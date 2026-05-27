{-# LANGUAGE ScopedTypeVariables #-}
module Main where

import Control.Exception (bracket, finally, try, catch, SomeException(..), IOException, throwIO, Exception(..))
import Control.Monad (forM_, when)
import qualified Data.Vector.Storable as V
import Data.Vector.HNSW
import Data.Vector.HNSW.Internal.AtomicSave
  ( AtomicSaveOps(..), realAtomicSaveOps, atomicSaveWith )
import System.Directory (createDirectory, doesFileExist,
                         getTemporaryDirectory, listDirectory, removePathForcibly,
                         renameDirectory)
import System.FilePath ((</>), takeDirectory, takeFileName)
import Data.List (isPrefixOf, isInfixOf, sort)

assertThrowsHNSW :: String -> IO a -> IO ()
assertThrowsHNSW label action = do
  result <- try (action >> pure ()) :: IO (Either HNSWException ())
  case result of
    Left (HNSWException _) -> pure ()
    Right _ -> error ("expected HNSWException: " ++ label)

nondecreasing :: (Ord a) => [a] -> Bool
nondecreasing xs = and (zipWith (<=) xs (drop 1 xs))

cleanupDir :: FilePath -> IO ()
cleanupDir dir = removePathForcibly dir `catch` (\(_ :: SomeException) -> pure ())

withCleanDir :: FilePath -> IO a -> IO a
withCleanDir dir action = do
  cleanupDir dir
  action `finally` cleanupDir dir

-- | Remove sibling .tmp-* and .old-* directories sharing the same parent+base as the given path.
cleanupSaveSiblings :: FilePath -> IO ()
cleanupSaveSiblings path = do
  let dir = takeDirectory path
      base = takeFileName path
  siblings <- listDirectory dir
  let toRemove =
        filter (\name -> (base ++ ".tmp-") `isPrefixOf` name
                      || (base ++ ".old-") `isPrefixOf` name)
               siblings
  forM_ toRemove $ \name -> removePathForcibly (dir </> name)
    `catch` (\(_ :: SomeException) -> pure ())

-- | Bracket-style resource helper: create index, run action, close index.
-- Ensures cleanup even if action throws.
withIndexForTest :: Metric -> Int -> Int -> Int -> Int -> (Index -> IO a) -> IO a
withIndexForTest metric dim maxElements m efConstruction action =
  bracket (new metric dim maxElements m efConstruction) close action

-- | Check that a list has no duplicate elements.
noDuplicates :: Eq a => [a] -> Bool
noDuplicates [] = True
noDuplicates (x:xs) = x `notElem` xs && noDuplicates xs

-- | Assert two values are equal, with a label.
assertEqual :: (Eq a, Show a) => String -> a -> a -> IO ()
assertEqual label expected actual =
  if expected == actual
    then pure ()
    else error (label ++ ": expected " ++ show expected ++ ", got " ++ show actual)

-- | Remove a path (file or dir), ignoring errors.
cleanupPath :: FilePath -> IO ()
cleanupPath path = removePathForcibly path `catch` (\(_ :: SomeException) -> pure ())

-- | Run an action with a path, cleaning up the path afterwards.
withCleanPath :: FilePath -> IO a -> IO a
withCleanPath path action = do
  cleanupPath path
  action `finally` cleanupPath path

-- ============================================================
-- Atomic save failure-injection test helpers
-- ============================================================

mkRenameInjectingOps :: (FilePath -> FilePath -> Maybe IOException) -> AtomicSaveOps
mkRenameInjectingOps shouldFail = realAtomicSaveOps
  { asoRenameDirectory = \src dst ->
      case shouldFail src dst of
        Just ex -> throwIO ex
        Nothing  -> renameDirectory src dst
  }

mkWriteIndexInjectingOps :: Exception e => e -> AtomicSaveOps
mkWriteIndexInjectingOps ex = realAtomicSaveOps
  { asoWriteIndex = \_ _ -> throwIO ex
  }

mkWriteMetadataInjectingOps :: Exception e => e -> AtomicSaveOps
mkWriteMetadataInjectingOps ex = realAtomicSaveOps
  { asoWriteMetadata = \_ _ -> throwIO ex
  }

loadAndAssertLabel :: FilePath -> Int -> IO ()
loadAndAssertLabel dir expectedLabel = do
  idx <- loadIndex dir
  results <- search idx queryVec 1
  case results of
    [(l, _)] | l == expectedLabel -> close idx >> pure ()
    _ -> close idx >> error "loadAndAssertLabel: wrong label"
  where
    queryVec = V.fromList [1.0, 0.0, 0.0, 0.0]

main :: IO ()
main = do
  -- L2 create/close
  do
    idx <- new L2 4 100 16 200
    close idx
    putStrLn "L2 create/close: ok"

  -- InnerProduct create/close
  do
    idx <- new InnerProduct 4 100 16 200
    close idx
    putStrLn "InnerProduct create/close: ok"

  -- double close is safe
  do
    idx <- new L2 4 100 16 200
    close idx
    close idx
    putStrLn "double close: ok"

  -- invalid dim=0 throws HNSWException
  do
    assertThrowsHNSW "dim=0" $ new L2 0 100 16 200
    putStrLn "dim=0 throws: ok"

  -- invalid maxElements=0 throws HNSWException
  do
    assertThrowsHNSW "maxElements=0" $ new L2 4 0 16 200
    putStrLn "maxElements=0 throws: ok"

  -- insert one vector
  do
    idx <- new L2 4 100 16 200
    insert idx 42 (V.fromList [1.0, 0.0, 0.0, 0.0])
    close idx
    putStrLn "insert one vector: ok"

  -- insert dimension mismatch throws HNSWException
  do
    idx <- new L2 4 100 16 200
    assertThrowsHNSW "dim mismatch" $ insert idx 1 (V.fromList [1.0, 0.0, 0.0])
    close idx
    putStrLn "insert dim mismatch throws: ok"

  -- negative label throws HNSWException
  do
    idx <- new L2 4 100 16 200
    assertThrowsHNSW "negative label" $ insert idx (-1) (V.fromList [1.0, 0.0, 0.0, 0.0])
    close idx
    putStrLn "negative label throws: ok"

  -- insert after close throws
  do
    idx <- new L2 4 100 16 200
    close idx
    assertThrowsHNSW "insert after close" $ insert idx 1 (V.fromList [1.0, 0.0, 0.0, 0.0])
    putStrLn "insert after close throws: ok"

  -- insert one vector and find it with k=1
  do
    idx <- new L2 4 100 16 200
    insert idx 42 (V.fromList [1.0, 0.0, 0.0, 0.0])
    results <- search idx (V.fromList [1.0, 0.0, 0.0, 0.0]) 1
    close idx
    case results of
      [(l, d)] | l == 42 -> putStrLn "find inserted vector: ok"
      _ -> error "find inserted vector: unexpected result"

  -- result ordering (ascending by distance)
  do
    idx <- new L2 4 100 16 200
    insert idx 10 (V.fromList [0.0, 1.0, 0.0, 0.0])
    insert idx 20 (V.fromList [1.0, 0.0, 0.0, 0.0])
    insert idx 30 (V.fromList [0.5, 0.5, 0.0, 0.0])
    results <- search idx (V.fromList [0.0, 1.0, 0.0, 0.0]) 3
    close idx
    let ds = map snd results
    if nondecreasing ds
      then putStrLn "result ordering: ok"
      else error "result ordering: distances not sorted"

  -- search dimension mismatch throws Haskell-side
  do
    idx <- new L2 4 100 16 200
    assertThrowsHNSW "search dim mismatch" $
      search idx (V.fromList [1.0, 0.0, 0.0]) 1
    close idx
    putStrLn "search dim mismatch throws: ok"

  -- k <= 0 throws Haskell-side
  do
    idx <- new L2 4 100 16 200
    assertThrowsHNSW "k=0" $ search idx (V.fromList [1.0, 0.0, 0.0, 0.0]) 0
    close idx
    putStrLn "k<=0 throws: ok"

  -- search on empty index returns empty list
  do
    idx <- new L2 4 100 16 200
    results <- search idx (V.fromList [1.0, 0.0, 0.0, 0.0]) 3
    close idx
    if null results
      then putStrLn "search empty index: ok"
      else error "search empty index: expected empty list"

  -- save/load round trip
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-round-trip-test"
    withCleanDir dir $ do
      idx <- new L2 4 100 16 200
      insert idx 42 (V.fromList [1.0, 0.0, 0.0, 0.0])
      let query = V.fromList [1.0, 0.0, 0.0, 0.0]
      resultsBefore <- search idx query 1
      case resultsBefore of
        [(l, _)] | l == 42 -> pure ()
        _ -> error "round trip: resultsBefore check failed"
      saveIndex idx dir
      close idx
      idx2 <- loadIndex dir
      resultsAfter <- search idx2 query 1
      close idx2
      case resultsAfter of
        [(l, _)] | l == 42 -> putStrLn "save/load round trip: ok"
        _ -> error "save/load round trip: label mismatch after load"

  -- load missing directory throws
  do
    let dir = "/nonexistent/hnsw-hs-12345-dir"
    assertThrowsHNSW "load missing dir" $ loadIndex dir
    putStrLn "load missing dir throws: ok"

  -- load missing metadata throws
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-missing-meta-test"
    withCleanDir dir $ do
      createDirectory dir
      writeFile (dir </> "hnsw.index") ""
      assertThrowsHNSW "load missing metadata" $ loadIndex dir
    putStrLn "load missing metadata throws: ok"

  -- malformed metadata throws (unknown version)
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-bad-meta-test"
    withCleanDir dir $ do
      createDirectory dir
      writeFile (dir </> "metadata.txt") "not-version-1\nmetric=L2\ndimension=4\nmax_elements=100"
      assertThrowsHNSW "load malformed metadata" $ loadIndex dir
    putStrLn "load malformed metadata throws: ok"

  -- malformed metadata throws (missing required field)
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-missing-field-test"
    withCleanDir dir $ do
      createDirectory dir
      writeFile (dir </> "metadata.txt") (unlines
        [ "hnsw-hs-metadata-v2"
        , "metric=L2"
        , "dimension=4"
        ])
      assertThrowsHNSW "load missing required field" $ loadIndex dir
    putStrLn "load missing required field throws: ok"

  -- malformed metadata throws (duplicate key)
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-dup-key-test"
    withCleanDir dir $ do
      createDirectory dir
      writeFile (dir </> "metadata.txt") (unlines
        [ "hnsw-hs-metadata-v2"
        , "metric=L2"
        , "metric=L2"
        , "dimension=4"
        , "max_elements=100"
        ])
      assertThrowsHNSW "load duplicate metadata key" $ loadIndex dir
    putStrLn "load duplicate metadata key throws: ok"

  -- load missing hnsw.index throws
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-missing-index-test"
    withCleanDir dir $ do
      createDirectory dir
      writeFile (dir </> "metadata.txt") (unlines
        [ "hnsw-hs-metadata-v2"
        , "metric=L2"
        , "dimension=4"
        , "max_elements=100"
        , "label=42"
        ])
      assertThrowsHNSW "load missing hnsw.index" $ loadIndex dir
    putStrLn "load missing hnsw.index throws: ok"

  -- save after close throws
  do
    idx <- new L2 4 100 16 200
    close idx
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-save-after-close-test"
    withCleanDir dir $ do
      assertThrowsHNSW "save after close" $ saveIndex idx dir
    putStrLn "save after close throws: ok"

  -- insert beyond maxElements throws HNSWException
  do
    idx <- new L2 4 2 2 20
    let cleanup = close idx
    (do
        insert idx 10 (V.fromList [1.0, 0.0, 0.0, 0.0])
        insert idx 20 (V.fromList [2.0, 0.0, 0.0, 0.0])
        assertThrowsHNSW "max_elements overflow" $
          insert idx 30 (V.fromList [3.0, 0.0, 0.0, 0.0])
        putStrLn "max_elements overflow throws: ok"
      ) `finally` cleanup

  -- search after close throws
  do
    idx <- new L2 4 100 16 200
    close idx
    assertThrowsHNSW "search after close" $
      search idx (V.fromList [1.0, 0.0, 0.0, 0.0]) 1
    putStrLn "search after close throws: ok"

  -- k greater than element count: all inserted found, no duplicates, labels valid, sorted
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      insert idx 10 (V.fromList [0.0, 1.0, 0.0, 0.0])
      insert idx 20 (V.fromList [1.0, 0.0, 0.0, 0.0])
      insert idx 30 (V.fromList [0.5, 0.5, 0.0, 0.0])
      results <- search idx (V.fromList [0.0, 1.0, 0.0, 0.0]) 10
      let labels = map fst results
          distances = map snd results
          insertedLabels = [10, 20, 30]
      assertEqual "k>count: result count" 3 (length results)
      assertEqual "k>count: no duplicate labels" True (noDuplicates labels)
      assertEqual "k>count: all labels in inserted set" True (all (`elem` insertedLabels) labels)
      assertEqual "k>count: distances nondecreasing" True (nondecreasing distances)
    putStrLn "k > count: ok"

  -- save to existing directory replaces it and removes stale files
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-overwrite-test"
    withCleanDir dir $ do
      let vecA = V.fromList [1.0, 0.0, 0.0, 0.0]
          vecB = V.fromList [0.0, 1.0, 0.0, 0.0]

      withIndexForTest L2 4 100 16 200 $ \idxA -> do
        insert idxA 1 vecA
        saveIndex idxA dir

      writeFile (dir </> "stale.txt") "junk"

      withIndexForTest L2 4 100 16 200 $ \idxB -> do
        insert idxB 2 vecB
        saveIndex idxB dir

      staleGone <- not <$> doesFileExist (dir </> "stale.txt")
      when (not staleGone) $
        error "overwrite: stale.txt still present after save"

      idxC <- loadIndex dir
      (do
        rB <- search idxC vecB 1
        case rB of
          [(l, _)] | l == 2 -> pure ()
          _ -> error "overwrite: label 2 not found after overwrite"
        rA <- search idxC vecA 10
        let labelsA = map fst rA
        when (1 `elem` labelsA) $
          error "overwrite: old label 1 still present after overwrite"
        ) `finally` close idxC

    putStrLn "save overwrite: ok"

  -- saveIndex to regular file throws
  do
    tmp <- getTemporaryDirectory
    let file = tmp </> "hnsw-hs-regular-file-test"
    withCleanPath file $ do
      writeFile file ""
      withIndexForTest L2 4 100 16 200 $ \idx -> do
        insert idx 42 (V.fromList [1.0, 0.0, 0.0, 0.0])
        assertThrowsHNSW "save to regular file" $ saveIndex idx file
      putStrLn "save to regular file throws: ok"

  -- load then save again works
  do
    tmp <- getTemporaryDirectory
    let dir1 = tmp </> "hnsw-hs-save2-dir1"
        dir2 = tmp </> "hnsw-hs-save2-dir2"
    withCleanDir dir1 $ withCleanDir dir2 $ do
      idx <- new L2 4 100 16 200
      let vec = V.fromList [1.0, 0.0, 0.0, 0.0]
      insert idx 42 vec
      saveIndex idx dir1
      close idx
      idx2 <- loadIndex dir1
      saveIndex idx2 dir2
      close idx2
      idx3 <- loadIndex dir2
      results <- search idx3 vec 1
      close idx3
      case results of
        [(l, _)] | l == 42 -> putStrLn "load then save again: ok"
        _ -> error "load then save again: label mismatch"

  -- loaded index double-close is safe
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-close-loaded-test"
    withCleanDir dir $ do
      idx <- new L2 4 100 16 200
      insert idx 42 (V.fromList [1.0, 0.0, 0.0, 0.0])
      saveIndex idx dir
      close idx
      idx2 <- loadIndex dir
      close idx2
      close idx2
      putStrLn "close loaded index safe: ok"

  -- search on loaded index after close throws
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-search-after-close-loaded"
    withCleanDir dir $ do
      idx <- new L2 4 100 16 200
      insert idx 42 (V.fromList [1.0, 0.0, 0.0, 0.0])
      saveIndex idx dir
      close idx
      idx2 <- loadIndex dir
      close idx2
      assertThrowsHNSW "search after close (loaded)" $
        search idx2 (V.fromList [1.0, 0.0, 0.0, 0.0]) 1
      putStrLn "search after close (loaded) throws: ok"

  -- malformed metadata throws (unknown key)
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-unknown-key-test"
    withCleanDir dir $ do
      createDirectory dir
      writeFile (dir </> "metadata.txt") (unlines
        [ "hnsw-hs-metadata-v2"
        , "metric=L2"
        , "dimension=4"
        , "max_elements=100"
        , "unknown_key=value"
        ])
      assertThrowsHNSW "load unknown metadata key" $ loadIndex dir
    putStrLn "load unknown metadata key throws: ok"

  -- malformed metadata throws (invalid dimension integer)
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-bad-dim-test"
    withCleanDir dir $ do
      createDirectory dir
      writeFile (dir </> "metadata.txt") (unlines
        [ "hnsw-hs-metadata-v2"
        , "metric=L2"
        , "dimension=abc"
        , "max_elements=100"
        ])
      assertThrowsHNSW "load invalid dimension" $ loadIndex dir
    putStrLn "load invalid dimension throws: ok"

  -- malformed metadata throws (negative max_elements)
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-neg-melem-test"
    withCleanDir dir $ do
      createDirectory dir
      writeFile (dir </> "metadata.txt") (unlines
        [ "hnsw-hs-metadata-v2"
        , "metric=L2"
        , "dimension=4"
        , "max_elements=-1"
        ])
      assertThrowsHNSW "load negative max_elements" $ loadIndex dir
    putStrLn "load negative max_elements throws: ok"

  -- duplicate insert throws
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      insert idx 42 (V.fromList [1.0, 0.0, 0.0, 0.0])
      assertThrowsHNSW "duplicate label" $
        insert idx 42 (V.fromList [2.0, 0.0, 0.0, 0.0])
    putStrLn "duplicate insert throws: ok"

  -- distinct labels still work
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      insert idx 10 (V.fromList [1.0, 0.0, 0.0, 0.0])
      insert idx 20 (V.fromList [0.0, 1.0, 0.0, 0.0])
      results <- search idx (V.fromList [0.0, 1.0, 0.0, 0.0]) 2
      let labels = map fst results
      when (not (all (`elem` labels) [10, 20])) $
        error "distinct labels: labels not found"
    putStrLn "distinct labels: ok"

  -- after load, duplicate insert throws
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-dup-after-load-test"
    withCleanDir dir $ do
      idx <- new L2 4 100 16 200
      insert idx 42 (V.fromList [1.0, 0.0, 0.0, 0.0])
      saveIndex idx dir
      close idx
      idx2 <- loadIndex dir
      (assertThrowsHNSW "duplicate after load" $
         insert idx2 42 (V.fromList [2.0, 0.0, 0.0, 0.0]))
        `finally` close idx2
    putStrLn "duplicate after load throws: ok"

  -- metadata duplicate label line throws
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-dup-label-meta-test"
    withCleanDir dir $ do
      createDirectory dir
      writeFile (dir </> "metadata.txt") (unlines
        [ "hnsw-hs-metadata-v2"
        , "metric=L2"
        , "dimension=4"
        , "max_elements=100"
        , "label=1"
        , "label=1"
        ])
      assertThrowsHNSW "metadata duplicate label" $ loadIndex dir
    putStrLn "metadata duplicate label throws: ok"

  -- metadata negative label throws
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-neg-label-meta-test"
    withCleanDir dir $ do
      createDirectory dir
      writeFile (dir </> "metadata.txt") (unlines
        [ "hnsw-hs-metadata-v2"
        , "metric=L2"
        , "dimension=4"
        , "max_elements=100"
        , "label=-1"
        ])
      assertThrowsHNSW "metadata negative label" $ loadIndex dir
    putStrLn "metadata negative label throws: ok"

  -- v1 metadata throws
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-v1-meta-test"
    withCleanDir dir $ do
      createDirectory dir
      writeFile (dir </> "metadata.txt") (unlines
        [ "hnsw-hs-metadata-v1"
        , "metric=L2"
        , "dimension=4"
        , "max_elements=100"
        ])
      assertThrowsHNSW "v1 metadata" $ loadIndex dir
    putStrLn "v1 metadata throws: ok"

  -- saved metadata contains sorted live_label lines
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-sorted-labels-test"
    withCleanDir dir $ do
      idx <- new L2 4 100 16 200
      insert idx 20 (V.fromList [0.0, 1.0, 0.0, 0.0])
      insert idx 10 (V.fromList [1.0, 0.0, 0.0, 0.0])
      insert idx 30 (V.fromList [0.5, 0.5, 0.0, 0.0])
      saveIndex idx dir
      close idx
      meta <- readFile (dir </> "metadata.txt")
      let labelLines = filter ("live_label=" `isPrefixOf`) (lines meta)
      assertEqual "sorted live_label lines" ["live_label=10", "live_label=20", "live_label=30"] labelLines
    putStrLn "sorted live_label lines: ok"

  -- loaded label set rejects duplicates
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-label-set-test"
    withCleanDir dir $ do
      idx <- new L2 4 100 16 200
      insert idx 10 (V.fromList [1.0, 0.0, 0.0, 0.0])
      insert idx 20 (V.fromList [0.0, 1.0, 0.0, 0.0])
      saveIndex idx dir
      close idx
      idx2 <- loadIndex dir
      (do
        -- label 10 already exists, should throw
        assertThrowsHNSW "duplicate after load" $
          insert idx2 10 (V.fromList [0.0, 0.0, 1.0, 0.0])
        -- label 30 is new, should succeed
        let v30 = V.fromList [0.5, 0.5, 0.0, 0.0]
        insert idx2 30 v30
        results <- search idx2 v30 1
        case results of
          [(l, _)] | l == 30 -> pure ()
          _ -> error "loaded label set: new insert not found"
        ) `finally` close idx2
    putStrLn "loaded label set: ok"

  -- atomic save creates complete directory with both required files
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-atomic-complete-test"
    withCleanDir dir $
      withIndexForTest L2 4 100 16 200 $ \idx -> do
        insert idx 42 (V.fromList [1.0, 0.0, 0.0, 0.0])
        saveIndex idx dir
        indexOk <- doesFileExist (dir </> "hnsw.index")
        metadataOk <- doesFileExist (dir </> "metadata.txt")
        when (not indexOk) $ error "atomic save: hnsw.index missing"
        when (not metadataOk) $ error "atomic save: metadata.txt missing"
        idx2 <- loadIndex dir
        (do
          results <- search idx2 (V.fromList [1.0, 0.0, 0.0, 0.0]) 1
          case results of
            [(l, _)] | l == 42 -> pure ()
            _ -> error "atomic save: label not found after load"
          ) `finally` close idx2
    putStrLn "atomic save complete directory: ok"

  -- no .tmp- or .old- sibling directories remain after save/replace
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-no-remnants-test"

    cleanupSaveSiblings dir
    withCleanDir dir $
      (do
        withIndexForTest L2 4 100 16 200 $ \idx1 -> do
          insert idx1 1 (V.fromList [1.0, 0.0, 0.0, 0.0])
          saveIndex idx1 dir

        withIndexForTest L2 4 100 16 200 $ \idx2 -> do
          insert idx2 2 (V.fromList [0.0, 1.0, 0.0, 0.0])
          saveIndex idx2 dir

        siblings <- listDirectory (takeDirectory dir)
        let base = takeFileName dir
            remnants =
              filter
                (\name -> (base ++ ".tmp-") `isPrefixOf` name
                      || (base ++ ".old-") `isPrefixOf` name)
                siblings
        when (not (null remnants)) $
          error ("no remnants: unexpected sibling directories: " ++ unwords remnants)
      ) `finally` cleanupSaveSiblings dir

    putStrLn "no sibling remnants: ok"

  -- load after repeated saves to same target yields latest data
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-repeated-save-test"
    withCleanDir dir $ do
      withIndexForTest L2 4 100 16 200 $ \idx1 -> do
        insert idx1 1 (V.fromList [1.0, 0.0, 0.0, 0.0])
        saveIndex idx1 dir

      withIndexForTest L2 4 100 16 200 $ \idx2 -> do
        insert idx2 2 (V.fromList [0.0, 1.0, 0.0, 0.0])
        saveIndex idx2 dir

      idx3 <- loadIndex dir
      (do
        results <- search idx3 (V.fromList [0.0, 1.0, 0.0, 0.0]) 2
        let labels = map fst results
        when (1 `elem` labels) $
          error "repeated save: old label 1 still present"
        when (not (2 `elem` labels)) $
          error "repeated save: new label 2 not found"
        ) `finally` close idx3

    putStrLn "repeated save to same target: ok"

  -- ============================================================
  -- Atomic save failure-injection tests
  -- ============================================================

  -- Test 1: write index failure leaves previous target and cleans temp
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-write-index-failure-test"
    withCleanDir dir $
      (do
        withIndexForTest L2 4 100 16 200 $ \idxA -> do
          insert idxA 42 (V.fromList [1.0, 0.0, 0.0, 0.0])
          saveIndex idxA dir

        withIndexForTest L2 4 100 16 200 $ \idxB -> do
          insert idxB 99 (V.fromList [0.0, 1.0, 0.0, 0.0])
          let injOps = mkWriteIndexInjectingOps (HNSWException "injected: write failed")
          (atomicSaveWith injOps idxB dir `catch` (\(_ :: HNSWException) -> pure ()))

        loadAndAssertLabel dir 42
      ) `finally` cleanupSaveSiblings dir
    putStrLn "atomic save write index failure: ok"

  -- Test 2: metadata write failure leaves previous target and cleans temp
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-metadata-write-failure-test"
    withCleanDir dir $
      (do
        withIndexForTest L2 4 100 16 200 $ \idxA -> do
          insert idxA 42 (V.fromList [1.0, 0.0, 0.0, 0.0])
          saveIndex idxA dir

        withIndexForTest L2 4 100 16 200 $ \idxB -> do
          insert idxB 99 (V.fromList [0.0, 1.0, 0.0, 0.0])
          let injOps = mkWriteMetadataInjectingOps (HNSWException "injected: metadata failed")
          (atomicSaveWith injOps idxB dir `catch` (\(_ :: HNSWException) -> pure ()))

        loadAndAssertLabel dir 42
      ) `finally` cleanupSaveSiblings dir
    putStrLn "atomic save metadata write failure: ok"

  -- Test 3: publish failure on new target cleans temp and leaves no target
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-publish-new-failure-test"
    withCleanDir dir $
      (do
        withIndexForTest L2 4 100 16 200 $ \idx -> do
          insert idx 99 (V.fromList [0.0, 1.0, 0.0, 0.0])
          let injOps = mkRenameInjectingOps $ \_src dst ->
                if dst == dir
                  then Just (userError "injected: publish failed")
                  else Nothing
          (atomicSaveWith injOps idx dir `catch` (\(_ :: HNSWException) -> pure ()))
          exists <- doesFileExist dir
          when exists $ error "publish failure test: target should not exist"
        cleanupSaveSiblings dir
      ) `finally` cleanupSaveSiblings dir
    putStrLn "atomic save publish new failure: ok"

  -- Test 4: publish failure replacing existing target rolls back
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-publish-replace-failure-test"
    withCleanDir dir $
      (do
        withIndexForTest L2 4 100 16 200 $ \idxA -> do
          insert idxA 42 (V.fromList [1.0, 0.0, 0.0, 0.0])
          saveIndex idxA dir

        withIndexForTest L2 4 100 16 200 $ \idxB -> do
          insert idxB 99 (V.fromList [0.0, 1.0, 0.0, 0.0])
          let injOps = mkRenameInjectingOps $ \src dst ->
                if dst == dir && ".tmp-" `isInfixOf` src
                  then Just (userError "injected: publish failed")
                  else Nothing
          (atomicSaveWith injOps idxB dir `catch` (\(_ :: HNSWException) -> pure ()))

        loadAndAssertLabel dir 42
        cleanupSaveSiblings dir
      ) `finally` cleanupSaveSiblings dir
    putStrLn "atomic save publish replace failure: ok"

  -- Test 5: rollback failure reports both errors
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-rollback-failure-test"
    withCleanDir dir $
      (do
        withIndexForTest L2 4 100 16 200 $ \idxA -> do
          insert idxA 42 (V.fromList [1.0, 0.0, 0.0, 0.0])
          saveIndex idxA dir

        withIndexForTest L2 4 100 16 200 $ \idxB -> do
          insert idxB 99 (V.fromList [0.0, 1.0, 0.0, 0.0])
          let injOps = mkRenameInjectingOps $ \src dst ->
                case (src, dst) of
                  (_, dst') | dst' == dir && ".tmp-" `isInfixOf` src ->
                    Just (userError "injected: publish failed")
                  (_, dst') | dst' == dir && ".old-" `isInfixOf` src ->
                    Just (userError "injected: rollback failed")
                  _ -> Nothing
          result <- try (atomicSaveWith injOps idxB dir) :: IO (Either SomeException ())
          case result of
            Left _ -> pure () -- expected exception
            Right _ -> error "rollback failure test: expected exception"
        cleanupSaveSiblings dir
      ) `finally` cleanupSaveSiblings dir
    putStrLn "atomic save rollback failure: ok"

  -- Test 6: backup cleanup failure after successful publish does not fail save
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-backup-cleanup-failure-test"
    withCleanDir dir $
      (do
        withIndexForTest L2 4 100 16 200 $ \idxA -> do
          insert idxA 42 (V.fromList [1.0, 0.0, 0.0, 0.0])
          saveIndex idxA dir

        withIndexForTest L2 4 100 16 200 $ \idxB -> do
          insert idxB 99 (V.fromList [0.0, 1.0, 0.0, 0.0])
          let injOps = realAtomicSaveOps
                {asoRemovePathForcibly = \p ->
                   if ".old-" `isInfixOf` p
                     then throwIO (userError "injected: cleanup failed")
                     else removePathForcibly p
                }
          atomicSaveWith injOps idxB dir -- should succeed despite cleanup failure
        loadAndAssertLabel dir 99
      ) `finally` cleanupSaveSiblings dir
    putStrLn "atomic save backup cleanup failure: ok"

  -- insertMany basic: insert 3 distinct vectors/labels, search k=3 finds all
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      insertMany idx
        [ (1, V.fromList [1.0, 0.0, 0.0, 0.0])
        , (2, V.fromList [0.0, 1.0, 0.0, 0.0])
        , (3, V.fromList [0.0, 0.0, 1.0, 0.0])
        ]
      results <- search idx (V.fromList [0.0, 0.0, 1.0, 0.0]) 3
      let labels = map fst results
      assertEqual "insertMany basic: found 3 labels" 3 (length results)
      assertEqual "insertMany basic: all expected labels" True (all (`elem` labels) [1, 2, 3])
    putStrLn "insertMany basic: ok"

  -- insertMany empty list: no-op, does not throw
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      insertMany idx []
      results <- search idx (V.fromList [1.0, 0.0, 0.0, 0.0]) 3
      assertEqual "insertMany empty: empty result" True (null results)
    putStrLn "insertMany empty list: ok"

  -- insertMany dimension mismatch throws: first insert succeeds, second throws,
  -- first remains searchable
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      insert idx 1 (V.fromList [1.0, 0.0, 0.0, 0.0])
      assertThrowsHNSW "insertMany dim mismatch" $
        insertMany idx [(2, V.fromList [0.0, 1.0, 0.0])]
      -- first item must still be present
      results <- search idx (V.fromList [1.0, 0.0, 0.0, 0.0]) 1
      case results of
        [(l, _)] | l == 1 -> pure ()
        _ -> error "insertMany dim mismatch: first insert not found after second failed"
    putStrLn "insertMany dimension mismatch throws: ok"

  -- insertMany negative label throws
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      assertThrowsHNSW "insertMany negative label" $
        insertMany idx [(1, V.fromList [1.0, 0.0, 0.0, 0.0]), (-1, V.fromList [0.0, 1.0, 0.0, 0.0])]
    putStrLn "insertMany negative label throws: ok"

  -- insertMany duplicate label (within batch) throws, first insert remains
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      let v1 = V.fromList [1.0, 0.0, 0.0, 0.0]
          v2 = V.fromList [0.0, 1.0, 0.0, 0.0]
      assertThrowsHNSW "insertMany duplicate within batch" $
        insertMany idx [(1, v1), (1, v2)]
      -- first insert must still be present
      results <- search idx v1 1
      case results of
        [(l, _)] | l == 1 -> pure ()
        _ -> error "insertMany duplicate within batch: first insert not found"
    putStrLn "insertMany duplicate within batch throws: ok"

  -- insertMany duplicate label (against existing) throws, existing remains
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      insert idx 1 (V.fromList [1.0, 0.0, 0.0, 0.0])
      assertThrowsHNSW "insertMany duplicate against existing" $
        insertMany idx [(2, V.fromList [0.0, 1.0, 0.0, 0.0]), (1, V.fromList [0.0, 0.0, 1.0, 0.0])]
      results <- search idx (V.fromList [1.0, 0.0, 0.0, 0.0]) 1
      case results of
        [(l, _)] | l == 1 -> pure ()
        _ -> error "insertMany duplicate against existing: original not found"
    putStrLn "insertMany duplicate against existing throws: ok"

  -- insertMany after close throws
  do
    idx <- new L2 4 100 16 200
    close idx
    assertThrowsHNSW "insertMany after close" $
      insertMany idx [(1, V.fromList [1.0, 0.0, 0.0, 0.0])]
    putStrLn "insertMany after close throws: ok"

  -- setEf accepts positive ef and search still works
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      setEf idx 10
      insert idx 1 (V.fromList [1.0, 0.0, 0.0, 0.0])
      results <- search idx (V.fromList [1.0, 0.0, 0.0, 0.0]) 1
      case results of
        [(l, _)] | l == 1 -> pure ()
        _ -> error "setEf positive: search failed after setEf"
    putStrLn "setEf accepts positive ef: ok"

  -- setEf rejects zero
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      assertThrowsHNSW "setEf ef=0" $ setEf idx 0
    putStrLn "setEf rejects zero: ok"

  -- setEf rejects negative
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      assertThrowsHNSW "setEf ef=-1" $ setEf idx (-1)
    putStrLn "setEf rejects negative: ok"

  -- setEf after close throws
  do
    idx <- new L2 4 100 16 200
    close idx
    assertThrowsHNSW "setEf after close" $ setEf idx 10
    putStrLn "setEf after close throws: ok"

  -- setEf on loaded index works and search still works
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-setef-loaded-test"
    withCleanDir dir $ do
      idx <- new L2 4 100 16 200
      insert idx 1 (V.fromList [1.0, 0.0, 0.0, 0.0])
      saveIndex idx dir
      close idx
      idx2 <- loadIndex dir
      (do
        setEf idx2 50
        results <- search idx2 (V.fromList [1.0, 0.0, 0.0, 0.0]) 1
        case results of
          [(l, _)] | l == 1 -> pure ()
          _ -> error "setEf on loaded index: search failed after setEf"
        ) `finally` close idx2
    putStrLn "setEf on loaded index: ok"

  -- saveIndex writes v3 header
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-v3-header-test"
    withCleanDir dir $ do
      idx <- new L2 4 100 16 200
      insert idx 1 (V.fromList [1.0, 0.0, 0.0, 0.0])
      insert idx 2 (V.fromList [0.0, 1.0, 0.0, 0.0])
      saveIndex idx dir
      close idx
      meta <- readFile (dir </> "metadata.txt")
      let firstLine = head (lines meta)
      assertEqual "v3 header" "hnsw-hs-metadata-v3" firstLine
      let liveLines = filter ("live_label=" `isPrefixOf`) (lines meta)
      assertEqual "v3 live_label count" 2 (length liveLines)
      let hasLegacyLabel = any (\l -> "label=" `isPrefixOf` l && not ("live_label=" `isPrefixOf` l)) (lines meta)
      assertEqual "v3 no legacy label= lines" False hasLegacyLabel
    putStrLn "saveIndex writes v3 header: ok"

  -- saveIndex writes live_label keys
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-v3-live-label-keys-test"
    withCleanDir dir $ do
      idx <- new L2 4 100 16 200
      insert idx 10 (V.fromList [1.0, 0.0, 0.0, 0.0])
      insert idx 20 (V.fromList [0.0, 1.0, 0.0, 0.0])
      saveIndex idx dir
      close idx
      meta <- readFile (dir </> "metadata.txt")
      let liveLines = filter ("live_label=" `isPrefixOf`) (lines meta)
      assertEqual "live_label lines present" 2 (length liveLines)
      assertEqual "live_label=10 present" True ("live_label=10" `elem` liveLines)
      assertEqual "live_label=20 present" True ("live_label=20" `elem` liveLines)
    putStrLn "saveIndex v3 live_label keys: ok"

  -- loadIndex v2 roundtrip
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-v2-roundtrip-test"
    withCleanDir dir $ do
      idx <- new L2 4 100 16 200
      insert idx 42 (V.fromList [1.0, 0.0, 0.0, 0.0])
      saveIndex idx dir
      close idx
      -- overwrite with v2 metadata
      writeFile (dir </> "metadata.txt") (unlines
        [ "hnsw-hs-metadata-v2"
        , "metric=L2"
        , "dimension=4"
        , "max_elements=100"
        , "label=42"
        ])
      idx2 <- loadIndex dir
      (do
        results <- search idx2 (V.fromList [1.0, 0.0, 0.0, 0.0]) 1
        case results of
          [(l, _)] | l == 42 -> pure ()
          _ -> error "v2 roundtrip: label not found after load"
        assertThrowsHNSW "v2 duplicate insert" $
          insert idx2 42 (V.fromList [2.0, 0.0, 0.0, 0.0])
        ) `finally` close idx2
    putStrLn "loadIndex v2 roundtrip: ok"

  -- loadIndex v3 live labels only
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-v3-live-test"
    withCleanDir dir $ do
      idx <- new L2 4 100 16 200
      insert idx 1 (V.fromList [1.0, 0.0, 0.0, 0.0])
      saveIndex idx dir
      close idx
      -- overwrite with v3 live labels only
      writeFile (dir </> "metadata.txt") (unlines
        [ "hnsw-hs-metadata-v3"
        , "metric=L2"
        , "dimension=4"
        , "max_elements=100"
        , "live_label=1"
        ])
      idx2 <- loadIndex dir
      (do
        assertThrowsHNSW "v3 live duplicate insert" $
          insert idx2 1 (V.fromList [2.0, 0.0, 0.0, 0.0])
        ) `finally` close idx2
    putStrLn "loadIndex v3 live labels: ok"

  -- loadIndex accepts v3 with reserved labels
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-v3-reserved-load-test"
    withCleanDir dir $ do
      idx <- new L2 4 100 16 200
      insert idx 1 (V.fromList [1.0, 0.0, 0.0, 0.0])
      saveIndex idx dir
      close idx
      -- overwrite with v3 containing reserved_label
      writeFile (dir </> "metadata.txt") (unlines
        [ "hnsw-hs-metadata-v3"
        , "metric=L2"
        , "dimension=4"
        , "max_elements=100"
        , "live_label=1"
        , "reserved_label=99"
        ])
      idx2 <- loadIndex dir
      (do
        assertThrowsHNSW "insert reserved label" $
          insert idx2 99 (V.fromList [2.0, 0.0, 0.0, 0.0])
        ) `finally` close idx2
    putStrLn "loadIndex accepts v3 with reserved labels: ok"

  -- reserved labels roundtrip after load
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-reserved-roundtrip-test"
    withCleanDir dir $ do
      -- load index with reserved label
      createDirectory dir
      writeFile (dir </> "metadata.txt") (unlines
        [ "hnsw-hs-metadata-v3"
        , "metric=L2"
        , "dimension=4"
        , "max_elements=100"
        , "live_label=1"
        , "reserved_label=99"
        ])
      -- create a minimal hnsw.index (hnswlib will reject it but we just want to test metadata roundtrip)
      -- Actually we need a valid index to load, so create and save a real one first
      idx <- new L2 4 100 16 200
      insert idx 1 (V.fromList [1.0, 0.0, 0.0, 0.0])
      saveIndex idx dir
      close idx
      -- overwrite with v3 containing reserved
      writeFile (dir </> "metadata.txt") (unlines
        [ "hnsw-hs-metadata-v3"
        , "metric=L2"
        , "dimension=4"
        , "max_elements=100"
        , "live_label=1"
        , "reserved_label=99"
        ])
      idx2 <- loadIndex dir
      saveIndex idx2 dir
      close idx2
      meta <- readFile (dir </> "metadata.txt")
      let reservedLines = filter ("reserved_label=" `isPrefixOf`) (lines meta)
      assertEqual "reserved_label persists" ["reserved_label=99"] reservedLines
    putStrLn "reserved labels roundtrip after load: ok"

  -- v3 disjointness throws
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-v3-disjoint-test"
    withCleanDir dir $ do
      createDirectory dir
      writeFile (dir </> "metadata.txt") (unlines
        [ "hnsw-hs-metadata-v3"
        , "metric=L2"
        , "dimension=4"
        , "max_elements=100"
        , "live_label=1"
        , "reserved_label=1"
        ])
      assertThrowsHNSW "v3 disjointness" $ loadIndex dir
    putStrLn "v3 disjointness throws: ok"

  -- v3 unknown key throws
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-v3-unknown-key-test"
    withCleanDir dir $ do
      createDirectory dir
      writeFile (dir </> "metadata.txt") (unlines
        [ "hnsw-hs-metadata-v3"
        , "metric=L2"
        , "dimension=4"
        , "max_elements=100"
        , "live_label=1"
        , "unknown_key=value"
        ])
      assertThrowsHNSW "v3 unknown key" $ loadIndex dir
    putStrLn "v3 unknown key throws: ok"

  -- v3 label key in v3 throws
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-v3-label-key-test"
    withCleanDir dir $ do
      createDirectory dir
      writeFile (dir </> "metadata.txt") (unlines
        [ "hnsw-hs-metadata-v3"
        , "metric=L2"
        , "dimension=4"
        , "max_elements=100"
        , "label=1"
        ])
      assertThrowsHNSW "v3 label key" $ loadIndex dir
    putStrLn "v3 label key in v3 throws: ok"

  -- v2 rejects live_label key
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-v2-live-label-test"
    withCleanDir dir $ do
      createDirectory dir
      writeFile (dir </> "metadata.txt") (unlines
        [ "hnsw-hs-metadata-v2"
        , "metric=L2"
        , "dimension=4"
        , "max_elements=100"
        , "live_label=1"
        ])
      assertThrowsHNSW "v2 live_label key" $ loadIndex dir
    putStrLn "v2 rejects live_label key: ok"

  -- v2 rejects reserved_label key
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-v2-reserved-label-test"
    withCleanDir dir $ do
      createDirectory dir
      writeFile (dir </> "metadata.txt") (unlines
        [ "hnsw-hs-metadata-v2"
        , "metric=L2"
        , "dimension=4"
        , "max_elements=100"
        , "reserved_label=1"
        ])
      assertThrowsHNSW "v2 reserved_label key" $ loadIndex dir
    putStrLn "v2 rejects reserved_label key: ok"

  -- v3 duplicate live_label throws
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-v3-dup-live-test"
    withCleanDir dir $ do
      createDirectory dir
      writeFile (dir </> "metadata.txt") (unlines
        [ "hnsw-hs-metadata-v3"
        , "metric=L2"
        , "dimension=4"
        , "max_elements=100"
        , "live_label=1"
        , "live_label=1"
        ])
      assertThrowsHNSW "v3 duplicate live_label" $ loadIndex dir
    putStrLn "v3 duplicate live_label throws: ok"

  -- v3 duplicate reserved_label throws
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-v3-dup-reserved-test"
    withCleanDir dir $ do
      createDirectory dir
      writeFile (dir </> "metadata.txt") (unlines
        [ "hnsw-hs-metadata-v3"
        , "metric=L2"
        , "dimension=4"
        , "max_elements=100"
        , "reserved_label=1"
        , "reserved_label=1"
        ])
      assertThrowsHNSW "v3 duplicate reserved_label" $ loadIndex dir
    putStrLn "v3 duplicate reserved_label throws: ok"

  -- v3 negative reserved_label throws
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-v3-neg-reserved-test"
    withCleanDir dir $ do
      createDirectory dir
      writeFile (dir </> "metadata.txt") (unlines
        [ "hnsw-hs-metadata-v3"
        , "metric=L2"
        , "dimension=4"
        , "max_elements=100"
        , "reserved_label=-1"
        ])
      assertThrowsHNSW "v3 negative reserved_label" $ loadIndex dir
    putStrLn "v3 negative reserved_label throws: ok"

  -- insert checks reserved set
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-insert-reserved-test"
    withCleanDir dir $ do
      idx <- new L2 4 100 16 200
      insert idx 1 (V.fromList [1.0, 0.0, 0.0, 0.0])
      saveIndex idx dir
      close idx
      writeFile (dir </> "metadata.txt") (unlines
        [ "hnsw-hs-metadata-v3"
        , "metric=L2"
        , "dimension=4"
        , "max_elements=100"
        , "live_label=1"
        , "reserved_label=99"
        ])
      idx2 <- loadIndex dir
      (do
        assertThrowsHNSW "insert reserved label" $
          insert idx2 99 (V.fromList [2.0, 0.0, 0.0, 0.0])
        ) `finally` close idx2
    putStrLn "insert checks reserved set: ok"

  -- remove existing label
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      let v1 = V.fromList [1.0, 0.0, 0.0, 0.0]
          v2 = V.fromList [0.0, 1.0, 0.0, 0.0]
      insert idx 1 v1
      insert idx 2 v2
      remove idx 1
      results <- search idx v1 2
      let labels = map fst results
      assertEqual "remove: label 1 not in results" False (1 `elem` labels)
      assertEqual "remove: label 2 still in results" True (2 `elem` labels)
    putStrLn "remove existing label: ok"

  -- remove negative label throws
  do
    idx <- new L2 4 100 16 200
    assertThrowsHNSW "remove negative label" $ remove idx (-1)
    close idx
    putStrLn "remove negative label throws: ok"

  -- remove missing label throws
  do
    idx <- new L2 4 100 16 200
    assertThrowsHNSW "remove missing label" $ remove idx 999
    close idx
    putStrLn "remove missing label throws: ok"

  -- remove double remove throws
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      insert idx 1 (V.fromList [1.0, 0.0, 0.0, 0.0])
      remove idx 1
      assertThrowsHNSW "double remove" $ remove idx 1
    putStrLn "remove double remove throws: ok"

  -- remove after close throws
  do
    idx <- new L2 4 100 16 200
    insert idx 1 (V.fromList [1.0, 0.0, 0.0, 0.0])
    close idx
    assertThrowsHNSW "remove after close" $ remove idx 1
    putStrLn "remove after close throws: ok"

  -- insert removed label throws
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      insert idx 1 (V.fromList [1.0, 0.0, 0.0, 0.0])
      remove idx 1
      assertThrowsHNSW "insert after remove" $
        insert idx 1 (V.fromList [2.0, 0.0, 0.0, 0.0])
    putStrLn "insert removed label throws: ok"

  -- remove save/load preserves behavior
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-remove-load-test"
    withCleanDir dir $ do
      idx <- new L2 4 100 16 200
      insert idx 1 (V.fromList [1.0, 0.0, 0.0, 0.0])
      insert idx 2 (V.fromList [0.0, 1.0, 0.0, 0.0])
      remove idx 1
      saveIndex idx dir
      close idx
      idx2 <- loadIndex dir
      (do
        results <- search idx2 (V.fromList [1.0, 0.0, 0.0, 0.0]) 2
        let labels = map fst results
        assertEqual "remove/load: label 1 absent" False (1 `elem` labels)
        assertEqual "remove/load: label 2 present" True (2 `elem` labels)
        assertThrowsHNSW "remove/load: insert removed label" $
          insert idx2 1 (V.fromList [2.0, 0.0, 0.0, 0.0])
        ) `finally` close idx2
    putStrLn "remove save/load preserves behavior: ok"

  -- metadata after remove contains reserved_label
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-remove-meta-test"
    withCleanDir dir $ do
      idx <- new L2 4 100 16 200
      insert idx 1 (V.fromList [1.0, 0.0, 0.0, 0.0])
      insert idx 2 (V.fromList [0.0, 1.0, 0.0, 0.0])
      remove idx 1
      saveIndex idx dir
      close idx
      meta <- readFile (dir </> "metadata.txt")
      let reservedLines = filter ("reserved_label=" `isPrefixOf`) (lines meta)
          liveLines = filter ("live_label=" `isPrefixOf`) (lines meta)
      assertEqual "remove meta: reserved_label=1 present" ["reserved_label=1"] reservedLines
      assertEqual "remove meta: live_label=1 absent" False ("live_label=1" `elem` liveLines)
      assertEqual "remove meta: live_label=2 present" True ("live_label=2" `elem` liveLines)
    putStrLn "metadata after remove contains reserved_label: ok"

  -- remove then insertMany same label throws
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      insert idx 1 (V.fromList [1.0, 0.0, 0.0, 0.0])
      remove idx 1
      assertThrowsHNSW "insertMany after remove" $
        insertMany idx [(1, V.fromList [2.0, 0.0, 0.0, 0.0])]
    putStrLn "remove then insertMany same label throws: ok"

  -- remove then searchBatch excludes removed label
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      let v1 = V.fromList [1.0, 0.0, 0.0, 0.0]
          v2 = V.fromList [0.0, 1.0, 0.0, 0.0]
      insert idx 1 v1
      insert idx 2 v2
      remove idx 1
      results <- searchBatch idx [v1, v2] 1
      assertEqual "searchBatch after remove: v1 result count" 1 (length (results !! 0))
      let labels1 = map fst (results !! 0)
      assertEqual "searchBatch after remove: v1 does not return 1" False (1 `elem` labels1)
    putStrLn "remove then searchBatch excludes removed label: ok"

  -- remove reserved label throws
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-hs-remove-reserved-test"
    withCleanDir dir $ do
      idx <- new L2 4 100 16 200
      insert idx 1 (V.fromList [1.0, 0.0, 0.0, 0.0])
      saveIndex idx dir
      close idx
      -- add reserved label 99 via metadata
      writeFile (dir </> "metadata.txt") (unlines
        [ "hnsw-hs-metadata-v3"
        , "metric=L2"
        , "dimension=4"
        , "max_elements=100"
        , "live_label=1"
        , "reserved_label=99"
        ])
      idx2 <- loadIndex dir
      (do
        assertThrowsHNSW "remove reserved label" $ remove idx2 99
        ) `finally` close idx2
    putStrLn "remove reserved label throws: ok"

  -- searchBatch basic: insert 3 vectors, searchBatch returns 3 results
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      let v1 = V.fromList [1.0, 0.0, 0.0, 0.0]
          v2 = V.fromList [0.0, 1.0, 0.0, 0.0]
          v3 = V.fromList [0.0, 0.0, 1.0, 0.0]
      insert idx 1 v1
      insert idx 2 v2
      insert idx 3 v3
      results <- searchBatch idx [v1, v2, v3] 1
      assertEqual "searchBatch basic: result count" 3 (length results)
      case results of
        [[(l1, _)], [(l2, _)], [(l3, _)]] ->
          assertEqual "searchBatch basic: labels" [1, 2, 3] [l1, l2, l3]
        _ -> error "searchBatch basic: unexpected result structure"
    putStrLn "searchBatch basic: ok"

  -- searchBatch empty list returns empty
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      results <- searchBatch idx [] 1
      assertEqual "searchBatch empty: result is []" True (null results)
    putStrLn "searchBatch empty list: ok"

  -- searchBatch k <= 0 throws
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      let v = V.fromList [1.0, 0.0, 0.0, 0.0]
      assertThrowsHNSW "searchBatch k=0" $
        searchBatch idx [v] 0
    putStrLn "searchBatch k<=0 throws: ok"

  -- searchBatch dimension mismatch throws
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      let valid = V.fromList [1.0, 0.0, 0.0, 0.0]
          badDim = V.fromList [1.0, 0.0, 0.0]  -- dim 3 instead of 4
      assertThrowsHNSW "searchBatch dim mismatch" $
        searchBatch idx [valid, badDim] 1
    putStrLn "searchBatch dimension mismatch throws: ok"

  -- searchBatch after close throws
  do
    idx <- new L2 4 100 16 200
    close idx
    let v = V.fromList [1.0, 0.0, 0.0, 0.0]
    assertThrowsHNSW "searchBatch after close" $
      searchBatch idx [v] 1
    putStrLn "searchBatch after close throws: ok"

  -- searchBatch honors setEf
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      let v = V.fromList [1.0, 0.0, 0.0, 0.0]
      insert idx 1 v
      setEf idx 50
      results <- searchBatch idx [v] 1
      case results of
        [[(l, _)]] | l == 1 -> pure ()
        _ -> error "searchBatch with setEf: unexpected result"
    putStrLn "searchBatch honors setEf: ok"

  -- Test 1: remove preserves other labels
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      let v1 = V.fromList [1.0, 0.0, 0.0, 0.0]
          v2 = V.fromList [0.0, 1.0, 0.0, 0.0]
          v3 = V.fromList [0.0, 0.0, 1.0, 0.0]
      insert idx 1 v1
      insert idx 2 v2
      insert idx 3 v3
      remove idx 2
      results1 <- search idx v1 1
      results3 <- search idx v3 1
      results2 <- search idx v2 1
      assertEqual "remove preserves label 1" [(1, 0.0)] results1
      assertEqual "remove preserves label 3" [(3, 0.0)] results3
      assertEqual "remove excludes label 2" True (null results2 || notElem 2 (map fst results2))
    putStrLn "remove preserves other labels: ok"

  -- Test 2: remove multiple labels
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      let v1 = V.fromList [1.0, 0.0, 0.0, 0.0]
          v2 = V.fromList [0.0, 1.0, 0.0, 0.0]
          v3 = V.fromList [0.0, 0.0, 1.0, 0.0]
      insert idx 1 v1
      insert idx 2 v2
      insert idx 3 v3
      remove idx 1
      remove idx 3
      results2 <- search idx v2 1
      assertEqual "remove multiple: label 2 still searchable" [(2, 0.0)] results2
      assertThrowsHNSW "remove multiple: insert 1 throws" $ insert idx 1 v1
      assertThrowsHNSW "remove multiple: insert 3 throws" $ insert idx 3 v3
    putStrLn "remove multiple labels: ok"

  -- Test 3: remove then save metadata has sorted live/reserved labels
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-slice22-test3"
    withCleanDir dir $ do
      createDirectory dir
      withIndexForTest L2 4 100 16 200 $ \idx -> do
        let v3 = V.fromList [3.0, 0.0, 0.0, 0.0]
            v1 = V.fromList [1.0, 0.0, 0.0, 0.0]
            v2 = V.fromList [0.0, 1.0, 0.0, 0.0]
        insert idx 3 v3
        insert idx 1 v1
        insert idx 2 v2
        remove idx 2
        saveIndex idx dir
      metadata <- readFile (dir </> "metadata.txt")
      let liveLines = filter ("live_label=" `isPrefixOf`) (lines metadata)
          reservedLines = filter ("reserved_label=" `isPrefixOf`) (lines metadata)
          liveNums = map (read . drop (length "live_label=")) liveLines
          reservedNums = map (read . drop (length "reserved_label=")) reservedLines
      assertEqual "live labels sorted" True (liveNums == sort liveNums)
      assertEqual "reserved labels sorted" True (reservedNums == sort reservedNums)
      assertEqual "live labels present" [1, 3] liveNums
      assertEqual "reserved labels present" [2] reservedNums
    putStrLn "remove then save metadata has sorted live/reserved labels: ok"

  -- Test 4: remove missing after load throws
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-slice22-test4"
    withCleanDir dir $ do
      withIndexForTest L2 4 100 16 200 $ \idx -> do
        let v = V.fromList [1.0, 0.0, 0.0, 0.0]
        insert idx 1 v
        saveIndex idx dir
      loadIndex dir >>= \idx' ->
        (assertThrowsHNSW "remove missing after load throws" $ remove idx' 999) `finally` close idx'
    putStrLn "remove missing after load throws: ok"

  -- Test 5: remove reserved after load throws
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-slice22-test5"
    withCleanDir dir $ do
      withIndexForTest L2 4 100 16 200 $ \idx -> do
        let v = V.fromList [1.0, 0.0, 0.0, 0.0]
        insert idx 1 v
        saveIndex idx dir
      -- Rewrite metadata to include reserved_label=99
      writeFile (dir </> "metadata.txt") (unlines
        [ "hnsw-hs-metadata-v3"
        , "metric=L2"
        , "dimension=4"
        , "max_elements=100"
        , "live_label=1"
        , "reserved_label=99"
        ])
      loadIndex dir >>= \idx' ->
        (assertThrowsHNSW "remove reserved after load throws" $ remove idx' 99) `finally` close idx'
    putStrLn "remove reserved after load throws: ok"

  -- Test 6: remove loaded v2 index then save writes v3 reserved label
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-slice22-test6"
    withCleanDir dir $ do
      withIndexForTest L2 4 100 16 200 $ \idx -> do
        let v = V.fromList [1.0, 0.0, 0.0, 0.0]
        insert idx 1 v
        insert idx 2 v
        saveIndex idx dir
      -- Rewrite to v2 format
      writeFile (dir </> "metadata.txt") (unlines
        [ "hnsw-hs-metadata-v2"
        , "metric=L2"
        , "dimension=4"
        , "max_elements=100"
        , "label=1"
        , "label=2"
        ])
      loadIndex dir >>= \idx' -> do
        remove idx' 2
        saveIndex idx' dir
        close idx'
      loadedMetadata <- readFile (dir </> "metadata.txt")
      assertEqual "v2->v3: first line is v3" "hnsw-hs-metadata-v3" (head (lines loadedMetadata))
      assertEqual "v2->v3: live_label=1 present" True $
        any ("live_label=1" `isPrefixOf`) (lines loadedMetadata)
      assertEqual "v2->v3: reserved_label=2 present" True $
        any ("reserved_label=2" `isPrefixOf`) (lines loadedMetadata)
    putStrLn "remove loaded v2 index then save writes v3 reserved label: ok"

  -- Test 7: remove loaded v3 reserved label remains reserved
  do
    tmp <- getTemporaryDirectory
    let dir = tmp </> "hnsw-slice22-test7"
    withCleanDir dir $ do
      -- Create a valid hnsw.index first via normal save
      withIndexForTest L2 4 100 16 200 $ \idx -> do
        let v = V.fromList [1.0, 0.0, 0.0, 0.0]
        insert idx 1 v
        saveIndex idx dir
      -- Rewrite metadata to v3 with reserved_label=99 but without the live_label=2 from prior insert
      writeFile (dir </> "metadata.txt") (unlines
        [ "hnsw-hs-metadata-v3"
        , "metric=L2"
        , "dimension=4"
        , "max_elements=100"
        , "live_label=1"
        , "reserved_label=99"
        ])
      loadIndex dir >>= \idx' -> do
        let v2 = V.fromList [2.0, 0.0, 0.0, 0.0]
        insert idx' 2 v2
        remove idx' 2
        saveIndex idx' dir
        close idx'
      loadIndex dir >>= \idx'' ->
        let v99 = V.fromList [1.0, 0.0, 0.0, 0.0] in
        (assertThrowsHNSW "reserved label remains reserved after save/load" $ insert idx'' 99 v99) `finally` close idx''
    putStrLn "remove loaded v3 reserved label remains reserved: ok"

  -- Test 8: remove after setEf still works
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      let v1 = V.fromList [1.0, 0.0, 0.0, 0.0]
          v2 = V.fromList [0.0, 1.0, 0.0, 0.0]
      setEf idx 50
      insert idx 1 v1
      insert idx 2 v2
      remove idx 1
      results2 <- search idx v2 1
      results1 <- search idx v1 1
      assertEqual "setEf then remove: label 2 still found" [(2, 0.0)] results2
      assertEqual "setEf then remove: label 1 excluded" True (null results1 || notElem 1 (map fst results1))
    putStrLn "remove after setEf still works: ok"

  -- Test 9: remove after insertMany works
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      let v1 = V.fromList [1.0, 0.0, 0.0, 0.0]
          v2 = V.fromList [0.0, 1.0, 0.0, 0.0]
          v3 = V.fromList [0.0, 0.0, 1.0, 0.0]
      insertMany idx [(1, v1), (2, v2), (3, v3)]
      remove idx 2
      results1 <- search idx v1 1
      results2 <- search idx v2 1
      results3 <- search idx v3 1
      assertEqual "insertMany then remove: label 1 found" [(1, 0.0)] results1
      assertEqual "insertMany then remove: label 2 excluded" True (null results2 || notElem 2 (map fst results2))
      assertEqual "insertMany then remove: label 3 found" [(3, 0.0)] results3
    putStrLn "remove after insertMany works: ok"

  -- Update tests
  -- Test 1: update existing label changes nearest-search result
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      let vA = V.fromList [1.0, 0.0, 0.0, 0.0]
          vB = V.fromList [0.0, 1.0, 0.0, 0.0]
          vC = V.fromList [0.0, 0.0, 1.0, 0.0]
      insert idx 1 vA
      insert idx 2 vB
      update idx 1 vC
      results <- search idx vC 1
      assertEqual "update: label 1 found near vC" [(1, 0.0)] results
    putStrLn "update existing label changes nearest-search result: ok"

  -- Test 2: update preserves label returns same label after update
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      let v1 = V.fromList [1.0, 0.0, 0.0, 0.0]
          v2 = V.fromList [0.0, 1.0, 0.0, 0.0]
      insert idx 1 v1
      update idx 1 v2
      results <- search idx v2 1
      assertEqual "update preserves label: label 1 returned" [(1, 0.0)] results
    putStrLn "update preserves label returns same label after update: ok"

  -- Test 3: update dim mismatch throws
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      let v3 = V.fromList [1.0, 0.0, 0.0]
      insert idx 1 (V.fromList [0.0, 0.0, 0.0, 0.0])
      assertThrowsHNSW "update dim mismatch throws" $ update idx 1 v3
    putStrLn "update dim mismatch throws: ok"

  -- Test 4: update negative label throws
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      let v = V.fromList [1.0, 0.0, 0.0, 0.0]
      insert idx 1 v
      assertThrowsHNSW "update negative label throws" $ update idx (-1) v
    putStrLn "update negative label throws: ok"

  -- Test 5: update missing label throws
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      let v = V.fromList [1.0, 0.0, 0.0, 0.0]
      assertThrowsHNSW "update missing label throws" $ update idx 99 v
    putStrLn "update missing label throws: ok"

  -- Test 6: update reserved label throws
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      let v = V.fromList [1.0, 0.0, 0.0, 0.0]
          v2 = V.fromList [0.0, 1.0, 0.0, 0.0]
      insert idx 1 v
      remove idx 1
      assertThrowsHNSW "update reserved label throws" $ update idx 1 v2
    putStrLn "update reserved label throws: ok"

  -- Test 7: update after close throws
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      let v = V.fromList [1.0, 0.0, 0.0, 0.0]
      insert idx 1 v
      close idx
      assertThrowsHNSW "update after close throws" $ update idx 1 v
    putStrLn "update after close throws: ok"

  -- Test 8: update after load works
  do
    createDirectory "tmp" `catch` (\(_ :: SomeException) -> pure ())
    let dir = "tmp/update-load"
    withCleanDir dir $ do
      withIndexForTest L2 4 100 16 200 $ \idx -> do
        let v1 = V.fromList [1.0, 0.0, 0.0, 0.0]
            v2 = V.fromList [0.0, 1.0, 0.0, 0.0]
        insert idx 1 v1
        update idx 1 v2
        saveIndex idx dir
        close idx
      loadIndex dir >>= \idx' -> do
        results <- search idx' (V.fromList [0.0, 1.0, 0.0, 0.0]) 1
        assertEqual "update after load: label 1 found near v2" [(1, 0.0)] results
        close idx'
    putStrLn "update after load works: ok"

  -- Test 9: update with ef set works
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      let v1 = V.fromList [1.0, 0.0, 0.0, 0.0]
          v2 = V.fromList [0.0, 1.0, 0.0, 0.0]
          v3 = V.fromList [0.0, 0.0, 1.0, 0.0]
      setEf idx 50
      insert idx 1 v1
      insert idx 2 v2
      update idx 1 v3
      results <- search idx v3 1
      assertEqual "update with ef set: label 1 found" [(1, 0.0)] results
    putStrLn "update with ef set works: ok"

  -- Test 10: update preserves live label bookkeeping
  do
    withIndexForTest L2 4 100 16 200 $ \idx -> do
      let v1 = V.fromList [1.0, 0.0, 0.0, 0.0]
          v2 = V.fromList [0.0, 1.0, 0.0, 0.0]
      insert idx 1 v1
      update idx 1 v2
      assertThrowsHNSW "update then re-insert duplicate throws" $ insert idx 1 v1
    putStrLn "update preserves live label bookkeeping: ok"

  putStrLn "all tests passed"
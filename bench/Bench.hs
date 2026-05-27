{-# LANGUAGE BangPatterns #-}

module Main where

import Control.Exception (bracket, catch, finally, IOException)
import Criterion.Main (bgroup, bench, defaultMain, whnfIO)
import Data.Vector.HNSW (Metric(..), Index, Vector, new, close, insert, search, setEf, saveIndex, loadIndex)
import qualified Data.Vector.Storable as SV
import System.Directory (createDirectoryIfMissing, removePathForcibly)
import System.FilePath ((</>))
import System.Info (arch, os)

-- Fixed benchmark parameters
dim :: Int
dim = 384

m :: Int
m = 16

efConstruction :: Int
efConstruction = 200

benchTmpRoot :: FilePath
benchTmpRoot = "tmp"

-- Deterministic vector generation (no RNG dependency)
vectorFor :: Int -> Int -> Vector
vectorFor !d !i = SV.generate d $ \j ->
  fromIntegral ((i * 131 + j * 17) `mod` 1000) / 1000

-- Bracket that creates an index and closes it on exit (prevents leaks)
withIndexForBench :: Int -> (Index -> IO a) -> IO a
withIndexForBench !maxElements !k =
  bracket (new L2 dim maxElements m efConstruction) close k

-- Build index with n inserts and close it
buildAndClose :: Int -> IO ()
buildAndClose !n = withIndexForBench n $ \idx ->
  mapM_ (\i -> insert idx i (vectorFor dim i)) [0 .. n - 1]

-- Build an index with n inserts, returning it (must be closed by caller)
buildIndex :: Int -> IO Index
buildIndex !n = do
  idx <- new L2 dim n m efConstruction
  mapM_ (\i -> insert idx i (vectorFor dim i)) [0 .. n - 1]
  pure idx

-- Search workload: build index, run queries, close
searchWorkload :: Int -> Int -> Int -> IO Int
searchWorkload !n !q !k = withIndexForBench n $ \idx -> do
  mapM_ (\i -> insert idx i (vectorFor dim i)) [0 .. n - 1]
  let queries = map (vectorFor dim) [0 .. q - 1]
  let loop [] !acc = return acc
      loop (query:qs) !acc = do
        !result <- search idx query k
        let !c = length result
        loop qs $! (acc + c)
  !total <- loop queries 0
  total `seq` return total

-- Search workload with ef tuning (honest: includes build + setEf + search)
searchWithEf :: Int -> Int -> Int -> Int -> IO Int
searchWithEf !n !q !k !ef = withIndexForBench n $ \idx -> do
  mapM_ (\i -> insert idx i (vectorFor dim i)) [0 .. n - 1]
  setEf idx ef
  let queries = map (vectorFor dim) [0 .. q - 1]
  let loop [] !acc = return acc
      loop (query:qs) !acc = do
        !result <- search idx query k
        let !c = length result
        loop qs $! (acc + c)
  !total <- loop queries 0
  total `seq` return total

-- Insert throughput: build empty index, insert n vectors
insertBatch :: Int -> IO ()
insertBatch !n = withIndexForBench n $ \idx ->
  mapM_ (\i -> insert idx i (vectorFor dim i)) [0 .. n - 1]

-- Run a single search and force the result
runSingleSearch :: Index -> IO Int
runSingleSearch !idx = do
  !result <- search idx (vectorFor dim 0) 10
  let !c = length result
  pure c

-- Run searches for multiple query vectors and count results
runSearchQueries :: Index -> Int -> Int -> IO Int
runSearchQueries !idx !q !k = do
  let queries = map (vectorFor dim) [0 .. q - 1]
  let loop [] !acc = return acc
      loop (query:qs) !acc = do
        !result <- search idx query k
        let !c = length result
        loop qs $! (acc + c)
  !total <- loop queries 0
  total `seq` return total

-- Temp directory helpers
makeBenchTempDir :: String -> IO FilePath
makeBenchTempDir !tag = do
  createDirectoryIfMissing True benchTmpRoot
  pure $ benchTmpRoot </> ("bench-" ++ tag)

cleanupBenchTempDir :: FilePath -> IO ()
cleanupBenchTempDir !path = removePathForcibly path `catch` ignoreIt
  where
    ignoreIt :: IOException -> IO ()
    ignoreIt _ = pure ()

withReusableBenchTempDir :: String -> (FilePath -> IO a) -> IO a
withReusableBenchTempDir !tag !action = do
  dir <- makeBenchTempDir tag
  cleanupBenchTempDir dir
  createDirectoryIfMissing True dir
  action dir `finally` cleanupBenchTempDir dir

main :: IO ()
main = do
  putStrLn "=== Benchmark environment ==="
  putStrLn $ "OS/arch: " ++ os ++ "/" ++ arch
  putStrLn "GHC: 9.10.3"
  putStrLn "Resolver: lts-24.41"
  putStrLn "Parameters: metric=L2, dim=384, M=16, efConstruction=200"
  putStrLn "hnswlib commit: d9b3608c83d83b46c96e25088cb1d729b29dcfe9"
  putStrLn "================================"

  -- Pre-build index for save benchmark (shared across all save iterations)
  saveIdx <- buildIndex 1000

  -- Pre-build and pre-save once for load benchmark
  loadDir <- makeBenchTempDir "load-src"
  createDirectoryIfMissing True loadDir
  do
    idx <- buildIndex 1000
    saveIndex idx loadDir
    close idx

  -- Pre-build index for search-latency benchmarks
  searchIdx <- buildIndex 1000
  setEf searchIdx 100

  defaultMain
    [ bgroup "create"
        [ bench "dim384-max1000" $
            whnfIO (withIndexForBench 1000 $ \_ -> pure ())
        ]
    , bgroup "build-index"
        [ bench "n100"  $ whnfIO (buildAndClose 100)
        , bench "n1000" $ whnfIO (buildAndClose 1000)
        ]
    , bgroup "search-workload"
        [ bench "build1000-query100-k10" $
            whnfIO (searchWorkload 1000 100 10)
        ]
    , bgroup "search-workload-ef"
        [ bench "build1000-query100-k10-ef10" $
            whnfIO (searchWithEf 1000 100 10 10)
        , bench "build1000-query100-k10-ef50" $
            whnfIO (searchWithEf 1000 100 10 50)
        , bench "build1000-query100-k10-ef100" $
            whnfIO (searchWithEf 1000 100 10 100)
        ]
    , bgroup "insert-throughput"
        [ bench "n100"  $ whnfIO (insertBatch 100)
        , bench "n1000" $ whnfIO (insertBatch 1000)
        ]
    , bgroup "search-latency"
        [ bench "build1000-query1-k10-ef100" $
            whnfIO (runSingleSearch searchIdx)
        , bench "build1000-query100-k10-ef100" $
            whnfIO (runSearchQueries searchIdx 100 10)
        ]
    , bgroup "persistence"
        [ bench "save-build1000" $
            whnfIO $
              withReusableBenchTempDir "save-target" $ \target ->
                saveIndex saveIdx target
        , bench "load-build1000" $
            whnfIO $ do
              idx <- loadIndex loadDir
              close idx
        ]
    ]

  close searchIdx
  close saveIdx
  cleanupBenchTempDir loadDir
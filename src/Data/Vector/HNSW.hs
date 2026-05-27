{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}
-- |
-- Module      : Data.Vector.HNSW
-- Description : Haskell bindings to hnswlib for approximate nearest-neighbor search.
--
-- This module exposes a wrapper around hnswlib indexes for vectors of
-- 'Float'. It supports insertion, search, search-depth tuning, deletion by
-- label, and directory-based persistence.
--
-- Labels are non-negative 'Int' values. Removed labels are reserved and cannot
-- be inserted again.
--
-- Persistence uses a directory containing @hnsw.index@ and @metadata.txt@.
module Data.Vector.HNSW
  ( Metric(..)
  , Vector
  , Index
  , HNSWException(..)
  , new
  , close
  , insert
  , insertMany
  , remove
  , update
  , setEf
  , search
  , searchBatch
  , saveIndex
  , loadIndex
  ) where

import Control.Concurrent.MVar (modifyMVar, modifyMVar_, newMVar, withMVar)
import Control.Exception (SomeException(..), displayException, throwIO, catch)
import Control.Monad (forM, when)
import qualified Data.Set as Set
import Foreign.C.String (peekCString, withCString)
import Foreign.C.Types (CInt(..), CSize(..))
import Foreign.ForeignPtr (FinalizerPtr, finalizeForeignPtr, newForeignPtr)
import Foreign.Marshal.Alloc (alloca)
import Foreign.Marshal.Array (allocaArray)
import Foreign.Ptr (Ptr, nullPtr)
import Foreign.Storable (peek, peekElemOff)
import qualified Data.Vector.Storable as Storable
import System.FilePath ((</>))

import Data.Vector.HNSW.Internal.Types
  ( Index(..)
  , Metric(..)
  , Vector
  , HNSWException(..)
  , HNSWIndex
  , withIndex
  , c_hnsw_create
  , c_hnsw_add_point
  , c_hnsw_last_error
  , c_hnsw_set_ef
  , c_hnsw_mark_delete
  , c_hnsw_update_point
  , c_hnsw_search_knn
  , c_hnsw_save_index
  , c_hnsw_load_index
  , c_hnsw_free_finalizer
  , Metadata(..)
  , parseMetadata
  , metricCode
  , catchIOException
  )

import Data.Vector.HNSW.Internal.AtomicSave (atomicSaveWith, realAtomicSaveOps)

-- | Create a new HNSW index.
--
-- Throws @HNSWException@ if parameters are invalid or the underlying
-- hnswlib construction fails (e.g. out-of-memory).
new :: Metric -> Int -> Int -> Int -> Int -> IO Index
new metric dim maxElements m efConstruction = do
  when (dim <= 0)           $ throwIO (HNSWException "new: dim must be positive")
  when (maxElements <= 0)  $ throwIO (HNSWException "new: maxElements must be positive")
  when (m <= 0)            $ throwIO (HNSWException "new: m must be positive")
  when (efConstruction <= 0) $ throwIO (HNSWException "new: efConstruction must be positive")
  let code = metricCode metric
  ptr <- c_hnsw_create
           (fromIntegral code)
           (fromIntegral dim)
           (fromIntegral maxElements)
           (fromIntegral m)
           (fromIntegral efConstruction)
  if ptr == nullPtr
    then do
      err <- c_hnsw_last_error >>= peekCString
      throwIO (HNSWException ("hnsw_create failed: " ++ err))
    else do
      fp <- newForeignPtr c_hnsw_free_finalizer ptr
      var <- newMVar (Just fp)
      labelsVar <- newMVar Set.empty
      reservedVar <- newMVar Set.empty
      pure (Index metric dim maxElements labelsVar reservedVar var)

-- | Close an index, freeing its resources.
--
-- Idempotent: calling 'close' on an already-closed index is a no-op.
close :: Index -> IO ()
close (Index _ _ _ _ _ var) = modifyMVar_ var $ \case
  Nothing -> pure Nothing
  Just fp -> do
    finalizeForeignPtr fp
    pure Nothing

-- | Insert a vector with the given label.
--
-- Validates label is non-negative and vector length matches the index
-- dimension before calling C. Throws @HNSWException@ on dimension mismatch
-- or if the underlying hnswlib insert fails.
insert :: Index -> Int -> Vector -> IO ()
insert idx label vec = do
  when (label < 0) $
    throwIO (HNSWException "insert: label must be non-negative")
  let expected = indexDim idx
      actual = Storable.length vec
  when (actual /= expected) $
    throwIO (HNSWException ("insert: dimension mismatch; expected " ++ show expected ++ ", got " ++ show actual))
  modifyMVar_ (indexLiveLabels idx) $ \liveLabels ->
    withMVar (indexReservedLabels idx) $ \reservedLabels -> do
      when (Set.member label liveLabels) $
        throwIO (HNSWException ("insert: duplicate label: " ++ show label))
      when (Set.member label reservedLabels) $
        throwIO (HNSWException ("insert: label was previously removed: " ++ show label))
      withIndex idx $ \ptr ->
        Storable.unsafeWith vec $ \vecPtr -> do
          rc <- c_hnsw_add_point ptr vecPtr (fromIntegral actual) (fromIntegral label)
          when (rc /= 0) $ do
            err <- c_hnsw_last_error >>= peekCString
            throwIO (HNSWException ("insert failed: " ++ err))
      pure (Set.insert label liveLabels)

-- | Insert multiple (label, vector) pairs in order.
--
-- Processing is in order and not atomic: if insertion fails partway
-- through, earlier successful inserts remain in the index.
insertMany :: Index -> [(Int, Vector)] -> IO ()
insertMany idx = mapM_ (uncurry (insert idx))

-- | Remove a label from the index.
--
-- Marks the label deleted in hnswlib, removes it from the live label set,
-- and adds it to the reserved label set. A removed label cannot be
-- re-inserted — the wrapper rejects it. Throws if the label is not live
-- or the index is closed.
remove :: Index -> Int -> IO ()
remove idx label
  | label < 0 = throwIO (HNSWException ("remove: negative label: " ++ show label))
  | otherwise =
      modifyMVar_ (indexLiveLabels idx) $ \liveLabels ->
        modifyMVar (indexReservedLabels idx) $ \reservedLabels -> do
          when (not (Set.member label liveLabels)) $
            throwIO (HNSWException ("remove: label not found: " ++ show label))
          withIndex idx $ \ptr -> do
            rc <- c_hnsw_mark_delete ptr (fromIntegral label)
            when (rc /= 0) $ do
              err <- c_hnsw_last_error >>= peekCString
              throwIO (HNSWException ("remove failed: " ++ err))
          let liveLabels' = Set.delete label liveLabels
              reservedLabels' = Set.insert label reservedLabels
          pure (reservedLabels', liveLabels')

-- | Update the vector stored for an existing live label.
--
-- The label must already exist (inserted but not removed). The vector
-- dimension must match the index dimension. The updated vector takes
-- effect on the next search.
--
-- /Throws/: 'HNSWException' if the index is closed, the label is missing,
-- the label was previously removed, or the vector dimension does not match.
--
-- /Since: 0.1.0.0
update :: Index -> Int -> Vector -> IO ()
update idx label vec
  | label < 0 = throwIO (HNSWException "update: label must be non-negative")
  | otherwise = do
      let actualDim = Storable.length vec
          expectedDim = indexDim idx
      when (actualDim /= expectedDim) $
        throwIO (HNSWException ("update: dimension mismatch; expected " ++ show expectedDim ++ ", got " ++ show actualDim))
      withMVar (indexLiveLabels idx) $ \live ->
        withMVar (indexReservedLabels idx) $ \reserved -> do
          when (not (Set.member label live)) $
            throwIO (HNSWException ("update: label not found: " ++ show label))
          when (Set.member label reserved) $
            throwIO (HNSWException ("update: label was previously removed: " ++ show label))
          withIndex idx $ \ptr ->
            Storable.unsafeWith vec $ \vecPtr -> do
              rc <- c_hnsw_update_point ptr vecPtr (fromIntegral expectedDim) (fromIntegral label)
              when (rc /= 0) $ do
                err <- c_hnsw_last_error >>= peekCString
                throwIO (HNSWException ("update failed: " ++ err))

-- | Set the search breadth parameter ef for subsequent searches.
--
-- Throws @HNSWException@ if ef is not positive, the index is closed,
-- or the underlying hnswlib call fails.
setEf :: Index -> Int -> IO ()
setEf idx ef
  | ef <= 0 = throwIO (HNSWException "setEf: ef must be positive")
  | otherwise =
      withIndex idx $ \ptr -> do
        rc <- c_hnsw_set_ef ptr (fromIntegral ef)
        when (rc /= 0) $ do
          err <- c_hnsw_last_error >>= peekCString
          throwIO (HNSWException ("setEf failed: " ++ err))

-- | Search for k nearest neighbors of a query vector.
--
-- Returns '[(label, distance)]' sorted ascending by distance (nearest first).
-- The number of results is at most k, subject to hnswlib behavior on an
-- index with fewer than k elements.
search :: Index -> Vector -> Int -> IO [(Int, Float)]
search idx vec k = do
  when (k <= 0) $
    throwIO (HNSWException "search: k must be positive")
  let expected = indexDim idx
      actual = Storable.length vec
  when (actual /= expected) $
    throwIO (HNSWException ("search: dimension mismatch; expected " ++ show expected ++ ", got " ++ show actual))
  withIndex idx $ \ptr ->
    Storable.unsafeWith vec $ \vecPtr ->
      allocaArray k $ \labelsBuf ->
        allocaArray k $ \distancesBuf ->
          alloca $ \countBuf -> do
            rc <- c_hnsw_search_knn ptr vecPtr (fromIntegral actual) (fromIntegral k) labelsBuf distancesBuf countBuf
            if rc /= 0
              then do
                err <- c_hnsw_last_error >>= peekCString
                throwIO (HNSWException ("search failed: " ++ err))
              else do
                n <- fromIntegral <$> peek countBuf
                when (n > k) $
                  throwIO (HNSWException "search: C shim returned more results than requested")
                forM [0 .. n - 1] $ \i -> do
                  labelSize <- peekElemOff labelsBuf i
                  let labelInteger = toInteger labelSize
                      maxInt = toInteger (maxBound :: Int)
                  when (labelInteger > maxInt) $
                    throwIO (HNSWException "search: returned label does not fit in Int")
                  let label = fromInteger labelInteger
                  d <- peekElemOff distancesBuf i
                  pure (label, d)

-- | Search for nearest neighbors for multiple query vectors.
--
-- This is a sequential convenience wrapper over 'search', not an
-- optimized C-level batch search. Queries are processed in order and
-- the function stops at the first failure.
searchBatch :: Index -> [Vector] -> Int -> IO [[(Int, Float)]]
searchBatch idx queries k =
  mapM (\query -> search idx query k) queries

-- | Save an index to a directory.
--
-- If the target directory itself is missing, saveIndex creates it by
-- publishing the completed temp directory (the parent directory must already
-- exist). Throws @HNSWException@ if the path is a regular file or if the
-- underlying hnswlib save fails. On successful return the directory contains
-- exactly 'hnsw.index' and 'metadata.txt'.
--
-- Uses a sibling temp directory and @renameDirectory@ to publish atomically
-- on local filesystems (atomic visibility, not crash durability; no
-- @fsync@ is performed). On network filesystems @renameDirectory@ may have
-- different atomicity semantics. Failures before publish leave the target
-- directory untouched. If publish fails after moving the previous target and
-- rollback also fails, both errors are reported and the backup sibling is
-- left for manual recovery.
saveIndex :: Index -> FilePath -> IO ()
saveIndex idx path = atomicSaveWith realAtomicSaveOps idx path

-- | Load an index from a directory.
--
-- Reads metadata.txt to determine metric, dimension, and max_elements,
-- then loads the hnsw.index file. Throws @HNSWException@ if metadata
-- is missing, malformed, or if the underlying hnswlib load fails.
loadIndex :: FilePath -> IO Index
loadIndex path = do
  meta <- parseMetadata =<< readFile (path </> "metadata.txt")
    `catch` \e -> throwIO (HNSWException ("loadIndex: " ++ displayException (e :: SomeException)))
  ptr <- loadIndexC path meta
  fp <- newForeignPtr c_hnsw_free_finalizer ptr
  var <- newMVar (Just fp)
  labelsVar <- newMVar (metadataLiveLabels meta)
  reservedVar <- newMVar (metadataReservedLabels meta)
  pure (Index
         (metadataMetric meta)
         (metadataDim meta)
         (metadataMaxElements meta)
         labelsVar
         reservedVar
         var)

loadIndexC :: FilePath -> Metadata -> IO (Ptr HNSWIndex)
loadIndexC path meta = do
  ptr <- withCString (path </> "hnsw.index") $ \cPath ->
    c_hnsw_load_index
      (metricCode (metadataMetric meta))
      cPath
      (fromIntegral (metadataDim meta))
      (fromIntegral (metadataMaxElements meta))
  if ptr == nullPtr
    then do
      err <- c_hnsw_last_error >>= peekCString
      throwIO (HNSWException ("loadIndex failed: " ++ err))
    else pure ptr
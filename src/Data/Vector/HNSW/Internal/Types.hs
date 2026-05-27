{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}
-- |
-- Module      : Data.Vector.HNSW.Internal.Types
-- Description : Internal types, FFI declarations, and low-level helpers.
--
-- This module is not part of the public API and should not be used
-- directly. It is placed in other-modules for compilation into the
-- library component.
module Data.Vector.HNSW.Internal.Types where

import Control.Concurrent.MVar (MVar, modifyMVar, newMVar, withMVar)
import Control.Exception
  ( Exception(..), IOException, SomeException(..), displayException
  , throwIO, catch, try, onException )
import Control.Monad (forM, when)
import Data.Typeable (Typeable)
import qualified Data.Set as Set
import Foreign.C.String (CString, peekCString, withCString)
import Foreign.C.Types (CInt(..), CSize(..))
import Foreign.ForeignPtr (ForeignPtr, FinalizerPtr, finalizeForeignPtr, newForeignPtr, withForeignPtr)
import Foreign.Marshal.Alloc (alloca)
import Foreign.Marshal.Array (allocaArray)
import Foreign.Ptr (Ptr, nullPtr)
import Foreign.Storable (peek, peekElemOff)
import System.Directory
  ( createDirectory, createDirectoryIfMissing
  , doesDirectoryExist, doesFileExist, doesPathExist
  , removePathForcibly, renameDirectory )
import System.FilePath ((</>))
import qualified Data.Vector.Storable as Storable

-- | Vector type accepted by this package.
--
-- The vector length must match the index dimension.
type Vector = Storable.Vector Float

-- | Distance metric used by the HNSW index.
data Metric = L2 | InnerProduct
  deriving (Eq, Ord, Show)

-- | Exception thrown for validation failures, closed indexes, persistence
-- errors, and errors reported by hnswlib.
data HNSWException = HNSWException String
  deriving (Show, Typeable)

instance Exception HNSWException

-- | Opaque C pointer type used only in FFI internals.
data HNSWIndex

-- | Opaque handle to an HNSW index.
data Index = Index
  { indexMetric          :: !Metric
  , indexDim            :: !Int
  , indexMaxElements    :: !Int
  , indexLiveLabels     :: !(MVar (Set.Set Int))
  , indexReservedLabels :: !(MVar (Set.Set Int))
  , indexPtr            :: !(MVar (Maybe (ForeignPtr HNSWIndex)))
  }

-- FFI imports

foreign import ccall safe "HNSW.h hnsw_create"
  c_hnsw_create
    :: CInt
    -> CSize
    -> CSize
    -> CSize
    -> CSize
    -> IO (Ptr HNSWIndex)

foreign import ccall unsafe "&hnsw_free"
  c_hnsw_free_finalizer :: FinalizerPtr HNSWIndex

foreign import ccall safe "HNSW.h hnsw_add_point"
  c_hnsw_add_point :: Ptr HNSWIndex -> Ptr Float -> CSize -> CSize -> IO CInt

foreign import ccall safe "HNSW.h hnsw_last_error"
  c_hnsw_last_error :: IO CString

foreign import ccall safe "HNSW.h hnsw_set_ef"
  c_hnsw_set_ef :: Ptr HNSWIndex -> CSize -> IO CInt

foreign import ccall safe "HNSW.h hnsw_mark_delete"
  c_hnsw_mark_delete :: Ptr HNSWIndex -> CSize -> IO CInt

foreign import ccall safe "HNSW.h hnsw_update_point"
  c_hnsw_update_point :: Ptr HNSWIndex -> Ptr Float -> CSize -> CSize -> IO CInt

foreign import ccall safe "HNSW.h hnsw_search_knn"
  c_hnsw_search_knn
    :: Ptr HNSWIndex
    -> Ptr Float
    -> CSize
    -> CSize
    -> Ptr CSize
    -> Ptr Float
    -> Ptr CSize
    -> IO CInt

foreign import ccall safe "HNSW.h hnsw_save_index"
  c_hnsw_save_index :: Ptr HNSWIndex -> CString -> IO CInt

foreign import ccall safe "HNSW.h hnsw_load_index"
  c_hnsw_load_index
    :: CInt
    -> CString
    -> CSize
    -> CSize
    -> IO (Ptr HNSWIndex)

-- | Internal helper: run an action with the raw index pointer.
withIndex :: Index -> (Ptr HNSWIndex -> IO a) -> IO a
withIndex (Index _ _ _ _ _ var) action =
  modifyMVar var $ \case
    Nothing ->
      throwIO (HNSWException "index is closed")
    Just fp -> do
      result <- withForeignPtr fp action
      pure (Just fp, result)

-- | Internal metadata type used for parsing and formatting persistence data.
data Metadata = Metadata
  { metadataMetric          :: !Metric
  , metadataDim            :: !Int
  , metadataMaxElements    :: !Int
  , metadataLiveLabels     :: !(Set.Set Int)
  , metadataReservedLabels :: !(Set.Set Int)
  }

-- | Parse metadata.txt content. Throws @HNSWException@ on parse failure.
parseMetadata :: String -> IO Metadata
parseMetadata content = case lines content of
  [] -> throwIO (HNSWException "parseMetadata: empty file")
  (v:rest) | v == "hnsw-hs-metadata-v2" ->
    go rest (MetadataFields Nothing Nothing Nothing Set.empty Set.empty) >>= finalizeMetadata
  (v:rest) | v == "hnsw-hs-metadata-v3" ->
    goV3 rest (MetadataFields Nothing Nothing Nothing Set.empty Set.empty) >>= finalizeMetadata
  (v:_) ->
    throwIO (HNSWException ("parseMetadata: unknown version: " ++ v))

-- v2 parser: only accepts label=, rejects live_label=/reserved_label=
go :: [String] -> MetadataFields -> IO MetadataFields
go [] acc = pure acc
go (l:ls) acc
  | null l = go ls acc
  | otherwise = case break (== '=') l of
      (k, '=':v)
        | null k -> throwIO (HNSWException "parseMetadata: empty key")
        | k == "metric" -> do
            m <- parseMetricValue v
            case fieldMetric acc of
              Nothing -> go ls acc { fieldMetric = Just m }
              Just _ -> throwIO (HNSWException "parseMetadata: duplicate key: metric")
        | k == "dimension" -> do
            n <- parsePositiveInt "dimension" v
            case fieldDim acc of
              Nothing -> go ls acc { fieldDim = Just n }
              Just _ -> throwIO (HNSWException "parseMetadata: duplicate key: dimension")
        | k == "max_elements" -> do
            n <- parsePositiveInt "max_elements" v
            case fieldMaxElements acc of
              Nothing -> go ls acc { fieldMaxElements = Just n }
              Just _ -> throwIO (HNSWException "parseMetadata: duplicate key: max_elements")
        | k == "label" -> do
            n <- parseNonNegativeInt "label" v
            case Set.member n (fieldLiveLabels acc) of
              True -> throwIO (HNSWException "parseMetadata: duplicate label value")
              False -> go ls acc { fieldLiveLabels = Set.insert n (fieldLiveLabels acc) }
        | k == "live_label" -> throwIO (HNSWException "parseMetadata: live_label not supported in v2")
        | k == "reserved_label" -> throwIO (HNSWException "parseMetadata: reserved_label not supported in v2")
        | otherwise -> throwIO (HNSWException ("parseMetadata: unknown key: " ++ k))
      _ -> throwIO (HNSWException "parseMetadata: malformed line")

-- v3 parser: accepts live_label= and reserved_label=, rejects label=
goV3 :: [String] -> MetadataFields -> IO MetadataFields
goV3 [] acc = pure acc
goV3 (l:ls) acc
  | null l = goV3 ls acc
  | otherwise = case break (== '=') l of
      (k, '=':v)
        | null k -> throwIO (HNSWException "parseMetadata: empty key")
        | k == "metric" -> do
            m <- parseMetricValue v
            case fieldMetric acc of
              Nothing -> goV3 ls acc { fieldMetric = Just m }
              Just _ -> throwIO (HNSWException "parseMetadata: duplicate key: metric")
        | k == "dimension" -> do
            n <- parsePositiveInt "dimension" v
            case fieldDim acc of
              Nothing -> goV3 ls acc { fieldDim = Just n }
              Just _ -> throwIO (HNSWException "parseMetadata: duplicate key: dimension")
        | k == "max_elements" -> do
            n <- parsePositiveInt "max_elements" v
            case fieldMaxElements acc of
              Nothing -> goV3 ls acc { fieldMaxElements = Just n }
              Just _ -> throwIO (HNSWException "parseMetadata: duplicate key: max_elements")
        | k == "live_label" -> do
            n <- parseNonNegativeInt "live_label" v
            case Set.member n (fieldLiveLabels acc) of
              True -> throwIO (HNSWException "parseMetadata: duplicate live_label value")
              False -> goV3 ls acc { fieldLiveLabels = Set.insert n (fieldLiveLabels acc) }
        | k == "reserved_label" -> do
            n <- parseNonNegativeInt "reserved_label" v
            case Set.member n (fieldReservedLabels acc) of
              True -> throwIO (HNSWException "parseMetadata: duplicate reserved_label value")
              False -> goV3 ls acc { fieldReservedLabels = Set.insert n (fieldReservedLabels acc) }
        | k == "label" -> throwIO (HNSWException "parseMetadata: label not supported in v3; use live_label")
        | otherwise -> throwIO (HNSWException ("parseMetadata: unknown key: " ++ k))
      _ -> throwIO (HNSWException "parseMetadata: malformed line")

data MetadataFields = MetadataFields
  { fieldMetric          :: !(Maybe Metric)
  , fieldDim           :: !(Maybe Int)
  , fieldMaxElements   :: !(Maybe Int)
  , fieldLiveLabels    :: !(Set.Set Int)
  , fieldReservedLabels :: !(Set.Set Int)
  }

finalizeMetadata :: MetadataFields -> IO Metadata
finalizeMetadata fields = do
  m <- requireField "metric" (fieldMetric fields)
  d <- requireField "dimension" (fieldDim fields)
  e <- requireField "max_elements" (fieldMaxElements fields)
  let live = fieldLiveLabels fields
      reserved = fieldReservedLabels fields
  when (not (Set.disjoint live reserved)) $
    throwIO (HNSWException "parseMetadata: live and reserved label sets are not disjoint")
  pure (Metadata m d e live reserved)

requireField :: String -> Maybe a -> IO a
requireField name = maybe
  (throwIO (HNSWException ("parseMetadata: missing required key: " ++ name)))
  pure

parseMetricValue :: String -> IO Metric
parseMetricValue "L2" = pure L2
parseMetricValue "InnerProduct" = pure InnerProduct
parseMetricValue s = throwIO (HNSWException ("parseMetadata: unknown metric: " ++ s))

parsePositiveInt :: String -> String -> IO Int
parsePositiveInt key s = case reads s :: [(Int, String)] of
  (n, ""):_ | n > 0 -> pure n
  _ -> throwIO (HNSWException ("parseMetadata: " ++ key ++ " must be positive int: " ++ s))

parseNonNegativeInt :: String -> String -> IO Int
parseNonNegativeInt key s = case reads s :: [(Int, String)] of
  (n, ""):_ | n >= 0 -> pure n
  _ -> throwIO (HNSWException ("parseMetadata: " ++ key ++ " must be non-negative int: " ++ s))

formatMetadata :: Index -> Set.Set Int -> Set.Set Int -> String
formatMetadata idx liveLabels reservedLabels = unlines $
  "hnsw-hs-metadata-v3" :
  ("metric=" ++ metricString (indexMetric idx)) :
  ("dimension=" ++ show (indexDim idx)) :
  ("max_elements=" ++ show (indexMaxElements idx)) :
  map (\l -> "live_label=" ++ show l) (Set.toAscList liveLabels) ++
  map (\l -> "reserved_label=" ++ show l) (Set.toAscList reservedLabels)

metricString :: Metric -> String
metricString L2 = "L2"
metricString InnerProduct = "InnerProduct"

metricCode :: Metric -> CInt
metricCode L2 = 0
metricCode InnerProduct = 1

catchIOException :: String -> IO a -> IO a
catchIOException context action =
  action `catch` \(e :: IOException) ->
    throwIO (HNSWException (context ++ displayException e))
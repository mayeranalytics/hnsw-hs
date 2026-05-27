{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}
-- |
-- Module      : Data.Vector.HNSW.Internal.AtomicSave
-- Description : Internal atomic-save helpers for tests.
--
-- This module is exposed for the package test suite and is not part of
-- the stable public API. Downstream users should not depend on it.
-- The names and behavior of this module may change without notice.
module Data.Vector.HNSW.Internal.AtomicSave
  ( -- * Atomic save operations for testing
    AtomicSaveOps(..)
  , realAtomicSaveOps
  , atomicSaveWith
  ) where

import Control.Concurrent.MVar (withMVar)
import Control.Exception (IOException, catch, displayException, onException, throwIO, try)
import Control.Monad (when)
import Foreign.C.String (peekCString, withCString)
import System.Directory
  ( createDirectory
  , doesDirectoryExist
  , doesFileExist
  , doesPathExist
  , removePathForcibly
  , renameDirectory
  )
import System.FilePath ((</>))

import Data.Vector.HNSW.Internal.Types
  ( Index(..), HNSWException(..), withIndex
  , c_hnsw_save_index, c_hnsw_last_error
  , formatMetadata, indexLiveLabels, indexReservedLabels
  , catchIOException )

-- | Operations for atomic save, injectable for testing.
data AtomicSaveOps = AtomicSaveOps
  { -- | Check if a path is an existing file.
    asoDoesFileExist      :: FilePath -> IO Bool
  , -- | Check if a path is an existing directory.
    asoDoesDirectoryExist :: FilePath -> IO Bool
  , -- | Check if a path exists (file or directory).
    asoDoesPathExist      :: FilePath -> IO Bool
  , -- | Create a directory.
    asoCreateDirectory    :: FilePath -> IO ()
  , -- | Rename a directory within the filesystem.
    asoRenameDirectory    :: FilePath -> FilePath -> IO ()
  , -- | Forcibly remove a path.
    asoRemovePathForcibly :: FilePath -> IO ()
  , -- | Write the index file.
    asoWriteIndex         :: Index -> FilePath -> IO ()
  , -- | Write the metadata file.
    asoWriteMetadata      :: FilePath -> String -> IO ()
  }

-- | Production @AtomicSaveOps@ using real filesystem operations.
realAtomicSaveOps :: AtomicSaveOps
realAtomicSaveOps = AtomicSaveOps
  { asoDoesFileExist      = doesFileExist
  , asoDoesDirectoryExist = doesDirectoryExist
  , asoDoesPathExist      = doesPathExist
  , asoCreateDirectory    = createDirectory
  , asoRenameDirectory    = renameDirectory
  , asoRemovePathForcibly = removePathForcibly
  ,asoWriteIndex          = writeIndexFile
  , asoWriteMetadata      = writeMetadataFile
  }

-- | Write the HNSW index file.
writeIndexFile :: Index -> FilePath -> IO ()
writeIndexFile idx path =
  withIndex idx $ \ptr ->
    withCString path $ \cPath -> do
      rc <- c_hnsw_save_index ptr cPath
      when (rc /= 0) $ do
        err <- c_hnsw_last_error >>= peekCString
        throwIO (HNSWException ("saveIndex failed: " ++ err))

-- | Write the metadata text file.
writeMetadataFile :: FilePath -> String -> IO ()
writeMetadataFile = writeFile

-- | Clean up a path, ignoring any errors.
cleanupPathIgnoreErrorsWith :: AtomicSaveOps -> FilePath -> IO ()
cleanupPathIgnoreErrorsWith ops p =
  asoRemovePathForcibly ops p `catch` \(_ :: IOException) -> pure ()

-- | Save an index to a directory using the given atomic-save operations.
atomicSaveWith :: AtomicSaveOps -> Index -> FilePath -> IO ()
atomicSaveWith ops idx path = do
  isFile <- catchIOException "saveIndex: failed to check path: " $
    asoDoesFileExist ops path
  when isFile $
    throwIO (HNSWException ("saveIndex: path is a regular file: " ++ path))

  tmpDir <- freshSiblingPathWith ops path ".tmp-"
  backupDir <- freshSiblingPathWith ops path ".old-"

  catchIOException "saveIndex: failed to create temp directory: " $
    asoCreateDirectory ops tmpDir

  let doWrite =
        withMVar (indexLiveLabels idx) $ \liveLabels ->
          withMVar (indexReservedLabels idx) $ \reservedLabels -> do
            asoWriteIndex ops idx (tmpDir </> "hnsw.index")
            catchIOException "saveIndex: failed to write metadata: " $
              asoWriteMetadata ops (tmpDir </> "metadata.txt")
                (formatMetadata idx liveLabels reservedLabels)

  (do
    doWrite
    publishDirectoryWith ops path tmpDir backupDir
    ) `onException` cleanupPathIgnoreErrorsWith ops tmpDir

-- | Find a sibling path that does not yet exist.
freshSiblingPathWith :: AtomicSaveOps -> FilePath -> String -> IO FilePath
freshSiblingPathWith ops base prefix = loop 0
  where
    loop n = do
      let candidate = base ++ prefix ++ show n
      exists <- asoDoesPathExist ops candidate
      if exists
        then if n >= 1000
               then throwIO (HNSWException "saveIndex: too many sibling paths")
               else loop (n + 1)
        else pure candidate

-- | Publish the temp directory as the target.
publishDirectoryWith :: AtomicSaveOps -> FilePath -> FilePath -> FilePath -> IO ()
publishDirectoryWith ops target tmp backup = do
  targetIsFile <- catchIOException "saveIndex: failed to check target at publish time: " $
    asoDoesFileExist ops target
  when targetIsFile $
    throwIO (HNSWException ("saveIndex: target became a regular file: " ++ target))

  targetIsDir <- catchIOException "saveIndex: failed to check target directory: " $
    asoDoesDirectoryExist ops target

  if not targetIsDir
    then catchIOException "saveIndex: failed to publish new directory: " $
           asoRenameDirectory ops tmp target
    else publishReplaceWith ops target tmp backup

-- | Replace an existing target directory with the temp directory.
publishReplaceWith :: AtomicSaveOps -> FilePath -> FilePath -> FilePath -> IO ()
publishReplaceWith ops target tmp backup = do
  catchIOException "saveIndex: failed to move existing target to backup: " $
    asoRenameDirectory ops target backup

  asoRenameDirectory ops tmp target `catch` \(publishErr :: IOException) -> do
    rollbackResult <- try (asoRenameDirectory ops backup target) :: IO (Either IOException ())
    case rollbackResult of
      Right () ->
        throwIO (HNSWException ("saveIndex: failed to publish new directory; restored previous target: " ++ displayException publishErr))
      Left rollbackErr ->
        throwIO (HNSWException
          ("saveIndex: failed to publish new directory and rollback failed; publish error: "
            ++ displayException publishErr
            ++ "; rollback error: "
            ++ displayException rollbackErr))

  cleanupPathIgnoreErrorsWith ops backup
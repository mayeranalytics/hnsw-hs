# hnsw-hs

`hnsw-hs` provides Haskell FFI bindings to [`hnswlib`](https://github.com/nmslib/hnswlib), a header-only C++ implementation of Hierarchical Navigable Small World (HNSW) approximate nearest-neighbor search.

## Install

```sh
stack build
stack test
```

## API overview

```haskell
import Data.Vector.HNSW

new :: Metric -> Int -> Int -> Int -> Int -> IO Index
insert :: Index -> Int -> Vector -> IO ()
insertMany :: Index -> [(Int, Vector)] -> IO ()
search :: Index -> Vector -> Int -> IO [(Int, Float)]
searchBatch :: Index -> [Vector] -> Int -> IO [[(Int, Float)]]
saveIndex :: Index -> FilePath -> IO ()
loadIndex :: FilePath -> IO Index
close :: Index -> IO ()
```

`Metric` is `L2` or `InnerProduct`. `Vector` is `Data.Vector.Storable.Vector Float`. Labels must be non-negative.

## Persistence

`saveIndex` writes a directory containing `metadata.txt` and `hnsw.index`. `loadIndex` reads it back. The format uses `live_label` and `reserved_label` fields.

## Benchmarks

```sh
stack build --bench --no-run-benchmarks
stack exec bench -- --list
```

Benchmark results are not published publicly yet. See the GitHub repository for the current benchmark policy.

## Vendoring

`hnswlib` is vendored at `src-include/hnswlib/` (not a submodule).

## License

BSD-3-Clause. See `LICENSE`.
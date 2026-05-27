#ifndef HNSW_H
#define HNSW_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hnsw_index hnsw_index_t;

typedef enum hnsw_metric {
  HNSW_METRIC_L2 = 0,
  HNSW_METRIC_INNER_PRODUCT = 1
} hnsw_metric_t;

hnsw_index_t* hnsw_create(
  hnsw_metric_t metric,
  size_t dim,
  size_t max_elements,
  size_t m,
  size_t ef_construction
);

void hnsw_free(hnsw_index_t* index);

int hnsw_add_point(hnsw_index_t* index, const float* vector, size_t dim, size_t label);

// hnsw_search_knn: search for k nearest neighbors.
// Output arrays labels_out and distances_out must have capacity for at least k entries.
// Returns 0 on success (count_out set to actual result count, which may be < k),
// -1 on error.
int hnsw_search_knn(
  hnsw_index_t* index,
  const float* vector,
  size_t dim,
  size_t k,
  size_t* labels_out,
  float* distances_out,
  size_t* count_out
);

int hnsw_save_index(hnsw_index_t* index, const char* path);

hnsw_index_t* hnsw_load_index(
  hnsw_metric_t metric,
  const char* path,
  size_t dim,
  size_t max_elements
);

int hnsw_set_ef(hnsw_index_t* index, size_t ef);

int hnsw_mark_delete(hnsw_index_t* index, size_t label);

int hnsw_update_point(hnsw_index_t* index, const float* vector, size_t dim, size_t label);

const char* hnsw_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
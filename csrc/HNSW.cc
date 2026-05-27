#include "hnswlib/hnswlib.h"
#include "HNSW.h"
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

thread_local std::string last_error_message;

struct hnsw_index_impl {
  hnswlib::SpaceInterface<float>* space;
  hnswlib::HierarchicalNSW<float>* index;
  hnsw_metric_t metric;
  size_t dim;
  size_t max_elements;
};

void clear_error() {
    last_error_message.clear();
}

void set_error(const char* msg) {
    last_error_message = msg ? msg : "unknown error";
}

void set_error_from_exception(const std::exception& e) {
    last_error_message = e.what();
}

void set_error_unknown() {
    last_error_message = "unknown exception";
}

} // anonymous namespace

extern "C" {

hnsw_index_t* hnsw_create(
  hnsw_metric_t metric,
  size_t dim,
  size_t max_elements,
  size_t m,
  size_t ef_construction
) {
    clear_error();

    hnswlib::SpaceInterface<float>* space = nullptr;
    switch (metric) {
        case HNSW_METRIC_L2:
            try {
                space = new hnswlib::L2Space(dim);
            } catch (const std::exception& e) {
                set_error_from_exception(e);
                return nullptr;
            } catch (...) {
                set_error_unknown();
                return nullptr;
            }
            break;
        case HNSW_METRIC_INNER_PRODUCT:
            try {
                space = new hnswlib::InnerProductSpace(dim);
            } catch (const std::exception& e) {
                set_error_from_exception(e);
                return nullptr;
            } catch (...) {
                set_error_unknown();
                return nullptr;
            }
            break;
        default:
            set_error("invalid metric");
            return nullptr;
    }

    hnswlib::HierarchicalNSW<float>* index = nullptr;
    try {
        index = new hnswlib::HierarchicalNSW<float>(space, max_elements, m, ef_construction);
    } catch (const std::exception& e) {
        delete space;
        set_error_from_exception(e);
        return nullptr;
    } catch (...) {
        delete space;
        set_error_unknown();
        return nullptr;
    }

    hnsw_index_impl* ctx = new hnsw_index_impl;
    ctx->space = space;
    ctx->index = index;
    ctx->metric = metric;
    ctx->dim = dim;
    ctx->max_elements = max_elements;
    return reinterpret_cast<hnsw_index_t*>(ctx);
}

void hnsw_free(hnsw_index_t* index) {
    if (!index) {
        return;
    }
    hnsw_index_impl* ctx = reinterpret_cast<hnsw_index_impl*>(index);
    delete ctx->index;
    delete ctx->space;
    delete ctx;
}

int hnsw_add_point(hnsw_index_t* index, const float* vector, size_t dim, size_t label) {
    clear_error();
    if (!index) {
        set_error("hnsw_add_point: index is null");
        return -1;
    }
    hnsw_index_impl* ctx = reinterpret_cast<hnsw_index_impl*>(index);
    if (!ctx->index) {
        set_error("hnsw_add_point: internal index is null");
        return -1;
    }
    if (!vector) {
        set_error("hnsw_add_point: vector is null");
        return -1;
    }
    if (dim != ctx->dim) {
        set_error("hnsw_add_point: dimension mismatch");
        return -1;
    }
    try {
        ctx->index->addPoint(vector, static_cast<hnswlib::labeltype>(label));
    } catch (const std::exception& e) {
        set_error_from_exception(e);
        return -1;
    } catch (...) {
        set_error_unknown();
        return -1;
    }
    return 0;
}

int hnsw_search_knn(
    hnsw_index_t* index,
    const float* vector,
    size_t dim,
    size_t k,
    size_t* labels_out,
    float* distances_out,
    size_t* count_out
) {
    clear_error();

    if (!count_out) {
        set_error("hnsw_search_knn: count_out is null");
        return -1;
    }
    *count_out = 0;

    if (!index) {
        set_error("hnsw_search_knn: index is null");
        return -1;
    }
    hnsw_index_impl* impl = reinterpret_cast<hnsw_index_impl*>(index);
    if (!impl->index) {
        set_error("hnsw_search_knn: internal index is null");
        return -1;
    }
    if (!vector) {
        set_error("hnsw_search_knn: vector is null");
        return -1;
    }
    if (!labels_out) {
        set_error("hnsw_search_knn: labels_out is null");
        return -1;
    }
    if (!distances_out) {
        set_error("hnsw_search_knn: distances_out is null");
        return -1;
    }
    if (k == 0) {
        set_error("hnsw_search_knn: k must be positive");
        return -1;
    }
    if (dim != impl->dim) {
        set_error("hnsw_search_knn: dimension mismatch");
        return -1;
    }
    try {
        auto results = impl->index->searchKnn(vector, k);
        size_t n = results.size();
        *count_out = n;
        std::vector<std::pair<float, size_t>> tmp;
        tmp.reserve(n);
        while (!results.empty()) {
            const auto item = results.top();
            tmp.emplace_back(
                static_cast<float>(item.first),
                static_cast<size_t>(item.second));
            results.pop();
        }
        for (size_t i = 0; i < n; ++i) {
            distances_out[i] = tmp[n - 1 - i].first;
            labels_out[i] = tmp[n - 1 - i].second;
        }
    } catch (const std::exception& e) {
        set_error_from_exception(e);
        return -1;
    } catch (...) {
        set_error_unknown();
        return -1;
    }
    return 0;
}

int hnsw_save_index(hnsw_index_t* index, const char* path) {
    clear_error();
    if (!index) {
        set_error("hnsw_save_index: index is null");
        return -1;
    }
    if (!path) {
        set_error("hnsw_save_index: path is null");
        return -1;
    }
    hnsw_index_impl* impl = reinterpret_cast<hnsw_index_impl*>(index);
    if (!impl->index) {
        set_error("hnsw_save_index: internal index is null");
        return -1;
    }
    try {
        impl->index->saveIndex(std::string(path));
    } catch (const std::exception& e) {
        set_error_from_exception(e);
        return -1;
    } catch (...) {
        set_error_unknown();
        return -1;
    }
    return 0;
}

hnsw_index_t* hnsw_load_index(
    hnsw_metric_t metric,
    const char* path,
    size_t dim,
    size_t max_elements
) {
    clear_error();
    if (!path) {
        set_error("hnsw_load_index: path is null");
        return nullptr;
    }
    if (dim == 0) {
        set_error("hnsw_load_index: dim must be positive");
        return nullptr;
    }
    if (max_elements == 0) {
        set_error("hnsw_load_index: max_elements must be positive");
        return nullptr;
    }
    hnswlib::SpaceInterface<float>* space = nullptr;
    switch (metric) {
        case HNSW_METRIC_L2:
            try {
                space = new hnswlib::L2Space(dim);
            } catch (const std::exception& e) {
                set_error_from_exception(e);
                return nullptr;
            } catch (...) {
                set_error_unknown();
                return nullptr;
            }
            break;
        case HNSW_METRIC_INNER_PRODUCT:
            try {
                space = new hnswlib::InnerProductSpace(dim);
            } catch (const std::exception& e) {
                set_error_from_exception(e);
                return nullptr;
            } catch (...) {
                set_error_unknown();
                return nullptr;
            }
            break;
        default:
            set_error("hnsw_load_index: invalid metric");
            return nullptr;
    }
    hnsw_index_impl* ctx = nullptr;
    hnswlib::HierarchicalNSW<float>* index = nullptr;
    try {
        index = new hnswlib::HierarchicalNSW<float>(space, std::string(path), false, max_elements);
        ctx = new hnsw_index_impl;
        ctx->space = space;
        ctx->index = index;
        ctx->metric = metric;
        ctx->dim = dim;
        ctx->max_elements = max_elements;
        return reinterpret_cast<hnsw_index_t*>(ctx);
    } catch (const std::exception& e) {
        delete index;
        delete space;
        set_error_from_exception(e);
        return nullptr;
    } catch (...) {
        delete index;
        delete space;
        set_error_unknown();
        return nullptr;
    }
}

const char* hnsw_last_error(void) {
    return last_error_message.c_str();
}

int hnsw_set_ef(hnsw_index_t* index, size_t ef) {
    clear_error();
    if (!index) {
        set_error("hnsw_set_ef: index is null");
        return -1;
    }
    if (ef == 0) {
        set_error("hnsw_set_ef: ef must be positive");
        return -1;
    }
    hnsw_index_impl* ctx = reinterpret_cast<hnsw_index_impl*>(index);
    if (!ctx->index) {
        set_error("hnsw_set_ef: internal index is null");
        return -1;
    }
    try {
        ctx->index->setEf(ef);
        return 0;
    } catch (const std::exception& e) {
        set_error_from_exception(e);
        return -1;
    } catch (...) {
        set_error_unknown();
        return -1;
    }
}

int hnsw_mark_delete(hnsw_index_t* index, size_t label) {
    clear_error();
    if (!index) {
        set_error("hnsw_mark_delete: index is null");
        return -1;
    }
    hnsw_index_impl* ctx = reinterpret_cast<hnsw_index_impl*>(index);
    if (!ctx->index) {
        set_error("hnsw_mark_delete: internal index is null");
        return -1;
    }
    try {
        ctx->index->markDelete(static_cast<hnswlib::labeltype>(label));
        return 0;
    } catch (const std::exception& e) {
        set_error_from_exception(e);
        return -1;
    } catch (...) {
        set_error_unknown();
        return -1;
    }
}

int hnsw_update_point(hnsw_index_t* index, const float* vector, size_t dim, size_t label) {
    clear_error();
    if (!index) {
        set_error("hnsw_update_point: index is null");
        return -1;
    }
    hnsw_index_impl* ctx = reinterpret_cast<hnsw_index_impl*>(index);
    if (!ctx->index) {
        set_error("hnsw_update_point: internal index is null");
        return -1;
    }
    if (!vector) {
        set_error("hnsw_update_point: vector is null");
        return -1;
    }
    if (dim != ctx->dim) {
        set_error("hnsw_update_point: dimension mismatch");
        return -1;
    }
    try {
        auto& mutex = ctx->index->getLabelOpMutex(static_cast<hnswlib::labeltype>(label));
        std::unique_lock<std::mutex> lock(mutex);
        auto it = ctx->index->label_lookup_.find(static_cast<hnswlib::labeltype>(label));
        if (it == ctx->index->label_lookup_.end()) {
            set_error("hnsw_update_point: label not found");
            return -1;
        }
        ctx->index->updatePoint(vector, it->second, 1.0f);
        return 0;
    } catch (const std::exception& e) {
        set_error_from_exception(e);
        return -1;
    } catch (...) {
        set_error_unknown();
        return -1;
    }
}

} // extern "C"
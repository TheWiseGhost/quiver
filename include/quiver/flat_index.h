// Exact brute-force index. The correctness baseline for the approximate indexes, and the right tool
// for small collections (it is often fastest below ~10k vectors).
//
// Thread-safe: searches run concurrently; add() takes an exclusive lock.
#pragma once

#include <cstddef>
#include <memory>
#include <shared_mutex>
#include <string>
#include <vector>

#include "quiver/common.h"
#include "quiver/filter.h"
#include "quiver/thread_pool.h"

namespace quiver {

class FlatIndex {
 public:
  explicit FlatIndex(std::size_t dim, Metric metric = Metric::kL2);

  /// Appends `n` row-major vectors. `labels` may be null, in which case vector i gets label
  /// size() + i. Labels are not checked for uniqueness. Holds at most 2^32 vectors.
  void add(const float* vectors, std::size_t n, const Label* labels = nullptr);

  /// Exact k nearest neighbors, closest first. Fewer than k results if the index (or filter
  /// admits) fewer than k vectors.
  std::vector<Neighbor> search(const float* query, std::size_t k,
                               const Filter* filter = nullptr) const;

  /// Searches `nq` row-major queries in parallel. Writes nq * k results; unfilled slots get
  /// kInvalidLabel and +infinity.
  void search_batch(const float* queries, std::size_t nq, std::size_t k, Label* labels,
                    float* distances, const Filter* filter = nullptr,
                    ThreadPool* pool = nullptr) const;

  std::size_t size() const;
  std::size_t dim() const noexcept { return dim_; }
  Metric metric() const noexcept { return metric_; }

  /// Stored vector `i` (normalized if the metric is cosine). Not synchronized with add().
  const float* vector(std::size_t i) const noexcept { return data_.data() + i * dim_; }
  Label label(std::size_t i) const noexcept { return labels_[i]; }

  void save(const std::string& path) const;
  static FlatIndex load(const std::string& path, const LoadOptions& options = {});

 private:
  void search_one(const float* query, std::size_t k, const Filter* filter, Label* labels,
                  float* distances, std::vector<float>& scratch) const;

  std::size_t dim_;
  Metric metric_;
  std::vector<float> data_;
  std::vector<Label> labels_;
  std::unique_ptr<std::shared_mutex> mu_ = std::make_unique<std::shared_mutex>();  // movable
};

}  // namespace quiver

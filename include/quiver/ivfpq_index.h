// IVF-PQ: inverted-file index with product-quantized residuals (Jégou et al., 2011).
//
// A coarse k-means splits space into `nlist` cells. Each vector is stored in its cell as the PQ
// code of its residual (vector - cell centroid), so memory is ~m bytes per vector. A query
// visits the `nprobe` closest cells and ranks their codes with ADC table lookups; optionally the
// best candidates are re-ranked with exact distances (`refine_factor`). This trades some recall
// for 10-30x less memory than HNSW: the method of choice when vectors do not fit in RAM.
//
// Concurrency: all methods are thread-safe. Searches run concurrently; add() encodes in parallel
// and takes an exclusive lock only to append. Cells are compared by L2 for kL2 and by inner
// product otherwise (cosine centroids are unit-normalized, so both views agree).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "quiver/common.h"
#include "quiver/filter.h"
#include "quiver/thread_pool.h"

namespace quiver {

struct IvfPqParams {
  std::size_t dim = 0;
  Metric metric = Metric::kL2;
  std::size_t nlist = 1024;  ///< coarse cells; ~sqrt(n) to 4*sqrt(n) is typical
  std::size_t m = 16;        ///< PQ sub-quantizers (bytes per vector); must divide dim
  std::size_t train_iterations = 20;
  /// Keep float32 copies of the vectors so results can be re-ranked exactly.
  bool store_vectors = false;
  std::uint64_t seed = 1234;
};

struct IvfPqSearchParams {
  std::size_t nprobe = 16;
  /// With store_vectors, re-rank the best k * refine_factor ADC candidates exactly (0 = off).
  std::size_t refine_factor = 0;
  const Filter* filter = nullptr;
};

class IvfPqIndex {
 public:
  explicit IvfPqIndex(const IvfPqParams& params);
  ~IvfPqIndex();
  IvfPqIndex(IvfPqIndex&&) noexcept;
  IvfPqIndex& operator=(IvfPqIndex&&) noexcept;

  /// Learns coarse centroids and the residual PQ codebooks. Needs n >= max(nlist, 256). An index
  /// is trained exactly once (a second call throws).
  void train(const float* vectors, std::size_t n, ThreadPool* pool = nullptr);
  bool is_trained() const noexcept;

  /// `labels` may be null (labels become size() + i). Labels are not checked for uniqueness.
  void add(const float* vectors, std::size_t n, const Label* labels = nullptr,
           ThreadPool* pool = nullptr);

  std::vector<Neighbor> search(const float* query, std::size_t k,
                               const IvfPqSearchParams& params = {}) const;
  void search_batch(const float* queries, std::size_t nq, std::size_t k, Label* labels,
                    float* distances, const IvfPqSearchParams& params = {},
                    ThreadPool* pool = nullptr) const;

  void save(const std::string& path) const;
  static IvfPqIndex load(const std::string& path, const LoadOptions& options = {});

  std::size_t size() const noexcept;
  std::size_t dim() const noexcept;
  const IvfPqParams& params() const noexcept;
  /// Bytes used by codes, labels, and (if stored) raw vectors.
  std::size_t memory_bytes() const noexcept;
  /// Number of vectors in each cell (for diagnosing imbalance).
  std::vector<std::size_t> list_sizes() const;

 private:
  struct Impl;
  explicit IvfPqIndex(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace quiver

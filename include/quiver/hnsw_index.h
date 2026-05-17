// Hierarchical Navigable Small World graph index (Malkov & Yashunin, 2018).
//
// Concurrency model
//   * search(), search_batch(), add() and remove() may all be called concurrently from any number
//     of threads. Searches take no locks: graph edges are published with release stores and read
//     with acquire loads, and node storage grows in fixed chunks that never move.
//   * Writers hold at most one per-node spinlock at a time, so inserts cannot deadlock.
//   * save() waits for in-flight add() calls and blocks new ones while it writes, so it always
//     captures a consistent graph (searches and removes are never blocked).
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

/// How vectors are stored inside the index.
enum class Storage : std::uint32_t {
  kFloat32 = 0,  ///< exact
  kSQ8 = 1,      ///< 8-bit scalar quantized: 4x less memory, small recall cost
};

struct HnswParams {
  std::size_t dim = 0;
  Metric metric = Metric::kL2;
  Storage storage = Storage::kFloat32;
  /// Max out-degree on layers >= 1. Layer 0 allows 2*M. Typical range 8-64.
  std::size_t M = 16;
  /// Beam width while building. Higher = better graph, slower inserts.
  std::size_t ef_construction = 200;
  /// Seeds the per-node level assignment; levels depend only on (seed, insertion slot).
  std::uint64_t seed = 0x5eed;
};

struct HnswSearchParams {
  /// Beam width at query time; raised to k if smaller. The main recall/latency knob.
  std::size_t ef = 64;
  /// Optional restriction of the result set.
  const Filter* filter = nullptr;
  /// When the filter knows its cardinality and admits at most this fraction of the index, the
  /// search scans the admitted vectors exactly instead of walking the graph (graph search
  /// degrades when most visited nodes are rejected). 0 disables the fallback.
  double exact_scan_threshold = 0.02;
};

class HnswIndex {
 public:
  explicit HnswIndex(const HnswParams& params);
  ~HnswIndex();
  HnswIndex(HnswIndex&&) noexcept;
  HnswIndex& operator=(HnswIndex&&) noexcept;

  /// SQ8 only: learns the quantization range from a representative sample. Can be called once.
  /// If it was not called, the first add() trains on its own batch (which must then hold at
  /// least 256 vectors, so the ranges are meaningful).
  void train(const float* vectors, std::size_t n);
  bool is_trained() const noexcept;

  /// Inserts `n` row-major vectors (finite values only). `labels` may be null, in which case
  /// vector i gets label `s + i`, where `s` counts every vector ever inserted (including removed
  /// ones), i.e. 0, 1, 2, ... in insertion order. Do not mix default and explicit labels unless
  /// the explicit ones cannot collide. Throws, leaving the index unchanged, if a label is already
  /// present (and not removed). Uses `pool` (or the global pool) to insert in parallel.
  void add(const float* vectors, std::size_t n, const Label* labels = nullptr,
           ThreadPool* pool = nullptr);

  /// Soft-deletes `label`: it stops appearing in results but keeps routing searches until the
  /// index is rebuilt. Returns false if the label is not present.
  bool remove(Label label);
  bool contains(Label label) const;

  /// The stored vector for `label` (decoded for SQ8, normalized for cosine).
  std::vector<float> get_vector(Label label) const;

  /// Approximate k nearest neighbors, closest first.
  std::vector<Neighbor> search(const float* query, std::size_t k,
                               const HnswSearchParams& params = {}) const;

  /// Searches `nq` row-major queries in parallel, writing nq * k results. Unfilled slots get
  /// kInvalidLabel and +infinity.
  void search_batch(const float* queries, std::size_t nq, std::size_t k, Label* labels,
                    float* distances, const HnswSearchParams& params = {},
                    ThreadPool* pool = nullptr) const;

  /// Pre-allocates room for `n` vectors (optional; storage also grows on demand).
  void reserve(std::size_t n);

  /// Atomically writes the index to `path` (temp file + fsync + rename).
  void save(const std::string& path) const;
  static HnswIndex load(const std::string& path, const LoadOptions& options = {});

  std::size_t size() const noexcept;  ///< live (non-removed) vectors
  std::size_t dim() const noexcept;
  Metric metric() const noexcept;
  Storage storage() const noexcept;
  const HnswParams& params() const noexcept;

  struct Stats {
    std::size_t num_nodes = 0;  ///< including removed
    std::size_t num_removed = 0;
    int max_level = -1;
    std::vector<std::size_t> nodes_per_level;
    double mean_degree_level0 = 0.0;
    std::size_t vector_bytes = 0;  ///< bytes of stored vector codes
    std::size_t graph_bytes = 0;   ///< bytes of adjacency lists
    bool memory_mapped = false;
  };
  Stats stats() const;

 private:
  struct Impl;
  explicit HnswIndex(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace quiver

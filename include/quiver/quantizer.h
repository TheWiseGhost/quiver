// Vector compression: per-dimension 8-bit scalar quantization and product quantization.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "quiver/common.h"
#include "quiver/thread_pool.h"

namespace quiver::internal {
class CentroidTable;
}

namespace quiver {

/// SQ8: each dimension is mapped linearly onto [0, 255] using the per-dimension range seen in
/// training. 4x smaller than float32; distances are computed asymmetrically (float query vs code),
/// so only the database side carries quantization error.
class ScalarQuantizer {
 public:
  ScalarQuantizer() = default;
  explicit ScalarQuantizer(std::size_t dim) : dim_(dim) {}

  /// Learns per-dimension [min, max] from `n` row-major training vectors.
  void train(const float* vectors, std::size_t n);
  /// Restores trained parameters (e.g. from a file). `vscale` must be positive and finite.
  void set_params(std::vector<float> vmin, std::vector<float> vscale);

  bool trained() const noexcept { return !vmin_.empty(); }
  std::size_t dim() const noexcept { return dim_; }
  std::size_t code_size() const noexcept { return dim_; }

  /// Values outside the trained range are clamped.
  void encode(const float* vector, std::uint8_t* code) const noexcept;
  void decode(const std::uint8_t* code, float* vector) const noexcept;

  const float* vmin() const noexcept { return vmin_.data(); }
  const float* vscale() const noexcept { return vscale_.data(); }

 private:
  std::size_t dim_ = 0;
  std::vector<float> vmin_;
  std::vector<float> vscale_;
};

struct KMeansParams {
  std::size_t k = 256;
  std::size_t iterations = 25;
  /// Training is subsampled to at most k * max_points_per_centroid points (as in FAISS).
  std::size_t max_points_per_centroid = 256;
  std::uint64_t seed = 1234;
};

/// Lloyd's k-means with parallel assignment and empty-cluster splitting. Returns k x dim
/// row-major centroids. Requires n >= k.
std::vector<float> kmeans(const float* vectors, std::size_t n, std::size_t dim,
                          const KMeansParams& params, ThreadPool* pool = nullptr);

/// Index of the centroid nearest to `x` (squared L2).
std::size_t nearest_centroid(const float* x, const float* centroids, std::size_t k,
                             std::size_t dim) noexcept;

/// PQ: the vector is split into `m` sub-vectors, each replaced by the id of its nearest of 256
/// sub-centroids, so a vector costs `m` bytes. Distances to a query are computed with
/// asymmetric distance computation (ADC): per query, build an m x 256 table of sub-distances,
/// then each code costs m table lookups and adds instead of `dim` multiply-adds.
class ProductQuantizer {
 public:
  static constexpr std::size_t kSubCentroids = 256;

  ProductQuantizer();
  /// `dim` must be divisible by `m`.
  ProductQuantizer(std::size_t dim, std::size_t m);
  ~ProductQuantizer();
  ProductQuantizer(const ProductQuantizer&);
  ProductQuantizer& operator=(const ProductQuantizer&);
  ProductQuantizer(ProductQuantizer&&) noexcept;
  ProductQuantizer& operator=(ProductQuantizer&&) noexcept;

  void train(const float* vectors, std::size_t n, ThreadPool* pool = nullptr,
             std::size_t iterations = 25, std::uint64_t seed = 1234);
  bool trained() const noexcept { return !centroids_.empty(); }

  std::size_t dim() const noexcept { return dim_; }
  std::size_t m() const noexcept { return m_; }
  std::size_t dsub() const noexcept { return dsub_; }
  std::size_t code_size() const noexcept { return m_; }

  void encode(const float* vector, std::uint8_t* code) const noexcept;
  void decode(const std::uint8_t* code, float* vector) const noexcept;

  /// table[j * 256 + c] = ||q_j - centroid_{j,c}||^2
  void compute_l2_table(const float* query, float* table) const noexcept;
  /// table[j * 256 + c] = <q_j, centroid_{j,c}>
  void compute_ip_table(const float* query, float* table) const noexcept;

  /// Sum of table entries selected by `code`.
  float adc(const float* table, const std::uint8_t* code) const noexcept;
  /// ADC for `n` contiguous codes; four codes are interleaved for instruction-level parallelism.
  void adc_batch(const float* table, const std::uint8_t* codes, std::size_t n,
                 float* out) const noexcept;

  /// m x 256 x dsub sub-centroids.
  const std::vector<float>& centroids() const noexcept { return centroids_; }
  void set_centroids(std::vector<float> centroids);

 private:
  std::size_t dim_ = 0;
  std::size_t m_ = 0;
  std::size_t dsub_ = 0;
  std::vector<float> centroids_;
  std::vector<internal::CentroidTable> tables_;  // per sub-space, transposed for fast scans
};

}  // namespace quiver

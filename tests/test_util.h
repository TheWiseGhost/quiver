#pragma once

#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "quiver/common.h"
#include "quiver/flat_index.h"

namespace quiver::testing {

/// n x dim standard normal vectors.
inline std::vector<float> random_vectors(std::size_t n, std::size_t dim, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> out(n * dim);
  for (float& x : out) x = dist(rng);
  return out;
}

/// Gaussian blobs around `clusters` centers: a harder, more realistic distribution than isotropic
/// noise. The centers depend only on (clusters, dim), so base and query sets drawn with different
/// seeds share the same distribution.
inline std::vector<float> clustered_vectors(std::size_t n, std::size_t dim, std::size_t clusters,
                                            std::uint64_t seed) {
  std::normal_distribution<float> unit(0.0f, 1.0f);
  std::mt19937_64 center_rng(clusters * 1000003 + dim);
  std::vector<float> centers(clusters * dim);
  for (float& x : centers) x = 2.0f * unit(center_rng);
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<std::size_t> pick(0, clusters - 1);
  std::vector<float> out(n * dim);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t c = pick(rng);
    for (std::size_t j = 0; j < dim; ++j) out[i * dim + j] = centers[c * dim + j] + unit(rng);
  }
  return out;
}

/// Mean fraction of the true top-k found, given nq x k result labels.
inline double recall_at_k(const std::vector<Label>& truth, const std::vector<Label>& got,
                          std::size_t nq, std::size_t k) {
  std::size_t hits = 0;
  for (std::size_t q = 0; q < nq; ++q) {
    std::unordered_set<Label> t(truth.begin() + static_cast<std::ptrdiff_t>(q * k),
                                truth.begin() + static_cast<std::ptrdiff_t>((q + 1) * k));
    for (std::size_t j = 0; j < k; ++j) hits += t.count(got[q * k + j]);
  }
  return static_cast<double>(hits) / static_cast<double>(nq * k);
}

/// Exact neighbors via FlatIndex.
inline std::vector<Label> ground_truth(const std::vector<float>& base,
                                       const std::vector<float>& queries, std::size_t dim,
                                       std::size_t k, Metric metric) {
  FlatIndex flat(dim, metric);
  flat.add(base.data(), base.size() / dim);
  const std::size_t nq = queries.size() / dim;
  std::vector<Label> labels(nq * k);
  std::vector<float> dist(nq * k);
  flat.search_batch(queries.data(), nq, k, labels.data(), dist.data());
  return labels;
}

/// A unique path in the system temp dir, removed on destruction.
class TempPath {
 public:
  explicit TempPath(const std::string& name) {
    static int counter = 0;
    path_ =
        (std::filesystem::temp_directory_path() / ("quiver_test_" + std::to_string(::getpid()) +
                                                   "_" + std::to_string(counter++) + "_" + name))
            .string();
  }
  ~TempPath() {
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }
  const std::string& str() const { return path_; }

 private:
  std::string path_;
};

}  // namespace quiver::testing

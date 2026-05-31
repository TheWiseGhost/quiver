#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>

#include "internal/centroid_table.h"
#include "quiver/distance.h"
#include "quiver/quantizer.h"

namespace quiver {

std::size_t nearest_centroid(const float* x, const float* centroids, std::size_t k,
                             std::size_t dim) noexcept {
  const auto l2 = simd::active().l2sqr;
  std::size_t best = 0;
  float best_d = std::numeric_limits<float>::infinity();
  for (std::size_t c = 0; c < k; ++c) {
    const float d = l2(x, centroids + c * dim, dim);
    if (d < best_d) {
      best_d = d;
      best = c;
    }
  }
  return best;
}

std::vector<float> kmeans(const float* vectors, std::size_t n, std::size_t dim,
                          const KMeansParams& params, ThreadPool* pool) {
  const std::size_t k = params.k;
  QUIVER_CHECK(k > 0 && dim > 0, "kmeans: k and dim must be positive");
  QUIVER_CHECK(n >= k, "kmeans: need at least k=" + std::to_string(k) + " training points, got " +
                           std::to_string(n));
  ThreadPool& tp = pool ? *pool : ThreadPool::global();
  std::mt19937_64 rng(params.seed);

  // Subsample: beyond ~256 points per centroid, more data barely moves the centroids.
  std::vector<float> sample;
  const float* x = vectors;
  const std::size_t cap = k * std::max<std::size_t>(1, params.max_points_per_centroid);
  if (n > cap) {
    std::vector<std::size_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    for (std::size_t i = 0; i < cap; ++i) {
      std::swap(perm[i], perm[i + rng() % (n - i)]);
    }
    sample.resize(cap * dim);
    for (std::size_t i = 0; i < cap; ++i) {
      std::memcpy(sample.data() + i * dim, vectors + perm[i] * dim, dim * sizeof(float));
    }
    x = sample.data();
    n = cap;
  }

  // Initialize from k distinct random points.
  std::vector<float> centroids(k * dim);
  {
    std::vector<std::size_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    for (std::size_t i = 0; i < k; ++i) {
      std::swap(perm[i], perm[i + rng() % (n - i)]);
      std::memcpy(centroids.data() + i * dim, x + perm[i] * dim, dim * sizeof(float));
    }
  }

  std::vector<std::uint32_t> assign(n);
  std::vector<double> sums(k * dim);
  std::vector<std::size_t> counts(k);
  internal::CentroidTable table;
  for (std::size_t iter = 0; iter < params.iterations; ++iter) {
    table.reset(centroids.data(), k, dim);
    tp.parallel_for(n, 256, [&](std::size_t b, std::size_t e) {
      std::vector<std::size_t> best(e - b);
      table.nearest_batch(x + b * dim, e - b, best.data());
      for (std::size_t i = b; i < e; ++i) assign[i] = static_cast<std::uint32_t>(best[i - b]);
    });
    std::fill(sums.begin(), sums.end(), 0.0);
    std::fill(counts.begin(), counts.end(), 0);
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t c = assign[i];
      ++counts[c];
      for (std::size_t j = 0; j < dim; ++j)
        sums[c * dim + j] += static_cast<double>(x[i * dim + j]);
    }
    for (std::size_t c = 0; c < k; ++c) {
      if (counts[c] == 0) continue;
      for (std::size_t j = 0; j < dim; ++j) {
        centroids[c * dim + j] =
            static_cast<float>(sums[c * dim + j] / static_cast<double>(counts[c]));
      }
    }
    // Empty clusters: split a large cluster by cloning its centroid with a tiny symmetric
    // perturbation, so the two copies separate on the next assignment.
    for (std::size_t c = 0; c < k; ++c) {
      if (counts[c] != 0) continue;
      std::size_t donor = 0;
      for (;;) {  // pick a donor with probability proportional to its size
        donor = rng() % k;
        if (static_cast<double>(counts[donor]) / static_cast<double>(n) * static_cast<double>(k) >
            std::uniform_real_distribution<double>(0.0, 1.0)(rng)) {
          break;
        }
      }
      constexpr float kEps = 1.0f / 1024.0f;
      for (std::size_t j = 0; j < dim; ++j) {
        const float v = centroids[donor * dim + j];
        const float sign = (j % 2 == 0) ? 1.0f : -1.0f;
        centroids[c * dim + j] = v * (1.0f + sign * kEps);
        centroids[donor * dim + j] = v * (1.0f - sign * kEps);
      }
      counts[c] = counts[donor] / 2;
      counts[donor] -= counts[c];
    }
  }
  return centroids;
}

ProductQuantizer::ProductQuantizer() = default;
ProductQuantizer::~ProductQuantizer() = default;
ProductQuantizer::ProductQuantizer(const ProductQuantizer&) = default;
ProductQuantizer& ProductQuantizer::operator=(const ProductQuantizer&) = default;
ProductQuantizer::ProductQuantizer(ProductQuantizer&&) noexcept = default;
ProductQuantizer& ProductQuantizer::operator=(ProductQuantizer&&) noexcept = default;

ProductQuantizer::ProductQuantizer(std::size_t dim, std::size_t m) : dim_(dim), m_(m) {
  QUIVER_CHECK(
      dim > 0 && m > 0 && dim % m == 0,
      "PQ: dimension " + std::to_string(dim) + " must be divisible by m=" + std::to_string(m));
  dsub_ = dim / m;
}

void ProductQuantizer::train(const float* vectors, std::size_t n, ThreadPool* pool,
                             std::size_t iterations, std::uint64_t seed) {
  QUIVER_CHECK(m_ > 0, "PQ is not configured");
  QUIVER_CHECK(n >= kSubCentroids,
               "PQ training needs at least 256 vectors, got " + std::to_string(n));
  std::vector<float> centroids(m_ * kSubCentroids * dsub_);
  std::vector<float> sub(n * dsub_);
  for (std::size_t j = 0; j < m_; ++j) {
    for (std::size_t i = 0; i < n; ++i) {
      std::memcpy(sub.data() + i * dsub_, vectors + i * dim_ + j * dsub_, dsub_ * sizeof(float));
    }
    KMeansParams kp;
    kp.k = kSubCentroids;
    kp.iterations = iterations;
    kp.seed = seed + j;
    const auto c = kmeans(sub.data(), n, dsub_, kp, pool);
    std::copy(c.begin(), c.end(),
              centroids.begin() + static_cast<std::ptrdiff_t>(j * kSubCentroids * dsub_));
  }
  set_centroids(std::move(centroids));
}

void ProductQuantizer::set_centroids(std::vector<float> centroids) {
  QUIVER_CHECK(centroids.size() == m_ * kSubCentroids * dsub_,
               "PQ centroid table has the wrong size");
  centroids_ = std::move(centroids);
  tables_.resize(m_);
  for (std::size_t j = 0; j < m_; ++j) {
    tables_[j].reset(centroids_.data() + j * kSubCentroids * dsub_, kSubCentroids, dsub_);
  }
}

void ProductQuantizer::encode(const float* vector, std::uint8_t* code) const noexcept {
  float dist[kSubCentroids];
  for (std::size_t j = 0; j < m_; ++j) {
    tables_[j].l2(vector + j * dsub_, dist);  // 256 is already a multiple of the tile width
    code[j] = static_cast<std::uint8_t>(tables_[j].argmin(dist));
  }
}

void ProductQuantizer::decode(const std::uint8_t* code, float* vector) const noexcept {
  for (std::size_t j = 0; j < m_; ++j) {
    std::memcpy(vector + j * dsub_, centroids_.data() + (j * kSubCentroids + code[j]) * dsub_,
                dsub_ * sizeof(float));
  }
}

void ProductQuantizer::compute_l2_table(const float* query, float* table) const noexcept {
  for (std::size_t j = 0; j < m_; ++j) tables_[j].l2(query + j * dsub_, table + j * kSubCentroids);
}

void ProductQuantizer::compute_ip_table(const float* query, float* table) const noexcept {
  for (std::size_t j = 0; j < m_; ++j) tables_[j].dot(query + j * dsub_, table + j * kSubCentroids);
}

float ProductQuantizer::adc(const float* table, const std::uint8_t* code) const noexcept {
  float sum = 0.0f;
  for (std::size_t j = 0; j < m_; ++j) sum += table[j * kSubCentroids + code[j]];
  return sum;
}

void ProductQuantizer::adc_batch(const float* table, const std::uint8_t* codes, std::size_t n,
                                 float* out) const noexcept {
  std::size_t i = 0;
  // Four independent accumulation chains hide the latency of the dependent table loads.
  for (; i + 4 <= n; i += 4) {
    const std::uint8_t* c0 = codes + i * m_;
    const std::uint8_t* c1 = c0 + m_;
    const std::uint8_t* c2 = c1 + m_;
    const std::uint8_t* c3 = c2 + m_;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (std::size_t j = 0; j < m_; ++j) {
      const float* t = table + j * kSubCentroids;
      s0 += t[c0[j]];
      s1 += t[c1[j]];
      s2 += t[c2[j]];
      s3 += t[c3[j]];
    }
    out[i] = s0;
    out[i + 1] = s1;
    out[i + 2] = s2;
    out[i + 3] = s3;
  }
  for (; i < n; ++i) out[i] = adc(table, codes + i * m_);
}

}  // namespace quiver

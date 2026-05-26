// Distances from vectors to many centroids, used by k-means, PQ encoding, and IVF probing.
//
// Centroids are stored transposed (dimension-major) and padded to a multiple of kTileC, so the
// inner loops run across centroids. Those loops are element-wise (no horizontal reduction), which
// lets the compiler vectorize them without -ffast-math and avoids any per-centroid call overhead
// - crucial for tiny vectors like 4-8 dimensional PQ sub-vectors.
//
// nearest_batch() goes further with a register-blocked microkernel: a tile of kTileP points x
// kTileC centroids is accumulated entirely in vector registers (4 x 16 floats = 16 NEON / 8 AVX
// registers), so each centroid value loaded from memory feeds kTileP fused multiply-adds instead
// of one load-modify-store. This is the same idea as a GEMM microkernel.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <vector>

namespace quiver::internal {

/// Portable 4 x float SIMD vector (GCC/Clang vector extension): lowers to NEON on AArch64 and
/// SSE on x86-64, with no intrinsics or per-ISA code.
using f32x4 = float __attribute__((vector_size(16)));

inline f32x4 load4(const float* p) noexcept {
  f32x4 v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}
inline void store4(float* p, f32x4 v) noexcept { std::memcpy(p, &v, sizeof(v)); }

class CentroidTable {
 public:
  static constexpr std::size_t kTileP = 4;   // points per microkernel tile
  static constexpr std::size_t kTileC = 16;  // centroids per microkernel tile

  CentroidTable() = default;
  CentroidTable(const float* centroids, std::size_t k, std::size_t dim) {
    reset(centroids, k, dim);
  }

  void reset(const float* centroids, std::size_t k, std::size_t dim) {
    k_ = k;
    dim_ = dim;
    kpad_ = (k + kTileC - 1) / kTileC * kTileC;
    t_.assign(kpad_ * dim, 0.0f);
    // Padding centroids get +inf norms so they can never win an argmin.
    norms_.assign(kpad_, std::numeric_limits<float>::infinity());
    for (std::size_t c = 0; c < k; ++c) {
      float norm = 0.0f;
      for (std::size_t j = 0; j < dim; ++j) {
        const float v = centroids[c * dim + j];
        t_[j * kpad_ + c] = v;
        norm += v * v;
      }
      norms_[c] = norm;
    }
  }

  std::size_t k() const noexcept { return k_; }

  /// out[c] = ||x - centroid_c||^2 for c < k, as an exact sum of squared differences.
  /// `out` must hold padded_k() floats.
  void l2(const float* x, float* out) const noexcept {
    std::fill(out, out + kpad_, 0.0f);
    for (std::size_t j = 0; j < dim_; ++j) {
      const float xj = x[j];
      const float* row = t_.data() + j * kpad_;
      for (std::size_t c = 0; c < kpad_; ++c) {
        const float d = xj - row[c];
        out[c] += d * d;
      }
    }
  }

  /// out[c] = <x, centroid_c>. `out` must hold padded_k() floats.
  void dot(const float* x, float* out) const noexcept {
    std::fill(out, out + kpad_, 0.0f);
    for (std::size_t j = 0; j < dim_; ++j) {
      const float xj = x[j];
      const float* row = t_.data() + j * kpad_;
      for (std::size_t c = 0; c < kpad_; ++c) out[c] += xj * row[c];
    }
  }

  std::size_t padded_k() const noexcept { return kpad_; }

  /// Index of the smallest of the first k values.
  std::size_t argmin(const float* v) const noexcept { return argmin_n(v, k_); }

  /// Nearest centroid of each of `n` row-major points: by squared L2 via ||c||^2 - 2<x, c> (the
  /// ||x||^2 term does not affect the argmin), or with `max_inner_product` by largest <x, c>.
  void nearest_batch(const float* x, std::size_t n, std::size_t* out,
                     bool max_inner_product = false) const {
    float tile_x[kTileP * kMaxTileDim];
    std::vector<float> xbuf;
    float best_d[kTileP];
    for (std::size_t p0 = 0; p0 < n; p0 += kTileP) {
      const std::size_t np = std::min(kTileP, n - p0);
      // Gather the tile transposed (dim-major), zero-padding missing points.
      float* xt = tile_x;
      if (dim_ > kMaxTileDim) {
        xbuf.resize(kTileP * dim_);
        xt = xbuf.data();
      }
      for (std::size_t j = 0; j < dim_; ++j) {
        for (std::size_t p = 0; p < kTileP; ++p)
          xt[j * kTileP + p] = p < np ? x[(p0 + p) * dim_ + j] : 0.0f;
      }
      for (std::size_t p = 0; p < kTileP; ++p) best_d[p] = std::numeric_limits<float>::infinity();
      for (std::size_t c0 = 0; c0 < kpad_; c0 += kTileC) {
        // 4 points x 16 centroids = 16 vector accumulators, all register-resident.
        f32x4 acc[kTileP][kTileC / 4] = {};
        for (std::size_t j = 0; j < dim_; ++j) {
          const float* row = t_.data() + j * kpad_ + c0;
          const f32x4 r0 = load4(row), r1 = load4(row + 4), r2 = load4(row + 8),
                      r3 = load4(row + 12);
          const float* xj = xt + j * kTileP;
          for (std::size_t p = 0; p < kTileP; ++p) {
            const f32x4 xv = f32x4{} + xj[p];  // broadcast
            acc[p][0] += xv * r0;
            acc[p][1] += xv * r1;
            acc[p][2] += xv * r2;
            acc[p][3] += xv * r3;
          }
        }
        float dots[kTileP][kTileC];
        for (std::size_t p = 0; p < kTileP; ++p) {
          for (std::size_t q = 0; q < kTileC / 4; ++q) store4(&dots[p][q * 4], acc[p][q]);
        }
        const std::size_t valid = std::min(kTileC, k_ - c0);  // skip padding centroids
        for (std::size_t p = 0; p < np; ++p) {
          for (std::size_t c = 0; c < valid; ++c) {
            const float d = max_inner_product ? -dots[p][c] : norms_[c0 + c] - 2.0f * dots[p][c];
            if (d < best_d[p]) {
              best_d[p] = d;
              out[p0 + p] = c0 + c;
            }
          }
        }
      }
    }
  }

  /// Nearest centroid of a single point.
  std::size_t nearest(const float* x, bool max_inner_product = false) const {
    std::size_t out = 0;
    nearest_batch(x, 1, &out, max_inner_product);
    return out;
  }

  /// Index of the smallest of v[0..n). Two passes with independent min chains, which is several
  /// times faster than the branchy single pass (whose loop-carried dependency serializes).
  static std::size_t argmin_n(const float* v, std::size_t n) noexcept {
    float m0 = std::numeric_limits<float>::infinity(), m1 = m0, m2 = m0, m3 = m0;
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
      m0 = std::min(m0, v[i]);
      m1 = std::min(m1, v[i + 1]);
      m2 = std::min(m2, v[i + 2]);
      m3 = std::min(m3, v[i + 3]);
    }
    for (; i < n; ++i) m0 = std::min(m0, v[i]);
    const float m = std::min(std::min(m0, m1), std::min(m2, m3));
    for (std::size_t j = 0; j < n; ++j) {
      if (v[j] == m) return j;
    }
    return 0;  // all NaN
  }

 private:
  static constexpr std::size_t kMaxTileDim = 256;

  std::size_t k_ = 0;
  std::size_t kpad_ = 0;
  std::size_t dim_ = 0;
  std::vector<float> t_;      // dim x kpad
  std::vector<float> norms_;  // ||centroid_c||^2, +inf for padding
};

}  // namespace quiver::internal

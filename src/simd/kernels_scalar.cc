// Portable reference kernels. These are deliberately plain loops: without -ffast-math the compiler
// cannot reassociate the float reductions, so they stay scalar and serve as an honest baseline.
#include "simd/kernels.h"

namespace quiver::simd::scalar {

float l2sqr(const float* a, const float* b, std::size_t dim) noexcept {
  float sum = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

float dot(const float* a, const float* b, std::size_t dim) noexcept {
  float sum = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

float l2sqr_sq8(const float* q, const std::uint8_t* code, const float* vmin, const float* vscale,
                std::size_t dim) noexcept {
  float sum = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) {
    const float d = q[i] - (vmin[i] + static_cast<float>(code[i]) * vscale[i]);
    sum += d * d;
  }
  return sum;
}

float dot_sq8(const float* q, const std::uint8_t* code, const float* vmin, const float* vscale,
              std::size_t dim) noexcept {
  float sum = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) {
    sum += q[i] * (vmin[i] + static_cast<float>(code[i]) * vscale[i]);
  }
  return sum;
}

}  // namespace quiver::simd::scalar

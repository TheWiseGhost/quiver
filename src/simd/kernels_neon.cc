// AArch64 NEON kernels for the runtime dispatch table. The bodies live in kernels_neon_inl.h so
// the HNSW search loop can also inline them directly.
#include "simd/kernels.h"

#if QUIVER_HAVE_NEON

#include "simd/kernels_neon_inl.h"

namespace quiver::simd::neon {

float l2sqr(const float* a, const float* b, std::size_t dim) noexcept {
  return l2sqr_inl(a, b, dim);
}
float dot(const float* a, const float* b, std::size_t dim) noexcept { return dot_inl(a, b, dim); }
float l2sqr_sq8(const float* q, const std::uint8_t* code, const float* vmin, const float* vscale,
                std::size_t dim) noexcept {
  return l2sqr_sq8_inl(q, code, vmin, vscale, dim);
}
float dot_sq8(const float* q, const std::uint8_t* code, const float* vmin, const float* vscale,
              std::size_t dim) noexcept {
  return dot_sq8_inl(q, code, vmin, vscale, dim);
}

}  // namespace quiver::simd::neon

#endif  // QUIVER_HAVE_NEON

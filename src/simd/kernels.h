// Per-ISA kernel declarations. Only dispatch.cc should include this.
#pragma once

#include <cstddef>
#include <cstdint>

#define QUIVER_DECLARE_KERNELS(ns)                                                                \
  namespace quiver::simd::ns {                                                                    \
  float l2sqr(const float* a, const float* b, std::size_t dim) noexcept;                          \
  float dot(const float* a, const float* b, std::size_t dim) noexcept;                            \
  float l2sqr_sq8(const float* q, const std::uint8_t* code, const float* vmin,                    \
                  const float* vscale, std::size_t dim) noexcept;                                 \
  float dot_sq8(const float* q, const std::uint8_t* code, const float* vmin, const float* vscale, \
                std::size_t dim) noexcept;                                                        \
  }

QUIVER_DECLARE_KERNELS(scalar)

#if defined(__aarch64__) && defined(__ARM_NEON)
#define QUIVER_HAVE_NEON 1
QUIVER_DECLARE_KERNELS(neon)
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define QUIVER_HAVE_AVX2 1
QUIVER_DECLARE_KERNELS(avx2)
#endif

#undef QUIVER_DECLARE_KERNELS

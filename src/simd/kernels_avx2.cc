// x86-64 AVX2 + FMA kernels.
//
// Functions carry __attribute__((target("avx2,fma"))) instead of the whole file being compiled with
// -mavx2, so the rest of the binary stays runnable on older CPUs; dispatch.cc only hands these out
// after CPUID confirms support.
#include "simd/kernels.h"

#if QUIVER_HAVE_AVX2

#include <immintrin.h>

#define QUIVER_AVX2 __attribute__((target("avx2,fma")))

namespace quiver::simd::avx2 {
namespace {

QUIVER_AVX2 inline float hsum(__m256 v) noexcept {
  __m128 lo = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  __m128 shuf = _mm_movehdup_ps(lo);
  __m128 sums = _mm_add_ps(lo, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  return _mm_cvtss_f32(_mm_add_ss(sums, shuf));
}

// Loads 8 codes and decodes them to vmin + code * vscale.
QUIVER_AVX2 inline __m256 sq8_decode8(const std::uint8_t* code, const float* vmin,
                                      const float* vscale) noexcept {
  const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(code));
  const __m256 cf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes));
  return _mm256_fmadd_ps(cf, _mm256_loadu_ps(vscale), _mm256_loadu_ps(vmin));
}

}  // namespace

QUIVER_AVX2 float l2sqr(const float* a, const float* b, std::size_t dim) noexcept {
  __m256 s0 = _mm256_setzero_ps();
  __m256 s1 = _mm256_setzero_ps();
  __m256 s2 = _mm256_setzero_ps();
  __m256 s3 = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 32 <= dim; i += 32) {
    const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
    const __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16));
    const __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24));
    s0 = _mm256_fmadd_ps(d0, d0, s0);
    s1 = _mm256_fmadd_ps(d1, d1, s1);
    s2 = _mm256_fmadd_ps(d2, d2, s2);
    s3 = _mm256_fmadd_ps(d3, d3, s3);
  }
  for (; i + 8 <= dim; i += 8) {
    const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    s0 = _mm256_fmadd_ps(d0, d0, s0);
  }
  float sum = hsum(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
  for (; i < dim; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

QUIVER_AVX2 float dot(const float* a, const float* b, std::size_t dim) noexcept {
  __m256 s0 = _mm256_setzero_ps();
  __m256 s1 = _mm256_setzero_ps();
  __m256 s2 = _mm256_setzero_ps();
  __m256 s3 = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 32 <= dim; i += 32) {
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
    s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
    s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
  }
  for (; i + 8 <= dim; i += 8) {
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
  }
  float sum = hsum(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
  for (; i < dim; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

QUIVER_AVX2 float l2sqr_sq8(const float* q, const std::uint8_t* code, const float* vmin,
                            const float* vscale, std::size_t dim) noexcept {
  __m256 s0 = _mm256_setzero_ps();
  __m256 s1 = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    const __m256 d0 =
        _mm256_sub_ps(_mm256_loadu_ps(q + i), sq8_decode8(code + i, vmin + i, vscale + i));
    const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(q + i + 8),
                                    sq8_decode8(code + i + 8, vmin + i + 8, vscale + i + 8));
    s0 = _mm256_fmadd_ps(d0, d0, s0);
    s1 = _mm256_fmadd_ps(d1, d1, s1);
  }
  for (; i + 8 <= dim; i += 8) {
    const __m256 d0 =
        _mm256_sub_ps(_mm256_loadu_ps(q + i), sq8_decode8(code + i, vmin + i, vscale + i));
    s0 = _mm256_fmadd_ps(d0, d0, s0);
  }
  float sum = hsum(_mm256_add_ps(s0, s1));
  for (; i < dim; ++i) {
    const float d = q[i] - (vmin[i] + static_cast<float>(code[i]) * vscale[i]);
    sum += d * d;
  }
  return sum;
}

QUIVER_AVX2 float dot_sq8(const float* q, const std::uint8_t* code, const float* vmin,
                          const float* vscale, std::size_t dim) noexcept {
  __m256 s0 = _mm256_setzero_ps();
  __m256 s1 = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + i), sq8_decode8(code + i, vmin + i, vscale + i), s0);
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(q + i + 8),
                         sq8_decode8(code + i + 8, vmin + i + 8, vscale + i + 8), s1);
  }
  for (; i + 8 <= dim; i += 8) {
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + i), sq8_decode8(code + i, vmin + i, vscale + i), s0);
  }
  float sum = hsum(_mm256_add_ps(s0, s1));
  for (; i < dim; ++i) {
    sum += q[i] * (vmin[i] + static_cast<float>(code[i]) * vscale[i]);
  }
  return sum;
}

}  // namespace quiver::simd::avx2

#endif  // QUIVER_HAVE_AVX2

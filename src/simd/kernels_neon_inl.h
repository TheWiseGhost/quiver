// NEON kernel bodies as inline functions, so code that is NEON-only by construction (AArch64,
// where NEON is mandatory) can call them directly and let the compiler inline them into hot loops.
// kernels_neon.cc wraps these for the runtime dispatch table.
#pragma once

#if defined(__aarch64__) && defined(__ARM_NEON)

#include <arm_neon.h>

#include <cstddef>
#include <cstdint>

namespace quiver::simd::neon {

inline float32x4_t sq8_decode(uint16x4_t c, const float* vmin, const float* vscale) noexcept {
  const float32x4_t cf = vcvtq_f32_u32(vmovl_u16(c));
  return vfmaq_f32(vld1q_f32(vmin), cf, vld1q_f32(vscale));
}

inline float l2sqr_inl(const float* a, const float* b, std::size_t dim) noexcept {
  float32x4_t s0 = vdupq_n_f32(0.0f);
  float32x4_t s1 = vdupq_n_f32(0.0f);
  float32x4_t s2 = vdupq_n_f32(0.0f);
  float32x4_t s3 = vdupq_n_f32(0.0f);
  std::size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    const float32x4_t d0 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
    const float32x4_t d1 = vsubq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    const float32x4_t d2 = vsubq_f32(vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    const float32x4_t d3 = vsubq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    s0 = vfmaq_f32(s0, d0, d0);
    s1 = vfmaq_f32(s1, d1, d1);
    s2 = vfmaq_f32(s2, d2, d2);
    s3 = vfmaq_f32(s3, d3, d3);
  }
  for (; i + 4 <= dim; i += 4) {
    const float32x4_t d0 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
    s0 = vfmaq_f32(s0, d0, d0);
  }
  float sum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
  for (; i < dim; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

inline float dot_inl(const float* a, const float* b, std::size_t dim) noexcept {
  float32x4_t s0 = vdupq_n_f32(0.0f);
  float32x4_t s1 = vdupq_n_f32(0.0f);
  float32x4_t s2 = vdupq_n_f32(0.0f);
  float32x4_t s3 = vdupq_n_f32(0.0f);
  std::size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
    s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
  }
  for (; i + 4 <= dim; i += 4) {
    s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
  }
  float sum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
  for (; i < dim; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

inline float l2sqr_sq8_inl(const float* q, const std::uint8_t* code, const float* vmin,
                           const float* vscale, std::size_t dim) noexcept {
  float32x4_t s0 = vdupq_n_f32(0.0f);
  float32x4_t s1 = vdupq_n_f32(0.0f);
  float32x4_t s2 = vdupq_n_f32(0.0f);
  float32x4_t s3 = vdupq_n_f32(0.0f);
  std::size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    // Widen 16 x u8 -> 2 x (8 x u16) -> 4 x (4 x f32), then decode with one FMA each.
    const uint8x16_t c = vld1q_u8(code + i);
    const uint16x8_t lo = vmovl_u8(vget_low_u8(c));
    const uint16x8_t hi = vmovl_high_u8(c);
    const float32x4_t d0 =
        vsubq_f32(vld1q_f32(q + i), sq8_decode(vget_low_u16(lo), vmin + i, vscale + i));
    const float32x4_t d1 = vsubq_f32(vld1q_f32(q + i + 4),
                                     sq8_decode(vget_high_u16(lo), vmin + i + 4, vscale + i + 4));
    const float32x4_t d2 =
        vsubq_f32(vld1q_f32(q + i + 8), sq8_decode(vget_low_u16(hi), vmin + i + 8, vscale + i + 8));
    const float32x4_t d3 = vsubq_f32(vld1q_f32(q + i + 12),
                                     sq8_decode(vget_high_u16(hi), vmin + i + 12, vscale + i + 12));
    s0 = vfmaq_f32(s0, d0, d0);
    s1 = vfmaq_f32(s1, d1, d1);
    s2 = vfmaq_f32(s2, d2, d2);
    s3 = vfmaq_f32(s3, d3, d3);
  }
  float sum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
  for (; i < dim; ++i) {
    const float d = q[i] - (vmin[i] + static_cast<float>(code[i]) * vscale[i]);
    sum += d * d;
  }
  return sum;
}

inline float dot_sq8_inl(const float* q, const std::uint8_t* code, const float* vmin,
                         const float* vscale, std::size_t dim) noexcept {
  float32x4_t s0 = vdupq_n_f32(0.0f);
  float32x4_t s1 = vdupq_n_f32(0.0f);
  float32x4_t s2 = vdupq_n_f32(0.0f);
  float32x4_t s3 = vdupq_n_f32(0.0f);
  std::size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    const uint8x16_t c = vld1q_u8(code + i);
    const uint16x8_t lo = vmovl_u8(vget_low_u8(c));
    const uint16x8_t hi = vmovl_high_u8(c);
    s0 = vfmaq_f32(s0, vld1q_f32(q + i), sq8_decode(vget_low_u16(lo), vmin + i, vscale + i));
    s1 = vfmaq_f32(s1, vld1q_f32(q + i + 4),
                   sq8_decode(vget_high_u16(lo), vmin + i + 4, vscale + i + 4));
    s2 = vfmaq_f32(s2, vld1q_f32(q + i + 8),
                   sq8_decode(vget_low_u16(hi), vmin + i + 8, vscale + i + 8));
    s3 = vfmaq_f32(s3, vld1q_f32(q + i + 12),
                   sq8_decode(vget_high_u16(hi), vmin + i + 12, vscale + i + 12));
  }
  float sum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
  for (; i < dim; ++i) {
    sum += q[i] * (vmin[i] + static_cast<float>(code[i]) * vscale[i]);
  }
  return sum;
}

}  // namespace quiver::simd::neon

#endif

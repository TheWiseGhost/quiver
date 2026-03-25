// SIMD distance kernels with runtime ISA dispatch.
//
// Every kernel exists in a portable scalar form plus hand-vectorized NEON (AArch64) and AVX2+FMA
// (x86-64) variants. The best variant the CPU supports is chosen once at startup; the environment
// variable QUIVER_SIMD=scalar|neon|avx2 overrides the choice (useful for benchmarking and for
// exercising the AVX2 path under emulators that do not advertise it via CPUID).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace quiver::simd {

enum class Isa : std::uint8_t { kScalar = 0, kNeon = 1, kAvx2 = 2 };

std::string_view to_string(Isa isa) noexcept;

/// Float32 kernel: a and b each hold `dim` floats. No alignment requirement.
using FloatKernel = float (*)(const float* a, const float* b, std::size_t dim) noexcept;

/// Asymmetric SQ8 kernel: compares a float query against a uint8 code that decodes to
/// x[i] = vmin[i] + code[i] * vscale[i].
using Sq8Kernel = float (*)(const float* query, const std::uint8_t* code, const float* vmin,
                            const float* vscale, std::size_t dim) noexcept;

struct KernelTable {
  Isa isa;
  FloatKernel l2sqr;    ///< sum (a_i - b_i)^2
  FloatKernel dot;      ///< sum a_i * b_i
  Sq8Kernel l2sqr_sq8;  ///< sum (q_i - x_i)^2
  Sq8Kernel dot_sq8;    ///< sum q_i * x_i
};

/// The table every index uses. Resolved once; thread-safe.
const KernelTable& active() noexcept;

/// The table for a specific ISA, or nullptr when it is not compiled in or not runnable here.
/// An ISA forced through QUIVER_SIMD counts as runnable.
const KernelTable* get(Isa isa) noexcept;

/// All ISAs `get` would return a table for, scalar first.
std::vector<Isa> available();

/// Scales `v` to unit L2 norm in place (no-op for the zero vector).
void normalize(float* v, std::size_t dim) noexcept;

/// Hints the CPU to pull `bytes` starting at `p` into cache.
inline void prefetch(const void* p, std::size_t bytes = 64) noexcept {
  const auto* c = static_cast<const char*>(p);
  for (std::size_t off = 0; off < bytes; off += 64) {
    __builtin_prefetch(c + off, 0, 3);
  }
}

}  // namespace quiver::simd

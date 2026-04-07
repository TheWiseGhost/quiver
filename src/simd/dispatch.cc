#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "quiver/distance.h"
#include "simd/kernels.h"

namespace quiver::simd {
namespace {

constexpr KernelTable kScalar{Isa::kScalar, scalar::l2sqr, scalar::dot, scalar::l2sqr_sq8,
                              scalar::dot_sq8};
#if QUIVER_HAVE_NEON
constexpr KernelTable kNeon{Isa::kNeon, neon::l2sqr, neon::dot, neon::l2sqr_sq8, neon::dot_sq8};
#endif
#if QUIVER_HAVE_AVX2
constexpr KernelTable kAvx2{Isa::kAvx2, avx2::l2sqr, avx2::dot, avx2::l2sqr_sq8, avx2::dot_sq8};
#endif

const KernelTable* compiled(Isa isa) noexcept {
  switch (isa) {
    case Isa::kScalar:
      return &kScalar;
    case Isa::kNeon:
#if QUIVER_HAVE_NEON
      return &kNeon;
#else
      return nullptr;
#endif
    case Isa::kAvx2:
#if QUIVER_HAVE_AVX2
      return &kAvx2;
#else
      return nullptr;
#endif
  }
  return nullptr;
}

bool cpu_supports(Isa isa) noexcept {
  switch (isa) {
    case Isa::kScalar:
      return true;
    case Isa::kNeon:
#if QUIVER_HAVE_NEON
      return true;  // mandatory on AArch64
#else
      return false;
#endif
    case Isa::kAvx2:
#if QUIVER_HAVE_AVX2
      __builtin_cpu_init();
      return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
      return false;
#endif
  }
  return false;
}

// ISA requested through QUIVER_SIMD, if any and if compiled in.
const KernelTable* forced() noexcept {
  static const KernelTable* table = []() -> const KernelTable* {
    // Read once, inside a thread-safe static initializer.
    const char* env = std::getenv("QUIVER_SIMD");  // NOLINT(concurrency-mt-unsafe)
    if (env == nullptr || *env == '\0') return nullptr;
    for (Isa isa : {Isa::kScalar, Isa::kNeon, Isa::kAvx2}) {
      if (to_string(isa) == env) {
        if (const KernelTable* t = compiled(isa)) return t;
      }
    }
    std::fprintf(stderr, "quiver: ignoring QUIVER_SIMD=%s (unknown or not compiled in)\n", env);
    return nullptr;
  }();
  return table;
}

}  // namespace

std::string_view to_string(Isa isa) noexcept {
  switch (isa) {
    case Isa::kScalar:
      return "scalar";
    case Isa::kNeon:
      return "neon";
    case Isa::kAvx2:
      return "avx2";
  }
  return "unknown";
}

const KernelTable* get(Isa isa) noexcept {
  const KernelTable* table = compiled(isa);
  if (table == nullptr) return nullptr;
  if (cpu_supports(isa) || forced() == table) return table;
  return nullptr;
}

std::vector<Isa> available() {
  std::vector<Isa> out;
  for (Isa isa : {Isa::kScalar, Isa::kNeon, Isa::kAvx2}) {
    if (get(isa) != nullptr) out.push_back(isa);
  }
  return out;
}

const KernelTable& active() noexcept {
  static const KernelTable& table = []() -> const KernelTable& {
    if (const KernelTable* f = forced()) return *f;
    for (Isa isa : {Isa::kAvx2, Isa::kNeon}) {
      if (const KernelTable* t = get(isa)) return *t;
    }
    return kScalar;
  }();
  return table;
}

void normalize(float* v, std::size_t dim) noexcept {
  float norm_sq = active().dot(v, v, dim);
  if (!std::isfinite(norm_sq)) {
    // Squares overflowed float (|x| > ~1.8e19); redo the reduction in double.
    double acc = 0.0;
    for (std::size_t i = 0; i < dim; ++i)
      acc += static_cast<double>(v[i]) * static_cast<double>(v[i]);
    const double inv = acc > 0.0 && std::isfinite(acc) ? 1.0 / std::sqrt(acc) : 0.0;
    if (inv == 0.0) return;
    for (std::size_t i = 0; i < dim; ++i)
      v[i] = static_cast<float>(static_cast<double>(v[i]) * inv);
    return;
  }
  if (norm_sq <= 0.0f) return;
  const float inv = 1.0f / std::sqrt(norm_sq);
  for (std::size_t i = 0; i < dim; ++i) v[i] *= inv;
}

}  // namespace quiver::simd

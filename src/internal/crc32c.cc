#include "internal/crc32c.h"

#include <array>
#include <cstring>

#if defined(__aarch64__)
#include <arm_acle.h>
#if defined(__APPLE__)
#define QUIVER_CRC_HW_ALWAYS 1  // every Apple Silicon core implements the CRC32 extension
#else
#include <sys/auxv.h>
#endif
#elif defined(__x86_64__)
#include <immintrin.h>
#endif

namespace quiver::internal {
namespace {

constexpr std::uint32_t kPoly = 0x82f63b78u;  // reflected Castagnoli polynomial

constexpr std::array<std::uint32_t, 256> make_table() {
  std::array<std::uint32_t, 256> t{};
  for (std::uint32_t i = 0; i < 256; ++i) {
    std::uint32_t c = i;
    for (int k = 0; k < 8; ++k) c = (c & 1) ? (c >> 1) ^ kPoly : c >> 1;
    t[i] = c;
  }
  return t;
}
constexpr auto kTable = make_table();

std::uint32_t crc_sw(const unsigned char* p, std::size_t n, std::uint32_t c) noexcept {
  for (std::size_t i = 0; i < n; ++i) c = kTable[(c ^ p[i]) & 0xff] ^ (c >> 8);
  return c;
}

#if defined(__aarch64__)
#if defined(__clang__)
#define QUIVER_TARGET_CRC __attribute__((target("crc")))
#else
#define QUIVER_TARGET_CRC __attribute__((target("+crc")))  // GCC's spelling
#endif
QUIVER_TARGET_CRC std::uint32_t crc_hw(const unsigned char* p, std::size_t n,
                                       std::uint32_t c) noexcept {
  for (; n >= 8; n -= 8, p += 8) {
    std::uint64_t v = 0;
    std::memcpy(&v, p, 8);
    c = __crc32cd(c, v);
  }
  for (; n > 0; --n, ++p) c = __crc32cb(c, *p);
  return c;
}
bool have_hw() noexcept {
#if defined(QUIVER_CRC_HW_ALWAYS)
  return true;
#elif defined(HWCAP_CRC32)
  static const bool ok = (getauxval(AT_HWCAP) & HWCAP_CRC32) != 0;
  return ok;
#else
  return false;
#endif
}
#elif defined(__x86_64__)
__attribute__((target("sse4.2"))) std::uint32_t crc_hw(const unsigned char* p, std::size_t n,
                                                       std::uint32_t c) noexcept {
  std::uint64_t c64 = c;
  for (; n >= 8; n -= 8, p += 8) {
    std::uint64_t v = 0;
    std::memcpy(&v, p, 8);
    c64 = _mm_crc32_u64(c64, v);
  }
  c = static_cast<std::uint32_t>(c64);
  for (; n > 0; --n, ++p) c = _mm_crc32_u8(c, *p);
  return c;
}
bool have_hw() noexcept {
  static const bool ok = [] {
    __builtin_cpu_init();
    return __builtin_cpu_supports("sse4.2") != 0;
  }();
  return ok;
}
#endif

}  // namespace

std::uint32_t crc32c(const void* data, std::size_t size, std::uint32_t crc) noexcept {
  const auto* p = static_cast<const unsigned char*>(data);
  std::uint32_t c = ~crc;
#if defined(__aarch64__) || defined(__x86_64__)
  c = have_hw() ? crc_hw(p, size, c) : crc_sw(p, size, c);
#else
  c = crc_sw(p, size, c);
#endif
  return ~c;
}

}  // namespace quiver::internal

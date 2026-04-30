#pragma once

#include <cstddef>
#include <cstdint>

namespace quiver::internal {

/// CRC-32C (Castagnoli). Uses the ARMv8 CRC32 or SSE4.2 instructions when available.
/// Pass a previous result as `crc` to checksum data incrementally.
std::uint32_t crc32c(const void* data, std::size_t size, std::uint32_t crc = 0) noexcept;

}  // namespace quiver::internal

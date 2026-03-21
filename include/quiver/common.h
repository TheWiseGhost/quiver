// Core vocabulary types shared by every Quiver index.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace quiver {

/// Caller-visible identifier attached to every stored vector.
using Label = std::uint64_t;

/// Dense internal node identifier (index into storage arrays).
using NodeId = std::uint32_t;

/// Largest supported vector dimensionality (all index types).
inline constexpr std::size_t kMaxDimension = 65536;

/// Returned in result slots that could not be filled (e.g. k > size()).
inline constexpr Label kInvalidLabel = std::numeric_limits<Label>::max();

/// Distance conventions (smaller is always closer):
///   kL2            squared Euclidean distance
///   kInnerProduct  1 - <q, x>
///   kCosine        1 - cos(q, x); vectors are normalized on insert and query
enum class Metric : std::uint32_t { kL2 = 0, kInnerProduct = 1, kCosine = 2 };

std::string_view to_string(Metric metric) noexcept;
/// Accepts "l2", "ip"/"inner_product", "cosine"/"cos". Throws Error otherwise.
Metric parse_metric(std::string_view name);

struct Neighbor {
  float distance;
  Label label;

  friend constexpr bool operator<(const Neighbor& a, const Neighbor& b) noexcept {
    return a.distance < b.distance || (a.distance == b.distance && a.label < b.label);
  }
  friend constexpr bool operator==(const Neighbor&, const Neighbor&) noexcept = default;
};

/// How an index file is brought into memory.
struct LoadOptions {
  /// HnswIndex only (other index types always read): map the file instead of reading it. The
  /// vector and graph arrays are used in place and pages fault in on first touch; the mapping is
  /// private and copy-on-write, so the index stays fully mutable and the file is never modified.
  /// Keep the file in place (do not truncate/overwrite it in place) while the index is alive;
  /// replacing it via save() to the same path is safe.
  bool mmap = false;
  /// Verify every section's CRC-32C. Costs one pass over the file (defeats lazy mmap paging).
  /// Header and section-table checksums and all structural invariants are always verified.
  bool verify_checksums = true;
};

/// All recoverable failures (bad arguments, corrupt files, I/O errors) throw this.
class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

namespace detail {
[[noreturn]] void throw_error(const std::string& message);
}  // namespace detail

}  // namespace quiver

/// Throws quiver::Error when `cond` is false. `message` is only evaluated on failure.
#define QUIVER_CHECK(cond, message)             \
  do {                                          \
    if (!(cond)) [[unlikely]] {                 \
      ::quiver::detail::throw_error((message)); \
    }                                           \
  } while (0)

#include "internal/chunked_storage.h"

#include <sys/mman.h>

#include <cstring>
#include <string>

namespace quiver::internal {

namespace {
unsigned pick_shift(std::size_t stride) {
  unsigned shift = 10;
  while (shift < 16 && (std::size_t{2} << shift) * stride <= ChunkedStorage::kTargetChunkBytes)
    ++shift;
  return shift;
}
}  // namespace

ChunkedStorage::ChunkedStorage(std::size_t stride)
    : stride_(stride),
      shift_(pick_shift(stride)),
      mask_(static_cast<NodeId>((std::size_t{1} << shift_) - 1)),
      chunks_(new std::atomic<std::byte*>[kMaxChunks]()) {
  QUIVER_CHECK(stride > 0, "ChunkedStorage stride must be positive");
  owned_.reserve(kMaxChunks);  // so recording a new chunk can never throw (and leak it)
}

ChunkedStorage::~ChunkedStorage() {
  const std::size_t n = num_chunks_.load(std::memory_order_relaxed);
  for (std::size_t c = 0; c < n; ++c) {
    if (owned_[c]) ::munmap(chunks_[c].load(std::memory_order_relaxed), chunk_bytes());
  }
}

std::byte* ChunkedStorage::allocate_chunk() const {
  void* p =
      ::mmap(nullptr, chunk_bytes(), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  QUIVER_CHECK(p != MAP_FAILED,
               "out of memory allocating " + std::to_string(chunk_bytes()) + "-byte storage chunk");
#if defined(MADV_HUGEPAGE)
  // Graph traversal touches vectors at random; huge pages cut TLB misses (Linux THP).
  ::madvise(p, chunk_bytes(), MADV_HUGEPAGE);
#endif
  return static_cast<std::byte*>(p);
}

void ChunkedStorage::reserve(std::size_t n) {
  QUIVER_CHECK(n <= max_nodes(), "capacity " + std::to_string(n) + " exceeds the maximum of " +
                                     std::to_string(max_nodes()) + " vectors");
  const std::size_t needed = (n + chunk_nodes() - 1) >> shift_;
  if (num_chunks_.load(std::memory_order_acquire) >= needed) return;
  std::lock_guard lock(grow_mu_);
  for (std::size_t c = num_chunks_.load(std::memory_order_relaxed); c < needed; ++c) {
    chunks_[c].store(allocate_chunk(), std::memory_order_release);
    owned_.push_back(true);
    num_chunks_.store(c + 1, std::memory_order_release);
  }
}

void ChunkedStorage::adopt(std::byte* base, std::size_t n) {
  std::lock_guard lock(grow_mu_);
  QUIVER_CHECK(num_chunks_.load(std::memory_order_relaxed) == 0, "adopt() requires empty storage");
  QUIVER_CHECK(n <= max_nodes(), "too many vectors to adopt");
  const std::size_t full = n >> shift_;
  for (std::size_t c = 0; c < full; ++c) {
    chunks_[c].store(base + c * chunk_bytes(), std::memory_order_relaxed);
    owned_.push_back(false);
  }
  const std::size_t tail = n & mask_;
  std::size_t total = full;
  if (tail != 0) {
    std::byte* chunk = allocate_chunk();
    std::memcpy(chunk, base + full * chunk_bytes(), tail * stride_);
    chunks_[full].store(chunk, std::memory_order_relaxed);
    owned_.push_back(true);
    ++total;
  }
  num_chunks_.store(total, std::memory_order_release);
}

std::size_t ChunkedStorage::owned_bytes() const noexcept {
  std::lock_guard lock(grow_mu_);
  std::size_t chunks = 0;
  for (bool owned : owned_) chunks += owned ? 1 : 0;
  return chunks * chunk_bytes();
}

}  // namespace quiver::internal

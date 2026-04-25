// Fixed-stride per-node storage that grows without ever relocating existing slots.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "quiver/common.h"

namespace quiver::internal {

/// Storage addressed by NodeId through a two-level table: id -> (chunk, slot). Chunks are allocated
/// on demand and stay put until destruction, so a reader holding a NodeId it obtained through a
/// properly synchronized path (e.g. an acquire load of a graph edge) can dereference it with no
/// further synchronization while other threads keep growing the storage. This is what lets HNSW
/// searches run lock-free concurrently with inserts.
///
/// Chunks are anonymous memory mappings: untouched pages cost nothing and arrive zeroed.
class ChunkedStorage {
 public:
  static constexpr std::size_t kMaxChunks = std::size_t{1} << 15;
  /// Chunks aim for this size; the node count per chunk is a power of two in [2^10, 2^16].
  static constexpr std::size_t kTargetChunkBytes = std::size_t{64} << 20;

  explicit ChunkedStorage(std::size_t stride);
  ~ChunkedStorage();

  ChunkedStorage(const ChunkedStorage&) = delete;
  ChunkedStorage& operator=(const ChunkedStorage&) = delete;

  std::byte* at(NodeId id) const noexcept {
    std::byte* chunk = chunks_[id >> shift_].load(std::memory_order_relaxed);
    return chunk + static_cast<std::size_t>(id & mask_) * stride_;
  }

  template <class T>
  T* as(NodeId id) const noexcept {
    return reinterpret_cast<T*>(at(id));
  }

  std::size_t stride() const noexcept { return stride_; }
  std::size_t chunk_nodes() const noexcept { return std::size_t{1} << shift_; }
  /// Largest number of slots this storage can ever hold.
  std::size_t max_nodes() const noexcept { return kMaxChunks << shift_; }

  /// Number of slots backed by chunks.
  std::size_t capacity() const noexcept {
    return num_chunks_.load(std::memory_order_acquire) << shift_;
  }

  /// Makes slots [0, n) addressable. Thread-safe; a cheap check when already satisfied.
  void reserve(std::size_t n);

  /// Serves the first `n` slots from external memory laid out contiguously at `base` (e.g. a
  /// private, copy-on-write file mapping that the caller keeps alive). Full chunks alias `base`;
  /// a trailing partial chunk is copied into owned memory so that later appends cannot run past
  /// the end of the mapping. Must be called on empty storage.
  void adopt(std::byte* base, std::size_t n);

  /// Bytes of address space this object reserved itself (adopted memory is not counted). Pages are
  /// committed lazily by the OS, so resident memory can be much lower.
  std::size_t owned_bytes() const noexcept;

 private:
  std::byte* allocate_chunk() const;
  std::size_t chunk_bytes() const noexcept { return chunk_nodes() * stride_; }

  std::size_t stride_;
  unsigned shift_;
  NodeId mask_;
  std::unique_ptr<std::atomic<std::byte*>[]> chunks_;
  std::vector<bool> owned_;  // guarded by grow_mu_
  std::atomic<std::size_t> num_chunks_{0};
  mutable std::mutex grow_mu_;
};

}  // namespace quiver::internal

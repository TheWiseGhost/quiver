// Hot-path helpers shared by the graph and flat indexes: candidate heaps, visited sets, spinlocks.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "quiver/common.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace quiver::internal {

struct Candidate {
  float dist;
  NodeId id;
};

/// Heap comparators for std::push_heap / std::pop_heap.
struct FarthestOnTop {  // max-heap: the bounded result set, worst element at front()
  bool operator()(const Candidate& a, const Candidate& b) const noexcept { return a.dist < b.dist; }
};
struct NearestOnTop {  // min-heap: the expansion frontier, best element at front()
  bool operator()(const Candidate& a, const Candidate& b) const noexcept { return a.dist > b.dist; }
};

template <class Cmp>
inline void heap_push(std::vector<Candidate>& heap, Candidate c) {
  heap.push_back(c);
  std::push_heap(heap.begin(), heap.end(), Cmp{});
}

template <class Cmp>
inline Candidate heap_pop(std::vector<Candidate>& heap) {
  std::pop_heap(heap.begin(), heap.end(), Cmp{});
  const Candidate top = heap.back();
  heap.pop_back();
  return top;
}

/// Epoch-tagged visited set: clearing is O(1) (bump the epoch) except once every 65,535 uses.
/// Grows on demand, so ids inserted concurrently with a search are handled safely.
class VisitedSet {
 public:
  void begin(std::size_t size_hint) {
    if (marks_.size() < size_hint) marks_.resize(size_hint, 0);
    if (++epoch_ == 0) {
      std::fill(marks_.begin(), marks_.end(), std::uint16_t{0});
      epoch_ = 1;
    }
  }

  /// Marks `id`; returns true if it had not been visited in this epoch.
  bool insert(NodeId id) {
    if (id >= marks_.size()) [[unlikely]] {
      marks_.resize(std::max<std::size_t>(std::size_t{id} + 1, marks_.size() * 2), 0);
    }
    if (marks_[id] == epoch_) return false;
    marks_[id] = epoch_;
    return true;
  }

  bool contains(NodeId id) const noexcept { return id < marks_.size() && marks_[id] == epoch_; }

  void prefetch(NodeId id) const noexcept {
    if (id < marks_.size()) __builtin_prefetch(&marks_[id], 1, 3);
  }

 private:
  std::vector<std::uint16_t> marks_;
  std::uint16_t epoch_ = 0;
};

inline void cpu_relax() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
  _mm_pause();
#elif defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#endif
}

/// One-byte test-and-test-and-set lock over a plain byte (so it can live inside POD node records).
class SpinGuard {
 public:
  explicit SpinGuard(std::uint8_t& byte) noexcept : flag_(byte) {
    for (int spins = 0; flag_.exchange(1, std::memory_order_acquire) != 0;) {
      while (flag_.load(std::memory_order_relaxed) != 0) {
        if (++spins < 64) {
          cpu_relax();
        } else {
          std::this_thread::yield();
        }
      }
    }
  }
  ~SpinGuard() { flag_.store(0, std::memory_order_release); }

  SpinGuard(const SpinGuard&) = delete;
  SpinGuard& operator=(const SpinGuard&) = delete;

 private:
  std::atomic_ref<std::uint8_t> flag_;
};

/// Leases per-thread scratch buffers from a thread-local stack. Each nested lease on the same
/// thread gets its own object, so a re-entrant call (e.g. a Filter whose contains() runs a search
/// on another index) cannot corrupt the buffers of the call it interrupted. The common depth-1
/// case allocates nothing after warm-up.
template <class T>
class ScratchLease {
 public:
  ScratchLease() {
    Stack& st = stack();
    if (st.depth == st.items.size()) st.items.push_back(std::make_unique<T>());
    item_ = st.items[st.depth++].get();
  }
  ~ScratchLease() { --stack().depth; }
  ScratchLease(const ScratchLease&) = delete;
  ScratchLease& operator=(const ScratchLease&) = delete;

  T& operator*() const noexcept { return *item_; }
  T* operator->() const noexcept { return item_; }

 private:
  struct Stack {
    std::vector<std::unique_ptr<T>> items;
    std::size_t depth = 0;
  };
  static Stack& stack() {
    thread_local Stack st;
    return st;
  }
  T* item_;
};

/// SplitMix64: a fast, well-mixed hash used to derive per-node random levels deterministically.
constexpr std::uint64_t splitmix64(std::uint64_t x) noexcept {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

}  // namespace quiver::internal

// A small fork-join pool for data-parallel loops (batch insert, batch search, k-means).
#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace quiver {

class ThreadPool {
 public:
  /// Range callback: process items [begin, end).
  using RangeFn = std::function<void(std::size_t begin, std::size_t end)>;

  /// `num_threads` counts the calling thread, so ThreadPool(1) runs everything inline.
  /// 0 means std::thread::hardware_concurrency().
  explicit ThreadPool(std::size_t num_threads = 0);
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  std::size_t num_threads() const noexcept { return workers_.size() + 1; }

  /// Splits [0, n) into chunks of at most `grain` items, runs `fn` on them across the pool
  /// (the caller participates), and blocks until all chunks finish. If any chunk throws, the
  /// remaining chunks are skipped and the first exception is rethrown here.
  ///
  /// Safe to call from several threads (calls are serialized) and from inside a running chunk of
  /// any pool (the nested loop then runs inline on that thread instead of deadlocking).
  void parallel_for(std::size_t n, std::size_t grain, const RangeFn& fn);

  /// Process-wide pool sized to the machine; created on first use.
  static ThreadPool& global();

 private:
  struct Job;

  void worker_loop();
  void shutdown() noexcept;
  static void run_chunks(Job& job) noexcept;

  std::vector<std::thread> workers_;
  std::mutex submit_mu_;  // serializes parallel_for calls

  std::mutex mu_;
  std::condition_variable work_cv_;
  std::condition_variable done_cv_;
  Job* job_ = nullptr;
  std::uint64_t generation_ = 0;
  std::size_t pending_workers_ = 0;
  bool stop_ = false;
};

}  // namespace quiver

#include "quiver/thread_pool.h"

#include <algorithm>
#include <atomic>
#include <exception>

namespace quiver {
namespace {
// The pool whose chunk the current thread is executing, used to detect nested parallel_for.
thread_local const ThreadPool* t_running_pool = nullptr;
}  // namespace

struct ThreadPool::Job {
  const RangeFn* fn = nullptr;
  std::size_t n = 0;
  std::size_t grain = 1;
  std::atomic<std::size_t> next{0};
  std::atomic<bool> failed{false};
  std::mutex error_mu;
  std::exception_ptr error;
  const ThreadPool* pool = nullptr;
};

ThreadPool::ThreadPool(std::size_t num_threads) {
  if (num_threads == 0) {
    num_threads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
  }
  workers_.reserve(num_threads - 1);
  try {
    for (std::size_t i = 0; i + 1 < num_threads; ++i) {
      workers_.emplace_back([this] { worker_loop(); });
    }
  } catch (...) {
    shutdown();  // join the threads that did start, then report the failure
    throw;
  }
}

ThreadPool::~ThreadPool() { shutdown(); }

void ThreadPool::shutdown() noexcept {
  {
    std::lock_guard lock(mu_);
    stop_ = true;
  }
  work_cv_.notify_all();
  for (auto& t : workers_) {
    if (t.joinable()) t.join();
  }
  workers_.clear();
}

ThreadPool& ThreadPool::global() {
  static ThreadPool pool;
  return pool;
}

void ThreadPool::run_chunks(Job& job) noexcept {
  const ThreadPool* previous = t_running_pool;
  t_running_pool = job.pool;
  while (!job.failed.load(std::memory_order_relaxed)) {
    const std::size_t begin = job.next.fetch_add(job.grain, std::memory_order_relaxed);
    if (begin >= job.n) break;
    const std::size_t end = std::min(job.n, begin + job.grain);
    try {
      (*job.fn)(begin, end);
    } catch (...) {
      std::lock_guard lock(job.error_mu);
      if (!job.error) job.error = std::current_exception();
      job.failed.store(true, std::memory_order_relaxed);
    }
  }
  t_running_pool = previous;
}

void ThreadPool::worker_loop() {
  std::uint64_t seen = 0;
  for (;;) {
    Job* job = nullptr;
    {
      std::unique_lock lock(mu_);
      work_cv_.wait(lock, [&] { return stop_ || generation_ != seen; });
      if (stop_) return;
      seen = generation_;
      job = job_;
    }
    run_chunks(*job);
    {
      std::lock_guard lock(mu_);
      if (--pending_workers_ == 0) done_cv_.notify_one();
    }
  }
}

void ThreadPool::parallel_for(std::size_t n, std::size_t grain, const RangeFn& fn) {
  if (n == 0) return;
  grain = std::max<std::size_t>(1, grain);
  // Inline when there is nothing to parallelize, or when called from inside any pool's chunk.
  // Nesting into the *same* pool would deadlock on submit_mu_; nesting across pools (A -> B -> A,
  // or two threads nesting A and B in opposite orders) could deadlock too, and the outer loop
  // already occupies the cores, so running nested loops inline is both safe and fast.
  if (workers_.empty() || n <= grain || t_running_pool != nullptr) {
    fn(0, n);
    return;
  }

  std::lock_guard submit(submit_mu_);
  Job job;
  job.fn = &fn;
  job.n = n;
  job.grain = grain;
  job.pool = this;
  {
    std::lock_guard lock(mu_);
    job_ = &job;
    pending_workers_ = workers_.size();
    ++generation_;
  }
  work_cv_.notify_all();
  run_chunks(job);
  {
    // Every worker must finish with `job` before it goes out of scope.
    std::unique_lock lock(mu_);
    done_cv_.wait(lock, [&] { return pending_workers_ == 0; });
    job_ = nullptr;
  }
  if (job.error) std::rethrow_exception(job.error);
}

}  // namespace quiver

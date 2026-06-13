#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

#include "quiver/thread_pool.h"

namespace quiver {
namespace {

TEST(ThreadPool, CoversEveryIndexExactlyOnce) {
  ThreadPool pool(4);
  for (std::size_t n : {0u, 1u, 7u, 1000u, 100003u}) {
    for (std::size_t grain : {1u, 3u, 64u, 1000000u}) {
      std::vector<std::atomic<int>> hits(n);
      pool.parallel_for(n, grain, [&](std::size_t b, std::size_t e) {
        ASSERT_LE(e - b, grain);
        for (std::size_t i = b; i < e; ++i) hits[i].fetch_add(1);
      });
      for (std::size_t i = 0; i < n; ++i) ASSERT_EQ(hits[i].load(), 1) << n << " " << grain;
    }
  }
}

TEST(ThreadPool, ActuallyRunsInParallel) {
  ThreadPool pool(4);
  std::mutex mu;
  std::set<std::thread::id> ids;
  pool.parallel_for(64, 1, [&](std::size_t, std::size_t) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    std::lock_guard lock(mu);
    ids.insert(std::this_thread::get_id());
  });
  EXPECT_GT(ids.size(), 1u);
}

TEST(ThreadPool, PropagatesFirstExceptionAndStaysUsable) {
  ThreadPool pool(4);
  EXPECT_THROW(pool.parallel_for(1000, 1,
                                 [](std::size_t b, std::size_t) {
                                   if (b == 500) throw std::runtime_error("boom");
                                 }),
               std::runtime_error);
  std::atomic<std::size_t> sum{0};
  pool.parallel_for(100, 7, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) sum += i;
  });
  EXPECT_EQ(sum.load(), 4950u);
}

TEST(ThreadPool, NestedParallelForRunsInline) {
  ThreadPool pool(4);
  std::atomic<int> total{0};
  pool.parallel_for(8, 1, [&](std::size_t, std::size_t) {
    pool.parallel_for(10, 1,
                      [&](std::size_t b, std::size_t e) { total += static_cast<int>(e - b); });
  });
  EXPECT_EQ(total.load(), 80);
}

TEST(ThreadPool, ConcurrentCallersAreSerialized) {
  ThreadPool pool(3);
  std::atomic<long> total{0};
  std::vector<std::thread> callers;
  for (int t = 0; t < 4; ++t) {
    callers.emplace_back([&] {
      for (int r = 0; r < 50; ++r) {
        pool.parallel_for(100, 5,
                          [&](std::size_t b, std::size_t e) { total += static_cast<long>(e - b); });
      }
    });
  }
  for (auto& c : callers) c.join();
  EXPECT_EQ(total.load(), 4 * 50 * 100);
}

TEST(ThreadPool, SingleThreadPoolRunsInline) {
  ThreadPool pool(1);
  EXPECT_EQ(pool.num_threads(), 1u);
  const auto caller = std::this_thread::get_id();
  pool.parallel_for(
      100, 1, [&](std::size_t, std::size_t) { EXPECT_EQ(std::this_thread::get_id(), caller); });
}

}  // namespace
}  // namespace quiver

namespace quiver {
namespace {

TEST(ThreadPool, CrossPoolNestingDoesNotDeadlock) {
  // Two threads nest A->B and B->A concurrently: the classic lock-order inversion.
  ThreadPool a(3), b(3);
  std::atomic<int> total{0};
  auto nest = [&](ThreadPool& outer, ThreadPool& inner) {
    for (int r = 0; r < 20; ++r) {
      outer.parallel_for(8, 1, [&](std::size_t, std::size_t) {
        inner.parallel_for(
            8, 1, [&](std::size_t bb, std::size_t e) { total += static_cast<int>(e - bb); });
      });
    }
  };
  std::thread t1([&] { nest(a, b); });
  std::thread t2([&] { nest(b, a); });
  t1.join();
  t2.join();
  EXPECT_EQ(total.load(), 2 * 20 * 8 * 8);
}

}  // namespace
}  // namespace quiver

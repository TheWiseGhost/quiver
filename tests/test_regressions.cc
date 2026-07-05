// Regression tests for defects found in code review. Each test names the failure it guards.
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <thread>

#include "internal/crc32c.h"
#include "internal/file_format.h"
#include "quiver/flat_index.h"
#include "quiver/hnsw_index.h"
#include "quiver/ivfpq_index.h"
#include "test_util.h"

namespace quiver {
namespace {

// Slots used to become visible through the label map before their storage chunk existed, so
// contains/remove/stats/duplicate-add racing the very first add() dereferenced a null chunk.
TEST(Regression, LabelLookupsRacingFirstAddAreSafe) {
  const std::vector<float> v(8, 1.0f);
  for (int mode = 0; mode < 4; ++mode) {
    for (int iter = 0; iter < 300; ++iter) {
      HnswParams p;
      p.dim = 8;
      HnswIndex index(p);
      std::atomic<bool> go{false}, done{false};
      std::thread reader([&] {
        while (!go.load()) {
        }
        const Label l = 7;
        while (!done.load()) {
          if (mode == 0) (void)index.contains(7);
          if (mode == 1) (void)index.stats();
          if (mode == 2) (void)index.remove(7);
          if (mode == 3) {
            try {
              index.add(v.data(), 1, &l);
            } catch (const Error&) {
            }
          }
        }
      });
      const Label l = 7;
      go = true;
      try {
        index.add(v.data(), 1, &l);
      } catch (const Error&) {  // mode 3: the reader may win the label
      }
      done = true;
      reader.join();
      EXPECT_LE(index.size(), 1u);
    }
  }
}

// A filter whose contains() runs a search on another index used to reuse (and corrupt) the
// outer search's thread-local scratch buffers.
TEST(Regression, ReentrantFilterGetsItsOwnScratch) {
  const std::size_t dim = 8, n = 2000;
  const auto base = testing::random_vectors(n, dim, 1);
  HnswParams p;
  p.dim = dim;
  HnswIndex outer(p), inner(p);
  outer.add(base.data(), n);
  inner.add(base.data(), n);
  PredicateFilter nested([&](Label l) {
    const auto r = inner.search(base.data() + (l % n) * dim, 3);  // re-enters Quiver
    return !r.empty() && l % 2 == 0;
  });
  HnswSearchParams sp;
  sp.filter = &nested;
  for (std::size_t q = 0; q < 20; ++q) {
    const auto got = outer.search(base.data() + q * dim, 5, sp);
    ASSERT_EQ(got.size(), 5u);
    for (const auto& nb : got) EXPECT_EQ(nb.label % 2, 0u);
    for (std::size_t j = 1; j < got.size(); ++j) EXPECT_LE(got[j - 1].distance, got[j].distance);
  }
}

TEST(Regression, NonFiniteInputsAreRejected) {
  HnswParams p;
  p.dim = 4;
  HnswIndex hnsw(p);
  std::vector<float> bad = {1, std::nanf(""), 2, 3};
  EXPECT_THROW(hnsw.add(bad.data(), 1), Error);
  EXPECT_THROW((void)hnsw.search(bad.data(), 1), Error);
  bad[1] = std::numeric_limits<float>::infinity();
  FlatIndex flat(4);
  EXPECT_THROW(flat.add(bad.data(), 1), Error);
  EXPECT_EQ(hnsw.size(), 0u);
}

// k * refine_factor wrapped to 0 and the search read heap.front() of an empty vector.
TEST(Regression, IvfPqRefineFactorOverflowIsSaturated) {
  const std::size_t dim = 8, n = 2000;
  const auto base = testing::random_vectors(n, dim, 2);
  IvfPqParams p;
  p.dim = dim;
  p.nlist = 16;
  p.m = 4;
  p.store_vectors = true;
  IvfPqIndex index(p);
  index.train(base.data(), n);
  index.add(base.data(), n);
  IvfPqSearchParams sp;
  sp.nprobe = 16;
  sp.refine_factor = std::size_t{1} << 63;  // k * rf == 0 (mod 2^64) for even k
  const auto r = index.search(base.data(), 2, sp);
  ASSERT_EQ(r.size(), 2u);
  EXPECT_EQ(r[0].label, 0u);  // refine over everything is exact
}

TEST(Regression, IvfPqTrainsAtMostOnce) {
  const auto base = testing::random_vectors(600, 8, 3);
  IvfPqParams p;
  p.dim = 8;
  p.nlist = 8;
  p.m = 2;
  IvfPqIndex index(p);
  EXPECT_FALSE(index.is_trained());
  index.train(base.data(), 600);
  EXPECT_TRUE(index.is_trained());
  EXPECT_THROW(index.train(base.data(), 600), Error);
}

TEST(Regression, IvfPqInnerProductAssignsByInnerProduct) {
  // With max-inner-product probing, cells must also be assigned by inner product; otherwise
  // nprobe = nlist / 2 misses many true neighbors that live in L2-nearest-but-low-IP cells.
  const std::size_t dim = 16, n = 6000, nq = 100, k = 10;
  const auto base = testing::clustered_vectors(n, dim, 30, 4);
  const auto queries = testing::clustered_vectors(nq, dim, 30, 5);
  const auto truth = testing::ground_truth(base, queries, dim, k, Metric::kInnerProduct);
  IvfPqParams p;
  p.dim = dim;
  p.metric = Metric::kInnerProduct;
  p.nlist = 32;
  p.m = 8;
  p.store_vectors = true;
  IvfPqIndex index(p);
  index.train(base.data(), n);
  index.add(base.data(), n);
  IvfPqSearchParams sp;
  sp.nprobe = 16;
  sp.refine_factor = 8;
  std::vector<Label> got(nq * k);
  std::vector<float> dist(nq * k);
  index.search_batch(queries.data(), nq, k, got.data(), dist.data(), sp);
  EXPECT_GE(testing::recall_at_k(truth, got, nq, k), 0.9);
}

// Overlapping upper-layer blocks passed validation; a later insert writing one node's list then
// silently rewrote another's, breaking the layer invariants that validation had established.
TEST(Regression, AliasedUpperLayerBlocksAreRejected) {
  const std::size_t dim = 4;
  HnswParams p;
  p.dim = dim;
  p.M = 4;  // small M -> many upper-layer nodes
  HnswIndex index(p);
  const auto base = testing::random_vectors(500, dim, 6);
  index.add(base.data(), 500);
  testing::TempPath path("alias.qvr");
  index.save(path.str());

  std::ifstream in(path.str(), std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  internal::FileHeader header;
  std::memcpy(&header, bytes.data(), sizeof(header));
  std::vector<internal::SectionEntry> table(header.num_sections);
  std::memcpy(table.data(), bytes.data() + sizeof(header),
              table.size() * sizeof(internal::SectionEntry));
  const auto it = std::find_if(table.begin(), table.end(),
                               [](const auto& e) { return e.tag == internal::fourcc("UPOF"); });
  ASSERT_NE(it, table.end());
  auto* off = reinterpret_cast<std::uint64_t*>(bytes.data() + it->offset);
  const std::size_t n = it->size / sizeof(std::uint64_t);
  // Point the second upper-layer node at the first one's block.
  std::size_t first = n, second = n;
  for (std::size_t i = 0; i < n; ++i) {
    if (off[i] == std::numeric_limits<std::uint64_t>::max()) continue;
    if (first == n) {
      first = i;
    } else {
      second = i;
      break;
    }
  }
  ASSERT_LT(second, n);
  off[second] = off[first];
  std::ofstream(path.str(), std::ios::binary | std::ios::trunc)
      .write(bytes.data(), static_cast<std::streamsize>(bytes.size()));

  LoadOptions opts;
  opts.verify_checksums = false;  // the attacker controls the bytes, so CRCs prove nothing
  EXPECT_THROW(HnswIndex::load(path.str(), opts), Error);
}

TEST(Regression, OverlappingSectionsAreRejected) {
  FlatIndex flat(4);
  const auto base = testing::random_vectors(10, 4, 7);
  flat.add(base.data(), 10);
  testing::TempPath path("overlap.qvr");
  flat.save(path.str());
  std::ifstream in(path.str(), std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  internal::FileHeader header;
  std::memcpy(&header, bytes.data(), sizeof(header));
  auto* table = reinterpret_cast<internal::SectionEntry*>(bytes.data() + sizeof(header));
  table[1].offset = table[0].offset;  // two sections now share bytes
  // Re-seal the table and header so only the overlap check can catch it.
  header.table_crc = internal::crc32c(table, header.num_sections * sizeof(internal::SectionEntry));
  header.header_crc = internal::crc32c(&header, offsetof(internal::FileHeader, header_crc));
  std::memcpy(bytes.data(), &header, sizeof(header));
  std::ofstream(path.str(), std::ios::binary | std::ios::trunc)
      .write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  LoadOptions opts;
  opts.verify_checksums = false;
  EXPECT_THROW(FlatIndex::load(path.str(), opts), Error);
}

TEST(Regression, FlatIndexConcurrentAddAndSearch) {
  const std::size_t dim = 8;
  const auto base = testing::random_vectors(4000, dim, 8);
  FlatIndex flat(dim);
  flat.add(base.data(), 100);
  std::atomic<bool> done{false};
  std::thread reader([&] {
    while (!done) {
      const auto r = flat.search(base.data(), 3);
      EXPECT_EQ(r.size(), 3u);
    }
  });
  for (std::size_t i = 100; i < 4000; i += 100) flat.add(base.data() + i * dim, 100);
  done = true;
  reader.join();
  EXPECT_EQ(flat.size(), 4000u);
}

// IVF-PQ add() held the index lock across parallel_for; with a concurrent search_batch on the
// same pool and a third writer waiting (writer-preferring shared_mutex), all three hung forever.
TEST(Regression, IvfPqAddSearchAddDoesNotDeadlock) {
  const std::size_t dim = 16, n = 8000, nq = 20000;
  const auto base = testing::random_vectors(n, dim, 9);
  const auto queries = testing::random_vectors(nq, dim, 10);
  IvfPqParams p;
  p.dim = dim;
  p.nlist = 32;
  p.m = 4;
  p.train_iterations = 3;
  IvfPqIndex index(p);
  ThreadPool pool(4), inline_pool(1);
  index.train(base.data(), n, &pool);
  index.add(base.data(), n, nullptr, &pool);
  std::vector<Label> labels(nq * 10);
  std::vector<float> dist(nq * 10);
  IvfPqSearchParams sp;
  sp.nprobe = 32;
  std::atomic<int> done{0};
  std::thread b([&] {
    index.search_batch(queries.data(), nq, 10, labels.data(), dist.data(), sp, &pool);
    ++done;
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  std::thread a([&] {
    index.add(base.data(), 2000, nullptr, &pool);
    ++done;
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  std::thread c([&] {
    index.add(base.data(), 100, nullptr, &inline_pool);
    ++done;
  });
  for (int i = 0; i < 600 && done < 3; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  if (done < 3) {
    ADD_FAILURE() << "deadlock: only " << done << "/3 threads finished";
    std::_Exit(1);  // the threads can never be joined
  }
  a.join();
  b.join();
  c.join();
  EXPECT_EQ(index.size(), n + 2100);
}

// num_live was incremented only after linking, so a remove() racing an insert could decrement
// it first and size() wrapped around to 2^64 - 1.
TEST(Regression, SizeNeverUnderflowsWhenRemoveRacesInsert) {
  const std::vector<float> v(8, 0.5f);
  for (int iter = 0; iter < 300; ++iter) {
    HnswParams p;
    p.dim = 8;
    HnswIndex index(p);
    std::atomic<bool> go{false};
    std::thread remover([&] {
      while (!go.load()) {
      }
      while (!index.remove(3)) {
        if (index.size() > 1) break;  // would mean the counter wrapped
      }
    });
    const Label l = 3;
    go = true;
    index.add(v.data(), 1, &l);
    remover.join();
    EXPECT_LE(index.size(), 1u);
  }
}

TEST(Regression, SentinelLabelIsRejected) {
  const std::vector<float> v(4, 1.0f);
  HnswParams p;
  p.dim = 4;
  HnswIndex hnsw(p);
  FlatIndex flat(4);
  EXPECT_THROW(hnsw.add(v.data(), 1, &kInvalidLabel), Error);
  EXPECT_THROW(flat.add(v.data(), 1, &kInvalidLabel), Error);
  EXPECT_EQ(hnsw.size(), 0u);
}

}  // namespace
}  // namespace quiver

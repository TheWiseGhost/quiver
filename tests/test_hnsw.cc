#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <limits>
#include <thread>

#include "quiver/hnsw_index.h"
#include "test_util.h"

namespace quiver {
namespace {

HnswParams make_params(std::size_t dim, Metric metric = Metric::kL2,
                       Storage storage = Storage::kFloat32) {
  HnswParams p;
  p.dim = dim;
  p.metric = metric;
  p.storage = storage;
  p.M = 16;
  p.ef_construction = 100;
  return p;
}

// ---- recall across metric x storage -----------------------------------------------------------

struct RecallCase {
  Metric metric;
  Storage storage;
  double min_recall;
};

class HnswRecall : public ::testing::TestWithParam<RecallCase> {};

TEST_P(HnswRecall, MeetsRecallTarget) {
  const auto [metric, storage, min_recall] = GetParam();
  const std::size_t dim = 32, n = 5000, nq = 200, k = 10;
  const auto base = testing::clustered_vectors(n, dim, 20, 1);
  const auto queries = testing::clustered_vectors(nq, dim, 20, 2);
  const auto truth = testing::ground_truth(base, queries, dim, k, metric);

  HnswIndex index(make_params(dim, metric, storage));
  index.add(base.data(), n);
  ASSERT_EQ(index.size(), n);

  std::vector<Label> got(nq * k);
  std::vector<float> dist(nq * k);
  HnswSearchParams sp;
  sp.ef = 128;
  index.search_batch(queries.data(), nq, k, got.data(), dist.data(), sp);
  const double recall = testing::recall_at_k(truth, got, nq, k);
  EXPECT_GE(recall, min_recall) << "metric=" << to_string(metric);

  // Distances come back sorted.
  for (std::size_t q = 0; q < nq; ++q) {
    for (std::size_t j = 1; j < k; ++j) EXPECT_LE(dist[q * k + j - 1], dist[q * k + j]);
  }
}

INSTANTIATE_TEST_SUITE_P(MetricsAndStorage, HnswRecall,
                         ::testing::Values(RecallCase{Metric::kL2, Storage::kFloat32, 0.97},
                                           RecallCase{Metric::kInnerProduct, Storage::kFloat32,
                                                      0.95},
                                           RecallCase{Metric::kCosine, Storage::kFloat32, 0.97},
                                           RecallCase{Metric::kL2, Storage::kSQ8, 0.90},
                                           RecallCase{Metric::kCosine, Storage::kSQ8, 0.90}),
                         [](const auto& param_info) {
                           return std::string(to_string(param_info.param.metric)) +
                                  (param_info.param.storage == Storage::kSQ8 ? "_sq8" : "_f32");
                         });

TEST(Hnsw, RecallImprovesWithEf) {
  const std::size_t dim = 24, n = 4000, nq = 100, k = 10;
  const auto base = testing::random_vectors(n, dim, 3);
  const auto queries = testing::random_vectors(nq, dim, 4);
  const auto truth = testing::ground_truth(base, queries, dim, k, Metric::kL2);
  HnswParams p = make_params(dim);
  p.M = 8;
  p.ef_construction = 40;
  HnswIndex index(p);
  index.add(base.data(), n);
  double prev = 0.0;
  for (std::size_t ef : {10u, 40u, 160u}) {
    std::vector<Label> got(nq * k);
    std::vector<float> dist(nq * k);
    HnswSearchParams sp;
    sp.ef = ef;
    index.search_batch(queries.data(), nq, k, got.data(), dist.data(), sp);
    const double r = testing::recall_at_k(truth, got, nq, k);
    EXPECT_GE(r + 0.01, prev) << "ef=" << ef;
    prev = r;
  }
  EXPECT_GT(prev, 0.95);
}

// ---- basic behavior ------------------------------------------------------------------------

TEST(Hnsw, EmptyIndexReturnsNothing) {
  HnswIndex index(make_params(8));
  std::vector<float> q(8, 0.5f);
  EXPECT_TRUE(index.search(q.data(), 5).empty());
  std::vector<Label> labels(5);
  std::vector<float> dist(5);
  index.search_batch(q.data(), 1, 5, labels.data(), dist.data());
  EXPECT_EQ(labels[0], kInvalidLabel);
  EXPECT_TRUE(std::isinf(dist[0]));
}

TEST(Hnsw, SingleVectorAndExactSelfMatch) {
  HnswIndex index(make_params(4));
  const std::vector<float> v = {1, 2, 3, 4};
  const Label label = 42;
  index.add(v.data(), 1, &label);
  const auto r = index.search(v.data(), 3);
  ASSERT_EQ(r.size(), 1u);
  EXPECT_EQ(r[0].label, 42u);
  EXPECT_FLOAT_EQ(r[0].distance, 0.0f);
}

TEST(Hnsw, FindsEveryVectorAsItsOwnNearestNeighbor) {
  const std::size_t dim = 16, n = 2000;
  const auto base = testing::random_vectors(n, dim, 5);
  HnswIndex index(make_params(dim));
  index.add(base.data(), n);
  std::size_t found = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const auto r = index.search(base.data() + i * dim, 1);
    found += (!r.empty() && r[0].label == i) ? 1u : 0u;
  }
  EXPECT_GE(static_cast<double>(found) / n, 0.99);
}

TEST(Hnsw, KLargerThanIndex) {
  const std::size_t dim = 8;
  const auto base = testing::random_vectors(5, dim, 6);
  HnswIndex index(make_params(dim));
  index.add(base.data(), 5);
  EXPECT_EQ(index.search(base.data(), 100).size(), 5u);
}

TEST(Hnsw, CustomLabelsAndDuplicateRejection) {
  const std::size_t dim = 8;
  const auto base = testing::random_vectors(3, dim, 7);
  HnswIndex index(make_params(dim));
  const std::vector<Label> labels = {1000, 2000, 3000};
  index.add(base.data(), 3, labels.data());
  EXPECT_TRUE(index.contains(2000));
  EXPECT_FALSE(index.contains(1));
  EXPECT_THROW(index.add(base.data(), 1, &labels[1]), Error);
  EXPECT_EQ(index.size(), 3u);  // the failed add left no trace

  const std::vector<Label> dup = {7, 7};
  EXPECT_THROW(index.add(base.data(), 2, dup.data()), Error);
  EXPECT_FALSE(index.contains(7));
}

TEST(Hnsw, GetVectorRoundTrips) {
  const std::size_t dim = 12;
  const auto base = testing::random_vectors(50, dim, 8);
  HnswIndex index(make_params(dim));
  index.add(base.data(), 50);
  const auto v = index.get_vector(17);
  for (std::size_t j = 0; j < dim; ++j) EXPECT_FLOAT_EQ(v[j], base[17 * dim + j]);
  EXPECT_THROW(index.get_vector(999), Error);
}

TEST(Hnsw, InvalidParamsThrow) {
  EXPECT_THROW(HnswIndex(make_params(0)), Error);
  HnswParams p = make_params(8);
  p.M = 1;
  EXPECT_THROW(HnswIndex{p}, Error);
  p.M = 100000;
  EXPECT_THROW(HnswIndex{p}, Error);
}

TEST(Hnsw, DeterministicWhenSingleThreaded) {
  const std::size_t dim = 16, n = 1500;
  const auto base = testing::random_vectors(n, dim, 9);
  ThreadPool one(1);
  HnswIndex a(make_params(dim)), b(make_params(dim));
  a.add(base.data(), n, nullptr, &one);
  b.add(base.data(), n, nullptr, &one);
  for (std::size_t i = 0; i < 50; ++i) {
    EXPECT_EQ(a.search(base.data() + i * dim, 10), b.search(base.data() + i * dim, 10));
  }
}

TEST(Hnsw, IncrementalAddsMatchBatchQuality) {
  const std::size_t dim = 16, n = 3000, nq = 100, k = 10;
  const auto base = testing::random_vectors(n, dim, 10);
  const auto queries = testing::random_vectors(nq, dim, 11);
  const auto truth = testing::ground_truth(base, queries, dim, k, Metric::kL2);
  HnswIndex index(make_params(dim));
  for (std::size_t i = 0; i < n; i += 250) index.add(base.data() + i * dim, 250);
  std::vector<Label> got(nq * k);
  std::vector<float> dist(nq * k);
  index.search_batch(queries.data(), nq, k, got.data(), dist.data());
  EXPECT_GE(testing::recall_at_k(truth, got, nq, k), 0.95);
}

TEST(Hnsw, StatsAreConsistent) {
  const std::size_t dim = 8, n = 3000;
  const auto base = testing::random_vectors(n, dim, 12);
  HnswIndex index(make_params(dim));
  index.add(base.data(), n);
  const auto st = index.stats();
  EXPECT_EQ(st.num_nodes, n);
  EXPECT_EQ(st.nodes_per_level[0], n);
  EXPECT_GE(st.max_level, 1);
  for (std::size_t l = 1; l < st.nodes_per_level.size(); ++l) {
    EXPECT_LE(st.nodes_per_level[l], st.nodes_per_level[l - 1]);
  }
  EXPECT_GT(st.mean_degree_level0, 8.0);
  EXPECT_LE(st.mean_degree_level0, 32.0);
  EXPECT_EQ(st.vector_bytes, n * dim * sizeof(float));
}

// ---- removal and filtering -----------------------------------------------------------------

TEST(Hnsw, RemovedLabelsNeverReturned) {
  const std::size_t dim = 16, n = 2000;
  const auto base = testing::random_vectors(n, dim, 13);
  HnswIndex index(make_params(dim));
  index.add(base.data(), n);
  for (Label l = 0; l < n; l += 2) ASSERT_TRUE(index.remove(l));
  EXPECT_FALSE(index.remove(0));      // already removed
  EXPECT_FALSE(index.remove(n + 5));  // never existed
  EXPECT_EQ(index.size(), n / 2);
  for (std::size_t i = 0; i < 100; ++i) {
    const auto r = index.search(base.data() + i * dim, 10);
    ASSERT_EQ(r.size(), 10u);
    for (const auto& nb : r) EXPECT_EQ(nb.label % 2, 1u);
  }
  // A removed label can be added again.
  const Label zero = 0;
  index.add(base.data(), 1, &zero);
  EXPECT_TRUE(index.contains(0));
  EXPECT_EQ(index.search(base.data(), 1)[0].label, 0u);
}

class HnswFilter : public ::testing::TestWithParam<double> {};

TEST_P(HnswFilter, ResultsRespectFilterWithHighRecall) {
  const double selectivity = GetParam();
  const std::size_t dim = 16, n = 5000, nq = 50, k = 10;
  const auto base = testing::random_vectors(n, dim, 14);
  const auto queries = testing::random_vectors(nq, dim, 15);
  HnswIndex index(make_params(dim));
  index.add(base.data(), n);

  BitsetFilter filter(n);
  std::mt19937 rng(1);
  std::uniform_real_distribution<double> u(0, 1);
  for (Label l = 0; l < n; ++l) {
    if (u(rng) < selectivity) filter.insert(l);
  }
  FlatIndex flat(dim);
  flat.add(base.data(), n);

  std::size_t hits = 0, total = 0;
  HnswSearchParams sp;
  sp.ef = 128;
  sp.filter = &filter;
  for (std::size_t q = 0; q < nq; ++q) {
    const auto got = index.search(queries.data() + q * dim, k, sp);
    const auto want = flat.search(queries.data() + q * dim, k, &filter);
    for (const auto& nb : got) EXPECT_TRUE(filter.contains(nb.label));
    std::unordered_set<Label> w;
    for (const auto& nb : want) w.insert(nb.label);
    for (const auto& nb : got) hits += w.count(nb.label);
    total += want.size();
  }
  EXPECT_GE(static_cast<double>(hits) / static_cast<double>(total), 0.9)
      << "selectivity=" << selectivity;
}

// 0.5%: exact-scan fallback. 5% and 50%: filtered graph search.
INSTANTIATE_TEST_SUITE_P(Selectivity, HnswFilter, ::testing::Values(0.005, 0.05, 0.5));

TEST(Hnsw, PredicateFilterUsesGraphPath) {
  const std::size_t dim = 8, n = 1000;
  const auto base = testing::random_vectors(n, dim, 16);
  HnswIndex index(make_params(dim));
  index.add(base.data(), n);
  PredicateFilter odd([](Label l) { return l % 2 == 1; });
  HnswSearchParams sp;
  sp.filter = &odd;
  const auto r = index.search(base.data(), 10, sp);
  ASSERT_EQ(r.size(), 10u);
  for (const auto& nb : r) EXPECT_EQ(nb.label % 2, 1u);
}

// ---- concurrency (run under the tsan preset to check for races) -----------------------------

TEST(HnswConcurrency, ParallelInsertKeepsRecall) {
  const std::size_t dim = 24, n = 6000, nq = 100, k = 10;
  const auto base = testing::clustered_vectors(n, dim, 30, 17);
  const auto queries = testing::clustered_vectors(nq, dim, 30, 18);
  const auto truth = testing::ground_truth(base, queries, dim, k, Metric::kL2);
  ThreadPool pool(8);
  HnswIndex index(make_params(dim));
  index.add(base.data(), n, nullptr, &pool);
  EXPECT_EQ(index.size(), n);
  std::vector<Label> got(nq * k);
  std::vector<float> dist(nq * k);
  index.search_batch(queries.data(), nq, k, got.data(), dist.data(), {}, &pool);
  EXPECT_GE(testing::recall_at_k(truth, got, nq, k), 0.95);
}

TEST(HnswConcurrency, SearchWhileInsertingAndRemoving) {
  const std::size_t dim = 16, n = 8000;
  const auto base = testing::random_vectors(n, dim, 19);
  HnswIndex index(make_params(dim));
  index.add(base.data(), 1000);  // seed graph

  std::atomic<bool> done{false};
  std::atomic<std::size_t> searches{0};
  std::vector<std::thread> readers;
  for (int t = 0; t < 3; ++t) {
    readers.emplace_back([&, t] {
      std::size_t i = static_cast<std::size_t>(t);
      while (!done.load()) {
        const auto r = index.search(base.data() + (i % 1000) * dim, 5);
        // Regression: a node must never be reachable before its own edges are written, or a
        // search can descend onto it and come back nearly empty.
        EXPECT_EQ(r.size(), 5u);
        for (const auto& nb : r) EXPECT_LT(nb.label, n);
        ++i;
        ++searches;
      }
    });
  }
  std::thread remover([&] {
    for (Label l = 0; l < 200; ++l) index.remove(l);
  });
  // Writers: individual adds from two threads plus a parallel batch.
  std::thread w1([&] {
    for (std::size_t i = 1000; i < 2500; ++i) {
      const Label l = i;
      index.add(base.data() + i * dim, 1, &l);
    }
  });
  std::thread w2([&] {
    for (std::size_t i = 2500; i < 4000; ++i) {
      const Label l = i;
      index.add(base.data() + i * dim, 1, &l);
    }
  });
  std::vector<Label> labels(n - 4000);
  for (std::size_t i = 0; i < labels.size(); ++i) labels[i] = 4000 + i;
  ThreadPool pool(3);
  index.add(base.data() + 4000 * dim, n - 4000, labels.data(), &pool);
  w1.join();
  w2.join();
  remover.join();
  done = true;
  for (auto& r : readers) r.join();

  EXPECT_EQ(index.size(), n - 200);
  EXPECT_GT(searches.load(), 0u);
  // Concurrent inserts that land near each other cannot see one another, so a parallel build
  // leaves slightly more weakly-reachable nodes than a sequential one (~0.05% here); demand
  // near-perfect self-recall rather than perfection.
  std::size_t found = 0, checked = 0;
  for (std::size_t i = 200; i < n; i += 7) {
    const auto r = index.search(base.data() + i * dim, 1);
    ASSERT_FALSE(r.empty());
    found += r[0].label == i ? 1u : 0u;
    ++checked;
  }
  EXPECT_GE(static_cast<double>(found) / static_cast<double>(checked), 0.99);
}

}  // namespace
}  // namespace quiver

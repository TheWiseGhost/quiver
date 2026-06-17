#include <gtest/gtest.h>

#include <algorithm>
#include <limits>

#include "quiver/flat_index.h"
#include "test_util.h"

namespace quiver {
namespace {

TEST(FlatIndex, MatchesNaiveSearch) {
  const std::size_t dim = 13, n = 300, k = 7;
  auto base = testing::random_vectors(n, dim, 1);
  auto q = testing::random_vectors(1, dim, 2);
  FlatIndex index(dim);
  index.add(base.data(), n);
  const auto got = index.search(q.data(), k);
  ASSERT_EQ(got.size(), k);

  std::vector<Neighbor> naive;
  for (std::size_t i = 0; i < n; ++i) {
    float d = 0;
    for (std::size_t j = 0; j < dim; ++j) {
      const float t = q[j] - base[i * dim + j];
      d += t * t;
    }
    naive.push_back({d, i});
  }
  std::sort(naive.begin(), naive.end());
  for (std::size_t j = 0; j < k; ++j) {
    EXPECT_EQ(got[j].label, naive[j].label);
    EXPECT_NEAR(got[j].distance, naive[j].distance, 1e-3f);
  }
}

TEST(FlatIndex, CosineAndInnerProduct) {
  const std::size_t dim = 4;
  std::vector<float> base = {1, 0, 0, 0, 0, 2, 0, 0, 1, 1, 0, 0};
  std::vector<float> q = {3, 0.1f, 0, 0};
  FlatIndex cosine(dim, Metric::kCosine);
  cosine.add(base.data(), 3);
  auto r = cosine.search(q.data(), 3);
  EXPECT_EQ(r[0].label, 0u);
  EXPECT_NEAR(r[0].distance, 1.0f - 3.0f / std::sqrt(9.01f), 1e-5f);

  FlatIndex ip(dim, Metric::kInnerProduct);
  ip.add(base.data(), 3);
  r = ip.search(q.data(), 3);
  EXPECT_EQ(r[0].label, 2u);  // dot = 3.1 beats 3.0
  EXPECT_NEAR(r[0].distance, 1.0f - 3.1f, 1e-5f);
}

TEST(FlatIndex, FewerThanKAndFilters) {
  const std::size_t dim = 8;
  auto base = testing::random_vectors(10, dim, 3);
  FlatIndex index(dim);
  std::vector<Label> labels = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109};
  index.add(base.data(), 10, labels.data());
  EXPECT_EQ(index.search(base.data(), 50).size(), 10u);

  LabelSetFilter only({103, 107});
  const auto r = index.search(base.data(), 5, &only);
  ASSERT_EQ(r.size(), 2u);
  for (const auto& nb : r) EXPECT_TRUE(nb.label == 103 || nb.label == 107);

  std::vector<Label> out(5);
  std::vector<float> dist(5);
  index.search_batch(base.data(), 1, 5, out.data(), dist.data(), &only);
  EXPECT_EQ(out[2], kInvalidLabel);
  EXPECT_EQ(dist[4], std::numeric_limits<float>::infinity());
}

TEST(FlatIndex, SaveLoadRoundTrip) {
  const std::size_t dim = 16;
  auto base = testing::random_vectors(100, dim, 4);
  FlatIndex index(dim, Metric::kCosine);
  index.add(base.data(), 100);
  testing::TempPath path("flat.qvr");
  index.save(path.str());
  const FlatIndex loaded = FlatIndex::load(path.str());
  EXPECT_EQ(loaded.size(), 100u);
  EXPECT_EQ(loaded.metric(), Metric::kCosine);
  EXPECT_EQ(loaded.search(base.data(), 5), index.search(base.data(), 5));
}

}  // namespace
}  // namespace quiver

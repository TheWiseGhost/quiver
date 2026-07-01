#include <gtest/gtest.h>

#include <cmath>

#include "quiver/distance.h"
#include "quiver/ivfpq_index.h"
#include "quiver/quantizer.h"
#include "test_util.h"

namespace quiver {
namespace {

double mse(const std::vector<float>& a, const std::vector<float>& b) {
  double s = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    s += static_cast<double>((a[i] - b[i]) * (a[i] - b[i]));
  return s / static_cast<double>(a.size());
}

TEST(KMeans, RecoversWellSeparatedClusters) {
  const std::size_t dim = 4, per = 200, k = 3;
  std::vector<float> data;
  const float centers[3][4] = {{0, 0, 0, 0}, {10, 10, 0, 0}, {0, 0, 10, 10}};
  auto noise = testing::random_vectors(per * k, dim, 1);
  for (std::size_t c = 0; c < k; ++c) {
    for (std::size_t i = 0; i < per; ++i) {
      for (std::size_t j = 0; j < dim; ++j)
        data.push_back(centers[c][j] + 0.1f * noise[(c * per + i) * dim + j]);
    }
  }
  KMeansParams kp;
  kp.k = k;
  const auto cent = kmeans(data.data(), per * k, dim, kp);
  for (const auto& c : centers) {
    const std::size_t nearest = nearest_centroid(c, cent.data(), k, dim);
    EXPECT_LT(simd::active().l2sqr(c, cent.data() + nearest * dim, dim), 0.05f);
  }
}

TEST(KMeans, HandlesDuplicatePointsWithoutNaN) {
  // Many identical points force empty clusters, exercising the split path.
  std::vector<float> data(1000 * 2, 1.0f);
  for (std::size_t i = 0; i < 20; ++i) data[i * 2] = static_cast<float>(i);
  KMeansParams kp;
  kp.k = 16;
  const auto cent = kmeans(data.data(), 1000, 2, kp);
  for (float x : cent) EXPECT_TRUE(std::isfinite(x));
}

TEST(ProductQuantizer, ReconstructionBeatsCoarserCodes) {
  const std::size_t dim = 32, n = 4000;
  const auto data = testing::clustered_vectors(n, dim, 10, 3);
  double prev = 1e30;
  for (std::size_t m : {4u, 8u, 16u}) {
    ProductQuantizer pq(dim, m);
    pq.train(data.data(), n);
    std::vector<float> back(n * dim);
    std::vector<std::uint8_t> code(m);
    for (std::size_t i = 0; i < n; ++i) {
      pq.encode(data.data() + i * dim, code.data());
      pq.decode(code.data(), back.data() + i * dim);
    }
    const double err = mse(data, back);
    EXPECT_LT(err, prev) << "m=" << m;
    prev = err;
  }
  EXPECT_LT(prev, 0.3);  // unit-variance noise per dimension; 16 bytes keeps most of it
}

TEST(ProductQuantizer, AdcEqualsDistanceToReconstruction) {
  const std::size_t dim = 16, n = 1000, m = 4;
  const auto data = testing::random_vectors(n, dim, 4);
  ProductQuantizer pq(dim, m);
  pq.train(data.data(), n);
  const auto q = testing::random_vectors(1, dim, 5);
  std::vector<float> l2t(m * 256), ipt(m * 256);
  pq.compute_l2_table(q.data(), l2t.data());
  pq.compute_ip_table(q.data(), ipt.data());
  std::vector<std::uint8_t> codes(7 * m);
  for (std::size_t i = 0; i < 7; ++i) pq.encode(data.data() + i * dim, codes.data() + i * m);
  std::vector<float> batch(7);
  pq.adc_batch(l2t.data(), codes.data(), 7, batch.data());
  for (std::size_t i = 0; i < 7; ++i) {
    std::vector<float> rec(dim);
    pq.decode(codes.data() + i * m, rec.data());
    EXPECT_NEAR(pq.adc(l2t.data(), codes.data() + i * m),
                simd::active().l2sqr(q.data(), rec.data(), dim), 1e-3f);
    EXPECT_NEAR(batch[i], pq.adc(l2t.data(), codes.data() + i * m), 1e-5f);
    EXPECT_NEAR(pq.adc(ipt.data(), codes.data() + i * m),
                simd::active().dot(q.data(), rec.data(), dim), 1e-3f);
  }
}

TEST(ProductQuantizer, RejectsBadShapes) {
  EXPECT_THROW(ProductQuantizer(10, 3), Error);
  ProductQuantizer pq(8, 2);
  const auto few = testing::random_vectors(100, 8, 1);
  EXPECT_THROW(pq.train(few.data(), 100), Error);
}

struct IvfCase {
  Metric metric;
  bool refine;
  double min_recall;
};

class IvfPqRecall : public ::testing::TestWithParam<IvfCase> {};

TEST_P(IvfPqRecall, MeetsTarget) {
  const auto [metric, refine, min_recall] = GetParam();
  const std::size_t dim = 32, n = 10000, nq = 100, k = 10;
  const auto base = testing::clustered_vectors(n, dim, 50, 6);
  const auto queries = testing::clustered_vectors(nq, dim, 50, 7);
  const auto truth = testing::ground_truth(base, queries, dim, k, metric);
  IvfPqParams p;
  p.dim = dim;
  p.metric = metric;
  p.nlist = 64;
  p.m = 16;
  p.store_vectors = refine;
  IvfPqIndex index(p);
  index.train(base.data(), n);
  index.add(base.data(), n);
  EXPECT_EQ(index.size(), n);
  IvfPqSearchParams sp;
  sp.nprobe = 16;
  sp.refine_factor = refine ? 4 : 0;
  std::vector<Label> got(nq * k);
  std::vector<float> dist(nq * k);
  index.search_batch(queries.data(), nq, k, got.data(), dist.data(), sp);
  EXPECT_GE(testing::recall_at_k(truth, got, nq, k), min_recall);
}

INSTANTIATE_TEST_SUITE_P(Configs, IvfPqRecall,
                         ::testing::Values(IvfCase{Metric::kL2, false, 0.55},
                                           IvfCase{Metric::kL2, true, 0.95},
                                           IvfCase{Metric::kCosine, true, 0.95},
                                           IvfCase{Metric::kInnerProduct, true, 0.90}));

TEST(IvfPq, FilterSaveLoadAndErrors) {
  const std::size_t dim = 16, n = 3000;
  const auto base = testing::random_vectors(n, dim, 8);
  IvfPqParams p;
  p.dim = dim;
  p.nlist = 32;
  p.m = 8;
  p.store_vectors = true;
  IvfPqIndex index(p);
  EXPECT_THROW(index.add(base.data(), 1), Error);  // untrained
  index.train(base.data(), n);
  index.add(base.data(), n);

  LabelSetFilter allowed({1, 2, 3, 2000});
  IvfPqSearchParams sp;
  sp.nprobe = 32;
  sp.filter = &allowed;
  const auto r = index.search(base.data(), 10, sp);
  ASSERT_EQ(r.size(), 4u);
  for (const auto& nb : r) EXPECT_TRUE(allowed.contains(nb.label));

  testing::TempPath path("ivf.qvr");
  index.save(path.str());
  const IvfPqIndex loaded = IvfPqIndex::load(path.str());
  EXPECT_EQ(loaded.size(), n);
  EXPECT_EQ(loaded.list_sizes(), index.list_sizes());
  sp.filter = nullptr;
  sp.refine_factor = 2;
  EXPECT_EQ(loaded.search(base.data() + 5 * dim, 5, sp),
            index.search(base.data() + 5 * dim, 5, sp));
  EXPECT_EQ(loaded.search(base.data() + 5 * dim, 1, sp)[0].label, 5u);
}

}  // namespace
}  // namespace quiver

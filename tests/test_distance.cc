#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "quiver/distance.h"
#include "quiver/quantizer.h"
#include "test_util.h"

namespace quiver {
namespace {

// Compares every compiled-in, runnable ISA against the scalar reference on awkward lengths
// (exercising main loop, 4/8-wide loop and scalar tail) and unaligned pointers.
class KernelTest : public ::testing::TestWithParam<simd::Isa> {};

constexpr std::size_t kDims[] = {1,  2,  3,  4,  5,   7,   8,   9,   15,  16,  17,  31,   32,  33,
                                 63, 64, 65, 96, 100, 127, 128, 129, 256, 384, 768, 1000, 1536};

float rel_err(float got, float want) {
  return std::abs(got - want) / std::max(1.0f, std::abs(want));
}

TEST_P(KernelTest, FloatKernelsMatchScalar) {
  const simd::KernelTable* k = simd::get(GetParam());
  ASSERT_NE(k, nullptr);
  const simd::KernelTable* ref = simd::get(simd::Isa::kScalar);
  for (std::size_t dim : kDims) {
    // +1 so that a + 1 is a misaligned pointer.
    auto a = testing::random_vectors(1, dim + 1, dim);
    auto b = testing::random_vectors(1, dim + 1, dim + 1000);
    for (int offset : {0, 1}) {
      const float* pa = a.data() + offset;
      const float* pb = b.data() + offset;
      EXPECT_LT(rel_err(k->l2sqr(pa, pb, dim), ref->l2sqr(pa, pb, dim)), 1e-5f) << "dim=" << dim;
      EXPECT_LT(rel_err(k->dot(pa, pb, dim), ref->dot(pa, pb, dim)), 1e-5f) << "dim=" << dim;
    }
  }
}

TEST_P(KernelTest, Sq8KernelsMatchScalar) {
  const simd::KernelTable* k = simd::get(GetParam());
  ASSERT_NE(k, nullptr);
  const simd::KernelTable* ref = simd::get(simd::Isa::kScalar);
  std::mt19937 rng(7);
  for (std::size_t dim : kDims) {
    auto q = testing::random_vectors(1, dim, dim);
    auto vmin = testing::random_vectors(1, dim, dim + 1);
    std::vector<float> vscale(dim);
    std::vector<std::uint8_t> code(dim);
    for (std::size_t i = 0; i < dim; ++i) {
      vscale[i] = 0.01f + static_cast<float>(i % 7) * 0.003f;
      code[i] = static_cast<std::uint8_t>(rng() & 0xff);
    }
    EXPECT_LT(rel_err(k->l2sqr_sq8(q.data(), code.data(), vmin.data(), vscale.data(), dim),
                      ref->l2sqr_sq8(q.data(), code.data(), vmin.data(), vscale.data(), dim)),
              1e-5f)
        << "dim=" << dim;
    EXPECT_LT(rel_err(k->dot_sq8(q.data(), code.data(), vmin.data(), vscale.data(), dim),
                      ref->dot_sq8(q.data(), code.data(), vmin.data(), vscale.data(), dim)),
              1e-5f)
        << "dim=" << dim;
  }
}

TEST_P(KernelTest, ZeroLengthIsZero) {
  const simd::KernelTable* k = simd::get(GetParam());
  ASSERT_NE(k, nullptr);
  const float x = 1.0f;
  EXPECT_EQ(k->l2sqr(&x, &x, 0), 0.0f);
  EXPECT_EQ(k->dot(&x, &x, 0), 0.0f);
}

INSTANTIATE_TEST_SUITE_P(AllIsas, KernelTest, ::testing::ValuesIn(simd::available()),
                         [](const auto& param_info) {
                           return std::string(simd::to_string(param_info.param));
                         });

TEST(Distance, ActiveIsBestAvailable) {
  const auto isas = simd::available();
  ASSERT_FALSE(isas.empty());
  EXPECT_EQ(isas.front(), simd::Isa::kScalar);
  if (std::getenv("QUIVER_SIMD") == nullptr) {
    EXPECT_EQ(simd::active().isa, isas.back());
  }
}

TEST(Distance, NormalizeProducesUnitVectors) {
  auto v = testing::random_vectors(1, 77, 3);
  simd::normalize(v.data(), v.size());
  EXPECT_NEAR(simd::active().dot(v.data(), v.data(), v.size()), 1.0f, 1e-5f);
  std::vector<float> zero(5, 0.0f);
  simd::normalize(zero.data(), zero.size());  // must not produce NaN
  for (float x : zero) EXPECT_EQ(x, 0.0f);
}

TEST(ScalarQuantizer, RoundTripErrorIsBoundedByHalfAStep) {
  const std::size_t dim = 40, n = 500;
  auto data = testing::random_vectors(n, dim, 11);
  ScalarQuantizer sq(dim);
  sq.train(data.data(), n);
  std::vector<std::uint8_t> code(dim);
  std::vector<float> back(dim);
  for (std::size_t i = 0; i < n; ++i) {
    sq.encode(data.data() + i * dim, code.data());
    sq.decode(code.data(), back.data());
    for (std::size_t j = 0; j < dim; ++j) {
      EXPECT_LE(std::abs(back[j] - data[i * dim + j]), 0.5f * sq.vscale()[j] + 1e-5f);
    }
  }
}

TEST(ScalarQuantizer, ClampsOutOfRangeAndHandlesConstantDims) {
  const std::size_t dim = 3;
  std::vector<float> train = {0, 5, 1, 1, 5, 2};  // dim 1 is constant
  ScalarQuantizer sq(dim);
  sq.train(train.data(), 2);
  std::vector<float> v = {-10, 5, 100};
  std::vector<std::uint8_t> code(dim);
  sq.encode(v.data(), code.data());
  EXPECT_EQ(code[0], 0);
  EXPECT_EQ(code[2], 255);
  std::vector<float> back(dim);
  sq.decode(code.data(), back.data());
  EXPECT_FLOAT_EQ(back[1], 5.0f);
}

}  // namespace
}  // namespace quiver

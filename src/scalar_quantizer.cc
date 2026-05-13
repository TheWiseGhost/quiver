#include <algorithm>
#include <cmath>
#include <limits>

#include "quiver/quantizer.h"

namespace quiver {

void ScalarQuantizer::train(const float* vectors, std::size_t n) {
  QUIVER_CHECK(dim_ > 0, "ScalarQuantizer has no dimension");
  QUIVER_CHECK(n > 0 && vectors != nullptr, "ScalarQuantizer::train needs at least one vector");
  std::vector<float> lo(dim_, std::numeric_limits<float>::infinity());
  std::vector<float> hi(dim_, -std::numeric_limits<float>::infinity());
  bool finite = true;
  for (std::size_t i = 0; i < n; ++i) {
    const float* x = vectors + i * dim_;
    for (std::size_t j = 0; j < dim_; ++j) {
      finite &= std::isfinite(x[j]);  // std::min/max silently skip NaN, so check explicitly
      lo[j] = std::min(lo[j], x[j]);
      hi[j] = std::max(hi[j], x[j]);
    }
  }
  QUIVER_CHECK(finite, "ScalarQuantizer::train: training data contains NaN or infinity");
  std::vector<float> scale(dim_);
  for (std::size_t j = 0; j < dim_; ++j) {
    const float range = hi[j] - lo[j];
    QUIVER_CHECK(std::isfinite(range), "ScalarQuantizer::train: value range overflows float");
    // A constant dimension still needs a positive scale; any value decodes back exactly.
    scale[j] = range > 0.0f ? range / 255.0f : 1.0f;
  }
  vmin_ = std::move(lo);
  vscale_ = std::move(scale);
}

void ScalarQuantizer::set_params(std::vector<float> vmin, std::vector<float> vscale) {
  QUIVER_CHECK(vmin.size() == dim_ && vscale.size() == dim_, "SQ8 parameter size mismatch");
  for (std::size_t j = 0; j < dim_; ++j) {
    QUIVER_CHECK(std::isfinite(vmin[j]) && std::isfinite(vscale[j]) && vscale[j] > 0.0f,
                 "invalid SQ8 parameters");
  }
  vmin_ = std::move(vmin);
  vscale_ = std::move(vscale);
}

void ScalarQuantizer::encode(const float* vector, std::uint8_t* code) const noexcept {
  for (std::size_t j = 0; j < dim_; ++j) {
    const float t = std::nearbyint((vector[j] - vmin_[j]) / vscale_[j]);
    // NaN compares false everywhere and lands on 0.
    code[j] = static_cast<std::uint8_t>(t >= 255.0f ? 255.0f : (t > 0.0f ? t : 0.0f));
  }
}

void ScalarQuantizer::decode(const std::uint8_t* code, float* vector) const noexcept {
  for (std::size_t j = 0; j < dim_; ++j) {
    vector[j] = vmin_[j] + static_cast<float>(code[j]) * vscale_[j];
  }
}

}  // namespace quiver

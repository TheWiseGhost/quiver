// Loaders for the standard ANN benchmark formats.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace quiver::bench {

struct Matrix {
  std::vector<float> data;
  std::size_t rows = 0;
  std::size_t cols = 0;
  const float* row(std::size_t i) const { return data.data() + i * cols; }
};

struct IntMatrix {
  std::vector<std::int32_t> data;
  std::size_t rows = 0;
  std::size_t cols = 0;
};

/// .fvecs / .bvecs (TEXMEX: per-row int32 dim header) and .fbin / .u8bin (big-ann-benchmarks:
/// int32 rows, int32 cols header). `max_rows` = 0 reads everything.
Matrix load_vectors(const std::string& path, std::size_t max_rows = 0);

/// .ivecs or .ibin ground-truth neighbor ids.
IntMatrix load_ids(const std::string& path, std::size_t max_rows = 0);

void save_ivecs(const std::string& path, const IntMatrix& m);

}  // namespace quiver::bench

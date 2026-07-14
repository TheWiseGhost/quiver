#include "dataset.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace quiver::bench {
namespace {

struct FileCloser {
  void operator()(std::FILE* f) const { std::fclose(f); }
};
using File = std::unique_ptr<std::FILE, FileCloser>;

File open_or_throw(const std::string& path, const char* mode) {
  File f(std::fopen(path.c_str(), mode));
  if (!f) throw std::runtime_error("cannot open " + path);
  return f;
}

bool ends_with(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::size_t file_size(std::FILE* f) {
  std::fseek(f, 0, SEEK_END);
  const long size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  return static_cast<std::size_t>(size);
}

void read_exact(std::FILE* f, void* dst, std::size_t bytes, const std::string& path) {
  if (std::fread(dst, 1, bytes, f) != bytes) throw std::runtime_error("short read in " + path);
}

// TEXMEX layout: each row is [int32 dim][dim elements of T].
template <class T, class Out>
void load_vecs(const std::string& path, std::size_t max_rows, std::vector<Out>& out,
               std::size_t& rows, std::size_t& cols) {
  File f = open_or_throw(path, "rb");
  const std::size_t size = file_size(f.get());
  std::int32_t dim = 0;
  read_exact(f.get(), &dim, 4, path);
  if (dim <= 0) throw std::runtime_error("bad dimension in " + path);
  cols = static_cast<std::size_t>(dim);
  const std::size_t row_bytes = 4 + cols * sizeof(T);
  rows = size / row_bytes;
  if (max_rows != 0 && max_rows < rows) rows = max_rows;
  std::fseek(f.get(), 0, SEEK_SET);
  out.resize(rows * cols);
  std::vector<T> buf(cols);
  for (std::size_t i = 0; i < rows; ++i) {
    read_exact(f.get(), &dim, 4, path);
    if (static_cast<std::size_t>(dim) != cols) throw std::runtime_error("ragged rows in " + path);
    read_exact(f.get(), buf.data(), cols * sizeof(T), path);
    for (std::size_t j = 0; j < cols; ++j) out[i * cols + j] = static_cast<Out>(buf[j]);
  }
}

// big-ann layout: [int32 rows][int32 cols] then rows*cols elements of T.
template <class T, class Out>
void load_bin(const std::string& path, std::size_t max_rows, std::vector<Out>& out,
              std::size_t& rows, std::size_t& cols) {
  File f = open_or_throw(path, "rb");
  std::int32_t header[2];
  read_exact(f.get(), header, 8, path);
  rows = static_cast<std::size_t>(header[0]);
  cols = static_cast<std::size_t>(header[1]);
  if (max_rows != 0 && max_rows < rows) rows = max_rows;
  std::vector<T> buf(rows * cols);
  read_exact(f.get(), buf.data(), buf.size() * sizeof(T), path);
  out.assign(buf.begin(), buf.end());
}

}  // namespace

Matrix load_vectors(const std::string& path, std::size_t max_rows) {
  Matrix m;
  if (ends_with(path, ".fvecs")) {
    load_vecs<float>(path, max_rows, m.data, m.rows, m.cols);
  } else if (ends_with(path, ".bvecs")) {
    load_vecs<std::uint8_t>(path, max_rows, m.data, m.rows, m.cols);
  } else if (ends_with(path, ".fbin")) {
    load_bin<float>(path, max_rows, m.data, m.rows, m.cols);
  } else if (ends_with(path, ".u8bin")) {
    load_bin<std::uint8_t>(path, max_rows, m.data, m.rows, m.cols);
  } else {
    throw std::runtime_error("unknown vector format: " + path);
  }
  return m;
}

IntMatrix load_ids(const std::string& path, std::size_t max_rows) {
  IntMatrix m;
  if (ends_with(path, ".ivecs")) {
    load_vecs<std::int32_t>(path, max_rows, m.data, m.rows, m.cols);
  } else if (ends_with(path, ".ibin")) {
    load_bin<std::int32_t>(path, max_rows, m.data, m.rows, m.cols);
  } else {
    throw std::runtime_error("unknown id format: " + path);
  }
  return m;
}

void save_ivecs(const std::string& path, const IntMatrix& m) {
  File f = open_or_throw(path, "wb");
  const auto dim = static_cast<std::int32_t>(m.cols);
  for (std::size_t i = 0; i < m.rows; ++i) {
    std::fwrite(&dim, 4, 1, f.get());
    std::fwrite(m.data.data() + i * m.cols, 4, m.cols, f.get());
  }
}

}  // namespace quiver::bench

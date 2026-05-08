#include "quiver/flat_index.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <shared_mutex>

#include "internal/file_format.h"
#include "internal/search_util.h"
#include "quiver/distance.h"

namespace quiver {
namespace {

using internal::Candidate;
using internal::FarthestOnTop;

constexpr std::uint32_t kMeta = internal::fourcc("META");
constexpr std::uint32_t kVectors = internal::fourcc("VECT");
constexpr std::uint32_t kLabels = internal::fourcc("LABL");

struct FlatMeta {
  std::uint64_t dim;
  std::uint64_t count;
  std::uint32_t metric;
  std::uint32_t reserved;
};

}  // namespace

namespace {
void check_finite(const float* v, std::size_t count) {
  bool ok = true;
  for (std::size_t i = 0; i < count; ++i) ok &= std::isfinite(v[i]);
  QUIVER_CHECK(ok, "vectors must not contain NaN or infinity");
}
}  // namespace

FlatIndex::FlatIndex(std::size_t dim, Metric metric) : dim_(dim), metric_(metric) {
  QUIVER_CHECK(dim >= 1 && dim <= kMaxDimension,
               "dimension must be in [1, " + std::to_string(kMaxDimension) + "]");
  QUIVER_CHECK(
      metric == Metric::kL2 || metric == Metric::kInnerProduct || metric == Metric::kCosine,
      "unknown metric");
}

void FlatIndex::add(const float* vectors, std::size_t n, const Label* labels) {
  if (n == 0) return;
  QUIVER_CHECK(vectors != nullptr, "vectors must not be null");
  check_finite(vectors, n * dim_);
  if (labels != nullptr) {
    QUIVER_CHECK(std::find(labels, labels + n, kInvalidLabel) == labels + n,
                 "label 2^64-1 is reserved (it marks empty result slots)");
  }
  std::unique_lock lock(*mu_);
  const std::size_t first = labels_.size();
  // Rows are tracked as 32-bit ids in the search heap.
  QUIVER_CHECK(first + n <= std::numeric_limits<NodeId>::max(),
               "FlatIndex holds at most 2^32 vectors");
  data_.insert(data_.end(), vectors, vectors + n * dim_);
  if (metric_ == Metric::kCosine) {
    for (std::size_t i = first; i < first + n; ++i) simd::normalize(data_.data() + i * dim_, dim_);
  }
  for (std::size_t i = 0; i < n; ++i) labels_.push_back(labels ? labels[i] : first + i);
}

void FlatIndex::search_one(const float* query, std::size_t k, const Filter* filter,
                           Label* out_labels, float* out_distances,
                           std::vector<float>& scratch) const {
  const simd::KernelTable& kt = simd::active();
  const float* q = query;
  if (metric_ == Metric::kCosine) {
    scratch.assign(query, query + dim_);
    simd::normalize(scratch.data(), dim_);
    q = scratch.data();
  }
  const bool l2 = metric_ == Metric::kL2;

  const std::size_t n = labels_.size();
  std::vector<Candidate> heap;
  heap.reserve(std::min(k, n) + 1);
  for (std::size_t i = 0; i < n; ++i) {
    if (filter != nullptr && !filter->contains(labels_[i])) continue;
    const float* x = data_.data() + i * dim_;
    float d = l2 ? kt.l2sqr(q, x, dim_) : 1.0f - kt.dot(q, x, dim_);
    d = d == d ? d : std::numeric_limits<float>::infinity();  // keep the heap's ordering strict
    if (heap.size() < k) {
      internal::heap_push<FarthestOnTop>(heap, {d, static_cast<NodeId>(i)});
    } else if (d < heap.front().dist) {
      std::pop_heap(heap.begin(), heap.end(), FarthestOnTop{});
      heap.back() = {d, static_cast<NodeId>(i)};
      std::push_heap(heap.begin(), heap.end(), FarthestOnTop{});
    }
  }
  std::sort_heap(heap.begin(), heap.end(), FarthestOnTop{});
  for (std::size_t j = 0; j < k; ++j) {
    if (j < heap.size()) {
      out_labels[j] = labels_[heap[j].id];
      out_distances[j] = heap[j].dist;
    } else {
      out_labels[j] = kInvalidLabel;
      out_distances[j] = std::numeric_limits<float>::infinity();
    }
  }
}

std::vector<Neighbor> FlatIndex::search(const float* query, std::size_t k,
                                        const Filter* filter) const {
  QUIVER_CHECK(query != nullptr, "query must not be null");
  check_finite(query, dim_);
  std::vector<Label> labels(k);
  std::vector<float> distances(k);
  std::vector<float> scratch;
  std::shared_lock lock(*mu_);
  search_one(query, k, filter, labels.data(), distances.data(), scratch);
  std::vector<Neighbor> out;
  out.reserve(k);
  for (std::size_t j = 0; j < k && labels[j] != kInvalidLabel; ++j) {
    out.push_back({distances[j], labels[j]});
  }
  return out;
}

void FlatIndex::search_batch(const float* queries, std::size_t nq, std::size_t k, Label* labels,
                             float* distances, const Filter* filter, ThreadPool* pool) const {
  if (nq == 0 || k == 0) return;
  QUIVER_CHECK(queries != nullptr && labels != nullptr && distances != nullptr,
               "search_batch: null buffer");
  check_finite(queries, nq * dim_);
  std::shared_lock lock(*mu_);  // held by this thread while the pool works on our behalf
  ThreadPool& tp = pool ? *pool : ThreadPool::global();
  tp.parallel_for(nq, 1, [&](std::size_t begin, std::size_t end) {
    std::vector<float> scratch;
    for (std::size_t i = begin; i < end; ++i) {
      search_one(queries + i * dim_, k, filter, labels + i * k, distances + i * k, scratch);
    }
  });
}

std::size_t FlatIndex::size() const {
  std::shared_lock lock(*mu_);
  return labels_.size();
}

void FlatIndex::save(const std::string& path) const {
  std::shared_lock lock(*mu_);
  internal::FileWriter w(internal::IndexKind::kFlat);
  w.add_pod(kMeta, FlatMeta{dim_, labels_.size(), static_cast<std::uint32_t>(metric_), 0});
  w.add_bytes(kVectors, data_.data(), data_.size() * sizeof(float));
  w.add_bytes(kLabels, labels_.data(), labels_.size() * sizeof(Label));
  w.commit(path);
}

FlatIndex FlatIndex::load(const std::string& path, const LoadOptions& options) {
  auto image = internal::FileImage::open(path, internal::IndexKind::kFlat,
                                         {options.mmap, options.verify_checksums});
  const auto meta = image->pod<FlatMeta>(kMeta);
  QUIVER_CHECK(meta.dim > 0 && meta.dim <= kMaxDimension, "corrupt index file: bad dimension");
  QUIVER_CHECK(meta.metric <= static_cast<std::uint32_t>(Metric::kCosine),
               "corrupt index file: unknown metric");
  QUIVER_CHECK(meta.count <= image->size() / sizeof(Label), "corrupt index file: bad vector count");
  FlatIndex index(static_cast<std::size_t>(meta.dim), static_cast<Metric>(meta.metric));
  const auto count = static_cast<std::size_t>(meta.count);
  QUIVER_CHECK(count <= image->size() / (index.dim_ * sizeof(float) + 1),
               "corrupt index file: bad vector count");
  const auto vectors = image->array<float>(kVectors, count * index.dim_);
  const auto labels = image->array<Label>(kLabels, count);
  QUIVER_CHECK(
      std::all_of(vectors.begin(), vectors.end(), [](float v) { return std::isfinite(v); }),
      "corrupt index file: non-finite vector values");
  index.data_.assign(vectors.begin(), vectors.end());
  index.labels_.assign(labels.begin(), labels.end());
  return index;
}

}  // namespace quiver

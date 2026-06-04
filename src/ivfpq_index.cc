#include "quiver/ivfpq_index.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <random>
#include <shared_mutex>

#include "internal/centroid_table.h"
#include "internal/file_format.h"
#include "internal/search_util.h"
#include "quiver/distance.h"
#include "quiver/quantizer.h"

namespace quiver {
namespace {

using internal::Candidate;

constexpr float kInf = std::numeric_limits<float>::infinity();

constexpr std::uint32_t kMetaTag = internal::fourcc("META");
constexpr std::uint32_t kCentroidsTag = internal::fourcc("CENT");
constexpr std::uint32_t kCodebookTag = internal::fourcc("PQCB");
constexpr std::uint32_t kOffsetsTag = internal::fourcc("LOFF");
constexpr std::uint32_t kCodesTag = internal::fourcc("CODE");
constexpr std::uint32_t kLabelsTag = internal::fourcc("LABL");
constexpr std::uint32_t kVectorsTag = internal::fourcc("VECT");

struct IvfPqMeta {
  std::uint64_t dim;
  std::uint64_t nlist;
  std::uint64_t m;
  std::uint64_t count;
  std::uint64_t train_iterations;
  std::uint64_t seed;
  std::uint32_t metric;
  std::uint32_t store_vectors;
  std::uint32_t trained;
  std::uint32_t reserved;
};

struct InvertedList {
  std::vector<std::uint8_t> codes;  // size() * m
  std::vector<Label> labels;
  std::vector<float> vectors;  // size() * dim, only with store_vectors
};

/// Per-thread buffers for search.
struct Scratch {
  std::vector<float> query;
  std::vector<float> residual;
  std::vector<float> table;
  std::vector<float> dists;
  std::vector<Candidate> probes;
  std::vector<std::pair<float, std::uint64_t>> top;
  std::vector<float> query_table;
  std::vector<float> coarse_buf;
};

using ScratchLease = internal::ScratchLease<Scratch>;

void check_finite(const float* v, std::size_t count) {
  bool ok = true;
  for (std::size_t i = 0; i < count; ++i) ok &= std::isfinite(v[i]);
  QUIVER_CHECK(ok, "vectors must not contain NaN or infinity");
}

constexpr std::size_t kMaxLists = std::size_t{1} << 24;

/// The precomputed L2 term table is used when it fits in this budget.
constexpr std::size_t kMaxPrecomputedBytes = std::size_t{256} << 20;

}  // namespace

struct IvfPqIndex::Impl {
  IvfPqParams params;
  std::vector<float> centroids;  // nlist x dim
  internal::CentroidTable coarse;
  ProductQuantizer pq;
  // L2 by-residual decomposition (as in FAISS):
  //   ||q - c - r||^2 = ||q - c||^2 + sum_j ( ||r_j||^2 + 2<c_j, r_j> )  -  2 sum_j <q_j, r_j>
  //                     \_ coarse _/       \____ precomputed[l][j][code_j] ____/   \_ per query _/
  // so probing a cell costs m*256 adds instead of computing a fresh 256*dim table.
  std::vector<float> precomputed;  // nlist x m x 256, empty if over budget or metric is IP
  std::vector<InvertedList> lists;
  std::size_t count = 0;
  // Guards lists/count. The trained state (centroids, coarse, pq, precomputed) is written once,
  // under the exclusive lock, and is read-only afterwards.
  mutable std::shared_mutex mu;
  bool trained = false;

  explicit Impl(const IvfPqParams& p) : params(p), pq(p.dim, p.m) {
    QUIVER_CHECK(p.dim >= 1 && p.dim <= kMaxDimension,
                 "dimension must be in [1, " + std::to_string(kMaxDimension) + "]");
    QUIVER_CHECK(p.nlist >= 1 && p.nlist <= kMaxLists, "nlist must be in [1, 2^24]");
    QUIVER_CHECK(
        p.metric == Metric::kL2 || p.metric == Metric::kInnerProduct || p.metric == Metric::kCosine,
        "unknown metric");
    lists.resize(p.nlist);
  }

  /// Cell assignment uses the same notion of "closest" as probing (L2, or max inner product).
  /// For cosine, centroids are unit-normalized, so the two coincide.
  bool assign_by_ip() const noexcept { return params.metric == Metric::kInnerProduct; }

  bool ip() const noexcept { return params.metric != Metric::kL2; }

  /// The L2 precomputed term table for (centroids, pq), or empty when not applicable.
  std::vector<float> precompute(const std::vector<float>& cents, const ProductQuantizer& q) const {
    const std::size_t ks = ProductQuantizer::kSubCentroids, m = q.m(), dsub = q.dsub();
    if (ip() || params.nlist * m * ks * sizeof(float) > kMaxPrecomputedBytes) return {};
    std::vector<float> table(params.nlist * m * ks);
    const auto& cb = q.centroids();
    for (std::size_t l = 0; l < params.nlist; ++l) {
      for (std::size_t j = 0; j < m; ++j) {
        const float* cl = cents.data() + l * params.dim + j * dsub;
        for (std::size_t c = 0; c < ks; ++c) {
          const float* r = cb.data() + (j * ks + c) * dsub;
          float v = 0.0f;
          for (std::size_t t = 0; t < dsub; ++t) v += r[t] * r[t] + 2.0f * cl[t] * r[t];
          table[(l * m + j) * ks + c] = v;
        }
      }
    }
    return table;
  }

  const float* prepare(const float* v, std::vector<float>& buf) const {
    if (params.metric != Metric::kCosine) return v;
    buf.assign(v, v + params.dim);
    simd::normalize(buf.data(), params.dim);
    return buf.data();
  }

  /// Exact distance under the index metric (for refinement).
  float exact(const float* q, const float* x) const noexcept {
    const auto& k = simd::active();
    return ip() ? 1.0f - k.dot(q, x, params.dim) : k.l2sqr(q, x, params.dim);
  }

  void train(const float* in, std::size_t n, ThreadPool* pool) {
    QUIVER_CHECK(n >= params.nlist && n >= ProductQuantizer::kSubCentroids,
                 "IVF-PQ training needs at least max(nlist, 256) vectors");
    check_finite(in, n * params.dim);
    {
      std::shared_lock lock(mu);
      QUIVER_CHECK(!trained, "index is already trained");
    }
    const std::size_t d = params.dim;
    std::vector<float> data(in, in + n * d);
    if (params.metric == Metric::kCosine) {
      for (std::size_t i = 0; i < n; ++i) simd::normalize(data.data() + i * d, d);
    }
    // Everything is built into locals and committed at the end, so a failure (or a concurrent
    // train()) can never leave a half-trained index.
    KMeansParams kp;
    kp.k = params.nlist;
    kp.iterations = params.train_iterations;
    kp.seed = params.seed;
    std::vector<float> new_centroids = kmeans(data.data(), n, d, kp, pool);
    if (params.metric == Metric::kCosine) {
      for (std::size_t l = 0; l < params.nlist; ++l)
        simd::normalize(new_centroids.data() + l * d, d);
    }

    // PQ is trained on residuals so codes describe the offset within a cell, which has far
    // less variance than the raw vector.
    ThreadPool& tp = pool ? *pool : ThreadPool::global();
    internal::CentroidTable new_coarse(new_centroids.data(), params.nlist, d);
    std::vector<float> residuals(n * d);
    tp.parallel_for(n, 256, [&](std::size_t b, std::size_t e) {
      std::vector<std::size_t> cell(e - b);
      new_coarse.nearest_batch(data.data() + b * d, e - b, cell.data(), assign_by_ip());
      for (std::size_t i = b; i < e; ++i) {
        const float* x = data.data() + i * d;
        const float* c = new_centroids.data() + cell[i - b] * d;
        for (std::size_t j = 0; j < d; ++j) residuals[i * d + j] = x[j] - c[j];
      }
    });
    // PQ codebooks barely improve beyond ~65k samples; take a random subset, not a prefix
    // (input order is often correlated, e.g. sorted by source or time).
    const std::size_t pq_n = std::min<std::size_t>(n, 65536);
    if (pq_n < n) {
      std::mt19937_64 rng(params.seed + 2);
      for (std::size_t i = 0; i < pq_n; ++i) {
        const std::size_t j = i + rng() % (n - i);
        std::swap_ranges(residuals.begin() + static_cast<std::ptrdiff_t>(i * d),
                         residuals.begin() + static_cast<std::ptrdiff_t>((i + 1) * d),
                         residuals.begin() + static_cast<std::ptrdiff_t>(j * d));
      }
    }
    ProductQuantizer new_pq(params.dim, params.m);
    new_pq.train(residuals.data(), pq_n, pool, params.train_iterations, params.seed + 1);
    std::vector<float> new_precomputed = precompute(new_centroids, new_pq);

    std::unique_lock lock(mu);
    QUIVER_CHECK(!trained, "index is already trained");
    centroids = std::move(new_centroids);
    coarse = std::move(new_coarse);
    pq = std::move(new_pq);
    precomputed = std::move(new_precomputed);
    trained = true;
  }

  void add(const float* in, std::size_t n, const Label* labels, ThreadPool* pool) {
    if (n == 0) return;
    check_finite(in, n * params.dim);
    if (labels != nullptr) {
      QUIVER_CHECK(std::find(labels, labels + n, kInvalidLabel) == labels + n,
                   "label 2^64-1 is reserved (it marks empty result slots)");
    }
    const std::size_t d = params.dim, m = params.m;
    std::vector<std::uint32_t> assign(n);
    std::vector<std::uint8_t> codes(n * m);
    std::vector<float> normed;
    const float* x = in;
    if (params.metric == Metric::kCosine) {
      normed.assign(in, in + n * d);
      for (std::size_t i = 0; i < n; ++i) simd::normalize(normed.data() + i * d, d);
      x = normed.data();
    }
    {
      // Observe `trained` under the lock (acquire). The trained state never changes after it is
      // set, so encoding then needs no lock. Holding the lock across parallel_for would deadlock:
      // pool chunks of a concurrent search_batch need the shared lock, and a waiting writer
      // blocks new shared lockers (writer-preferring shared_mutex).
      {
        std::shared_lock lock(mu);
        QUIVER_CHECK(trained, "IVF-PQ index must be trained before add()");
      }
      ThreadPool& tp = pool ? *pool : ThreadPool::global();
      tp.parallel_for(n, 64, [&](std::size_t b, std::size_t e) {
        std::vector<float> residual(d);
        std::vector<std::size_t> cell(e - b);
        coarse.nearest_batch(x + b * d, e - b, cell.data(), assign_by_ip());
        for (std::size_t i = b; i < e; ++i) {
          const float* v = x + i * d;
          const std::size_t l = cell[i - b];
          assign[i] = static_cast<std::uint32_t>(l);
          for (std::size_t j = 0; j < d; ++j) residual[j] = v[j] - centroids[l * d + j];
          pq.encode(residual.data(), codes.data() + i * m);
        }
      });
    }

    std::unique_lock lock(mu);
    // Reserve every touched list first; after that the appends cannot throw, so a failure
    // leaves the lists exactly as they were (codes, labels and vectors stay in step).
    std::vector<std::size_t> extra(params.nlist, 0);
    for (std::uint32_t l : assign) ++extra[l];
    for (std::size_t l = 0; l < params.nlist; ++l) {
      if (extra[l] == 0) continue;
      InvertedList& list = lists[l];
      QUIVER_CHECK(list.labels.size() + extra[l] <= std::numeric_limits<std::uint32_t>::max(),
                   "too many vectors in one IVF cell (2^32 limit); use a larger nlist");
      list.codes.reserve(list.codes.size() + extra[l] * m);
      list.labels.reserve(list.labels.size() + extra[l]);
      if (params.store_vectors) list.vectors.reserve(list.vectors.size() + extra[l] * d);
    }
    for (std::size_t i = 0; i < n; ++i) {
      InvertedList& list = lists[assign[i]];
      const auto ci = codes.begin() + static_cast<std::ptrdiff_t>(i * m);
      list.codes.insert(list.codes.end(), ci, ci + static_cast<std::ptrdiff_t>(m));
      list.labels.push_back(labels ? labels[i] : count + i);
      if (params.store_vectors) list.vectors.insert(list.vectors.end(), x + i * d, x + (i + 1) * d);
    }
    count += n;
  }

  /// One query. The caller holds `mu` (shared) for the whole batch: pool chunks must never take
  /// an index lock themselves (see add()).
  void search_into(const float* query, std::size_t k, const IvfPqSearchParams& p, Label* out_labels,
                   float* out_dist) const {
    std::fill(out_labels, out_labels + k, kInvalidLabel);
    std::fill(out_dist, out_dist + k, kInf);
    if (k == 0 || !trained) return;
    ScratchLease lease;
    Scratch& s = *lease;
    const std::size_t d = params.dim, nlist = params.nlist;
    const float* q = prepare(query, s.query);

    // 1. Pick the nprobe closest cells.
    const std::size_t nprobe = std::min(std::max<std::size_t>(p.nprobe, 1), nlist);
    s.probes.resize(nlist);
    s.coarse_buf.resize(coarse.padded_k());
    if (ip()) {
      coarse.dot(q, s.coarse_buf.data());
      for (std::size_t l = 0; l < nlist; ++l)
        s.probes[l] = {-s.coarse_buf[l], static_cast<NodeId>(l)};
    } else {
      coarse.l2(q, s.coarse_buf.data());
      for (std::size_t l = 0; l < nlist; ++l)
        s.probes[l] = {s.coarse_buf[l], static_cast<NodeId>(l)};
    }
    std::partial_sort(s.probes.begin(), s.probes.begin() + static_cast<std::ptrdiff_t>(nprobe),
                      s.probes.end(),
                      [](const Candidate& a, const Candidate& b) { return a.dist < b.dist; });

    // 2. ADC-scan those cells, keeping the best `keep` candidates.
    const bool refine = p.refine_factor > 0 && params.store_vectors;
    // Saturating: k * refine_factor must neither wrap (to 0) nor exceed what exists.
    const std::size_t factor = refine ? p.refine_factor : 1;
    const std::size_t keep =
        std::min(count, factor > count / std::max<std::size_t>(k, 1) ? count : k * factor);
    if (keep == 0) return;
    const std::size_t table_size = pq.m() * ProductQuantizer::kSubCentroids;
    s.table.resize(table_size);
    const bool use_precomputed = !precomputed.empty();
    if (ip()) {
      pq.compute_ip_table(q, s.table.data());  // <q, r> does not depend on the cell
    } else if (use_precomputed) {
      s.query_table.resize(table_size);
      pq.compute_ip_table(q, s.query_table.data());
      for (float& v : s.query_table) v *= -2.0f;
    }
    s.residual.resize(d);

    // Max-heap of (distance, list << 32 | offset-in-list).
    auto& heap = s.top;
    heap.clear();
    auto cmp = [](const auto& a, const auto& b) { return a.first < b.first; };

    for (std::size_t pi = 0; pi < nprobe; ++pi) {
      const std::size_t l = s.probes[pi].id;
      const InvertedList& list = lists[l];
      const std::size_t len = list.labels.size();
      if (len == 0) continue;
      float base = 0.0f;
      if (ip()) {
        base = -s.probes[pi].dist;  // <q, c>
      } else if (use_precomputed) {
        base = s.probes[pi].dist;  // ||q - c||^2
        const float* pre = precomputed.data() + l * table_size;
        for (std::size_t t = 0; t < table_size; ++t) s.table[t] = pre[t] + s.query_table[t];
      } else {
        const float* c = centroids.data() + l * d;
        for (std::size_t j = 0; j < d; ++j) s.residual[j] = q[j] - c[j];
        pq.compute_l2_table(s.residual.data(), s.table.data());
      }
      s.dists.resize(len);
      pq.adc_batch(s.table.data(), list.codes.data(), len, s.dists.data());
      for (std::size_t i = 0; i < len; ++i) {
        float dist = ip() ? 1.0f - (base + s.dists[i]) : base + s.dists[i];
        dist = dist == dist ? dist : kInf;  // NaN would break the heap's strict weak ordering
        if (heap.size() == keep && dist >= heap.front().first) continue;
        if (p.filter != nullptr && !p.filter->contains(list.labels[i])) continue;
        const std::uint64_t ref = (static_cast<std::uint64_t>(l) << 32) | i;
        if (heap.size() < keep) {
          heap.emplace_back(dist, ref);
          std::push_heap(heap.begin(), heap.end(), cmp);
        } else {
          std::pop_heap(heap.begin(), heap.end(), cmp);
          heap.back() = {dist, ref};
          std::push_heap(heap.begin(), heap.end(), cmp);
        }
      }
    }

    // 3. Optional exact re-ranking.
    if (refine) {
      for (auto& [dist, ref] : heap) {
        const InvertedList& list = lists[ref >> 32];
        dist = exact(q, list.vectors.data() + (ref & 0xffffffffu) * d);
      }
      std::make_heap(heap.begin(), heap.end(), cmp);
      while (heap.size() > k) {
        std::pop_heap(heap.begin(), heap.end(), cmp);
        heap.pop_back();
      }
    }
    std::sort_heap(heap.begin(), heap.end(), cmp);
    const std::size_t n = std::min(k, heap.size());
    for (std::size_t j = 0; j < n; ++j) {
      const std::uint64_t ref = heap[j].second;
      out_labels[j] = lists[ref >> 32].labels[ref & 0xffffffffu];
      out_dist[j] = heap[j].first;
    }
  }
};

IvfPqIndex::IvfPqIndex(const IvfPqParams& params) : impl_(std::make_unique<Impl>(params)) {}
IvfPqIndex::IvfPqIndex(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
IvfPqIndex::~IvfPqIndex() = default;
IvfPqIndex::IvfPqIndex(IvfPqIndex&&) noexcept = default;
IvfPqIndex& IvfPqIndex::operator=(IvfPqIndex&&) noexcept = default;

void IvfPqIndex::train(const float* vectors, std::size_t n, ThreadPool* pool) {
  QUIVER_CHECK(vectors != nullptr, "vectors must not be null");
  impl_->train(vectors, n, pool);  // locks internally; training happens at most once
}

bool IvfPqIndex::is_trained() const noexcept {
  std::shared_lock lock(impl_->mu);
  return impl_->trained;
}

void IvfPqIndex::add(const float* vectors, std::size_t n, const Label* labels, ThreadPool* pool) {
  QUIVER_CHECK(vectors != nullptr || n == 0, "vectors must not be null");
  impl_->add(vectors, n, labels, pool);
}

std::vector<Neighbor> IvfPqIndex::search(const float* query, std::size_t k,
                                         const IvfPqSearchParams& params) const {
  QUIVER_CHECK(query != nullptr, "query must not be null");
  check_finite(query, impl_->params.dim);
  std::vector<Label> labels(k);
  std::vector<float> dist(k);
  {
    std::shared_lock lock(impl_->mu);
    impl_->search_into(query, k, params, labels.data(), dist.data());
  }
  std::vector<Neighbor> out;
  for (std::size_t j = 0; j < k && labels[j] != kInvalidLabel; ++j)
    out.push_back({dist[j], labels[j]});
  return out;
}

void IvfPqIndex::search_batch(const float* queries, std::size_t nq, std::size_t k, Label* labels,
                              float* distances, const IvfPqSearchParams& params,
                              ThreadPool* pool) const {
  if (nq == 0 || k == 0) return;
  QUIVER_CHECK(queries != nullptr && labels != nullptr && distances != nullptr,
               "search_batch: null buffer");
  const std::size_t d = impl_->params.dim;
  check_finite(queries, nq * d);
  // Taken once by the calling thread; workers rely on it (see Impl::search_into).
  std::shared_lock lock(impl_->mu);
  ThreadPool& tp = pool ? *pool : ThreadPool::global();
  tp.parallel_for(nq, 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      impl_->search_into(queries + i * d, k, params, labels + i * k, distances + i * k);
    }
  });
}

std::size_t IvfPqIndex::size() const noexcept {
  std::shared_lock lock(impl_->mu);
  return impl_->count;
}
std::size_t IvfPqIndex::dim() const noexcept { return impl_->params.dim; }
const IvfPqParams& IvfPqIndex::params() const noexcept { return impl_->params; }

std::size_t IvfPqIndex::memory_bytes() const noexcept {
  std::shared_lock lock(impl_->mu);
  std::size_t bytes =
      (impl_->centroids.size() + impl_->pq.centroids().size() + impl_->precomputed.size()) *
      sizeof(float);
  for (const auto& l : impl_->lists) {
    bytes += l.codes.size() + l.labels.size() * sizeof(Label) + l.vectors.size() * sizeof(float);
  }
  return bytes;
}

std::vector<std::size_t> IvfPqIndex::list_sizes() const {
  std::shared_lock lock(impl_->mu);
  std::vector<std::size_t> out;
  out.reserve(impl_->lists.size());
  for (const auto& l : impl_->lists) out.push_back(l.labels.size());
  return out;
}

void IvfPqIndex::save(const std::string& path) const {
  std::shared_lock lock(impl_->mu);
  const Impl& ix = *impl_;
  const auto& p = ix.params;
  IvfPqMeta meta{p.dim,
                 p.nlist,
                 p.m,
                 ix.count,
                 p.train_iterations,
                 p.seed,
                 static_cast<std::uint32_t>(p.metric),
                 p.store_vectors ? 1u : 0u,
                 ix.trained ? 1u : 0u,
                 0};
  std::vector<std::uint64_t> offsets(p.nlist + 1, 0);
  for (std::size_t l = 0; l < p.nlist; ++l) offsets[l + 1] = offsets[l] + ix.lists[l].labels.size();

  internal::FileWriter w(internal::IndexKind::kIvfPq);
  w.add_pod(kMetaTag, meta);
  w.add_bytes(kCentroidsTag, ix.centroids.data(), ix.centroids.size() * sizeof(float));
  w.add_bytes(kCodebookTag, ix.pq.centroids().data(), ix.pq.centroids().size() * sizeof(float));
  w.add_bytes(kOffsetsTag, offsets.data(), offsets.size() * sizeof(std::uint64_t));
  w.add(kCodesTag, ix.count * p.m, [&ix](internal::SectionSink& s) {
    for (const auto& l : ix.lists) {
      if (!l.codes.empty()) s.write(l.codes.data(), l.codes.size());
    }
  });
  w.add(kLabelsTag, ix.count * sizeof(Label), [&ix](internal::SectionSink& s) {
    for (const auto& l : ix.lists) {
      if (!l.labels.empty()) s.write(l.labels.data(), l.labels.size() * sizeof(Label));
    }
  });
  if (p.store_vectors) {
    w.add(kVectorsTag, ix.count * p.dim * sizeof(float), [&ix](internal::SectionSink& s) {
      for (const auto& l : ix.lists) {
        if (!l.vectors.empty()) s.write(l.vectors.data(), l.vectors.size() * sizeof(float));
      }
    });
  }
  w.commit(path);
}

IvfPqIndex IvfPqIndex::load(const std::string& path, const LoadOptions& options) {
  auto image = internal::FileImage::open(path, internal::IndexKind::kIvfPq,
                                         {options.mmap, options.verify_checksums});
  const std::string corrupt = "corrupt index file '" + path + "': ";
  const auto meta = image->pod<IvfPqMeta>(kMetaTag);
  QUIVER_CHECK(meta.dim >= 1 && meta.dim <= 65536, corrupt + "bad dimension");
  QUIVER_CHECK(meta.m >= 1 && meta.m <= meta.dim && meta.dim % meta.m == 0, corrupt + "bad m");
  QUIVER_CHECK(meta.nlist >= 1 && meta.nlist <= kMaxLists, corrupt + "bad nlist");
  // Bound allocations by the file size before constructing anything sized by the header.
  QUIVER_CHECK((meta.nlist + 1) * sizeof(std::uint64_t) <= image->size(), corrupt + "bad nlist");
  QUIVER_CHECK(meta.metric <= static_cast<std::uint32_t>(Metric::kCosine), corrupt + "bad metric");
  QUIVER_CHECK(meta.count <= image->size(), corrupt + "bad count");

  IvfPqParams p;
  p.dim = static_cast<std::size_t>(meta.dim);
  p.nlist = static_cast<std::size_t>(meta.nlist);
  p.m = static_cast<std::size_t>(meta.m);
  p.train_iterations = static_cast<std::size_t>(meta.train_iterations);
  p.seed = meta.seed;
  p.metric = static_cast<Metric>(meta.metric);
  p.store_vectors = meta.store_vectors != 0;
  auto impl = std::make_unique<Impl>(p);
  const auto count = static_cast<std::size_t>(meta.count);

  if (meta.trained != 0) {
    const auto cent = image->array<float>(kCentroidsTag, p.nlist * p.dim);
    const auto cb =
        image->array<float>(kCodebookTag, p.m * ProductQuantizer::kSubCentroids * (p.dim / p.m));
    const bool finite =
        std::all_of(cent.begin(), cent.end(), [](float v) { return std::isfinite(v); }) &&
        std::all_of(cb.begin(), cb.end(), [](float v) { return std::isfinite(v); });
    QUIVER_CHECK(finite, corrupt + "non-finite centroids");
    impl->centroids.assign(cent.begin(), cent.end());
    impl->coarse.reset(impl->centroids.data(), p.nlist, p.dim);
    impl->pq.set_centroids(std::vector<float>(cb.begin(), cb.end()));
    impl->precomputed = impl->precompute(impl->centroids, impl->pq);
    impl->trained = true;
  } else {
    QUIVER_CHECK(count == 0, corrupt + "untrained index with vectors");
  }
  const auto offsets = image->array<std::uint64_t>(kOffsetsTag, p.nlist + 1);
  QUIVER_CHECK(offsets[0] == 0 && offsets[p.nlist] == count, corrupt + "bad list offsets");
  for (std::size_t l = 0; l < p.nlist; ++l) {
    QUIVER_CHECK(offsets[l] <= offsets[l + 1], corrupt + "bad list offsets");
  }
  const auto codes = image->array<std::uint8_t>(kCodesTag, count * p.m);
  const auto labels = image->array<Label>(kLabelsTag, count);
  std::span<float> vectors;
  if (p.store_vectors) {
    vectors = image->array<float>(kVectorsTag, count * p.dim);
    QUIVER_CHECK(
        std::all_of(vectors.begin(), vectors.end(), [](float v) { return std::isfinite(v); }),
        corrupt + "non-finite stored vectors");
  }
  for (std::size_t l = 0; l < p.nlist; ++l) {
    const auto b = static_cast<std::size_t>(offsets[l]);
    const auto e = static_cast<std::size_t>(offsets[l + 1]);
    InvertedList& list = impl->lists[l];
    list.codes.assign(codes.begin() + static_cast<std::ptrdiff_t>(b * p.m),
                      codes.begin() + static_cast<std::ptrdiff_t>(e * p.m));
    list.labels.assign(labels.begin() + static_cast<std::ptrdiff_t>(b),
                       labels.begin() + static_cast<std::ptrdiff_t>(e));
    if (p.store_vectors) {
      list.vectors.assign(vectors.begin() + static_cast<std::ptrdiff_t>(b * p.dim),
                          vectors.begin() + static_cast<std::ptrdiff_t>(e * p.dim));
    }
  }
  impl->count = count;
  return IvfPqIndex(std::move(impl));
}

}  // namespace quiver

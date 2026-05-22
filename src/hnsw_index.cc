#include "quiver/hnsw_index.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

#include "internal/chunked_storage.h"
#include "internal/file_format.h"
#include "internal/search_util.h"
#include "quiver/distance.h"
#include "quiver/quantizer.h"
#include "simd/kernels.h"
#if QUIVER_HAVE_NEON
#include "simd/kernels_neon_inl.h"
#endif

namespace quiver {
namespace {

using internal::Candidate;
using internal::ChunkedStorage;
using internal::FarthestOnTop;
using internal::NearestOnTop;
using internal::SpinGuard;

constexpr unsigned kMaxLevel = 31;
constexpr std::size_t kMaxM = 512;
constexpr std::size_t kMaxDim = kMaxDimension;
constexpr std::size_t kMinAutoTrain = 256;  // smallest first batch SQ8 may train itself on
constexpr float kInf = std::numeric_limits<float>::infinity();

/// NaN breaks the strict weak ordering every heap and sort below relies on. Inputs are checked
/// on entry, but a loaded file (with checksums off) could still hold NaN vectors, so distances
/// are also mapped NaN -> +inf at the source (one compare-and-select).
inline float sanitize(float d) noexcept { return d == d ? d : kInf; }

void check_finite(const float* v, std::size_t count) {
  bool ok = true;
  for (std::size_t i = 0; i < count; ++i) ok &= std::isfinite(v[i]);
  QUIVER_CHECK(ok, "vectors must not contain NaN or infinity");
}

// Node lifecycle flags.
constexpr std::uint8_t kReady = 1;    // fully initialized; set once, before the node is linked
constexpr std::uint8_t kRemoved = 2;  // soft-deleted

/// Per-node bookkeeping. Plain bytes so it can live in zero-initialized chunk memory; the
/// concurrently accessed fields are only touched through std::atomic_ref.
struct NodeMeta {
  Label label;
  std::uint32_t* upper;  // `level` blocks of [count, ids[M]] for layers 1..level
  std::uint8_t level;
  std::uint8_t flags;
  std::uint8_t lock;
  std::uint8_t reserved[5];
};
static_assert(sizeof(NodeMeta) == 24 && std::is_trivially_copyable_v<NodeMeta>);

inline std::uint32_t load_acquire(const std::uint32_t& x) noexcept {
  return std::atomic_ref<std::uint32_t>(const_cast<std::uint32_t&>(x))
      .load(std::memory_order_acquire);
}
inline void store_release(std::uint32_t& x, std::uint32_t v) noexcept {
  std::atomic_ref<std::uint32_t>(x).store(v, std::memory_order_release);
}
inline std::uint8_t load_flags(const NodeMeta& m, std::memory_order order) noexcept {
  return std::atomic_ref<std::uint8_t>(const_cast<std::uint8_t&>(m.flags)).load(order);
}

// Entry point packed into one word so (node, top level) are read and published together.
// 0 means "empty index".
constexpr std::uint64_t pack_entry(NodeId id, unsigned level) noexcept {
  return (static_cast<std::uint64_t>(level + 1) << 32) | id;
}
constexpr NodeId entry_id(std::uint64_t e) noexcept { return static_cast<NodeId>(e & 0xffffffffu); }
constexpr unsigned entry_level(std::uint64_t e) noexcept {
  return static_cast<unsigned>(e >> 32) - 1;
}

/// Bump allocator for upper-layer adjacency blocks (only ~1/M of nodes need any).
class UpperArena {
 public:
  std::uint32_t* allocate(std::size_t words) {
    std::lock_guard lock(mu_);
    if (words > left_) {
      const std::size_t size = std::max(words, kBlockWords);
      blocks_.emplace_back(new std::uint32_t[size]());
      cursor_ = blocks_.back().get();
      left_ = size;
      allocated_ += size;
    }
    std::uint32_t* p = cursor_;
    cursor_ += words;
    left_ -= words;
    return p;
  }
  std::size_t allocated_words() const {
    std::lock_guard lock(mu_);
    return allocated_;
  }

 private:
  static constexpr std::size_t kBlockWords = std::size_t{1} << 16;
  mutable std::mutex mu_;
  std::vector<std::unique_ptr<std::uint32_t[]>> blocks_;
  std::uint32_t* cursor_ = nullptr;
  std::size_t left_ = 0;
  std::size_t allocated_ = 0;
};

/// Per-thread working memory, reused across calls so the hot paths never allocate.
struct Scratch {
  internal::VisitedSet visited;
  std::vector<Candidate> frontier;
  std::vector<Candidate> results;
  std::vector<Candidate> candidates;
  std::vector<Candidate> selected;
  std::vector<Candidate> merge;
  std::vector<Candidate> pruned;
  std::vector<std::vector<Candidate>> chosen;  // per layer, during one insert
  std::vector<NodeId> ids;
  std::vector<float> query;
  std::vector<float> decoded_a;
  std::vector<float> decoded_b;
};

using ScratchLease = internal::ScratchLease<Scratch>;

// ---- Distance spaces ------------------------------------------------------------------------
// Each space knows how to compare a float query to a stored node. Search code is templated on
// the space, so the storage/metric branch is resolved once per call instead of per distance.

// Kernel policies. The dynamic ones call through the runtime dispatch table; on AArch64 (where
// NEON is mandatory) the static ones let the compiler inline the kernel into the search loop, so
// distance loads can be scheduled together with the surrounding graph traversal.
struct DynamicFloat {
  simd::FloatKernel fn;
  float operator()(const float* a, const float* b, std::size_t d) const noexcept {
    return fn(a, b, d);
  }
};
struct DynamicSq8 {
  simd::Sq8Kernel fn;
  float operator()(const float* q, const std::uint8_t* c, const float* lo, const float* sc,
                   std::size_t d) const noexcept {
    return fn(q, c, lo, sc, d);
  }
};
#if QUIVER_HAVE_NEON
struct NeonL2 {
  float operator()(const float* a, const float* b, std::size_t d) const noexcept {
    return simd::neon::l2sqr_inl(a, b, d);
  }
};
struct NeonDot {
  float operator()(const float* a, const float* b, std::size_t d) const noexcept {
    return simd::neon::dot_inl(a, b, d);
  }
};
struct NeonL2Sq8 {
  float operator()(const float* q, const std::uint8_t* c, const float* lo, const float* sc,
                   std::size_t d) const noexcept {
    return simd::neon::l2sqr_sq8_inl(q, c, lo, sc, d);
  }
};
struct NeonDotSq8 {
  float operator()(const float* q, const std::uint8_t* c, const float* lo, const float* sc,
                   std::size_t d) const noexcept {
    return simd::neon::dot_sq8_inl(q, c, lo, sc, d);
  }
};
#endif

template <bool kIp, class Kernel>
struct FloatSpace {
  const ChunkedStorage* vectors;
  Kernel kernel;
  std::size_t dim;
  std::size_t prefetch_bytes;

  float operator()(const float* q, NodeId id) const noexcept {
    const float r = kernel(q, vectors->as<float>(id), dim);
    if constexpr (kIp) {
      return sanitize(1.0f - r);
    } else {
      return sanitize(r);
    }
  }
  /// The node's vector in a form usable as a query.
  const float* as_query(NodeId id, std::vector<float>& /*buf*/) const noexcept {
    return vectors->as<float>(id);
  }
  void prefetch(NodeId id) const noexcept { simd::prefetch(vectors->at(id), prefetch_bytes); }
};

template <bool kIp, class Kernel>
struct Sq8Space {
  const ChunkedStorage* codes;
  Kernel kernel;
  const ScalarQuantizer* sq;
  std::size_t dim;
  std::size_t prefetch_bytes;

  float operator()(const float* q, NodeId id) const noexcept {
    const float r = kernel(q, codes->as<std::uint8_t>(id), sq->vmin(), sq->vscale(), dim);
    if constexpr (kIp) {
      return sanitize(1.0f - r);
    } else {
      return sanitize(r);
    }
  }
  const float* as_query(NodeId id, std::vector<float>& buf) const {
    buf.resize(dim);
    sq->decode(codes->as<std::uint8_t>(id), buf.data());
    return buf.data();
  }
  void prefetch(NodeId id) const noexcept { simd::prefetch(codes->at(id), prefetch_bytes); }
};

// ---- File layout ----------------------------------------------------------------------------

constexpr std::uint32_t kMetaTag = internal::fourcc("META");
constexpr std::uint32_t kVectorsTag = internal::fourcc("VECT");
constexpr std::uint32_t kLinks0Tag = internal::fourcc("LNK0");
constexpr std::uint32_t kLabelsTag = internal::fourcc("LABL");
constexpr std::uint32_t kLevelsTag = internal::fourcc("LEVL");
constexpr std::uint32_t kFlagsTag = internal::fourcc("FLAG");
constexpr std::uint32_t kUpperOffsetsTag = internal::fourcc("UPOF");
constexpr std::uint32_t kUpperLinksTag = internal::fourcc("UPLK");
constexpr std::uint32_t kSq8Tag = internal::fourcc("SQ8P");

constexpr std::uint64_t kNoUpper = std::numeric_limits<std::uint64_t>::max();

struct HnswMeta {
  std::uint64_t dim;
  std::uint64_t M;
  std::uint64_t ef_construction;
  std::uint64_t seed;
  std::uint64_t num_slots;
  std::uint64_t upper_words;
  std::uint32_t metric;
  std::uint32_t storage;
  std::uint32_t entry_node;
  std::uint32_t entry_level_plus1;  // 0 = empty
};
static_assert(std::is_trivially_copyable_v<HnswMeta>);

}  // namespace

// =============================================================================================

struct HnswIndex::Impl {
  explicit Impl(const HnswParams& p);

  // --- configuration (immutable after construction) ---
  HnswParams params;
  std::size_t M;
  std::size_t M0;
  double level_mult;
  std::size_t code_size;
  const simd::KernelTable& kernels = simd::active();

  // --- storage ---
  ChunkedStorage vectors;  // code_size bytes per node
  ChunkedStorage links0;   // (1 + M0) uint32 per node: [count, ids...]
  ChunkedStorage metas;    // NodeMeta per node
  UpperArena arena;
  std::shared_ptr<internal::FileImage> image;  // keeps adopted file memory alive

  // --- graph state ---
  std::atomic<std::uint64_t> entry{0};
  std::mutex entry_mu;  // held by an insert that may become the new top-level entry point
  std::atomic<NodeId> next_slot{0};
  std::atomic<std::size_t> num_live{0};
  std::atomic<std::size_t> num_removed{0};

  mutable std::shared_mutex labels_mu;
  std::unordered_map<Label, NodeId> label_to_id;

  ScalarQuantizer sq;
  std::atomic<bool> trained{false};  // SQ8 parameters are written once, before this is set
  std::mutex train_mu;

  // Shared by add(), exclusive for save(): a save sees a quiescent graph. Searches never take it.
  mutable std::shared_mutex writers_mu;

  // ---- accessors ----
  NodeMeta& meta(NodeId id) const noexcept { return *metas.as<NodeMeta>(id); }
  std::uint32_t* links(NodeId id, unsigned level) const noexcept {
    return level == 0 ? links0.as<std::uint32_t>(id)
                      : meta(id).upper + static_cast<std::size_t>(level - 1) * (1 + M);
  }
  bool is_live(NodeId id) const noexcept {
    return load_flags(meta(id), std::memory_order_acquire) == kReady;
  }

  /// Resolves (storage, metric, ISA) once and calls f with the matching distance space.
  template <class F>
  decltype(auto) with_space(F&& f) const {
    const bool ip = params.metric != Metric::kL2;
    const std::size_t pf = std::min<std::size_t>(code_size, 256);
    const std::size_t d = params.dim;
#if QUIVER_HAVE_NEON
    if (kernels.isa == simd::Isa::kNeon) {
      if (params.storage == Storage::kFloat32) {
        if (ip) return f(FloatSpace<true, NeonDot>{&vectors, {}, d, pf});
        return f(FloatSpace<false, NeonL2>{&vectors, {}, d, pf});
      }
      if (ip) return f(Sq8Space<true, NeonDotSq8>{&vectors, {}, &sq, d, pf});
      return f(Sq8Space<false, NeonL2Sq8>{&vectors, {}, &sq, d, pf});
    }
#endif
    if (params.storage == Storage::kFloat32) {
      if (ip) return f(FloatSpace<true, DynamicFloat>{&vectors, {kernels.dot}, d, pf});
      return f(FloatSpace<false, DynamicFloat>{&vectors, {kernels.l2sqr}, d, pf});
    }
    if (ip) return f(Sq8Space<true, DynamicSq8>{&vectors, {kernels.dot_sq8}, &sq, d, pf});
    return f(Sq8Space<false, DynamicSq8>{&vectors, {kernels.l2sqr_sq8}, &sq, d, pf});
  }

  unsigned random_level(NodeId slot) const noexcept {
    const std::uint64_t h = internal::splitmix64(params.seed ^ internal::splitmix64(slot));
    const double u = (static_cast<double>(h >> 11) + 0.5) * 0x1.0p-53;  // uniform in (0, 1)
    const double level = -std::log(u) * level_mult;
    return level >= kMaxLevel ? kMaxLevel : static_cast<unsigned>(level);
  }

  /// Applies the metric's preprocessing (cosine -> unit norm) into `buf` if needed.
  const float* prepare(const float* v, std::vector<float>& buf) const {
    if (params.metric != Metric::kCosine) return v;
    buf.assign(v, v + params.dim);
    simd::normalize(buf.data(), params.dim);
    return buf.data();
  }

  // ---- graph primitives ----

  /// ef=1 descent on one layer: walks to the closest neighbor until no neighbor is closer.
  template <class Space>
  Candidate greedy(const Space& sp, const float* q, Candidate cur, unsigned level) const {
    const std::size_t cap = level == 0 ? M0 : M;
    for (bool improved = true; improved;) {
      improved = false;
      const std::uint32_t* block = links(cur.id, level);
      const std::size_t cnt = std::min<std::size_t>(load_acquire(block[0]), cap);
      for (std::size_t j = 0; j < cnt; ++j) {
        const NodeId n = load_acquire(block[1 + j]);
        const float d = sp(q, n);
        if (d < cur.dist) {
          cur = {d, n};
          improved = true;
        }
      }
    }
    return cur;
  }

  /// Beam search on one layer (Algorithm 2 of the HNSW paper). Leaves the best `ef` admitted
  /// nodes in s.results as a max-heap. With kCheck, nodes failing `accept` still route the
  /// search but are kept out of the results, and the search keeps going until it has `ef`.
  template <bool kCheck, class Space, class Accept>
  void search_layer(const Space& sp, const float* q, Candidate ep, std::size_t ef, unsigned level,
                    const Accept& accept, Scratch& s) const {
    auto& frontier = s.frontier;
    auto& results = s.results;
    frontier.clear();
    results.clear();
    s.visited.begin(next_slot.load(std::memory_order_relaxed));
    if (s.ids.size() < M0) s.ids.resize(M0);

    s.visited.insert(ep.id);
    frontier.push_back(ep);
    if (!kCheck || accept(ep.id)) results.push_back(ep);
    float bound = results.empty() ? kInf : ep.dist;
    const std::size_t cap = level == 0 ? M0 : M;

    while (!frontier.empty()) {
      const Candidate c = frontier.front();
      if (c.dist > bound && (!kCheck || results.size() >= ef)) break;
      internal::heap_pop<NearestOnTop>(frontier);
      // The next node to expand is (usually) the new frontier top: start pulling its adjacency
      // list now so that miss overlaps with this node's distance computations.
      if (!frontier.empty()) __builtin_prefetch(links(frontier.front().id, level), 0, 3);

      // Pass 1: collect unvisited neighbors and issue their prefetches so the memory system
      // fetches them in parallel. Pass 2: compute distances.
      const std::uint32_t* block = links(c.id, level);
      const std::size_t cnt = std::min<std::size_t>(load_acquire(block[0]), cap);
      std::size_t fresh = 0;
      for (std::size_t j = 0; j < cnt; ++j) {
        const NodeId n = load_acquire(block[1 + j]);
        if (s.visited.insert(n)) {
          sp.prefetch(n);
          s.ids[fresh++] = n;
        }
      }
      for (std::size_t j = 0; j < fresh; ++j) {
        const NodeId n = s.ids[j];
        const float d = sp(q, n);
        if (results.size() < ef || d < bound) {
          internal::heap_push<NearestOnTop>(frontier, {d, n});
          if (!kCheck || accept(n)) {
            internal::heap_push<FarthestOnTop>(results, {d, n});
            if (results.size() > ef) internal::heap_pop<FarthestOnTop>(results);
          }
          if (!results.empty()) bound = results.front().dist;
        }
      }
    }
  }

  /// Neighbor selection heuristic (Algorithm 4): walking candidates from nearest to farthest,
  /// keep c only if it is closer to the base than to every neighbor kept so far. This favors
  /// edges in diverse directions, which keeps clustered data navigable.
  template <class Space>
  void select_neighbors(const Space& sp, const std::vector<Candidate>& sorted, std::size_t max_n,
                        std::vector<Candidate>& out, std::vector<float>& buf) const {
    out.clear();
    if (sorted.size() <= max_n) {
      out = sorted;
      return;
    }
    for (const Candidate& c : sorted) {
      if (out.size() >= max_n) break;
      const float* cq = sp.as_query(c.id, buf);
      bool keep = true;
      for (const Candidate& r : out) {
        if (sp(cq, r.id) < c.dist) {
          keep = false;
          break;
        }
      }
      if (keep) out.push_back(c);
    }
  }

  /// Adds directed edges node -> add[i] on `level`, re-pruning with the heuristic when the list
  /// overflows. `node_q` is the node's vector as a query (null: decode it from storage).
  template <class Space>
  void add_edges(const Space& sp, NodeId node, const float* node_q, const Candidate* add,
                 std::size_t n_add, unsigned level, Scratch& s) const {
    const std::size_t cap = level == 0 ? M0 : M;
    SpinGuard guard(meta(node).lock);
    std::uint32_t* block = links(node, level);
    std::size_t cnt = std::min<std::size_t>(block[0], cap);

    s.merge.clear();
    for (std::size_t i = 0; i < n_add; ++i) {
      const NodeId target = add[i].id;
      if (target == node) continue;
      bool present = false;
      for (std::size_t j = 0; j < cnt && !present; ++j) present = block[1 + j] == target;
      if (present) continue;
      if (cnt < cap) {
        store_release(block[1 + cnt], target);
        ++cnt;
      } else {
        s.merge.push_back(add[i]);
      }
    }
    if (!s.merge.empty()) {
      // Overflow: choose the best `cap` among existing + new edges.
      const float* nq = node_q != nullptr ? node_q : sp.as_query(node, s.decoded_a);
      for (std::size_t j = 0; j < cnt; ++j) {
        const NodeId e = block[1 + j];
        s.merge.push_back({sp(nq, e), e});
      }
      std::sort(s.merge.begin(), s.merge.end(),
                [](const Candidate& a, const Candidate& b) { return a.dist < b.dist; });
      select_neighbors(sp, s.merge, cap, s.pruned, s.decoded_b);
      for (std::size_t j = 0; j < s.pruned.size(); ++j) store_release(block[1 + j], s.pruned[j].id);
      cnt = s.pruned.size();
    }
    store_release(block[0], static_cast<std::uint32_t>(cnt));
  }

  /// Wires an initialized node into the graph.
  template <class Space>
  void link(const Space& sp, NodeId id, const float* v, Scratch& s) {
    const unsigned level = meta(id).level;

    std::unique_lock top_lock(entry_mu, std::defer_lock);
    std::uint64_t e = entry.load(std::memory_order_acquire);
    if (e == 0 || level > entry_level(e)) {
      top_lock.lock();
      e = entry.load(std::memory_order_acquire);
      if (e == 0) {
        entry.store(pack_entry(id, level), std::memory_order_release);
        return;
      }
      if (level <= entry_level(e)) top_lock.unlock();  // someone else raised the top meanwhile
    }

    const unsigned top = entry_level(e);
    Candidate cur{sp(v, entry_id(e)), entry_id(e)};
    for (unsigned lc = top; lc > level; --lc) cur = greedy(sp, v, cur, lc);

    // Phase 1, top-down: find and write this node's own neighbors on every layer. The node is
    // not reachable yet, so concurrent searches cannot land on a half-wired node.
    const unsigned link_top = std::min(level, top);
    if (s.chosen.size() <= link_top) s.chosen.resize(link_top + 1);
    auto no_filter = [](NodeId) { return true; };
    for (int lc = static_cast<int>(link_top); lc >= 0; --lc) {
      const auto ul = static_cast<unsigned>(lc);
      search_layer<false>(sp, v, cur, params.ef_construction, ul, no_filter, s);

      s.candidates.clear();
      for (const Candidate& c : s.results) {
        if (c.id != id) s.candidates.push_back(c);
      }
      std::sort(s.candidates.begin(), s.candidates.end(),
                [](const Candidate& a, const Candidate& b) { return a.dist < b.dist; });
      std::vector<Candidate>& chosen = s.chosen[ul];
      if (s.candidates.empty()) {
        chosen.clear();
        continue;
      }
      select_neighbors(sp, s.candidates, M, chosen, s.decoded_b);
      add_edges(sp, id, v, chosen.data(), chosen.size(), ul, s);
      cur = s.candidates.front();
    }

    // Phase 2: publish the reverse edges, which is what makes the node reachable.
    for (unsigned lc = 0; lc <= link_top; ++lc) {
      for (const Candidate& c : s.chosen[lc]) {
        const Candidate back{c.dist, id};
        add_edges(sp, c.id, nullptr, &back, 1, lc, s);
      }
    }

    if (top_lock.owns_lock()) entry.store(pack_entry(id, level), std::memory_order_release);
  }

  // ---- operations ----

  void train_locked(const float* vectors_in, std::size_t n) {
    if (params.metric == Metric::kCosine) {
      std::vector<float> normed(vectors_in, vectors_in + n * params.dim);
      for (std::size_t i = 0; i < n; ++i)
        simd::normalize(normed.data() + i * params.dim, params.dim);
      sq.train(normed.data(), n);
    } else {
      sq.train(vectors_in, n);
    }
    trained.store(true, std::memory_order_release);
  }

  void init_node(NodeId id, Label label, const float* v) {
    NodeMeta& m = meta(id);
    const unsigned level = random_level(id);
    m.label = label;
    m.level = static_cast<std::uint8_t>(level);
    m.upper = level > 0 ? arena.allocate(level * (1 + M)) : nullptr;
    links0.as<std::uint32_t>(id)[0] = 0;
    if (params.storage == Storage::kFloat32) {
      std::memcpy(vectors.at(id), v, code_size);
    } else {
      sq.encode(v, vectors.as<std::uint8_t>(id));
    }
    // Count the node before it becomes removable, so a racing remove() can never decrement
    // num_live below the number of nodes it has seen.
    num_live.fetch_add(1, std::memory_order_relaxed);
    std::atomic_ref<std::uint8_t>(m.flags).store(kReady, std::memory_order_release);
  }

  /// Largest number of slots every per-node storage can hold.
  std::size_t max_slots() const noexcept {
    return std::min({vectors.max_nodes(), links0.max_nodes(), metas.max_nodes(),
                     static_cast<std::size_t>(std::numeric_limits<NodeId>::max())});
  }

  /// Validates labels and claims slots [base, base + n) under the label lock.
  ///
  /// Storage for the slots is reserved *before* any of them becomes visible (through the label
  /// map or next_slot), so every id another thread can obtain is backed by memory. Fresh slots
  /// read as flags == 0 ("not ready") until init_node() publishes them.
  NodeId claim_slots(const Label* labels, std::size_t n) {
    std::unique_lock lock(labels_mu);
    const NodeId base = next_slot.load(std::memory_order_relaxed);
    QUIVER_CHECK(static_cast<std::size_t>(base) + n <= max_slots(),
                 "index is full (maximum " + std::to_string(max_slots()) + " vectors)");
    auto label_of = [&](std::size_t i) { return labels ? labels[i] : Label{base} + i; };
    // Validate everything before mutating, so a rejected batch leaves no trace.
    for (std::size_t i = 0; i < n; ++i) {
      const Label l = label_of(i);
      const auto it = label_to_id.find(l);
      // A mapping to a removed node may be replaced; one to a live or in-flight node may not.
      QUIVER_CHECK(it == label_to_id.end() ||
                       (load_flags(meta(it->second), std::memory_order_acquire) & kRemoved) != 0,
                   "label " + std::to_string(l) + " is already in the index");
    }
    const std::size_t end = static_cast<std::size_t>(base) + n;
    vectors.reserve(end);  // may throw; nothing is published yet
    links0.reserve(end);
    metas.reserve(end);

    // Publish the mappings, remembering what each replaced, so a failure (allocation inside the
    // hash map) can be undone without allocating: restore replaced values, erase new keys.
    constexpr NodeId kNone = std::numeric_limits<NodeId>::max();
    std::vector<NodeId> previous(n, kNone);
    std::size_t done = 0;
    try {
      for (; done < n; ++done) {
        const auto [it, fresh] =
            label_to_id.try_emplace(label_of(done), base + static_cast<NodeId>(done));
        if (!fresh) {
          previous[done] = it->second;
          it->second = base + static_cast<NodeId>(done);
        }
      }
    } catch (...) {
      for (std::size_t i = 0; i < done; ++i) {
        if (previous[i] == kNone) {
          label_to_id.erase(label_of(i));
        } else if (const auto it = label_to_id.find(label_of(i)); it != label_to_id.end()) {
          it->second = previous[i];  // always found: the key existed before and is still there
        }
      }
      throw;
    }
    next_slot.store(base + static_cast<NodeId>(n), std::memory_order_release);
    return base;
  }

  /// After a failed insert: forget labels whose slots never became ready. Their slots stay as
  /// permanent holes (flags == 0), which every reader already skips.
  void release_unready(NodeId base, std::size_t n) noexcept {
    std::unique_lock lock(labels_mu);
    for (auto it = label_to_id.begin(); it != label_to_id.end();) {
      const NodeId id = it->second;
      const bool ours = id >= base && id < base + n;
      if (ours && load_flags(meta(id), std::memory_order_acquire) == 0) {
        it = label_to_id.erase(it);
      } else {
        ++it;
      }
    }
  }

  void add(const float* in, std::size_t n, const Label* labels, ThreadPool* pool) {
    if (n == 0) return;
    QUIVER_CHECK(in != nullptr, "vectors must not be null");
    check_finite(in, n * params.dim);
    if (labels != nullptr) {
      QUIVER_CHECK(std::find(labels, labels + n, kInvalidLabel) == labels + n,
                   "label 2^64-1 is reserved (it marks empty result slots)");
    }
    if (labels != nullptr && n > 1) {
      std::vector<Label> sorted(labels, labels + n);
      std::sort(sorted.begin(), sorted.end());
      QUIVER_CHECK(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(),
                   "duplicate labels within one add() call");
    }
    if (params.storage == Storage::kSQ8 && !trained.load(std::memory_order_acquire)) {
      std::lock_guard lock(train_mu);
      if (!trained.load(std::memory_order_relaxed)) {
        QUIVER_CHECK(n >= kMinAutoTrain,
                     "SQ8 index is untrained: call train() with a representative sample, or "
                     "make the first add() at least " +
                         std::to_string(kMinAutoTrain) + " vectors");
        train_locked(in, n);
      }
    }

    std::shared_lock writer(writers_mu);  // excludes save(); concurrent adds are fine
    const NodeId base = claim_slots(labels, n);
    try {
      auto insert_range = [&](std::size_t begin, std::size_t stop) {
        ScratchLease lease;
        Scratch& s = *lease;
        with_space([&](const auto& sp) {
          for (std::size_t i = begin; i < stop; ++i) {
            const NodeId id = base + static_cast<NodeId>(i);
            const float* v = prepare(in + i * params.dim, s.query);
            init_node(id, labels ? labels[i] : Label{base} + i, v);
            link(sp, id, v, s);
          }
        });
      };
      if (n == 1) {
        insert_range(0, 1);
      } else {
        ThreadPool& tp = pool ? *pool : ThreadPool::global();
        tp.parallel_for(n, 1, insert_range);
      }
    } catch (...) {
      release_unready(base, n);
      throw;
    }
  }

  template <class Space>
  void search_graph(const Space& sp, const float* q, std::size_t k, const HnswSearchParams& p,
                    Scratch& s) const {
    const std::uint64_t e = entry.load(std::memory_order_acquire);
    if (e == 0) {
      s.results.clear();
      return;
    }
    Candidate cur{sp(q, entry_id(e)), entry_id(e)};
    for (unsigned lc = entry_level(e); lc > 0; --lc) cur = greedy(sp, q, cur, lc);

    const std::size_t ef = std::max(p.ef, k);
    if (p.filter != nullptr || num_removed.load(std::memory_order_relaxed) > 0) {
      const Filter* filter = p.filter;
      auto accept = [&](NodeId id) {
        const NodeMeta& m = meta(id);
        if ((load_flags(m, std::memory_order_relaxed) & kRemoved) != 0) return false;
        return filter == nullptr || filter->contains(m.label);
      };
      search_layer<true>(sp, q, cur, ef, 0, accept, s);
    } else {
      search_layer<false>(sp, q, cur, ef, 0, [](NodeId) { return true; }, s);
    }
  }

  /// Exact scan over the labels a selective filter admits. Returns false if unsupported.
  template <class Space>
  bool search_exact(const Space& sp, const float* q, std::size_t k, const Filter& filter,
                    Scratch& s) const {
    auto& heap = s.results;
    heap.clear();
    std::shared_lock lock(labels_mu);
    return filter.for_each([&](Label l) {
      const auto it = label_to_id.find(l);
      if (it == label_to_id.end() || !is_live(it->second)) return;
      const float d = sp(q, it->second);
      if (heap.size() < k) {
        internal::heap_push<FarthestOnTop>(heap, {d, it->second});
      } else if (d < heap.front().dist) {
        internal::heap_pop<FarthestOnTop>(heap);
        internal::heap_push<FarthestOnTop>(heap, {d, it->second});
      }
    });
  }

  void search_into(const float* query, std::size_t k, const HnswSearchParams& p, Label* out_labels,
                   float* out_dist) const {
    std::fill(out_labels, out_labels + k, kInvalidLabel);
    std::fill(out_dist, out_dist + k, kInf);
    if (k == 0) return;
    QUIVER_CHECK(query != nullptr, "query must not be null");
    check_finite(query, params.dim);
    ScratchLease lease;
    Scratch& s = *lease;
    const float* q = prepare(query, s.query);

    with_space([&](const auto& sp) {
      bool done = false;
      if (p.filter != nullptr && p.exact_scan_threshold > 0.0) {
        const auto card = p.filter->cardinality();
        const double live = static_cast<double>(num_live.load(std::memory_order_relaxed));
        if (card && static_cast<double>(*card) <= p.exact_scan_threshold * live) {
          done = search_exact(sp, q, k, *p.filter, s);
        }
      }
      if (!done) search_graph(sp, q, k, p, s);
    });

    // results is a max-heap; sort_heap yields ascending distance.
    std::sort_heap(s.results.begin(), s.results.end(), FarthestOnTop{});
    const std::size_t n = std::min(k, s.results.size());
    for (std::size_t j = 0; j < n; ++j) {
      out_labels[j] = meta(s.results[j].id).label;
      out_dist[j] = s.results[j].dist;
    }
  }

  NodeId find_live(Label label) const {
    std::shared_lock lock(labels_mu);
    const auto it = label_to_id.find(label);
    if (it == label_to_id.end() || !is_live(it->second)) return std::numeric_limits<NodeId>::max();
    return it->second;
  }

  void save(const std::string& path) const;
  static std::unique_ptr<Impl> load(const std::string& path, const LoadOptions& options);
};

HnswIndex::Impl::Impl(const HnswParams& p)
    : params(p),
      M(p.M),
      M0(2 * p.M),
      level_mult(1.0 / std::log(static_cast<double>(std::max<std::size_t>(p.M, 2)))),
      code_size(p.storage == Storage::kFloat32 ? p.dim * sizeof(float) : p.dim),
      vectors(std::max<std::size_t>(code_size, 1)),
      links0((1 + 2 * p.M) * sizeof(std::uint32_t)),
      metas(sizeof(NodeMeta)),
      sq(p.dim) {
  QUIVER_CHECK(p.dim >= 1 && p.dim <= kMaxDim,
               "dimension must be in [1, " + std::to_string(kMaxDim) + "]");
  QUIVER_CHECK(p.M >= 2 && p.M <= kMaxM, "M must be in [2, " + std::to_string(kMaxM) + "]");
  QUIVER_CHECK(p.ef_construction >= 1, "ef_construction must be positive");
  QUIVER_CHECK(
      p.metric == Metric::kL2 || p.metric == Metric::kInnerProduct || p.metric == Metric::kCosine,
      "unknown metric");
  QUIVER_CHECK(p.storage == Storage::kFloat32 || p.storage == Storage::kSQ8, "unknown storage");
  if (p.storage == Storage::kFloat32) trained.store(true, std::memory_order_relaxed);
}

// ---- persistence -----------------------------------------------------------------------------

void HnswIndex::Impl::save(const std::string& path) const {
  std::unique_lock writer(writers_mu);
  const std::size_t n = next_slot.load(std::memory_order_acquire);

  // Upper layers are scattered across the arena; compact them into one array.
  std::vector<std::uint64_t> upper_offsets(n, kNoUpper);
  std::uint64_t upper_words = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const NodeMeta& m = meta(static_cast<NodeId>(i));
    if (m.level > 0 && load_flags(m, std::memory_order_acquire) != 0) {
      upper_offsets[i] = upper_words;
      upper_words += m.level * (1 + M);
    }
  }

  const std::uint64_t e = entry.load(std::memory_order_acquire);
  HnswMeta hm{};
  hm.dim = params.dim;
  hm.M = M;
  hm.ef_construction = params.ef_construction;
  hm.seed = params.seed;
  hm.num_slots = n;
  hm.upper_words = upper_words;
  hm.metric = static_cast<std::uint32_t>(params.metric);
  hm.storage = static_cast<std::uint32_t>(params.storage);
  hm.entry_node = e == 0 ? 0 : entry_id(e);
  hm.entry_level_plus1 = e == 0 ? 0 : entry_level(e) + 1;

  internal::FileWriter w(internal::IndexKind::kHnsw);
  w.add_pod(kMetaTag, hm);

  // Chunked arrays are written chunk by chunk so the file is one contiguous array.
  auto add_chunked = [&](std::uint32_t tag, const ChunkedStorage& st) {
    w.add(tag, n * st.stride(), [&st, n](internal::SectionSink& sink) {
      for (std::size_t first = 0; first < n; first += st.chunk_nodes()) {
        const std::size_t count = std::min(st.chunk_nodes(), n - first);
        sink.write(st.at(static_cast<NodeId>(first)), count * st.stride());
      }
    });
  };
  add_chunked(kVectorsTag, vectors);
  add_chunked(kLinks0Tag, links0);

  w.add(kLabelsTag, n * sizeof(Label), [this, n](internal::SectionSink& sink) {
    for (std::size_t i = 0; i < n; ++i)
      sink.write(&meta(static_cast<NodeId>(i)).label, sizeof(Label));
  });
  w.add(kLevelsTag, n, [this, n](internal::SectionSink& sink) {
    for (std::size_t i = 0; i < n; ++i) sink.write(&meta(static_cast<NodeId>(i)).level, 1);
  });
  w.add(kFlagsTag, n, [this, n](internal::SectionSink& sink) {
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint8_t f = load_flags(meta(static_cast<NodeId>(i)), std::memory_order_acquire);
      sink.write(&f, 1);
    }
  });
  w.add_bytes(kUpperOffsetsTag, upper_offsets.data(), n * sizeof(std::uint64_t));
  w.add(kUpperLinksTag, upper_words * sizeof(std::uint32_t),
        [this, n, &upper_offsets](internal::SectionSink& sink) {
          for (std::size_t i = 0; i < n; ++i) {
            if (upper_offsets[i] == kNoUpper) continue;
            const NodeMeta& m = meta(static_cast<NodeId>(i));
            sink.write(m.upper, m.level * (1 + M) * sizeof(std::uint32_t));
          }
        });
  std::vector<float> sq_params;
  if (params.storage == Storage::kSQ8) {
    QUIVER_CHECK(trained.load(std::memory_order_acquire) || n == 0, "SQ8 index is not trained");
    if (trained.load(std::memory_order_acquire)) {
      sq_params.assign(sq.vmin(), sq.vmin() + params.dim);
      sq_params.insert(sq_params.end(), sq.vscale(), sq.vscale() + params.dim);
    }
    w.add_bytes(kSq8Tag, sq_params.data(), sq_params.size() * sizeof(float));
  }
  w.commit(path);
}

std::unique_ptr<HnswIndex::Impl> HnswIndex::Impl::load(const std::string& path,
                                                       const LoadOptions& options) {
  auto image = internal::FileImage::open(path, internal::IndexKind::kHnsw,
                                         {options.mmap, options.verify_checksums});
  const std::string corrupt = "corrupt index file '" + path + "': ";
  const auto hm = image->pod<HnswMeta>(kMetaTag);
  QUIVER_CHECK(hm.dim >= 1 && hm.dim <= kMaxDim, corrupt + "bad dimension");
  QUIVER_CHECK(hm.M >= 2 && hm.M <= kMaxM, corrupt + "bad M");
  QUIVER_CHECK(hm.ef_construction >= 1, corrupt + "bad ef_construction");
  QUIVER_CHECK(hm.metric <= static_cast<std::uint32_t>(Metric::kCosine), corrupt + "bad metric");
  QUIVER_CHECK(hm.storage <= static_cast<std::uint32_t>(Storage::kSQ8), corrupt + "bad storage");

  HnswParams p;
  p.dim = static_cast<std::size_t>(hm.dim);
  p.M = static_cast<std::size_t>(hm.M);
  p.ef_construction = static_cast<std::size_t>(hm.ef_construction);
  p.seed = hm.seed;
  p.metric = static_cast<Metric>(hm.metric);
  p.storage = static_cast<Storage>(hm.storage);
  auto impl = std::make_unique<Impl>(p);
  Impl& ix = *impl;

  // Every count is bounded by the file size before it is used in any size computation.
  QUIVER_CHECK(hm.num_slots <= ix.vectors.max_nodes() && hm.num_slots <= image->size(),
               corrupt + "bad vector count");
  const auto n = static_cast<std::size_t>(hm.num_slots);
  QUIVER_CHECK(hm.upper_words <= image->size() / sizeof(std::uint32_t), corrupt + "bad upper size");
  const auto upper_words = static_cast<std::size_t>(hm.upper_words);

  const auto vec = image->array<std::byte>(kVectorsTag, n * ix.code_size);
  const auto lnk0 = image->array<std::uint32_t>(kLinks0Tag, n * (1 + ix.M0));
  const auto labels = image->array<Label>(kLabelsTag, n);
  const auto levels = image->array<std::uint8_t>(kLevelsTag, n);
  const auto flags = image->array<std::uint8_t>(kFlagsTag, n);
  const auto up_off = image->array<std::uint64_t>(kUpperOffsetsTag, n);
  const auto up = image->array<std::uint32_t>(kUpperLinksTag, upper_words);

  if (p.storage == Storage::kSQ8) {
    const auto sqp = image->bytes(kSq8Tag);
    if (!sqp.empty()) {
      QUIVER_CHECK(sqp.size() == 2 * p.dim * sizeof(float), corrupt + "bad SQ8 parameters");
      std::vector<float> vmin(p.dim), vscale(p.dim);
      std::memcpy(vmin.data(), sqp.data(), p.dim * sizeof(float));
      std::memcpy(vscale.data(), sqp.data() + p.dim * sizeof(float), p.dim * sizeof(float));
      ix.sq.set_params(std::move(vmin), std::move(vscale));
      ix.trained.store(true, std::memory_order_relaxed);
    } else {
      QUIVER_CHECK(n == 0, corrupt + "SQ8 index without quantizer parameters");
    }
  }

  // Structural validation. A hostile file must not be able to make a later search read out of
  // bounds, so every edge, level and offset is checked here, once.
  // Upper-layer blocks must be laid out exactly as save() writes them: consecutive, in node
  // order. Besides bounds, this guarantees no two blocks alias, so a later insert writing one
  // node's list can never corrupt another's (all edges were validated against that layout).
  std::uint64_t expected_off = 0;
  for (std::size_t i = 0; i < n; ++i) {
    QUIVER_CHECK((flags[i] & ~(kReady | kRemoved)) == 0, corrupt + "bad node flags");
    QUIVER_CHECK(flags[i] != kRemoved, corrupt + "bad node flags");
    QUIVER_CHECK(levels[i] <= kMaxLevel, corrupt + "bad node level");
    if ((flags[i] & kReady) != 0 && levels[i] > 0) {
      QUIVER_CHECK(up_off[i] == expected_off, corrupt + "bad upper-layer offsets");
      expected_off += std::uint64_t{levels[i]} * (1 + ix.M);
      QUIVER_CHECK(expected_off <= upper_words, corrupt + "upper-layer offset out of bounds");
    } else {
      QUIVER_CHECK(up_off[i] == kNoUpper, corrupt + "bad upper-layer offsets");
    }
  }
  QUIVER_CHECK(expected_off == upper_words, corrupt + "bad upper-layer size");
  // An edge (u -> v) on layer l requires v to be a ready node that exists on layer l.
  auto check_block = [&](const std::uint32_t* block, std::size_t cap, std::size_t level) {
    QUIVER_CHECK(block[0] <= cap, corrupt + "adjacency list too long");
    for (std::size_t j = 0; j < block[0]; ++j) {
      const std::uint32_t v = block[1 + j];
      QUIVER_CHECK(v < n && (flags[v] & kReady) != 0 && levels[v] >= level,
                   corrupt + "edge to an invalid node");
    }
  };
  for (std::size_t i = 0; i < n; ++i) {
    if ((flags[i] & kReady) == 0) continue;
    check_block(lnk0.data() + i * (1 + ix.M0), ix.M0, 0);
    for (std::size_t l = 1; l <= levels[i]; ++l) {
      check_block(up.data() + up_off[i] + (l - 1) * (1 + ix.M), ix.M, l);
    }
  }
  if (hm.entry_level_plus1 != 0) {
    const std::uint32_t ep = hm.entry_node;
    QUIVER_CHECK(ep < n && (flags[ep] & kReady) != 0 && levels[ep] + 1u == hm.entry_level_plus1,
                 corrupt + "bad entry point");
    ix.entry.store(pack_entry(ep, levels[ep]), std::memory_order_relaxed);
  } else {
    for (std::size_t i = 0; i < n; ++i) {
      QUIVER_CHECK((flags[i] & kReady) == 0, corrupt + "nodes present but no entry point");
    }
  }

  // Adopt the large arrays in place; rebuild per-node metadata in owned memory.
  ix.vectors.adopt(vec.data(), n);
  ix.links0.adopt(reinterpret_cast<std::byte*>(lnk0.data()), n);
  ix.metas.reserve(n);
  std::size_t live = 0, removed = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const auto id = static_cast<NodeId>(i);
    NodeMeta& m = ix.meta(id);
    m.label = labels[i];
    m.level = levels[i];
    m.flags = flags[i];
    m.upper = (flags[i] & kReady) != 0 && levels[i] > 0 ? up.data() + up_off[i] : nullptr;
    if (flags[i] == kReady) {
      ++live;
      const bool fresh = ix.label_to_id.emplace(labels[i], id).second;
      QUIVER_CHECK(fresh, corrupt + "duplicate label " + std::to_string(labels[i]));
    } else if (flags[i] == (kReady | kRemoved)) {
      ++removed;
    }
  }
  ix.next_slot.store(static_cast<NodeId>(n), std::memory_order_relaxed);
  ix.num_live.store(live, std::memory_order_relaxed);
  ix.num_removed.store(removed, std::memory_order_relaxed);
  ix.image = std::move(image);
  return impl;
}

// =============================================================================================
// Public API

HnswIndex::HnswIndex(const HnswParams& params) : impl_(std::make_unique<Impl>(params)) {}
HnswIndex::HnswIndex(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
HnswIndex::~HnswIndex() = default;
HnswIndex::HnswIndex(HnswIndex&&) noexcept = default;
HnswIndex& HnswIndex::operator=(HnswIndex&&) noexcept = default;

void HnswIndex::train(const float* vectors, std::size_t n) {
  QUIVER_CHECK(impl_->params.storage == Storage::kSQ8, "train() only applies to SQ8 storage");
  QUIVER_CHECK(vectors != nullptr && n > 0, "train() needs at least one vector");
  check_finite(vectors, n * impl_->params.dim);
  std::lock_guard lock(impl_->train_mu);
  // Parameters are read without locks once trained, so they may only be set once.
  QUIVER_CHECK(!impl_->trained.load(std::memory_order_acquire),
               "index is already trained (SQ8 parameters cannot change once set)");
  impl_->train_locked(vectors, n);
}

bool HnswIndex::is_trained() const noexcept {
  return impl_->trained.load(std::memory_order_acquire);
}

void HnswIndex::add(const float* vectors, std::size_t n, const Label* labels, ThreadPool* pool) {
  impl_->add(vectors, n, labels, pool);
}

bool HnswIndex::remove(Label label) {
  std::shared_lock lock(impl_->labels_mu);
  const auto it = impl_->label_to_id.find(label);
  if (it == impl_->label_to_id.end()) return false;
  std::atomic_ref<std::uint8_t> f(impl_->meta(it->second).flags);
  std::uint8_t cur = f.load(std::memory_order_acquire);
  while (cur == kReady) {
    if (f.compare_exchange_weak(cur, kReady | kRemoved, std::memory_order_acq_rel)) {
      impl_->num_live.fetch_sub(1, std::memory_order_relaxed);
      impl_->num_removed.fetch_add(1, std::memory_order_relaxed);
      return true;
    }
  }
  return false;
}

bool HnswIndex::contains(Label label) const {
  return impl_->find_live(label) != std::numeric_limits<NodeId>::max();
}

std::vector<float> HnswIndex::get_vector(Label label) const {
  const NodeId id = impl_->find_live(label);
  QUIVER_CHECK(id != std::numeric_limits<NodeId>::max(),
               "label " + std::to_string(label) + " not found");
  std::vector<float> out(impl_->params.dim);
  if (impl_->params.storage == Storage::kFloat32) {
    std::memcpy(out.data(), impl_->vectors.at(id), impl_->code_size);
  } else {
    impl_->sq.decode(impl_->vectors.as<std::uint8_t>(id), out.data());
  }
  return out;
}

std::vector<Neighbor> HnswIndex::search(const float* query, std::size_t k,
                                        const HnswSearchParams& params) const {
  std::vector<Label> labels(k);
  std::vector<float> dist(k);
  impl_->search_into(query, k, params, labels.data(), dist.data());
  std::vector<Neighbor> out;
  out.reserve(k);
  for (std::size_t j = 0; j < k && labels[j] != kInvalidLabel; ++j)
    out.push_back({dist[j], labels[j]});
  return out;
}

void HnswIndex::search_batch(const float* queries, std::size_t nq, std::size_t k, Label* labels,
                             float* distances, const HnswSearchParams& params,
                             ThreadPool* pool) const {
  if (nq == 0 || k == 0) return;
  QUIVER_CHECK(queries != nullptr && labels != nullptr && distances != nullptr,
               "search_batch: null buffer");
  // Pool chunks below take no index-wide lock that a thread waiting on this pool could hold,
  // which is what keeps search_batch deadlock-free alongside add() and save().
  ThreadPool& tp = pool ? *pool : ThreadPool::global();
  const std::size_t grain = std::max<std::size_t>(1, nq / (tp.num_threads() * 16));
  const std::size_t dim = impl_->params.dim;
  tp.parallel_for(nq, grain, [&](std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      impl_->search_into(queries + i * dim, k, params, labels + i * k, distances + i * k);
    }
  });
}

void HnswIndex::reserve(std::size_t n) {
  impl_->vectors.reserve(n);
  impl_->links0.reserve(n);
  impl_->metas.reserve(n);
}

void HnswIndex::save(const std::string& path) const { impl_->save(path); }

HnswIndex HnswIndex::load(const std::string& path, const LoadOptions& options) {
  return HnswIndex(Impl::load(path, options));
}

std::size_t HnswIndex::size() const noexcept {
  return impl_->num_live.load(std::memory_order_relaxed);
}
std::size_t HnswIndex::dim() const noexcept { return impl_->params.dim; }
Metric HnswIndex::metric() const noexcept { return impl_->params.metric; }
Storage HnswIndex::storage() const noexcept { return impl_->params.storage; }
const HnswParams& HnswIndex::params() const noexcept { return impl_->params; }

HnswIndex::Stats HnswIndex::stats() const {
  const Impl& ix = *impl_;
  Stats st;
  const std::size_t n = ix.next_slot.load(std::memory_order_acquire);
  std::size_t edges0 = 0, ready = 0, upper_words = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const auto id = static_cast<NodeId>(i);
    const NodeMeta& m = ix.meta(id);
    const std::uint8_t f = load_flags(m, std::memory_order_acquire);
    if ((f & kReady) == 0) continue;
    ++ready;
    if ((f & kRemoved) != 0) ++st.num_removed;
    if (st.nodes_per_level.size() <= m.level) st.nodes_per_level.resize(m.level + 1u, 0);
    for (std::size_t l = 0; l <= m.level; ++l) ++st.nodes_per_level[l];
    edges0 += load_acquire(ix.links0.as<std::uint32_t>(id)[0]);
    upper_words += m.level * (1 + ix.M);
  }
  st.num_nodes = ready;
  st.max_level = static_cast<int>(st.nodes_per_level.size()) - 1;
  st.mean_degree_level0 = ready ? static_cast<double>(edges0) / static_cast<double>(ready) : 0.0;
  st.vector_bytes = ready * ix.code_size;
  st.graph_bytes = ready * ix.links0.stride() + upper_words * sizeof(std::uint32_t);
  st.memory_mapped = ix.image != nullptr && ix.image->mapped();
  return st;
}

}  // namespace quiver

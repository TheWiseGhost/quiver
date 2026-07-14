// End-to-end ANN benchmark: build an index, then sweep the search-effort knob and report
// recall@k against exact ground truth together with throughput and tail latency.
//
//   ann_bench --base sift_base.fvecs --query sift_query.fvecs --gt sift_groundtruth.ivecs
//             --index hnsw --M 16 --efc 200 --ef 10,20,40,80,160 --threads 10 --json out.jsonl
//
//   ann_bench --synthetic 100000,128,1000 --index hnsw     # no files needed
#include <sys/resource.h>
#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "dataset.h"
#include "quiver/distance.h"
#include "quiver/flat_index.h"
#include "quiver/hnsw_index.h"
#include "quiver/ivfpq_index.h"
#include "quiver/thread_pool.h"

namespace {

using Clock = std::chrono::steady_clock;
using quiver::bench::IntMatrix;
using quiver::bench::Matrix;

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

struct Args {
  std::map<std::string, std::string> kv;

  std::string get(const std::string& key, const std::string& fallback = "") const {
    const auto it = kv.find(key);
    return it == kv.end() ? fallback : it->second;
  }
  std::size_t num(const std::string& key, std::size_t fallback) const {
    const auto v = get(key);
    return v.empty() ? fallback : static_cast<std::size_t>(std::stoull(v));
  }
  bool has(const std::string& key) const { return kv.count(key) != 0; }
};

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key.rfind("--", 0) != 0) throw std::runtime_error("unexpected argument " + key);
    key = key.substr(2);
    if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
      a.kv[key] = argv[++i];
    } else {
      a.kv[key] = "1";
    }
  }
  return a;
}

std::vector<std::size_t> parse_list(const std::string& s) {
  std::vector<std::size_t> out;
  std::stringstream ss(s);
  for (std::string item; std::getline(ss, item, ',');) out.push_back(std::stoull(item));
  return out;
}

std::size_t peak_rss_bytes() {
  rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
  return static_cast<std::size_t>(ru.ru_maxrss);  // bytes
#else
  return static_cast<std::size_t>(ru.ru_maxrss) * 1024;  // KiB
#endif
}

Matrix synthetic(std::size_t n, std::size_t dim, std::size_t clusters, std::uint64_t seed,
                 std::uint64_t center_seed) {
  std::mt19937_64 crng(center_seed);
  std::normal_distribution<float> unit(0.0f, 1.0f);
  std::vector<float> centers(clusters * dim);
  for (float& x : centers) x = 2.0f * unit(crng);
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<std::size_t> pick(0, clusters - 1);
  Matrix m;
  m.rows = n;
  m.cols = dim;
  m.data.resize(n * dim);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t c = pick(rng);
    for (std::size_t j = 0; j < dim; ++j) m.data[i * dim + j] = centers[c * dim + j] + unit(rng);
  }
  return m;
}

IntMatrix exact_ground_truth(const Matrix& base, const Matrix& queries, std::size_t k,
                             quiver::Metric metric, quiver::ThreadPool& pool) {
  quiver::FlatIndex flat(base.cols, metric);
  flat.add(base.data.data(), base.rows);
  std::vector<quiver::Label> labels(queries.rows * k);
  std::vector<float> dist(queries.rows * k);
  flat.search_batch(queries.data.data(), queries.rows, k, labels.data(), dist.data(), nullptr,
                    &pool);
  IntMatrix gt;
  gt.rows = queries.rows;
  gt.cols = k;
  gt.data.assign(labels.begin(), labels.end());
  return gt;
}

double recall(const IntMatrix& gt, const std::vector<quiver::Label>& got, std::size_t nq,
              std::size_t k) {
  std::size_t hits = 0;
  for (std::size_t q = 0; q < nq; ++q) {
    std::unordered_set<std::int64_t> truth;
    for (std::size_t j = 0; j < k; ++j) truth.insert(gt.data[q * gt.cols + j]);
    for (std::size_t j = 0; j < k; ++j) {
      hits += truth.count(static_cast<std::int64_t>(got[q * k + j]));
    }
  }
  return static_cast<double>(hits) / static_cast<double>(nq * k);
}

/// Adapter so the sweep logic is index-agnostic.
struct Searcher {
  virtual ~Searcher() = default;
  virtual void batch(const float* q, std::size_t nq, std::size_t k, quiver::Label* labels,
                     float* dist, std::size_t effort, quiver::ThreadPool& pool) const = 0;
  virtual void one(const float* q, std::size_t k, quiver::Label* labels, float* dist,
                   std::size_t effort) const = 0;
};

struct HnswSearcher final : Searcher {
  const quiver::HnswIndex* index;
  explicit HnswSearcher(const quiver::HnswIndex* i) : index(i) {}
  void batch(const float* q, std::size_t nq, std::size_t k, quiver::Label* labels, float* dist,
             std::size_t ef, quiver::ThreadPool& pool) const override {
    quiver::HnswSearchParams p;
    p.ef = ef;
    index->search_batch(q, nq, k, labels, dist, p, &pool);
  }
  void one(const float* q, std::size_t k, quiver::Label* labels, float* dist,
           std::size_t ef) const override {
    static quiver::ThreadPool inline_pool(1);
    quiver::HnswSearchParams p;
    p.ef = ef;
    index->search_batch(q, 1, k, labels, dist, p, &inline_pool);
  }
};

struct FlatSearcher final : Searcher {
  const quiver::FlatIndex* index;
  explicit FlatSearcher(const quiver::FlatIndex* i) : index(i) {}
  void batch(const float* q, std::size_t nq, std::size_t k, quiver::Label* labels, float* dist,
             std::size_t, quiver::ThreadPool& pool) const override {
    index->search_batch(q, nq, k, labels, dist, nullptr, &pool);
  }
  void one(const float* q, std::size_t k, quiver::Label* labels, float* dist,
           std::size_t) const override {
    static quiver::ThreadPool inline_pool(1);
    index->search_batch(q, 1, k, labels, dist, nullptr, &inline_pool);
  }
};

struct IvfPqSearcher final : Searcher {
  const quiver::IvfPqIndex* index;
  std::size_t refine;
  IvfPqSearcher(const quiver::IvfPqIndex* i, std::size_t r) : index(i), refine(r) {}
  quiver::IvfPqSearchParams params(std::size_t nprobe) const {
    quiver::IvfPqSearchParams p;
    p.nprobe = nprobe;
    p.refine_factor = refine;
    return p;
  }
  void batch(const float* q, std::size_t nq, std::size_t k, quiver::Label* labels, float* dist,
             std::size_t nprobe, quiver::ThreadPool& pool) const override {
    index->search_batch(q, nq, k, labels, dist, params(nprobe), &pool);
  }
  void one(const float* q, std::size_t k, quiver::Label* labels, float* dist,
           std::size_t nprobe) const override {
    static quiver::ThreadPool inline_pool(1);
    index->search_batch(q, 1, k, labels, dist, params(nprobe), &inline_pool);
  }
};

}  // namespace

int main(int argc, char** argv) try {
#if defined(__APPLE__)
  // Ask for performance cores. Threads created afterwards (the pool) inherit this QoS class;
  // without it, macOS may schedule a benchmark on efficiency cores and add 2x noise.
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
  const Args args = parse(argc, argv);
  if (args.has("help") || argc == 1) {
    std::printf(
        "usage: ann_bench (--base F --query F [--gt F] | --synthetic n,dim,nq[,clusters])\n"
        "  --index hnsw|ivfpq|flat   --metric l2|ip|cosine   --storage f32|sq8\n"
        "  ivfpq: --nlist 1024 --pqm 16 --refine 0 (--ef is nprobe)\n"
        "  --M 16 --efc 200 --ef 10,20,40,80,160,320 --k 10 --threads N --reps 3\n"
        "  --max-base N --max-query N --save PATH --load PATH [--mmap] --json OUT.jsonl --tag "
        "NAME\n");
    return 0;
  }

  const std::size_t threads = args.num("threads", 0);
  quiver::ThreadPool pool(threads);
  const std::size_t k = args.num("k", 10);
  const std::size_t reps = std::max<std::size_t>(1, args.num("reps", 3));
  const quiver::Metric metric = quiver::parse_metric(args.get("metric", "l2"));
  const std::string index_kind = args.get("index", "hnsw");

  // ---- data ----
  Matrix base, queries;
  IntMatrix gt;
  std::string dataset;
  if (args.has("synthetic")) {
    const auto spec = parse_list(args.get("synthetic"));
    if (spec.size() < 3) throw std::runtime_error("--synthetic needs n,dim,nq[,clusters]");
    const std::size_t clusters = spec.size() > 3 ? spec[3] : 100;
    base = synthetic(spec[0], spec[1], clusters, 1, 42);
    queries = synthetic(spec[2], spec[1], clusters, 2, 42);
    dataset = "synthetic-" + std::to_string(spec[0]) + "x" + std::to_string(spec[1]);
  } else {
    base = quiver::bench::load_vectors(args.get("base"), args.num("max-base", 0));
    queries = quiver::bench::load_vectors(args.get("query"), args.num("max-query", 0));
    dataset = args.get("tag", args.get("base"));
  }
  if (args.has("gt") && !args.has("max-base")) {
    gt = quiver::bench::load_ids(args.get("gt"), queries.rows);
    if (gt.cols < k) throw std::runtime_error("ground truth has fewer than k columns");
  } else {
    const auto t0 = Clock::now();
    gt = exact_ground_truth(base, queries, k, metric, pool);
    std::fprintf(stderr, "computed exact ground truth in %.2fs\n", seconds_since(t0));
  }
  std::fprintf(stderr, "dataset %s: %zu base, %zu queries, dim %zu, simd=%s, threads=%zu\n",
               dataset.c_str(), base.rows, queries.rows, base.cols,
               std::string(quiver::simd::to_string(quiver::simd::active().isa)).c_str(),
               pool.num_threads());

  // ---- build ----
  std::unique_ptr<quiver::HnswIndex> hnsw;
  std::unique_ptr<quiver::FlatIndex> flat;
  std::unique_ptr<quiver::IvfPqIndex> ivfpq_holder;
  std::unique_ptr<Searcher> searcher;
  double build_s = 0.0;
  std::size_t index_bytes = 0;
  std::string config;
  std::vector<std::size_t> efforts;

  if (index_kind == "hnsw") {
    const auto t0 = Clock::now();
    if (args.has("load")) {
      quiver::LoadOptions lo;
      lo.mmap = args.has("mmap");
      lo.verify_checksums = !args.has("mmap");
      hnsw = std::make_unique<quiver::HnswIndex>(quiver::HnswIndex::load(args.get("load"), lo));
      std::fprintf(stderr, "loaded %s in %.3fs (%s)\n", args.get("load").c_str(), seconds_since(t0),
                   lo.mmap ? "mmap" : "read");
    } else {
      quiver::HnswParams p;
      p.dim = base.cols;
      p.metric = metric;
      p.storage =
          args.get("storage", "f32") == "sq8" ? quiver::Storage::kSQ8 : quiver::Storage::kFloat32;
      p.M = args.num("M", 16);
      p.ef_construction = args.num("efc", 200);
      hnsw = std::make_unique<quiver::HnswIndex>(p);
      if (p.storage == quiver::Storage::kSQ8)
        hnsw->train(base.data.data(), std::min<std::size_t>(base.rows, 100000));
      hnsw->add(base.data.data(), base.rows, nullptr, &pool);
      build_s = seconds_since(t0);
    }
    const auto st = hnsw->stats();
    index_bytes = st.vector_bytes + st.graph_bytes;
    const auto& p = hnsw->params();
    config = "M=" + std::to_string(p.M) + ",efc=" + std::to_string(p.ef_construction) +
             ",storage=" + (p.storage == quiver::Storage::kSQ8 ? "sq8" : "f32");
    std::fprintf(stderr,
                 "built HNSW (%s) in %.2fs: %zu nodes, %d levels, mean degree %.1f, %.1f MiB\n",
                 config.c_str(), build_s, st.num_nodes, st.max_level + 1, st.mean_degree_level0,
                 static_cast<double>(index_bytes) / (1 << 20));
    if (args.has("save")) {
      const auto ts = Clock::now();
      hnsw->save(args.get("save"));
      std::fprintf(stderr, "saved to %s in %.2fs\n", args.get("save").c_str(), seconds_since(ts));
    }
    searcher = std::make_unique<HnswSearcher>(hnsw.get());
    efforts = parse_list(args.get("ef", "10,20,40,80,160,320"));
  } else if (index_kind == "ivfpq") {
    quiver::IvfPqParams p;
    p.dim = base.cols;
    p.metric = metric;
    p.nlist = args.num("nlist", 1024);
    p.m = args.num("pqm", 16);
    const std::size_t refine = args.num("refine", 0);
    p.store_vectors = refine > 0;
    std::unique_ptr<quiver::IvfPqIndex> ivf = std::make_unique<quiver::IvfPqIndex>(p);
    const auto t0 = Clock::now();
    ivf->train(base.data.data(),
               std::min<std::size_t>(base.rows, std::max<std::size_t>(100000, 64 * p.nlist)),
               &pool);
    const double train_s = seconds_since(t0);
    ivf->add(base.data.data(), base.rows, nullptr, &pool);
    build_s = seconds_since(t0);
    index_bytes = ivf->memory_bytes();
    config = "nlist=" + std::to_string(p.nlist) + ",m=" + std::to_string(p.m) +
             ",refine=" + std::to_string(refine);
    std::fprintf(stderr, "built IVF-PQ (%s) in %.2fs (train %.2fs): %.1f MiB\n", config.c_str(),
                 build_s, train_s, static_cast<double>(index_bytes) / (1 << 20));
    searcher = std::make_unique<IvfPqSearcher>(ivf.get(), refine);
    ivfpq_holder = std::move(ivf);
    efforts = parse_list(args.get("ef", "1,2,4,8,16,32,64,128"));
  } else if (index_kind == "flat") {
    const auto t0 = Clock::now();
    flat = std::make_unique<quiver::FlatIndex>(base.cols, metric);
    flat->add(base.data.data(), base.rows);
    build_s = seconds_since(t0);
    index_bytes = base.rows * base.cols * sizeof(float);
    config = "exact";
    searcher = std::make_unique<FlatSearcher>(flat.get());
    efforts = {0};
  } else {
    throw std::runtime_error("unknown --index " + index_kind);
  }

  // ---- sweep ----
  std::FILE* json = args.has("json") ? std::fopen(args.get("json").c_str(), "a") : nullptr;
  const std::size_t nq = queries.rows;
  std::vector<quiver::Label> labels(nq * k);
  std::vector<float> dist(nq * k);
  std::vector<double> lat(nq);

  std::printf("%-8s %8s %12s %12s %10s %10s\n", "effort", "recall", "QPS(batch)", "QPS(1 thr)",
              "p50 us", "p99 us");
  for (std::size_t effort : efforts) {
    // Batch throughput across the pool: best of `reps`.
    double best_batch = 1e30;
    for (std::size_t r = 0; r < reps; ++r) {
      const auto t0 = Clock::now();
      searcher->batch(queries.data.data(), nq, k, labels.data(), dist.data(), effort, pool);
      best_batch = std::min(best_batch, seconds_since(t0));
    }
    const double rec = recall(gt, labels, nq, k);

    // Single-threaded latency: one query at a time on this thread. Best of `reps` per query
    // filters out scheduler noise without hiding systematic tail behavior.
    std::fill(lat.begin(), lat.end(), 1e30);
    double total_single = 0.0;
    for (std::size_t r = 0; r < reps; ++r) {
      const auto t_all = Clock::now();
      for (std::size_t q = 0; q < nq; ++q) {
        const auto t0 = Clock::now();
        searcher->one(queries.row(q), k, labels.data() + q * k, dist.data() + q * k, effort);
        lat[q] = std::min(lat[q], seconds_since(t0));
      }
      const double t = seconds_since(t_all);
      total_single = r == 0 ? t : std::min(total_single, t);
    }
    std::vector<double> sorted = lat;
    std::sort(sorted.begin(), sorted.end());
    const double p50 = sorted[nq / 2] * 1e6;
    const double p99 = sorted[std::min(nq - 1, nq * 99 / 100)] * 1e6;
    const double qps_batch = static_cast<double>(nq) / best_batch;
    const double qps_single = static_cast<double>(nq) / total_single;

    std::printf("%-8zu %8.4f %12.0f %12.0f %10.1f %10.1f\n", effort, rec, qps_batch, qps_single,
                p50, p99);
    if (json != nullptr) {
      std::fprintf(json,
                   "{\"library\":\"quiver\",\"dataset\":\"%s\",\"index\":\"%s\",\"config\":\"%s\","
                   "\"metric\":\"%s\",\"k\":%zu,\"effort\":%zu,\"recall\":%.5f,\"qps_batch\":%.1f,"
                   "\"qps_single\":%.1f,\"p50_us\":%.2f,\"p99_us\":%.2f,\"build_s\":%.3f,"
                   "\"index_bytes\":%zu,\"threads\":%zu,\"simd\":\"%s\"}\n",
                   dataset.c_str(), index_kind.c_str(), config.c_str(),
                   std::string(quiver::to_string(metric)).c_str(), k, effort, rec, qps_batch,
                   qps_single, p50, p99, build_s, index_bytes, pool.num_threads(),
                   std::string(quiver::simd::to_string(quiver::simd::active().isa)).c_str());
    }
  }
  if (json != nullptr) std::fclose(json);
  std::fprintf(stderr, "peak RSS %.1f MiB\n", static_cast<double>(peak_rss_bytes()) / (1 << 20));
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "error: %s\n", e.what());
  return 1;
}

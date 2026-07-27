// nanobind bindings. Kept thin: argument normalization (dtype/contiguity) happens in Python
// (quiver/__init__.py); this layer releases the GIL around every compute-heavy call.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>

#include "quiver/distance.h"
#include "quiver/filter.h"
#include "quiver/flat_index.h"
#include "quiver/hnsw_index.h"
#include "quiver/ivfpq_index.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using FloatMatrix = nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using LabelArray = nb::ndarray<const std::uint64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

/// Pools are expensive to spin up, so one is cached per requested size.
quiver::ThreadPool* pool_for(std::size_t num_threads) {
  if (num_threads == 0) return &quiver::ThreadPool::global();
  if (num_threads > 1024) throw nb::value_error("num_threads must be at most 1024");
  static std::mutex mu;
  static std::map<std::size_t, std::unique_ptr<quiver::ThreadPool>> pools;
  std::lock_guard lock(mu);
  auto& p = pools[num_threads];
  if (!p) p = std::make_unique<quiver::ThreadPool>(num_threads);
  return p.get();
}

/// Builds the cheapest filter for a set of allowed labels.
std::unique_ptr<quiver::Filter> make_filter(const LabelArray& allowed) {
  const std::uint64_t* p = allowed.data();
  const std::size_t n = allowed.shape(0);
  const std::uint64_t max_label = n == 0 ? 0 : *std::max_element(p, p + n);
  // A bitset costs max_label/8 bytes; use it unless labels are very sparse.
  if (max_label < (std::uint64_t{1} << 32) && max_label / 64 <= 16 * n + 1024) {
    auto f = std::make_unique<quiver::BitsetFilter>(static_cast<std::size_t>(max_label) + 1);
    for (std::size_t i = 0; i < n; ++i) f->insert(p[i]);
    return f;
  }
  return std::make_unique<quiver::LabelSetFilter>(p, p + n);
}

void check_dim(const FloatMatrix& x, std::size_t dim) {
  if (x.shape(1) != dim) {
    throw nb::value_error(("expected vectors of dimension " + std::to_string(dim) + ", got " +
                           std::to_string(x.shape(1)))
                              .c_str());
  }
}

/// nq * k, rejecting sizes that would overflow (or that no machine could allocate).
std::size_t result_size(std::size_t nq, std::size_t k) {
  std::size_t total = 0;
  if (__builtin_mul_overflow(nq, k, &total) || total > (std::size_t{1} << 40)) {
    throw nb::value_error("nq * k is too large");
  }
  return total;
}

/// (labels int64 [nq, k] with -1 for empty slots, distances float32 [nq, k]).
nb::tuple make_results(std::size_t nq, std::size_t k, std::vector<std::uint64_t>&& labels,
                       std::vector<float>&& dist) {
  auto* l = new std::vector<std::uint64_t>(std::move(labels));
  auto* d = new std::vector<float>(std::move(dist));
  nb::capsule lo(l, [](void* p) noexcept { delete static_cast<std::vector<std::uint64_t>*>(p); });
  nb::capsule dco(d, [](void* p) noexcept { delete static_cast<std::vector<float>*>(p); });
  // kInvalidLabel (2^64 - 1) reinterpreted as int64 is exactly -1.
  auto la = nb::ndarray<nb::numpy, std::int64_t, nb::ndim<2>>(
      reinterpret_cast<std::int64_t*>(l->data()), {nq, k}, lo);
  auto da = nb::ndarray<nb::numpy, float, nb::ndim<2>>(d->data(), {nq, k}, dco);
  return nb::make_tuple(la, da);
}

}  // namespace

NB_MODULE(_quiver, m) {
  m.doc() = "Quiver: SIMD-accelerated approximate nearest neighbor search (C++ core)";

  nb::register_exception_translator([](const std::exception_ptr& p, void*) {
    try {
      std::rethrow_exception(p);
    } catch (const quiver::Error& e) {
      PyErr_SetString(PyExc_ValueError, e.what());
    }
  });

  m.def(
      "simd_isa", [] { return std::string(quiver::simd::to_string(quiver::simd::active().isa)); },
      "Instruction set the distance kernels are using.");

  // ---- HNSW ----
  nb::class_<quiver::HnswIndex>(m, "HnswIndex")
      .def(
          "__init__",
          [](quiver::HnswIndex* self, std::size_t dim, const std::string& metric, std::size_t M,
             std::size_t ef_construction, const std::string& storage, std::uint64_t seed) {
            quiver::HnswParams p;
            p.dim = dim;
            p.metric = quiver::parse_metric(metric);
            p.M = M;
            p.ef_construction = ef_construction;
            p.seed = seed;
            if (storage == "float32" || storage == "f32") {
              p.storage = quiver::Storage::kFloat32;
            } else if (storage == "sq8" || storage == "int8" || storage == "uint8") {
              p.storage = quiver::Storage::kSQ8;
            } else {
              throw nb::value_error("storage must be 'float32' or 'sq8'");
            }
            new (self) quiver::HnswIndex(p);
          },
          "dim"_a, "metric"_a = "l2", "M"_a = 16, "ef_construction"_a = 200,
          "storage"_a = "float32", "seed"_a = 0x5eed)
      .def("_train",
           [](quiver::HnswIndex& self, const FloatMatrix& x) {
             check_dim(x, self.dim());
             nb::gil_scoped_release release;
             self.train(x.data(), x.shape(0));
           })
      .def(
          "_add",
          [](quiver::HnswIndex& self, const FloatMatrix& x, nb::object labels,
             std::size_t threads) {
            check_dim(x, self.dim());
            const std::uint64_t* lp = nullptr;
            LabelArray la;
            if (!labels.is_none()) {
              la = nb::cast<LabelArray>(labels);
              if (la.shape(0) != x.shape(0))
                throw nb::value_error("labels and vectors differ in length");
              lp = la.data();
            }
            quiver::ThreadPool* pool = pool_for(threads);
            nb::gil_scoped_release release;
            self.add(x.data(), x.shape(0), lp, pool);
          },
          "vectors"_a, "labels"_a.none(), "num_threads"_a)
      .def(
          "_search",
          [](const quiver::HnswIndex& self, const FloatMatrix& q, std::size_t k, std::size_t ef,
             nb::object allowed, double exact_threshold, std::size_t threads) {
            check_dim(q, self.dim());
            std::unique_ptr<quiver::Filter> filter;
            if (!allowed.is_none()) filter = make_filter(nb::cast<LabelArray>(allowed));
            const std::size_t nq = q.shape(0);
            std::vector<std::uint64_t> labels(result_size(nq, k));
            std::vector<float> dist(labels.size());
            quiver::HnswSearchParams p;
            p.ef = ef;
            p.filter = filter.get();
            p.exact_scan_threshold = exact_threshold;
            quiver::ThreadPool* pool = pool_for(threads);
            {
              nb::gil_scoped_release release;
              self.search_batch(q.data(), nq, k, labels.data(), dist.data(), p, pool);
            }
            return make_results(nq, k, std::move(labels), std::move(dist));
          },
          "queries"_a, "k"_a, "ef"_a, "allowed"_a.none(), "exact_threshold"_a, "num_threads"_a)
      // These may wait on an index lock (e.g. behind a large add()), so they release the GIL.
      .def("remove", &quiver::HnswIndex::remove, "label"_a,
           nb::call_guard<nb::gil_scoped_release>(),
           "Soft-delete a label. Returns False if it was not present.")
      .def("__contains__", &quiver::HnswIndex::contains, nb::call_guard<nb::gil_scoped_release>())
      .def("__len__", &quiver::HnswIndex::size)
      .def("get_vector",
           [](const quiver::HnswIndex& self, std::uint64_t label) {
             std::vector<float>* v = nullptr;
             {
               nb::gil_scoped_release release;
               v = new std::vector<float>(self.get_vector(label));
             }
             nb::capsule owner(
                 v, [](void* p) noexcept { delete static_cast<std::vector<float>*>(p); });
             return nb::ndarray<nb::numpy, float, nb::ndim<1>>(v->data(), {v->size()}, owner);
           })
      .def("reserve", &quiver::HnswIndex::reserve, nb::call_guard<nb::gil_scoped_release>())
      .def(
          "save",
          [](const quiver::HnswIndex& self, const std::string& path) {
            nb::gil_scoped_release release;
            self.save(path);
          },
          "path"_a)
      .def_static(
          "load",
          [](const std::string& path, bool mmap, bool verify_checksums) {
            quiver::LoadOptions o;
            o.mmap = mmap;
            o.verify_checksums = verify_checksums;
            nb::gil_scoped_release release;
            return quiver::HnswIndex::load(path, o);
          },
          "path"_a, "mmap"_a = false, "verify_checksums"_a = true)
      .def_prop_ro("dim", &quiver::HnswIndex::dim)
      .def_prop_ro(
          "metric",
          [](const quiver::HnswIndex& s) { return std::string(quiver::to_string(s.metric())); })
      .def_prop_ro("storage",
                   [](const quiver::HnswIndex& s) {
                     return s.storage() == quiver::Storage::kSQ8 ? "sq8" : "float32";
                   })
      .def_prop_ro("M", [](const quiver::HnswIndex& s) { return s.params().M; })
      .def_prop_ro("ef_construction",
                   [](const quiver::HnswIndex& s) { return s.params().ef_construction; })
      .def("stats", [](const quiver::HnswIndex& s) {
        quiver::HnswIndex::Stats st;
        {
          nb::gil_scoped_release release;  // O(n) scan
          st = s.stats();
        }
        nb::dict d;
        d["num_nodes"] = st.num_nodes;
        d["num_removed"] = st.num_removed;
        d["max_level"] = st.max_level;
        d["nodes_per_level"] = st.nodes_per_level;
        d["mean_degree_level0"] = st.mean_degree_level0;
        d["vector_bytes"] = st.vector_bytes;
        d["graph_bytes"] = st.graph_bytes;
        d["memory_mapped"] = st.memory_mapped;
        return d;
      });

  // ---- IVF-PQ ----
  nb::class_<quiver::IvfPqIndex>(m, "IvfPqIndex")
      .def(
          "__init__",
          [](quiver::IvfPqIndex* self, std::size_t dim, const std::string& metric,
             std::size_t nlist, std::size_t pq_m, bool store_vectors, std::size_t train_iterations,
             std::uint64_t seed) {
            quiver::IvfPqParams p;
            p.dim = dim;
            p.metric = quiver::parse_metric(metric);
            p.nlist = nlist;
            p.m = pq_m;
            p.store_vectors = store_vectors;
            p.train_iterations = train_iterations;
            p.seed = seed;
            new (self) quiver::IvfPqIndex(p);
          },
          "dim"_a, "metric"_a = "l2", "nlist"_a = 1024, "m"_a = 16, "store_vectors"_a = false,
          "train_iterations"_a = 20, "seed"_a = 1234)
      .def(
          "_train",
          [](quiver::IvfPqIndex& self, const FloatMatrix& x, std::size_t threads) {
            check_dim(x, self.dim());
            quiver::ThreadPool* pool = pool_for(threads);
            nb::gil_scoped_release release;
            self.train(x.data(), x.shape(0), pool);
          },
          "vectors"_a, "num_threads"_a)
      .def(
          "_add",
          [](quiver::IvfPqIndex& self, const FloatMatrix& x, nb::object labels,
             std::size_t threads) {
            check_dim(x, self.dim());
            const std::uint64_t* lp = nullptr;
            LabelArray la;
            if (!labels.is_none()) {
              la = nb::cast<LabelArray>(labels);
              if (la.shape(0) != x.shape(0))
                throw nb::value_error("labels and vectors differ in length");
              lp = la.data();
            }
            quiver::ThreadPool* pool = pool_for(threads);
            nb::gil_scoped_release release;
            self.add(x.data(), x.shape(0), lp, pool);
          },
          "vectors"_a, "labels"_a.none(), "num_threads"_a)
      .def(
          "_search",
          [](const quiver::IvfPqIndex& self, const FloatMatrix& q, std::size_t k,
             std::size_t nprobe, std::size_t refine_factor, nb::object allowed,
             std::size_t threads) {
            check_dim(q, self.dim());
            std::unique_ptr<quiver::Filter> filter;
            if (!allowed.is_none()) filter = make_filter(nb::cast<LabelArray>(allowed));
            const std::size_t nq = q.shape(0);
            std::vector<std::uint64_t> labels(result_size(nq, k));
            std::vector<float> dist(labels.size());
            quiver::IvfPqSearchParams p;
            p.nprobe = nprobe;
            p.refine_factor = refine_factor;
            p.filter = filter.get();
            quiver::ThreadPool* pool = pool_for(threads);
            {
              nb::gil_scoped_release release;
              self.search_batch(q.data(), nq, k, labels.data(), dist.data(), p, pool);
            }
            return make_results(nq, k, std::move(labels), std::move(dist));
          },
          "queries"_a, "k"_a, "nprobe"_a, "refine_factor"_a, "allowed"_a.none(), "num_threads"_a)
      .def("__len__", &quiver::IvfPqIndex::size, nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("is_trained", &quiver::IvfPqIndex::is_trained)
      .def_prop_ro("dim", &quiver::IvfPqIndex::dim)
      .def_prop_ro("nlist", [](const quiver::IvfPqIndex& s) { return s.params().nlist; })
      .def_prop_ro("m", [](const quiver::IvfPqIndex& s) { return s.params().m; })
      .def_prop_ro("memory_bytes",
                   [](const quiver::IvfPqIndex& s) {
                     nb::gil_scoped_release release;
                     return s.memory_bytes();
                   })
      .def("list_sizes", &quiver::IvfPqIndex::list_sizes, nb::call_guard<nb::gil_scoped_release>())
      .def("save",
           [](const quiver::IvfPqIndex& self, const std::string& path) {
             nb::gil_scoped_release release;
             self.save(path);
           })
      .def_static(
          "load",
          [](const std::string& path, bool verify_checksums) {
            quiver::LoadOptions o;
            o.verify_checksums = verify_checksums;
            nb::gil_scoped_release release;
            return quiver::IvfPqIndex::load(path, o);
          },
          "path"_a, "verify_checksums"_a = true);

  // ---- Flat ----
  nb::class_<quiver::FlatIndex>(m, "FlatIndex")
      .def(
          "__init__",
          [](quiver::FlatIndex* self, std::size_t dim, const std::string& metric) {
            new (self) quiver::FlatIndex(dim, quiver::parse_metric(metric));
          },
          "dim"_a, "metric"_a = "l2")
      .def(
          "_add",
          [](quiver::FlatIndex& self, const FloatMatrix& x, nb::object labels) {
            check_dim(x, self.dim());
            const std::uint64_t* lp = nullptr;
            LabelArray la;
            if (!labels.is_none()) {
              la = nb::cast<LabelArray>(labels);
              if (la.shape(0) != x.shape(0))
                throw nb::value_error("labels and vectors differ in length");
              lp = la.data();
            }
            nb::gil_scoped_release release;
            self.add(x.data(), x.shape(0), lp);
          },
          "vectors"_a, "labels"_a.none())
      .def(
          "_search",
          [](const quiver::FlatIndex& self, const FloatMatrix& q, std::size_t k, nb::object allowed,
             std::size_t threads) {
            check_dim(q, self.dim());
            std::unique_ptr<quiver::Filter> filter;
            if (!allowed.is_none()) filter = make_filter(nb::cast<LabelArray>(allowed));
            const std::size_t nq = q.shape(0);
            std::vector<std::uint64_t> labels(result_size(nq, k));
            std::vector<float> dist(labels.size());
            quiver::ThreadPool* pool = pool_for(threads);
            {
              nb::gil_scoped_release release;
              self.search_batch(q.data(), nq, k, labels.data(), dist.data(), filter.get(), pool);
            }
            return make_results(nq, k, std::move(labels), std::move(dist));
          },
          "queries"_a, "k"_a, "allowed"_a.none(), "num_threads"_a)
      .def("__len__", &quiver::FlatIndex::size, nb::call_guard<nb::gil_scoped_release>())
      .def("save",
           [](const quiver::FlatIndex& self, const std::string& path) {
             nb::gil_scoped_release release;
             self.save(path);
           })
      .def_static(
          "load",
          [](const std::string& path, bool verify_checksums) {
            quiver::LoadOptions o;
            o.verify_checksums = verify_checksums;
            nb::gil_scoped_release release;
            return quiver::FlatIndex::load(path, o);
          },
          "path"_a, "verify_checksums"_a = true)
      .def_prop_ro("dim", &quiver::FlatIndex::dim)
      .def_prop_ro("metric", [](const quiver::FlatIndex& s) {
        return std::string(quiver::to_string(s.metric()));
      });
}

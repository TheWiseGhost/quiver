// Microbenchmarks for the distance kernels, one series per ISA so the SIMD speedup is explicit.
//   ./bench_kernels --benchmark_filter=l2sqr
#include <benchmark/benchmark.h>

#include <random>
#include <string>
#include <vector>

#include "quiver/distance.h"

namespace {

using quiver::simd::Isa;
using quiver::simd::KernelTable;

std::vector<float> random_floats(std::size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d;
  std::vector<float> v(n);
  for (float& x : v) x = d(rng);
  return v;
}

// A pool of vectors larger than L1 would make this a memory benchmark; keep both operands hot
// so the numbers isolate arithmetic throughput.
void float_kernel(benchmark::State& state, Isa isa, bool is_dot) {
  const KernelTable* k = quiver::simd::get(isa);
  if (k == nullptr) {
    state.SkipWithError("ISA not available");
    return;
  }
  const auto dim = static_cast<std::size_t>(state.range(0));
  const auto a = random_floats(dim, 1);
  const auto b = random_floats(dim, 2);
  const auto fn = is_dot ? k->dot : k->l2sqr;
  for (auto _ : state) {
    benchmark::DoNotOptimize(fn(a.data(), b.data(), dim));
  }
  const double flops = 2.0 * static_cast<double>(dim) * static_cast<double>(state.iterations());
  state.counters["GFLOP/s"] = benchmark::Counter(flops / 1e9, benchmark::Counter::kIsRate);
}

void sq8_kernel(benchmark::State& state, Isa isa) {
  const KernelTable* k = quiver::simd::get(isa);
  if (k == nullptr) {
    state.SkipWithError("ISA not available");
    return;
  }
  const auto dim = static_cast<std::size_t>(state.range(0));
  const auto q = random_floats(dim, 1);
  const auto vmin = random_floats(dim, 2);
  std::vector<float> vscale(dim, 0.01f);
  std::vector<std::uint8_t> code(dim);
  for (std::size_t i = 0; i < dim; ++i) code[i] = static_cast<std::uint8_t>(i * 37);
  for (auto _ : state) {
    benchmark::DoNotOptimize(k->l2sqr_sq8(q.data(), code.data(), vmin.data(), vscale.data(), dim));
  }
  state.counters["dims/s"] =
      benchmark::Counter(static_cast<double>(dim) * static_cast<double>(state.iterations()),
                         benchmark::Counter::kIsRate);
}

void register_all() {
  for (Isa isa : quiver::simd::available()) {
    const std::string name(quiver::simd::to_string(isa));
    for (auto* b : {
             benchmark::RegisterBenchmark(("l2sqr/" + name).c_str(), float_kernel, isa, false),
             benchmark::RegisterBenchmark(("dot/" + name).c_str(), float_kernel, isa, true),
             benchmark::RegisterBenchmark(("l2sqr_sq8/" + name).c_str(), sq8_kernel, isa),
         }) {
      b->Arg(32)->Arg(128)->Arg(384)->Arg(768)->Arg(1536);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  register_all();
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}

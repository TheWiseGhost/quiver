// libFuzzer target: arbitrary bytes as an index file. Loading must either throw quiver::Error or
// produce an index on which searches and inserts are memory-safe.
//
// Checksum verification is disabled so the fuzzer explores the structural validator rather than
// bouncing off the CRCs.
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "quiver/hnsw_index.h"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  static const std::string path = "/tmp/quiver_fuzz_" + std::to_string(::getpid()) + ".qvr";
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) return 0;
  std::fwrite(data, 1, size, f);
  std::fclose(f);
  try {
    quiver::LoadOptions opts;
    opts.verify_checksums = false;
    quiver::HnswIndex index = quiver::HnswIndex::load(path, opts);
    if (index.dim() > 4096) return 0;
    std::vector<float> q(index.dim(), 0.5f);
    (void)index.search(q.data(), 5);
    const quiver::Label label = 1u << 30;
    index.add(q.data(), 1, &label);
    (void)index.search(q.data(), 5);
  } catch (const quiver::Error&) {
  }
  return 0;
}

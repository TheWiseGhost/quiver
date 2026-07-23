#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "quiver/hnsw_index.h"

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "corpus";
  std::mt19937 rng(1);
  std::normal_distribution<float> d;
  int i = 0;
  for (auto storage : {quiver::Storage::kFloat32, quiver::Storage::kSQ8}) {
    for (std::size_t n : {0u, 1u, 7u, 60u}) {
      quiver::HnswParams p;
      p.dim = 4;
      p.M = 4;
      p.storage = storage;
      quiver::HnswIndex index(p);
      if (storage == quiver::Storage::kSQ8) {
        std::vector<float> sample(300 * 4);
        for (float& x : sample) x = d(rng);
        index.train(sample.data(), 300);
      }
      std::vector<float> v(n * 4);
      for (float& x : v) x = d(rng);
      index.add(v.data(), n);
      if (n > 3) index.remove(2);
      index.save(dir + "/seed" + std::to_string(i++) + ".qvr");
    }
  }
  std::printf("wrote %d seeds to %s\n", i, dir.c_str());
}

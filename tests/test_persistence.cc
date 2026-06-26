#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>

#include "quiver/flat_index.h"
#include "quiver/hnsw_index.h"
#include "test_util.h"

namespace quiver {
namespace {

std::vector<char> read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void write_file(const std::string& path, const std::vector<char>& bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

HnswIndex build(std::size_t n, std::size_t dim, Storage storage = Storage::kFloat32,
                Metric metric = Metric::kL2) {
  HnswParams p;
  p.dim = dim;
  p.storage = storage;
  p.metric = metric;
  p.M = 12;
  p.ef_construction = 64;
  HnswIndex index(p);
  const auto base = testing::random_vectors(n, dim, 99);
  index.add(base.data(), n);
  return index;
}

void expect_same_results(const HnswIndex& a, const HnswIndex& b, std::size_t dim) {
  const auto queries = testing::random_vectors(50, dim, 123);
  for (std::size_t q = 0; q < 50; ++q) {
    EXPECT_EQ(a.search(queries.data() + q * dim, 10), b.search(queries.data() + q * dim, 10));
  }
}

class HnswRoundTrip : public ::testing::TestWithParam<std::tuple<Storage, bool>> {};

TEST_P(HnswRoundTrip, LoadedIndexIsIdentical) {
  const auto [storage, mmap] = GetParam();
  const std::size_t dim = 20;
  // > 1 chunk of metadata would be slow here; one partial chunk still exercises the tail copy.
  HnswIndex index = build(3000, dim, storage, Metric::kCosine);
  index.remove(5);
  index.remove(77);
  testing::TempPath path("hnsw.qvr");
  index.save(path.str());

  LoadOptions opts;
  opts.mmap = mmap;
  const HnswIndex loaded = HnswIndex::load(path.str(), opts);
  EXPECT_EQ(loaded.size(), index.size());
  EXPECT_EQ(loaded.storage(), storage);
  EXPECT_EQ(loaded.metric(), Metric::kCosine);
  EXPECT_FALSE(loaded.contains(5));
  EXPECT_TRUE(loaded.contains(6));
  EXPECT_EQ(loaded.stats().memory_mapped, mmap);
  EXPECT_EQ(loaded.stats().num_nodes, index.stats().num_nodes);
  EXPECT_EQ(loaded.stats().nodes_per_level, index.stats().nodes_per_level);
  expect_same_results(index, loaded, dim);
}

INSTANTIATE_TEST_SUITE_P(StorageAndMode, HnswRoundTrip,
                         ::testing::Combine(::testing::Values(Storage::kFloat32, Storage::kSQ8),
                                            ::testing::Bool()));

TEST(Persistence, MappedIndexIsMutableAndFileIsUntouched) {
  const std::size_t dim = 16;
  HnswIndex index = build(2000, dim);
  testing::TempPath path("cow.qvr");
  index.save(path.str());
  const auto before = read_file(path.str());

  LoadOptions opts;
  opts.mmap = true;
  HnswIndex mapped = HnswIndex::load(path.str(), opts);
  const auto extra = testing::random_vectors(500, dim, 5);
  std::vector<Label> labels(500);
  for (std::size_t i = 0; i < 500; ++i) labels[i] = 10000 + i;
  mapped.add(extra.data(), 500, labels.data());  // writes back-links into mapped pages
  mapped.remove(3);
  EXPECT_EQ(mapped.size(), 2000u + 500u - 1u);
  EXPECT_EQ(mapped.search(extra.data(), 1)[0].label, 10000u);

  EXPECT_EQ(read_file(path.str()), before) << "copy-on-write mapping leaked writes to disk";

  // Saving over the file we are mapped from is safe (atomic rename keeps the old inode alive).
  mapped.save(path.str());
  EXPECT_EQ(mapped.search(extra.data(), 1)[0].label, 10000u);
  const HnswIndex reloaded = HnswIndex::load(path.str());
  EXPECT_EQ(reloaded.size(), mapped.size());
  expect_same_results(mapped, reloaded, dim);
}

TEST(Persistence, EmptyIndexRoundTrips) {
  HnswParams p;
  p.dim = 8;
  p.storage = Storage::kSQ8;
  HnswIndex empty(p);
  testing::TempPath path("empty.qvr");
  empty.save(path.str());
  HnswIndex loaded = HnswIndex::load(path.str());
  EXPECT_EQ(loaded.size(), 0u);
  const auto v = testing::random_vectors(300, 8, 1);
  EXPECT_THROW(loaded.add(v.data(), 10), Error);     // too few vectors to auto-train SQ8 on
  loaded.train(v.data(), 300);                       // still trainable after load
  EXPECT_THROW(loaded.train(v.data(), 300), Error);  // but only once
  loaded.add(v.data(), 10);
  EXPECT_EQ(loaded.size(), 10u);
}

TEST(Persistence, RejectsWrongFiles) {
  testing::TempPath path("bad.qvr");
  write_file(path.str(), std::vector<char>(200, 'x'));
  EXPECT_THROW(HnswIndex::load(path.str()), Error);
  write_file(path.str(), {});
  EXPECT_THROW(HnswIndex::load(path.str()), Error);
  EXPECT_THROW(HnswIndex::load("/nonexistent/dir/x.qvr"), Error);

  FlatIndex flat(4);
  flat.save(path.str());
  EXPECT_THROW(HnswIndex::load(path.str()), Error);  // wrong index kind
}

TEST(Persistence, DetectsTruncationAndBitFlips) {
  HnswIndex index = build(500, 8);
  testing::TempPath path("flip.qvr");
  index.save(path.str());
  const auto good = read_file(path.str());

  auto truncated = good;
  truncated.resize(good.size() - 100);
  write_file(path.str(), truncated);
  EXPECT_THROW(HnswIndex::load(path.str()), Error);

  std::mt19937 rng(3);
  for (int trial = 0; trial < 50; ++trial) {
    auto bad = good;
    const std::size_t pos = rng() % bad.size();
    bad[pos] = static_cast<char>(bad[pos] ^ (1 << (rng() % 8)));
    write_file(path.str(), bad);
    // Every single-bit flip is caught by a checksum.
    EXPECT_THROW(HnswIndex::load(path.str()), Error) << "flip at byte " << pos;
  }
}

// Without checksum verification, corruption must still never cause memory unsafety: structural
// validation rejects anything that could send a search out of bounds. (Most useful under ASan.)
TEST(Persistence, CorruptGraphNeverCrashesWithoutChecksums) {
  const std::size_t dim = 8;
  HnswIndex index = build(400, dim);
  testing::TempPath path("fuzz.qvr");
  index.save(path.str());
  const auto good = read_file(path.str());
  const auto queries = testing::random_vectors(5, dim, 8);

  std::mt19937 rng(4);
  LoadOptions opts;
  opts.verify_checksums = false;
  int rejected = 0, accepted = 0;
  for (int trial = 0; trial < 300; ++trial) {
    auto bad = good;
    const int flips = 1 + static_cast<int>(rng() % 8);
    for (int f = 0; f < flips; ++f) {
      // Skip the header + table, which are always checksummed; target the payload.
      const std::size_t pos = 256 + rng() % (bad.size() - 256);
      bad[pos] = static_cast<char>(rng());
    }
    write_file(path.str(), bad);
    try {
      HnswIndex loaded = HnswIndex::load(path.str(), opts);
      for (std::size_t q = 0; q < 5; ++q) (void)loaded.search(queries.data() + q * dim, 5);
      const auto extra = testing::random_vectors(3, dim, static_cast<std::uint64_t>(trial));
      std::vector<Label> labels = {900000, 900001, 900002};
      loaded.add(extra.data(), 3, labels.data());
      ++accepted;
    } catch (const Error&) {
      ++rejected;
    }
  }
  EXPECT_GT(rejected, 0);
  EXPECT_GT(accepted, 0);
}

}  // namespace
}  // namespace quiver

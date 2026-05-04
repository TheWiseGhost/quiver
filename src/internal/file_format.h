// Quiver's on-disk container: a checksummed header, a section table, and 64-byte-aligned sections.
//
//   offset 0     FileHeader (64 bytes)
//   offset 64    SectionEntry[num_sections]
//   ...          sections, each starting on a 64-byte boundary
//
// Sections are raw little-endian arrays, laid out so they can be used in place from a memory
// mapping (no parsing or copying for the large vector and graph arrays). Every section carries a
// CRC-32C, and the header and table are always verified on load.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include "quiver/common.h"

namespace quiver::internal {

inline constexpr std::uint32_t kFormatVersion = 1;
inline constexpr std::size_t kSectionAlignment = 64;
inline constexpr std::size_t kMaxSections = 64;

enum class IndexKind : std::uint32_t { kFlat = 1, kHnsw = 2, kIvfPq = 3 };

constexpr std::uint32_t fourcc(const char (&s)[5]) noexcept {
  return static_cast<std::uint32_t>(static_cast<unsigned char>(s[0])) |
         static_cast<std::uint32_t>(static_cast<unsigned char>(s[1])) << 8 |
         static_cast<std::uint32_t>(static_cast<unsigned char>(s[2])) << 16 |
         static_cast<std::uint32_t>(static_cast<unsigned char>(s[3])) << 24;
}

struct FileHeader {
  char magic[8];
  std::uint32_t version;
  std::uint32_t kind;
  std::uint32_t endian_tag;
  std::uint32_t num_sections;
  std::uint64_t file_size;
  std::uint32_t table_crc;
  std::uint32_t header_crc;  // CRC of every byte before this field
  std::uint8_t reserved[24];
};
static_assert(sizeof(FileHeader) == 64 && std::is_trivially_copyable_v<FileHeader>);

struct SectionEntry {
  std::uint32_t tag;
  std::uint32_t crc;
  std::uint64_t offset;
  std::uint64_t size;
};
static_assert(sizeof(SectionEntry) == 24 && std::is_trivially_copyable_v<SectionEntry>);

/// Streams one section's bytes to disk while checksumming them.
class SectionSink {
 public:
  virtual void write(const void* data, std::size_t size) = 0;

 protected:
  ~SectionSink() = default;
};

/// Builds a file from sections and commits it atomically (temp file + fsync + rename), so readers
/// (including live memory mappings of the previous version) never see a torn file.
class FileWriter {
 public:
  using Producer = std::function<void(SectionSink&)>;

  explicit FileWriter(IndexKind kind) : kind_(kind) {}

  /// `produce` must write exactly `size` bytes.
  void add(std::uint32_t tag, std::size_t size, Producer produce);

  /// Section backed by caller memory that must stay valid until commit().
  void add_bytes(std::uint32_t tag, const void* data, std::size_t size);

  /// Section holding a copy of a trivially copyable value.
  template <class T>
  void add_pod(std::uint32_t tag, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    auto copy = std::make_shared<T>(value);
    add(tag, sizeof(T), [copy](SectionSink& s) { s.write(copy.get(), sizeof(T)); });
  }

  void commit(const std::string& path) const;

 private:
  struct Pending {
    std::uint32_t tag;
    std::size_t size;
    Producer produce;
  };
  IndexKind kind_;
  std::vector<Pending> sections_;
};

/// A validated, loaded file. The bytes are writable: either a private copy-on-write mapping of
/// the file (mmap mode; pages load lazily and writes never reach the file) or an owned copy.
class FileImage {
 public:
  struct Options {
    bool mmap = false;
    bool verify_checksums = true;
  };

  static std::shared_ptr<FileImage> open(const std::string& path, IndexKind kind,
                                         const Options& options);
  ~FileImage();

  FileImage(const FileImage&) = delete;
  FileImage& operator=(const FileImage&) = delete;

  bool has(std::uint32_t tag) const noexcept { return find(tag) != nullptr; }

  /// Raw bytes of a required section.
  std::span<std::byte> bytes(std::uint32_t tag) const;

  /// A required section reinterpreted as exactly `count` elements of T.
  template <class T>
  std::span<T> array(std::uint32_t tag, std::size_t count) const {
    static_assert(std::is_trivially_copyable_v<T>);
    const std::span<std::byte> b = bytes(tag);
    QUIVER_CHECK(count <= b.size() / sizeof(T) && b.size() == count * sizeof(T),
                 "corrupt index file: section " + tag_name(tag) + " has the wrong size");
    return {reinterpret_cast<T*>(b.data()), count};
  }

  /// A required section holding exactly one T, copied out.
  template <class T>
  T pod(std::uint32_t tag) const {
    static_assert(std::is_trivially_copyable_v<T>);
    const std::span<std::byte> b = bytes(tag);
    QUIVER_CHECK(b.size() == sizeof(T),
                 "corrupt index file: section " + tag_name(tag) + " has the wrong size");
    T value;
    std::memcpy(&value, b.data(), sizeof(T));
    return value;
  }

  bool mapped() const noexcept { return mapped_; }
  std::size_t size() const noexcept { return size_; }

  static std::string tag_name(std::uint32_t tag);

 private:
  FileImage() = default;
  const SectionEntry* find(std::uint32_t tag) const noexcept;

  std::byte* data_ = nullptr;
  std::size_t size_ = 0;
  bool mapped_ = false;
  std::vector<SectionEntry> sections_;
};

}  // namespace quiver::internal

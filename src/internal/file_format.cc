#include "internal/file_format.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <filesystem>
#include <system_error>

#include "internal/crc32c.h"

namespace quiver::internal {
namespace {

constexpr char kMagic[8] = {'Q', 'U', 'I', 'V', 'E', 'R', 'D', 'B'};
constexpr std::uint32_t kEndianTag = 0x01020304u;

constexpr std::size_t align_up(std::size_t x) noexcept {
  return (x + kSectionAlignment - 1) & ~(kSectionAlignment - 1);
}

// std::strerror may return a shared static buffer; the error_category message is thread-safe.
std::string errno_text() { return std::generic_category().message(errno); }

/// Owns a POSIX fd for the writer; unlinks the temp file unless released.
class TempFile {
 public:
  explicit TempFile(std::string path) : path_(std::move(path)) {
    fd_ = ::open(path_.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    QUIVER_CHECK(fd_ >= 0, "cannot create '" + path_ + "': " + errno_text());
  }
  ~TempFile() {
    if (fd_ >= 0) ::close(fd_);
    if (!committed_) ::unlink(path_.c_str());
  }
  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;

  void write_at(const void* data, std::size_t size, std::size_t offset) {
    const auto* p = static_cast<const char*>(data);
    while (size > 0) {
      const ssize_t n = ::pwrite(fd_, p, size, static_cast<off_t>(offset));
      if (n < 0 && errno == EINTR) continue;
      QUIVER_CHECK(n > 0, "write to '" + path_ + "' failed: " + errno_text());
      p += n;
      size -= static_cast<std::size_t>(n);
      offset += static_cast<std::size_t>(n);
    }
  }

  void sync_and_close() {
#if defined(__APPLE__)
    // fsync() on macOS only reaches the drive cache; F_FULLFSYNC flushes to stable storage.
    if (::fcntl(fd_, F_FULLFSYNC) != 0) {
      QUIVER_CHECK(::fsync(fd_) == 0, "fsync '" + path_ + "' failed: " + errno_text());
    }
#else
    QUIVER_CHECK(::fsync(fd_) == 0, "fsync '" + path_ + "' failed: " + errno_text());
#endif
    const int fd = fd_;
    fd_ = -1;
    QUIVER_CHECK(::close(fd) == 0, "close '" + path_ + "' failed: " + errno_text());
  }

  void resize(std::size_t size) {
    QUIVER_CHECK(::ftruncate(fd_, static_cast<off_t>(size)) == 0,
                 "resize '" + path_ + "' failed: " + errno_text());
  }

  void mark_committed() noexcept { committed_ = true; }
  const std::string& path() const noexcept { return path_; }

 private:
  std::string path_;
  int fd_ = -1;
  bool committed_ = false;
};

/// Buffers section bytes, writes them at increasing offsets, and accumulates the CRC.
class BufferedSink final : public SectionSink {
 public:
  BufferedSink(TempFile& file, std::size_t offset) : file_(file), offset_(offset) {
    buffer_.reserve(kBufferSize);
  }

  void write(const void* data, std::size_t size) override {
    crc_ = crc32c(data, size, crc_);
    written_ += size;
    const auto* p = static_cast<const char*>(data);
    if (size >= kBufferSize) {
      flush();
      file_.write_at(p, size, offset_);
      offset_ += size;
      return;
    }
    if (buffer_.size() + size > kBufferSize) flush();
    buffer_.insert(buffer_.end(), p, p + size);
  }

  void flush() {
    if (buffer_.empty()) return;
    file_.write_at(buffer_.data(), buffer_.size(), offset_);
    offset_ += buffer_.size();
    buffer_.clear();
  }

  std::uint32_t crc() const noexcept { return crc_; }
  std::size_t written() const noexcept { return written_; }

 private:
  static constexpr std::size_t kBufferSize = std::size_t{1} << 20;
  TempFile& file_;
  std::size_t offset_;
  std::vector<char> buffer_;
  std::uint32_t crc_ = 0;
  std::size_t written_ = 0;
};

void fsync_parent_dir(const std::string& path) {
  std::filesystem::path dir = std::filesystem::path(path).parent_path();
  if (dir.empty()) dir = ".";
  const int fd = ::open(dir.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd >= 0) {
    ::fsync(fd);  // best effort: makes the rename itself durable
    ::close(fd);
  }
}

}  // namespace

void FileWriter::add(std::uint32_t tag, std::size_t size, Producer produce) {
  for (const auto& s : sections_) {
    QUIVER_CHECK(s.tag != tag, "duplicate section " + FileImage::tag_name(tag));
  }
  QUIVER_CHECK(sections_.size() < kMaxSections, "too many sections");
  sections_.push_back({tag, size, std::move(produce)});
}

void FileWriter::add_bytes(std::uint32_t tag, const void* data, std::size_t size) {
  add(tag, size, [data, size](SectionSink& s) {
    if (size > 0) s.write(data, size);
  });
}

void FileWriter::commit(const std::string& path) const {
  // Unique per process and per call, so concurrent saves never share a temp file.
  static std::atomic<std::uint64_t> counter{0};
  TempFile file(path + ".tmp." + std::to_string(::getpid()) + "." +
                std::to_string(counter.fetch_add(1, std::memory_order_relaxed)));

  std::vector<SectionEntry> table(sections_.size());
  std::size_t offset = align_up(sizeof(FileHeader) + table.size() * sizeof(SectionEntry));
  for (std::size_t i = 0; i < sections_.size(); ++i) {
    const Pending& s = sections_[i];
    BufferedSink sink(file, offset);
    s.produce(sink);
    sink.flush();
    QUIVER_CHECK(sink.written() == s.size, "internal error: section " + FileImage::tag_name(s.tag) +
                                               " wrote " + std::to_string(sink.written()) +
                                               " bytes, expected " + std::to_string(s.size));
    table[i] = SectionEntry{s.tag, sink.crc(), offset, s.size};
    offset = align_up(offset + s.size);
  }

  FileHeader header{};
  std::memcpy(header.magic, kMagic, sizeof(kMagic));
  header.version = kFormatVersion;
  header.kind = static_cast<std::uint32_t>(kind_);
  header.endian_tag = kEndianTag;
  header.num_sections = static_cast<std::uint32_t>(table.size());
  header.file_size = offset;
  header.table_crc = crc32c(table.data(), table.size() * sizeof(SectionEntry));
  header.header_crc = crc32c(&header, offsetof(FileHeader, header_crc));

  file.write_at(table.data(), table.size() * sizeof(SectionEntry), sizeof(FileHeader));
  file.write_at(&header, sizeof(header), 0);
  // Pad to the declared size so the final section's alignment padding exists on disk.
  file.resize(offset);
  file.sync_and_close();
  QUIVER_CHECK(std::rename(file.path().c_str(), path.c_str()) == 0,
               "rename to '" + path + "' failed: " + errno_text());
  file.mark_committed();
  fsync_parent_dir(path);
}

std::string FileImage::tag_name(std::uint32_t tag) {
  std::string s(4, '?');
  for (int i = 0; i < 4; ++i) {
    const char c = static_cast<char>((tag >> (8 * i)) & 0xff);
    s[static_cast<std::size_t>(i)] = (c >= 32 && c < 127) ? c : '?';
  }
  return "'" + s + "'";
}

std::shared_ptr<FileImage> FileImage::open(const std::string& path, IndexKind kind,
                                           const Options& options) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  QUIVER_CHECK(fd >= 0, "cannot open '" + path + "': " + errno_text());
  struct FdCloser {
    int fd;
    ~FdCloser() { ::close(fd); }
  } closer{fd};

  struct stat st{};
  QUIVER_CHECK(::fstat(fd, &st) == 0, "cannot stat '" + path + "': " + errno_text());
  QUIVER_CHECK(S_ISREG(st.st_mode), "'" + path + "' is not a regular file");
  const auto size = static_cast<std::size_t>(st.st_size);
  QUIVER_CHECK(size >= sizeof(FileHeader), "'" + path + "' is too small to be a Quiver index");

  std::shared_ptr<FileImage> image(new FileImage());
  image->size_ = size;
  if (options.mmap) {
    // Private + writable: pages are shared with the page cache until written, then copied.
    void* p = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    QUIVER_CHECK(p != MAP_FAILED, "cannot mmap '" + path + "': " + errno_text());
    image->data_ = static_cast<std::byte*>(p);
  } else {
    void* p = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    QUIVER_CHECK(p != MAP_FAILED, "out of memory loading '" + path + "'");
    image->data_ = static_cast<std::byte*>(p);
    std::size_t done = 0;
    while (done < size) {
      const ssize_t n = ::pread(fd, image->data_ + done, size - done, static_cast<off_t>(done));
      if (n < 0 && errno == EINTR) continue;
      QUIVER_CHECK(n > 0,
                   "read '" + path + "' failed: " + (n == 0 ? "unexpected EOF" : errno_text()));
      done += static_cast<std::size_t>(n);
    }
  }
  image->mapped_ = options.mmap;

  const std::string corrupt = "corrupt index file '" + path + "': ";
  FileHeader header;
  std::memcpy(&header, image->data_, sizeof(header));
  QUIVER_CHECK(std::memcmp(header.magic, kMagic, sizeof(kMagic)) == 0,
               "'" + path + "' is not a Quiver index (bad magic)");
  QUIVER_CHECK(header.endian_tag == kEndianTag,
               corrupt + "written on a machine with different endianness");
  QUIVER_CHECK(header.header_crc == crc32c(&header, offsetof(FileHeader, header_crc)),
               corrupt + "header checksum mismatch");
  QUIVER_CHECK(header.version == kFormatVersion,
               "'" + path + "' uses format version " + std::to_string(header.version) +
                   "; this build reads version " + std::to_string(kFormatVersion));
  QUIVER_CHECK(header.kind == static_cast<std::uint32_t>(kind),
               "'" + path + "' holds a different index type");
  QUIVER_CHECK(header.file_size == size, corrupt + "truncated or padded (size " +
                                             std::to_string(size) + ", header says " +
                                             std::to_string(header.file_size) + ")");
  QUIVER_CHECK(header.num_sections <= kMaxSections, corrupt + "too many sections");
  const std::size_t table_bytes = header.num_sections * sizeof(SectionEntry);
  QUIVER_CHECK(sizeof(FileHeader) + table_bytes <= size, corrupt + "section table out of bounds");
  QUIVER_CHECK(header.table_crc == crc32c(image->data_ + sizeof(FileHeader), table_bytes),
               corrupt + "section table checksum mismatch");

  image->sections_.resize(header.num_sections);
  std::memcpy(image->sections_.data(), image->data_ + sizeof(FileHeader), table_bytes);
  const std::size_t data_start = sizeof(FileHeader) + table_bytes;
  for (std::size_t i = 0; i < image->sections_.size(); ++i) {
    const SectionEntry& s = image->sections_[i];
    const std::string name = tag_name(s.tag);
    QUIVER_CHECK(s.offset % kSectionAlignment == 0, corrupt + "misaligned section " + name);
    QUIVER_CHECK(s.offset >= data_start && s.offset <= size && s.size <= size - s.offset,
                 corrupt + "section " + name + " out of bounds");
    for (std::size_t j = 0; j < i; ++j) {
      QUIVER_CHECK(image->sections_[j].tag != s.tag, corrupt + "duplicate section " + name);
    }
    if (options.verify_checksums) {
      QUIVER_CHECK(crc32c(image->data_ + s.offset, s.size) == s.crc,
                   corrupt + "checksum mismatch in section " + name);
    }
  }
  // Sections are used in place and some are written to later (copy-on-write), so two sections
  // sharing bytes could let a write through one corrupt invariants validated through the other.
  std::vector<SectionEntry> by_offset = image->sections_;
  std::sort(by_offset.begin(), by_offset.end(),
            [](const SectionEntry& a, const SectionEntry& b) { return a.offset < b.offset; });
  for (std::size_t i = 1; i < by_offset.size(); ++i) {
    QUIVER_CHECK(by_offset[i - 1].offset + by_offset[i - 1].size <= by_offset[i].offset,
                 corrupt + "overlapping sections");
  }
  return image;
}

FileImage::~FileImage() {
  if (data_ != nullptr) ::munmap(data_, size_);
}

const SectionEntry* FileImage::find(std::uint32_t tag) const noexcept {
  for (const auto& s : sections_) {
    if (s.tag == tag) return &s;
  }
  return nullptr;
}

std::span<std::byte> FileImage::bytes(std::uint32_t tag) const {
  const SectionEntry* s = find(tag);
  QUIVER_CHECK(s != nullptr, "corrupt index file: missing section " + tag_name(tag));
  return {data_ + s->offset, static_cast<std::size_t>(s->size)};
}

}  // namespace quiver::internal

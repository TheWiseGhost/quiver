// Label filters for restricting a search to a subset of the index ("pre-filtering").
#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

#include "quiver/common.h"

namespace quiver {

/// Decides which labels a search may return. Filtered-out vectors are still used to navigate the
/// graph (so connectivity is preserved) but never appear in results.
///
/// contains() may run on any search thread and may itself search *other* indexes, but must not
/// call methods of the index being searched (searches hold that index's read lock, and a
/// re-entrant call could deadlock behind a waiting writer).
class Filter {
 public:
  virtual ~Filter() = default;

  virtual bool contains(Label label) const = 0;

  /// Exact number of admitted labels, when cheap to know. Together with for_each() this lets an
  /// index switch to an exact scan when a filter is so selective that graph search would wander.
  virtual std::optional<std::size_t> cardinality() const { return std::nullopt; }

  /// Calls `fn` for every admitted label. Returns false if enumeration is unsupported.
  virtual bool for_each(const std::function<void(Label)>& /*fn*/) const { return false; }
};

/// Dense bitset over labels [0, universe). The right choice when labels are small integers.
class BitsetFilter final : public Filter {
 public:
  explicit BitsetFilter(std::size_t universe)
      : words_((universe + 63) / 64, 0), universe_(universe) {}

  void insert(Label label) {
    QUIVER_CHECK(label < universe_, "label outside BitsetFilter universe");
    std::uint64_t& w = words_[label >> 6];
    const std::uint64_t bit = std::uint64_t{1} << (label & 63);
    count_ += (w & bit) ? 0 : 1;
    w |= bit;
  }

  bool contains(Label label) const override {
    return label < universe_ && ((words_[label >> 6] >> (label & 63)) & 1) != 0;
  }

  std::optional<std::size_t> cardinality() const override { return count_; }

  bool for_each(const std::function<void(Label)>& fn) const override {
    for (std::size_t w = 0; w < words_.size(); ++w) {
      for (std::uint64_t bits = words_[w]; bits != 0; bits &= bits - 1) {
        fn((w << 6) | static_cast<unsigned>(std::countr_zero(bits)));
      }
    }
    return true;
  }

 private:
  std::vector<std::uint64_t> words_;
  std::size_t universe_;
  std::size_t count_ = 0;
};

/// Hash set of labels: for sparse or very large label values.
class LabelSetFilter final : public Filter {
 public:
  LabelSetFilter() = default;
  LabelSetFilter(std::initializer_list<Label> labels) : labels_(labels) {}
  template <class It>
  LabelSetFilter(It first, It last) : labels_(first, last) {}

  void insert(Label label) { labels_.insert(label); }
  bool contains(Label label) const override { return labels_.count(label) != 0; }
  std::optional<std::size_t> cardinality() const override { return labels_.size(); }
  bool for_each(const std::function<void(Label)>& fn) const override {
    for (Label l : labels_) fn(l);
    return true;
  }

 private:
  std::unordered_set<Label> labels_;
};

/// Arbitrary predicate. Selectivity is unknown, so searches always use the graph.
template <class Pred>
class PredicateFilter final : public Filter {
 public:
  explicit PredicateFilter(Pred pred) : pred_(std::move(pred)) {}
  bool contains(Label label) const override { return pred_(label); }

 private:
  Pred pred_;
};

}  // namespace quiver

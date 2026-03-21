#include "quiver/common.h"

namespace quiver {

std::string_view to_string(Metric metric) noexcept {
  switch (metric) {
    case Metric::kL2:
      return "l2";
    case Metric::kInnerProduct:
      return "ip";
    case Metric::kCosine:
      return "cosine";
  }
  return "unknown";
}

Metric parse_metric(std::string_view name) {
  if (name == "l2" || name == "euclidean") return Metric::kL2;
  if (name == "ip" || name == "inner_product" || name == "dot") return Metric::kInnerProduct;
  if (name == "cosine" || name == "cos" || name == "angular") return Metric::kCosine;
  throw Error("unknown metric '" + std::string(name) + "' (expected l2, ip, or cosine)");
}

namespace detail {
void throw_error(const std::string& message) { throw Error(message); }
}  // namespace detail

}  // namespace quiver

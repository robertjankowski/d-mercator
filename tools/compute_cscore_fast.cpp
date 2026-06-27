#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using count_t = std::int64_t;

constexpr double PI = 3.141592653589793238462643383279502884;
constexpr double TWO_PI = 2.0 * PI;

struct Point {
  double original;
  double inferred;
};

struct Group {
  std::size_t begin;
  std::size_t end;
  double original;
};

class FenwickTree {
 public:
  explicit FenwickTree(std::size_t size) : tree_(size + 1, 0) {}

  void add(std::size_t index, count_t delta) {
    for (++index; index < tree_.size(); index += index & -index) {
      tree_[index] += delta;
    }
  }

  // Returns the sum over ranks [0, count).
  count_t prefix(std::size_t count) const {
    count_t result = 0;
    for (; count > 0; count -= count & -count) {
      result += tree_[count];
    }
    return result;
  }

 private:
  std::vector<count_t> tree_;
};

double normalize_angle(double angle) {
  double normalized = std::fmod(angle, TWO_PI);
  if (normalized < 0.0) {
    normalized += TWO_PI;
  }
  if (normalized == 0.0 || normalized == TWO_PI) {
    return 0.0;
  }
  return normalized;
}

count_t choose_two(count_t value) {
  return value * (value - 1) / 2;
}

std::size_t rank_of(const std::vector<double>& coordinates, double value) {
  const auto it = std::lower_bound(coordinates.begin(), coordinates.end(), value);
  if (it == coordinates.end() || *it != value) {
    throw std::runtime_error("Internal error: inferred angle is absent from coordinate compression");
  }
  return static_cast<std::size_t>(it - coordinates.begin());
}

count_t count_exact(
    const FenwickTree& tree, const std::vector<double>& coordinates, double value) {
  const std::size_t rank = rank_of(coordinates, value);
  return tree.prefix(rank + 1) - tree.prefix(rank);
}

// Counts active angles z for which wrap(z - angle) is strictly positive.
count_t count_positive_direction(
    const FenwickTree& tree,
    const std::vector<double>& coordinates,
    double angle) {
  const auto equal_begin =
      std::lower_bound(coordinates.begin(), coordinates.end(), angle);
  const auto equal_end =
      std::upper_bound(equal_begin, coordinates.end(), angle);
  const auto forward_end = std::partition_point(
      equal_end, coordinates.end(), [angle](double value) {
        return value - angle < PI;
      });
  const auto wrapped_end = std::partition_point(
      coordinates.begin(), equal_begin, [angle](double value) {
        return value - angle < -PI;
      });
  const auto rank = [&coordinates](auto iterator) {
    return static_cast<std::size_t>(iterator - coordinates.begin());
  };
  return tree.prefix(rank(forward_end)) - tree.prefix(rank(equal_end)) +
         tree.prefix(rank(wrapped_end));
}

count_t brute_force_matches(const std::vector<Point>& points) {
  count_t matches = 0;
  for (std::size_t i = 0; i < points.size(); ++i) {
    for (std::size_t j = i + 1; j < points.size(); ++j) {
      const double original =
          normalize_angle(points[j].original - points[i].original + PI) - PI;
      const double inferred =
          normalize_angle(points[j].inferred - points[i].inferred + PI) - PI;
      const int original_sign = (original > 0.0) - (original < 0.0);
      const int inferred_sign = (inferred > 0.0) - (inferred < 0.0);
      matches += original_sign == inferred_sign;
    }
  }
  return matches;
}

count_t fast_matches(std::vector<Point> points) {
  const count_t size = static_cast<count_t>(points.size());
  if (size < 2) {
    return 0;
  }

  for (Point& point : points) {
    point.original = normalize_angle(point.original);
    point.inferred = normalize_angle(point.inferred);
  }
  std::sort(points.begin(), points.end(), [](const Point& left, const Point& right) {
    if (left.original != right.original) {
      return left.original < right.original;
    }
    return left.inferred < right.inferred;
  });

  std::vector<Group> groups;
  count_t original_tie_pairs = 0;
  count_t both_tie_pairs = 0;
  for (std::size_t begin = 0; begin < points.size();) {
    std::size_t end = begin + 1;
    while (end < points.size() && points[end].original == points[begin].original) {
      ++end;
    }
    groups.push_back({begin, end, points[begin].original});
    original_tie_pairs += choose_two(static_cast<count_t>(end - begin));

    for (std::size_t y_begin = begin; y_begin < end;) {
      std::size_t y_end = y_begin + 1;
      while (y_end < end && points[y_end].inferred == points[y_begin].inferred) {
        ++y_end;
      }
      both_tie_pairs += choose_two(static_cast<count_t>(y_end - y_begin));
      y_begin = y_end;
    }
    begin = end;
  }

  std::vector<double> inferred_coordinates;
  inferred_coordinates.reserve(points.size());
  for (const Point& point : points) {
    inferred_coordinates.push_back(point.inferred);
  }
  std::sort(inferred_coordinates.begin(), inferred_coordinates.end());
  inferred_coordinates.erase(
      std::unique(inferred_coordinates.begin(), inferred_coordinates.end()),
      inferred_coordinates.end());

  count_t inferred_tie_pairs = 0;
  {
    std::vector<count_t> frequencies(inferred_coordinates.size(), 0);
    for (const Point& point : points) {
      ++frequencies[rank_of(inferred_coordinates, point.inferred)];
    }
    for (const count_t frequency : frequencies) {
      inferred_tie_pairs += choose_two(frequency);
    }
  }

  // B: inferred-positive pairs across all pairs with different original angles.
  count_t inferred_positive_all = 0;
  {
    FenwickTree future(inferred_coordinates.size());
    for (std::size_t group_index = groups.size(); group_index-- > 0;) {
      const Group& group = groups[group_index];
      for (std::size_t index = group.begin; index < group.end; ++index) {
        inferred_positive_all += count_positive_direction(
            future, inferred_coordinates, points[index].inferred);
      }
      for (std::size_t index = group.begin; index < group.end; ++index) {
        future.add(rank_of(inferred_coordinates, points[index].inferred), 1);
      }
    }
  }

  // A: inferred-positive pairs whose original direction is also positive.
  // The same sliding window also counts inferred ties in this near half-circle.
  count_t original_positive_pairs = 0;
  count_t both_positive_pairs = 0;
  count_t inferred_ties_near = 0;
  {
    FenwickTree near(inferred_coordinates.size());
    count_t near_count = 0;
    std::vector<bool> active(groups.size(), false);
    std::size_t right = 0;

    for (std::size_t group_index = 0; group_index < groups.size(); ++group_index) {
      if (active[group_index]) {
        const Group& expired = groups[group_index];
        for (std::size_t index = expired.begin; index < expired.end; ++index) {
          near.add(rank_of(inferred_coordinates, points[index].inferred), -1);
          --near_count;
        }
        active[group_index] = false;
      }
      if (right < group_index) {
        right = group_index;
      }
      while (right + 1 < groups.size() &&
             groups[right + 1].original - groups[group_index].original < PI) {
        ++right;
        const Group& added = groups[right];
        for (std::size_t index = added.begin; index < added.end; ++index) {
          near.add(rank_of(inferred_coordinates, points[index].inferred), 1);
          ++near_count;
        }
        active[right] = true;
      }

      const Group& group = groups[group_index];
      original_positive_pairs +=
          static_cast<count_t>(group.end - group.begin) * near_count;
      for (std::size_t index = group.begin; index < group.end; ++index) {
        both_positive_pairs += count_positive_direction(
            near, inferred_coordinates, points[index].inferred);
        inferred_ties_near +=
            count_exact(near, inferred_coordinates, points[index].inferred);
      }
    }
  }

  const count_t total_pairs = choose_two(size);
  const count_t between_original_groups = total_pairs - original_tie_pairs;
  const count_t original_negative_pairs =
      between_original_groups - original_positive_pairs;
  const count_t inferred_ties_far =
      (inferred_tie_pairs - both_tie_pairs) - inferred_ties_near;

  const count_t matches_between =
      2 * both_positive_pairs + original_negative_pairs -
      inferred_positive_all - inferred_ties_far;
  const count_t matches = matches_between + both_tie_pairs;
  if (matches < 0 || matches > total_pairs) {
    throw std::runtime_error("Internal error: C-score match count is out of range");
  }
  return matches;
}

std::unordered_map<std::string, double> read_original_angles(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Could not open generated coordinates: " + path);
  }
  std::unordered_map<std::string, double> angles;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream fields(line);
    std::string name;
    double kappa = 0.0;
    double angle = 0.0;
    if (!(fields >> name >> kappa >> angle)) {
      throw std::runtime_error("Malformed generated-coordinate row in " + path);
    }
    if (!angles.emplace(std::move(name), angle).second) {
      throw std::runtime_error("Duplicate vertex in generated coordinates: " + path);
    }
  }
  return angles;
}

std::vector<Point> read_aligned_points(
    const std::string& inferred_path,
    const std::unordered_map<std::string, double>& originals) {
  std::ifstream input(inferred_path);
  if (!input) {
    throw std::runtime_error("Could not open inferred coordinates: " + inferred_path);
  }
  std::vector<Point> points;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream fields(line);
    std::string name;
    double kappa = 0.0;
    double angle = 0.0;
    if (!(fields >> name >> kappa >> angle)) {
      throw std::runtime_error("Malformed inferred-coordinate row in " + inferred_path);
    }
    const auto original = originals.find(name);
    if (original == originals.end()) {
      throw std::runtime_error("Inferred vertex is absent from generated coordinates: " + name);
    }
    points.push_back({original->second, angle});
  }
  return points;
}

void run_self_test() {
  std::mt19937_64 rng(20260627);
  std::uniform_real_distribution<double> continuous(0.0, TWO_PI);
  // Repeated values exercise exact angle ties. Avoid exactly antipodal values:
  // the production coordinate files are decimal, and antipodal floating-point
  // boundary behavior is not part of the C-score contract.
  const std::vector<double> tied_values = {
      0.0, 0.125, 0.75, 1.625, 3.25, 5.875};

  for (int trial = 0; trial < 500; ++trial) {
    const std::size_t size = 2 + static_cast<std::size_t>(rng() % 45);
    std::vector<Point> points;
    points.reserve(size);
    for (std::size_t index = 0; index < size; ++index) {
      const bool use_ties = trial % 2 == 0;
      points.push_back({
          use_ties ? tied_values[rng() % tied_values.size()] : continuous(rng),
          use_ties ? tied_values[rng() % tied_values.size()] : continuous(rng),
      });
    }
    const count_t expected = brute_force_matches(points);
    const count_t observed = fast_matches(points);
    if (expected != observed) {
      std::ostringstream message;
      message << "Self-test failed in trial " << trial << ": expected " << expected
              << " matches, observed " << observed;
      throw std::runtime_error(message.str());
    }
  }
  std::cout << "C-score self-test passed" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--self-test") {
      run_self_test();
      return 0;
    }
    if (argc != 3) {
      std::cerr << "Usage: " << argv[0]
                << " GENERATED_COORDINATES INFERRED_COORDINATES\n";
      return 2;
    }

    const auto originals = read_original_angles(argv[1]);
    std::vector<Point> points = read_aligned_points(argv[2], originals);
    if (points.size() < 2) {
      throw std::runtime_error("At least two aligned vertices are required");
    }
    const count_t aligned_vertices = static_cast<count_t>(points.size());
    const count_t total_pairs = choose_two(static_cast<count_t>(points.size()));
    const count_t matches = fast_matches(std::move(points));
    const count_t orientation_invariant_matches =
        std::max(matches, total_pairs - matches);
    const double c_score =
        static_cast<double>(orientation_invariant_matches) /
        static_cast<double>(total_pairs);

    std::cout << std::setprecision(17)
              << "{\"vertices\":" << originals.size()
              << ",\"aligned_vertices\":" << aligned_vertices
              << ",\"total_pairs\":" << total_pairs
              << ",\"matches\":" << matches
              << ",\"orientation_invariant_matches\":"
              << orientation_invariant_matches
              << ",\"c_score\":" << c_score << "}" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << std::endl;
    return 1;
  }
}

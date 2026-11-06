// Copyright 2026 AI Robot Association (AIRoA)
// SPDX-License-Identifier: Apache-2.0
// The checks a motion input passes before it reaches the realtime loop: pure, so the tests can call them.
#pragma once

#include <algorithm>
#include <cmath>
#include <iterator>

#include "geometry_msgs/msg/transform.hpp"
#include "oap_control/model.hpp"

namespace oap
{
constexpr double kQuatNormTol = 1e-2;  // a quaternion typed with 4 decimals is still a rotation
constexpr double kMaxPathLate = 0.1;  // [s] how far in the past a stamped joint path may start: build, DDS, executor
constexpr double kMaxPathEarly = 0.1;  // [s] and how far in the future: further is a skewed clock, not a schedule

// A stamped joint path that starts late seconds ago (negative: in the future) is on time to play.
inline bool path_on_time(double late) { return late <= kMaxPathLate && late >= -kMaxPathEarly; }

// All seven components finite, and the rotation a unit quaternion.
inline bool valid_transform(const geometry_msgs::msg::Transform & t)
{
  const auto & p = t.translation;
  const auto & r = t.rotation;
  for (const double x : {p.x, p.y, p.z, r.x, r.y, r.z, r.w}) {
    if (!std::isfinite(x)) {
      return false;
    }
  }
  return std::abs(std::sqrt(r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w) - 1.0) <= kQuatNormTol;
}

// Every time after the one before it (and none NaN): equal times give a zero-length spline segment.
template <class It>
bool strictly_increasing(It begin, It end)
{
  return std::adjacent_find(begin, end, [](double a, double b) { return !(a < b); }) == end &&
         std::all_of(begin, end, [](double x) { return std::isfinite(x); });
}

// Where a hold puts the command: the measured position, or the last finite command when a measurement is not finite.
inline const Vec & hold_position(const Vec & measured, const Vec & last_finite)
{
  return measured.allFinite() ? measured : last_finite;
}
}  // namespace oap

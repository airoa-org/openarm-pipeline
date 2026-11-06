// Copyright 2026 AI Robot Association (AIRoA)
// SPDX-License-Identifier: Apache-2.0
// Motion's inputs as the controller holds them, and its decisions on them: pure, so the tests can call them.
// parse_chunk and parse_path run in the subscription callbacks; path_start and trip_cause run in update() and do not
// allocate.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <string>

#include <pinocchio/spatial/se3.hpp>

#include "builtin_interfaces/msg/duration.hpp"
#include "builtin_interfaces/msg/time.hpp"
#include "geometry_msgs/msg/transform.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "trajectory_msgs/msg/multi_dof_joint_trajectory.hpp"

#include "oap_control/model.hpp"
#include "oap_control/spline.hpp"
#include "oap_control/validate.hpp"

namespace oap
{
constexpr double kMaxPathStartGap = 0.05;  // [rad] how far a joint path may start from the command

inline double seconds(const builtin_interfaces::msg::Time & t) { return t.sec + 1e-9 * t.nanosec; }
inline double seconds(const builtin_interfaces::msg::Duration & d) { return d.sec + 1e-9 * d.nanosec; }

inline pinocchio::SE3 to_se3(const geometry_msgs::msg::Transform & t)
{
  const Eigen::Quaterniond q(t.rotation.w, t.rotation.x, t.rotation.y, t.rotation.z);
  return pinocchio::SE3(q.normalized().toRotationMatrix(),
                        Eigen::Vector3d(t.translation.x, t.translation.y, t.translation.z));
}

// A chunk as received: knot times and poses, relative to the commanded hand when frame_id is the tip.
struct PoseChunk
{
  bool relative = false;
  int n = 0;
  std::array<double, kMaxKnots> t{};
  std::array<pinocchio::SE3, kMaxKnots> pose{};
};

// A joint path as received. An unstamped one (rebase) has times from its start, which is the update that takes it.
struct JointPath
{
  bool rebase = false;
  Hermite h;
};

enum class ChunkCheck
{
  kOk,
  kFrame,   // frame_id is neither the tip nor world
  kCount,   // not 1 to kMaxKnots - 1 points
  kPoint,   // a point without exactly one finite unit transform, or a time not after the one before
};

// Fills c from m when the chunk is acceptable for the hand whose frame is `tip`.
inline ChunkCheck parse_chunk(const trajectory_msgs::msg::MultiDOFJointTrajectory & m, const std::string & tip,
                              PoseChunk & c)
{
  if (m.header.frame_id != tip && m.header.frame_id != "world") {
    return ChunkCheck::kFrame;
  }
  if (m.points.empty() || m.points.size() > kMaxKnots - 1) {
    return ChunkCheck::kCount;
  }
  c = PoseChunk{};
  c.relative = m.header.frame_id == tip;
  const double t0 = seconds(m.header.stamp);
  double prev = -std::numeric_limits<double>::infinity();
  for (const auto & p : m.points) {
    const double t = t0 + seconds(p.time_from_start);
    if (p.transforms.size() != 1 || !(t > prev) || !valid_transform(p.transforms[0])) {
      return ChunkCheck::kPoint;
    }
    c.t[c.n] = t;
    c.pose[c.n++] = to_se3(p.transforms[0]);
    prev = t;
  }
  return ChunkCheck::kOk;
}

enum class PathCheck
{
  kOk,
  kNames,   // a model joint missing from joint_names
  kSize,    // a point whose positions do not match joint_names
  kLimits,  // a point not finite or outside the joint limits
  kTimes,   // fewer than two points, or times not strictly increasing
  kLate,    // stamped, and its first point not on time (path_on_time): too far in the past or in the future
};

struct PathVerdict
{
  PathCheck check = PathCheck::kOk;
  std::size_t positions = 0;  // kSize: the bad point's position count
  double late = 0.0;          // kLate: how long ago it starts [s] (negative: in the future)
};

constexpr double kLimitTol = 1e-6;  // [rad] a planner's sample on a limit (park: j4 = 0) can end a hair outside it

// Fills p from m, its columns reordered to `names`. Unstamped (stamp 0): times from the path's start, which the loop
// re-bases. Stamped: wall clock times, checked against `now` here and against the loop's time again when taken.
inline PathVerdict parse_path(const trajectory_msgs::msg::JointTrajectory & m,
                              const std::array<std::string, kJoints> & names, const Vec & lower, const Vec & upper,
                              double now, JointPath & p)
{
  std::array<std::size_t, kJoints> col{};
  for (int k = 0; k < kJoints; ++k) {
    const auto it = std::find(m.joint_names.begin(), m.joint_names.end(), names[k]);
    if (it == m.joint_names.end()) {
      return {PathCheck::kNames};
    }
    col[k] = static_cast<std::size_t>(it - m.joint_names.begin());
  }
  for (const auto & pt : m.points) {
    if (pt.positions.size() != m.joint_names.size()) {
      return {PathCheck::kSize, pt.positions.size()};
    }
  }
  p = JointPath{};
  p.rebase = !(m.header.stamp.sec || m.header.stamp.nanosec);
  const double t0 = p.rebase ? 0.0 : seconds(m.header.stamp);
  for (const auto & pt : m.points) {
    Vec q, v = Vec::Zero();
    for (int k = 0; k < kJoints; ++k) {
      q[k] = pt.positions[col[k]];
      if (pt.velocities.size() == m.joint_names.size()) {
        v[k] = pt.velocities[col[k]];
      }
    }
    if (!q.allFinite() || !v.allFinite() || (q.array() < lower.array() - kLimitTol).any() ||
        (q.array() > upper.array() + kLimitTol).any()) {
      return {PathCheck::kLimits};
    }
    p.h.t.push_back(t0 + seconds(pt.time_from_start));
    p.h.q.push_back(q);
    p.h.v.push_back(v);
  }
  if (p.h.t.size() < 2 || !strictly_increasing(p.h.t.begin(), p.h.t.end())) {
    return {PathCheck::kTimes};
  }
  if (!p.rebase && !path_on_time(now - p.h.t.front())) {
    return {PathCheck::kLate, 0, now - p.h.t.front()};
  }
  return {};
}

enum class PathStart
{
  kOk,
  kLate,  // a stamped path whose first point is not on time at t (path_on_time): too late, or too early
  kFar,   // where it plays from at t is more than kMaxPathStartGap from the command
};

// Whether the loop at time t takes path p, with the command at q_cmd. Sets q, v, a to where p plays from at t: part way
// along for a stamped path a little late, its first point for a stamped path a little early or an unstamped one.
inline PathStart path_start(const JointPath & p, double t, const Vec & q_cmd, Vec & q, Vec & v, Vec & a)
{
  if (!p.rebase && !path_on_time(t - p.h.t.front())) {
    return PathStart::kLate;
  }
  p.h.sample(p.rebase ? 0.0 : t, q, v, a);
  return (q - q_cmd).cwiseAbs().maxCoeff() > kMaxPathStartGap ? PathStart::kFar : PathStart::kOk;
}

enum class TripCause
{
  kNone,
  kNotFinite,  // the measured position or a reference is NaN or infinite: never write it
  kLead,       // a joint's command leads its measurement by more than lead_max: stuck, blocked or relaxed
};

inline TripCause trip_cause(const Vec & q, const Vec & q_cmd, const Vec & v_cmd, const Vec & a_cmd, double lead_max)
{
  if (!q.allFinite() || !q_cmd.allFinite() || !v_cmd.allFinite() || !a_cmd.allFinite()) {
    return TripCause::kNotFinite;
  }
  return (q_cmd - q).cwiseAbs().maxCoeff() > lead_max ? TripCause::kLead : TripCause::kNone;
}
}  // namespace oap

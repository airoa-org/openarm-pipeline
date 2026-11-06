// Copyright 2026 AI Robot Association (AIRoA)
// SPDX-License-Identifier: Apache-2.0
// Motion's decisions on its inputs (inputs.hpp): chunk acceptance, a joint path's start, and the trip.
#include <array>
#include <cmath>
#include <limits>
#include <string>

#include <gtest/gtest.h>

#include "oap_control/inputs.hpp"

namespace
{
using oap::ChunkCheck;
using oap::PathCheck;
using oap::PathStart;
using oap::TripCause;
using oap::Vec;
using trajectory_msgs::msg::MultiDOFJointTrajectory;
using trajectory_msgs::msg::MultiDOFJointTrajectoryPoint;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
const std::string kTip = "left_hand_root";

builtin_interfaces::msg::Duration dur(double s)
{
  builtin_interfaces::msg::Duration d;
  d.sec = static_cast<int32_t>(std::floor(s));
  d.nanosec = static_cast<uint32_t>(std::lround((s - std::floor(s)) * 1e9));
  return d;
}

// n knots 0.1 s apart from a stamp of 100 s, moving 1 cm per knot along x, in `frame`.
MultiDOFJointTrajectory chunk(int n, const std::string & frame = kTip)
{
  MultiDOFJointTrajectory m;
  m.header.frame_id = frame;
  m.header.stamp.sec = 100;
  for (int k = 0; k < n; ++k) {
    MultiDOFJointTrajectoryPoint p;
    geometry_msgs::msg::Transform t;
    t.translation.x = 0.01 * (k + 1);
    t.rotation.w = 1.0;
    p.transforms.push_back(t);
    p.time_from_start = dur(0.1 * (k + 1));
    m.points.push_back(p);
  }
  return m;
}

oap::JointPath path(bool rebase, double t0)  // 0 → 1 rad on every joint over 2 s, at rest at both ends
{
  oap::JointPath p;
  p.rebase = rebase;
  p.h.t = {t0, t0 + 2.0};
  p.h.q = {Vec::Zero(), Vec::Constant(1.0)};
  p.h.v = {Vec::Zero(), Vec::Zero()};
  return p;
}
}  // namespace

TEST(Chunk, AcceptsTipAndWorldFrames)
{
  oap::PoseChunk c;
  ASSERT_EQ(oap::parse_chunk(chunk(10), kTip, c), ChunkCheck::kOk);
  EXPECT_TRUE(c.relative);
  EXPECT_EQ(c.n, 10);
  EXPECT_NEAR(c.t[0], 100.1, 1e-9);
  EXPECT_NEAR(c.t[9], 101.0, 1e-9);
  EXPECT_NEAR(c.pose[9].translation().x(), 0.1, 1e-12);
  ASSERT_EQ(oap::parse_chunk(chunk(3, "world"), kTip, c), ChunkCheck::kOk);
  EXPECT_FALSE(c.relative);
  EXPECT_EQ(c.n, 3);  // the earlier, longer chunk leaves nothing behind
}

TEST(Chunk, RejectsOtherFrames)
{
  oap::PoseChunk c;
  EXPECT_EQ(oap::parse_chunk(chunk(3, "right_hand_root"), kTip, c), ChunkCheck::kFrame);
  EXPECT_EQ(oap::parse_chunk(chunk(3, ""), kTip, c), ChunkCheck::kFrame);
}

TEST(Chunk, RejectsEmptyOrTooLong)
{
  oap::PoseChunk c;
  EXPECT_EQ(oap::parse_chunk(chunk(0), kTip, c), ChunkCheck::kCount);
  EXPECT_EQ(oap::parse_chunk(chunk(oap::kMaxKnots - 1), kTip, c), ChunkCheck::kOk);
  EXPECT_EQ(oap::parse_chunk(chunk(oap::kMaxKnots), kTip, c), ChunkCheck::kCount);
}

TEST(Chunk, RejectsABadPoint)
{
  oap::PoseChunk c;
  auto m = chunk(5);
  m.points[2].transforms[0].translation.y = kNaN;
  EXPECT_EQ(oap::parse_chunk(m, kTip, c), ChunkCheck::kPoint);
  m = chunk(5);
  m.points[4].transforms[0].rotation.w = 0.0;
  EXPECT_EQ(oap::parse_chunk(m, kTip, c), ChunkCheck::kPoint);
  m = chunk(5);
  m.points[1].transforms.clear();
  EXPECT_EQ(oap::parse_chunk(m, kTip, c), ChunkCheck::kPoint);
  m = chunk(5);
  m.points[1].transforms.push_back(m.points[1].transforms[0]);
  EXPECT_EQ(oap::parse_chunk(m, kTip, c), ChunkCheck::kPoint);
  m = chunk(5);
  m.points[3].time_from_start = m.points[2].time_from_start;  // equal times
  EXPECT_EQ(oap::parse_chunk(m, kTip, c), ChunkCheck::kPoint);
  m = chunk(5);
  m.points[3].time_from_start = dur(0.05);  // back in time
  EXPECT_EQ(oap::parse_chunk(m, kTip, c), ChunkCheck::kPoint);
}

TEST(PathStart, StampedOnTimeOrALittleLatePlays)
{
  Vec q, v, a;
  const auto p = path(false, 10.0);
  EXPECT_EQ(oap::path_start(p, 10.0 - oap::kMaxPathEarly + 1e-6, Vec::Zero(), q, v, a), PathStart::kOk);
  EXPECT_TRUE(q.isZero());  // a little early: from its first point
  EXPECT_EQ(oap::path_start(p, 10.0, Vec::Zero(), q, v, a), PathStart::kOk);
  EXPECT_EQ(oap::path_start(p, 10.0 + oap::kMaxPathLate - 1e-6, Vec::Zero(), q, v, a), PathStart::kOk);
  EXPECT_GT(q[0], 0.0);  // from part way along
  EXPECT_LT(q[0], oap::kMaxPathStartGap);
}

TEST(PathStart, StampedTooLateIsDropped)
{
  Vec q, v, a;
  const auto p = path(false, 10.0);
  EXPECT_EQ(oap::path_start(p, 10.0 + oap::kMaxPathLate + 1e-3, Vec::Zero(), q, v, a), PathStart::kLate);
  EXPECT_EQ(oap::path_start(p, 11.0, Vec::Constant(0.5), q, v, a), PathStart::kLate);  // even on the command
}

TEST(PathStart, StampedTooEarlyIsDropped)  // a clock ahead of ours, not a schedule
{
  Vec q, v, a;
  const auto p = path(false, 10.0);
  EXPECT_EQ(oap::path_start(p, 10.0 - oap::kMaxPathEarly - 1e-3, Vec::Zero(), q, v, a), PathStart::kLate);
  EXPECT_EQ(oap::path_start(p, 7.0, Vec::Zero(), q, v, a), PathStart::kLate);
  EXPECT_EQ(oap::path_start(path(true, 0.0), -1e6, Vec::Zero(), q, v, a), PathStart::kOk);  // unstamped: never
}

TEST(PathStart, UnstampedIsNeverLate)
{
  Vec q, v, a;
  const auto p = path(true, 0.0);
  EXPECT_EQ(oap::path_start(p, 1e6, Vec::Zero(), q, v, a), PathStart::kOk);
  EXPECT_TRUE(q.isZero());
}

TEST(PathStart, FarFromTheCommandIsDropped)
{
  Vec q, v, a;
  Vec cmd = Vec::Zero();
  cmd[13] = oap::kMaxPathStartGap + 1e-3;
  EXPECT_EQ(oap::path_start(path(true, 0.0), 5.0, cmd, q, v, a), PathStart::kFar);
  EXPECT_EQ(oap::path_start(path(false, 10.0), 10.0, cmd, q, v, a), PathStart::kFar);
  cmd[13] = oap::kMaxPathStartGap - 1e-3;
  EXPECT_EQ(oap::path_start(path(true, 0.0), 5.0, cmd, q, v, a), PathStart::kOk);
}

TEST(Trip, NoneOnAFiniteCommandWithinTheLead)
{
  const Vec q = Vec::Constant(0.3);
  EXPECT_EQ(oap::trip_cause(q, q + Vec::Constant(0.1), Vec::Zero(), Vec::Zero(), 0.15), TripCause::kNone);
}

TEST(Trip, OnALead)
{
  const Vec q = Vec::Zero();
  Vec cmd = q;
  cmd[6] = -0.2;
  EXPECT_EQ(oap::trip_cause(q, cmd, Vec::Zero(), Vec::Zero(), 0.15), TripCause::kLead);
}

TEST(Trip, OnAnyNonFiniteValue)  // the lead check alone is false for NaN
{
  for (int which = 0; which < 4; ++which) {
    for (const double bad : {kNaN, std::numeric_limits<double>::infinity()}) {
      Vec vs[4] = {Vec::Zero(), Vec::Zero(), Vec::Zero(), Vec::Zero()};  // q, q_cmd, v_cmd, a_cmd
      vs[which][3] = bad;
      EXPECT_EQ(oap::trip_cause(vs[0], vs[1], vs[2], vs[3], 0.15), TripCause::kNotFinite) << which;
    }
  }
}

TEST(Trip, HoldsTheLastFiniteCommandOnANonFiniteMeasurement)
{
  Vec q = Vec::Constant(0.2);
  const Vec last = Vec::Constant(0.1);
  EXPECT_EQ(oap::hold_position(q, last), q);
  q[0] = kNaN;
  EXPECT_EQ(oap::hold_position(q, last), last);
  // the held command is finite, so the trip goes on only while the measurement stays non-finite
  EXPECT_EQ(oap::trip_cause(q, oap::hold_position(q, last), Vec::Zero(), Vec::Zero(), 0.15), TripCause::kNotFinite);
  q[0] = 0.1;
  EXPECT_EQ(oap::trip_cause(q, oap::hold_position(q, last), Vec::Zero(), Vec::Zero(), 0.15), TripCause::kNone);
}

namespace
{
using trajectory_msgs::msg::JointTrajectory;
using trajectory_msgs::msg::JointTrajectoryPoint;

std::array<std::string, oap::kJoints> names()
{
  std::array<std::string, oap::kJoints> n;
  for (int k = 0; k < oap::kJoints; ++k) {
    n[k] = "openarm_" + oap::kSides[k / oap::kDof] + "_joint" + std::to_string(k % oap::kDof + 1);
  }
  return n;
}

const Vec kLower = Vec::Constant(-1.0), kUpper = Vec::Constant(1.0);

// Point i at 0.5 i s, model joint k at 0.01 k + 0.1 i rad; its joint names in reverse model order, plus an extra joint
// the model does not have (a path may name more joints).
JointTrajectory joint_path(int n = 3)
{
  JointTrajectory m;
  const auto model = names();
  for (int k = oap::kJoints - 1; k >= 0; --k) {
    m.joint_names.push_back(model[k]);
  }
  m.joint_names.push_back("openarm_left_finger_joint1");
  for (int i = 0; i < n; ++i) {
    JointTrajectoryPoint pt;
    for (int c = 0; c < oap::kJoints; ++c) {
      const int k = oap::kJoints - 1 - c;  // column c is model joint k
      pt.positions.push_back(0.01 * k + 0.1 * i);
    }
    pt.positions.push_back(9.0);  // the extra joint: not checked against the arm's limits
    pt.time_from_start = dur(0.5 * i);
    m.points.push_back(pt);
  }
  return m;
}
}  // namespace

TEST(Path, UnstampedIsReorderedAndRebased)
{
  oap::JointPath p;
  const auto r = oap::parse_path(joint_path(), names(), kLower, kUpper, 1e6, p);
  ASSERT_EQ(r.check, PathCheck::kOk);
  EXPECT_TRUE(p.rebase);
  ASSERT_EQ(p.h.t.size(), 3u);
  EXPECT_DOUBLE_EQ(p.h.t[0], 0.0);  // from the path's start, whatever now is
  EXPECT_DOUBLE_EQ(p.h.t[2], 1.0);
  for (int k = 0; k < oap::kJoints; ++k) {
    EXPECT_DOUBLE_EQ(p.h.q[2][k], 0.01 * k + 0.2) << k;
  }
  EXPECT_TRUE(p.h.v[1].isZero());  // no velocities given
}

TEST(Path, VelocitiesAreTakenOnlyForEveryJoint)
{
  auto m = joint_path();
  for (auto & pt : m.points) {
    pt.velocities.assign(m.joint_names.size(), 0.0);
    pt.velocities[0] = 0.5;  // column 0 is right joint 7, model joint 13
  }
  oap::JointPath p;
  ASSERT_EQ(oap::parse_path(m, names(), kLower, kUpper, 0.0, p).check, PathCheck::kOk);
  EXPECT_DOUBLE_EQ(p.h.v[0][13], 0.5);
  for (auto & pt : m.points) {
    pt.velocities.resize(3);  // partial: ignored, as before
  }
  ASSERT_EQ(oap::parse_path(m, names(), kLower, kUpper, 0.0, p).check, PathCheck::kOk);
  EXPECT_TRUE(p.h.v[0].isZero());
}

TEST(Path, StampedOnTimeOrALittleLateIsTaken)
{
  auto m = joint_path();
  m.header.stamp.sec = 100;
  oap::JointPath p;
  ASSERT_EQ(oap::parse_path(m, names(), kLower, kUpper, 100.0, p).check, PathCheck::kOk);
  EXPECT_FALSE(p.rebase);
  EXPECT_DOUBLE_EQ(p.h.t[0], 100.0);
  EXPECT_EQ(oap::parse_path(m, names(), kLower, kUpper, 100.0 + oap::kMaxPathLate - 1e-6, p).check, PathCheck::kOk);
  m.header.stamp.sec = 0;
  m.header.stamp.nanosec = 1;  // any non-zero stamp counts as stamped
  ASSERT_EQ(oap::parse_path(m, names(), kLower, kUpper, 0.0, p).check, PathCheck::kOk);
  EXPECT_FALSE(p.rebase);
}

TEST(Path, StampedTooLateIsDropped)
{
  auto m = joint_path();
  m.header.stamp.sec = 100;
  oap::JointPath p;
  const auto r = oap::parse_path(m, names(), kLower, kUpper, 101.0, p);
  EXPECT_EQ(r.check, PathCheck::kLate);
  EXPECT_DOUBLE_EQ(r.late, 1.0);
}

TEST(Path, StampedTooEarlyIsDropped)
{
  auto m = joint_path();
  m.header.stamp.sec = 100;
  oap::JointPath p;
  EXPECT_EQ(oap::parse_path(m, names(), kLower, kUpper, 100.0 - oap::kMaxPathEarly + 1e-6, p).check, PathCheck::kOk);
  const auto r = oap::parse_path(m, names(), kLower, kUpper, 97.0, p);  // the sender's clock 3 s ahead
  EXPECT_EQ(r.check, PathCheck::kLate);
  EXPECT_DOUBLE_EQ(r.late, -3.0);
}

TEST(Path, AMissingJointIsDropped)
{
  auto m = joint_path();
  m.joint_names[3] = "nobody";
  oap::JointPath p;
  EXPECT_EQ(oap::parse_path(m, names(), kLower, kUpper, 0.0, p).check, PathCheck::kNames);
}

TEST(Path, APointWithTheWrongPositionCountIsDropped)  // this used to throw in the callback
{
  oap::JointPath p;
  for (const std::size_t n : {std::size_t{0}, std::size_t{14}, std::size_t{16}}) {
    auto m = joint_path();
    m.points[1].positions.resize(n);
    const auto r = oap::parse_path(m, names(), kLower, kUpper, 0.0, p);
    EXPECT_EQ(r.check, PathCheck::kSize) << n;
    EXPECT_EQ(r.positions, n);
  }
}

TEST(Path, APointOutsideTheLimitsOrNotFiniteIsDropped)
{
  oap::JointPath p;
  auto m = joint_path();
  m.points[2].positions[0] = 1.0 + oap::kLimitTol / 2;  // a hair outside: accepted
  EXPECT_EQ(oap::parse_path(m, names(), kLower, kUpper, 0.0, p).check, PathCheck::kOk);
  m.points[2].positions[0] = 1.0 + 2 * oap::kLimitTol;
  EXPECT_EQ(oap::parse_path(m, names(), kLower, kUpper, 0.0, p).check, PathCheck::kLimits);
  m = joint_path();
  m.points[0].positions[5] = -1.1;
  EXPECT_EQ(oap::parse_path(m, names(), kLower, kUpper, 0.0, p).check, PathCheck::kLimits);
  m = joint_path();
  m.points[1].positions[2] = kNaN;
  EXPECT_EQ(oap::parse_path(m, names(), kLower, kUpper, 0.0, p).check, PathCheck::kLimits);
  m = joint_path();
  m.points[1].velocities.assign(m.joint_names.size(), 0.0);
  m.points[1].velocities[4] = kNaN;
  EXPECT_EQ(oap::parse_path(m, names(), kLower, kUpper, 0.0, p).check, PathCheck::kLimits);
}

TEST(Path, TooFewPointsOrBadTimesAreDropped)
{
  oap::JointPath p;
  EXPECT_EQ(oap::parse_path(joint_path(1), names(), kLower, kUpper, 0.0, p).check, PathCheck::kTimes);
  EXPECT_EQ(oap::parse_path(joint_path(0), names(), kLower, kUpper, 0.0, p).check, PathCheck::kTimes);
  auto m = joint_path();
  m.points[2].time_from_start = m.points[1].time_from_start;  // h = 0: two points at the same time
  EXPECT_EQ(oap::parse_path(m, names(), kLower, kUpper, 0.0, p).check, PathCheck::kTimes);
  m = joint_path();
  m.points[0].time_from_start = dur(2.0);
  EXPECT_EQ(oap::parse_path(m, names(), kLower, kUpper, 0.0, p).check, PathCheck::kTimes);
}

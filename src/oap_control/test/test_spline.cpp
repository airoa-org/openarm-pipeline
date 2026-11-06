// Copyright 2026 AI Robot Association (AIRoA)
// SPDX-License-Identifier: Apache-2.0
// The input checks (validate.hpp) and the splines they protect (spline.hpp).
#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include <pinocchio/spatial/explog.hpp>

#include "oap_control/spline.hpp"
#include "oap_control/validate.hpp"

namespace
{
using oap::Vec;
using oap::Vec6;
using pinocchio::SE3;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

geometry_msgs::msg::Transform unit()
{
  geometry_msgs::msg::Transform t;
  t.translation.x = 0.3;
  t.translation.y = -0.2;
  t.translation.z = 0.1;
  t.rotation.w = 1.0;
  return t;
}
}  // namespace

TEST(Validate, UnitTransformPasses)
{
  EXPECT_TRUE(oap::valid_transform(unit()));
  auto t = unit();
  t.rotation.w = t.rotation.z = 0.7071;  // four decimals
  EXPECT_TRUE(oap::valid_transform(t));
}

TEST(Validate, AnyNonFiniteComponentFails)
{
  for (int i = 0; i < 7; ++i) {
    for (const double bad : {kNaN, std::numeric_limits<double>::infinity()}) {
      auto t = unit();
      double * c[7] = {&t.translation.x, &t.translation.y, &t.translation.z, &t.rotation.x,
                       &t.rotation.y,    &t.rotation.z,    &t.rotation.w};
      *c[i] = bad;
      EXPECT_FALSE(oap::valid_transform(t)) << "component " << i;
    }
  }
}

TEST(Validate, NonUnitQuaternionFails)
{
  auto t = unit();
  t.rotation.w = 0.0;
  EXPECT_FALSE(oap::valid_transform(t));
  t.rotation.w = 1.1;
  EXPECT_FALSE(oap::valid_transform(t));
}

TEST(Validate, StrictlyIncreasing)
{
  const std::vector<double> ok{0.0, 0.1, 0.2}, equal{0.0, 0.1, 0.1}, back{0.0, 0.2, 0.1}, nan{0.0, kNaN, 0.2},
    one{kNaN}, empty{};
  EXPECT_TRUE(oap::strictly_increasing(ok.begin(), ok.end()));
  EXPECT_FALSE(oap::strictly_increasing(equal.begin(), equal.end()));
  EXPECT_FALSE(oap::strictly_increasing(back.begin(), back.end()));
  EXPECT_FALSE(oap::strictly_increasing(nan.begin(), nan.end()));
  EXPECT_FALSE(oap::strictly_increasing(one.begin(), one.end()));
  EXPECT_TRUE(oap::strictly_increasing(empty.begin(), empty.end()));
}

TEST(Hermite, InterpolatesAndHoldsAtTheEnd)
{
  oap::Hermite h;
  h.t = {0.0, 1.0};
  h.q = {Vec::Zero(), Vec::Constant(1.0)};
  h.v = {Vec::Zero(), Vec::Zero()};
  Vec q, v, a;
  h.sample(0.5, q, v, a);
  EXPECT_NEAR(q[0], 0.5, 1e-12);
  EXPECT_NEAR(v[0], 1.5, 1e-12);  // the peak of a rest-to-rest cubic: 3/2 the mean
  EXPECT_TRUE(q.allFinite() && v.allFinite() && a.allFinite());
  h.sample(2.0, q, v, a);
  EXPECT_NEAR(q[3], 1.0, 1e-12);
  EXPECT_TRUE(v.isZero() && a.isZero());
}

TEST(Spline, StartsAtTheStartAndEndsAtTheLastKnot)
{
  oap::Spline s;
  const SE3 start = SE3::Identity();
  const double t[3] = {0.1, 0.2, 0.3};
  SE3 pose[3];
  for (int k = 0; k < 3; ++k) {
    pose[k] = SE3(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0.01 * (k + 1), 0.0, 0.0));
  }
  ASSERT_TRUE(s.fit(0.0, start, Vec6::Zero(), t, pose, 3));
  EXPECT_DOUBLE_EQ(s.end(), 0.3);
  SE3 out;
  Vec6 twist;
  s.sample(0.0, out, twist);
  EXPECT_LT(out.translation().norm(), 1e-12);  // isApprox is relative, so useless against zero
  EXPECT_TRUE(out.rotation().isApprox(start.rotation(), 1e-12));
  for (double ts = 0.0; ts <= 0.35; ts += 0.001) {
    s.sample(ts, out, twist);
    ASSERT_TRUE(out.translation().allFinite() && twist.allFinite()) << ts;
  }
  s.sample(0.4, out, twist);
  EXPECT_TRUE(out.isApprox(pose[2], 1e-12));
  EXPECT_TRUE(twist.isZero());
}

TEST(Spline, NoKnotDueLeavesItUnchanged)
{
  oap::Spline s;
  const double t[2] = {0.1, 0.2};
  const SE3 pose[2] = {SE3(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0.01, 0.0, 0.0)),
                       SE3(Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitZ()).toRotationMatrix(),
                           Eigen::Vector3d(0.02, 0.0, 0.0))};
  ASSERT_TRUE(s.fit(0.0, SE3::Identity(), Vec6::Zero(), t, pose, 2));
  const double ts[4] = {0.0, 0.05, 0.15, 0.3};
  SE3 before[4], out;
  Vec6 twist_before[4], twist;
  for (int i = 0; i < 4; ++i) {
    s.sample(ts[i], before[i], twist_before[i]);
  }
  // A different start, twist and knot, none of it due after t0 = 1.0.
  const double late[1] = {0.5};
  const SE3 other[1] = {SE3(Eigen::Matrix3d::Identity(), Eigen::Vector3d(1.0, 1.0, 1.0))};
  const SE3 start(Eigen::AngleAxisd(1.0, Eigen::Vector3d::UnitX()).toRotationMatrix(), Eigen::Vector3d(0.5, 0.5, 0.5));
  EXPECT_FALSE(s.fit(1.0, start, Vec6::Constant(1.0), late, other, 1));
  EXPECT_DOUBLE_EQ(s.end(), 0.2);
  for (int i = 0; i < 4; ++i) {
    s.sample(ts[i], out, twist);
    EXPECT_EQ(out.translation(), before[i].translation()) << ts[i];
    EXPECT_EQ(out.rotation(), before[i].rotation()) << ts[i];
    EXPECT_EQ(twist, twist_before[i]) << ts[i];
  }
}

// 9af2725: a first knot due within half a knot spacing of the start is dropped, so it cannot bend the start hard.
class FirstKnot : public ::testing::Test
{
protected:
  void fit(double t0)
  {
    for (int k = 0; k < 4; ++k) {
      t[k] = 0.1 * (k + 1);  // 0.1 s spacing: half a spacing is 0.05 s
      pose[k] = SE3(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0.01 * (k + 1), 0.0, 0.2));
    }
    ASSERT_TRUE(s.fit(t0, start, Vec6::Zero(), t, pose, 4));
  }
  double gap(double ts, const SE3 & want)
  {
    SE3 out;
    Vec6 twist;
    s.sample(ts, out, twist);
    return (out.translation() - want.translation()).norm() +
           pinocchio::log3(Eigen::Matrix3d(out.rotation().transpose() * want.rotation())).norm();
  }
  oap::Spline s;
  const SE3 start{Eigen::Matrix3d::Identity(), Eigen::Vector3d(0.0, 0.0, 0.2)};
  double t[4];
  SE3 pose[4];
};

TEST_F(FirstKnot, KeptWhenDueAfterHalfASpacing)
{
  fit(0.04);  // the first knot is due 0.06 s after the start
  EXPECT_EQ(s.knots(), 5);
  EXPECT_LT(gap(0.04, start), 1e-12);
  EXPECT_LT(gap(0.1, pose[0]), 1e-12);
  EXPECT_DOUBLE_EQ(s.end(), 0.4);
}

TEST_F(FirstKnot, DroppedWhenDueWithinHalfASpacing)
{
  fit(0.06);  // the first knot is due 0.04 s after the start
  EXPECT_EQ(s.knots(), 4);
  EXPECT_LT(gap(0.06, start), 1e-12);
  EXPECT_GT(gap(0.1, pose[0]), 1e-6);  // no longer forced through it
  EXPECT_LT(gap(0.2, pose[1]), 1e-12);
  EXPECT_DOUBLE_EQ(s.end(), 0.4);
}

TEST_F(FirstKnot, TheOnlyKnotLeftIsKept)
{
  oap::Spline one;
  const double tl[1] = {0.1};
  const SE3 pl[1] = {SE3(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0.01, 0.0, 0.2))};
  ASSERT_TRUE(one.fit(0.09, start, Vec6::Zero(), tl, pl, 1));
  EXPECT_EQ(one.knots(), 2);
}

TEST(Validate, HoldOnAFiniteMeasurementOrTheLastFiniteCommand)
{
  const Vec measured = Vec::Constant(0.2), last = Vec::Constant(0.1);
  EXPECT_EQ(oap::hold_position(measured, last), measured);
  Vec bad = measured;
  bad[5] = kNaN;
  EXPECT_EQ(oap::hold_position(bad, last), last);
}

TEST(Hermite, ALateStartSamplesPartWayAlong)  // what the path start check compares with the command
{
  oap::Hermite h;
  h.t = {10.0, 12.0};
  h.q = {Vec::Zero(), Vec::Constant(1.0)};
  h.v = {Vec::Zero(), Vec::Zero()};
  Vec q, v, a;
  h.sample(10.05, q, v, a);  // 50 ms late on a 2 s rest-to-rest move: a small step
  EXPECT_GT(q[0], 0.0);
  EXPECT_LT(q[0], 0.01);
  h.sample(11.0, q, v, a);  // 1 s late: half way
  EXPECT_NEAR(q[0], 0.5, 1e-12);
}

TEST(Validate, PathOnTime)  // seconds since a stamped path's start: negative is in the future
{
  EXPECT_TRUE(oap::path_on_time(0.0));
  EXPECT_TRUE(oap::path_on_time(oap::kMaxPathLate));
  EXPECT_TRUE(oap::path_on_time(-oap::kMaxPathEarly));
  EXPECT_FALSE(oap::path_on_time(1.0));
  EXPECT_FALSE(oap::path_on_time(-3.0));  // the sender's clock seconds ahead
  EXPECT_FALSE(oap::path_on_time(kNaN));
}

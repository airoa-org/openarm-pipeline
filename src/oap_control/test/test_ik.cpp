// Copyright 2026 AI Robot Association (AIRoA)
// SPDX-License-Identifier: Apache-2.0
// The DLS IK on the installed URDF: from a start near a random in-limit pose, stepping at 1 kHz toward that pose's
// hands reaches them, with the command inside the joint limits less the margin throughout. The parameters are the
// motion controller's, read from config/controllers.yaml.
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <regex>
#include <string>

#include <gtest/gtest.h>
#include <pinocchio/spatial/explog.hpp>

#include "oap_control/ik.hpp"
#include "oap_control/model.hpp"

namespace
{
using oap::kJoints;
using oap::Vec;
using pinocchio::SE3;

constexpr double kDt = 1e-3;

// The motion controller's scalar parameters from the package's config/controllers.yaml (the file the launch installs),
// so a retune there retunes the test: `key: number` lines under `motion:`, up to the next top level key.
std::map<std::string, double> motion_params()
{
  std::ifstream in(OAP_CONTROLLERS_YAML);
  std::map<std::string, double> out;
  const std::regex top(R"(^(\w+):\s*$)"), scalar(R"(^\s+(\w+):\s*([-+0-9.eE]+)\s*(#.*)?$)");
  bool inside = false;
  for (std::string line; std::getline(in, line);) {
    std::smatch m;
    if (std::regex_match(line, m, top)) {
      inside = m[1] == "motion";
    } else if (inside && std::regex_match(line, m, scalar)) {
      out[m[1]] = std::stod(m[2]);
    }
  }
  return out;
}

class IkRoundTrip : public ::testing::Test
{
protected:
  static void SetUpTestSuite()
  {
    model_ = std::make_unique<oap::Model>(oap::read_file(oap::default_urdf()));
    const auto p = motion_params();
    for (const auto * key : {"max_joint_velocity", "max_joint_acceleration", "ik_damping", "limit_margin"}) {
      ASSERT_EQ(p.count(key), 1u) << key << " not found under motion: in " << OAP_CONTROLLERS_YAML;
    }
    v_max_ = p.at("max_joint_velocity");
    a_max_ = p.at("max_joint_acceleration");
    damping_ = p.at("ik_damping");
    margin_ = p.at("limit_margin");
  }
  static void TearDownTestSuite() { model_.reset(); }
  void SetUp() override { ASSERT_GT(v_max_, 0.0) << "controllers.yaml was not read"; }

  // q inside the limits less the margin, which the IK keeps up to a discrete overshoot: braking at a_max, its
  // acceleration bound overrides the braking bound sqrt(2 a d) on the last steps. StopsAtTheMargin settles 1.6e-4 rad
  // past the margin (0.0198 rad inside the URDF limit); allow about three times that.
  void expect_inside(const Vec & q, const std::string & where) const
  {
    constexpr double slack = 5e-4;  // [rad]
    for (int k = 0; k < kJoints; ++k) {
      ASSERT_GE(q[k], model_->lower()[k] + margin_ - slack) << where << " " << model_->names()[k];
      ASSERT_LE(q[k], model_->upper()[k] - margin_ + slack) << where << " " << model_->names()[k];
    }
  }

  static std::unique_ptr<oap::Model> model_;
  static double v_max_, a_max_, damping_, margin_;
};
std::unique_ptr<oap::Model> IkRoundTrip::model_;
double IkRoundTrip::v_max_ = 0.0, IkRoundTrip::a_max_ = 0.0, IkRoundTrip::damping_ = 0.0, IkRoundTrip::margin_ = 0.0;

double hand_error(oap::Model & m, const Vec & q, const std::array<SE3, 2> & target)
{
  m.set(q);
  double e = 0.0;
  for (int s = 0; s < 2; ++s) {
    e = std::max(e, pinocchio::log6(m.hand(s).actInv(target[s])).toVector().norm());
  }
  return e;
}
}  // namespace

TEST_F(IkRoundTrip, ConvergesInsideTheLimits)
{
  oap::Model & m = *model_;
  oap::Ik ik(m, v_max_, a_max_, damping_, margin_);
  std::mt19937 rng(73);  // fixed: the same poses every run
  std::uniform_real_distribution<double> unit(0.0, 1.0), offset(-0.1, 0.1);
  constexpr int kTrials = 20, kMaxSteps = 10000;  // 10 s each at 1 kHz
  const double edge = margin_ + 0.13;  // [rad] the goal this far inside the limits, so a 0.1 rad start is inside too
  for (int trial = 0; trial < kTrials; ++trial) {
    Vec goal, q;
    for (int k = 0; k < kJoints; ++k) {
      const double lo = m.lower()[k] + edge, hi = m.upper()[k] - edge;
      ASSERT_LT(lo, hi) << m.names()[k];
      goal[k] = lo + unit(rng) * (hi - lo);
      q[k] = std::clamp(goal[k] + offset(rng), m.lower()[k] + margin_, m.upper()[k] - margin_);
    }
    m.set(goal);
    const std::array<SE3, 2> target{m.hand(0), m.hand(1)};
    Vec v = Vec::Zero(), a = Vec::Zero();
    const double e0 = hand_error(m, q, target);
    int steps = 0;
    for (; steps < kMaxSteps && hand_error(m, q, target) > 1e-5; ++steps) {
      ik.step(target, kDt, q, v, a);
      const std::string where = "trial " + std::to_string(trial) + " step " + std::to_string(steps);
      ASSERT_TRUE(q.allFinite() && v.allFinite() && a.allFinite()) << where;
      ASSERT_NO_FATAL_FAILURE(expect_inside(q, where));
      ASSERT_LE(v.cwiseAbs().maxCoeff(), v_max_ + 1e-12) << where;
    }
    EXPECT_GT(steps, 0);
    EXPECT_LE(hand_error(m, q, target), 1e-5) << "trial " << trial << ": from " << e0 << " after " << steps
                                              << " steps";
  }
}

// A goal whose elbow sits inside the margin: the IK drives the elbow to the margin and stops there.
TEST_F(IkRoundTrip, StopsAtTheMargin)
{
  oap::Model & m = *model_;
  oap::Ik ik(m, v_max_, a_max_, damping_, margin_);
  constexpr int kElbow = 3;  // left joint 4: the only joint that sets the shoulder to wrist distance
  Vec goal = 0.5 * (m.lower() + m.upper());
  goal[kElbow] = m.upper()[kElbow] - margin_ / 4;
  Vec q = goal;
  q[kElbow] = m.upper()[kElbow] - margin_ - 0.1;
  m.set(goal);
  const std::array<SE3, 2> target{m.hand(0), m.hand(1)};
  Vec v = Vec::Zero(), a = Vec::Zero();
  for (int i = 0; i < 5000; ++i) {
    ik.step(target, kDt, q, v, a);
    ASSERT_NO_FATAL_FAILURE(expect_inside(q, "step " + std::to_string(i)));
  }
  EXPECT_NEAR(q[kElbow], m.upper()[kElbow] - margin_, 2e-3);
}

TEST_F(IkRoundTrip, AtTheTargetItStaysPut)
{
  oap::Model & m = *model_;
  oap::Ik ik(m, v_max_, a_max_, damping_, margin_);
  Vec q = 0.5 * (m.lower() + m.upper());
  m.set(q);
  const std::array<SE3, 2> target{m.hand(0), m.hand(1)};
  const Vec q0 = q;
  Vec v = Vec::Zero(), a = Vec::Zero();
  for (int i = 0; i < 1000; ++i) {
    ik.step(target, kDt, q, v, a);
  }
  EXPECT_LT((q - q0).cwiseAbs().maxCoeff(), 1e-9);
  EXPECT_LT(v.cwiseAbs().maxCoeff(), 1e-9);
}


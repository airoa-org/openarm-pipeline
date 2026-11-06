// Copyright 2026 AI Robot Association (AIRoA)
// SPDX-License-Identifier: Apache-2.0
// Trajectories sampled once per update: a hand pose spline through a chunk's knots, and a joint path for both arms.
#pragma once

#include <array>
#include <vector>

#include <pinocchio/spatial/se3.hpp>

#include "oap_control/model.hpp"

namespace oap
{
constexpr int kMaxKnots = 64;  // a policy chunk holds 16 rows, a test chunk 10

// A clamped C² cubic spline through timed hand poses: position, and the rotation vector from the start rotation.
class Spline
{
public:
  // Knot 0 is `start` at t0 with `twist` (in the start frame), then the knots due after t0, less the first when it is
  // due within half a knot spacing: so close a knot bends the start hard. Ends at rest. False when no knot is due, and
  // the spline is unchanged.
  bool fit(double t0, const pinocchio::SE3 & start, const Vec6 & twist, const double * t, const pinocchio::SE3 * pose,
           int n);
  void sample(double t, pinocchio::SE3 & pose, Vec6 & twist) const;  // twist in the sampled pose's frame
  double end() const { return t_[n_ - 1]; }
  int knots() const { return n_; }  // the start knot included

private:
  int n_ = 0;
  std::array<double, kMaxKnots> t_{};
  std::array<Vec6, kMaxKnots> y_{}, m_{};  // values and second derivatives
  Eigen::Matrix3d R0_ = Eigen::Matrix3d::Identity();
};

// Cubic Hermite between timed joint positions and velocities.
struct Hermite
{
  std::vector<double> t;
  std::vector<Vec> q, v;

  void sample(double s, Vec & q_out, Vec & v_out, Vec & a_out) const;
};
}  // namespace oap

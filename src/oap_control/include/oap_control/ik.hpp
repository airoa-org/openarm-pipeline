// Copyright 2026 AI Robot Association (AIRoA)
// SPDX-License-Identifier: Apache-2.0
// One damped least squares step per update from the joint command toward both hands' targets.
#pragma once

#include <array>

#include <pinocchio/spatial/se3.hpp>

#include "oap_control/model.hpp"

namespace oap
{
class Ik
{
public:
  Ik(Model & model, double v_max, double a_max, double damping, double margin)
  : model_(model), v_max_(v_max), a_max_(a_max), damping_(damping), margin_(margin)
  {
  }

  // Advances q and v by one step of dt and sets a to zero. Per joint: the target's own motion (twist, in the hand frame,
  // through the same damped inverse) plus the speed that still stops at the step's end, inside the position limits (less
  // margin), bounded by v_max. a_max shapes the approach and the braking at the limits, not the step's change of v.
  void step(const std::array<pinocchio::SE3, 2> & target, const std::array<Vec6, 2> & twist, double dt, Vec & q, Vec & v,
            Vec & a);
  void step(const std::array<pinocchio::SE3, 2> & target, double dt, Vec & q, Vec & v, Vec & a)
  {
    step(target, {Vec6::Zero(), Vec6::Zero()}, dt, q, v, a);
  }

private:
  Model & model_;
  double v_max_, a_max_, damping_, margin_;
};
}  // namespace oap

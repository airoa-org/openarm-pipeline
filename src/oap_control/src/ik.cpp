// Copyright 2026 AI Robot Association (AIRoA)
// SPDX-License-Identifier: Apache-2.0
#include "oap_control/ik.hpp"

#include <algorithm>
#include <cmath>

#include <pinocchio/spatial/explog.hpp>

namespace oap
{
void Ik::step(const std::array<pinocchio::SE3, 2> & target, const std::array<Vec6, 2> & twist, double dt, Vec & q,
              Vec & v, Vec & a)
{
  model_.set(q);
  Vec dq = Vec::Zero(), ff = Vec::Zero();
  for (int s = 0; s < 2; ++s) {
    const Jac J = model_.jacobian(s);
    const Vec6 err = pinocchio::log6(model_.hand(s).actInv(target[s])).toVector();
    const Eigen::Matrix<double, 6, 6> A = J * J.transpose() + damping_ * damping_ * Eigen::Matrix<double, 6, 6>::Identity();
    const auto solver = A.ldlt();
    dq.segment<kDof>(s * kDof) = J.transpose() * solver.solve(err);
    ff.segment<kDof>(s * kDof) = J.transpose() * solver.solve(twist[s]);
  }
  for (int k = 0; k < kJoints; ++k) {
    const double lo = model_.lower()[k] + margin_, hi = model_.upper()[k] - margin_;
    double vk = ff[k] + std::copysign(std::min(std::abs(dq[k]) / dt, std::sqrt(2.0 * a_max_ * std::abs(dq[k]))), dq[k]);
    vk = std::clamp(vk, -std::sqrt(2.0 * a_max_ * std::max(0.0, q[k] - lo)),
                    std::sqrt(2.0 * a_max_ * std::max(0.0, hi - q[k])));
    v[k] = std::clamp(vk, -v_max_, v_max_);  // no a_max per step: with it, clipped turns overshot
  }
  a.setZero();  // no inertial feedforward from the IK path
  q += v * dt;
}
}  // namespace oap

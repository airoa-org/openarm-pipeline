// Copyright 2026 AI Robot Association (AIRoA)
// SPDX-License-Identifier: Apache-2.0
#include "oap_control/spline.hpp"

#include <algorithm>

#include <pinocchio/spatial/explog.hpp>

namespace oap
{
using pinocchio::SE3;

bool Spline::fit(double t0, const SE3 & start, const Vec6 & twist, const double * t, const SE3 * pose, int n)
{
  int first = static_cast<int>(std::find_if(t, t + n, [t0](double tk) { return tk > t0 + 1e-4; }) - t);
  if (first == n) {
    return false;
  }
  if (first < n - 1 && t[first] - t0 < 0.5 * (t[n - 1] - t[0]) / (n - 1)) {
    ++first;
  }
  R0_ = start.rotation();
  n_ = 0;
  t_[n_] = t0;
  y_[n_++] << start.translation(), Eigen::Vector3d::Zero();
  for (int k = first; k < n; ++k) {
    t_[n_] = t[k];
    y_[n_++] << pose[k].translation(), pinocchio::log3(Eigen::Matrix3d(R0_.transpose() * pose[k].rotation()));
  }
  Vec6 s0;
  s0 << R0_ * twist.head<3>(), twist.tail<3>();  // at knot 0 the rotation vector's rate is ω in the start frame

  // Start slope s0, end slope 0; the Thomas algorithm on the tridiagonal system for the second derivatives.
  std::array<double, kMaxKnots> c{};
  std::array<Vec6, kMaxKnots> d{};
  const int last = n_ - 1;
  for (int i = 0; i <= last; ++i) {
    const double hl = i > 0 ? t_[i] - t_[i - 1] : 0.0, hr = i < last ? t_[i + 1] - t_[i] : 0.0;
    const Vec6 sl = i > 0 ? Vec6((y_[i] - y_[i - 1]) / hl) : s0;
    const Vec6 sr = i < last ? Vec6((y_[i + 1] - y_[i]) / hr) : Vec6::Zero();
    const double b = 2.0 * (hl + hr);
    const Vec6 rhs = 6.0 * (sr - sl);
    const double denom = b - (i > 0 ? hl * c[i - 1] : 0.0);
    c[i] = hr / denom;
    d[i] = (rhs - (i > 0 ? Vec6(hl * d[i - 1]) : Vec6::Zero())) / denom;
  }
  m_[last] = d[last];
  for (int i = last - 1; i >= 0; --i) {
    m_[i] = d[i] - c[i] * m_[i + 1];
  }
  return true;
}

void Spline::sample(double s, SE3 & pose, Vec6 & twist) const
{
  Vec6 value, slope = Vec6::Zero();
  if (s >= t_[n_ - 1]) {
    value = y_[n_ - 1];
  } else {
    s = std::max(s, t_[0]);
    int i = 0;
    while (t_[i + 1] < s) {
      ++i;
    }
    const double h = t_[i + 1] - t_[i], a = t_[i + 1] - s, b = s - t_[i];
    value = m_[i] * (a * a * a / (6 * h)) + m_[i + 1] * (b * b * b / (6 * h)) + (y_[i] / h - m_[i] * (h / 6)) * a +
            (y_[i + 1] / h - m_[i + 1] * (h / 6)) * b;
    slope = -m_[i] * (a * a / (2 * h)) + m_[i + 1] * (b * b / (2 * h)) - (y_[i] / h - m_[i] * (h / 6)) +
            (y_[i + 1] / h - m_[i + 1] * (h / 6));
  }
  pose = SE3(R0_ * pinocchio::exp3(value.tail<3>()), value.head<3>());
  const Eigen::Matrix3d Rt = pose.rotation().transpose();
  twist << Rt * slope.head<3>(), Rt * R0_ * slope.tail<3>();  // ≈ ω: the rotation vector's rate, rotated
}

void Hermite::sample(double s, Vec & q_out, Vec & v_out, Vec & a_out) const
{
  if (s >= t.back()) {
    q_out = q.back();
    v_out.setZero();
    a_out.setZero();
    return;
  }
  std::size_t i = 0;
  while (t[i + 1] < s) {
    ++i;
  }
  const double h = t[i + 1] - t[i], u = std::clamp((s - t[i]) / h, 0.0, 1.0);
  const auto &q0 = q[i], &q1 = q[i + 1], &v0 = v[i], &v1 = v[i + 1];
  q_out = (2 * u * u * u - 3 * u * u + 1) * q0 + (u * u * u - 2 * u * u + u) * h * v0 +
          (-2 * u * u * u + 3 * u * u) * q1 + (u * u * u - u * u) * h * v1;
  v_out = ((6 * u * u - 6 * u) * q0 + (3 * u * u - 4 * u + 1) * h * v0 + (-6 * u * u + 6 * u) * q1 +
           (3 * u * u - 2 * u) * h * v1) / h;
  a_out = ((12 * u - 6) * q0 + (6 * u - 4) * h * v0 + (-12 * u + 6) * q1 + (6 * u - 2) * h * v1) / (h * h);
}
}  // namespace oap

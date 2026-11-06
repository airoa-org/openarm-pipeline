// Copyright 2026 AI Robot Association (AIRoA)
// SPDX-License-Identifier: Apache-2.0
// Both arms' kinematics and dynamics, from the URDF reduced to the 14 arm joints. Joint order: left 1-7, right 1-7.
#pragma once

#include <array>
#include <string>

#include <Eigen/Core>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

namespace oap
{
constexpr int kDof = 7, kJoints = 14;
using Vec = Eigen::Matrix<double, kJoints, 1>;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Jac = Eigen::Matrix<double, 6, kDof>;
const std::array<std::string, 2> kSides = {"left", "right"};

std::string read_file(const std::string & path);
std::string default_urdf();  // oap_description's

class Model
{
public:
  explicit Model(const std::string & urdf_xml);

  void set(const Vec & q);  // placements and Jacobians at q
  const pinocchio::SE3 & hand(int arm) const;  // <side>_hand_root in world, after set()
  Jac jacobian(int arm);  // in the hand frame, after set()
  Vec rnea(const Vec & q, const Vec & v, const Vec & a);
  Vec gravity(const Vec & q);

  const Vec & lower() const { return lower_; }
  const Vec & upper() const { return upper_; }
  const std::array<std::string, kJoints> & names() const { return names_; }

private:
  Vec to_pin(const Vec & x) const;
  Vec from_pin(const Eigen::VectorXd & x) const;

  pinocchio::Model model_;
  pinocchio::Data data_;
  std::array<std::string, kJoints> names_;
  std::array<int, kJoints> idx_;  // our order → pinocchio's
  std::array<pinocchio::FrameIndex, 2> hand_;
  Vec lower_, upper_;
  Eigen::Matrix<double, 6, Eigen::Dynamic> j6_;
};
}  // namespace oap

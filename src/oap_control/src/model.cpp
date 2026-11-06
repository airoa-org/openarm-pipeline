// Copyright 2026 AI Robot Association (AIRoA)
// SPDX-License-Identifier: Apache-2.0
#include "oap_control/model.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/parsers/urdf.hpp>

namespace oap
{
std::string read_file(const std::string & path)
{
  std::ifstream f(path);
  if (!f) {
    throw std::invalid_argument("cannot read " + path);
  }
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::string default_urdf()
{
  return ament_index_cpp::get_package_share_directory("oap_description") + "/urdf/openarm_bimanual.urdf";
}

Model::Model(const std::string & urdf_xml)
{
  pinocchio::Model full;
  pinocchio::urdf::buildModelFromXML(urdf_xml, full);
  for (int s = 0; s < 2; ++s) {
    for (int j = 0; j < kDof; ++j) {
      names_[s * kDof + j] = "openarm_" + kSides[s] + "_joint" + std::to_string(j + 1);
    }
  }
  std::vector<pinocchio::JointIndex> lock;
  for (pinocchio::JointIndex j = 1; j < full.joints.size(); ++j) {
    if (std::find(names_.begin(), names_.end(), full.names[j]) == names_.end()) {
      lock.push_back(j);
    }
  }
  model_ = pinocchio::buildReducedModel(full, lock, pinocchio::neutral(full));
  data_ = pinocchio::Data(model_);
  if (model_.nq != kJoints) {
    throw std::invalid_argument("the URDF has " + std::to_string(model_.nq) + " arm joints, expected 14");
  }
  for (int k = 0; k < kJoints; ++k) {
    if (!model_.existJointName(names_[k])) {
      throw std::invalid_argument("the URDF has no joint " + names_[k]);
    }
    idx_[k] = model_.idx_qs[model_.getJointId(names_[k])];
    lower_[k] = model_.lowerPositionLimit[idx_[k]];
    upper_[k] = model_.upperPositionLimit[idx_[k]];
  }
  for (int s = 0; s < 2; ++s) {
    const auto f = kSides[s] + "_hand_root";
    if (!model_.existFrame(f)) {
      throw std::invalid_argument("the URDF has no frame " + f);
    }
    hand_[s] = model_.getFrameId(f);
  }
  j6_.setZero(6, model_.nv);
}

Vec Model::to_pin(const Vec & x) const
{
  Vec y;
  for (int k = 0; k < kJoints; ++k) {
    y[idx_[k]] = x[k];
  }
  return y;
}

Vec Model::from_pin(const Eigen::VectorXd & x) const
{
  Vec y;
  for (int k = 0; k < kJoints; ++k) {
    y[k] = x[idx_[k]];
  }
  return y;
}

void Model::set(const Vec & q)
{
  pinocchio::computeJointJacobians(model_, data_, to_pin(q));
  pinocchio::updateFramePlacements(model_, data_);
}

const pinocchio::SE3 & Model::hand(int arm) const { return data_.oMf[hand_[arm]]; }

Jac Model::jacobian(int arm)
{
  j6_.setZero();
  pinocchio::getFrameJacobian(model_, data_, hand_[arm], pinocchio::LOCAL, j6_);
  Jac J;
  for (int j = 0; j < kDof; ++j) {
    J.col(j) = j6_.col(idx_[arm * kDof + j]);
  }
  return J;
}

Vec Model::rnea(const Vec & q, const Vec & v, const Vec & a)
{
  return from_pin(pinocchio::rnea(model_, data_, to_pin(q), to_pin(v), to_pin(a)));
}

Vec Model::gravity(const Vec & q)
{
  return from_pin(pinocchio::computeGeneralizedGravity(model_, data_, to_pin(q)));
}
}  // namespace oap

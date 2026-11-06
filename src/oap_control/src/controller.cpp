// Copyright 2026 AI Robot Association (AIRoA)
// SPDX-License-Identifier: Apache-2.0
// Torque feedforward for one arm, chained below motion.

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include <fstream>
#include <sstream>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "controller_interface/chainable_controller_interface.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/model.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace oap_control
{
// A joint's own body (its link and the fixed links on it) with a refitted mass and centre of mass in the joint frame,
// and a rotational inertia about that centre of mass; without one, the URDF's stays.
struct BodyOverride
{
  std::string joint;
  double mass = 0.0;
  Eigen::Vector3d com = Eigen::Vector3d::Zero();
  std::optional<Eigen::Matrix3d> inertia;
};

// One arm's model: the URDF reduced to `joints`, with `body` (when it names a joint) replacing that joint's own body.
// The reduced model locks the other joints, so their links stay on it.
inline pinocchio::Model load_arm_model(const std::string & file, const std::vector<std::string> & joints,
                                       const BodyOverride & body = {})
{
  std::ifstream f(file);
  if (!f) {
    throw std::runtime_error("cannot read " + file);
  }
  std::stringstream xml;
  xml << f.rdbuf();
  pinocchio::Model full;
  pinocchio::urdf::buildModelFromXML(xml.str(), full);
  std::vector<pinocchio::JointIndex> locked;
  for (pinocchio::JointIndex j = 1; j < full.joints.size(); ++j) {
    if (std::find(joints.begin(), joints.end(), full.names[j]) == joints.end()) {
      locked.push_back(j);
    }
  }
  pinocchio::Model model = pinocchio::buildReducedModel(full, locked, pinocchio::neutral(full));
  for (const auto & joint : joints) {
    if (!model.existJointName(joint)) {
      throw std::runtime_error("joint " + joint + " is not in " + file);
    }
  }
  if (!body.joint.empty()) {
    if (std::find(joints.begin(), joints.end(), body.joint) == joints.end() || !(body.mass > 0.0) ||
        !body.com.allFinite()) {
      throw std::runtime_error("body_override: joint " + body.joint + " is not one of joints, or mass/com is invalid");
    }
    const auto & own = full.inertias[full.getJointId(body.joint)];
    auto & merged = model.inertias[model.getJointId(body.joint)];
    const pinocchio::Inertia refit(body.mass, body.com, body.inertia ? *body.inertia : own.inertia().matrix());
    merged = pinocchio::Inertia::FromDynamicParameters(merged.toDynamicParameters() - own.toDynamicParameters() +
                                                       refit.toDynamicParameters());
  }
  return model;
}

inline std::string default_model_file()
{
  return ament_index_cpp::get_package_share_directory("oap_description") + "/urdf/openarm_bimanual.urdf";
}

using controller_interface::CallbackReturn;
using controller_interface::InterfaceConfiguration;
using controller_interface::return_type;
namespace hw = hardware_interface;

class TorqueFeedforwardController : public controller_interface::ChainableControllerInterface
{
public:
  CallbackReturn on_init() override
  {
    auto_declare<std::vector<std::string>>("joints", {});
    auto_declare<std::string>("mode", "gravity_comp");
    auto_declare<std::vector<double>>("max_tau", {});
    auto_declare<std::string>("model_file", "");
    auto_declare<std::string>("body_override.joint", "");
    auto_declare<double>("body_override.mass", 0.0);
    auto_declare<std::vector<double>>("body_override.com", {0.0, 0.0, 0.0});
    auto_declare<std::vector<double>>("body_override.inertia", {});  // ixx iyy izz ixy ixz iyz about the com
    auto_declare<std::vector<double>>("friction.coulomb", {});
    auto_declare<std::vector<double>>("friction.static", {});
    auto_declare<double>("friction.stribeck_velocity", 0.025);
    auto_declare<double>("friction.transition_velocity", 0.001);
    auto_declare<std::vector<double>>("integral.gain", {});
    auto_declare<std::vector<double>>("integral.limit", {});
    auto_declare<double>("integral.band", 0.1);
    auto_declare<double>("integral.still_velocity", 0.001);
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_configure(const rclcpp_lifecycle::State &) override
  {
    joints_ = get_node()->get_parameter("joints").as_string_array();
    mode_ = get_node()->get_parameter("mode").as_string();
    max_tau_ = get_node()->get_parameter("max_tau").as_double_array();
    if (max_tau_.size() != joints_.size()) {
      RCLCPP_ERROR(get_node()->get_logger(), "max_tau needs one value per joint");
      return CallbackReturn::ERROR;
    }
    fc_ = get_node()->get_parameter("friction.coulomb").as_double_array();
    fs_ = get_node()->get_parameter("friction.static").as_double_array();
    vs_ = get_node()->get_parameter("friction.stribeck_velocity").as_double();
    vt_ = get_node()->get_parameter("friction.transition_velocity").as_double();
    if (fc_.empty() != fs_.empty() || (!fc_.empty() && (fc_.size() != joints_.size() || fs_.size() != joints_.size())) ||
        !(vs_ > 0.0) || !(vt_ > 0.0)) {
      RCLCPP_ERROR(get_node()->get_logger(), "friction.coulomb and friction.static need one value per joint, or none; "
                                             "the velocities must be positive");
      return CallbackReturn::ERROR;
    }
    ki_ = get_node()->get_parameter("integral.gain").as_double_array();
    ki_limit_ = get_node()->get_parameter("integral.limit").as_double_array();
    ki_band_ = get_node()->get_parameter("integral.band").as_double();
    ki_still_ = get_node()->get_parameter("integral.still_velocity").as_double();
    if (ki_.size() != ki_limit_.size() || (!ki_.empty() && ki_.size() != joints_.size()) || !(ki_band_ > 0.0) ||
        !(ki_still_ >= 0.0)) {
      RCLCPP_ERROR(get_node()->get_logger(), "integral.gain and integral.limit need one value per joint, or none; "
                                             "integral.band must be positive, integral.still_velocity not negative");
      return CallbackReturn::ERROR;
    }
    if (mode_ != "gravity_comp" && mode_ != "inverse_dynamics") {
      RCLCPP_ERROR(get_node()->get_logger(), "mode must be gravity_comp or inverse_dynamics, got '%s'", mode_.c_str());
      return CallbackReturn::ERROR;
    }
    std::string file = get_node()->get_parameter("model_file").as_string();
    try {
      BodyOverride body;
      body.joint = get_node()->get_parameter("body_override.joint").as_string();
      body.mass = get_node()->get_parameter("body_override.mass").as_double();
      const auto com = get_node()->get_parameter("body_override.com").as_double_array();
      if (com.size() != 3) {
        throw std::runtime_error("body_override.com needs 3 values");
      }
      body.com = Eigen::Vector3d(com[0], com[1], com[2]);
      const auto in = get_node()->get_parameter("body_override.inertia").as_double_array();
      if (!in.empty()) {
        if (in.size() != 6) {
          throw std::runtime_error("body_override.inertia needs 6 values (ixx iyy izz ixy ixz iyz), or none");
        }
        Eigen::Matrix3d I;
        I << in[0], in[3], in[4], in[3], in[1], in[5], in[4], in[5], in[2];
        body.inertia = I;
      }
      model_ = load_arm_model(file.empty() ? default_model_file() : file, joints_, body);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_node()->get_logger(), "%s", e.what());
      return CallbackReturn::ERROR;
    }
    data_ = pinocchio::Data(model_);
    idx_q_.clear();
    idx_v_.clear();
    for (const auto & joint : joints_) {
      const auto & j = model_.joints[model_.getJointId(joint)];
      idx_q_.push_back(j.idx_q());
      idx_v_.push_back(j.idx_v());
    }
    q_ = pinocchio::neutral(model_);
    v_ = Eigen::VectorXd::Zero(model_.nv);
    a_ = Eigen::VectorXd::Zero(model_.nv);
    return CallbackReturn::SUCCESS;
  }

  // Hardware commands, grouped by type: all positions, then velocities, then efforts.
  InterfaceConfiguration command_interface_configuration() const override
  {
    InterfaceConfiguration conf{controller_interface::interface_configuration_type::INDIVIDUAL, {}};
    for (const auto * type : {hw::HW_IF_POSITION, hw::HW_IF_VELOCITY, hw::HW_IF_EFFORT}) {
      for (const auto & joint : joints_) {
        conf.names.push_back(joint + "/" + type);
      }
    }
    return conf;
  }

  InterfaceConfiguration state_interface_configuration() const override
  {
    InterfaceConfiguration conf{controller_interface::interface_configuration_type::INDIVIDUAL, {}};
    for (const auto * type : {hw::HW_IF_POSITION, hw::HW_IF_VELOCITY}) {
      for (const auto & joint : joints_) {
        conf.names.push_back(joint + "/" + type);
      }
    }
    return conf;
  }

  CallbackReturn on_activate(const rclcpp_lifecycle::State &) override
  {
    // Start at rest on the measured pose, until motion writes its first reference.
    const size_t n = joints_.size();
    for (size_t i = 0; i < n; ++i) {
      reference_interfaces_[i] = state_interfaces_[i].get_value();
      reference_interfaces_[n + i] = 0.0;
      reference_interfaces_[2 * n + i] = 0.0;
    }
    integral_.assign(n, 0.0);
    return CallbackReturn::SUCCESS;
  }

  return_type update_and_write_commands(const rclcpp::Time &, const rclcpp::Duration & period) override
  {
    const size_t n = joints_.size();
    for (size_t i = 0; i < n; ++i) {
      q_[idx_q_[i]] = state_interfaces_[i].get_value();
      v_[idx_v_[i]] = reference_interfaces_[n + i];
      a_[idx_v_[i]] = reference_interfaces_[2 * n + i];
    }
    const Eigen::VectorXd & tau = mode_ == "gravity_comp"
                                    ? pinocchio::computeGeneralizedGravity(model_, data_, q_)
                                    : pinocchio::rnea(model_, data_, q_, v_, a_);
    const auto refs = Eigen::Map<const Eigen::VectorXd>(reference_interfaces_.data(), 2 * n);
    if (!tau.allFinite() || !refs.allFinite()) {
      RCLCPP_ERROR_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 2000,
                            "non-finite state or reference: the motors keep the last command");
      return return_type::OK;
    }
    bool clipped = false;
    for (size_t i = 0; i < n; ++i) {
      const double t = tau[idx_v_[i]] + friction(i) + integrate(i, period.seconds());
      clipped |= std::abs(t) > max_tau_[i];
      command_interfaces_[i].set_value(reference_interfaces_[i]);
      command_interfaces_[n + i].set_value(reference_interfaces_[n + i]);
      command_interfaces_[2 * n + i].set_value(std::clamp(t, -max_tau_[i], max_tau_[i]));
    }
    if (clipped) {
      RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 2000, "tau_ff clipped at max_tau");
    }
    return return_type::OK;
  }

protected:
  // Coulomb plus the Stribeck rise toward static as the measured joint slows, signed by the commanded velocity.
  double friction(size_t i) const
  {
    if (fc_.empty()) {
      return 0.0;
    }
    const double v_meas = state_interfaces_[joints_.size() + i].get_value() / vs_;
    return (fc_[i] + (fs_[i] - fc_[i]) * std::exp(-v_meas * v_meas)) * std::tanh(v_[idx_v_[i]] / vt_);
  }

  // The I of a PID on the joint error: beyond the band (contact or a lagging command) it holds.
  // It holds too while the command is still: at rest it grew until it broke static friction, and the joint stepped.
  double integrate(size_t i, double dt)
  {
    if (ki_.empty()) {
      return 0.0;
    }
    const double e = reference_interfaces_[i] - state_interfaces_[i].get_value();
    if (ki_[i] > 0.0 && std::abs(e) <= ki_band_ && std::abs(reference_interfaces_[joints_.size() + i]) > ki_still_) {
      integral_[i] = std::clamp(integral_[i] + ki_[i] * e * dt, -ki_limit_[i], ki_limit_[i]);
    }
    return integral_[i];
  }

  // What motion writes: <this controller>/<joint>/{position, velocity, acceleration}.
  std::vector<hw::CommandInterface> on_export_reference_interfaces() override
  {
    reference_interfaces_.assign(3 * joints_.size(), 0.0);
    std::vector<hw::CommandInterface> refs;
    size_t k = 0;
    for (const auto * type : {hw::HW_IF_POSITION, hw::HW_IF_VELOCITY, hw::HW_IF_ACCELERATION}) {
      for (const auto & joint : joints_) {
        refs.emplace_back(get_node()->get_name(), joint + "/" + type, &reference_interfaces_[k++]);
      }
    }
    return refs;
  }

  // Only meaningful chained below motion: there is nothing to read otherwise.
  bool on_set_chained_mode(bool) override { return true; }
  return_type update_reference_from_subscribers() override { return return_type::OK; }

private:
  std::vector<std::string> joints_;
  std::string mode_;
  std::vector<double> max_tau_, fc_, fs_;
  double vs_ = 0.0, vt_ = 0.0;
  std::vector<double> ki_, ki_limit_, integral_;
  double ki_band_ = 0.0, ki_still_ = 0.0;
  pinocchio::Model model_;
  pinocchio::Data data_;
  std::vector<int> idx_q_, idx_v_;
  Eigen::VectorXd q_, v_, a_;
};

}  // namespace oap_control

PLUGINLIB_EXPORT_CLASS(
  oap_control::TorqueFeedforwardController, controller_interface::ChainableControllerInterface)

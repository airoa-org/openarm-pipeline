// Copyright 2026 AI Robot Association (AIRoA)
// SPDX-License-Identifier: Apache-2.0
// The motion inputs → q, q̇, q̈ references for the two torque feedforward controllers, once per update: a hand pose
// chunk becomes a spline (spline.hpp), sampled and followed by one IK step (ik.hpp); a planner path plays directly
// through the same references, so the chain never switches. Also the trip on a joint that falls behind, and /arm/hold.
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "controller_interface/controller_interface.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "realtime_tools/realtime_buffer.hpp"
#include "realtime_tools/realtime_publisher.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_srvs/srv/set_bool.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "trajectory_msgs/msg/multi_dof_joint_trajectory.hpp"

#include "oap_control/ik.hpp"
#include "oap_control/inputs.hpp"
#include "oap_control/model.hpp"
#include "oap_control/spline.hpp"
#include "oap_control/validate.hpp"

namespace oap_control
{
using controller_interface::CallbackReturn;
using controller_interface::InterfaceConfiguration;
using controller_interface::return_type;
using oap::kDof;
using oap::kJoints;
using oap::Vec;
using oap::Vec6;
using pinocchio::SE3;
namespace hw = hardware_interface;

using oap::JointPath;
using oap::PoseChunk;

class Motion : public controller_interface::ControllerInterface
{
public:
  CallbackReturn on_init() override
  {
    auto_declare<std::vector<std::string>>("feedforward_controllers", {"left_torque_controller", "right_torque_controller"});
    auto_declare<std::string>("model_file", "");
    auto_declare<double>("max_joint_velocity", 1.256);
    auto_declare<double>("max_joint_acceleration", 1.005);
    auto_declare<double>("max_joint_lead", 0.15);
    auto_declare<double>("ik_damping", 0.02);
    auto_declare<double>("limit_margin", 0.02);
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_configure(const rclcpp_lifecycle::State &) override
  {
    auto node = get_node();
    ff_ = node->get_parameter("feedforward_controllers").as_string_array();
    lead_max_ = node->get_parameter("max_joint_lead").as_double();
    if (ff_.size() != 2) {
      RCLCPP_ERROR(node->get_logger(), "feedforward_controllers needs the left and the right one");
      return CallbackReturn::ERROR;
    }
    std::string file = node->get_parameter("model_file").as_string();
    try {
      model_ = std::make_unique<oap::Model>(oap::read_file(file.empty() ? oap::default_urdf() : file));
      ik_ = std::make_unique<oap::Ik>(*model_, node->get_parameter("max_joint_velocity").as_double(),
                                      node->get_parameter("max_joint_acceleration").as_double(),
                                      node->get_parameter("ik_damping").as_double(),
                                      node->get_parameter("limit_margin").as_double());
    } catch (const std::exception & e) {
      RCLCPP_ERROR(node->get_logger(), "%s", e.what());
      return CallbackReturn::ERROR;
    }
    for (int s = 0; s < 2; ++s) {
      chunk_sub_[s] = node->create_subscription<trajectory_msgs::msg::MultiDOFJointTrajectory>(
        "/arm/" + oap::kSides[s] + "/cmd_pose_trajectory", 10,
        [this, s](trajectory_msgs::msg::MultiDOFJointTrajectory::ConstSharedPtr m) { on_chunk(s, *m); });
      hand_pub_[s] = std::make_unique<PosePub>(
        node->create_publisher<geometry_msgs::msg::PoseStamped>("/arm/" + oap::kSides[s] + "/hand_pose", 10));
      command_pub_[s] = std::make_unique<PosePub>(
        node->create_publisher<geometry_msgs::msg::PoseStamped>("/arm/" + oap::kSides[s] + "/command_pose", 10));
      ik_pub_[s] = std::make_unique<PosePub>(
        node->create_publisher<geometry_msgs::msg::PoseStamped>("/arm/" + oap::kSides[s] + "/ik_pose", 10));
    }
    path_sub_ = node->create_subscription<trajectory_msgs::msg::JointTrajectory>(
      "/arm/joint_trajectory", 10, [this](trajectory_msgs::msg::JointTrajectory::ConstSharedPtr m) { on_path(*m); });
    hold_srv_ = node->create_service<std_srvs::srv::Trigger>(
      "/arm/hold", [this](std_srvs::srv::Trigger::Request::SharedPtr, std_srvs::srv::Trigger::Response::SharedPtr res) {
        follow_ = false;
        hold_req_ = true;
        res->success = true;
        res->message = "holding the measured pose";
      });
    // Manual (the hardware's kp ramped to 0): the command follows the arms moved by hand instead of tripping on them,
    // so the next action starts from where they are. /arm/hold ends it.
    follow_srv_ = node->create_service<std_srvs::srv::SetBool>(
      "/arm/follow", [this](std_srvs::srv::SetBool::Request::SharedPtr req,
                            std_srvs::srv::SetBool::Response::SharedPtr res) {
        follow_ = req->data;
        res->success = true;
        res->message = req->data ? "the command follows the measured pose" : "following off";
      });
    cmd_pub_ = std::make_unique<JointPub>(node->create_publisher<sensor_msgs::msg::JointState>("/arm/command", 10));
    trip_pub_ = std::make_unique<BoolPub>(
      node->create_publisher<std_msgs::msg::Bool>("/arm/tripped", rclcpp::QoS(1).transient_local()));
    cmd_pub_->msg_.name.assign(model_->names().begin(), model_->names().end());
    cmd_pub_->msg_.position.resize(kJoints);
    cmd_pub_->msg_.velocity.resize(kJoints);
    cmd_pub_->msg_.effort.resize(kJoints);  // q̈_cmd, in the effort field
    return CallbackReturn::SUCCESS;
  }

  // Per arm: the feedforward controller's position, velocity and acceleration references.
  InterfaceConfiguration command_interface_configuration() const override
  {
    InterfaceConfiguration conf{controller_interface::interface_configuration_type::INDIVIDUAL, {}};
    for (int s = 0; s < 2; ++s) {
      for (const auto * type : {hw::HW_IF_POSITION, hw::HW_IF_VELOCITY, hw::HW_IF_ACCELERATION}) {
        for (int j = 0; j < kDof; ++j) {
          conf.names.push_back(ff_[s] + "/" + model_->names()[s * kDof + j] + "/" + type);
        }
      }
    }
    return conf;
  }

  InterfaceConfiguration state_interface_configuration() const override
  {
    InterfaceConfiguration conf{controller_interface::interface_configuration_type::INDIVIDUAL, {}};
    for (const auto * type : {hw::HW_IF_POSITION, hw::HW_IF_VELOCITY}) {
      for (const auto & name : model_->names()) {
        conf.names.push_back(name + "/" + type);
      }
    }
    return conf;
  }

  CallbackReturn on_activate(const rclcpp_lifecycle::State &) override
  {
    // Whatever was queued before activation is stale: mark it taken, so it never plays (on the realtime side of the
    // buffers, without writing them). Best effort: readFromRT swaps only when its try_lock wins, so a message being
    // written at this instant is taken by the first update, still behind the trip, lateness and start checks.
    last_path_ = *path_in_.readFromRT();
    for (int s = 0; s < 2; ++s) {
      last_chunk_[s] = *chunk_in_[s].readFromRT();
    }
    read_state();
    if (!q_.allFinite()) {  // no pose to hold: refuse, rather than hold zero or the last activation's command
      RCLCPP_ERROR(get_node()->get_logger(), "the joint state is not finite: not activating");
      return CallbackReturn::FAILURE;
    }
    q_finite_ = q_;
    clear_trip();
    return CallbackReturn::SUCCESS;
  }

  return_type update(const rclcpp::Time & time, const rclcpp::Duration & period) override
  {
    const double t = time.seconds(), dt = std::max(period.seconds(), 1e-4);
    read_state();
    if (hold_req_.exchange(false)) {
      clear_trip();
    }
    if (follow_) {  // chunks and paths that arrive meanwhile are dropped, not played once it ends
      for (int s = 0; s < 2; ++s) {
        last_chunk_[s] = *chunk_in_[s].readFromRT();
      }
      last_path_ = *path_in_.readFromRT();
      to_measured();
      return finish(time);
    }
    // A message is new when its pointer differs from the last one taken, so the loop never writes to the buffers.
    // While tripped, nothing moves the command until /arm/hold puts it on the measured pose again.
    if (const auto p = *path_in_.readFromRT(); p && p != last_path_) {
      last_path_ = p;
      if (tripped_) {
        RCLCPP_WARN(get_node()->get_logger(), "joint trajectory dropped: tripped, call /arm/hold first");
      } else {
        switch (oap::path_start(*p, t, q_cmd_, q_start_, v_start_, a_start_)) {
          case oap::PathStart::kLate: {
            const double late = t - p->h.t.front();
            RCLCPP_WARN(get_node()->get_logger(), "joint trajectory dropped: it starts %.3f s in the %s",
                        std::abs(late), late > 0 ? "past" : "future");
            break;
          }
          case oap::PathStart::kFar:
            RCLCPP_WARN(get_node()->get_logger(),
                        "joint trajectory dropped: its start is more than %.2f rad from the command",
                        oap::kMaxPathStartGap);
            break;
          case oap::PathStart::kOk:
            path_ = p;
            path_t0_ = p->rebase ? t : 0.0;  // what the path's times count from
            for (auto & a : active_) {
              a = false;
            }
            break;
        }
      }
    }
    for (int s = 0; s < 2; ++s) {
      if (const auto c = *chunk_in_[s].readFromRT(); c && c != last_chunk_[s]) {
        last_chunk_[s] = c;
        if (tripped_) {
          RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 2000,
                               "%s chunk dropped: tripped, call /arm/hold first", oap::kSides[s].c_str());
        } else if (!path_) {
          install(s, *c, t);
        }
      }
    }

    if (path_) {
      path_->h.sample(t - path_t0_, q_cmd_, v_cmd_, a_cmd_);
      if (t - path_t0_ < path_->h.t.front()) {  // stamped a little early: wait on its first point, at rest
        v_cmd_.setZero();
        a_cmd_.setZero();
      }
      if (t - path_t0_ >= path_->h.t.back()) {
        path_.reset();
        hold();
      }
    } else {
      for (int s = 0; s < 2; ++s) {
        if (active_[s]) {
          spline_[s].sample(t, target_[s], twist_[s]);
          active_[s] = t < spline_[s].end();
        }
      }
      for (int s = 0; s < 2; ++s) {
        if (!active_[s]) {
          twist_[s].setZero();  // a finished spline rests at its last knot
        }
      }
      ik_->step(target_, twist_, dt, q_cmd_, v_cmd_, a_cmd_);
    }
    switch (oap::trip_cause(q_, q_cmd_, v_cmd_, a_cmd_, lead_max_)) {
      case oap::TripCause::kNotFinite:  // never write NaN
        if (!tripped_) {
          RCLCPP_ERROR(get_node()->get_logger(), "the joint state or command is not finite: the arms hold");
        }
        trip();
        break;
      case oap::TripCause::kLead:  // stuck or blocked, or relaxed: stop chasing the target
        if (!tripped_) {
          RCLCPP_WARN(get_node()->get_logger(), "a joint fell more than %.2f rad behind: the arms hold", lead_max_);
        }
        trip();
        break;
      case oap::TripCause::kNone:
        break;
    }
    return finish(time);
  }

  return_type finish(const rclcpp::Time & time)
  {
    if (q_cmd_.allFinite()) {
      q_finite_ = q_cmd_;
    }
    write_references();
    publish(time);
    publish_trip();
    return return_type::OK;
  }

private:
  using PosePub = realtime_tools::RealtimePublisher<geometry_msgs::msg::PoseStamped>;
  using JointPub = realtime_tools::RealtimePublisher<sensor_msgs::msg::JointState>;
  using BoolPub = realtime_tools::RealtimePublisher<std_msgs::msg::Bool>;

  void on_chunk(int s, const trajectory_msgs::msg::MultiDOFJointTrajectory & m)
  {
    const auto tip = oap::kSides[s] + "_hand_root";
    PoseChunk c;  // on the stack: the heap only for a chunk that is taken
    switch (oap::parse_chunk(m, tip, c)) {
      case oap::ChunkCheck::kFrame:
        RCLCPP_WARN(get_node()->get_logger(), "%s chunk in frame '%s' dropped: only '%s' or 'world'",
                    oap::kSides[s].c_str(), m.header.frame_id.c_str(), tip.c_str());
        return;
      case oap::ChunkCheck::kCount:
        RCLCPP_WARN(get_node()->get_logger(), "%s chunk dropped: 1 to %d points", oap::kSides[s].c_str(),
                    oap::kMaxKnots - 1);
        return;
      case oap::ChunkCheck::kPoint:
        RCLCPP_WARN(get_node()->get_logger(),
                    "%s chunk dropped: one finite transform with a unit quaternion per point, strictly increasing times",
                    oap::kSides[s].c_str());
        return;
      case oap::ChunkCheck::kOk:
        break;
    }
    chunk_in_[s].writeFromNonRT(std::make_shared<PoseChunk>(c));
  }

  void on_path(const trajectory_msgs::msg::JointTrajectory & m)
  {
    JointPath p;
    const auto r = oap::parse_path(m, model_->names(), model_->lower(), model_->upper(),
                                   get_node()->now().seconds(), p);
    switch (r.check) {
      case oap::PathCheck::kNames:
        RCLCPP_WARN(get_node()->get_logger(), "joint trajectory dropped: it must name all 14 arm joints");
        return;
      case oap::PathCheck::kSize:
        RCLCPP_WARN(get_node()->get_logger(), "joint trajectory dropped: a point has %zu positions for %zu joint names",
                    r.positions, m.joint_names.size());
        return;
      case oap::PathCheck::kLimits:
        RCLCPP_WARN(get_node()->get_logger(), "joint trajectory dropped: a point is outside the joint limits");
        return;
      case oap::PathCheck::kTimes:
        RCLCPP_WARN(get_node()->get_logger(), "joint trajectory dropped: it needs two points at strictly increasing times");
        return;
      case oap::PathCheck::kLate:  // checked again when the loop takes it
        RCLCPP_WARN(get_node()->get_logger(), "joint trajectory dropped: it starts %.3f s in the %s", std::abs(r.late),
                    r.late > 0 ? "past" : "future");
        return;
      case oap::PathCheck::kOk:
        break;
    }
    path_in_.writeFromNonRT(std::make_shared<JointPath>(std::move(p)));
  }

  // The spline starts at the target now, with its twist: the target's lead over the command is what moves the IK, so
  // starting at the command would brake it at every chunk.
  void install(int s, const PoseChunk & c, double t)
  {
    Vec6 twist = Vec6::Zero();
    if (active_[s]) {
      SE3 now;
      spline_[s].sample(t, now, twist);
    }
    model_->set(q_cmd_);
    const SE3 anchor = c.relative ? model_->hand(s) : SE3::Identity();  // the commanded hand: the target can run ahead
    std::array<SE3, oap::kMaxKnots> pose;
    for (int k = 0; k < c.n; ++k) {
      pose[k] = anchor * c.pose[k];
    }
    if (spline_[s].fit(t, target_[s], twist, c.t.data(), pose.data(), c.n)) {
      active_[s] = true;
    }
  }

  void read_state()
  {
    for (int k = 0; k < kJoints; ++k) {
      q_[k] = state_interfaces_[k].get_value();
      v_[k] = state_interfaces_[kJoints + k].get_value();
    }
  }

  // The only places tripped_ changes. Both put the command on the measured pose, so clearing a trip can never resume
  // from a stale command.
  // Both mark /arm/tripped for publishing; publish_trip() retries every cycle until the publisher takes it.
  void trip()
  {
    to_measured();
    trip_dirty_ = trip_dirty_ || !tripped_;
    tripped_ = true;
    publish_trip();
  }

  void clear_trip()
  {
    to_measured();
    tripped_ = false;
    trip_dirty_ = true;
    publish_trip();
  }

  void to_measured()
  {
    path_.reset();
    q_cmd_ = oap::hold_position(q_, q_finite_);
    hold();
  }

  void hold()
  {
    v_cmd_.setZero();
    a_cmd_.setZero();
    model_->set(q_cmd_);
    for (int s = 0; s < 2; ++s) {
      target_[s] = model_->hand(s);
      active_[s] = false;
    }
  }

  void write_references()
  {
    for (int s = 0; s < 2; ++s) {
      for (int j = 0; j < kDof; ++j) {
        const int k = s * kDof + j, base = s * 3 * kDof;
        command_interfaces_[base + j].set_value(q_cmd_[k]);
        command_interfaces_[base + kDof + j].set_value(v_cmd_[k]);
        command_interfaces_[base + 2 * kDof + j].set_value(a_cmd_[k]);
      }
    }
  }

  static void fill(geometry_msgs::msg::PoseStamped & p, const rclcpp::Time & stamp, const SE3 & T)
  {
    p.header.stamp = stamp;
    p.header.frame_id = "world";
    const Eigen::Quaterniond q(T.rotation());
    p.pose.position.x = T.translation().x();
    p.pose.position.y = T.translation().y();
    p.pose.position.z = T.translation().z();
    p.pose.orientation.x = q.x();
    p.pose.orientation.y = q.y();
    p.pose.orientation.z = q.z();
    p.pose.orientation.w = q.w();
  }

  void publish_trip()
  {
    if (trip_dirty_ && trip_pub_->trylock()) {
      trip_pub_->msg_.data = tripped_;
      trip_pub_->unlockAndPublish();
      trip_dirty_ = false;
    }
  }

  void publish(const rclcpp::Time & stamp)
  {
    if (cmd_pub_->trylock()) {
      cmd_pub_->msg_.header.stamp = stamp;
      for (int k = 0; k < kJoints; ++k) {
        cmd_pub_->msg_.position[k] = q_cmd_[k];
        cmd_pub_->msg_.velocity[k] = v_cmd_[k];
        cmd_pub_->msg_.effort[k] = a_cmd_[k];
      }
      cmd_pub_->unlockAndPublish();
    }
    if (++tick_ % 4 == 0) {  // 250 Hz at 1 kHz: the runner anchors each chunk on the commanded hand at its observation
      // The spline's pose, which a new chunk's spline continues from, so chunks attach to chunks; FK(q_cmd) lags it.
      // A planner path has no spline, so it reports FK(q_cmd).
      model_->set(q_cmd_);
      for (int s = 0; s < 2; ++s) {
        if (command_pub_[s]->trylock()) {
          fill(command_pub_[s]->msg_, stamp, path_ ? model_->hand(s) : target_[s]);
          command_pub_[s]->unlockAndPublish();
        }
        if (ik_pub_[s]->trylock()) {  // FK(q_cmd): where the IK got, short of the target at a joint limit
          fill(ik_pub_[s]->msg_, stamp, model_->hand(s));
          ik_pub_[s]->unlockAndPublish();
        }
      }
      model_->set(q_);
      for (int s = 0; s < 2; ++s) {
        if (hand_pub_[s]->trylock()) {
          fill(hand_pub_[s]->msg_, stamp, model_->hand(s));
          hand_pub_[s]->unlockAndPublish();
        }
      }
    }
  }

  std::vector<std::string> ff_;
  std::unique_ptr<oap::Model> model_;
  std::unique_ptr<oap::Ik> ik_;
  double lead_max_ = 0.0;

  Vec q_ = Vec::Zero(), v_ = Vec::Zero(), q_cmd_ = Vec::Zero(), v_cmd_ = Vec::Zero(), a_cmd_ = Vec::Zero();
  std::array<SE3, 2> target_{SE3::Identity(), SE3::Identity()};
  std::array<Vec6, 2> twist_{Vec6::Zero(), Vec6::Zero()};  // the spline's, fed forward to the IK
  std::array<oap::Spline, 2> spline_;
  std::array<bool, 2> active_{false, false};
  std::shared_ptr<JointPath> path_;
  double path_t0_ = 0.0;  // the start of a rebased path_, else 0
  Vec q_finite_ = Vec::Zero();  // the last finite command (on_activate: the measured pose), for a hold on a NaN state
  Vec q_start_ = Vec::Zero(), v_start_ = Vec::Zero(), a_start_ = Vec::Zero();  // a new path's sample now
  bool tripped_ = false;
  bool trip_dirty_ = false;  // /arm/tripped has a change not yet published
  long tick_ = 0;

  std::array<realtime_tools::RealtimeBuffer<std::shared_ptr<PoseChunk>>, 2> chunk_in_;
  realtime_tools::RealtimeBuffer<std::shared_ptr<JointPath>> path_in_;
  std::array<std::shared_ptr<PoseChunk>, 2> last_chunk_;
  std::shared_ptr<JointPath> last_path_;
  std::atomic<bool> hold_req_{false};
  std::atomic<bool> follow_{false};

  std::array<rclcpp::Subscription<trajectory_msgs::msg::MultiDOFJointTrajectory>::SharedPtr, 2> chunk_sub_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr path_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr hold_srv_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr follow_srv_;
  std::array<std::unique_ptr<PosePub>, 2> hand_pub_, command_pub_, ik_pub_;
  std::unique_ptr<JointPub> cmd_pub_;
  std::unique_ptr<BoolPub> trip_pub_;
};
}  // namespace oap_control

PLUGINLIB_EXPORT_CLASS(oap_control::Motion, controller_interface::ControllerInterface)

// Copyright 2026 AI Robot Association (AIRoA)
// SPDX-License-Identifier: Apache-2.0
// Both YUBI fingers on one U2D2 bus in current-based position mode. A finger's user angle is 0 closed and grows as
// it opens: raw = user·direction + offset.
#include <dynamixel_sdk/dynamixel_sdk.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

namespace
{
// X-series control table, protocol 2.0
constexpr uint16_t kOperatingMode = 11, kTorqueEnable = 64, kPositionDGain = 80, kPositionPGain = 84;
constexpr uint16_t kGoalCurrent = 102, kGoalPosition = 116, kPresentCurrent = 126, kPresentVelocity = 128;
constexpr uint16_t kPresentPosition = 132, kStateBytes = 10;  // current (2), velocity (4), position (4)
constexpr uint8_t kCurrentBasedPosition = 5;
constexpr double kMaPerUnit = 2.69, kRadPerSecPerUnit = 0.229 * 2.0 * M_PI / 60.0, kTicksPerRev = 4096.0;

// One turn of the motor is kTicksPerRev ticks with 0 rad at its middle tick; an angle folds onto [-pi, pi].
constexpr int32_t kMiddleTick = 2048;

int32_t rad_to_tick(double rad)
{
  return kMiddleTick + static_cast<int32_t>(std::lround(std::remainder(rad, 2.0 * M_PI) / (2.0 * M_PI) * kTicksPerRev));
}

double tick_to_rad(int32_t tick)
{
  return std::remainder((tick - kMiddleTick) * 2.0 * M_PI / kTicksPerRev, 2.0 * M_PI);
}

struct Finger
{
  std::string name;
  uint8_t id;
  double offset;
  int direction;
  std::optional<double> goal;  // user rad, not yet written
};
}  // namespace

class Gripper : public rclcpp::Node
{
public:
  Gripper() : Node("gripper")
  {
    const auto port = declare_parameter<std::string>("port", "");
    const auto baud = declare_parameter<int>("baudrate", 57600);
    const auto names = declare_parameter<std::vector<std::string>>("names", std::vector<std::string>{});
    const auto ids = declare_parameter<std::vector<int64_t>>("ids", std::vector<int64_t>{});
    const auto offsets = declare_parameter<std::vector<double>>("offsets_rad", std::vector<double>{});
    const auto dirs = declare_parameter<std::vector<int64_t>>("directions", std::vector<int64_t>{});
    const auto range = declare_parameter<std::vector<double>>("range_rad", std::vector<double>{0.0, 0.7854});
    const auto current = declare_parameter<int>("goal_current_ma", 1500);
    const auto p_gain = declare_parameter<int>("position_p_gain", -1);
    const auto d_gain = declare_parameter<int>("position_d_gain", -1);
    const auto rate = declare_parameter<double>("rate_hz", 60.0);
    if (names.empty() || ids.size() != names.size() || offsets.size() != names.size() ||
        dirs.size() != names.size() || range.size() != 2 || !(range[0] < range[1]) || !(rate > 0.0)) {
      throw std::invalid_argument("names, ids, offsets_rad and directions need one value per finger, range_rad two");
    }
    lo_ = range[0];
    hi_ = range[1];
    for (std::size_t i = 0; i < names.size(); ++i) {
      fingers_.push_back({names[i], static_cast<uint8_t>(ids[i]), offsets[i], dirs[i] < 0 ? -1 : 1, std::nullopt});
    }

    port_ = dynamixel::PortHandler::getPortHandler(port.c_str());
    packet_ = dynamixel::PacketHandler::getPacketHandler(2.0);
    if (!port_->openPort()) {
      throw std::runtime_error("cannot open U2D2 port '" + port +
                               "': set port in config/gripper.yaml to yours (ls /dev/serial/by-id)");
    }
    if (!port_->setBaudRate(baud)) {
      throw std::runtime_error("cannot set " + port + " to " + std::to_string(baud) + " baud");
    }
    for (const auto & f : fingers_) {
      // Mode and gains need the torque off; the goal is the present position, so the finger holds when it comes on.
      write(f.id, kTorqueEnable, 1, 0);
      write(f.id, kOperatingMode, 1, kCurrentBasedPosition);
      if (p_gain >= 0) {
        write(f.id, kPositionPGain, 2, static_cast<uint32_t>(p_gain));
      }
      if (d_gain >= 0) {
        write(f.id, kPositionDGain, 2, static_cast<uint32_t>(d_gain));
      }
      write(f.id, kGoalCurrent, 2, static_cast<uint32_t>(std::lround(current / kMaPerUnit)));
      uint32_t present = 0;
      uint8_t err = 0;
      if (packet_->read4ByteTxRx(port_, f.id, kPresentPosition, &present, &err) != COMM_SUCCESS || err) {
        throw std::runtime_error("cannot read finger " + f.name);
      }
      write(f.id, kGoalPosition, 4, present);
      write(f.id, kTorqueEnable, 1, 1);
    }
    read_ = std::make_unique<dynamixel::GroupSyncRead>(port_, packet_, kPresentCurrent, kStateBytes);
    for (const auto & f : fingers_) {
      read_->addParam(f.id);
    }

    pub_ = create_publisher<sensor_msgs::msg::JointState>("/gripper/joint_states", 10);
    sub_ = create_subscription<trajectory_msgs::msg::JointTrajectory>(
      "/gripper/joint_trajectory", 10, [this](trajectory_msgs::msg::JointTrajectory::ConstSharedPtr m) { command(*m); });
    timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / rate), [this] { tick(); });
    RCLCPP_INFO(get_logger(), "%zu fingers on %s, goal current %ld mA", fingers_.size(), port.c_str(), static_cast<long>(current));
  }

  ~Gripper() override { port_->closePort(); }  // the torque stays on, so the fingers keep their grip

private:
  void write(uint8_t id, uint16_t addr, int bytes, uint32_t value)
  {
    uint8_t err = 0;
    const int r = bytes == 1   ? packet_->write1ByteTxRx(port_, id, addr, static_cast<uint8_t>(value), &err)
                  : bytes == 2 ? packet_->write2ByteTxRx(port_, id, addr, static_cast<uint16_t>(value), &err)
                               : packet_->write4ByteTxRx(port_, id, addr, value, &err);
    if (r != COMM_SUCCESS || err) {
      throw std::runtime_error("write to id " + std::to_string(id) + " address " + std::to_string(addr) + " failed");
    }
  }

  // The last point is the goal; the motor's own profile moves the finger there.
  void command(const trajectory_msgs::msg::JointTrajectory & m)
  {
    if (m.points.empty() || m.points.back().positions.size() != m.joint_names.size()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "trajectory without one position per joint dropped");
      return;
    }
    for (std::size_t k = 0; k < m.joint_names.size(); ++k) {
      for (auto & f : fingers_) {
        const double v = m.points.back().positions[k];
        if (f.name == m.joint_names[k] && std::isfinite(v)) {
          f.goal = std::clamp(v, lo_, hi_);
        }
      }
    }
  }

  void tick()
  {
    dynamixel::GroupSyncWrite goals(port_, packet_, kGoalPosition, 4);
    bool any = false;
    for (auto & f : fingers_) {
      if (f.goal) {
        const auto t = static_cast<uint32_t>(rad_to_tick(*f.goal * f.direction + f.offset));
        uint8_t b[4] = {DXL_LOBYTE(DXL_LOWORD(t)), DXL_HIBYTE(DXL_LOWORD(t)), DXL_LOBYTE(DXL_HIWORD(t)),
                        DXL_HIBYTE(DXL_HIWORD(t))};
        goals.addParam(f.id, b);
        f.goal.reset();
        any = true;
      }
    }
    if (any && goals.txPacket() != COMM_SUCCESS) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "goal position write failed");
    }
    if (read_->txRxPacket() != COMM_SUCCESS) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "state read failed");
      return;
    }
    sensor_msgs::msg::JointState js;
    js.header.stamp = now();
    for (const auto & f : fingers_) {
      if (!read_->isAvailable(f.id, kPresentCurrent, kStateBytes)) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "no state from %s", f.name.c_str());
        return;
      }
      const auto cur = static_cast<int16_t>(read_->getData(f.id, kPresentCurrent, 2));
      const auto vel = static_cast<int32_t>(read_->getData(f.id, kPresentVelocity, 4));
      const auto pos = static_cast<int32_t>(read_->getData(f.id, kPresentPosition, 4));
      js.name.push_back(f.name);
      js.position.push_back((tick_to_rad(pos) - f.offset) * f.direction);
      js.velocity.push_back(vel * kRadPerSecPerUnit * f.direction);
      js.effort.push_back(cur * kMaPerUnit);  // [mA]
    }
    pub_->publish(js);
  }

  std::vector<Finger> fingers_;
  double lo_ = 0.0, hi_ = 0.0;
  dynamixel::PortHandler * port_ = nullptr;
  dynamixel::PacketHandler * packet_ = nullptr;
  std::unique_ptr<dynamixel::GroupSyncRead> read_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  int code = 0;
  try {
    rclcpp::spin(std::make_shared<Gripper>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("gripper"), "%s", e.what());
    code = 1;
  }
  rclcpp::shutdown();
  return code;
}

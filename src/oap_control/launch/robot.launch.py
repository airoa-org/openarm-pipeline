# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Both arms under ros2_control (motion → torque feedforward → openarm_hardware), and the gripper on real.

    ros2 launch oap_control robot.launch.py hardware:=real   # the CAN interfaces up first (scripts/canup.sh)
    ros2 launch oap_control robot.launch.py hardware:=real left_can:=can2 right_can:=can3  # defaults can1, can0
    ros2 launch oap_control robot.launch.py                  # mock_components: commands become the state
"""
import os
import re

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

KP = [60.0, 60.0, 60.0, 60.0, 20.0, 20.0, 20.0]  # bring-up gains
KD = [4.5, 4.5, 4.5, 4.5, 1.5, 1.5, 1.5]
SIGTERM_TIMEOUT_S, SIGKILL_TIMEOUT_S = "20", "10"  # ros2_control_node's shutdown budget, see nodes()
LINK_FAULT_CYCLES = 20  # 1 kHz cycles, set from the reply gaps measured on the reference rig


def ros2_control(side, real, can):
    if real:
        params = {"arm_type": "v10", "can_interface": can, "arm_prefix": f"{side}_", "hand": "False",
                  "can_fd": "True", "link_fault_cycles": str(LINK_FAULT_CYCLES)}
        params.update({f"kp{i + 1}": str(v) for i, v in enumerate(KP)})
        params.update({f"kd{i + 1}": str(v) for i, v in enumerate(KD)})
        plugin = "openarm_hardware/OpenArm_v10HW"
    else:
        params, plugin = {}, "mock_components/GenericSystem"
    hw = f"<plugin>{plugin}</plugin>" + "".join(f'<param name="{k}">{v}</param>' for k, v in params.items())
    joints = "".join(
        f'<joint name="openarm_{side}_joint{j}">'
        + "".join(f'<command_interface name="{i}"/>' for i in ("position", "velocity", "effort"))
        + "".join(f'<state_interface name="{i}"><param name="initial_value">0.0</param></state_interface>'
                  for i in ("position", "velocity", "effort"))
        + "</joint>" for j in range(1, 8))
    return f'<ros2_control name="openarm_{side}_hardware_interface" type="system"><hardware>{hw}</hardware>{joints}</ros2_control>'


def nodes(context):
    real = LaunchConfiguration("hardware").perform(context) == "real"
    can = {s: LaunchConfiguration(f"{s}_can").perform(context) for s in ("left", "right")}
    if real:
        for s, i in can.items():
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,15}", i):
                raise RuntimeError(f"{s}_can:={i!r} is not a network interface name")
        if can["left"] == can["right"]:
            raise RuntimeError(f"left_can and right_can are both {can['left']}: each arm needs its own bus")
    with open(os.path.join(get_package_share_directory("oap_description"), "urdf", "openarm_bimanual.urdf")) as f:
        urdf = f.read()
    urdf = urdf.replace("</robot>", "".join(ros2_control(s, real, can[s])
                                            for s in ("left", "right")) + "</robot>")
    controllers = os.path.join(get_package_share_directory("oap_control"), "config", "controllers.yaml")

    def spawner(*names):
        return Node(package="controller_manager", executable="spawner", arguments=[*names], output="screen")
    out = [
        # On shutdown launch SIGTERMs a node 5 s after the SIGINT by default (SIGKILL 5 s later); deactivating and
        # disabling the real motors can take longer (DDS teardown), and a SIGTERM after rclcpp's shutdown skips it.
        Node(package="controller_manager", executable="ros2_control_node", output="screen",
             sigterm_timeout=SIGTERM_TIMEOUT_S, sigkill_timeout=SIGKILL_TIMEOUT_S,
             parameters=[{"robot_description": urdf}, controllers]),
        spawner("joint_state_broadcaster"),
        spawner("left_torque_controller", "right_torque_controller", "motion"),  # chained: the lower ones first
    ]
    if real:
        out.append(Node(package="oap_gripper", executable="gripper_node", name="gripper", output="screen",
                        parameters=[os.path.join(get_package_share_directory("oap_gripper"), "config", "gripper.yaml")]))
    return out


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("hardware", default_value="mock", choices=["mock", "real"]),
        DeclareLaunchArgument("left_can", default_value="can1"),
        DeclareLaunchArgument("right_can", default_value="can0"),
        OpaqueFunction(function=nodes),
    ])

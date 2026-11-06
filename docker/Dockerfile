# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
# The image, FROM the base (docker/base.Dockerfile: OS, ROS and apt packages, pinocchio, openarm_can, uv), which
# changes rarely and takes most of a cold build. BASE_IMAGE picks it: scripts/oap passes its local
# openarm-pipeline-base:$USER, or OAP_BASE_IMAGE.
ARG BASE_IMAGE=openarm-pipeline-base:local
FROM ${BASE_IMAGE}

# runner and gui: OpenCV for the wrist cameras, PyQt5; iproute2: scripts/canup.sh, which the GUI runs before a real
# bring-up. The policy server runs in its own venv (scripts/serve_policy.sh), so nothing of it is installed here.
RUN apt-get update && apt-get install -y --no-install-recommends python3-opencv python3-pyqt5 iproute2 \
    && rm -rf /var/lib/apt/lists/*

# openarm_hardware from openarm_ros2 0.9.2 (v1.0, matches the rig; main points at a missing xacro), with these patches,
# applied in this order (each builds on the previous ones):
#   hw_ros_node:           a ROS node inside the hardware plugin, /openarm_hw_relax_<side>, spun on its own thread
#   relax_mode:            its ~/set_relax service: kp 0, damping kept (relax_kd), so the arm sinks slowly under
#                          gravity; relax off starts from the measured pose, then tracks the command (callers
#                          /arm/hold before and after)
#   live_gains:            per-joint kp/kd as its parameters kp_joints/kd_joints, changed live and ramped over
#                          gain_ramp_s; the target gains are published on ~/gains (latched, repeated at 1 Hz);
#                          ~/set_manual ramps kp to 0 and back, keeping kd and the effort command (gravity)
#   no_refresh_in_read:    read() only copies the replies write() already parsed; the extra refresh round overloaded
#                          the CAN bus and the motors stopped following commands
#   hold_pose_on_activate: hold the measured pose on activate instead of driving every joint to zero; init fails
#                          when an arm motor does not answer while disabled, activation when it sends no valid
#                          state; on_shutdown, on_error and the destructor disable the motors as a last resort
#   velocity_half_step:    add vMax/4095 to each joint velocity: the motor truncates its 12-bit velocity, which reads
#                          low by that much (-0.011 rad/s at rest on a DM8009)
#   link_monitor:          log each motor's longest gap between replies every 10 s, and relax the arm when a motor
#                          reports an error or (with link_fault_cycles > 0) misses more than that many replies in a row
# openarm_description 1.0.4: the upstream arm and body meshes, for a URDF that refers to package://openarm_description.
ARG OPENARM_ROS2_REF=0.9.2
ARG OPENARM_DESCRIPTION_REF=1.0.4
COPY docker/patches /tmp/patches
RUN mkdir -p /opt/openarm_ws/src && cd /opt/openarm_ws \
    && git clone --branch ${OPENARM_DESCRIPTION_REF} --depth 1 https://github.com/enactic/openarm_description.git \
        src/openarm_description \
    && git clone --branch ${OPENARM_ROS2_REF} --depth 1 https://github.com/enactic/openarm_ros2.git src/openarm_ros2 \
    && cd src/openarm_ros2 && for p in hw_ros_node relax_mode live_gains no_refresh_in_read hold_pose_on_activate \
        velocity_half_step link_monitor; do git apply /tmp/patches/$p.patch || exit 1; done \
    && cd /opt/openarm_ws && . /opt/ros/humble/setup.sh \
    && colcon build --packages-select openarm_hardware openarm_description --cmake-args -DCMAKE_BUILD_TYPE=Release \
    && rm -rf build log

COPY docker/entrypoint.sh /entrypoint.sh
WORKDIR /ws
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]

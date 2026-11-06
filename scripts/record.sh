#!/usr/bin/env bash
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
# Records a run into output/openarm_<date>[_<tag>]; Ctrl-C closes the bag. In the container: ./scripts/record.sh [tag]
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
name=openarm_$(date +%Y%m%d_%H%M%S)${1:+_$1}
mkdir -p "$root/output"
exec ros2 bag record -o "$root/output/$name" /joint_states /arm/command /arm/tripped \
  /arm/left/hand_pose /arm/right/hand_pose /arm/left/command_pose /arm/right/command_pose \
  /arm/left/cmd_pose_trajectory /arm/right/cmd_pose_trajectory /arm/joint_trajectory \
  /gripper/joint_states /gripper/joint_trajectory \
  /left/left_arm_cam/image_raw/compressed /right/right_arm_cam/image_raw/compressed

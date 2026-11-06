#!/bin/bash
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
# shellcheck disable=SC1091  # the setup files exist only in the image
source /opt/ros/humble/setup.bash
source /opt/openarm_ws/install/setup.bash
[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash
exec "$@"

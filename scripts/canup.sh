#!/usr/bin/env bash
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
# Bring up an OpenArm CAN-FD bus. Needs root, because it runs `ip link`.
#
# Run it on the host, once per channel after each boot or re-plug of the robot:
#
#     sudo ./scripts/canup.sh can0
#     sudo ./scripts/canup.sh can1
#
# The GUI also runs it for both channels before a `hardware:=real` bring-up, inside
# the privileged container, which is root and shares the host's network namespace,
# so no host sudo is needed there.
#
# Which channel is which arm is set by the launch args left_can / right_can
# (defaults: left_can:=can1, right_can:=can0).
#
# The Damiao motors are on SocketCAN channels exposed by the robot's CAN-FD USB
# adapter. Settings must match the motors: 1 Mbit arbitration, 5 Mbit data,
# CAN-FD on. Healthy result = CAN state ERROR-ACTIVE, mtu 72.
set -euo pipefail

IFACE="${1:-can0}"
BITRATE=1000000        # nominal / arbitration phase (1 Mbit/s)
DBITRATE=5000000       # data phase (5 Mbit/s, CAN-FD)

if [[ $EUID -ne 0 ]]; then
  echo "needs root -- re-run as:  sudo $0 ${IFACE}" >&2
  exit 1
fi

if ! ip link show "$IFACE" &>/dev/null; then
  echo "ERROR: interface '$IFACE' not found. Is the robot USB plugged in?" >&2
  echo "       (list interfaces with:  ip link)" >&2
  exit 1
fi

echo "configuring $IFACE: bitrate ${BITRATE} dbitrate ${DBITRATE} fd on ..."
ip link set "$IFACE" down 2>/dev/null || true     # must be down to (re)set the link type
ip link set "$IFACE" type can bitrate "$BITRATE" dbitrate "$DBITRATE" fd on
ip link set "$IFACE" up

# Read back the CAN-specific state (not the netdev UP/DOWN): ERROR-ACTIVE = healthy,
# BUS-OFF/STOPPED = un-ACKed TX drove it off the bus -> just re-run this script.
# One awk, no grep: a grep that finds nothing would fail the pipeline and, under pipefail, abort before the WARN below.
state=$(ip -details link show "$IFACE" | awk '{
          for (i = 1; i < NF; i++)
            if ($i == "state" && $(i + 1) ~ /^(ERROR-ACTIVE|ERROR-WARNING|ERROR-PASSIVE|BUS-OFF|STOPPED|SLEEPING)$/) {
              if (s == "") s = $(i + 1)
            }
        } END { print s }' || true)
mtu=$(cat "/sys/class/net/${IFACE}/mtu" 2>/dev/null || echo '?')

echo "---"
ip -details -statistics link show "$IFACE" | sed -n '1,4p'
echo "---"
if [[ "${state:-}" == "ERROR-ACTIVE" ]]; then
  echo "OK: $IFACE up  state=$state  mtu=$mtu  (healthy -- ready for bring-up)"
else
  echo "WARN: $IFACE up but state=${state:-unknown} (mtu=$mtu). Expected ERROR-ACTIVE." >&2
  echo "      If BUS-OFF/STOPPED, re-run this script (restart-ms is 0, no auto-recovery)." >&2
  exit 1
fi

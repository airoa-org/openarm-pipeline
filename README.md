# openarm-pipeline

Runs the AIRoA-0 robot foundation model on a bimanual OpenArm v1.0 with YUBI grippers: policy inference, arm control
and a GUI, in one repository. ROS 2 Humble, ros2_control, Docker. The packages and scripts use the prefix `oap`.

The AIRoA-0 checkpoints are on Hugging Face: [airoa-org/AIRoA-0](https://huggingface.co/airoa-org/AIRoA-0) (see
[Policy](#policy) for how one plugs in).

AIRoA-0 is a vision-language-action (VLA) policy: from the two wrist camera images, the pose of one gripper tip seen from
the other, the finger openings and a text instruction it predicts a chunk of future tip poses and finger openings. Here
it runs on the policy server in `policy/` (see [Policy](#policy)); the runner sends its chunks to a 1 kHz controller
that turns them into joint commands.

## Hardware

- A bimanual OpenArm v1.0: two 7-DoF arms with Damiao motors, one CAN-FD bus per arm.
- A CAN-FD USB adapter with two channels (SocketCAN), 1 Mbit/s arbitration and 5 Mbit/s data.
- A YUBI gripper on each arm, its finger driven by a Dynamixel motor; both motors on one U2D2 (ids 0 and 1).
- A USB camera on each wrist (640×480 at 30 fps by default).
- An emergency stop that cuts the motor power, within reach of the operator.
- For the policy server: an NVIDIA GPU, in the same host or another one reachable over the network.

The planner keeps clear of the table and of the pillar between the arms. The table's height and footprint and both
clearance margins are in `src/oap_planner/config/workspace.yaml`; the pillar is part of the OpenArm body, so its shape
is fixed in `oap_planner` (`PILLAR_SLABS` in `geometry.py`, which both the optimizer and the checker use).

## Prerequisites

- Linux (the robot host).
- Docker. The image holds ROS 2 Humble and everything else; nothing ROS is needed on the host.
- The NVIDIA driver and the NVIDIA Container Toolkit, for the policy server. Without them `scripts/oap` starts the
  container without `--gpus` and the rest still runs.
- An X11 session for the GUI (the container draws on the host's display).
- `iproute2` on the host if you bring the CAN interfaces up by hand.

## Quick start (mock hardware, no robot)

Pick a `ROS_DOMAIN_ID` (0–101) no one else on the host or the network uses. On the host:

```sh
ROS_DOMAIN_ID=42 ./scripts/oap shell     # first run builds the image (about 15–20 min), starts the container, enters it
colcon build --symlink-install && source install/setup.bash
ros2 launch oap_control robot.launch.py  # mock_components: the command becomes the state
```

In a second terminal on the host:

```sh
./scripts/oap exec bash                  # the same container
source install/setup.bash
ros2 run oap_planner plan --target ready --dry-run   # plans and checks, sends nothing
ros2 run oap_planner plan --target ready
ros2 run oap_planner circle --radius 0.03            # tracking test from ready
ros2 run oap_planner plan --target park
```

On mock these are expected and harmless: `relax mode not switched off (no service)` (mock has no relax service) and
`no /gripper/joint_states: checking with open fingers` (the gripper node runs only on real). The runner needs the
wrist cameras and a running policy server, and the server and `prewarm` need a checkpoint (see [Policy](#policy)), so
they do not run in this quick start. `exit` both shells; `./scripts/oap down` stops the container.

## Adapt to your rig

| What | Where | Default |
|---|---|---|
| wrist cameras | `cameras.<side>.dev` in `src/oap_runner/config/runner.yaml` | `/dev/v4l/by-path/...` of the reference rig |
| U2D2 serial port | `port` in `src/oap_gripper/config/gripper.yaml` | `/dev/serial/by-id/usb-FTDI_...` of the reference rig |
| finger calibration | `offsets_rad`, `directions` in `src/oap_gripper/config/gripper.yaml` | the reference rig's |
| CAN interfaces | `left_can`, `right_can` arguments of `robot.launch.py`; `OAP_LEFT_CAN`, `OAP_RIGHT_CAN` for `scripts/gui` | `can1`, `can0` |
| ROS domain | `ROS_DOMAIN_ID` for `scripts/oap`, `OAP_ROS_DOMAIN_ID` for `scripts/gui` | none, 88 |
| table and clearance margins | `src/oap_planner/config/workspace.yaml` | the reference rig's |

Use stable device names (`/dev/v4l/by-path/`, `/dev/serial/by-id/`), not `/dev/videoN` or `/dev/ttyUSBN`, which change
between boots. `ls /dev/v4l/by-path/ /dev/serial/by-id/` lists them. Every process must see the same ROS domain: a
mismatch hides every topic without an error.

## Processes

| Process | Package | What |
|---|---|---|
| policy server | `policy/`, through `scripts/serve_policy.sh` (GPU, HTTP :8000) | the policy; checkpoints in `checkpoints/` (gitignored, a link is fine), AIRoA-0 and its venv in `.cache/` |
| runner | `oap_runner` (Python) | cameras, observation, policy client: one request at a time; sends each chunk's hand poses |
| control | `oap_control` (ros2_control, 1 kHz) | `motion` → `left/right_torque_controller` → `openarm_hardware` (or `mock_components`), plus `joint_state_broadcaster` |
| gripper | `oap_gripper` (C++) | both YUBI fingers over one U2D2 bus (`dynamixel_sdk`) |
| gui | `oap_gui` (PyQt5) | one window: control chain, Lifted / Ready / Park / Circle, Hold, Relax, Manual, gripper, the runner with its camera views, a bag |
| planner | `oap_planner` (Python) | the lifted / ready / park move: one CasADi trajectory over both arms, checked against the table and the pillar; `circle`, a tracking test |
| description | `oap_description` | the URDF of both arms, with a hand frame `<side>_hand_root` on each last link; the arm meshes come from `openarm_description`, built in the image |

ROS 2 Humble connects the processes.

Planner targets (`plan --target <name>`): `ready`, the start pose for inference; `park`, every joint at zero, the arms
hanging down, where control can stop; `lifted`, the hands about 5.5 cm above park, off its singularity,
where the arms can relax and sink without hitting anything.

### Glossary

- **YUBI gripper**: the parallel gripper on each wrist; one Dynamixel motor drives its fingers (`hand_<side>_motor_joint`,
  0 closed to about 0.785 rad open).
- **hand_root**: a frame fixed in each gripper's base (`<side>_hand_root` in the URDF); IK targets are this frame's pose
  in `world`. The policy's poses are of the gripper tip, `tcp_offset_m` (0.088 m) along hand_root's x; the runner
  converts between the two.
- **MIT command**: the Damiao motors' control frame: target position and velocity, kp, kd and a feedforward torque;
  the motor computes τ = kp·(q_ref − q) + kd·(q̇_ref − q̇) + τ_ff.
- **PA**: the sub-task a checkpoint is conditioned on next to its task, e.g. "Pick up one cup and place it onto the
  plate."; the server takes one per start: the checkpoint's own PA for `--stage` (default 0), or `--pa-text`.

### Control chain, per update

1. `motion` fits a C² cubic spline through each new chunk, from the current target and its velocity, samples
   it, and takes one DLS IK step under the joint velocity, acceleration and position limits. It writes q, q̇ and q̈ as
   references. A joint trajectory from the planner plays through the same references, so nothing switches.
2. `left/right_torque_controller` (`inverse_dynamics` mode) passes q and q̇ on and adds τ_ff = rnea(q, q̇, q̈):
   gravity, Coriolis and M·q̈, from the URDF with pinocchio.
3. `openarm_hardware` (enactic/openarm_ros2 0.9.2, with the patches in `docker/patches/`) sends the MIT command with
   kp, kd from the launch.

The patches add to the hardware plugin: a ROS node with a relax service, live gains, no refresh round in `read()` (1 kHz
needs it), holding the measured pose on activate, a velocity half-step, and a link monitor that relaxes the arm when a
motor reports an error or misses more than `LINK_FAULT_CYCLES` replies in a row.

## Interfaces

| Topic / service | Type | From → to |
|---|---|---|
| `/arm/<side>/cmd_pose_trajectory` | `trajectory_msgs/MultiDOFJointTrajectory` | runner → motion: hand_root poses, relative to the commanded hand when the chunk arrives (`frame_id` = `<side>_hand_root`) or absolute (`world`), each due at `header.stamp + time_from_start` |
| `/arm/joint_trajectory` | `trajectory_msgs/JointTrajectory` | planner → motion: a joint path for both arms |
| `/arm/hold` | `std_srvs/Trigger` | the command takes the measured pose and stops; clears a trip |
| `/arm/follow` | `std_srvs/SetBool` | true: the command stays on the measured pose as the arms move by hand (Manual), with no trip and no chunk or path played; `/arm/hold` ends it |
| `/arm/tripped` | `std_msgs/Bool` | motion → runner: a joint fell behind its command by more than `max_joint_lead`, so the arms hold |
| `/openarm_hw_relax_<side>/set_relax` | `std_srvs/SetBool` | relax mode of `openarm_hardware`: kp 0, the arm sinks damped |
| `/openarm_hw_relax_<side>/set_manual` | `std_srvs/SetBool` | manual mode, to move the arm by hand: kp ramps to 0 over `gain_ramp_s` (and back when it ends); kd and the torque controller's effort command (gravity at the measured pose while motion holds) stay; `set_relax` also ends it |
| `/joint_states` | `sensor_msgs/JointState` | joint_state_broadcaster → all: the 14 arm joints |
| `/arm/command` | `sensor_msgs/JointState` | motion → logs: q_cmd, q̇_cmd, and q̈_cmd in `effort` |
| `/arm/<side>/hand_pose`, `command_pose` | `geometry_msgs/PoseStamped` | motion → runner: the measured hand_root, and the hand pose spline (FK(q_cmd) during a planner path), in `world`; chunks anchor on the latter |
| `/gripper/joint_trajectory` | `trajectory_msgs/JointTrajectory` | runner, planner → gripper |
| `/gripper/joint_states` | `sensor_msgs/JointState` | gripper → all: `hand_<side>_motor_joint` |
| `/inference/state` | `std_msgs/String` | runner → gui: `running` or `paused: <why>` |
| `/inference/start`, `/inference/pause` | `std_srvs/Trigger` | gui → runner: start or pause inference; a paused run lets the current chunk's rows play out |
| `/<side>/<side>_arm_cam/image_raw/compressed` | `sensor_msgs/CompressedImage` | runner → gui, bags: the wrist images the policy sees (`publish_images`) |
| `/openarm_hw_relax_<side>/gains` | `std_msgs/Float64MultiArray` | `openarm_hardware` → logs: the current kp and kd; its parameters `kp_joints`, `kd_joints`, `gain_ramp_s` and `relax_kd` change them live |

## Policy

The policy server is `policy/airoa0_inference` (its own [README](policy/README.md)), on top of AIRoA-0 (pinned in
`scripts/serve_policy.sh`). A checkpoint is a directory in `checkpoints/` (mounted at `/checkpoints` in the container),
in the layout the AIRoA-0 release publishes under `examples/<task>/`: `model_config.json`, `params/`, `norm_stats.json`.
The repo ships no weights; get them from [huggingface.co/airoa-org/AIRoA-0](https://huggingface.co/airoa-org/AIRoA-0).
The repository is gated: log in to Hugging Face and accept the terms on the model page first. The checkpoints are Gemma
Model Derivatives, built on π0.5 / [openpi](https://github.com/Physical-Intelligence/openpi): their use is subject to the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms) and the
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy). See [NOTICE](NOTICE).

`scripts/serve_policy.sh /checkpoints/<task> --port 8000 --pa-text "<sub-task>"` clones AIRoA-0 into `.cache/AIRoA-0`
on its first start (or uses `AIRoA0_DIR`), builds its venv in `.cache/venvs/` (minutes, once), then serves. The image
provides uv and the GPU (`--gpus all`); CUDA comes from the venv's wheels. The server compiles before it listens, so
once `GET /metadata` answers it is ready (`ros2 run oap_runner prewarm` waits for that).

The runner posts one observation at a time to `POST /infer`: both wrist images, the right gripper tip seen from the left
one (xyz and a 6D rotation) and both finger openings, with the time and a session that restarts with every start. The
reply holds rows at 20 Hz: per arm the target tip pose relative to that tip at the observation, then both finger
openings. Layouts: [doc/policy.md](doc/policy.md).

The GUI lists every directory in `checkpoints/` with a `model_config.json`, and starts the server with the PA typed in its
server panel.

## Build

```sh
ROS_DOMAIN_ID=42 ./scripts/oap shell     # builds the image if needed, starts the container, a shell in it at /ws
colcon build --symlink-install && source install/setup.bash
```

`scripts/oap up`, `run`, and `shell` when no container runs, build the image when it is missing or `docker/`
changed since it was built (about 15–20 minutes the first time: pinocchio compiles from source). `./scripts/oap build [docker build args]` forces a
build, e.g. `--no-cache`. `OAP_BUILD_ARGS` goes to every build, e.g.
`export OAP_BUILD_ARGS="--build-arg CMEEL_JOBS=4"` on a host with little RAM.

The image provides every dependency of the packages, some of them built from source (pinocchio with CasADi support,
the patched `openarm_hardware`, `openarm_description`), so build the workspace in the container, not with
`rosdep install` on the host. After pulling a change under `docker/`, the next `oap up` (or `shell`, when no container runs) rebuilds the
image and recreates the container, unless control is running: then it keeps the old one until `./scripts/oap down`.
`oap shell` always enters a running container as it is, and only notes when it is stale or on another ROS domain;
`oap up` refuses a `ROS_DOMAIN_ID` that differs from the running container's.

`scripts/oap up` starts one long-running container, `openarm-pipeline-$USER`, in the domain `ROS_DOMAIN_ID` (which it
requires). `shell` starts it if needed and enters it; `exec <cmd>` runs a command in it and needs it running; `down`
stops what runs in the container, control last, then removes it; while control runs it asks first. `run <cmd>` is a throwaway container.

On a shared host give each user their own `ROS_DOMAIN_ID`: with `--network host` everyone in the same domain sees and
commands everyone's topics.

The container runs as root, so what it writes into the checkout (`build/`, `install/`, `log/`, `output/`, `.cache/`)
is owned by root. Remove it from the running container, `./scripts/oap exec rm -rf build install log`, or with `sudo` on the host.

## Container privileges

`scripts/oap` runs the container with `--privileged -v /dev:/dev --network host`, and `scripts/gui` runs
`xhost +SI:localuser:root`. Each is needed, and each widens what the container can do:

| Setting | Needed for | Effect |
|---|---|---|
| `--privileged` | `ip link` on the CAN interfaces (`scripts/canup.sh`), raw device access | the container is root on the host in all but name: all capabilities, no seccomp or AppArmor profile, every device |
| `-v /dev:/dev` | the CAN adapter, the U2D2 and the cameras, which appear and change at run time | every host device is visible in the container |
| `--network host` | SocketCAN interfaces live in the host's network namespace; DDS discovery; the policy server's port | the container shares the host's network: every port it opens is open on the host, port 8000 included |
| `xhost +SI:localuser:root` | the GUI, which runs as root in the container, drawing on the host's X display | local processes running as root, the container's among them, can connect to your X server and read its input and screen |

Run it only on a dedicated robot host, not on a shared or internet-facing machine. The policy server listens on all
interfaces without authentication: firewall port 8000, or keep the host on a closed network. Revert X access when done:

```sh
xhost -SI:localuser:root
```

## Safety

- Keep a hand on the e-stop whenever the motors have power. Nothing in software replaces it.
- Keep people, and anything that is not part of the task, out of both arms' reach while the motors have power. The
  planner checks only the table and the pillar, and policy chunks are not checked for collisions at all.
- First run everything on mock hardware (the default, no `hardware:=real`): the chain runs on `mock_components`, where
  the command becomes the state, so the planner, `circle` and the GUI work without the robot.
- Check a planner move with `--dry-run` before you send it: `plan --dry-run` plans from where the arms are, checks
  against the table and the pillar, and sends nothing; `circle --dry-run` checks the circle from the current command
  the same way. Without robot state (no control chain running, or the wrong `ROS_DOMAIN_ID`) a dry run fails like a
  real run; `--dry-run --from-canonical` checks from a canonical start instead (plan: the other of park/ready; circle:
  ready) and warns that it is not from where the arms are.
- On `real`, both arms hold where they are when control starts. Move to ready before inference; the GUI asks when the
  arms are not there.
- The runner refuses a chunk, and pauses, when an action is not finite, when its first row is more than
  `max_step_mm` / `max_step_deg` from the commanded hand, or when two rows in a row are further apart than
  `max_row_step_mm` / `max_row_step_deg`. It pauses when a camera or a state stream has no new sample for 3 s, and
  holds the arms (`/arm/hold`) when the server fails or does not answer.
- The arms hold when a joint lags its command by more than `max_joint_lead` or the command is not finite, and an arm
  relaxes when one of its motors stops replying or reports an error. When a motor does not reply at start, control does
  not start (neither arm is enabled); the log names the joint.
- Park before stopping control: the motors are disabled and the arms go limp.
- Relax is not yet verified on the robot: the arms may sink faster than expected. Use it only from Lifted, with a hand
  on the e-stop.
- If control stops during bring-up (a motor that does not reply is named in the control log), assume an arm may still
  be powered and holding: keep a hand on the e-stop, cut the motor power if an arm is enabled, fix the named joint,
  then start control again.

## Data and privacy

The wrist cameras see the workspace and anyone in it.

- The runner sends every observation, both wrist images included, to the policy server over plain HTTP
  (`server_host`, `server_port` in `src/oap_runner/config/runner.yaml`). Keep the server on the robot host or on a
  closed network (see [Container privileges](#container-privileges)). The server writes nothing to disk.
- With `publish_images: true` (the default) the runner also publishes the images as JPEG on
  `/<side>/<side>_arm_cam/image_raw/compressed`, for the GUI's camera views and for bags. Any process in the same ROS
  domain can subscribe to them, other hosts on the network included.
- `scripts/record.sh` and the GUI's bag button record the camera images, joint states and commands into `output/`
  (gitignored), and `oap_gui overlay` writes `overlay.mp4` into the bag's folder. Nothing is uploaded. To delete a
  recording, delete its folder; it is owned by root, so `./scripts/oap exec rm -rf output/<bag>`.
- The GUI keeps the last checkpoint and PA in `.cache/gui.json`.

## Run (real robot)

Pin `ROS_DOMAIN_ID`: a mismatch hides every topic without an error.

```sh
sudo ./scripts/canup.sh can0 && sudo ./scripts/canup.sh can1     # host, once per boot (or in the container)
ROS_DOMAIN_ID=88 ./scripts/oap shell                               # then, in the container:
colcon build --symlink-install && source install/setup.bash
ros2 launch oap_control robot.launch.py hardware:=real             # arms + gripper node: both arms hold where they are
  # left_can:=can1 right_can:=can0 are the defaults
ros2 run oap_planner plan --target ready                           # in another shell (./scripts/oap exec bash); --dry-run checks only
ros2 run oap_planner circle --radius 0.03                          # tracking test from ready
# the next three need a checkpoint in checkpoints/ (see Policy); the runner also needs the wrist cameras
./scripts/serve_policy.sh /checkpoints/<task> --port 8000 --pa-text "<sub-task>"   # in another shell
ros2 run oap_runner prewarm --port 8000                            # optional: waits until the server serves
ros2 run oap_runner runner --config src/oap_runner/config/runner.yaml
./scripts/record.sh cups                                           # a bag in output/
ros2 run oap_planner plan --target park                            # before stopping the launch: the arms go limp
```

Or all of it from one window, which starts the same commands and brings the CAN interfaces up on `real`:

```sh
./scripts/gui
```

The GUI needs an X11 display on the host. `scripts/gui` allows root's local X clients (`xhost +SI:localuser:root`),
builds the image if needed, starts the container in ROS domain `OAP_ROS_DOMAIN_ID` (default 88; set your own on a
shared host; it must match a container already running, e.g. `OAP_ROS_DOMAIN_ID=42 ./scripts/gui` after the quick
start), builds the workspace and opens the window. Its hardware selector picks mock or real. Closing the window
leaves every process it started running (control included); reopening it shows them again. Stop them with the GUI's
buttons, or all at once with `./scripts/oap down`, which stops what runs in the container, control last, and removes
it. While control runs, `down` first asks whether the arms are parked (without a terminal it refuses unless given
`--force` or `OAP_FORCE=1`).

The GUI's Start server runs the server on the chosen checkpoint with the PA from its server panel, in the same
container, and waits until it serves. The first start builds its venv (minutes). The GUI remembers the last checkpoint
and PA in `.cache/gui.json`. To change the PA, stop the server, edit it and start again. The runner stops when the
server does not answer within `REPLY_TIMEOUT_S`.

Relax lets both arms sink, damped with the startup kd (only from Lifted; not yet verified on the robot, see
[Safety](#safety)). Hold pauses inference and ends a move. Every
action, Hold included, first calls `/arm/hold`, which puts the command on the measured pose and clears a trip, then
leaves relax mode, so the arms never snap back to an old command. Start inference asks first when the arms are not at ready.

## Tests

Run them before a pull request (GitHub CI only lints: the image takes hours to build on a hosted runner):

```sh
./scripts/oap test   # colcon build -Werror, pytest, gtests, in a throwaway container; builds into .cache/test/
```

It needs no robot and no `ROS_DOMAIN_ID`, and a skipped test fails it. By hand, in the container from `/ws` (pin
pytest: newer releases break the `launch_testing` plugin that ROS 2 Humble loads into pytest):

```sh
colcon build && source install/setup.bash
pip3 install pytest==8.3.5 pytest-timeout==2.3.1
python3 -m pytest --timeout=120 tests/                        # unit tests: runner, planner, GUI
colcon test --packages-select oap_control && colcon test-result --verbose   # gtests: spline, validation, IK
```

`OAP_TESTS_NO_SKIP=1` turns a skipped test into a failure, as `oap test` does.

## Features and limitations

- Both arms and both grippers, driven by pose chunks from the AIRoA-0 policy server, or by joint trajectories from the
  planner, through one controller chain with no controller switching.
- Torque feedforward from the URDF (gravity, Coriolis, inertia), a static friction term and an integral on the joint
  error, on top of the motors' MIT position control; no impedance or force control.
- Collision checks against the table and the pillar in the ready / park planner only; policy chunks are checked by
  step size and joint lag, not for collisions.
- One robot: the bimanual OpenArm v1.0 with YUBI grippers and two wrist cameras. Other robots or camera layouts need
  code changes, not only configuration.
- One request at a time: the arms hold while the server answers, so they pause briefly between chunks.
- One sub-task (PA) per server start: the server does not move on to the next stage by itself.

## License

Apache-2.0: [LICENSE](LICENSE). Third-party code and credits: [NOTICE](NOTICE).

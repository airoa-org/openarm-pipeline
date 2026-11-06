# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
# The heavy, rarely changing part of the image: the OS, ROS and the other apt packages, pinocchio built from source
# (most of a cold build: over an hour on 2 cores), openarm_can and uv. docker/Dockerfile builds FROM it (BASE_IMAGE).
# scripts/oap builds it locally as openarm-pipeline-base:$USER before each build of docker/Dockerfile, with the same
# build args.
# Build context: the repository root, as for docker/Dockerfile.

# Pinned by digest (the multi-arch index of ros:humble-ros-base on 2026-11-03), so a rebuild starts from the same base.
FROM ros:humble-ros-base@sha256:d9bcf2e6f72df127fdee07048c2e87b1b4b9a59a74af0592f4489fc2844aa855
ENV DEBIAN_FRONTEND=noninteractive RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# ROS packages from a dated snapshot of packages.ros.org, which itself keeps only the latest version of each package:
# ROS_SNAPSHOT is the Humble sync the base image above was built from (ros2_control 2.54.3), so its ROS packages and
# the ones installed here match. The snapshots' signing key (4B63 CF8F DE49 746E 98FA 01DD AD19 BAB3 CBF1 25EA,
# expires 2027-06-01) is in docker/ros-snapshots.asc. Ubuntu's own packages still come from its archive. The base
# image's ros2.sources (packages.ros.org) belongs to ros2-apt-source, which the snapshot has a newer version of: purged,
# so that no upgrade puts packages.ros.org back (nothing depends on it).
ARG ROS_SNAPSHOT=2026-10-12
COPY docker/ros-snapshots.asc /usr/share/keyrings/ros-snapshots.asc
RUN apt-get purge -y ros2-apt-source && test ! -e /etc/apt/sources.list.d/ros2.sources \
    && printf '%s\n' 'Types: deb' "URIs: http://snapshots.ros.org/humble/${ROS_SNAPSHOT}/ubuntu" 'Suites: jammy' \
        'Components: main' 'Signed-By: /usr/share/keyrings/ros-snapshots.asc' \
        > /etc/apt/sources.list.d/ros2-snapshot.sources \
    && apt-get update && apt-get install -y --no-install-recommends can-utils \
        ros-humble-rmw-cyclonedds-cpp ros-humble-dynamixel-sdk ros-humble-xacro \
        ros-humble-ros2-control ros-humble-ros2-controllers ros-humble-realtime-tools \
        python3-pip python3-yaml \
    && rm -rf /var/lib/apt/lists/*

# The openarm_hardware patches rely on how ros2_control 2.54 behaves: every hardware component's on_init runs before
# any activates, and an exception in a constructor ends the node. The snapshot fixes the version; this check
# fails a change of ROS_SNAPSHOT to another minor version, so those assumptions are re-checked first.
RUN for p in ros-humble-ros2-control ros-humble-controller-manager ros-humble-hardware-interface; do \
        v=$(dpkg-query -W -f '${Version}' "$p") && case $v in 2.54.*) ;; *) \
            echo "$p is $v, not 2.54.x: re-check the ros2_control assumptions above, then update this check" >&2; \
            exit 1 ;; esac; \
    done

# Every pip install here takes its versions, transitive ones included, from base-constraints.txt; docker/Dockerfile's
# constraints.txt adds its own layers' pins to these (-c base-constraints.txt), so changing one of those leaves this
# image alone. Everything installed here is a wheel, apart from pinocchio, built without isolation.
COPY docker/base-constraints.txt /opt/oap/base-constraints.txt

# pinocchio from source against the casadi wheel: the ready planner needs pinocchio.casadi, which the pin wheel lacks.
# The C++ arm node links the same build through its cmeel prefix, so one pinocchio serves both. cmeel-boost requires
# numpy>=2, so the build runs with NUMPY_BUILD_VERSION (base-constraints.txt with that numpy) and numpy then goes down
# to base-constraints.txt's 1.24.4, which the image keeps: the numpy 2 builds also run on numpy 1.x, and the check
# below imports pinocchio.casadi with it.
# CMEEL_JOBS: parallel compile jobs for this build. Each job can take a few GB of RAM, so when empty it is the CPU
# count capped at one job per 4 GB of RAM (at least 1). Set it to override, in OAP_BUILD_ARGS so that the builds
# oap up and oap run start get it too (another value rebuilds this layer): export OAP_BUILD_ARGS="--build-arg CMEEL_JOBS=4"
ARG PINOCCHIO_VERSION=4.0.0
ARG CMEEL_JOBS=
ARG NUMPY_BUILD_VERSION=2.2.6
RUN grep -v '^numpy==' /opt/oap/base-constraints.txt > /tmp/build-constraints.txt \
    && echo "numpy==${NUMPY_BUILD_VERSION}" >> /tmp/build-constraints.txt \
    && pip3 install --no-cache-dir -c /tmp/build-constraints.txt casadi "cmeel[build]" cmeel-boost \
        "cmeel-urdfdom[build]" "libcoal[build]" "coal[build]" example-robot-data \
    && jobs=$(nproc) && mem=$(awk '/MemTotal/ {print int($2 / 4194304)}' /proc/meminfo) \
    && if [ "$mem" -lt "$jobs" ]; then jobs=$mem; fi && if [ "$jobs" -lt 1 ]; then jobs=1; fi \
    && export CMEEL_JOBS="${CMEEL_JOBS:-$jobs}" CMEEL_RUN_TESTS=OFF CMEEL_CMAKE_ARGS="-DBUILD_WITH_CASADI_SUPPORT=ON -Dcasadi_DIR=$(python3 -c 'import casadi, os; print(os.path.join(os.path.dirname(casadi.__file__), "cmake"))')" \
    && for p in libpinocchio==${PINOCCHIO_VERSION} pin==${PINOCCHIO_VERSION}; do \
        pip3 install --no-cache-dir -c /tmp/build-constraints.txt --no-build-isolation --no-deps --force-reinstall \
            --no-binary "${p%%==*}" "$p" || exit 1; \
    done \
    && pip3 install --no-cache-dir -c /opt/oap/base-constraints.txt numpy && rm /tmp/build-constraints.txt
# casadi lives in its wheel: pinocchio loads libcasadi without an `import casadi` first, and its CMake config finds it.
ENV CMAKE_PREFIX_PATH=/usr/local/lib/python3.10/dist-packages/cmeel.prefix:/usr/local/lib/python3.10/dist-packages/casadi \
    LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/cmeel.prefix/lib:/usr/local/lib/python3.10/dist-packages/casadi
RUN python3 -c "import numpy, pinocchio.casadi; assert numpy.__version__ < '1.25', numpy.__version__"

# openarm_can (Apache-2.0), the CAN library openarm_hardware is built against (docker/Dockerfile), from a pinned tag
# of enactic/openarm_can (its PPA publishes only the latest version). It installs to /usr/local, with its CMake package
# (find_package(OpenArmCAN)) under lib/cmake/OpenArmCAN; CLI11 is for its command-line tools.
ARG OPENARM_CAN_REF=1.4.1
RUN apt-get update && apt-get install -y --no-install-recommends libcli11-dev \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --branch ${OPENARM_CAN_REF} --depth 1 https://github.com/enactic/openarm_can.git /tmp/openarm_can \
    && cmake -S /tmp/openarm_can -B /tmp/openarm_can/build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
    && cmake --build /tmp/openarm_can/build -j"$(nproc)" && cmake --install /tmp/openarm_can/build \
    && ldconfig && rm -rf /tmp/openarm_can

# policy server: a bundle's run_policy_server.sh builds its own venv with uv; CUDA comes from the venv's wheels
# (jax[cuda12]), so the image needs only uv, its Python 3.11, the driver that --gpus passes in, and clang for the
# sdists the venv builds (evdev, through lerobot)
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
RUN apt-get update && apt-get install -y --no-install-recommends git-lfs clang python3-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir -c /opt/oap/base-constraints.txt uv && uv python install 3.11

#!/bin/bash
# Usage: source /path/to/state_nav/docker/startup.sh
#
# Edit STATENAV_SRC in the "User configuration" block below, or override via env:
#   export STATENAV_SRC=...
#
# Python deps (torch, cupy, numpy, etc.) are installed in the Docker image via
# Dockerfile.x64 + requirements.txt. This script registers the mounted source
# as an editable package and runs colcon build so `ros2 run statenav_global ...` works.
#
# Source other ROS workspaces (core_ws, elevation_mapping_ws, etc.) separately
# before running nodes that depend on them.

echo "Running initial setup..."

# =============================================================================
# User configuration — edit the default path below
# =============================================================================
HOST_HOME_DIR="${HOST_HOME_DIR:-$HOME}"

STATENAV_SRC="${STATENAV_SRC:-${HOST_HOME_DIR}/ros2_ws/statenav_ws/src/state_nav}"
STATENAV_WS="${STATENAV_WS:-$(cd "$(dirname "${STATENAV_SRC}")/.." && pwd)}"
# =============================================================================

export HOST_HOME_DIR STATENAV_SRC STATENAV_WS

echo "HOST_HOME_DIR: ${HOST_HOME_DIR}"
echo "STATENAV_SRC:  ${STATENAV_SRC}"
echo "STATENAV_WS:   ${STATENAV_WS}"

BASHRC_MARKER="# state_nav docker workspace"

if ! grep -qF "${BASHRC_MARKER}" ~/.bashrc; then
    cat >> ~/.bashrc <<EOF

# state_nav docker workspace
export HOST_HOME_DIR="${HOST_HOME_DIR}"
export STATENAV_SRC="${STATENAV_SRC}"
export STATENAV_WS="${STATENAV_WS}"
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
source /opt/ros/humble/setup.bash
source "${STATENAV_WS}/install/setup.bash"
EOF
    echo ".bashrc configured successfully"
else
    sed -i "s|^export HOST_HOME_DIR=.*|export HOST_HOME_DIR=\"${HOST_HOME_DIR}\"|" ~/.bashrc
    sed -i "s|^export STATENAV_SRC=.*|export STATENAV_SRC=\"${STATENAV_SRC}\"|" ~/.bashrc
    if grep -qF 'export STATENAV_WS=' ~/.bashrc; then
        sed -i "s|^export STATENAV_WS=.*|export STATENAV_WS=\"${STATENAV_WS}\"|" ~/.bashrc
    else
        sed -i "/^export STATENAV_SRC=/a export STATENAV_WS=\"${STATENAV_WS}\"" ~/.bashrc
    fi
    if ! grep -qF "${STATENAV_WS}/install/setup.bash" ~/.bashrc; then
        sed -i '/source \/opt\/ros\/humble\/setup.bash/a source "'"${STATENAV_WS}"'/install/setup.bash"' ~/.bashrc
    fi
    echo ".bashrc updated with current paths"
fi

source /opt/ros/humble/setup.bash

echo -e "Installing STATE-NAV (editable)..."
if [ ! -d "${STATENAV_SRC}" ]; then
    echo "Warning: ${STATENAV_SRC} not found"
    exit 1
fi

python3 -m pip install -e "${STATENAV_SRC}"

echo -e "Building ROS2 workspace at ${STATENAV_WS}..."
if ! command -v colcon >/dev/null 2>&1; then
    echo "Installing colcon..."
    apt-get update && apt-get install -y --no-install-recommends python3-colcon-common-extensions
fi

cd "${STATENAV_WS}" || exit 1
# pip install -e above keeps Python imports editable; avoid --symlink-install here
# because newer setuptools rejects setup.py develop --editable.
if ! colcon build --packages-select statenav_global; then
    echo "ERROR: colcon build failed. ros2 run will not work until this succeeds."
    return 1 2>/dev/null || exit 1
fi
source "${STATENAV_WS}/install/setup.bash"

source ~/.bashrc

echo -e "Done. Try: ros2 run statenav_global main_worldmodel"
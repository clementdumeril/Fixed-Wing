#!/bin/bash
# One-time setup on Ubuntu 24.04 with ROS 2 Jazzy, after PX4's Tools/setup/ubuntu.sh and `make px4_sitl`:
# the Micro XRCE-DDS agent (PX4 <-> ROS 2), px4_msgs generated from this PX4 checkout's messages, and the policy node.
# No sudo. bash setup_wsl.sh [PX4 checkout, default ~/PX4-Autopilot] [RLtools include directory]
set -e
PX4=${1:-$HOME/PX4-Autopilot}
RL_TOOLS_INCLUDE=${2:?RLtools include directory, e.g. /mnt/d/Work/fixed-wing-cpp/build/_deps/rl_tools-src/include}
HERE=$(cd "$(dirname "$0")" && pwd)

if [ ! -x "$HOME/xrce/bin/MicroXRCEAgent" ]; then
    [ -d "$HOME/Micro-XRCE-DDS-Agent" ] || git clone -b v2.4.3 https://github.com/eProsima/Micro-XRCE-DDS-Agent.git "$HOME/Micro-XRCE-DDS-Agent"
    cmake -S "$HOME/Micro-XRCE-DDS-Agent" -B "$HOME/Micro-XRCE-DDS-Agent/build" -DCMAKE_INSTALL_PREFIX="$HOME/xrce"
    cmake --build "$HOME/Micro-XRCE-DDS-Agent/build" -j4  # superbuild: the first build fetches and builds Fast DDS
    cmake --install "$HOME/Micro-XRCE-DDS-Agent/build"
fi

mkdir -p "$HOME/px4_ws/src"
if [ ! -d "$HOME/px4_ws/src/px4_msgs" ]; then
    git clone https://github.com/PX4/px4_msgs.git "$HOME/px4_ws/src/px4_msgs"
fi
# the messages of the PX4 that runs, so that the two sides agree
rm -f "$HOME/px4_ws/src/px4_msgs/msg/"*.msg
cp "$PX4/msg/"*.msg "$PX4/msg/versioned/"*.msg "$HOME/px4_ws/src/px4_msgs/msg/"
ln -sfn "$HERE/fw_policy" "$HOME/px4_ws/src/fw_policy"

source /opt/ros/jazzy/setup.bash
cd "$HOME/px4_ws"
colcon build --packages-select px4_msgs fw_policy --cmake-args -DRL_TOOLS_INCLUDE="$RL_TOOLS_INCLUDE"
echo "setup done"

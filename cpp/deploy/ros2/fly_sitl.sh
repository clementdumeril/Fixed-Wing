#!/bin/bash
# One SITL flight of the policy: PX4 with Gazebo's advanced_plane (headless), the DDS agent, PX4's own takeoff,
# then the policy node, which switches PX4 to offboard and hands back to loiter at the end.
# bash fly_sitl.sh [output directory] [extra ROS parameters, e.g. -p start_bearing_deg:=120.0]
PX4=${PX4:-$HOME/PX4-Autopilot}
OUT=${1:-$HOME/fw_flights/$(date +%Y%m%d_%H%M%S)}
shift
mkdir -p "$OUT"

cleanup(){ pkill -f MicroXRCEAgent; pkill -x px4; pkill -f "gz sim"; }
trap cleanup EXIT
cleanup 2>/dev/null; sleep 1

LD_LIBRARY_PATH="$HOME/xrce/lib:$LD_LIBRARY_PATH" "$HOME/xrce/bin/MicroXRCEAgent" udp4 -p 8888 > "$OUT/agent.log" 2>&1 &
# PX4's shell reads stdin: keep it open
(cd "$PX4" && tail -f /dev/null | HEADLESS=1 make px4_sitl gz_advanced_plane > "$OUT/px4.log" 2>&1) &
for i in $(seq 180); do grep -q "Gazebo world is ready" "$OUT/px4.log" && break; sleep 1; done
sleep 5
cd "$PX4/build/px4_sitl_default/rootfs"
# no ground station here: the airframe's datalink-loss action (NAV_DLL_ACT 2) would block arming
../bin/px4-param set NAV_DLL_ACT 0 > /dev/null
../bin/px4-param set MIS_TAKEOFF_ALT 60 > /dev/null
for i in $(seq 120); do grep -q "Ready for takeoff" "$OUT/px4.log" && break; sleep 1; done
grep -q "Ready for takeoff" "$OUT/px4.log" || { echo "PX4 not ready, see $OUT/px4.log"; exit 1; }
../bin/px4-commander takeoff
echo "PX4 is taking off; the policy takes over above 40 m"
# ROS only now: its setup points gz to ROS's own Gazebo configuration, which hides the system's gz sim
source /opt/ros/jazzy/setup.bash
source "$HOME/px4_ws/install/setup.bash"
ros2 run fw_policy fw_policy_node --ros-args -p log_path:="$OUT/flight.csv" "$@" 2>&1 | tee "$OUT/node.log"
echo "flight written to $OUT"

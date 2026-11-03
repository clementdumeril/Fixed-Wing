#!/bin/bash
# One SITL flight of the policy: PX4 with Gazebo's advanced_plane (headless), the DDS agent, PX4's own takeoff,
# then the policy node, which switches PX4 to offboard and hands back to loiter at the end.
# bash fly_sitl.sh [output directory] [extra ROS parameters, e.g. -p start_bearing_deg:=120.0]
# IDENTIFY=1 bash fly_sitl.sh ...: identification flight (doublets, see fw_policy_node.cpp), higher and straight ahead,
# with Gazebo's joint states (servo angles, rotor speed) recorded to joints.jsonl. It adds Gazebo's JointStatePublisher
# to the advanced_plane model of the PX4 checkout (it only publishes; git checkout of the model undoes it).
PX4=${PX4:-$HOME/PX4-Autopilot}
OUT=${1:-$HOME/fw_flights/$(date +%Y%m%d_%H%M%S)}
[ $# -gt 0 ] && shift
mkdir -p "$OUT"
MODEL="$PX4/Tools/simulation/gz/models/advanced_plane/model.sdf"

cleanup(){ pkill -f MicroXRCEAgent; pkill -x px4; pkill -f "gz sim"; pkill -f "gz topic"; }
trap cleanup EXIT
cleanup 2>/dev/null; sleep 1

if [ "$IDENTIFY" = 1 ] && ! grep -q JointStatePublisher "$MODEL"; then
    # before the last </model>: the model's own closing tag
    tac "$MODEL" | sed '0,/<\/model>/s//<\/model>\n    <plugin filename="gz-sim-joint-state-publisher-system" name="gz::sim::systems::JointStatePublisher"\/>/' | tac > "$MODEL.new"
    mv "$MODEL.new" "$MODEL"
fi

LD_LIBRARY_PATH="$HOME/xrce/lib:$LD_LIBRARY_PATH" "$HOME/xrce/bin/MicroXRCEAgent" udp4 -p 8888 > "$OUT/agent.log" 2>&1 &
# PX4's shell reads stdin: keep it open
(cd "$PX4" && tail -f /dev/null | HEADLESS=1 make px4_sitl gz_advanced_plane > "$OUT/px4.log" 2>&1) &
for i in $(seq 180); do grep -q "Gazebo world is ready" "$OUT/px4.log" && break; sleep 1; done
sleep 5
if [ "$IDENTIFY" = 1 ]; then
    gz topic -e -t /world/default/model/advanced_plane_0/joint_state --json-output > "$OUT/joints.jsonl" 2>/dev/null &
fi
cd "$PX4/build/px4_sitl_default/rootfs"
# no ground station here: the airframe's datalink-loss action (NAV_DLL_ACT 2) would block arming
../bin/px4-param set NAV_DLL_ACT 0 > /dev/null
START=40
if [ "$IDENTIFY" = 1 ]; then
    ../bin/px4-param set MIS_TAKEOFF_ALT 90 > /dev/null
    START=70
    set -- -p identify:=true -p wait_for_loiter:=true -p start_bearing_deg:=0.0 -p start_altitude:=$START.0 -p duration:=48.0 "$@"
else
    ../bin/px4-param set MIS_TAKEOFF_ALT 60 > /dev/null
fi
for i in $(seq 120); do grep -q "Ready for takeoff" "$OUT/px4.log" && break; sleep 1; done
grep -q "Ready for takeoff" "$OUT/px4.log" || { echo "PX4 not ready, see $OUT/px4.log"; exit 1; }
../bin/px4-commander takeoff
echo "PX4 is taking off; the policy takes over above $START m"
# ROS only now: its setup points gz to ROS's own Gazebo configuration, which hides the system's gz sim
source /opt/ros/jazzy/setup.bash
source "$HOME/px4_ws/install/setup.bash"
ros2 run fw_policy fw_policy_node --ros-args -p log_path:="$OUT/flight.csv" "$@" 2>&1 | tee "$OUT/node.log"
echo "flight written to $OUT"

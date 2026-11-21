cmake_minimum_required(VERSION 3.16)
project(fixed-wing-rltools CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(MSVC)
    add_compile_options(/bigobj /permissive- /fp:fast)
endif()

# RLtools (https://github.com/rl-tools/rl-tools), header-only: a local checkout, or the tested commit fetched into the
# build directory
set(RL_TOOLS_DIR "" CACHE PATH "RLtools checkout (fetched if empty)")
if(RL_TOOLS_DIR)
    add_subdirectory(${RL_TOOLS_DIR} rl_tools)
else()
    include(FetchContent)
    FetchContent_Declare(rl_tools
        GIT_REPOSITORY https://github.com/rl-tools/rl-tools.git
        GIT_TAG b32d9985c65a5e098a6bbf190fd994962d288b99)
    FetchContent_MakeAvailable(rl_tools)
endif()

find_package(Threads REQUIRED)
set(FW_TARGETS)
function(fw_executable target source)
    add_executable(${target} ${source})
    target_include_directories(${target} PRIVATE include)
    target_link_libraries(${target} PRIVATE RLtools::RLtools)
    if(ARGN)
        target_compile_definitions(${target} PRIVATE ${ARGN})
    endif()
    set(FW_TARGETS ${FW_TARGETS} ${target} PARENT_SCOPE)
endfunction()

# replays flights recorded in Python, for tools/compare_with_python.py
fw_executable(check src/check.cpp)

# first task (airspeed, altitude, course) on the Aerosonde with the settings of python/examples/train_sac.py: SAC,
# critic 256, batch 256, discount 0.99, an update at every step
fw_executable(train_sac src/train_sac.cpp)

# tracking a moving point on PX4's advanced_plane, with the position error also observed on a fine scale (65 inputs):
# SAC, critic 128, batch 256, discount 0.995, the critics updated every 10 steps and the actor every 20 (L2F paper).
# The precision recipe (slurm/tracking_px4plane_reward.slurm)
fw_executable(train_sac_c128_b256_tracking_g995_px4plane_i10a20_fine src/train_sac.cpp
    FW_CRITIC_HIDDEN=128 FW_BATCH=256 FW_TRACKING=1 FW_GAMMA=0.995 FW_AIRPLANE=1 FW_TRAINING_INTERVAL=10
    FW_ACTOR_INTERVAL_FACTOR=2 FW_FINE_POSITION=1)

# the same in wind, turbulence, gusts and thermals, with randomized airplanes (slurm/tracking_px4plane_simple.slurm)
fw_executable(train_sac_c128_b256_tracking_g995_px4plane_i10a20_fine_wind src/train_sac.cpp
    FW_CRITIC_HIDDEN=128 FW_BATCH=256 FW_TRACKING=1 FW_GAMMA=0.995 FW_AIRPLANE=1 FW_TRAINING_INTERVAL=10
    FW_ACTOR_INTERVAL_FACTOR=2 FW_FINE_POSITION=1 FW_DISTURBANCES=1)

# the wind models against the Python ones, for tools/compare_wind_with_python.py
fw_executable(check_wind src/check_wind.cpp FW_TRACKING=1 FW_AIRPLANE=1 FW_DISTURBANCES=1)

# tracking on the lab's E-flite UMX Eratix 3D FF 450 mm (FW_AIRPLANE=2, docs/eratix_model.md), same networks and update
# intervals, without the fine observation (slurm/tracking_eratix.slurm)
fw_executable(train_sac_c128_b256_tracking_g995_eratix_i10a20 src/train_sac.cpp
    FW_CRITIC_HIDDEN=128 FW_BATCH=256 FW_TRACKING=1 FW_GAMMA=0.995 FW_AIRPLANE=2 FW_TRAINING_INTERVAL=10
    FW_ACTOR_INTERVAL_FACTOR=2)

# trim, thrust, stall speed, linear modes and a hands-off flight of the Eratix model
fw_executable(eratix_trim src/eratix_trim.cpp FW_TRACKING=1 FW_AIRPLANE=2)

# a flight of the deployed policy in Gazebo (deploy/ros2/fly_sitl.sh) against this simulator
fw_executable(compare_sitl src/compare_sitl.cpp FW_TRACKING=1 FW_AIRPLANE=1)
target_include_directories(compare_sitl PRIVATE deploy/ros2/fw_policy/policy)

# a trained advanced_plane tracking actor (actor.h or best_actor.h, read at run time) when the reference point
# accelerates: circles, helices, figure-eights and speed oscillations, from level and random starts. evaluate_reference
# for the actors without the fine observation (62 inputs, as the deployed policy), evaluate_reference_fine for the
# actors of the _fine targets (65 inputs)
fw_executable(evaluate_reference src/evaluate_reference.cpp FW_TRACKING=1 FW_AIRPLANE=1)
fw_executable(evaluate_reference_fine src/evaluate_reference.cpp FW_TRACKING=1 FW_AIRPLANE=1 FW_FINE_POSITION=1)
target_link_libraries(evaluate_reference PRIVATE Threads::Threads)
target_link_libraries(evaluate_reference_fine PRIVATE Threads::Threads)

list(REMOVE_ITEM FW_TARGETS check check_wind)
if(NOT MSVC)
    foreach(target ${FW_TARGETS})
        target_compile_options(${target} PRIVATE -O3 -march=native)
    endforeach()
endif()

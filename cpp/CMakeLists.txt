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

# replays flights recorded in Python, for tools/compare_with_python.py
add_executable(check src/check.cpp)
target_include_directories(check PRIVATE include)
target_link_libraries(check PRIVATE RLtools::RLtools)

# SAC training and evaluation on the Python test scenarios
add_executable(train_sac src/train_sac.cpp)
target_include_directories(train_sac PRIVATE include)
target_link_libraries(train_sac PRIVATE RLtools::RLtools)

# same with the L2F network and batch sizes (critic 64, batch 64): much less computation per step
add_executable(train_sac_small src/train_sac.cpp)
target_include_directories(train_sac_small PRIVATE include)
target_link_libraries(train_sac_small PRIVATE RLtools::RLtools)
target_compile_definitions(train_sac_small PRIVATE FW_CRITIC_HIDDEN=64 FW_BATCH=64)

# more critic:batch sizes, for a sweep between the two above (slurm/sweep.slurm)
set(FW_SWEEP "256:64;128:128;128:64;64:256" CACHE STRING "critic:batch sizes of the extra train_sac variants")
set(TRAIN_TARGETS train_sac train_sac_small)
foreach(size ${FW_SWEEP})
    string(REPLACE ":" ";" parts ${size})
    list(GET parts 0 critic)
    list(GET parts 1 batch)
    set(target train_sac_c${critic}_b${batch})
    add_executable(${target} src/train_sac.cpp)
    target_include_directories(${target} PRIVATE include)
    target_link_libraries(${target} PRIVATE RLtools::RLtools)
    target_compile_definitions(${target} PRIVATE FW_CRITIC_HIDDEN=${critic} FW_BATCH=${batch})
    list(APPEND TRAIN_TARGETS ${target})
endforeach()

# critic 256, batch 64, with the time integrals of the errors in the observation (63 values instead of 60)
add_executable(train_sac_c256_b64_integral src/train_sac.cpp)
target_include_directories(train_sac_c256_b64_integral PRIVATE include)
target_link_libraries(train_sac_c256_b64_integral PRIVATE RLtools::RLtools)
target_compile_definitions(train_sac_c256_b64_integral PRIVATE FW_CRITIC_HIDDEN=256 FW_BATCH=64 FW_ERROR_INTEGRAL=1)
list(APPEND TRAIN_TARGETS train_sac_c256_b64_integral)

# critic 256, batch 64, on the position and velocity tracking task
add_executable(train_sac_c256_b64_tracking src/train_sac.cpp)
target_include_directories(train_sac_c256_b64_tracking PRIVATE include)
target_link_libraries(train_sac_c256_b64_tracking PRIVATE RLtools::RLtools)
target_compile_definitions(train_sac_c256_b64_tracking PRIVATE FW_CRITIC_HIDDEN=256 FW_BATCH=64 FW_TRACKING=1)
list(APPEND TRAIN_TARGETS train_sac_c256_b64_tracking)

# the same with a longer horizon: discount factor 0.995 (about 4 s) and 0.998 (about 10 s) instead of 0.99 (2 s)
foreach(gamma 995 998)
    set(target train_sac_c256_b64_tracking_g${gamma})
    add_executable(${target} src/train_sac.cpp)
    target_include_directories(${target} PRIVATE include)
    target_link_libraries(${target} PRIVATE RLtools::RLtools)
    target_compile_definitions(${target} PRIVATE FW_CRITIC_HIDDEN=256 FW_BATCH=64 FW_TRACKING=1 FW_GAMMA=0.${gamma})
    list(APPEND TRAIN_TARGETS ${target})
endforeach()

# tracking with a discount factor of 0.995 and the integral of the position error in the observation (65 values)
add_executable(train_sac_c256_b64_tracking_g995_integral src/train_sac.cpp)
target_include_directories(train_sac_c256_b64_tracking_g995_integral PRIVATE include)
target_link_libraries(train_sac_c256_b64_tracking_g995_integral PRIVATE RLtools::RLtools)
target_compile_definitions(train_sac_c256_b64_tracking_g995_integral PRIVATE FW_CRITIC_HIDDEN=256 FW_BATCH=64 FW_TRACKING=1 FW_GAMMA=0.995 FW_ERROR_INTEGRAL=1)
list(APPEND TRAIN_TARGETS train_sac_c256_b64_tracking_g995_integral)

# the same as tracking_g995 with 4, 8 or 16 airplanes in parallel: one network update per step of all of them
foreach(n 4 8 16)
    set(target train_sac_c256_b64_tracking_g995_e${n})
    add_executable(${target} src/train_sac.cpp)
    target_include_directories(${target} PRIVATE include)
    target_link_libraries(${target} PRIVATE RLtools::RLtools)
    target_compile_definitions(${target} PRIVATE FW_CRITIC_HIDDEN=256 FW_BATCH=64 FW_TRACKING=1 FW_GAMMA=0.995 FW_N_ENVIRONMENTS=${n})
    list(APPEND TRAIN_TARGETS ${target})
endforeach()

# the same as tracking_g995 with a smaller critic: the 256 x 256 layer is most of the cost of an update
foreach(critic 128 64)
    set(target train_sac_c${critic}_b64_tracking_g995)
    add_executable(${target} src/train_sac.cpp)
    target_include_directories(${target} PRIVATE include)
    target_link_libraries(${target} PRIVATE RLtools::RLtools)
    target_compile_definitions(${target} PRIVATE FW_CRITIC_HIDDEN=${critic} FW_BATCH=64 FW_TRACKING=1 FW_GAMMA=0.995)
    list(APPEND TRAIN_TARGETS ${target})
endforeach()

# the same with a network update every 4 or 8 steps only: steps 4 or 8 times cheaper, so more of them for the same time
foreach(critic 128 64)
    foreach(interval 4 8)
        set(target train_sac_c${critic}_b64_tracking_g995_i${interval})
        add_executable(${target} src/train_sac.cpp)
        target_include_directories(${target} PRIVATE include)
        target_link_libraries(${target} PRIVATE RLtools::RLtools)
        target_compile_definitions(${target} PRIVATE FW_CRITIC_HIDDEN=${critic} FW_BATCH=64 FW_TRACKING=1 FW_GAMMA=0.995 FW_TRAINING_INTERVAL=${interval})
        list(APPEND TRAIN_TARGETS ${target})
    endforeach()
endforeach()

# the tracking task on PX4's advanced_plane instead of the Aerosonde, to fly the policy in Gazebo with PX4 SITL;
# critic:batch sizes, with the batch of 256 of Learning to Fly in Seconds (critic 64)
foreach(size 256:64 128:64 128:256 64:256)
    string(REPLACE ":" ";" parts ${size})
    list(GET parts 0 critic)
    list(GET parts 1 batch)
    set(target train_sac_c${critic}_b${batch}_tracking_g995_px4plane)
    add_executable(${target} src/train_sac.cpp)
    target_include_directories(${target} PRIVATE include)
    target_link_libraries(${target} PRIVATE RLtools::RLtools)
    target_compile_definitions(${target} PRIVATE FW_CRITIC_HIDDEN=${critic} FW_BATCH=${batch} FW_TRACKING=1 FW_GAMMA=0.995 FW_AIRPLANE=1)
    list(APPEND TRAIN_TARGETS ${target})
endforeach()

# critic 128, batch 256 on the advanced_plane with the update intervals of the L2F paper: the critics every 10 steps,
# the actor every 20
add_executable(train_sac_c128_b256_tracking_g995_px4plane_i10a20 src/train_sac.cpp)
target_include_directories(train_sac_c128_b256_tracking_g995_px4plane_i10a20 PRIVATE include)
target_link_libraries(train_sac_c128_b256_tracking_g995_px4plane_i10a20 PRIVATE RLtools::RLtools)
target_compile_definitions(train_sac_c128_b256_tracking_g995_px4plane_i10a20 PRIVATE FW_CRITIC_HIDDEN=128 FW_BATCH=256 FW_TRACKING=1 FW_GAMMA=0.995 FW_AIRPLANE=1 FW_TRAINING_INTERVAL=10 FW_ACTOR_INTERVAL_FACTOR=2)
list(APPEND TRAIN_TARGETS train_sac_c128_b256_tracking_g995_px4plane_i10a20)

# the last two differences with the L2F paper, with its update intervals and batch: the critic of 64 (SAC), and TD3
# (its critic of 64 too, see src/train_sac.cpp)
foreach(algorithm sac td3)
    set(target train_${algorithm}_c64_b256_tracking_g995_px4plane_i10a20)
    add_executable(${target} src/train_sac.cpp)
    target_include_directories(${target} PRIVATE include)
    target_link_libraries(${target} PRIVATE RLtools::RLtools)
    target_compile_definitions(${target} PRIVATE FW_CRITIC_HIDDEN=64 FW_BATCH=256 FW_TRACKING=1 FW_GAMMA=0.995 FW_AIRPLANE=1 FW_TRAINING_INTERVAL=10 FW_ACTOR_INTERVAL_FACTOR=2)
    if(algorithm STREQUAL "td3")
        target_compile_definitions(${target} PRIVATE FW_TD3=1)
    endif()
    list(APPEND TRAIN_TARGETS ${target})
endforeach()

# the L2F recipe (SAC, critic 128) with the integral of the position error in the observation (65 values instead of
# 62), to remove the steady offset to the point (results/precision_px4plane)
add_executable(train_sac_c128_b256_tracking_g995_px4plane_i10a20_integral src/train_sac.cpp)
target_include_directories(train_sac_c128_b256_tracking_g995_px4plane_i10a20_integral PRIVATE include)
target_link_libraries(train_sac_c128_b256_tracking_g995_px4plane_i10a20_integral PRIVATE RLtools::RLtools)
target_compile_definitions(train_sac_c128_b256_tracking_g995_px4plane_i10a20_integral PRIVATE FW_CRITIC_HIDDEN=128 FW_BATCH=256 FW_TRACKING=1 FW_GAMMA=0.995 FW_AIRPLANE=1 FW_TRAINING_INTERVAL=10 FW_ACTOR_INTERVAL_FACTOR=2 FW_ERROR_INTEGRAL=1)
list(APPEND TRAIN_TARGETS train_sac_c128_b256_tracking_g995_px4plane_i10a20_integral)

# the L2F recipe with the position error also observed on a fine scale (65 values instead of 62), so that a few meters
# of error count in the commands (results/near_px4plane: the error left is a steady offset of 3-4 m)
add_executable(train_sac_c128_b256_tracking_g995_px4plane_i10a20_fine src/train_sac.cpp)
target_include_directories(train_sac_c128_b256_tracking_g995_px4plane_i10a20_fine PRIVATE include)
target_link_libraries(train_sac_c128_b256_tracking_g995_px4plane_i10a20_fine PRIVATE RLtools::RLtools)
target_compile_definitions(train_sac_c128_b256_tracking_g995_px4plane_i10a20_fine PRIVATE FW_CRITIC_HIDDEN=128 FW_BATCH=256 FW_TRACKING=1 FW_GAMMA=0.995 FW_AIRPLANE=1 FW_TRAINING_INTERVAL=10 FW_ACTOR_INTERVAL_FACTOR=2 FW_FINE_POSITION=1)
list(APPEND TRAIN_TARGETS train_sac_c128_b256_tracking_g995_px4plane_i10a20_fine)

# TD3 as above with half the exploration noise (0.05 instead of 0.1): its policies switched the surfaces between their
# limits at about the control rate (results/l2f_td3_px4plane)
add_executable(train_td3_c64_b256_tracking_g995_px4plane_i10a20_n005 src/train_sac.cpp)
target_include_directories(train_td3_c64_b256_tracking_g995_px4plane_i10a20_n005 PRIVATE include)
target_link_libraries(train_td3_c64_b256_tracking_g995_px4plane_i10a20_n005 PRIVATE RLtools::RLtools)
target_compile_definitions(train_td3_c64_b256_tracking_g995_px4plane_i10a20_n005 PRIVATE FW_CRITIC_HIDDEN=64 FW_BATCH=256 FW_TRACKING=1 FW_GAMMA=0.995 FW_AIRPLANE=1 FW_TRAINING_INTERVAL=10 FW_ACTOR_INTERVAL_FACTOR=2 FW_TD3=1 FW_EXPLORATION_NOISE=0.05)
list(APPEND TRAIN_TARGETS train_td3_c64_b256_tracking_g995_px4plane_i10a20_n005)

# the lab's E-flite UMX Eratix 3D FF 450 mm (FW_AIRPLANE=2, docs/eratix_model.md) with the current best recipe of the
# advanced_plane: SAC, critic 128, batch 256, the L2F update intervals (critics every 10 steps, actor every 20)
add_executable(train_sac_c128_b256_tracking_g995_eratix_i10a20 src/train_sac.cpp)
target_include_directories(train_sac_c128_b256_tracking_g995_eratix_i10a20 PRIVATE include)
target_link_libraries(train_sac_c128_b256_tracking_g995_eratix_i10a20 PRIVATE RLtools::RLtools)
target_compile_definitions(train_sac_c128_b256_tracking_g995_eratix_i10a20 PRIVATE FW_CRITIC_HIDDEN=128 FW_BATCH=256 FW_TRACKING=1 FW_GAMMA=0.995 FW_AIRPLANE=2 FW_TRAINING_INTERVAL=10 FW_ACTOR_INTERVAL_FACTOR=2)
list(APPEND TRAIN_TARGETS train_sac_c128_b256_tracking_g995_eratix_i10a20)

# trim, thrust, stall speed, linear modes and a hands-off flight of the Eratix model
add_executable(eratix_trim src/eratix_trim.cpp)
target_include_directories(eratix_trim PRIVATE include)
target_link_libraries(eratix_trim PRIVATE RLtools::RLtools)
target_compile_definitions(eratix_trim PRIVATE FW_TRACKING=1 FW_AIRPLANE=2)
list(APPEND TRAIN_TARGETS eratix_trim)

# a flight of the deployed policy in Gazebo (deploy/ros2/fly_sitl.sh) against this simulator
add_executable(compare_sitl src/compare_sitl.cpp)
target_include_directories(compare_sitl PRIVATE include deploy/ros2/fw_policy/policy)
target_link_libraries(compare_sitl PRIVATE RLtools::RLtools)
target_compile_definitions(compare_sitl PRIVATE FW_TRACKING=1 FW_AIRPLANE=1)
list(APPEND TRAIN_TARGETS compare_sitl)

# a trained tracking actor of the advanced_plane (actor.h or best_actor.h, read at run time) when the reference point
# accelerates: circles, helices, figure-eights and speed oscillations, from level and random starts
add_executable(evaluate_reference src/evaluate_reference.cpp)
target_include_directories(evaluate_reference PRIVATE include)
find_package(Threads REQUIRED)
target_link_libraries(evaluate_reference PRIVATE RLtools::RLtools Threads::Threads)
target_compile_definitions(evaluate_reference PRIVATE FW_TRACKING=1 FW_AIRPLANE=1)
list(APPEND TRAIN_TARGETS evaluate_reference)

if(NOT MSVC)
    foreach(target ${TRAIN_TARGETS})
        target_compile_options(${target} PRIVATE -O3 -march=native)
    endforeach()
endif()

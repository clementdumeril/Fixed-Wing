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

# more critic:batch sizes, for a sweep between the two above (savio/sweep.slurm)
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

if(NOT MSVC)
    foreach(target ${TRAIN_TARGETS})
        target_compile_options(${target} PRIVATE -O3 -march=native)
    endforeach()
endif()

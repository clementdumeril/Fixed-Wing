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

if(NOT MSVC)
    target_compile_options(train_sac PRIVATE -O3 -march=native)
    target_compile_options(train_sac_small PRIVATE -O3 -march=native)
endif()

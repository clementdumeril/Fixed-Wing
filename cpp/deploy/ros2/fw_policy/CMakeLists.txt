cmake_minimum_required(VERSION 3.16)
project(fw_policy CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(px4_msgs REQUIRED)

# the simulator headers of this repository (observation), RLtools (only for the environment base class) and the
# exported policy (tools/export_policy.py); the package directory may be a symbolic link into a colcon workspace
get_filename_component(PACKAGE_DIR ${CMAKE_CURRENT_SOURCE_DIR} REALPATH)
set(FIXED_WING_INCLUDE "${PACKAGE_DIR}/../../../include" CACHE PATH "Fixed-Wing/cpp/include")
set(RL_TOOLS_INCLUDE "" CACHE PATH "RLtools include directory")
set(POLICY_DIR "${PACKAGE_DIR}/policy" CACHE PATH "directory of policy_weights.h")
if(NOT EXISTS "${RL_TOOLS_INCLUDE}/rl_tools/rl/environments/environments.h")
    message(FATAL_ERROR "set RL_TOOLS_INCLUDE to an RLtools checkout's include directory")
endif()
if(NOT EXISTS "${POLICY_DIR}/policy_weights.h")
    message(FATAL_ERROR "no policy_weights.h in ${POLICY_DIR}: run tools/export_policy.py <run>/actor.h")
endif()

add_executable(fw_policy_node src/fw_policy_node.cpp)
target_include_directories(fw_policy_node PRIVATE ${FIXED_WING_INCLUDE} ${RL_TOOLS_INCLUDE} ${POLICY_DIR})
# the task the policy was trained on: tracking, on PX4's advanced_plane
target_compile_definitions(fw_policy_node PRIVATE FW_TRACKING=1 FW_AIRPLANE=1)
ament_target_dependencies(fw_policy_node rclcpp px4_msgs)
install(TARGETS fw_policy_node DESTINATION lib/${PROJECT_NAME})

ament_package()

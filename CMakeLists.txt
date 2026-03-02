cmake_minimum_required(VERSION 3.8)
cmake_policy(SET CMP0074 NEW)
project(ur20_zed_point_cloud)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

link_directories(
  ${PCL_LIBRARY_DIRS}
)

add_definitions(
  ${PCL_DEFINITIONS}
)

add_executable(echo_point_cloud src/echo_point_cloud.cpp)
add_executable(stats_point_cloud src/stats_point_cloud.cpp)
add_executable(filter_point_cloud src/filter_point_cloud.cpp)
add_executable(pcl_demo src/pcl_demo.cpp)


ament_target_dependencies(echo_point_cloud
  rclcpp
  sensor_msgs
  pcl_conversions
)
ament_target_dependencies(stats_point_cloud
  rclcpp
  sensor_msgs
  pcl_conversions
)
ament_target_dependencies(filter_point_cloud
  rclcpp
  sensor_msgs
  pcl_conversions
)
ament_target_dependencies(pcl_demo
  rclcpp
  sensor_msgs
  pcl_conversions
)

target_link_libraries(echo_point_cloud ${PCL_LIBRARIES})
target_link_libraries(stats_point_cloud ${PCL_LIBRARIES})
target_link_libraries(filter_point_cloud ${PCL_LIBRARIES})
target_link_libraries(pcl_demo ${PCL_LIBRARIES})

install(TARGETS
  echo_point_cloud
  stats_point_cloud
  filter_point_cloud
  pcl_demo
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

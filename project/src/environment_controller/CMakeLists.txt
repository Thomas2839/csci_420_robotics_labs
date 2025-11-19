cmake_minimum_required(VERSION 3.14.4)
project(environment_controller)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem thread)

# Create the executable
add_executable(environment_controller_node src/environment_controller_node.cpp)

# target_link_libraries(environment_controller ${Boost_LIBRARIES})

find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/UseKey.srv"
  DEPENDENCIES geometry_msgs
)

include_directories(include ${Boost_INCLUDE_DIRS})

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(environment_controller_node PUBLIC
  ${geometry_msgs_TARGETS}
  ${std_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
  ${nav_msgs_TARGETS}
  rclcpp::rclcpp
  tf2::tf2
  tf2_ros::static_transform_broadcaster_node
  tf2_ros::tf2_ros
  tf2_geometry_msgs::tf2_geometry_msgs
  "${cpp_typesupport_target}"
)

ament_export_dependencies(rosidl_default_runtime)

# Install the executable
install(TARGETS
  environment_controller_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
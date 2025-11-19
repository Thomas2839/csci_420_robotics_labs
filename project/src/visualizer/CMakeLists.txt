cmake_minimum_required(VERSION 3.14.4)
project(visualizer)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

## Find catkin and any catkin packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(src)

install(PROGRAMS
  src/gui.py
  DESTINATION lib/visualizer )

install(PROGRAMS
  src/gui_controller.py
  DESTINATION lib/visualizer )

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_package()
cmake_minimum_required(VERSION 3.14.4)
project(flightcontroller)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosgraph_msgs REQUIRED)

###########
## Build ##
###########

include_directories(include ${Boost_INCLUDE_DIRS})

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY maps DESTINATION share/${PROJECT_NAME})

install(PROGRAMS
  src/angle_calculator.py
  DESTINATION lib/flightcontroller )

install(PROGRAMS
  src/angle_controller.py
  DESTINATION lib/flightcontroller )

install(PROGRAMS
  src/position_controller.py
  DESTINATION lib/flightcontroller )

install(PROGRAMS
  src/velocity_controller.py
  DESTINATION lib/flightcontroller )

install(PROGRAMS
  src/pid_class.py
  DESTINATION lib/flightcontroller )

ament_package()
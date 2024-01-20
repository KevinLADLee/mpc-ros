cmake_minimum_required(VERSION 3.0.2)
project(mpc_ros)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  nav_msgs
)

catkin_python_setup()

catkin_package(CATKIN_DEPENDS rospy)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

file(GLOB_RECURSE PY_FILES src/*.py)

message(STATUS "PY_FILES: ${PY_FILES}")

catkin_install_python (
  PROGRAMS ${PY_FILES}
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)


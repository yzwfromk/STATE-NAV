cmake_minimum_required(VERSION 3.14)
project(statenav_global)

# Find Python3
find_package(Python3 REQUIRED)

# Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  grid_map_msgs
  genmsg
)

# Install Python package via setup.py
catkin_python_setup()

###################################
## catkin specific configuration ##
###################################
catkin_package(
  CATKIN_DEPENDS 
    roscpp 
    rospy 
    std_msgs 
    geometry_msgs 
    nav_msgs 
    grid_map_msgs
)

#############
## Install ##
#############

# Install Python ROS nodes
catkin_install_python(PROGRAMS
  executables/main_traversability_estimation.py
  executables/main_global_planning.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install config files
install(DIRECTORY executables/configs/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
  FILES_MATCHING PATTERN "*.yaml"
)
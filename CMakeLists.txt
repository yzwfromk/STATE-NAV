cmake_minimum_required(VERSION 3.14)
project(statenav_global)

# Find ament packages
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  # executables/main_traversability_estimation.py
  executables/main_global_planning.py
  executables/main_worldmodel.py
  executables/main_multiprocess.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install config files
install(DIRECTORY executables/configs/
  DESTINATION share/${PROJECT_NAME}/config
  FILES_MATCHING PATTERN "*.yaml"
)

# Install launch files (if any)
install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING PATTERN "*.py"
  PATTERN ".gitkeep" EXCLUDE
)

ament_package()
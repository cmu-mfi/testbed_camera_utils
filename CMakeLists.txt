cmake_minimum_required(VERSION 3.0.2)
project(testbed_camera_utils)

find_package(catkin REQUIRED)

#catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  scripts/calibrate_camera.py
  scripts/detect_amr.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
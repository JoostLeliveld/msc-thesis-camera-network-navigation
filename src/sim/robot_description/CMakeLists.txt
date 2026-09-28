cmake_minimum_required(VERSION 3.5)
project(robot_description)

install(
  DIRECTORY meshes urdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

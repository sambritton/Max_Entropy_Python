cmake_minimum_required(VERSION 3.15)
project(Max_Entropy_Python CXX)

message("CMake looking for pybind11...")
add_subdirectory(pybind11)

enable_testing()

add_subdirectory(potential_step_module)

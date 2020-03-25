
set(PYBIND11_CPP_STANDARD -std=c++14)
include(FindPythonInterp)

add_library(
    potential_step
    MODULE main.cpp)

target_link_libraries(
    potential_step
    PRIVATE pybind11::module)

target_include_directories(
    potential_step
    PRIVATE "include")

set_target_properties(
    potential_step
    PROPERTIES
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}")

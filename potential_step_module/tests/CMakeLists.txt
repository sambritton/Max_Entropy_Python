add_executable(test_driver driver.cpp)

target_include_directories(
    test_driver
    PRIVATE SYSTEM "../include")

target_include_directories(
    test_driver
    PRIVATE "../pstep")

target_link_libraries(test_driver PRIVATE options)

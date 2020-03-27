cmake_minimum_required(VERSION 3.15)
project(Max_Entropy_Python CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message("CMake looking for pybind11...")
add_subdirectory(pybind11)

#find_package(OpenMP REQUIRED)

enable_testing()

add_library(options INTERFACE)
add_library(warnings INTERFACE)

target_compile_features(options
    INTERFACE
        cxx_std_17
        cxx_alignas
        cxx_alignof
        cxx_attributes
        cxx_auto_type
        cxx_constexpr
        cxx_defaulted_functions
        cxx_deleted_functions
        cxx_final
        cxx_lambdas
        cxx_noexcept
        cxx_override
        cxx_range_for
        cxx_rvalue_references
        cxx_static_assert
        cxx_strong_enums
        cxx_trailing_return_types
        cxx_unicode_literals
        cxx_user_literals
        cxx_variadic_macros
)

# Some options taken from Jason Turner
# <https://github.com/lefticus/cpp_box/blob/master/CMakeLists.txt>
target_compile_options(warnings
    INTERFACE
        -Wall
        -Wextra
        -Wshadow            # warn the user if a variable
                            # declaration shadows one from a
                            # parent context
        -Wnon-virtual-dtor  # warn the user if a class
                            # with virtual functions
                            # has a non-virtual
                            # destructor. This helps
                            # catch hard to track down
                            # memory errors
        -Wpedantic          # Warn on nonstandard C++
        -Wunused
        -Wconversion        # Warn on lossy conversions
        -Wnull-dereference
        -Wdouble-promotion  # Warn on implicit conversion from
                            # float to double
        -Wformat=2          # Warn on possibly insecure formatting
                            # when using printf
)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    message("Using extra GNU-specific compiler warnings...")
    target_compile_options(warnings
        INTERFACE
            -Wmisleading-indentation    # Warn on indented blocks
                                        # that are not really blocks
            -Wduplicated-cond           # Warn on if/else chain with
                                        # duplicated conditions
            -Wduplicated-branches       # Warn on if/else chains with
                                        # duplicated code
            -Wuseless-cast              # Warn when casting to the same type
    )
endif()

add_subdirectory(potential_step_module)

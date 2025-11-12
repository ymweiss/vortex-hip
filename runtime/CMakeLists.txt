cmake_minimum_required(VERSION 3.10)
project(vortex_hip VERSION 1.0.0 LANGUAGES C CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Vortex installation path
if(NOT DEFINED VORTEX_ROOT)
    set(VORTEX_ROOT "$ENV{VORTEX_ROOT}" CACHE PATH "Path to Vortex installation")
    if(NOT VORTEX_ROOT)
        set(VORTEX_ROOT "${CMAKE_SOURCE_DIR}/../vortex" CACHE PATH "Path to Vortex installation")
    endif()
endif()

message(STATUS "Vortex root: ${VORTEX_ROOT}")

# Find Vortex libraries and headers
find_path(VORTEX_INCLUDE_DIR
    NAMES vortex.h
    PATHS ${VORTEX_ROOT}/runtime/include
    NO_DEFAULT_PATH
)

# Find Vortex runtime library
find_library(VORTEX_LIBRARY
    NAMES vortex
    PATHS
        ${VORTEX_ROOT}/build/runtime
        ${VORTEX_ROOT}/runtime/stub
        ${VORTEX_ROOT}/stub
    NO_DEFAULT_PATH
)

# Find Vortex kernel headers
find_path(VORTEX_KERNEL_INCLUDE_DIR
    NAMES vx_spawn.h vx_intrinsics.h
    PATHS ${VORTEX_ROOT}/kernel/include
    NO_DEFAULT_PATH
)

if(NOT VORTEX_INCLUDE_DIR)
    message(FATAL_ERROR "Vortex headers not found at ${VORTEX_ROOT}/runtime/include")
endif()

if(NOT VORTEX_LIBRARY)
    message(FATAL_ERROR "Vortex library not found. Searched in:\n"
            "  ${VORTEX_ROOT}/build/runtime\n"
            "  ${VORTEX_ROOT}/runtime/stub\n"
            "  ${VORTEX_ROOT}/stub\n"
            "Please build Vortex first with: cd ${VORTEX_ROOT}/build && make")
endif()

message(STATUS "Vortex include dir: ${VORTEX_INCLUDE_DIR}")
message(STATUS "Vortex kernel include dir: ${VORTEX_KERNEL_INCLUDE_DIR}")
message(STATUS "Vortex library: ${VORTEX_LIBRARY}")

# Vortex HIP Runtime Library
add_library(vortex_hip SHARED
    src/vortex_hip_runtime.cpp
)

target_include_directories(vortex_hip
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${VORTEX_INCLUDE_DIR}
        ${VORTEX_ROOT}/build/hw           # Generated headers (VX_config.h, VX_types.h)
        ${VORTEX_ROOT}/hw                 # Hardware source headers
        ${VORTEX_ROOT}/runtime/common     # Runtime common headers
)

target_link_libraries(vortex_hip
    PRIVATE
        ${VORTEX_LIBRARY}
)

# Enable warnings
target_compile_options(vortex_hip PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# Set output name to match HIP convention
set_target_properties(vortex_hip PROPERTIES
    OUTPUT_NAME "hip_vortex"
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)

# Installation
install(TARGETS vortex_hip
    EXPORT vortex_hip-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# Export configuration
install(EXPORT vortex_hip-targets
    FILE vortex_hip-targets.cmake
    NAMESPACE vortex_hip::
    DESTINATION lib/cmake/vortex_hip
)

# Create config file
include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vortex_hip-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/vortex_hip-config.cmake"
    INSTALL_DESTINATION lib/cmake/vortex_hip
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/vortex_hip-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/vortex_hip-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/vortex_hip-config-version.cmake"
    DESTINATION lib/cmake/vortex_hip
)

# Build examples (optional)
option(BUILD_EXAMPLES "Build example programs" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Print summary
message(STATUS "=== Vortex HIP Configuration Summary ===")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Vortex Root: ${VORTEX_ROOT}")
message(STATUS "  Build Examples: ${BUILD_EXAMPLES}")
message(STATUS "  Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "========================================")

# Note: HIP tests are in ../tests/ directory
# They require hipcc compiler and are not built by default
# See ../tests/README.md for more information

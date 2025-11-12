cmake_minimum_required(VERSION 3.14)
project(VortexHIPUnitTests)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Try to find Google Test installed on system
find_package(GTest QUIET)

if(NOT GTest_FOUND)
    message(STATUS "Google Test not found, fetching from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        release-1.12.1
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
else()
    message(STATUS "Using system Google Test")
endif()

# Enable testing
enable_testing()
include(GoogleTest)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/../../runtime/include
    ${CMAKE_SOURCE_DIR}/../../runtime/src
)

# Link directories (if runtime is built)
link_directories(
    ${CMAKE_SOURCE_DIR}/../../runtime/build
)

# Test: Metadata Structure
add_executable(test_metadata_structure
    test_metadata_structure.cpp
)
target_link_libraries(test_metadata_structure
    GTest::gtest_main
)
add_test(NAME MetadataStructure COMMAND test_metadata_structure)

# Test: Argument Layout
add_executable(test_argument_layout
    test_argument_layout.cpp
)
target_link_libraries(test_argument_layout
    GTest::gtest_main
)
add_test(NAME ArgumentLayout COMMAND test_argument_layout)

# Test: Type Sizes (RV32/RV64)
add_executable(test_type_sizes
    test_type_sizes.cpp
)
target_link_libraries(test_type_sizes
    GTest::gtest_main
)
add_test(NAME TypeSizes COMMAND test_type_sizes)

# Run all tests
add_custom_target(run_unit_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS test_metadata_structure test_argument_layout test_type_sizes
)

# Coverage target (requires gcov/lcov)
if(CMAKE_BUILD_TYPE MATCHES Coverage)
    if(CMAKE_COMPILER_IS_GNUCXX)
        include(CodeCoverage)
        setup_target_for_coverage(
            NAME coverage
            EXECUTABLE ctest
            DEPENDENCIES run_unit_tests
        )
    endif()
endif()

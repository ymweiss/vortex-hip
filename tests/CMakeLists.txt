# Vortex HIP Test Suite
cmake_minimum_required(VERSION 3.10)

# These tests require HIP compiler support
# Currently, they serve as reference implementations

# List all test programs
set(HIP_TESTS
    basic
    vecadd
    dotproduct
    sgemm
    sgemm2
    sgemm_tcu
    sgemv
    relu
    dropout
    conv3
    diverge
    cta
    sort
    stencil3d
    fence
    printf
    madmax
    mstress
    io_addr
    demo
    dogfood
)

# Note: Building these tests requires a HIP compiler configured for Vortex
# This is a placeholder build system for when hipcc/Vortex integration is complete

option(BUILD_HIP_TESTS "Build HIP test programs (requires hipcc for Vortex)" OFF)

if(BUILD_HIP_TESTS)
    # Find HIP compiler
    find_program(HIPCC hipcc)

    if(NOT HIPCC)
        message(WARNING "hipcc not found. HIP tests will not be built.")
        message(WARNING "Set HIPCC=/path/to/hipcc or disable BUILD_HIP_TESTS")
        return()
    endif()

    message(STATUS "Found hipcc: ${HIPCC}")
    message(STATUS "Building HIP tests for Vortex target")

    # HIP compiler flags for Vortex target
    set(HIPCC_FLAGS
        --offload-arch=vortex  # Vortex target (placeholder)
        -I${CMAKE_SOURCE_DIR}/runtime/include
        -L${CMAKE_BINARY_DIR}/runtime
        -lhip_vortex
    )

    # Build each test
    foreach(TEST_NAME ${HIP_TESTS})
        add_custom_command(
            OUTPUT ${TEST_NAME}
            COMMAND ${HIPCC} ${HIPCC_FLAGS}
                    ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_NAME}.cpp
                    -o ${TEST_NAME}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_NAME}.cpp
            COMMENT "Building HIP test: ${TEST_NAME}"
        )

        add_custom_target(test_${TEST_NAME} ALL DEPENDS ${TEST_NAME})

        # Install test
        install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}
                DESTINATION bin/tests)
    endforeach()

else()
    message(STATUS "HIP tests disabled. Enable with -DBUILD_HIP_TESTS=ON")
    message(STATUS "Note: Requires hipcc configured for Vortex target")

    # Install source files for reference
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
            DESTINATION share/vortex_hip/tests/src)

    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/README.md
            DESTINATION share/vortex_hip/tests)
endif()

# Add test sources to IDE projects even if not building
file(GLOB TEST_SOURCES "*.cpp")
add_custom_target(hip_test_sources SOURCES ${TEST_SOURCES})

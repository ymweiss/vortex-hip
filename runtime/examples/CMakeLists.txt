# Vortex HIP Examples

# Vector Addition Example
add_executable(vector_add vector_add.cpp)

target_link_libraries(vector_add
    PRIVATE vortex_hip
)

target_include_directories(vector_add
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

# Argument Marshaling Test
add_executable(test_marshaling test_marshaling.cpp)

target_link_libraries(test_marshaling
    PRIVATE vortex_hip
)

target_include_directories(test_marshaling
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

# Install examples
install(TARGETS vector_add test_marshaling
    RUNTIME DESTINATION bin/examples
)

# Copy source files for reference
install(FILES
    vector_add.cpp
    test_marshaling.cpp
    DESTINATION share/vortex_hip/examples
)

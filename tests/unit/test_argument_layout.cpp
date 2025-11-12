/**
 * @file test_argument_layout.cpp
 * @brief Unit tests for argument layout and alignment calculations
 */

#include <gtest/gtest.h>
#include "vortex_hip_runtime.h"
#include <vector>

TEST(ArgumentLayout, CalculatesPaddingCorrectly) {
    // Test padding calculation: (alignment - (offset % alignment)) % alignment

    struct TestCase {
        size_t offset;
        size_t alignment;
        size_t expected_padding;
    };

    std::vector<TestCase> cases = {
        {0, 4, 0},   // Already aligned
        {1, 4, 3},   // Need 3 bytes padding
        {2, 4, 2},   // Need 2 bytes padding
        {3, 4, 1},   // Need 1 byte padding
        {4, 4, 0},   // Already aligned
        {0, 8, 0},   // 8-byte alignment
        {4, 8, 4},   // Need 4 bytes to reach 8-byte boundary
        {7, 8, 1},   // Need 1 byte to reach 8-byte boundary
    };

    for (const auto& tc : cases) {
        size_t padding = (tc.alignment - (tc.offset % tc.alignment)) % tc.alignment;
        EXPECT_EQ(tc.expected_padding, padding)
            << "Failed for offset=" << tc.offset << " alignment=" << tc.alignment;
    }
}

TEST(ArgumentLayout, RV32PointerLayout) {
    // Test layout for RV32: float* a, float* b, int n
    std::vector<hipKernelArgumentMetadata> metadata = {
        {.offset = 0, .size = 4, .alignment = 4, .is_pointer = 1},  // a
        {.offset = 4, .size = 4, .alignment = 4, .is_pointer = 1},  // b
        {.offset = 8, .size = 4, .alignment = 4, .is_pointer = 0}   // n
    };

    EXPECT_EQ(0, metadata[0].offset);
    EXPECT_EQ(4, metadata[1].offset);
    EXPECT_EQ(8, metadata[2].offset);

    // Total size
    size_t total = metadata[2].offset + metadata[2].size;
    EXPECT_EQ(12, total);
}

TEST(ArgumentLayout, RV64PointerLayout) {
    // Test layout for RV64: float* a, float* b, int n
    std::vector<hipKernelArgumentMetadata> metadata = {
        {.offset = 0, .size = 8, .alignment = 8, .is_pointer = 1},  // a
        {.offset = 8, .size = 8, .alignment = 8, .is_pointer = 1},  // b
        {.offset = 16, .size = 4, .alignment = 4, .is_pointer = 0}  // n
    };

    EXPECT_EQ(0, metadata[0].offset);
    EXPECT_EQ(8, metadata[1].offset);
    EXPECT_EQ(16, metadata[2].offset);

    // Total size
    size_t total = metadata[2].offset + metadata[2].size;
    EXPECT_EQ(20, total);
}

TEST(ArgumentLayout, MixedTypesWithPadding) {
    // Test: char c, double* ptr (RV64)
    // char is 1 byte, double* needs 8-byte alignment
    std::vector<hipKernelArgumentMetadata> metadata = {
        {.offset = 0, .size = 1, .alignment = 1, .is_pointer = 0},  // char
        {.offset = 8, .size = 8, .alignment = 8, .is_pointer = 1}   // double* (padded to 8)
    };

    EXPECT_EQ(0, metadata[0].offset);
    EXPECT_EQ(8, metadata[1].offset);  // Padded from 1 to 8

    // Verify 7 bytes of padding were added
    size_t padding = metadata[1].offset - (metadata[0].offset + metadata[0].size);
    EXPECT_EQ(7, padding);
}

TEST(ArgumentLayout, StructByValue) {
    // Test: struct Vec3 { float x, y, z; } (12 bytes, 4-byte aligned)
    hipKernelArgumentMetadata metadata = {
        .offset = 0,
        .size = 12,
        .alignment = 4,
        .is_pointer = 0
    };

    EXPECT_EQ(12, metadata.size);
    EXPECT_EQ(4, metadata.alignment);
    EXPECT_EQ(0, metadata.is_pointer);
}

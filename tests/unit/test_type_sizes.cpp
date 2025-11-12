/**
 * @file test_type_sizes.cpp
 * @brief Unit tests for type size validation across RV32 and RV64
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cstddef>

// Test actual C++ type sizes
TEST(TypeSizes, BasicTypes) {
    EXPECT_EQ(1, sizeof(char));
    EXPECT_EQ(1, sizeof(int8_t));
    EXPECT_EQ(2, sizeof(int16_t));
    EXPECT_EQ(4, sizeof(int32_t));
    EXPECT_EQ(8, sizeof(int64_t));
    EXPECT_EQ(4, sizeof(float));
    EXPECT_EQ(8, sizeof(double));
}

TEST(TypeSizes, PointerSize) {
    // Pointer size depends on architecture
    size_t ptr_size = sizeof(void*);

    // Should be either 4 (RV32) or 8 (RV64)
    EXPECT_TRUE(ptr_size == 4 || ptr_size == 8)
        << "Pointer size is " << ptr_size << " bytes";
}

TEST(TypeSizes, IntSize) {
    // int is always 4 bytes on both RV32 and RV64
    EXPECT_EQ(4, sizeof(int));
    EXPECT_EQ(4, sizeof(unsigned int));
}

TEST(TypeSizes, LongSize) {
    // long differs between RV32 (4 bytes) and RV64 (8 bytes)
    size_t long_size = sizeof(long);

    // Should match pointer size
    EXPECT_EQ(sizeof(void*), long_size);
}

TEST(TypeSizes, FloatingPoint) {
    // Floating point sizes are consistent
    EXPECT_EQ(4, sizeof(float));
    EXPECT_EQ(8, sizeof(double));
}

// Test struct packing
TEST(TypeSizes, StructAlignment) {
    struct TestStruct {
        char c;
        int i;
    };

    // Struct should have padding
    EXPECT_GT(sizeof(TestStruct), sizeof(char) + sizeof(int));
}

TEST(TypeSizes, PackedStruct) {
    struct __attribute__((packed)) PackedStruct {
        char c;
        int i;
    };

    // Packed struct should have no padding
    EXPECT_EQ(sizeof(char) + sizeof(int), sizeof(PackedStruct));
}

// Parameterized test for RV32 metadata
class RV32TypeSizeTest : public ::testing::TestWithParam<std::tuple<size_t, size_t>> {
};

TEST_P(RV32TypeSizeTest, ValidateMetadata) {
    auto [size, alignment] = GetParam();

    EXPECT_GT(size, 0) << "Size must be positive";
    EXPECT_GT(alignment, 0) << "Alignment must be positive";
    EXPECT_TRUE((alignment & (alignment - 1)) == 0)
        << "Alignment must be power of 2";
    EXPECT_LE(alignment, size)
        << "Alignment should not exceed size for basic types";
}

INSTANTIATE_TEST_SUITE_P(
    RV32Types,
    RV32TypeSizeTest,
    ::testing::Values(
        std::make_tuple(1, 1),   // char
        std::make_tuple(2, 2),   // short
        std::make_tuple(4, 4),   // int, float, pointer (RV32)
        std::make_tuple(8, 8)    // long long, double
    )
);

// Parameterized test for RV64 metadata
class RV64TypeSizeTest : public ::testing::TestWithParam<std::tuple<size_t, size_t>> {
};

TEST_P(RV64TypeSizeTest, ValidateMetadata) {
    auto [size, alignment] = GetParam();

    EXPECT_GT(size, 0) << "Size must be positive";
    EXPECT_GT(alignment, 0) << "Alignment must be positive";
    EXPECT_TRUE((alignment & (alignment - 1)) == 0)
        << "Alignment must be power of 2";
}

INSTANTIATE_TEST_SUITE_P(
    RV64Types,
    RV64TypeSizeTest,
    ::testing::Values(
        std::make_tuple(1, 1),   // char
        std::make_tuple(2, 2),   // short
        std::make_tuple(4, 4),   // int, float
        std::make_tuple(8, 8)    // long, long long, double, pointer (RV64)
    )
);

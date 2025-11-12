/**
 * @file test_metadata_structure.cpp
 * @brief Unit tests for HIP kernel metadata structure
 */

#include <gtest/gtest.h>
#include "vortex_hip_runtime.h"

TEST(MetadataStructure, HasCorrectFields) {
    hipKernelArgumentMetadata meta = {
        .offset = 16,
        .size = 4,
        .alignment = 4,
        .is_pointer = 0
    };

    EXPECT_EQ(16, meta.offset);
    EXPECT_EQ(4, meta.size);
    EXPECT_EQ(4, meta.alignment);
    EXPECT_EQ(0, meta.is_pointer);
}

TEST(MetadataStructure, PointerFlagWorks) {
    hipKernelArgumentMetadata ptr_meta = {
        .offset = 0,
        .size = 8,
        .alignment = 8,
        .is_pointer = 1
    };

    EXPECT_NE(0, ptr_meta.is_pointer);
    EXPECT_EQ(1, ptr_meta.is_pointer);
}

TEST(MetadataStructure, ScalarFlagWorks) {
    hipKernelArgumentMetadata scalar_meta = {
        .offset = 0,
        .size = 4,
        .alignment = 4,
        .is_pointer = 0
    };

    EXPECT_EQ(0, scalar_meta.is_pointer);
}

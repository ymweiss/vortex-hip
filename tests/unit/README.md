# Unit Tests for Vortex HIP

**Test-Driven Development (TDD) is required for all new code.**

## Overview

This directory contains unit tests for the Vortex HIP runtime library. All tests use Google Test framework and follow TDD principles.

## Structure

```
tests/unit/
├── README.md                    # This file
├── CMakeLists.txt               # Test build configuration
├── run.sh                       # Test runner script
├── test_metadata_marshaling.cpp # Metadata marshaling tests
├── test_argument_layout.cpp     # Argument layout calculation tests
├── test_type_sizes.cpp          # Type size validation tests
└── test_registration.cpp        # Kernel registration tests
```

## Running Tests

### All Tests
```bash
cd tests/unit
./run.sh
```

### Specific Test
```bash
cd tests/unit
./test_metadata_marshaling
```

### With Verbose Output
```bash
cd tests/unit
./test_metadata_marshaling --gtest_verbose
```

### With Coverage
```bash
cd tests/unit
make coverage
```

## Writing Tests

### Test File Template

```cpp
#include <gtest/gtest.h>
#include "vortex_hip_runtime.h"

// Test fixture for grouped tests
class MetadataMarshalingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code before each test
    }

    void TearDown() override {
        // Cleanup code after each test
    }

    // Helper functions
    void validateMetadata(const hipKernelArgumentMetadata& meta) {
        EXPECT_GT(meta.size, 0);
        EXPECT_GT(meta.alignment, 0);
    }
};

// Simple test
TEST(MetadataBasic, SizeCalculation) {
    // Arrange
    size_t size = 4;
    size_t alignment = 4;

    // Act
    size_t offset = calculateOffset(size, alignment);

    // Assert
    EXPECT_EQ(0, offset);
}

// Test with fixture
TEST_F(MetadataMarshalingTest, PackArguments) {
    // Arrange
    float* ptr = nullptr;
    int value = 42;
    void* args[] = {&ptr, &value};

    // Act
    auto result = packArguments(args, 2, metadata);

    // Assert
    EXPECT_EQ(8, result.size());
    validateMetadata(metadata[0]);
}
```

### Test Naming Convention

- **Test Suite Name:** Component being tested (e.g., `MetadataMarshaling`)
- **Test Name:** Specific behavior being tested (e.g., `HandlesNullPointer`)

**Format:** `TEST(TestSuiteName, TestName)`

**Examples:**
```cpp
TEST(ArgumentLayout, CalculatesOffsetCorrectly)
TEST(ArgumentLayout, HandlesAlignment)
TEST(ArgumentLayout, WorksWithMixedTypes)
```

### TDD Workflow

1. **Write the test FIRST** (it should fail)
```cpp
TEST(NewFeature, DoesWhatItShould) {
    auto result = newFunction(input);
    EXPECT_EQ(expected, result);
}
```

2. **Run the test** (verify it fails)
```bash
./run.sh
# Should see: FAILED NewFeature.DoesWhatItShould
```

3. **Implement the minimum code to pass**
```cpp
int newFunction(int input) {
    return expected_value;
}
```

4. **Run the test again** (should pass)
```bash
./run.sh
# Should see: PASSED NewFeature.DoesWhatItShould
```

5. **Refactor if needed** (keeping tests green)

## Test Categories

### Required Tests for New Features

#### Functionality Tests
- ✅ Happy path (normal usage)
- ✅ Edge cases (boundary values)
- ✅ Error conditions
- ✅ Invalid input handling

#### Example:
```cpp
TEST(ArgumentMarshaling, HandlesEmptyArguments)
TEST(ArgumentMarshaling, HandlesSingleArgument)
TEST(ArgumentMarshaling, HandlesMaximumArguments)
TEST(ArgumentMarshaling, RejectsNullMetadata)
TEST(ArgumentMarshaling, RejectsMismatchedArgCount)
```

### Performance Tests

For performance-critical code:
```cpp
TEST(ArgumentMarshaling, MarshalingPerformance) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10000; i++) {
        marshallArguments(args, metadata);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Should complete in less than 10ms for 10k iterations
    EXPECT_LT(duration.count(), 10000);
}
```

## Assertions

### Common Assertions

```cpp
// Equality
EXPECT_EQ(expected, actual);
ASSERT_EQ(expected, actual);  // Fatal - stops test on failure

// Inequality
EXPECT_NE(val1, val2);

// Comparisons
EXPECT_LT(val1, val2);  // Less than
EXPECT_LE(val1, val2);  // Less than or equal
EXPECT_GT(val1, val2);  // Greater than
EXPECT_GE(val1, val2);  // Greater than or equal

// Boolean
EXPECT_TRUE(condition);
EXPECT_FALSE(condition);

// Pointers
EXPECT_EQ(nullptr, ptr);
EXPECT_NE(nullptr, ptr);

// Floating point (with tolerance)
EXPECT_NEAR(expected, actual, tolerance);

// Strings
EXPECT_STREQ("expected", actual_cstr);

// Exceptions
EXPECT_THROW(function(), ExceptionType);
EXPECT_NO_THROW(function());
```

### Custom Matchers

```cpp
// For complex validation
EXPECT_THAT(value, MatcherFunction());
```

## Coverage Requirements

**Minimum: 80% code coverage**

### Checking Coverage

```bash
cd tests/unit
make coverage

# Opens HTML report
xdg-open coverage/index.html
```

### What to Cover

- ✅ All public API functions
- ✅ All error paths
- ✅ All conditional branches
- ✅ Edge cases and boundary conditions

### What NOT to Cover

- ❌ Third-party libraries
- ❌ Generated code
- ❌ Trivial getters/setters (if truly trivial)

## Mocking

For dependencies that are hard to test:

```cpp
class MockVortexDevice {
public:
    MOCK_METHOD(int, vx_start, (vx_device_h, vx_buffer_h, vx_buffer_h));
    MOCK_METHOD(int, vx_ready_wait, (vx_device_h, uint64_t));
};

TEST(KernelLaunch, CallsVortexStart) {
    MockVortexDevice mock;

    EXPECT_CALL(mock, vx_start(_, _, _))
        .WillOnce(Return(0));

    launchKernel(kernel, grid, block);
}
```

## Test Data

### Test Fixtures

For shared setup:

```cpp
class MetadataTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // One-time setup for all tests in suite
    }

    void SetUp() override {
        // Setup before each test
        metadata = createTestMetadata();
    }

    hipKernelArgumentMetadata* metadata;
};
```

### Parameterized Tests

For testing multiple inputs:

```cpp
class TypeSizeTest : public ::testing::TestWithParam<std::tuple<size_t, size_t>> {
};

TEST_P(TypeSizeTest, CorrectSizes) {
    auto [size, alignment] = GetParam();
    EXPECT_GT(size, 0);
    EXPECT_GT(alignment, 0);
}

INSTANTIATE_TEST_SUITE_P(
    RV32Types,
    TypeSizeTest,
    ::testing::Values(
        std::make_tuple(4, 4),  // int
        std::make_tuple(4, 4),  // float
        std::make_tuple(4, 4)   // pointer
    )
);
```

## Debugging Failed Tests

### Run with debugger
```bash
gdb ./test_metadata_marshaling
(gdb) run --gtest_filter=TestName
```

### Verbose output
```bash
./test_metadata_marshaling --gtest_verbose
```

### Specific test
```bash
./test_metadata_marshaling --gtest_filter=TestSuite.TestName
```

### Print all tests
```bash
./test_metadata_marshaling --gtest_list_tests
```

## Integration with CI/CD

All tests are run automatically on:
- Every commit
- Every pull request
- Nightly builds

Failed tests block merging.

## Best Practices

### DO:
- ✅ Write tests before implementation (TDD)
- ✅ Keep tests simple and focused
- ✅ Test one thing per test
- ✅ Use descriptive test names
- ✅ Clean up resources in TearDown()
- ✅ Use EXPECT over ASSERT unless failure is fatal
- ✅ Test error conditions
- ✅ Document complex test setups

### DON'T:
- ❌ Test implementation details
- ❌ Write tests that depend on execution order
- ❌ Use sleep() or timing assumptions
- ❌ Leave commented-out tests
- ❌ Skip writing tests for "simple" code
- ❌ Ignore failing tests
- ❌ Commit code without running tests

## Questions?

See the main project [.development-rules.md](../../.development-rules.md) or ask in:
- GitHub Issues
- Project chat

---

**Remember: If it's not tested, it's broken.**

**Last Updated:** 2025-11-06

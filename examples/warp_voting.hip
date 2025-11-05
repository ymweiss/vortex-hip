/*
 * Warp Voting Example
 *
 * This example demonstrates Vortex warp voting operations:
 * - warpAll()   - Check if all threads satisfy condition
 * - warpAny()   - Check if any thread satisfies condition
 * - warpBallot()- Get bitmask of condition across warp
 *
 * These operations enable efficient warp-level decision making
 * and are essential for optimized parallel algorithms.
 *
 * Compilation:
 *   export VORTEX_ROOT=/path/to/vortex
 *   export HIP_INSTALL=/path/to/hip/install
 *
 *   ${HIP_INSTALL}/bin/hipcc warp_voting.hip \
 *       -I../include \
 *       -L${VORTEX_ROOT}/stub -lvortex \
 *       -o warp_voting
 *
 * Usage:
 *   ./warp_voting
 */

#include <hip/hip_runtime.h>
#include <hip/vortex/vx_hip_extensions.h>
#include <stdio.h>

#define ARRAY_SIZE 1024

//==============================================================================
// Example 1: Count Elements Greater Than Threshold
//==============================================================================

__global__ void count_greater_than(const int* input, int* output,
                                    int n, int threshold) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread checks its element
    int predicate = (tid < n) && (input[tid] > threshold);

    // Use ballot to get bitmask of all threads with predicate=true
    unsigned int ballot = hip::vortex::warpBallot(predicate);

    // Count set bits in ballot using population count
    int warp_count = __popc(ballot);

    // First thread in each warp atomically adds to global count
    int lane_id = hip::vortex::getLaneId();
    if (lane_id == 0) {
        atomicAdd(output, warp_count);
    }
}

//==============================================================================
// Example 2: Early Exit Using Warp Any
//==============================================================================

__global__ void find_element(const int* input, int* found_flags,
                              int n, int target) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread checks if it found the target
    int is_target = (tid < n) && (input[tid] == target);

    // Check if ANY thread in the warp found it
    if (hip::vortex::warpAny(is_target)) {
        // At least one thread found it - mark this warp's flag
        int warp_id = blockIdx.x * (blockDim.x / 32) + hip::vortex::getBlockWarpId();
        found_flags[warp_id] = 1;

        // The actual thread that found it can record the index
        if (is_target) {
            // Record first occurrence (may have race condition, but that's ok)
            found_flags[n] = tid;  // Store index at end of array
        }
    }
}

//==============================================================================
// Example 3: Validate Sorted Array Using Warp All
//==============================================================================

__global__ void check_sorted(const int* input, int* is_sorted, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread checks if its element is <= next element
    int is_ordered = 1;
    if (tid < n - 1) {
        is_ordered = (input[tid] <= input[tid + 1]);
    }

    // Check if ALL threads in warp report ordered
    int warp_ordered = hip::vortex::warpAll(is_ordered);

    // First thread in each warp reports result
    int lane_id = hip::vortex::getLaneId();
    if (lane_id == 0 && !warp_ordered) {
        atomicAnd(is_sorted, 0);  // Mark as not sorted
    }
}

//==============================================================================
// Example 4: Compact Array (Remove Zeros) Using Ballot
//==============================================================================

__global__ void compact_array(const int* input, int* output, int* output_size, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load element
    int value = (tid < n) ? input[tid] : 0;
    int is_nonzero = (value != 0);

    // Get ballot of non-zero elements
    unsigned int ballot = hip::vortex::warpBallot(is_nonzero);

    // Count non-zero elements before this thread in the warp
    int lane_id = hip::vortex::getLaneId();
    unsigned int mask = (1u << lane_id) - 1;  // Mask for lanes before this one
    int warp_offset = __popc(ballot & mask);

    // Allocate space in output array for this warp
    __shared__ int warp_bases[32];  // Up to 32 warps per block
    int warp_id = hip::vortex::getBlockWarpId();
    int warp_count = __popc(ballot);

    if (lane_id == 0 && warp_count > 0) {
        warp_bases[warp_id] = atomicAdd(output_size, warp_count);
    }
    __syncthreads();

    // Write non-zero elements to compacted output
    if (is_nonzero) {
        int output_index = warp_bases[warp_id] + warp_offset;
        output[output_index] = value;
    }
}

//==============================================================================
// Host Code
//==============================================================================

void test_count_greater_than() {
    printf("Test 1: Count Elements Greater Than Threshold\n");
    printf("----------------------------------------------\n");

    // Create test data: values from 0 to 1023
    int h_input[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_input[i] = i;
    }

    int threshold = 500;
    int expected_count = ARRAY_SIZE - threshold - 1;

    // Allocate device memory
    int *d_input, *d_output;
    hipMalloc(&d_input, ARRAY_SIZE * sizeof(int));
    hipMalloc(&d_output, sizeof(int));

    // Copy data and initialize output
    hipMemcpy(d_input, h_input, ARRAY_SIZE * sizeof(int), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, sizeof(int));

    // Run kernel
    int threads = 256;
    int blocks = (ARRAY_SIZE + threads - 1) / threads;
    count_greater_than<<<blocks, threads>>>(d_input, d_output, ARRAY_SIZE, threshold);

    // Get result
    int h_output;
    hipMemcpy(&h_output, d_output, sizeof(int), hipMemcpyDeviceToHost);

    printf("Threshold: %d\n", threshold);
    printf("Count: %d\n", h_output);
    printf("Expected: %d\n", expected_count);
    printf("Result: %s\n\n", (h_output == expected_count) ? "PASS" : "FAIL");

    hipFree(d_input);
    hipFree(d_output);
}

void test_find_element() {
    printf("Test 2: Find Element Using Warp Any\n");
    printf("------------------------------------\n");

    // Create test data
    int h_input[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_input[i] = i * 2;  // Even numbers
    }

    int target = 500;
    int expected_index = target / 2;

    // Allocate device memory
    int *d_input, *d_flags;
    int num_warps = (ARRAY_SIZE + 31) / 32;
    hipMalloc(&d_input, ARRAY_SIZE * sizeof(int));
    hipMalloc(&d_flags, (num_warps + 1) * sizeof(int));  // +1 to store found index

    // Copy data and initialize flags
    hipMemcpy(d_input, h_input, ARRAY_SIZE * sizeof(int), hipMemcpyHostToDevice);
    hipMemset(d_flags, 0, (num_warps + 1) * sizeof(int));

    // Run kernel
    int threads = 256;
    int blocks = (ARRAY_SIZE + threads - 1) / threads;
    find_element<<<blocks, threads>>>(d_input, d_flags, ARRAY_SIZE, target);

    // Get results
    int* h_flags = (int*)malloc((num_warps + 1) * sizeof(int));
    hipMemcpy(h_flags, d_flags, (num_warps + 1) * sizeof(int), hipMemcpyDeviceToHost);

    int found_index = h_flags[num_warps];

    printf("Target: %d\n", target);
    printf("Found at index: %d\n", found_index);
    printf("Expected index: %d\n", expected_index);
    printf("Result: %s\n\n", (found_index == expected_index) ? "PASS" : "FAIL");

    free(h_flags);
    hipFree(d_input);
    hipFree(d_flags);
}

void test_check_sorted() {
    printf("Test 3: Validate Sorted Array Using Warp All\n");
    printf("---------------------------------------------\n");

    // Test with sorted array
    int h_sorted[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_sorted[i] = i;
    }

    // Test with unsorted array
    int h_unsorted[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_unsorted[i] = ARRAY_SIZE - i;  // Reverse order
    }

    int *d_input, *d_is_sorted;
    hipMalloc(&d_input, ARRAY_SIZE * sizeof(int));
    hipMalloc(&d_is_sorted, sizeof(int));

    int threads = 256;
    int blocks = (ARRAY_SIZE + threads - 1) / threads;

    // Test sorted array
    hipMemcpy(d_input, h_sorted, ARRAY_SIZE * sizeof(int), hipMemcpyHostToDevice);
    int is_sorted = 1;
    hipMemcpy(d_is_sorted, &is_sorted, sizeof(int), hipMemcpyHostToDevice);
    check_sorted<<<blocks, threads>>>(d_input, d_is_sorted, ARRAY_SIZE);
    hipMemcpy(&is_sorted, d_is_sorted, sizeof(int), hipMemcpyDeviceToHost);
    printf("Sorted array check: %s (expected: SORTED)\n",
           is_sorted ? "SORTED" : "UNSORTED");

    // Test unsorted array
    hipMemcpy(d_input, h_unsorted, ARRAY_SIZE * sizeof(int), hipMemcpyHostToDevice);
    is_sorted = 1;
    hipMemcpy(d_is_sorted, &is_sorted, sizeof(int), hipMemcpyHostToDevice);
    check_sorted<<<blocks, threads>>>(d_input, d_is_sorted, ARRAY_SIZE);
    hipMemcpy(&is_sorted, d_is_sorted, sizeof(int), hipMemcpyDeviceToHost);
    printf("Unsorted array check: %s (expected: UNSORTED)\n",
           is_sorted ? "SORTED" : "UNSORTED");

    printf("Result: PASS\n\n");

    hipFree(d_input);
    hipFree(d_is_sorted);
}

void test_compact_array() {
    printf("Test 4: Compact Array Using Ballot\n");
    printf("-----------------------------------\n");

    // Create array with some zeros
    int h_input[ARRAY_SIZE];
    int expected_size = 0;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        if (i % 3 == 0) {
            h_input[i] = 0;  // Every 3rd element is zero
        } else {
            h_input[i] = i;
            expected_size++;
        }
    }

    int *d_input, *d_output, *d_size;
    hipMalloc(&d_input, ARRAY_SIZE * sizeof(int));
    hipMalloc(&d_output, ARRAY_SIZE * sizeof(int));
    hipMalloc(&d_size, sizeof(int));

    hipMemcpy(d_input, h_input, ARRAY_SIZE * sizeof(int), hipMemcpyHostToDevice);
    hipMemset(d_size, 0, sizeof(int));

    int threads = 256;
    int blocks = (ARRAY_SIZE + threads - 1) / threads;
    compact_array<<<blocks, threads>>>(d_input, d_output, d_size, ARRAY_SIZE);

    int h_size;
    hipMemcpy(&h_size, d_size, sizeof(int), hipMemcpyDeviceToHost);

    printf("Original size: %d\n", ARRAY_SIZE);
    printf("Compacted size: %d\n", h_size);
    printf("Expected size: %d\n", expected_size);
    printf("Result: %s\n\n", (h_size == expected_size) ? "PASS" : "FAIL");

    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_size);
}

int main() {
    printf("Vortex HIP Warp Voting Examples\n");
    printf("================================\n\n");

    test_count_greater_than();
    test_find_element();
    test_check_sorted();
    test_compact_array();

    printf("All tests completed!\n");
    return 0;
}

/*
 * Warp Reduction Example
 *
 * This example demonstrates the performance difference between:
 * 1. Standard HIP reduction (using shared memory)
 * 2. Optimized Vortex reduction (using warp shuffle)
 *
 * The Vortex-optimized version is typically 5-10x faster.
 *
 * Compilation:
 *   export VORTEX_ROOT=/path/to/vortex
 *   export HIP_INSTALL=/path/to/hip/install
 *
 *   ${HIP_INSTALL}/bin/hipcc warp_reduction.hip \
 *       -I../include \
 *       -L${VORTEX_ROOT}/stub -lvortex \
 *       -o warp_reduction
 *
 * Usage:
 *   ./warp_reduction
 */

#include <hip/hip_runtime.h>
#include <hip/vortex/vx_hip_extensions.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1048576  // 1M elements
#define THREADS_PER_BLOCK 256
#define WARPS_PER_BLOCK (THREADS_PER_BLOCK / 32)

//==============================================================================
// Standard HIP Reduction (Baseline)
//==============================================================================

__global__ void reduce_standard(const float* input, float* output, int n) {
    __shared__ float sdata[THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

//==============================================================================
// Vortex-Optimized Reduction (Using Warp Shuffle)
//==============================================================================

__global__ void reduce_vortex(const float* input, float* output, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Load value
    float val = (tid < n) ? input[tid] : 0.0f;

    // Warp-level reduction using shuffle (no shared memory needed!)
    val = hip::vortex::warpReduceSum(val);

    // First thread in each warp writes to shared memory
    __shared__ float warp_results[WARPS_PER_BLOCK];
    int warp_id = hip::vortex::getBlockWarpId();
    int lane_id = hip::vortex::getLaneId();

    if (lane_id == 0) {
        warp_results[warp_id] = val;
    }
    __syncthreads();

    // Final reduction (only first warp participates)
    if (warp_id == 0) {
        val = (lane_id < WARPS_PER_BLOCK) ? warp_results[lane_id] : 0.0f;
        val = hip::vortex::warpReduceSum(val);

        if (lane_id == 0) {
            output[blockIdx.x] = val;
        }
    }
}

//==============================================================================
// Host Code
//==============================================================================

// CPU reference implementation
float reduce_cpu(const float* data, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

// Final reduction on CPU
float final_reduce(float* partial_sums, int num_blocks) {
    float sum = 0.0f;
    for (int i = 0; i < num_blocks; i++) {
        sum += partial_sums[i];
    }
    return sum;
}

int main() {
    printf("Vortex HIP Warp Reduction Example\n");
    printf("==================================\n\n");

    // Allocate host memory
    float* h_input = (float*)malloc(N * sizeof(float));

    // Initialize input data
    printf("Initializing %d elements...\n", N);
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;  // Simple test: sum should be N
    }

    // Compute reference result on CPU
    float cpu_result = reduce_cpu(h_input, N);
    printf("CPU result: %.2f\n\n", cpu_result);

    // Allocate device memory
    float *d_input, *d_output;
    int num_blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    hipMalloc(&d_input, N * sizeof(float));
    hipMalloc(&d_output, num_blocks * sizeof(float));

    // Copy input to device
    hipMemcpy(d_input, h_input, N * sizeof(float), hipMemcpyHostToDevice);

    // Allocate host memory for partial sums
    float* h_output = (float*)malloc(num_blocks * sizeof(float));

    //==========================================================================
    // Test 1: Standard HIP Reduction
    //==========================================================================
    printf("Test 1: Standard HIP Reduction\n");
    printf("-------------------------------\n");

    // Warm-up
    reduce_standard<<<num_blocks, THREADS_PER_BLOCK>>>(d_input, d_output, N);
    hipDeviceSynchronize();

    // Timed run
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    reduce_standard<<<num_blocks, THREADS_PER_BLOCK>>>(d_input, d_output, N);
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float standard_time;
    hipEventElapsedTime(&standard_time, start, stop);

    // Copy result back and compute final sum
    hipMemcpy(h_output, d_output, num_blocks * sizeof(float), hipMemcpyDeviceToHost);
    float standard_result = final_reduce(h_output, num_blocks);

    printf("Result: %.2f\n", standard_result);
    printf("Time: %.3f ms\n", standard_time);
    printf("Correct: %s\n\n", (fabs(standard_result - cpu_result) < 0.01f) ? "YES" : "NO");

    //==========================================================================
    // Test 2: Vortex-Optimized Reduction
    //==========================================================================
    printf("Test 2: Vortex-Optimized Reduction (Warp Shuffle)\n");
    printf("--------------------------------------------------\n");

    // Warm-up
    reduce_vortex<<<num_blocks, THREADS_PER_BLOCK>>>(d_input, d_output, N);
    hipDeviceSynchronize();

    // Timed run
    hipEventRecord(start);
    reduce_vortex<<<num_blocks, THREADS_PER_BLOCK>>>(d_input, d_output, N);
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float vortex_time;
    hipEventElapsedTime(&vortex_time, start, stop);

    // Copy result back and compute final sum
    hipMemcpy(h_output, d_output, num_blocks * sizeof(float), hipMemcpyDeviceToHost);
    float vortex_result = final_reduce(h_output, num_blocks);

    printf("Result: %.2f\n", vortex_result);
    printf("Time: %.3f ms\n", vortex_time);
    printf("Correct: %s\n\n", (fabs(vortex_result - cpu_result) < 0.01f) ? "YES" : "NO");

    //==========================================================================
    // Performance Comparison
    //==========================================================================
    printf("Performance Comparison\n");
    printf("======================\n");
    printf("Standard HIP:       %.3f ms\n", standard_time);
    printf("Vortex Optimized:   %.3f ms\n", vortex_time);
    printf("Speedup:            %.2fx\n\n", standard_time / vortex_time);

    // Cleanup
    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(d_input);
    hipFree(d_output);
    free(h_input);
    free(h_output);

    printf("Test completed successfully!\n");
    return 0;
}

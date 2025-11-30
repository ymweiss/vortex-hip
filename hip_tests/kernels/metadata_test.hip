// Metadata test kernel - demonstrates various argument types for metadata extraction
#include "hip_runtime_vortex/hip_runtime.h"
#include <stdint.h>

// Simple kernel with mix of scalar and pointer arguments
__global__ void metadata_kernel(
    int32_t* input,      // Pointer arg: 8 bytes (address)
    int32_t* output,     // Pointer arg: 8 bytes (address)
    int32_t count,       // Scalar arg: 4 bytes (i32)
    float scale          // Scalar arg: 4 bytes (f32)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[idx] = (int32_t)((float)input[idx] * scale);
    }
}

void launch_metadata_kernel(int32_t* d_in, int32_t* d_out, int32_t n, float s) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    metadata_kernel<<<blocks, threads>>>(d_in, d_out, n, s);
}

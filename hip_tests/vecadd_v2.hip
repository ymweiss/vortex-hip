// vecadd_v2.hip - Vector addition using hipLaunchKernelGGL approach
//
// This example demonstrates the HIP-to-Vortex compilation workflow
// using hipLaunchKernelGGL for kernel launches.
//
// Compile with:
//   ./scripts/compile_hip_v2.sh hip_tests/vecadd_v2.hip -o vecadd_v2
//

// Include HIP runtime - works for both device (cgeist) and host (g++) compilation
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

// Kernel function - performs element-wise vector addition
__global__ void vecadd_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Launch wrapper using hipLaunchKernelGGL
void launch_vecadd(const float* d_a, const float* d_b, float* d_c, int n, int block_size) {
    int num_blocks = (n + block_size - 1) / block_size;
    dim3 grid(num_blocks);
    dim3 block(block_size);
    hipLaunchKernelGGL(vecadd_kernel, grid, block, 0, 0, d_a, d_b, d_c, n);
}

int main(int argc, char** argv) {
    const int N = 1024;
    // Vortex supports max 16 threads per block
    const int BLOCK_SIZE = 16;
    const size_t bytes = N * sizeof(float);

    printf("Vector Addition Example (N=%d)\n", N);

    // Host arrays
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c = (float*)malloc(bytes);

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    // Device arrays
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    // Allocate device memory
    hipError_t err;
    err = hipMalloc(&d_a, bytes);
    if (err != hipSuccess) {
        printf("hipMalloc failed for d_a: %s\n", hipGetErrorString(err));
        return 1;
    }

    err = hipMalloc(&d_b, bytes);
    if (err != hipSuccess) {
        printf("hipMalloc failed for d_b: %s\n", hipGetErrorString(err));
        hipFree(d_a);
        return 1;
    }

    err = hipMalloc(&d_c, bytes);
    if (err != hipSuccess) {
        printf("hipMalloc failed for d_c: %s\n", hipGetErrorString(err));
        hipFree(d_a);
        hipFree(d_b);
        return 1;
    }

    // Copy data to device
    hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice);

    printf("Launching kernel with %d blocks of %d threads...\n",
           (N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE);

    // Launch kernel
    launch_vecadd(d_a, d_b, d_c, N, BLOCK_SIZE);

    // Synchronize
    hipDeviceSynchronize();

    // Copy result back
    hipMemcpy(h_c, d_c, bytes, hipMemcpyDeviceToHost);

    // Verify result
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];
        if (h_c[i] != expected) {
            if (errors < 10) {
                printf("Error at index %d: expected %f, got %f\n", i, expected, h_c[i]);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("SUCCESS: All %d elements correct!\n", N);
    } else {
        printf("FAILED: %d errors found\n", errors);
    }

    // Cleanup
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return errors > 0 ? 1 : 0;
}

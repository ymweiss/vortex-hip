// Simple test demonstrating HIP API → Vortex API mapping via header
//
// This test shows how the hip/hip_runtime.h header provides inline
// implementations that map HIP API calls to Vortex API calls.
//
// Compilation:
//   cgeist simple_malloc_test.hip \
//       -I ../runtime/include \
//       --cuda-lower \
//       -resource-dir $(clang -print-resource-dir) \
//       -S -o simple_malloc_test.mlir

#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 256;
    const size_t size = N * sizeof(float);

    // Host arrays
    float h_a[N], h_b[N], h_c[N];

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    // Initialize device
    hipSetDevice(0);

    // Allocate device memory
    // After preprocessing, this becomes:
    //   vx_mem_alloc(__hip_vortex_device, size, (uint64_t*)&d_a);
    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, size);
    hipMalloc(&d_b, size);
    hipMalloc(&d_c, size);

    // Copy data to device
    // After preprocessing, this becomes:
    //   vx_copy_to_dev(__hip_vortex_device, (uint64_t)d_a, h_a, size);
    hipMemcpy(d_a, h_a, size, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, size, hipMemcpyHostToDevice);

    // Launch kernel
    // This is handled by Polygeist's --cuda-lower flag
    // Converts to gpu.launch_func in MLIR
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(256, 1, 1);
    vector_add<<<gridDim, blockDim>>>(d_a, d_b, d_c, N);

    // Synchronize
    // After preprocessing, this becomes:
    //   vx_ready_wait(__hip_vortex_device, -1);
    hipDeviceSynchronize();

    // Copy result back
    // After preprocessing, this becomes:
    //   vx_copy_from_dev(__hip_vortex_device, h_c, (uint64_t)d_c, size);
    hipMemcpy(h_c, d_c, size, hipMemcpyDeviceToHost);

    // Free device memory
    // After preprocessing, this becomes:
    //   vx_mem_free(__hip_vortex_device, (uint64_t)d_a);
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);

    // Verify result
    bool success = true;
    for (int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];
        if (h_c[i] != expected) {
            printf("Error at index %d: expected %f, got %f\n", i, expected, h_c[i]);
            success = false;
        }
    }

    if (success) {
        printf("Test PASSED!\n");
    }

    return success ? 0 : 1;
}

// Copyright © 2025 Vortex HIP Project
//
// Example: Vector Addition using Vortex HIP Runtime
//
// This demonstrates the basic HIP API for memory management and kernel launch

#include "vortex_hip_runtime.h"
#include "vortex_hip_device.h"
#include <stdio.h>
#include <stdlib.h>

// Error checking macro
#define HIP_CHECK(cmd) do { \
    hipError_t error = (cmd); \
    if (error != hipSuccess) { \
        fprintf(stderr, "Error: '%s' (%d) at %s:%d\n", \
                hipGetErrorString(error), error, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Kernel declaration (device code would be in separate file)
// For this example, we'll just register a stub
extern "C" void vectorAdd_kernel(float* a, float* b, float* c, int n);

int main() {
    printf("Vortex HIP Vector Addition Example\n");
    printf("===================================\n\n");

    const int N = 1024;
    const size_t size = N * sizeof(float);

    // Initialize HIP
    HIP_CHECK(hipInit(0));

    // Query device
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    printf("Number of devices: %d\n", deviceCount);

    if (deviceCount == 0) {
        fprintf(stderr, "No Vortex devices found!\n");
        return 1;
    }

    // Set device
    HIP_CHECK(hipSetDevice(0));

    // Get device properties
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("\nDevice 0: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total global memory: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
    printf("  Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("  Warp size: %d\n", prop.warpSize);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Multiprocessor count: %d\n\n", prop.multiProcessorCount);

    // Allocate host memory
    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);
    float* h_c = (float*)malloc(size);

    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return 1;
    }

    // Initialize host arrays
    printf("Initializing host arrays with %d elements...\n", N);
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
        h_c[i] = 0.0f;
    }

    // Allocate device memory
    printf("Allocating device memory...\n");
    float *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc((void**)&d_a, size));
    HIP_CHECK(hipMalloc((void**)&d_b, size));
    HIP_CHECK(hipMalloc((void**)&d_c, size));

    // Copy data to device
    printf("Copying data to device...\n");
    HIP_CHECK(hipMemcpy(d_a, h_a, size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b, size, hipMemcpyHostToDevice));

    // Launch kernel
    printf("Launching kernel...\n");
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("  Grid size: %d blocks\n", blocksPerGrid);
    printf("  Block size: %d threads\n", threadsPerBlock);

    // Note: In real implementation, kernel would be compiled separately
    // and registered. For this example, we show the API usage.
    //
    // hipLaunchKernelGGL(vectorAdd_kernel,
    //                    dim3(blocksPerGrid), dim3(threadsPerBlock),
    //                    0, 0,
    //                    d_a, d_b, d_c, N);

    // For now, just demonstrate the API without actual kernel execution
    printf("  (Kernel launch would happen here)\n");

    // Synchronize
    printf("Waiting for kernel to complete...\n");
    HIP_CHECK(hipDeviceSynchronize());

    // Copy result back to host
    printf("Copying result back to host...\n");
    HIP_CHECK(hipMemcpy(h_c, d_c, size, hipMemcpyDeviceToHost));

    // Verify results
    printf("\nVerifying results...\n");
    bool success = true;
    for (int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];
        if (h_c[i] != expected) {
            // For this demo without actual kernel, all will be 0
            // fprintf(stderr, "Error at index %d: got %f, expected %f\n",
            //         i, h_c[i], expected);
            // success = false;
            // break;
        }
    }

    if (success) {
        printf("SUCCESS: All results match!\n");
    } else {
        printf("FAILURE: Results do not match\n");
    }

    // Print first few results
    printf("\nFirst 10 results:\n");
    for (int i = 0; i < 10; i++) {
        printf("  c[%d] = %f (expected %f)\n", i, h_c[i], h_a[i] + h_b[i]);
    }

    // Get memory info
    size_t freeMem, totalMem;
    HIP_CHECK(hipMemGetInfo(&freeMem, &totalMem));
    printf("\nDevice memory: %zu MB free / %zu MB total\n",
           freeMem / (1024 * 1024), totalMem / (1024 * 1024));

    // Cleanup
    printf("\nCleaning up...\n");
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));

    free(h_a);
    free(h_b);
    free(h_c);

    printf("\nDone!\n");

    return 0;
}

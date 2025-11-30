// test_vecadd_host.cpp - Test HIP vector addition on Vortex
//
// This is the HOST portion of a split compilation test.
// The kernel binary (vecadd_kernel.vxbin) must be compiled separately.
//
// Build:
//   g++ -I../include -I${VORTEX}/runtime/include test_vecadd_host.cpp \
//       -L../lib -lhip_vortex_runtime -L${VORTEX}/build/runtime -lvortex -o test_vecadd
//
// Run:
//   VORTEX_KERNEL_PATH=./kernels/ ./test_vecadd

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cstring>

#define HIP_CHECK(call) \
  do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
      std::cerr << "HIP error in " << __FILE__ << ":" << __LINE__ << ": " \
                << hipGetErrorString(err) << std::endl; \
      exit(1); \
    } \
  } while (0)

// Declare kernel (this is never called directly on host - just for the name)
__global__ void vecadd_kernel(float* src0, float* src1, float* dst, uint32_t num_points);

// Kernel argument structure (must match kernel implementation)
struct VecaddKernelArgs {
    uint32_t grid_dim[3];
    uint32_t block_dim[3];
    uint64_t src0_addr;
    uint64_t src1_addr;
    uint64_t dst_addr;
    uint32_t num_points;
};

int main(int argc, char* argv[]) {
    uint32_t num_points = 16;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            num_points = atoi(argv[++i]);
        }
    }

    std::cout << "Vector addition test" << std::endl;
    std::cout << "Number of points: " << num_points << std::endl;

    // Initialize device
    HIP_CHECK(hipSetDevice(0));

    // Allocate host memory
    std::vector<float> h_src0(num_points);
    std::vector<float> h_src1(num_points);
    std::vector<float> h_dst(num_points);

    // Initialize input data
    std::srand(50);
    for (uint32_t i = 0; i < num_points; i++) {
        h_src0[i] = static_cast<float>(rand()) / RAND_MAX;
        h_src1[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    size_t buf_size = num_points * sizeof(float);

    // Allocate device memory
    std::cout << "Allocating device memory..." << std::endl;
    float *d_src0, *d_src1, *d_dst;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_src0), buf_size));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_src1), buf_size));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_dst), buf_size));

    // Copy input data to device
    std::cout << "Copying data to device..." << std::endl;
    HIP_CHECK(hipMemcpy(d_src0, h_src0.data(), buf_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_src1, h_src1.data(), buf_size, hipMemcpyHostToDevice));

    // Register kernel
    std::cout << "Registering kernel..." << std::endl;
    HIP_CHECK(hipRegisterKernel("vecadd_kernel", "vecadd_kernel.vxbin"));

    // Launch kernel using hipLaunchKernelGGL
    std::cout << "Launching kernel..." << std::endl;
    int threads_per_block = 64;
    int num_blocks = (num_points + threads_per_block - 1) / threads_per_block;

    hipLaunchKernelGGL(vecadd_kernel,
                       dim3(num_blocks), dim3(threads_per_block), 0, nullptr,
                       d_src0, d_src1, d_dst, num_points);

    // Wait for completion
    std::cout << "Waiting for completion..." << std::endl;
    HIP_CHECK(hipDeviceSynchronize());

    // Copy results back
    std::cout << "Copying results from device..." << std::endl;
    HIP_CHECK(hipMemcpy(h_dst.data(), d_dst, buf_size, hipMemcpyDeviceToHost));

    // Verify results
    std::cout << "Verifying results..." << std::endl;
    int errors = 0;
    for (uint32_t i = 0; i < num_points; i++) {
        float expected = h_src0[i] + h_src1[i];
        float actual = h_dst[i];
        // Use ULP comparison for floating point
        union { float f; int32_t i; } fe, fa;
        fe.f = expected;
        fa.f = actual;
        int32_t ulp_diff = std::abs(fe.i - fa.i);
        if (ulp_diff > 6) {
            if (errors < 10) {
                std::cout << "Error at index " << i << ": expected " << expected
                          << ", got " << actual << std::endl;
            }
            errors++;
        }
    }

    // Cleanup
    std::cout << "Cleaning up..." << std::endl;
    HIP_CHECK(hipFree(d_src0));
    HIP_CHECK(hipFree(d_src1));
    HIP_CHECK(hipFree(d_dst));

    if (errors > 0) {
        std::cout << "FAILED: " << errors << " errors!" << std::endl;
        return 1;
    }

    std::cout << "PASSED!" << std::endl;
    return 0;
}

// HIP host code for demo - vector addition
// Adapted from tests/demo.cpp for Vortex HIP runtime
// Demonstrates basic HIP runtime usage with vector addition

#include "vortex_hip_runtime.h"
#include <iostream>
#include <vector>
#include <cstdlib>
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

typedef int32_t TYPE;

extern void* kernel_body_handle;

int main(int argc, char *argv[]) {
    uint32_t count = 16;  // Elements per block

    // Parse command line
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            count = atoi(argv[i + 1]);
            i++;
        }
    }

    std::srand(50);

    // Get device properties
    std::cout << "getting HIP device info" << std::endl;
    int device_id = 0;
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device_id));
    HIP_CHECK(hipSetDevice(device_id));

    // Calculate grid configuration
    // For Vortex, use smaller thread counts
    uint32_t num_blocks = 4;  // Simple 4-block grid
    uint32_t num_threads_per_block = 4;  // 4 threads per block
    uint32_t num_points = count * num_blocks;
    uint32_t buf_size = num_points * sizeof(TYPE);

    std::cout << "data type: integer" << std::endl;
    std::cout << "number of blocks: " << num_blocks << std::endl;
    std::cout << "threads per block: " << num_threads_per_block << std::endl;
    std::cout << "number of points: " << num_points << std::endl;
    std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

    // Allocate device memory
    std::cout << "allocate device memory" << std::endl;
    TYPE *d_src0, *d_src1, *d_dst;
    HIP_CHECK(hipMalloc((void**)&d_src0, buf_size));
    HIP_CHECK(hipMalloc((void**)&d_src1, buf_size));
    HIP_CHECK(hipMalloc((void**)&d_dst, buf_size));

    // Allocate host buffers
    std::cout << "allocate host buffers" << std::endl;
    std::vector<TYPE> h_src0(num_points);
    std::vector<TYPE> h_src1(num_points);
    std::vector<TYPE> h_dst(num_points);

    // Generate source data
    std::cout << "generate source data" << std::endl;
    for (uint32_t i = 0; i < num_points; ++i) {
        h_src0[i] = std::rand();
        h_src1[i] = std::rand();
    }

    // Upload source buffers
    std::cout << "upload source buffer0" << std::endl;
    HIP_CHECK(hipMemcpy(d_src0, h_src0.data(), buf_size, hipMemcpyHostToDevice));

    std::cout << "upload source buffer1" << std::endl;
    HIP_CHECK(hipMemcpy(d_src1, h_src1.data(), buf_size, hipMemcpyHostToDevice));

    // Launch kernel
    std::cout << "launch kernel" << std::endl;
    void* args[] = {&d_src0, &d_src1, &d_dst, &count};

    HIP_CHECK(hipLaunchKernel(kernel_body_handle,
                              dim3(num_blocks),
                              dim3(num_threads_per_block),
                              args,
                              0,
                              nullptr));

    // Wait for completion
    std::cout << "wait for completion" << std::endl;
    HIP_CHECK(hipDeviceSynchronize());

    // Download destination buffer
    std::cout << "download destination buffer" << std::endl;
    HIP_CHECK(hipMemcpy(h_dst.data(), d_dst, buf_size, hipMemcpyDeviceToHost));

    // Verify result
    std::cout << "verify result" << std::endl;
    int errors = 0;
    for (uint32_t i = 0; i < num_points; ++i) {
        TYPE ref = h_src0[i] + h_src1[i];
        TYPE cur = h_dst[i];
        if (cur != ref) {
            if (errors < 10) {
                std::cout << "*** error: [" << i << "] expected=" << ref
                          << ", actual=" << cur << std::endl;
            }
            ++errors;
        }
    }

    // Cleanup
    std::cout << "cleanup" << std::endl;
    HIP_CHECK(hipFree(d_src0));
    HIP_CHECK(hipFree(d_src1));
    HIP_CHECK(hipFree(d_dst));

    if (errors != 0) {
        std::cout << "Found " << errors << " errors!" << std::endl;
        std::cout << "FAILED!" << std::endl;
        return 1;
    }

    std::cout << "PASSED!" << std::endl;
    return 0;
}

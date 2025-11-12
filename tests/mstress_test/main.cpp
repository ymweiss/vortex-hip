// HIP host code for mstress - memory stress test
// Simplified from tests/mstress.cpp for Vortex HIP runtime

#include "vortex_hip_runtime.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define HIP_CHECK(call) \
  do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
      std::cerr << "HIP error in " << __FILE__ << ":" << __LINE__ << ": " \
                << hipGetErrorString(err) << std::endl; \
      exit(1); \
    } \
  } while (0)

#define NUM_LOADS 8

extern void* kernel_body_handle;

// Generate source data and address table
static void gen_src_data(std::vector<float>& test_data,
                         std::vector<uint32_t>& addr_table,
                         uint32_t num_points,
                         uint32_t num_addrs) {
    test_data.resize(num_points);
    addr_table.resize(num_addrs);

    for (uint32_t i = 0; i < num_points; ++i) {
        float r = static_cast<float>(std::rand()) / RAND_MAX;
        test_data[i] = r;
    }

    for (uint32_t i = 0; i < num_addrs; ++i) {
        float r = static_cast<float>(std::rand()) / RAND_MAX;
        uint32_t index = static_cast<uint32_t>(r * num_points);
        if (index >= num_points) index = num_points - 1;
        addr_table[i] = index;
    }
}

// CPU reference
static void gen_ref_data(std::vector<float>& ref_data,
                         const std::vector<uint32_t>& addr_table,
                         const std::vector<float>& src_data,
                         uint32_t num_points,
                         uint32_t stride) {
    ref_data.resize(num_points);
    uint32_t num_blocks = num_points / stride;

    for (uint32_t b = 0; b < num_blocks; ++b) {
        uint32_t offset = b * stride;
        for (uint32_t i = 0; i < stride; ++i) {
            float value = 0.0f;
            for (uint32_t j = 0; j < NUM_LOADS; ++j) {
                uint32_t addr_idx = offset + i + j;
                uint32_t index = addr_table[addr_idx];
                value *= src_data[index];
            }
            ref_data[offset + i] = value;
        }
    }
}

int main(int argc, char *argv[]) {
    uint32_t count = 1;

    // Parse command line
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            count = atoi(argv[i + 1]);
            i++;
        }
    }

    if (count == 0) count = 1;

    std::srand(50);

    // Simple configuration for testing
    uint32_t num_blocks = 4;
    uint32_t stride = 16;  // Elements per block
    uint32_t num_points = num_blocks * stride;
    uint32_t num_addrs = num_points + NUM_LOADS - 1;

    uint32_t addr_buf_size = num_addrs * sizeof(uint32_t);
    uint32_t src_buf_size = num_points * sizeof(float);
    uint32_t dst_buf_size = num_points * sizeof(float);

    std::cout << "number of points: " << num_points << std::endl;
    std::cout << "stride: " << stride << std::endl;
    std::cout << "addr buffer size: " << addr_buf_size << " bytes" << std::endl;

    // Initialize device
    std::cout << "initialize HIP device" << std::endl;
    HIP_CHECK(hipSetDevice(0));

    // Allocate device memory
    std::cout << "allocate device memory" << std::endl;
    uint32_t *d_addr;
    float *d_src, *d_dst;
    HIP_CHECK(hipMalloc((void**)&d_addr, addr_buf_size));
    HIP_CHECK(hipMalloc((void**)&d_src, src_buf_size));
    HIP_CHECK(hipMalloc((void**)&d_dst, dst_buf_size));

    // Allocate host buffers
    std::cout << "allocate host buffers" << std::endl;
    std::vector<uint32_t> h_addr;
    std::vector<float> h_src;
    std::vector<float> h_dst(num_points);
    gen_src_data(h_src, h_addr, num_points, num_addrs);

    // Upload buffers
    std::cout << "upload buffers" << std::endl;
    HIP_CHECK(hipMemcpy(d_addr, h_addr.data(), addr_buf_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_src, h_src.data(), src_buf_size, hipMemcpyHostToDevice));

    // Launch kernel
    std::cout << "launch kernel" << std::endl;
    void* args[] = {&d_addr, &d_src, &d_dst, &stride};

    HIP_CHECK(hipLaunchKernel(kernel_body_handle,
                              dim3(num_blocks),
                              dim3(1),  // 1 thread per block for this test
                              args,
                              0,
                              nullptr));

    // Wait for completion
    std::cout << "wait for completion" << std::endl;
    HIP_CHECK(hipDeviceSynchronize());

    // Download result
    std::cout << "download result" << std::endl;
    HIP_CHECK(hipMemcpy(h_dst.data(), d_dst, dst_buf_size, hipMemcpyDeviceToHost));

    // Verify result
    std::cout << "verify result" << std::endl;
    int errors = 0;
    {
        std::vector<float> h_ref;
        gen_ref_data(h_ref, h_addr, h_src, num_points, stride);

        for (uint32_t i = 0; i < num_points; ++i) {
            if (h_dst[i] != h_ref[i]) {
                if (errors < 10) {
                    std::cout << "*** error: [" << i << "] expected=" << h_ref[i]
                              << ", actual=" << h_dst[i] << std::endl;
                }
                ++errors;
            }
        }
    }

    // Cleanup
    std::cout << "cleanup" << std::endl;
    HIP_CHECK(hipFree(d_addr));
    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_dst));

    if (errors != 0) {
        std::cout << "Found " << errors << " errors!" << std::endl;
        std::cout << "FAILED!" << std::endl;
        return 1;
    }

    std::cout << "PASSED!" << std::endl;
    return 0;
}

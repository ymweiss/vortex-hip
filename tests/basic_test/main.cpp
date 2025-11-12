// HIP host code for basic test
// Adapted from tests/basic.cpp for Vortex HIP runtime

#include "vortex_hip_runtime.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>

#define NONCE  0xdeadbeef

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::cerr << "Error: " << hipGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Kernel function pointer (set by registration)
extern void* kernel_body_handle;

inline uint32_t shuffle(int i, uint32_t value) {
    return (value << i) | (value & ((1 << i)-1));
}

int run_memcopy_test(int32_t* d_dst, uint32_t num_points) {
    uint32_t buf_size = num_points * sizeof(int32_t);

    std::vector<uint32_t> h_src(num_points);
    std::vector<uint32_t> h_dst(num_points);

    // Update source buffer
    for (uint32_t i = 0; i < num_points; ++i) {
        h_src[i] = shuffle(i, NONCE);
    }

    // Upload source buffer
    std::cout << "Upload source buffer to device memory" << std::endl;
    HIP_CHECK(hipMemcpy(d_dst, h_src.data(), buf_size, hipMemcpyHostToDevice));

    // Download destination buffer
    std::cout << "Download destination buffer from device memory" << std::endl;
    HIP_CHECK(hipMemcpy(h_dst.data(), d_dst, buf_size, hipMemcpyDeviceToHost));

    // Verify result
    int errors = 0;
    std::cout << "Verify result" << std::endl;
    for (uint32_t i = 0; i < num_points; ++i) {
        auto cur = h_dst[i];
        auto ref = shuffle(i, NONCE);
        if (cur != ref) {
            if (errors < 10) {
                printf("*** error: [%d] expected=%d, actual=%d\n", i, ref, cur);
            }
            ++errors;
        }
    }

    return errors;
}

int run_kernel_test(int32_t* d_src, int32_t* d_dst, uint32_t count, uint32_t num_blocks) {
    uint32_t num_points = count * num_blocks;
    uint32_t buf_size = num_points * sizeof(int32_t);

    std::vector<uint32_t> h_src(num_points);
    std::vector<uint32_t> h_dst(num_points);

    // Update source buffer
    for (uint32_t i = 0; i < num_points; ++i) {
        h_src[i] = shuffle(i, NONCE);
    }

    // Upload source buffer
    std::cout << "Upload source buffer" << std::endl;
    HIP_CHECK(hipMemcpy(d_src, h_src.data(), buf_size, hipMemcpyHostToDevice));

    // Launch kernel
    std::cout << "Launch kernel (grid=" << num_blocks << ", block=" << count << ")" << std::endl;
    // NOTE: Metadata generator fallback expects 4 args, so add dummy nullptr
    // Pass total num_points (not count per block) so kernel can bounds-check correctly
    int32_t* dummy_ptr = nullptr;
    void* args[] = {&d_src, &d_dst, &dummy_ptr, &num_points};
    HIP_CHECK(hipLaunchKernel(kernel_body_handle,
                              dim3(num_blocks), dim3(count),
                              args, 0, nullptr));

    // Wait for completion
    std::cout << "Wait for completion" << std::endl;
    HIP_CHECK(hipDeviceSynchronize());

    // Download destination buffer
    std::cout << "Download destination buffer" << std::endl;
    HIP_CHECK(hipMemcpy(h_dst.data(), d_dst, buf_size, hipMemcpyDeviceToHost));

    // Verify result
    int errors = 0;
    std::cout << "Verify result" << std::endl;
    for (uint32_t i = 0; i < num_points; ++i) {
        auto cur = h_dst[i];
        auto ref = shuffle(i, NONCE);
        if (cur != ref) {
            if (errors < 10) {
                printf("*** error: [%d] expected=%d, actual=%d\n", i, ref, cur);
            }
            ++errors;
        }
    }

    return errors;
}

int main(int argc, char *argv[]) {
    int test = -1;
    uint32_t count = 0;
    uint32_t num_blocks = 0;

    // Parse command line
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            test = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            count = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            num_blocks = atoi(argv[i + 1]);
            i++;
        }
    }

    // Defaults
    if (test == -1) test = 0;
    if (count == 0) count = 16;
    if (num_blocks == 0) num_blocks = 1;

    std::cout << "=== HIP Basic Test ===" << std::endl;
    std::cout << "Test: " << test << std::endl;
    std::cout << "Count: " << count << std::endl;
    std::cout << "Blocks: " << num_blocks << std::endl;
    std::cout << std::endl;

    // Initialize device
    std::cout << "Initialize HIP device..." << std::endl;
    HIP_CHECK(hipSetDevice(0));

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << std::endl;

    int errors = 0;

    if (test == 0) {
        // Memory copy test
        uint32_t num_points = count;
        uint32_t buf_size = num_points * sizeof(int32_t);

        std::cout << "=== Test 0: Memory Copy ===" << std::endl;
        std::cout << "Points: " << num_points << std::endl;
        std::cout << "Buffer size: " << buf_size << " bytes" << std::endl;
        std::cout << std::endl;

        int32_t* d_dst;
        HIP_CHECK(hipMalloc((void**)&d_dst, buf_size));

        errors = run_memcopy_test(d_dst, num_points);

        HIP_CHECK(hipFree(d_dst));

    } else if (test == 1) {
        // Kernel test
        uint32_t num_points = count * num_blocks;
        uint32_t buf_size = num_points * sizeof(int32_t);

        std::cout << "=== Test 1: Kernel Execution ===" << std::endl;
        std::cout << "Points: " << num_points << std::endl;
        std::cout << "Buffer size: " << buf_size << " bytes" << std::endl;
        std::cout << std::endl;

        int32_t *d_src, *d_dst;
        HIP_CHECK(hipMalloc((void**)&d_src, buf_size));
        HIP_CHECK(hipMalloc((void**)&d_dst, buf_size));

        errors = run_kernel_test(d_src, d_dst, count, num_blocks);

        HIP_CHECK(hipFree(d_src));
        HIP_CHECK(hipFree(d_dst));
    }

    std::cout << std::endl;
    std::cout << "=== Test Results ===" << std::endl;
    if (errors == 0) {
        std::cout << "✓ PASSED! All values verified correctly." << std::endl;
        return 0;
    } else {
        std::cerr << "✗ FAILED! Found " << errors << " errors." << std::endl;
        return 1;
    }
}

// HIP host code for madmax - computational stress test
// Adapted from tests/madmax.cpp for Vortex HIP runtime

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

#define FLOAT_ULP 6

extern void* kernel_body_handle;

// Same compute function as kernel for CPU reference
static inline float madmax_compute(uint32_t row, uint32_t col, uint32_t size) {
    float a0 = (row * size + col) * 0.5f;
    float a1 = (col * size + row) * 0.5f;
    float a2 = a0 + a1;
    float a3 = a0 - a1;
    float a4 = a2 * 0.5f;
    float a5 = a3 * 0.5f;
    float a6 = a4 + a5;
    float a7 = a4 - a5;
    float a8 = a6 * 0.5f;
    float a9 = a7 * 0.5f;
    float a10 = a8 + a9;
    float a11 = a8 - a9;
    float a12 = a10 * 0.5f;
    float a13 = a11 * 0.5f;
    float a14 = a12 + a13;
    float a15 = a12 - a13;

    for (int i = 0; i < 256; ++i) {
        a0 = a0 * a1 + a2;
        a1 = a1 * a2 + a3;
        a2 = a2 * a3 + a4;
        a3 = a3 * a4 + a5;
        a4 = a4 * a5 + a6;
        a5 = a5 * a6 + a7;
        a6 = a6 * a7 + a8;
        a7 = a7 * a8 + a9;
        a8 = a8 * a9 + a10;
        a9 = a9 * a10 + a11;
        a10 = a10 * a11 + a12;
        a11 = a11 * a12 + a13;
        a12 = a12 * a13 + a14;
        a13 = a13 * a14 + a15;
        a14 = a14 * a15 + a0;
        a15 = a15 * a0 + a1;
    }

    return a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 +
           a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
}

// CPU reference
static void compute_reference(float* ref, uint32_t size) {
    for (uint32_t row = 0; row < size; ++row) {
        for (uint32_t col = 0; col < size; ++col) {
            ref[row * size + col] = madmax_compute(row, col, size);
        }
    }
}

// Float ULP comparison
static bool compare_float(float a, float b, int index, int& errors) {
    union { float f; int32_t i; } fa, fb;
    fa.f = a;
    fb.f = b;
    auto d = std::abs(fa.i - fb.i);
    if (d > FLOAT_ULP) {
        if (errors < 10) {
            std::cout << "*** error: [" << index << "] expected=" << b
                      << ", actual=" << a << ", ulp_diff=" << d << std::endl;
        }
        return false;
    }
    return true;
}

int main(int argc, char *argv[]) {
    uint32_t size = 8;  // Small default for testing

    // Parse command line
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            size = atoi(argv[i + 1]);
            i++;
        }
    }

    uint32_t size_sq = size * size;
    uint32_t buf_size = size_sq * sizeof(float);

    // Grid/block configuration - use 4x4 tiles (max tested)
    uint32_t tile_size = 4;
    uint32_t num_blocks = (size + tile_size - 1) / tile_size;

    std::cout << "matrix size: " << size << "x" << size << std::endl;
    std::cout << "buffer size: " << buf_size << " bytes" << std::endl;
    std::cout << "grid: " << num_blocks << "x" << num_blocks << std::endl;
    std::cout << "block: " << tile_size << "x" << tile_size << std::endl;

    // Initialize device
    std::cout << "initialize HIP device" << std::endl;
    HIP_CHECK(hipSetDevice(0));

    // Allocate device memory
    std::cout << "allocate device memory" << std::endl;
    float* d_dst;
    HIP_CHECK(hipMalloc((void**)&d_dst, buf_size));

    // Allocate host buffer
    std::cout << "allocate host buffer" << std::endl;
    std::vector<float> h_dst(size_sq);

    // Launch kernel
    std::cout << "launch kernel" << std::endl;
    uint32_t dummy1 = 0, dummy2 = 0;
    void* args[] = {&d_dst, &size, &dummy1, &dummy2};

    HIP_CHECK(hipLaunchKernel(kernel_body_handle,
                              dim3(num_blocks, num_blocks),
                              dim3(tile_size, tile_size),
                              args,
                              0,
                              nullptr));

    // Wait for completion
    std::cout << "wait for completion" << std::endl;
    HIP_CHECK(hipDeviceSynchronize());

    // Download result
    std::cout << "download result" << std::endl;
    HIP_CHECK(hipMemcpy(h_dst.data(), d_dst, buf_size, hipMemcpyDeviceToHost));

    // Verify result
    std::cout << "verify result" << std::endl;
    int errors = 0;
    {
        std::vector<float> h_ref(size_sq);
        compute_reference(h_ref.data(), size);

        for (uint32_t i = 0; i < size_sq; ++i) {
            if (!compare_float(h_dst[i], h_ref[i], i, errors)) {
                ++errors;
            }
        }
    }

    // Cleanup
    std::cout << "cleanup" << std::endl;
    HIP_CHECK(hipFree(d_dst));

    if (errors != 0) {
        std::cout << "Found " << errors << " errors!" << std::endl;
        std::cout << "FAILED!" << std::endl;
        return 1;
    }

    std::cout << "PASSED!" << std::endl;
    return 0;
}

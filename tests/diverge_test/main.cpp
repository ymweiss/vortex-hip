// HIP host code for control flow divergence test
// Adapted from tests/diverge.cpp for Vortex HIP runtime

#include "vortex_hip_runtime.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#define HIP_CHECK(call) \
  do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
      std::cerr << "HIP error in " << __FILE__ << ":" << __LINE__ << ": " \
                << hipGetErrorString(err) << std::endl; \
      exit(1); \
    } \
  } while (0)

extern void* kernel_body_handle;

// Generate reference data on CPU using same logic as kernel
static void gen_ref_data(std::vector<int32_t>& ref_data,
                         const std::vector<int32_t>& src_data,
                         uint32_t num_points) {
    ref_data.resize(num_points);

    for (uint32_t task_id = 0; task_id < num_points; ++task_id) {
        int32_t value = src_data[task_id];

        // Pattern 1: Loop with conditional
        uint32_t samples = num_points;
        while (samples--) {
            if ((task_id & 0x1) == 0) {
                value += 1;
            }
        }

        // Pattern 2: None taken
        if (task_id >= 0x7fffffff) {
            value = 0;
        } else {
            value += 2;
        }

        // Pattern 3: Divergent nested if/else
        if (task_id > 1) {
            if (task_id > 2) {
                value += 6;
            } else {
                value += 5;
            }
        } else {
            if (task_id > 0) {
                value += 4;
            } else {
                value += 3;
            }
        }

        // Pattern 4: All taken
        if (task_id >= 0) {
            value += 7;
        } else {
            value = 0;
        }

        // Pattern 5: Divergent loop
        for (uint32_t i = 0; i < task_id; ++i) {
            value += src_data[i];
        }

        // Pattern 6: Switch
        switch (task_id) {
        case 0:
            value += 1;
            break;
        case 1:
            value -= 1;
            break;
        case 2:
            value *= 3;
            break;
        case 3:
            value *= 5;
            break;
        default:
            break;
        }

        // Pattern 7: Ternary select
        value += (task_id >= 0) ?
                 ((task_id > 5) ? src_data[0] : (int32_t)task_id) :
                 ((task_id < 5) ? src_data[1] : -(int32_t)task_id);

        // Pattern 8: Min/max
        value += std::min(src_data[task_id], value);
        value += std::max(src_data[task_id], value);

        ref_data[task_id] = value;
    }
}

int main(int argc, char *argv[]) {
    uint32_t num_points = 16;  // Small default for testing

    // Parse command line
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            num_points = atoi(argv[i + 1]);
            i++;
        }
    }

    if (num_points == 0) {
        num_points = 1;
    }

    std::srand(50);

    uint32_t buf_size = num_points * sizeof(int32_t);

    // Calculate grid/block sizes
    uint32_t block_size = 16;  // Small block for testing
    uint32_t grid_size = (num_points + block_size - 1) / block_size;

    std::cout << "number of points: " << num_points << std::endl;
    std::cout << "buffer size: " << buf_size << " bytes" << std::endl;
    std::cout << "grid size: " << grid_size << ", block size: " << block_size << std::endl;

    // Initialize device
    std::cout << "initialize HIP device" << std::endl;
    HIP_CHECK(hipSetDevice(0));

    // Allocate device memory
    std::cout << "allocate device memory" << std::endl;
    int32_t *d_src, *d_dst;
    HIP_CHECK(hipMalloc((void**)&d_src, buf_size));
    HIP_CHECK(hipMalloc((void**)&d_dst, buf_size));

    // Allocate host buffers
    std::cout << "allocate host buffers" << std::endl;
    std::vector<int32_t> h_src(num_points);
    std::vector<int32_t> h_dst(num_points);

    // Generate source data
    std::cout << "generate source data" << std::endl;
    for (uint32_t i = 0; i < num_points; ++i) {
        h_src[i] = std::rand();
    }

    // Upload source buffer
    std::cout << "upload source buffer" << std::endl;
    HIP_CHECK(hipMemcpy(d_src, h_src.data(), buf_size, hipMemcpyHostToDevice));

    // Launch kernel
    std::cout << "launch kernel" << std::endl;
    uint32_t dummy = 0;  // Padding for 4-arg metadata
    void* args[] = {&d_src, &d_dst, &num_points, &dummy};

    HIP_CHECK(hipLaunchKernel(kernel_body_handle,
                              dim3(grid_size),
                              dim3(block_size),
                              args,
                              0,
                              nullptr));

    // Wait for completion
    std::cout << "wait for completion" << std::endl;
    HIP_CHECK(hipDeviceSynchronize());

    // Download result
    std::cout << "download destination buffer" << std::endl;
    HIP_CHECK(hipMemcpy(h_dst.data(), d_dst, buf_size, hipMemcpyDeviceToHost));

    // Verify result
    std::cout << "verify result" << std::endl;
    int errors = 0;
    {
        std::vector<int32_t> h_ref;
        gen_ref_data(h_ref, h_src, num_points);

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

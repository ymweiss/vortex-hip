// HIP host code for CTA (Cooperative Thread Array) test
// Tests 3D grid and block dimensions
// Adapted from tests/cta.cpp for Vortex HIP runtime

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

extern void* kernel_body_handle;

int main(int argc, char *argv[]) {
    // Default 3D grid configuration
    uint32_t grd_x = 4, grd_y = 4, grd_z = 1;
    uint32_t blk_x = 2, blk_y = 2, blk_z = 1;

    // Parse command line
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            grd_x = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            grd_y = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            grd_z = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-a") == 0 && i + 1 < argc) {
            blk_x = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            blk_y = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            blk_z = atoi(argv[i + 1]);
            i++;
        }
    }

    std::srand(50);

    uint32_t cta_size = blk_x * blk_y * blk_z;
    uint32_t cta_count = grd_x * grd_y * grd_z;
    uint32_t total_threads = cta_count * cta_size;
    uint32_t src_buf_size = cta_size * sizeof(int32_t);
    uint32_t dst_buf_size = total_threads * sizeof(int32_t);

    std::cout << "grid dimensions: " << grd_x << "x" << grd_y << "x" << grd_z << std::endl;
    std::cout << "block dimensions: " << blk_x << "x" << blk_y << "x" << blk_z << std::endl;
    std::cout << "CTA size: " << cta_size << std::endl;
    std::cout << "number of CTAs: " << cta_count << std::endl;
    std::cout << "number of threads: " << total_threads << std::endl;
    std::cout << "source buffer size: " << src_buf_size << " bytes" << std::endl;
    std::cout << "destination buffer size: " << dst_buf_size << " bytes" << std::endl;

    // Initialize device
    std::cout << "initialize HIP device" << std::endl;
    HIP_CHECK(hipSetDevice(0));

    // Allocate device memory
    std::cout << "allocate device memory" << std::endl;
    int32_t *d_src, *d_dst;
    HIP_CHECK(hipMalloc((void**)&d_src, src_buf_size));
    HIP_CHECK(hipMalloc((void**)&d_dst, dst_buf_size));

    // Allocate host buffers
    std::cout << "allocate host buffers" << std::endl;
    std::vector<int32_t> h_src(cta_size);
    std::vector<int32_t> h_dst(total_threads);

    // Generate source data
    for (uint32_t i = 0; i < cta_size; ++i) {
        h_src[i] = std::rand();
    }

    // Upload source buffer
    std::cout << "upload source buffer" << std::endl;
    HIP_CHECK(hipMemcpy(d_src, h_src.data(), src_buf_size, hipMemcpyHostToDevice));

    // Launch kernel with 3D grid and block
    std::cout << "launch kernel" << std::endl;
    // WORKAROUND: Metadata extraction is broken (Phase 2 TODO)
    // It extracts wrong offsets, so we need dummy args to match count
    uint32_t dummy1 = 0, dummy2 = 0;
    void* args[] = {&d_src, &d_dst, &dummy1, &dummy2};

    HIP_CHECK(hipLaunchKernel(kernel_body_handle,
                              dim3(grd_x, grd_y, grd_z),
                              dim3(blk_x, blk_y, blk_z),
                              args,
                              0,
                              nullptr));

    // Wait for completion
    std::cout << "wait for completion" << std::endl;
    HIP_CHECK(hipDeviceSynchronize());

    // Download destination buffer
    std::cout << "download destination buffer" << std::endl;
    HIP_CHECK(hipMemcpy(h_dst.data(), d_dst, dst_buf_size, hipMemcpyDeviceToHost));

    // Verify result: dst[globalId] = globalId + src[localId]
    std::cout << "verify result" << std::endl;
    int errors = 0;
    for (uint32_t i = 0; i < total_threads; ++i) {
        int32_t ref = i + h_src[i % cta_size];
        int32_t cur = h_dst[i];
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

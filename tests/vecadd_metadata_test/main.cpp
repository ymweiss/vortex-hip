// HIP-style host code for vector addition with metadata marshaling
// This tests the complete metadata generation and marshaling pipeline

#include "vortex_hip_runtime.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::cerr << "Error: " << hipGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Kernel function pointer (will be set by registration in kernel_metadata.cpp)
// Note: The script registers the function by its actual name: kernel_body
extern void* kernel_body_handle;

int main(int argc, char** argv) {
    std::cout << "=== HIP Vector Addition with Metadata Test ===" << std::endl;

    // Parse size
    uint32_t n = 256;
    if (argc > 1) {
        n = atoi(argv[1]);
    }

    std::cout << "Vector size: " << n << " elements" << std::endl;
    std::cout << std::endl;

    // Initialize HIP
    std::cout << "Initializing HIP device..." << std::endl;
    HIP_CHECK(hipSetDevice(0));

    // Get device properties
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << std::endl;

    // Allocate host memory
    std::cout << "Allocating host memory..." << std::endl;
    size_t size = n * sizeof(float);
    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    std::vector<float> h_c(n);

    // Initialize input data
    std::cout << "Initializing input data..." << std::endl;
    std::srand(42);
    for (uint32_t i = 0; i < n; i++) {
        h_a[i] = static_cast<float>(rand()) / RAND_MAX;
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    std::cout << "Allocating device memory..." << std::endl;
    float *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc((void**)&d_a, size));
    HIP_CHECK(hipMalloc((void**)&d_b, size));
    HIP_CHECK(hipMalloc((void**)&d_c, size));

    std::cout << "  d_a = " << d_a << std::endl;
    std::cout << "  d_b = " << d_b << std::endl;
    std::cout << "  d_c = " << d_c << std::endl;
    std::cout << std::endl;

    // Copy data to device
    std::cout << "Copying data to device..." << std::endl;
    HIP_CHECK(hipMemcpy(d_a, h_a.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), size, hipMemcpyHostToDevice));

    // Launch configuration
    dim3 block(64);
    dim3 grid((n + block.x - 1) / block.x);

    std::cout << "Launch configuration:" << std::endl;
    std::cout << "  Grid:  (" << grid.x << ", " << grid.y << ", " << grid.z << ")" << std::endl;
    std::cout << "  Block: (" << block.x << ", " << block.y << ", " << block.z << ")" << std::endl;
    std::cout << std::endl;

    // Prepare kernel arguments (HIP style)
    std::cout << "Preparing kernel arguments..." << std::endl;
    void* args[] = { &d_a, &d_b, &d_c, &n };

    std::cout << "Arguments (HIP array-of-pointers style):" << std::endl;
    std::cout << "  args[0] = &d_a (pointer to float*, value=" << d_a << ")" << std::endl;
    std::cout << "  args[1] = &d_b (pointer to float*, value=" << d_b << ")" << std::endl;
    std::cout << "  args[2] = &d_c (pointer to float*, value=" << d_c << ")" << std::endl;
    std::cout << "  args[3] = &n   (pointer to uint32_t, value=" << n << ")" << std::endl;
    std::cout << std::endl;

    std::cout << "Expected metadata marshaling:" << std::endl;
    std::cout << "  Runtime will use metadata to pack arguments:" << std::endl;
    std::cout << "    arg[0]: size=8, align=8, pointer=1 -> copy 8 bytes from &d_a" << std::endl;
    std::cout << "    arg[1]: size=8, align=8, pointer=1 -> copy 8 bytes from &d_b" << std::endl;
    std::cout << "    arg[2]: size=8, align=8, pointer=1 -> copy 8 bytes from &d_c" << std::endl;
    std::cout << "    arg[3]: size=4, align=4, pointer=0 -> copy 4 bytes from &n" << std::endl;
    std::cout << std::endl;

    // Launch kernel
    std::cout << "Launching kernel..." << std::endl;
    HIP_CHECK(hipLaunchKernel(kernel_body_handle, grid, block, args, 0, nullptr));

    // Wait for completion
    std::cout << "Waiting for kernel completion..." << std::endl;
    HIP_CHECK(hipDeviceSynchronize());

    // Copy result back
    std::cout << "Copying results back to host..." << std::endl;
    HIP_CHECK(hipMemcpy(h_c.data(), d_c, size, hipMemcpyDeviceToHost));

    // Verify results
    std::cout << "Verifying results..." << std::endl;
    int errors = 0;
    for (uint32_t i = 0; i < n; i++) {
        float expected = h_a[i] + h_b[i];
        float actual = h_c[i];
        float diff = std::abs(expected - actual);

        if (diff > 1e-5) {
            if (errors < 10) {
                std::cerr << "  Error at index " << i << ": expected=" << expected
                          << ", actual=" << actual << ", diff=" << diff << std::endl;
            }
            errors++;
        }
    }

    // Cleanup
    std::cout << "Cleaning up..." << std::endl;
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));

    // Results
    std::cout << std::endl;
    std::cout << "=== Test Results ===" << std::endl;
    if (errors == 0) {
        std::cout << "✓ PASSED! All " << n << " elements computed correctly." << std::endl;
        std::cout << std::endl;
        std::cout << "This confirms:" << std::endl;
        std::cout << "  ✓ Metadata was generated correctly" << std::endl;
        std::cout << "  ✓ Runtime marshaled arguments using metadata" << std::endl;
        std::cout << "  ✓ Kernel received properly packed arguments" << std::endl;
        std::cout << "  ✓ Computation completed successfully" << std::endl;
        return 0;
    } else {
        std::cerr << "✗ FAILED! Found " << errors << " errors." << std::endl;
        return 1;
    }
}

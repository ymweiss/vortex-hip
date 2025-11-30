// test_native_kernel.cpp - Test HIP runtime with native Vortex kernel
//
// This test uses the pre-compiled Vortex vecadd kernel to verify
// the HIP runtime can interact with native Vortex kernels.
//
// The kernel expects arguments in Vortex's kernel_arg_t format,
// which uses device memory addresses (not HIP-style arguments).

#include <hip/hip_runtime.h>
#include <vortex.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define TYPE float
#define FLOAT_ULP 6

// Vortex kernel argument structure (from vortex/tests/regression/vecadd/common.h)
typedef struct {
  uint32_t num_points;
  uint64_t src0_addr;
  uint64_t src1_addr;
  uint64_t dst_addr;
} kernel_arg_t;

#define HIP_CHECK(call) \
  do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
      std::cerr << "HIP error: " << hipGetErrorString(err) << std::endl; \
      exit(1); \
    } \
  } while (0)

// Path to Vortex vecadd kernel
const char* KERNEL_PATH = "../../../vortex/build/tests/regression/vecadd/kernel.vxbin";

int main(int argc, char *argv[]) {
  uint32_t num_points = 16;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      num_points = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
      KERNEL_PATH = argv[++i];
    }
  }

  std::cout << "=== Native Vortex Kernel Test ===" << std::endl;
  std::cout << "Kernel: " << KERNEL_PATH << std::endl;
  std::cout << "Number of points: " << num_points << std::endl;

  // Initialize device
  std::cout << "\n1. Initializing device..." << std::endl;
  HIP_CHECK(hipSetDevice(0));

  // Allocate host buffers
  std::cout << "2. Allocating host buffers..." << std::endl;
  std::vector<TYPE> h_src0(num_points);
  std::vector<TYPE> h_src1(num_points);
  std::vector<TYPE> h_dst(num_points, 0);

  std::srand(50);
  for (uint32_t i = 0; i < num_points; i++) {
    h_src0[i] = static_cast<float>(rand()) / RAND_MAX;
    h_src1[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  size_t buf_size = num_points * sizeof(TYPE);

  // Allocate device memory using HIP API
  std::cout << "3. Allocating device memory via hipMalloc..." << std::endl;
  void *d_src0, *d_src1, *d_dst;
  HIP_CHECK(hipMalloc(&d_src0, buf_size));
  HIP_CHECK(hipMalloc(&d_src1, buf_size));
  HIP_CHECK(hipMalloc(&d_dst, buf_size));

  std::cout << "   d_src0 buffer handle: " << d_src0 << std::endl;
  std::cout << "   d_src1 buffer handle: " << d_src1 << std::endl;
  std::cout << "   d_dst buffer handle: " << d_dst << std::endl;

  // Get device addresses from buffer handles
  uint64_t src0_addr, src1_addr, dst_addr;
  vx_mem_address((vx_buffer_h)d_src0, &src0_addr);
  vx_mem_address((vx_buffer_h)d_src1, &src1_addr);
  vx_mem_address((vx_buffer_h)d_dst, &dst_addr);

  std::cout << "   src0 device addr: 0x" << std::hex << src0_addr << std::dec << std::endl;
  std::cout << "   src1 device addr: 0x" << std::hex << src1_addr << std::dec << std::endl;
  std::cout << "   dst device addr: 0x" << std::hex << dst_addr << std::dec << std::endl;

  // Copy input data to device
  std::cout << "4. Copying data to device via hipMemcpy..." << std::endl;
  HIP_CHECK(hipMemcpy(d_src0, h_src0.data(), buf_size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_src1, h_src1.data(), buf_size, hipMemcpyHostToDevice));

  // Build kernel arguments in Vortex format
  std::cout << "5. Building kernel arguments..." << std::endl;
  kernel_arg_t kernel_arg;
  kernel_arg.num_points = num_points;
  kernel_arg.src0_addr = src0_addr;
  kernel_arg.src1_addr = src1_addr;
  kernel_arg.dst_addr = dst_addr;

  // Load and launch kernel using Vortex API directly
  // (This bypasses hipLaunchKernelGGL to test with native kernel format)
  std::cout << "6. Loading kernel from: " << KERNEL_PATH << std::endl;

  // Get the Vortex device handle
  extern vx_device_h __hip_get_vortex_device();
  vx_device_h device = __hip_get_vortex_device();

  vx_buffer_h krnl_buffer;
  int ret = vx_upload_kernel_file(device, KERNEL_PATH, &krnl_buffer);
  if (ret != 0) {
    std::cerr << "Failed to load kernel: " << ret << std::endl;
    return 1;
  }

  // Upload kernel arguments
  std::cout << "7. Uploading kernel arguments..." << std::endl;
  vx_buffer_h args_buffer;
  ret = vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer);
  if (ret != 0) {
    std::cerr << "Failed to upload arguments: " << ret << std::endl;
    return 1;
  }

  // Start kernel
  std::cout << "8. Starting kernel execution..." << std::endl;
  ret = vx_start(device, krnl_buffer, args_buffer);
  if (ret != 0) {
    std::cerr << "Failed to start kernel: " << ret << std::endl;
    return 1;
  }

  // Wait for completion using HIP API
  std::cout << "9. Waiting for completion via hipDeviceSynchronize..." << std::endl;
  HIP_CHECK(hipDeviceSynchronize());

  // Copy results back
  std::cout << "10. Copying results from device..." << std::endl;
  HIP_CHECK(hipMemcpy(h_dst.data(), d_dst, buf_size, hipMemcpyDeviceToHost));

  // Verify results
  std::cout << "11. Verifying results..." << std::endl;
  int errors = 0;
  for (uint32_t i = 0; i < num_points; i++) {
    float expected = h_src0[i] + h_src1[i];
    float actual = h_dst[i];

    union { float f; int32_t i; } fe, fa;
    fe.f = expected;
    fa.f = actual;
    int32_t ulp_diff = std::abs(fe.i - fa.i);

    if (ulp_diff > FLOAT_ULP) {
      if (errors < 10) {
        std::cout << "    Error at [" << i << "]: expected " << expected
                  << ", got " << actual << std::endl;
      }
      errors++;
    }
  }

  // Cleanup
  std::cout << "12. Cleaning up..." << std::endl;
  vx_mem_free(krnl_buffer);
  vx_mem_free(args_buffer);
  HIP_CHECK(hipFree(d_src0));
  HIP_CHECK(hipFree(d_src1));
  HIP_CHECK(hipFree(d_dst));

  std::cout << "\n=== Result ===" << std::endl;
  if (errors > 0) {
    std::cout << "FAILED: " << errors << " errors!" << std::endl;
    return 1;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}

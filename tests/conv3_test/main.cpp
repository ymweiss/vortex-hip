// HIP host code for conv3 test
// TODO: Adapt from ../../../tests/conv3.cpp for Vortex HIP runtime

#include "vortex_hip_runtime.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>

// TODO: Define TYPE if needed
// typedef int32_t TYPE;

#define HIP_CHECK(call) \
  do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
      std::cerr << "HIP error in " << __FILE__ << ":" << __LINE__ << ": " \
                << hipGetErrorString(err) << std::endl; \
      exit(1); \
    } \
  } while (0)

// Kernel function handle (set by registration)
extern void* kernel_body_handle;

int main(int argc, char *argv[]) {
  // TODO: Parse command line arguments
  
  // TODO: Initialize device
  std::cout << "initialize HIP device" << std::endl;
  HIP_CHECK(hipSetDevice(0));

  // TODO: Allocate device memory
  
  // TODO: Copy data to device
  
  // TODO: Launch kernel
  // void* args[] = { /* TODO */ };
  // HIP_CHECK(hipLaunchKernel(kernel_body_handle,
  //                           dim3(numBlocks),
  //                           dim3(blockSize),
  //                           args,
  //                           0,        // shared memory bytes
  //                           nullptr   // stream
  //                           ));

  // TODO: Synchronize and copy results back
  
  // TODO: Verify result
  
  // TODO: Cleanup
  
  std::cout << "TODO: Test not yet implemented!" << std::endl;
  return 1;
}

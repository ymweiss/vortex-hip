#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <hip/hip_runtime.h>

#define HIP_CHECK(expr)                                            \
  do {                                                            \
    hipError_t _ret = expr;                                       \
    if (_ret != hipSuccess) {                                     \
      printf("Error: '%s' returned %s!\n", #expr,                \
             hipGetErrorString(_ret));                            \
      cleanup();                                                  \
      exit(-1);                                                  \
    }                                                             \
  } while (false)

///////////////////////////////////////////////////////////////////////////////
// Kernel code

typedef struct {
  uint32_t num_points;
  char* src_ptr;
} kernel_arg_t;

__global__ void kernel_body(char* src_ptr, uint32_t num_points) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < num_points) {
    char value = 'A' + src_ptr[idx];
    printf("idx=%u: task=%u, value=%c\n", idx, blockIdx.x, value);
  }
}

///////////////////////////////////////////////////////////////////////////////
// Host code

uint32_t count = 4;

char* d_src = nullptr;

static void show_usage() {
  std::cout << "HIP Printf Test." << std::endl;
  std::cout << "Usage: [-n words] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:h")) != -1) {
    switch (c) {
    case 'n':
      count = atoi(optarg);
      break;
    case 'h':
      show_usage();
      exit(0);
      break;
    default:
      show_usage();
      exit(-1);
    }
  }
}

void cleanup() {
  if (d_src) {
    HIP_CHECK(hipFree(d_src));
  }
}

int main(int argc, char *argv[]) {
  // parse command arguments
  parse_args(argc, argv);

  if (count == 0) {
    count = 1;
  }

  std::cout << "Initialize HIP device" << std::endl;
  HIP_CHECK(hipSetDevice(0));

  // Query device properties
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));

  std::cout << "Device: " << props.name << std::endl;
  std::cout << "Max threads per block: " << props.maxThreadsPerBlock << std::endl;

  // Calculate buffer size
  uint32_t num_points = count * 256; // Simplified for HIP
  uint32_t buf_size = num_points * sizeof(char);

  std::cout << "Number of points: " << num_points << std::endl;
  std::cout << "Buffer size: " << buf_size << " bytes" << std::endl;

  // Allocate device memory
  std::cout << "Allocate device memory" << std::endl;
  HIP_CHECK(hipMalloc((void**)&d_src, buf_size));

  // Allocate and initialize host buffer
  std::cout << "Allocate host buffer" << std::endl;
  std::vector<char> h_src(num_points);

  // Generate input data
  for (uint32_t i = 0; i < num_points; ++i) {
    h_src[i] = (char)i;
  }

  // Copy data to device
  std::cout << "Copy data to device" << std::endl;
  HIP_CHECK(hipMemcpy(d_src, h_src.data(), buf_size, hipMemcpyHostToDevice));

  // Launch kernel
  std::cout << "Launch kernel" << std::endl;
  int blockSize = 64;
  int gridSize = (num_points + blockSize - 1) / blockSize;
  hipLaunchKernelGGL(kernel_body, dim3(gridSize), dim3(blockSize), 0, 0, d_src, num_points);

  // Wait for kernel to complete
  std::cout << "Wait for kernel completion" << std::endl;
  HIP_CHECK(hipDeviceSynchronize());

  // Cleanup
  std::cout << "Cleanup" << std::endl;
  cleanup();

  std::cout << "PASSED!" << std::endl;

  return 0;
}

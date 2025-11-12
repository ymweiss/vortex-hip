#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <unistd.h>

#define NUM_ADDRS 16
#define HIP_CHECK(call)                                                     \
  do {                                                                      \
    hipError_t error = call;                                               \
    if (error != hipSuccess) {                                             \
      std::cerr << "HIP error: " << hipGetErrorString(error) << " at "     \
                << __FILE__ << ":" << __LINE__ << std::endl;               \
      cleanup();                                                            \
      exit(-1);                                                             \
    }                                                                       \
  } while (false)

///////////////////////////////////////////////////////////////////////////////

// Kernel structure
typedef struct {
  uint32_t num_points;
  uint64_t src_addr;
  uint64_t dst_addr;
} kernel_arg_t;

// Global variables
const char* kernel_file = "kernel.vxbin";
uint32_t count = 0;
static uint64_t io_base_addr = 0;

// Device memory pointers
uint32_t* d_usr_test_buffer = nullptr;
uint32_t* d_io_test_buffer = nullptr;
uint64_t* d_src_buffer = nullptr;
int32_t* d_dst_buffer = nullptr;
kernel_arg_t* d_args = nullptr;

kernel_arg_t kernel_arg = {};

///////////////////////////////////////////////////////////////////////////////
// HIP Kernel
///////////////////////////////////////////////////////////////////////////////

__global__ void io_addr_kernel(kernel_arg_t* args) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= args->num_points) {
    return;
  }

  uint64_t* src_ptr = (uint64_t*)args->src_addr;
  int32_t* dst_ptr = (int32_t*)args->dst_addr;

  // Read address from source buffer
  uint64_t addr = src_ptr[idx];

  // Dereference address to get data
  int32_t* addr_ptr = (int32_t*)addr;

  // Store dereferenced value in destination
  dst_ptr[idx] = *addr_ptr;
}

///////////////////////////////////////////////////////////////////////////////
// Host Functions
///////////////////////////////////////////////////////////////////////////////

static void show_usage() {
  std::cout << "HIP io_addr Test." << std::endl;
  std::cout << "Usage: [-n words] [-h: help]" << std::endl;
}

static void parse_args(int argc, char** argv) {
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
  if (d_usr_test_buffer) {
    HIP_CHECK(hipFree(d_usr_test_buffer));
  }
  if (d_io_test_buffer) {
    HIP_CHECK(hipFree(d_io_test_buffer));
  }
  if (d_src_buffer) {
    HIP_CHECK(hipFree(d_src_buffer));
  }
  if (d_dst_buffer) {
    HIP_CHECK(hipFree(d_dst_buffer));
  }
  if (d_args) {
    HIP_CHECK(hipFree(d_args));
  }
}

void gen_src_addrs(std::vector<uint64_t>& src_addrs, uint32_t size,
                   uint64_t usr_test_addr) {
  src_addrs.resize(size);
  uint32_t u = 0, k = 0;
  for (uint32_t i = 0; i < size; ++i) {
    if (0 == (i % 4)) {
      k = (i + u) % NUM_ADDRS;
      ++u;
    }
    uint32_t j = i % NUM_ADDRS;
    uint64_t a = ((j == k) ? usr_test_addr : io_base_addr) + j * sizeof(uint32_t);
    std::cout << std::dec << i << "," << k << ": value=0x" << std::hex << a
              << std::endl;
    src_addrs[i] = a;
  }
}

void gen_ref_data(std::vector<int32_t>& ref_data, uint32_t size) {
  ref_data.resize(size);
  for (uint32_t i = 0; i < size; ++i) {
    int32_t j = i % NUM_ADDRS;
    ref_data[i] = j * j;
  }
}

int main(int argc, char* argv[]) {
  // parse command arguments
  parse_args(argc, argv);

  if (count == 0) {
    count = 1;
  }

  std::srand(50);

  // Get HIP device info
  std::cout << "Initializing HIP device" << std::endl;
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cerr << "No HIP devices found!" << std::endl;
    return -1;
  }
  HIP_CHECK(hipSetDevice(0));

  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  std::cout << "Using device: " << props.name << std::endl;

  // Use device properties for thread count
  uint32_t num_threads_per_block = 256; // Standard HIP block size
  uint32_t num_points = count * 1024; // Equivalent to multi-core setup

  uint32_t addr_buf_size = NUM_ADDRS * sizeof(int32_t);
  uint32_t src_buf_size = num_points * sizeof(uint64_t);
  uint32_t dst_buf_size = num_points * sizeof(int32_t);

  std::cout << "number of points: " << std::dec << num_points << std::endl;
  std::cout << "addr buffer size: " << addr_buf_size << " bytes" << std::endl;
  std::cout << "src buffer size: " << src_buf_size << " bytes" << std::endl;
  std::cout << "dst buffer size: " << dst_buf_size << " bytes" << std::endl;

  kernel_arg.num_points = num_points;

  // Allocate device memory
  std::cout << "allocate device memory" << std::endl;
  HIP_CHECK(hipMalloc(&d_usr_test_buffer, addr_buf_size));
  HIP_CHECK(hipMalloc(&d_io_test_buffer, addr_buf_size));
  HIP_CHECK(hipMalloc(&d_src_buffer, src_buf_size));
  HIP_CHECK(hipMalloc(&d_dst_buffer, dst_buf_size));
  HIP_CHECK(hipMalloc(&d_args, sizeof(kernel_arg_t)));

  // Set kernel argument addresses to device pointers
  kernel_arg.src_addr = (uint64_t)d_src_buffer;
  kernel_arg.dst_addr = (uint64_t)d_dst_buffer;

  std::cout << "dev_src=0x" << std::hex << kernel_arg.src_addr << std::endl;
  std::cout << "dev_dst=0x" << std::hex << kernel_arg.dst_addr << std::endl;

  // Allocate host buffers
  std::cout << "allocate host buffers" << std::endl;
  std::vector<uint64_t> h_src;
  std::vector<uint32_t> h_addr(NUM_ADDRS);
  std::vector<int32_t> h_dst(num_points);

  // Generate source data
  uint64_t usr_test_addr = (uint64_t)d_usr_test_buffer;
  io_base_addr = (uint64_t)d_io_test_buffer;

  gen_src_addrs(h_src, num_points, usr_test_addr);
  for (uint32_t i = 0; i < NUM_ADDRS; ++i) {
    h_addr[i] = i * i;
  }

  // Upload address data to device
  std::cout << "upload address buffers" << std::endl;
  HIP_CHECK(hipMemcpy(d_usr_test_buffer, h_addr.data(), addr_buf_size,
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_io_test_buffer, h_addr.data(), addr_buf_size,
                      hipMemcpyHostToDevice));

  // Upload source buffer (addresses)
  std::cout << "upload source buffer" << std::endl;
  HIP_CHECK(hipMemcpy(d_src_buffer, h_src.data(), src_buf_size,
                      hipMemcpyHostToDevice));

  // Upload kernel arguments
  std::cout << "upload kernel arguments" << std::endl;
  HIP_CHECK(hipMemcpy(d_args, &kernel_arg, sizeof(kernel_arg_t),
                      hipMemcpyHostToDevice));

  // Launch kernel
  std::cout << "launch kernel" << std::endl;
  uint32_t num_blocks = (num_points + num_threads_per_block - 1) / num_threads_per_block;
  hipLaunchKernelGGL(io_addr_kernel, dim3(num_blocks), dim3(num_threads_per_block),
                     0, 0, d_args);
  HIP_CHECK(hipGetLastError());

  // Wait for kernel completion
  std::cout << "wait for completion" << std::endl;
  HIP_CHECK(hipDeviceSynchronize());

  // Download destination buffer
  std::cout << "download destination buffer" << std::endl;
  HIP_CHECK(hipMemcpy(h_dst.data(), d_dst_buffer, dst_buf_size,
                      hipMemcpyDeviceToHost));

  // Verify result
  std::cout << "verify result" << std::endl;
  int errors = 0;
  {
    std::vector<int32_t> h_ref;
    gen_ref_data(h_ref, num_points);

    for (uint32_t i = 0; i < num_points; ++i) {
      int ref = h_ref[i];
      int cur = h_dst[i];
      if (cur != ref) {
        std::cout << "error at result #" << std::dec << i << std::hex
                  << ": actual 0x" << cur << ", expected 0x" << ref
                  << std::endl;
        ++errors;
      }
    }
  }

  // Cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  std::cout << "PASSED!" << std::endl;

  return 0;
}

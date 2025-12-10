// Indirect Memory Access (IO_ADDR) - HIP version
// Converted from Vortex regression test: vortex/tests/regression/io_addr

#include <hip/hip_runtime.h>
#include <stdlib.h>

#define NUM_ADDRS 16

#define HIP_CHECK(call) \
  do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
      std::cerr << "HIP error in " << __FILE__ << ":" << __LINE__ << ": " \
                << hipGetErrorString(err) << std::endl; \
      exit(1); \
    } \
  } while (0)

///////////////////////////////////////////////////////////////////////////////

// HIP kernel - performs indirect memory access
// Each thread reads an address from src_ptr and dereferences it
__global__ void io_addr_kernel(uint64_t* src_ptr, uint32_t* dst_ptr, uint32_t num_points) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_points) {
    // Read the address from src_ptr
    int32_t* addr_ptr = (int32_t*)(src_ptr[idx]);
    // Dereference it and store result
    dst_ptr[idx] = *addr_ptr;
  }
}

///////////////////////////////////////////////////////////////////////////////

void gen_src_addrs(std::vector<uint64_t>& src_addrs, uint64_t addr_base, uint32_t size) {
  src_addrs.resize(size);
  uint32_t u = 0, k = 0;
  for (uint32_t i = 0; i < size; ++i) {
    if (0 == (i % 4)) {
      k = (i + u) % NUM_ADDRS;
      ++u;
    }
    uint32_t j = i % NUM_ADDRS;
    // All addresses point to the same buffer in HIP (no IO vs user memory distinction)
    uint64_t a = addr_base + j * sizeof(uint32_t);
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

///////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
  uint32_t count = 1;

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      count = atoi(argv[i + 1]);
      i++;
    }
  }

  std::srand(50);

  uint32_t num_points = count;
  uint32_t addr_buf_size = NUM_ADDRS * sizeof(int32_t);
  uint32_t src_buf_size = num_points * sizeof(uint64_t);
  uint32_t dst_buf_size = num_points * sizeof(int32_t);

  printf("number of points: %u\n", num_points);
  std::cout << "addr buffer size: " << addr_buf_size << " bytes" << std::endl;
  std::cout << "src buffer size: " << src_buf_size << " bytes" << std::endl;
  std::cout << "dst buffer size: " << dst_buf_size << " bytes" << std::endl;

  // Allocate host buffers
  printf("allocate host buffers\n");
  std::vector<uint64_t> h_src;
  std::vector<uint32_t> h_addr(NUM_ADDRS);
  std::vector<int32_t> h_dst(num_points);

  // Generate address data - each location contains i*i
  for (uint32_t i = 0; i < NUM_ADDRS; ++i) {
    h_addr[i] = i * i;
  }

  // Allocate device memory
  printf("allocate device memory\n");
  uint32_t *d_addr;
  uint64_t *d_src;
  uint32_t *d_dst;

  HIP_CHECK(hipMalloc(&d_addr, addr_buf_size));
  HIP_CHECK(hipMalloc(&d_src, src_buf_size));
  HIP_CHECK(hipMalloc(&d_dst, dst_buf_size));

  // Upload address buffer
  printf("upload address buffer\n");
  HIP_CHECK(hipMemcpy(d_addr, h_addr.data(), addr_buf_size, hipMemcpyHostToDevice));

  // Get the device address of d_addr to generate source addresses
  uint64_t d_addr_base = (uint64_t)d_addr;

  // Generate source addresses (pointers into d_addr)
  gen_src_addrs(h_src, d_addr_base, num_points);

  // Upload source buffer (contains addresses)
  printf("upload source buffer\n");
  HIP_CHECK(hipMemcpy(d_src, h_src.data(), src_buf_size, hipMemcpyHostToDevice));

  // Launch kernel
  printf("launch kernel\n");
  int threads_per_block = 64;
  int num_blocks = (num_points + threads_per_block - 1) / threads_per_block;

  hipLaunchKernelGGL(io_addr_kernel, dim3(num_blocks), dim3(threads_per_block), 0, 0,
                     d_src, d_dst, num_points);

  // Wait for completion
  printf("wait for completion\n");
  HIP_CHECK(hipDeviceSynchronize());

  // Download destination buffer
  printf("download destination buffer\n");
  HIP_CHECK(hipMemcpy(h_dst.data(), d_dst, dst_buf_size, hipMemcpyDeviceToHost));

  // Verify result
  printf("verify result\n");
  int errors = 0;
  std::vector<int32_t> h_ref;
  gen_ref_data(h_ref, num_points);

  for (uint32_t i = 0; i < num_points; ++i) {
    int ref = h_ref[i];
    int cur = h_dst[i];
    if (cur != ref) {
      if (errors < 100) {
        std::cout << "*** error: [" << i << "] expected=" << ref
                  << ", actual=" << cur << std::endl;
      }
      ++errors;
    }
  }

  // Cleanup
  printf("cleanup\n");
  HIP_CHECK(hipFree(d_addr));
  HIP_CHECK(hipFree(d_src));
  HIP_CHECK(hipFree(d_dst));

  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    printf("FAILED!\n");
    return 1;
  }

  printf("PASSED!\n");
  return 0;
}

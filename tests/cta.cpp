#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <unistd.h>

// HIP error checking macro
#define HIP_CHECK(call)                                             \
  do {                                                              \
    hipError_t err = call;                                          \
    if (hipSuccess != err) {                                        \
      printf("Error: HIP call '%s' failed with code %d\n", #call, (int)err); \
      cleanup();                                                    \
      exit(-1);                                                    \
    }                                                               \
  } while (false)

// Kernel argument structure
typedef struct {
  uint32_t block_dim[3];
  uint32_t grid_dim[3];
  int* src_ptr;
  int* dst_ptr;
} kernel_arg_t;

// HIP kernel
__global__ void kernel_body(int* src_ptr, int* dst_ptr, uint32_t cta_size) {
  uint32_t blockId = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
  uint32_t localId = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
  uint32_t globalId = localId + blockId * cta_size;
  dst_ptr[globalId] = globalId + src_ptr[localId];
}

///////////////////////////////////////////////////////////////////////////////

template <typename Type>
class Comparator {};

template <>
class Comparator<int> {
public:
  static const char* type_str() {
    return "integer";
  }
  static int generate() {
    return rand();
  }
  static bool compare(int a, int b, int index, int errors) {
    if (a != b) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%d, actual=%d\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

// Global variables
uint32_t grd_x = 4;
uint32_t grd_y = 4;
uint32_t grd_z = 1;
uint32_t blk_x = 1;
uint32_t blk_y = 1;
uint32_t blk_z = 1;

int* d_src = nullptr;
int* d_dst = nullptr;

static void show_usage() {
  std::cout << "HIP CTA Test." << std::endl;
  std::cout << "Usage: [-x grid.x] [-y grid.y] [-z grid.z] [-a block.x] [-b block.y] [-c block.z] [-h: help]" << std::endl;
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "x:y:z:a:b:c:h")) != -1) {
    switch (c) {
    case 'a':
      blk_x = atoi(optarg);
      break;
    case 'b':
      blk_y = atoi(optarg);
      break;
    case 'c':
      blk_z = atoi(optarg);
      break;
    case 'x':
      grd_x = atoi(optarg);
      break;
    case 'y':
      grd_y = atoi(optarg);
      break;
    case 'z':
      grd_z = atoi(optarg);
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
    d_src = nullptr;
  }
  if (d_dst) {
    HIP_CHECK(hipFree(d_dst));
    d_dst = nullptr;
  }
}

int main(int argc, char* argv[]) {
  // Parse command arguments
  parse_args(argc, argv);

  std::srand(50);

  uint32_t cta_size = blk_x * blk_y * blk_z;
  uint32_t cta_count = grd_x * grd_y * grd_z;
  uint32_t total_threads = cta_count * cta_size;
  uint32_t src_buf_size = cta_size * sizeof(int);
  uint32_t dst_buf_size = total_threads * sizeof(int);

  std::cout << "CTA size: " << cta_size << std::endl;
  std::cout << "number of CTAs: " << cta_count << std::endl;
  std::cout << "number of threads: " << total_threads << std::endl;
  std::cout << "source buffer size: " << src_buf_size << " bytes" << std::endl;
  std::cout << "destination buffer size: " << dst_buf_size << " bytes" << std::endl;

  // Allocate device memory
  std::cout << "allocate device memory" << std::endl;
  HIP_CHECK(hipMalloc(&d_src, src_buf_size));
  HIP_CHECK(hipMalloc(&d_dst, dst_buf_size));

  // Allocate and initialize host buffers
  std::cout << "allocate host buffers" << std::endl;
  std::vector<int> h_src(cta_size);
  std::vector<int> h_dst(total_threads);

  for (uint32_t i = 0; i < cta_size; ++i) {
    h_src[i] = Comparator<int>::generate();
  }

  // Upload source buffer
  std::cout << "upload source buffer" << std::endl;
  HIP_CHECK(hipMemcpy(d_src, h_src.data(), src_buf_size, hipMemcpyHostToDevice));

  // Launch kernel
  std::cout << "launch kernel" << std::endl;
  dim3 grid(grd_x, grd_y, grd_z);
  dim3 block(blk_x, blk_y, blk_z);
  hipLaunchKernelGGL(kernel_body, grid, block, 0, 0, d_src, d_dst, cta_size);
  HIP_CHECK(hipGetLastError());

  // Wait for completion
  std::cout << "wait for completion" << std::endl;
  HIP_CHECK(hipDeviceSynchronize());

  // Download destination buffer
  std::cout << "download destination buffer" << std::endl;
  HIP_CHECK(hipMemcpy(h_dst.data(), d_dst, dst_buf_size, hipMemcpyDeviceToHost));

  // Verify result
  std::cout << "verify result" << std::endl;
  int errors = 0;
  for (uint32_t i = 0; i < total_threads; ++i) {
    auto ref = i + h_src[i % cta_size];
    auto cur = h_dst[i];
    if (!Comparator<int>::compare(cur, ref, i, errors)) {
      ++errors;
    }
  }

  // Cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  std::cout << "PASSED!" << std::endl;

  return 0;
}

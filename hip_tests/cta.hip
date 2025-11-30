// CTA (Cooperative Thread Array) Test - HIP version
// Converted from Vortex regression test: vortex/tests/regression/cta
//
// This test verifies that thread indexing works correctly across both blocks and threads.
// Each thread computes: globalId + src[localId] and writes to dst[globalId]

#include <hip/hip_runtime.h>
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

///////////////////////////////////////////////////////////////////////////////

class Comparator {
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

///////////////////////////////////////////////////////////////////////////////

// HIP kernel
// Each thread computes: globalId + src[localId]
// where globalId is the unique global thread ID across all blocks
// and localId is the thread ID within the current block
__global__ void cta_kernel(int* src_ptr, int* dst_ptr) {
  uint32_t cta_size = blockDim.x * blockDim.y * blockDim.z;
  uint32_t blockId  = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
  uint32_t localId  = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
  uint32_t globalId = localId + blockId * cta_size;
  dst_ptr[globalId] = globalId + src_ptr[localId];
}

///////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
  // Default grid and block dimensions
  uint32_t grd_x = 4;
  uint32_t grd_y = 4;
  uint32_t grd_z = 1;
  uint32_t blk_x = 1;
  uint32_t blk_y = 1;
  uint32_t blk_z = 1;

  // Parse command line arguments
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
    } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      std::cout << "CTA Test." << std::endl;
      std::cout << "Usage: [-x grid.x] [-y grid.y] [-z grid.z] [-a block.x] [-b block.y] [-c block.z] [-h help]" << std::endl;
      return 0;
    }
  }

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

  // Allocate host buffers
  std::cout << "allocate host buffers" << std::endl;
  std::vector<int> h_src(cta_size);
  std::vector<int> h_dst(total_threads);

  for (uint32_t i = 0; i < cta_size; ++i) {
    h_src[i] = Comparator::generate();
  }

  // Allocate device memory
  std::cout << "allocate device memory" << std::endl;
  int *d_src, *d_dst;
  HIP_CHECK(hipMalloc(&d_src, src_buf_size));
  HIP_CHECK(hipMalloc(&d_dst, dst_buf_size));

  // Upload source buffer
  std::cout << "upload source buffer" << std::endl;
  HIP_CHECK(hipMemcpy(d_src, h_src.data(), src_buf_size, hipMemcpyHostToDevice));

  // Launch kernel
  std::cout << "launch kernel" << std::endl;
  dim3 grid(grd_x, grd_y, grd_z);
  dim3 block(blk_x, blk_y, blk_z);

  hipLaunchKernelGGL(cta_kernel, grid, block, 0, 0, d_src, d_dst);

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
    if (!Comparator::compare(cur, ref, i, errors)) {
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

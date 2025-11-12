#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <cstring>

#define NONCE  0xdeadbeef

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

inline uint32_t shuffle(int i, uint32_t value) {
  return (value << i) | (value & ((1 << i)-1));
}

// HIP kernel - copy data from src to dst
__global__ void basic_kernel(int32_t* src, int32_t* dst, uint32_t count) {
  int idx = blockIdx.x * count + threadIdx.x;
  if (threadIdx.x < count) {
    dst[idx] = src[idx];
  }
}

///////////////////////////////////////////////////////////////////////////////

int run_memcopy_test(int32_t* d_dst, uint32_t num_points) {
  uint32_t buf_size = num_points * sizeof(int32_t);

  std::vector<uint32_t> h_src(num_points);
  std::vector<uint32_t> h_dst(num_points);

  // update source buffer
  for (uint32_t i = 0; i < num_points; ++i) {
    h_src[i] = shuffle(i, NONCE);
  }

  auto time_start = std::chrono::high_resolution_clock::now();

  // upload source buffer
  std::cout << "write source buffer to local memory" << std::endl;
  auto t0 = std::chrono::high_resolution_clock::now();
  HIP_CHECK(hipMemcpy(d_dst, h_src.data(), buf_size, hipMemcpyHostToDevice));
  auto t1 = std::chrono::high_resolution_clock::now();

  // download destination buffer
  std::cout << "read destination buffer from local memory" << std::endl;
  auto t2 = std::chrono::high_resolution_clock::now();
  HIP_CHECK(hipMemcpy(h_dst.data(), d_dst, buf_size, hipMemcpyDeviceToHost));
  auto t3 = std::chrono::high_resolution_clock::now();

  // verify result
  int errors = 0;
  std::cout << "verify result" << std::endl;
  for (uint32_t i = 0; i < num_points; ++i) {
    auto cur = h_dst[i];
    auto ref = shuffle(i, NONCE);
    if (cur != ref) {
      printf("*** error: [%d] expected=%d, actual=%d\n", i, ref, cur);
      ++errors;
    }
  }

  auto time_end = std::chrono::high_resolution_clock::now();

  double elapsed;
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  printf("upload time: %lg ms\n", elapsed);
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
  printf("download time: %lg ms\n", elapsed);
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Total elapsed time: %lg ms\n", elapsed);

  return errors;
}

int run_kernel_test(int32_t* d_src, int32_t* d_dst, uint32_t count, uint32_t num_blocks) {
  uint32_t num_points = count * num_blocks;
  uint32_t buf_size = num_points * sizeof(int32_t);

  std::vector<uint32_t> h_src(num_points);
  std::vector<uint32_t> h_dst(num_points);

  // update source buffer
  for (uint32_t i = 0; i < num_points; ++i) {
    h_src[i] = shuffle(i, NONCE);
  }

  auto time_start = std::chrono::high_resolution_clock::now();

  // upload source buffer
  auto t0 = std::chrono::high_resolution_clock::now();
  HIP_CHECK(hipMemcpy(d_src, h_src.data(), buf_size, hipMemcpyHostToDevice));
  auto t1 = std::chrono::high_resolution_clock::now();

  // start device
  std::cout << "start execution" << std::endl;
  auto t2 = std::chrono::high_resolution_clock::now();
  hipLaunchKernelGGL(basic_kernel, dim3(num_blocks), dim3(count), 0, 0,
                     d_src, d_dst, count);
  HIP_CHECK(hipDeviceSynchronize());
  auto t3 = std::chrono::high_resolution_clock::now();

  // download destination buffer
  std::cout << "read destination buffer from local memory" << std::endl;
  auto t4 = std::chrono::high_resolution_clock::now();
  HIP_CHECK(hipMemcpy(h_dst.data(), d_dst, buf_size, hipMemcpyDeviceToHost));
  auto t5 = std::chrono::high_resolution_clock::now();

  // verify result
  int errors = 0;
  std::cout << "verify result" << std::endl;
  for (uint32_t i = 0; i < num_points; ++i) {
    auto cur = h_dst[i];
    auto ref = shuffle(i, NONCE);
    if (cur != ref) {
      printf("*** error: [%d] expected=%d, actual=%d\n", i, ref, cur);
      ++errors;
    }
  }

  auto time_end = std::chrono::high_resolution_clock::now();

  double elapsed;
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  printf("upload time: %lg ms\n", elapsed);
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
  printf("execute time: %lg ms\n", elapsed);
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count();
  printf("download time: %lg ms\n", elapsed);
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Total elapsed time: %lg ms\n", elapsed);

  return errors;
}

///////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
  int test = -1;
  uint32_t count = 0;

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      count = atoi(argv[i + 1]);
      i++;
    } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
      test = atoi(argv[i + 1]);
      i++;
    }
  }

  if (count == 0) {
    count = 1;
  }

  // Get device properties to determine number of compute units (similar to num_cores)
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  uint32_t num_blocks = prop.multiProcessorCount;

  uint32_t num_points = count * num_blocks;
  uint32_t buf_size = num_points * sizeof(int32_t);

  std::cout << "number of points: " << num_points << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  // Allocate device memory
  std::cout << "allocate device memory" << std::endl;
  int32_t *d_src, *d_dst;
  HIP_CHECK(hipMalloc(&d_src, buf_size));
  HIP_CHECK(hipMalloc(&d_dst, buf_size));

  int errors = 0;

  // run tests
  if (0 == test || -1 == test) {
    std::cout << "run memcopy test" << std::endl;
    errors = run_memcopy_test(d_dst, num_points);
  }

  if (1 == test || -1 == test) {
    std::cout << "run kernel test" << std::endl;
    errors = run_kernel_test(d_src, d_dst, count, num_blocks);
  }

  // cleanup
  std::cout << "cleanup" << std::endl;
  HIP_CHECK(hipFree(d_src));
  HIP_CHECK(hipFree(d_dst));

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  std::cout << "Test PASSED" << std::endl;
  return 0;
}

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>

#define HIP_CHECK(call)                                                 \
  do {                                                                  \
    hipError_t err = call;                                             \
    if (hipSuccess != err) {                                           \
      fprintf(stderr, "HIP error: %s (code: %d)\n",                   \
              hipGetErrorString(err), err);                            \
      exit(-1);                                                        \
    }                                                                  \
  } while (false)

#ifndef TYPE
#define TYPE int
#endif

// Kernel argument structure
typedef struct {
  uint32_t num_points;
  TYPE* src_ptr;
  TYPE* dst_ptr;
} kernel_arg_t;

// HIP Kernel - Bitonic sort kernel
// Each thread computes the final position for one element
__global__ void kernel_body(const kernel_arg_t* arg) {
  uint32_t num_points = arg->num_points;
  const TYPE* src_ptr = arg->src_ptr;
  TYPE* dst_ptr = arg->dst_ptr;

  uint32_t blockIdx_x = blockIdx.x;

  if (blockIdx_x >= num_points) {
    return;
  }

  auto ref_value = src_ptr[blockIdx_x];

  uint32_t pos = 0;
  for (uint32_t i = 0; i < num_points; ++i) {
    auto cur_value = src_ptr[i];
    pos += (cur_value < ref_value) || ((cur_value == ref_value) && (i < blockIdx_x));
  }
  dst_ptr[pos] = ref_value;
}

///////////////////////////////////////////////////////////////////////////////

static void show_usage(const char* program) {
  std::cout << "HIP Sort Test." << std::endl;
  std::cout << "Usage: " << program << " [-n count] [-h]" << std::endl;
}

void gen_src_data(std::vector<TYPE>& src_data, uint32_t size) {
  src_data.resize(size);
  for (uint32_t i = 0; i < size; ++i) {
    auto r = static_cast<float>(std::rand()) / RAND_MAX;
    auto value = static_cast<TYPE>(r * size);
    src_data[i] = value;
  }
}

void gen_ref_data(std::vector<TYPE>& ref_data, const std::vector<TYPE>& src_data, uint32_t size) {
  ref_data.resize(size);
  for (uint32_t i = 0; i < size; ++i) {
    TYPE ref_value = src_data.at(i);
    uint32_t pos = 0;
    for (uint32_t j = 0; j < size; ++j) {
      TYPE cur_value = src_data.at(j);
      pos += (cur_value < ref_value) || (cur_value == ref_value && j < i);
    }
    ref_data.at(pos) = ref_value;
  }
}

int main(int argc, char* argv[]) {
  uint32_t count = 1;

  // Parse arguments
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "-n" && i + 1 < argc) {
      count = std::atoi(argv[i + 1]);
      ++i;
    } else if (std::string(argv[i]) == "-h") {
      show_usage(argv[0]);
      return 0;
    }
  }

  if (count == 0) {
    count = 1;
  }

  std::srand(50);

  // Get device info
  std::cout << "Getting HIP device info" << std::endl;
  int device = 0;
  HIP_CHECK(hipSetDevice(device));

  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, device));

  uint32_t num_points = count * props.maxThreadsPerBlock;
  uint32_t buf_size = num_points * sizeof(TYPE);

  std::cout << "Device: " << props.name << std::endl;
  std::cout << "Number of points: " << num_points << std::endl;
  std::cout << "Buffer size: " << buf_size << " bytes" << std::endl;

  // Allocate host memory
  std::cout << "Allocating host memory" << std::endl;
  std::vector<TYPE> h_src;
  std::vector<TYPE> h_dst(num_points);
  gen_src_data(h_src, num_points);

  // Allocate device memory
  std::cout << "Allocating device memory" << std::endl;
  TYPE* d_src = nullptr;
  TYPE* d_dst = nullptr;
  HIP_CHECK(hipMalloc((void**)&d_src, buf_size));
  HIP_CHECK(hipMalloc((void**)&d_dst, buf_size));

  // Copy input to device
  std::cout << "Copying input to device" << std::endl;
  HIP_CHECK(hipMemcpy(d_src, h_src.data(), buf_size, hipMemcpyHostToDevice));

  // Prepare kernel arguments
  kernel_arg_t kernel_arg;
  kernel_arg.num_points = num_points;
  kernel_arg.src_ptr = d_src;
  kernel_arg.dst_ptr = d_dst;

  // Allocate device memory for kernel arguments
  kernel_arg_t* d_arg = nullptr;
  HIP_CHECK(hipMalloc((void**)&d_arg, sizeof(kernel_arg_t)));
  HIP_CHECK(hipMemcpy(d_arg, &kernel_arg, sizeof(kernel_arg_t), hipMemcpyHostToDevice));

  // Launch kernel
  std::cout << "Launching kernel" << std::endl;
  hipLaunchKernelGGL(kernel_body, dim3(num_points), dim3(1), 0, 0, d_arg);
  HIP_CHECK(hipGetLastError());

  // Wait for kernel completion
  std::cout << "Waiting for kernel completion" << std::endl;
  HIP_CHECK(hipDeviceSynchronize());

  // Copy result back to host
  std::cout << "Copying result to host" << std::endl;
  HIP_CHECK(hipMemcpy(h_dst.data(), d_dst, buf_size, hipMemcpyDeviceToHost));

  // Verify result
  std::cout << "Verifying result" << std::endl;
  int errors = 0;
  {
    std::vector<TYPE> h_ref;
    gen_ref_data(h_ref, h_src, num_points);

    for (uint32_t i = 0; i < num_points; ++i) {
      TYPE ref = h_ref[i];
      TYPE cur = h_dst[i];
      if (cur != ref) {
        std::cout << "Error at result #" << std::dec << i
                  << std::hex << ": actual=" << cur << ", expected=" << ref << std::endl;
        ++errors;
        if (errors > 10) {  // Limit error output
          std::cout << "Additional errors omitted..." << std::endl;
          break;
        }
      }
    }
  }

  // Cleanup
  std::cout << "Cleaning up" << std::endl;
  HIP_CHECK(hipFree(d_src));
  HIP_CHECK(hipFree(d_dst));
  HIP_CHECK(hipFree(d_arg));

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  std::cout << "PASSED!" << std::endl;

  return 0;
}

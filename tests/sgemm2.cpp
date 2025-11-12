#include <iostream>
#include <vector>
#include <cmath>
#include <hip/hip_runtime.h>

#define HIP_CHECK(cmd) { \
  hipError_t error = cmd; \
  if (error != hipSuccess) { \
    std::cerr << "Error: HIP API call failed with error '" << hipGetErrorString(error) << "'" << std::endl; \
    exit(-1); \
  } \
}

#define FLOAT_ULP 6

// Kernel argument structure
typedef struct {
  uint32_t grid_dim[2];
  uint32_t block_dim[2];
  uint32_t size;
  uint32_t tile_size;
  float* A_ptr;
  float* B_ptr;
  float* C_ptr;
} kernel_arg_t;

// HIP kernel for SGEMM with tiling
__global__ void sgemm2_kernel(kernel_arg_t arg) {
  // Setup buffer arguments
  float* A_ptr = arg.A_ptr;
  float* B_ptr = arg.B_ptr;
  float* C_ptr = arg.C_ptr;

  uint32_t size = arg.size;
  uint32_t tile_size = arg.tile_size;

  // Allocate shared memory for tiles
  extern __shared__ char shared_mem[];
  float* local_A = (float*)shared_mem;
  float* local_B = (float*)shared_mem + blockDim.x * blockDim.y;

  // Determine global row and column indices
  uint32_t g_row = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t g_col = blockIdx.y * blockDim.y + threadIdx.y;

  // Determine local row and column indices
  uint32_t l_row = threadIdx.x;
  uint32_t l_col = threadIdx.y;

  float sum = 0.0f;

  // Loop over tiles
  for (uint32_t k = 0; k < size; k += tile_size) {
    // Load tile of matrix A & B to shared memory
    local_A[l_row * tile_size + l_col] = A_ptr[g_row * size + (k + l_col)];
    local_B[l_row * tile_size + l_col] = B_ptr[(k + l_row) * size + g_col];

    // Synchronize all threads in current block
    __syncthreads();

    // Compute partial sum for the local tile
    for (uint32_t j = 0; j < tile_size; ++j) {
      sum += local_A[l_row * tile_size + j] * local_B[j * tile_size + l_col];
    }

    // Synchronize all threads in current block
    __syncthreads();
  }

  // Store the computed sum into the result matrix C
  C_ptr[g_row * size + g_col] = sum;
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
        std::cout << "*** error: [" << index << "] expected=" << a << ", actual=" << b << std::endl;
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<float> {
private:
  union Float_t { float f; int i; };
public:
  static const char* type_str() {
    return "float";
  }
  static float generate() {
    return static_cast<float>(rand()) / RAND_MAX;
  }
  static bool compare(float a, float b, int index, int errors) {
    union fi_t { float f; int32_t i; };
    fi_t fa, fb;
    fa.f = a;
    fb.f = b;
    auto d = std::abs(fa.i - fb.i);
    if (d > FLOAT_ULP) {
      if (errors < 100) {
        std::cout << "*** error: [" << index << "] expected=" << a << ", actual=" << b << std::endl;
      }
      return false;
    }
    return true;
  }
};

static void matmul_cpu(float* out, const float* A, const float* B, uint32_t width, uint32_t height) {
  for (uint32_t row = 0; row < height; ++row) {
    for (uint32_t col = 0; col < width; ++col) {
      float sum = 0.0f;
      for (uint32_t e = 0; e < width; ++e) {
        float a = A[row * width + e];
        float b = B[e * width + col];
        float c = a * b;
        sum += c;
      }
      out[row * width + col] = sum;
    }
  }
}

uint32_t size = 16;
uint32_t tile_size = 4;

static void show_usage() {
  std::cout << "HIP SGEMM2 Test." << std::endl;
  std::cout << "Usage: [-n matrix_size] [-t:tile_size] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:t:h")) != -1) {
    switch (c) {
    case 'n':
      size = atoi(optarg);
      break;
    case 't':
      tile_size = atoi(optarg);
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

int main(int argc, char *argv[]) {
  // parse command arguments
  parse_args(argc, argv);

  if ((size / tile_size) * tile_size != size) {
    std::cerr << "Error: matrix size " << size << " must be a multiple of tile size " << tile_size << std::endl;
    return -1;
  }

  std::srand(50);

  // Initialize HIP
  std::cout << "initialize HIP" << std::endl;
  HIP_CHECK(hipSetDevice(0));

  uint32_t size_sq = size * size;
  uint32_t buf_size = size_sq * sizeof(float);
  uint32_t group_size = tile_size * tile_size;
  uint32_t local_mem = 2 * group_size * sizeof(float);

  std::cout << "data type: " << Comparator<float>::type_str() << std::endl;
  std::cout << "matrix size: " << size << "x" << size << std::endl;
  std::cout << "tile size: " << tile_size << "x" << tile_size << std::endl;
  std::cout << "shared memory: " << local_mem << " bytes" << std::endl;

  // Allocate device memory
  std::cout << "allocate device memory" << std::endl;
  float* d_A = nullptr;
  float* d_B = nullptr;
  float* d_C = nullptr;
  HIP_CHECK(hipMalloc(&d_A, buf_size));
  HIP_CHECK(hipMalloc(&d_B, buf_size));
  HIP_CHECK(hipMalloc(&d_C, buf_size));

  // Allocate host buffers
  std::cout << "allocate host buffers" << std::endl;
  std::vector<float> h_A(size_sq);
  std::vector<float> h_B(size_sq);
  std::vector<float> h_C(size_sq);

  // Generate source data
  std::cout << "generate source data" << std::endl;
  for (uint32_t i = 0; i < size_sq; ++i) {
    h_A[i] = Comparator<float>::generate();
    h_B[i] = Comparator<float>::generate();
  }

  // Upload source buffers
  std::cout << "upload source buffers" << std::endl;
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), buf_size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_B, h_B.data(), buf_size, hipMemcpyHostToDevice));

  // Setup kernel arguments
  kernel_arg_t kernel_arg = {};
  kernel_arg.grid_dim[0] = size / tile_size;
  kernel_arg.grid_dim[1] = size / tile_size;
  kernel_arg.block_dim[0] = tile_size;
  kernel_arg.block_dim[1] = tile_size;
  kernel_arg.size = size;
  kernel_arg.tile_size = tile_size;
  kernel_arg.A_ptr = d_A;
  kernel_arg.B_ptr = d_B;
  kernel_arg.C_ptr = d_C;

  // Launch kernel
  std::cout << "launch kernel" << std::endl;
  dim3 grid(kernel_arg.grid_dim[0], kernel_arg.grid_dim[1]);
  dim3 block(kernel_arg.block_dim[0], kernel_arg.block_dim[1]);
  hipLaunchKernelGGL(sgemm2_kernel, grid, block, local_mem, 0, kernel_arg);
  HIP_CHECK(hipGetLastError());

  // Wait for kernel completion
  std::cout << "wait for completion" << std::endl;
  HIP_CHECK(hipDeviceSynchronize());

  // Download result buffer
  std::cout << "download destination buffer" << std::endl;
  HIP_CHECK(hipMemcpy(h_C.data(), d_C, buf_size, hipMemcpyDeviceToHost));

  // Verify result
  std::cout << "verify result" << std::endl;
  int errors = 0;
  {
    std::vector<float> h_ref(size_sq);
    matmul_cpu(h_ref.data(), h_A.data(), h_B.data(), size, size);

    for (uint32_t i = 0; i < h_ref.size(); ++i) {
      if (!Comparator<float>::compare(h_C[i], h_ref[i], i, errors)) {
        ++errors;
      }
    }
  }

  // Cleanup
  std::cout << "cleanup" << std::endl;
  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_B));
  HIP_CHECK(hipFree(d_C));

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  std::cout << "PASSED!" << std::endl;

  return 0;
}

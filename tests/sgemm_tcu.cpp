#include <hip/hip_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string.h>
#include <vector>
#include <cstdlib>
#include <cstdint>

// Configuration constants
#ifndef NUM_THREADS
#define NUM_THREADS 4
#endif

#ifndef ITYPE_SIZE
#define ITYPE_SIZE 2  // 16-bit (fp16)
#endif

#ifndef OTYPE_SIZE
#define OTYPE_SIZE 4  // 32-bit (fp32)
#endif

#define FLOAT_ULP 6
#define MAX_ERRORS 100

// Tile dimensions for TCU-like behavior
#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

// Helper macro for error checking
#define HIP_CHECK(expr)                                             \
  do {                                                              \
    hipError_t _ret = expr;                                         \
    if (hipSuccess != _ret) {                                       \
      printf("HIP Error: %s returned %d!\n", #expr, (int)_ret);     \
      exit(-1);                                                     \
    }                                                               \
  } while (false)

typedef struct {
  uint32_t grid_dim[2];
  uint32_t block_dim[2];
  uint32_t M, N, K;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;
} kernel_arg_t;

// Kernel argument structure
__constant__ kernel_arg_t kernel_arg;

static bool g_enable_sparse = false;

///////////////////////////////////////////////////////////////////////////////
// Data conversion utilities
///////////////////////////////////////////////////////////////////////////////

static void convert_row_to_col_major_4bit(uint8_t *dst, uint32_t width, uint32_t height, const uint8_t *src) {
  uint32_t out_bytes = (width * height + 1) / 2;
  memset(dst, 0, out_bytes);
  uint32_t dst_stride = (height + 1) / 2;

  for (uint32_t c = 0; c < width; ++c) {
    uint32_t base = c * dst_stride;
    for (uint32_t r = 0; r < height; r += 2) {
      uint32_t idx_even = r * width + c;
      uint32_t idx_odd = (r + 1) * width + c;

      uint8_t b_even = src[idx_even / 2];
      uint8_t b_odd = (r + 1 < height) ? src[idx_odd / 2] : 0;

      uint8_t nib_even = (idx_even & 1) ? (b_even >> 4) : (b_even & 0x0F);
      uint8_t nib_odd = (r + 1 < height)
                            ? ((idx_odd & 1) ? (b_odd >> 4) : (b_odd & 0x0F))
                            : 0;

      dst[base + r / 2] = (nib_odd << 4) | nib_even;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// Data accessor templates
///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct data_accessor_t {
  static T read(const T *ptr, uint32_t offset) {
    return ptr[offset];
  }
  static void write(T *ptr, uint32_t offset, T value) {
    ptr[offset] = value;
  }
};

template <>
struct data_accessor_t<uint8_t> {
  static uint8_t read(const uint8_t *ptr, uint32_t offset) {
    uint32_t row_off = offset / 2;
    bool odd = offset & 0x1;
    uint8_t value8 = ptr[row_off];
    return odd ? (value8 >> 4) : (value8 & 0x0f);
  }
  static void write(uint8_t *ptr, uint32_t offset, uint32_t value) {
    uint32_t row_off = offset / 2;
    bool odd = offset & 0x1;
    uint8_t old_value = ptr[row_off];
    uint8_t new_value = odd ? ((old_value & 0x0f) | (value << 4))
                            : ((old_value & 0xf0) | (value & 0x0f));
    ptr[offset / 2] = new_value;
  }
};

///////////////////////////////////////////////////////////////////////////////
// Comparator templates for different data types
///////////////////////////////////////////////////////////////////////////////

template <typename Type>
class Comparator {};

template <>
class Comparator<int8_t> {
public:
  static int8_t generate() {
    return (int8_t)rand();
  }
  static bool compare(int8_t a, int8_t b, int index, int errors) {
    if (a != b) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=0x%x, actual=0x%x\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<uint8_t> {
public:
  static uint8_t generate() {
    return (uint8_t)rand();
  }
  static bool compare(uint8_t a, uint8_t b, int index, int errors) {
    if (a != b) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=0x%x, actual=0x%x\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<int32_t> {
public:
  static int32_t generate() {
    return (int32_t)rand();
  }
  static bool compare(int32_t a, int32_t b, int index, int errors) {
    if (a != b) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=0x%x, actual=0x%x\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<uint16_t> {
public:
  static uint16_t generate() {
    return (uint16_t)rand();
  }
  static bool compare(uint16_t a, uint16_t b, int index, int errors) {
    if (a != b) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=0x%x, actual=0x%x\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<float> {
public:
  static float generate() {
    return static_cast<float>(rand()) / RAND_MAX;
  }
  static bool compare(float a, float b, int index, int errors) {
    union fi_t {
      float f;
      int32_t i;
    };
    fi_t fa, fb;
    fa.f = a;
    fb.f = b;
    auto d = std::abs(fa.i - fb.i);
    if (d > FLOAT_ULP) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=%f, actual=%f\n", index, fb.f, fa.f);
      }
      return false;
    }
    return true;
  }
};

///////////////////////////////////////////////////////////////////////////////
// Multiply-add templates
///////////////////////////////////////////////////////////////////////////////

template <typename SType, typename DType>
struct muladd_t {
  static DType eval(SType a, SType b, DType c) {
    return static_cast<DType>(a) * static_cast<DType>(b) + c;
  }
};

template <>
struct muladd_t<uint16_t, float> {
  static float eval(uint16_t a, uint16_t b, float c) {
    // For fp16: simple conversion to float
    float fa = __half2float(__ushort_as_half(a));
    float fb = __half2float(__ushort_as_half(b));
    return fa * fb + c;
  }
};

///////////////////////////////////////////////////////////////////////////////
// CPU reference implementation
///////////////////////////////////////////////////////////////////////////////

template <typename IType, typename OType>
static void matmul_cpu(OType *C, const IType *A, const IType *B,
                       uint32_t M, uint32_t N, uint32_t K) {
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      OType sum(0);
      for (uint32_t k = 0; k < K; ++k) {
        auto a = data_accessor_t<IType>::read(A, m * K + k);
        auto b = data_accessor_t<IType>::read(B, k * N + n);
        sum = muladd_t<IType, OType>::eval(a, b, sum);
      }
      data_accessor_t<OType>::write(C, m * N + n, sum);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// Simple SGEMM kernel - performs basic matrix multiplication without TCU
// This is adapted from the vortex TCU version
///////////////////////////////////////////////////////////////////////////////

template <typename IType, typename OType>
__global__ void sgemm_kernel(IType *A, IType *B, OType *C,
                             uint32_t M, uint32_t N, uint32_t K) {
  uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    OType sum = 0;
    for (uint32_t k = 0; k < K; ++k) {
      IType a = A[row * K + k];
      IType b = B[k * N + col];
      sum = muladd_t<IType, OType>::eval(a, b, sum);
    }
    C[row * N + col] = sum;
  }
}

// Specialized kernel for fp16 input and fp32 output
__global__ void sgemm_kernel_fp16_fp32(__half *A, __half *B, float *C,
                                       uint32_t M, uint32_t N, uint32_t K) {
  uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float sum = 0.0f;
    for (uint32_t k = 0; k < K; ++k) {
      float a = __half2float(A[row * K + k]);
      float b = __half2float(B[k * N + col]);
      sum += a * b;
    }
    C[row * N + col] = sum;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Sparse matrix pruning
///////////////////////////////////////////////////////////////////////////////

struct SparseMat {
  std::vector<uint16_t> values;
  std::vector<uint8_t> meta;
  uint32_t rows, cols;
};

static SparseMat pruneAndCompressMatrixA(const std::vector<uint16_t>& denseA,
                                         uint32_t M, uint32_t K) {
  SparseMat out;
  out.rows = M;
  out.cols = K;
  out.values.reserve(M * K / 2);
  out.meta.reserve(M * K / 4);

  const uint16_t* src = denseA.data();

  for (uint32_t r = 0; r < M; ++r) {
    for (uint32_t c = 0; c < K; c += 4) {
      uint16_t blk[4] = {src[r * K + c],
                        src[r * K + c + 1],
                        src[r * K + c + 2],
                        src[r * K + c + 3]};

      uint32_t idx[4] = {0, 1, 2, 3};
      std::sort(idx, idx + 4,
        [&](uint32_t a, uint32_t b) {
          return std::abs((int)blk[a]) < std::abs((int)blk[b]);
        });

      uint8_t keep0 = idx[3];
      uint8_t keep1 = idx[2];

      out.values.push_back(blk[keep0]);
      out.values.push_back(blk[keep1]);

      uint8_t m = (1u << keep0) | (1u << keep1);
      out.meta.push_back(m);
    }
  }
  return out;
}

static void test_pruneA() {
  const uint32_t M = 4, K = 8;
  std::vector<uint16_t> denseA(M * K);
  for (auto& v : denseA) v = Comparator<uint16_t>::generate();

  auto spA = pruneAndCompressMatrixA(denseA, M, K);

  std::vector<uint16_t> recovered(M * K, 0);
  size_t v_idx = 0, m_idx = 0;
  for (uint32_t r = 0; r < M; ++r)
    for (uint32_t c = 0; c < K; c += 4) {
      uint8_t m = spA.meta[m_idx++];
      for (uint32_t i = 0; i < 4; ++i)
        if (m & (1u << i))
          recovered[r * K + c + i] = spA.values[v_idx++];
    }

  for (uint32_t i = 0; i < M * K; ++i)
    assert(recovered[i] == denseA[i] || recovered[i] == 0);
  std::cout << "pruneAndCompressMatrixA passed\n";
}

///////////////////////////////////////////////////////////////////////////////
// Host code
///////////////////////////////////////////////////////////////////////////////

static void show_usage() {
  std::cout << "HIP SGEMM Test (adapted from Vortex TCU)." << std::endl;
  std::cout << "Usage: [-m M] [-n N] [-k K] [-s] [-h]" << std::endl;
  std::cout << "  -m M  Matrix dimension M (default 32)" << std::endl;
  std::cout << "  -n N  Matrix dimension N (default 32)" << std::endl;
  std::cout << "  -k K  Matrix dimension K (default 32)" << std::endl;
  std::cout << "  -s    Enable 2:4 structured sparsity" << std::endl;
  std::cout << "  -h    Show this help message" << std::endl;
}

static void parse_args(int argc, char **argv, uint32_t &xm, uint32_t &xn, uint32_t &xk) {
  xm = 32;
  xn = 32;
  xk = 32;

  int c;
  while ((c = getopt(argc, argv, "m:n:k:hs")) != -1) {
    switch (c) {
    case 'm':
      xm = atoi(optarg);
      break;
    case 'n':
      xn = atoi(optarg);
      break;
    case 'k':
      xk = atoi(optarg);
      break;
    case 's':
      g_enable_sparse = true;
      std::cout << "Sparse mode enabled (-s)" << std::endl;
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
  uint32_t xm, xn, xk;
  parse_args(argc, argv, xm, xn, xk);

  if (g_enable_sparse) {
    test_pruneA();
  }

  std::srand(50);

  // Get device info
  std::cout << "Getting HIP device info" << std::endl;
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cout << "No HIP devices found!" << std::endl;
    return -1;
  }
  HIP_CHECK(hipSetDevice(0));

  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  std::cout << "Device: " << props.name << std::endl;
  std::cout << "Max threads per block: " << props.maxThreadsPerBlock << std::endl;

  uint32_t M = xm;
  uint32_t N = xn;
  uint32_t K = xk;

  // Validate dimensions
  if ((M % TILE_M) != 0) {
    std::cout << "Error: M must be a multiple of " << TILE_M << std::endl;
    return -1;
  }
  if ((N % TILE_N) != 0) {
    std::cout << "Error: N must be a multiple of " << TILE_N << std::endl;
    return -1;
  }
  if ((K % TILE_K) != 0) {
    std::cout << "Error: K must be a multiple of " << TILE_K << std::endl;
    return -1;
  }

  size_t sizeA = M * K;
  size_t sizeB = K * N;
  size_t sizeC = M * N;

  std::cout << "Matrix dimensions:" << std::endl;
  std::cout << "  A: " << M << "x" << K << std::endl;
  std::cout << "  B: " << K << "x" << N << std::endl;
  std::cout << "  C: " << M << "x" << N << std::endl;

  // Allocate host memory
  std::cout << "Allocating host memory" << std::endl;
  std::vector<__half> h_A(sizeA);
  std::vector<__half> h_B(sizeB);
  std::vector<float> h_C(sizeC);
  std::vector<float> h_ref(sizeC);

  // Generate random data
  std::cout << "Generating random test data" << std::endl;
  for (uint32_t i = 0; i < sizeA; ++i) {
    h_A[i] = __float2half(Comparator<float>::generate());
  }
  for (uint32_t i = 0; i < sizeB; ++i) {
    h_B[i] = __float2half(Comparator<float>::generate());
  }

  // Allocate device memory
  std::cout << "Allocating device memory" << std::endl;
  __half *d_A, *d_B;
  float *d_C;
  HIP_CHECK(hipMalloc(&d_A, sizeA * sizeof(__half)));
  HIP_CHECK(hipMalloc(&d_B, sizeB * sizeof(__half)));
  HIP_CHECK(hipMalloc(&d_C, sizeC * sizeof(float)));

  // Copy data to device
  std::cout << "Copying data to device" << std::endl;
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), sizeA * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_B, h_B.data(), sizeB * sizeof(__half), hipMemcpyHostToDevice));

  // Setup kernel launch parameters
  dim3 block(16, 16);  // 16x16 threads per block
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

  std::cout << "Grid: " << grid.x << "x" << grid.y << std::endl;
  std::cout << "Block: " << block.x << "x" << block.y << std::endl;

  auto time_start = std::chrono::high_resolution_clock::now();

  // Launch kernel
  std::cout << "Launching SGEMM kernel" << std::endl;
  hipLaunchKernelGGL(sgemm_kernel_fp16_fp32, grid, block, 0, 0,
                     d_A, d_B, d_C, M, N, K);
  HIP_CHECK(hipGetLastError());

  // Wait for kernel completion
  HIP_CHECK(hipDeviceSynchronize());

  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Kernel execution time: %lg ms\n", elapsed);

  // Copy result back to host
  std::cout << "Copying result from device" << std::endl;
  HIP_CHECK(hipMemcpy(h_C.data(), d_C, sizeC * sizeof(float), hipMemcpyDeviceToHost));

  // CPU reference computation
  std::cout << "Computing CPU reference" << std::endl;
  matmul_cpu<__half, float>(h_ref.data(), h_A.data(), h_B.data(), M, N, K);

  // Verify results
  std::cout << "Verifying results" << std::endl;
  int errors = 0;
  for (uint32_t i = 0; i < h_ref.size(); ++i) {
    if (!Comparator<float>::compare(h_C[i], h_ref[i], i, errors)) {
      ++errors;
    }
  }

  // Cleanup
  std::cout << "Cleaning up" << std::endl;
  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_B));
  HIP_CHECK(hipFree(d_C));

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " / " << sizeC << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  std::cout << "PASSED!" << std::endl;

  return 0;
}

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <hip/hip_runtime.h>

// Data type definition
#ifndef TYPE
#define TYPE float
#endif

// Float comparison tolerance
#define FLOAT_ULP 6

// HIP error checking macro
#define HIP_CHECK(call) \
  do { \
    hipError_t err = call; \
    if (hipSuccess != err) { \
      fprintf(stderr, "HIP error: %s returned %s\n", #call, hipGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while (false)

///////////////////////////////////////////////////////////////////////////////
// Kernel argument structure
///////////////////////////////////////////////////////////////////////////////

typedef struct {
  uint32_t grid_dim[2];
  uint32_t width;
  TYPE*    I_ptr;
  TYPE*    W_ptr;
  TYPE*    O_ptr;
  bool     use_lmem;
} kernel_arg_t;

///////////////////////////////////////////////////////////////////////////////
// HIP Kernel
///////////////////////////////////////////////////////////////////////////////

__global__ void conv3_kernel(TYPE* I, TYPE* W, TYPE* O, uint32_t width, bool use_lmem) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col >= width || row >= width) {
    return;
  }

  // Adjust for padded borders
  int paddedWidth = width + 2;
  int paddedX = col + 1;
  int paddedY = row + 1;

  // Compute 3x3 convolution sum
  TYPE sum = 0.0f;

  sum += I[(paddedY - 1) * paddedWidth + (paddedX - 1)] * W[0]; // Top-left
  sum += I[(paddedY - 1) * paddedWidth + paddedX] * W[1];       // Top-center
  sum += I[(paddedY - 1) * paddedWidth + (paddedX + 1)] * W[2]; // Top-right

  sum += I[paddedY * paddedWidth + (paddedX - 1)] * W[3];       // Middle-left
  sum += I[paddedY * paddedWidth + paddedX] * W[4];             // Center
  sum += I[paddedY * paddedWidth + (paddedX + 1)] * W[5];       // Middle-right

  sum += I[(paddedY + 1) * paddedWidth + (paddedX - 1)] * W[6]; // Bottom-left
  sum += I[(paddedY + 1) * paddedWidth + paddedX] * W[7];       // Bottom-center
  sum += I[(paddedY + 1) * paddedWidth + (paddedX + 1)] * W[8]; // Bottom-right

  O[row * width + col] = sum;
}

///////////////////////////////////////////////////////////////////////////////
// Comparator Templates
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

template <>
class Comparator<float> {
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
        printf("*** error: [%d] expected=%f, actual=%f\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

///////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation
///////////////////////////////////////////////////////////////////////////////

static void convolution_cpu(TYPE *O, TYPE *I, TYPE *W, int32_t width, int32_t height) {
  int paddedWidth = width + 2;
  for (int32_t y = 0; y < height; ++y) {
    for (int32_t x = 0; x < width; ++x) {
      int paddedY = y + 1;
      int paddedX = x + 1;
      TYPE sum(0);
      for (int32_t ky = -1; ky <= 1; ++ky) {
        for (int32_t kx = -1; kx <= 1; ++kx) {
          int32_t iy = paddedY + ky;
          int32_t ix = paddedX + kx;
          TYPE value = I[iy * paddedWidth + ix];
          TYPE weight = W[(ky + 1) * 3 + (kx + 1)];
          sum += value * weight;
        }
      }
      O[y * width + x] = sum;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// Host Code
///////////////////////////////////////////////////////////////////////////////

void show_usage() {
  std::cout << "HIP Conv3 Test." << std::endl;
  std::cout << "Usage: [-l: local memory] [-n size] [-h|?: help]" << std::endl;
}

int main(int argc, char *argv[]) {
  int size = 32;
  bool use_lmem = false;

  // Parse command arguments
  {
    int c;
    int opt_index = 1;
    while (opt_index < argc) {
      if (argv[opt_index][0] == '-') {
        c = argv[opt_index][1];
        switch (c) {
        case 'n':
          if (opt_index + 1 < argc) {
            size = atoi(argv[opt_index + 1]);
            opt_index += 2;
          } else {
            opt_index++;
          }
          break;
        case 'l':
          use_lmem = true;
          opt_index++;
          break;
        case 'h':
        case '?':
          show_usage();
          return 0;
        default:
          show_usage();
          return -1;
        }
      } else {
        opt_index++;
      }
    }
  }

  std::srand(50);

  std::cout << "Initialize HIP device" << std::endl;
  HIP_CHECK(hipSetDevice(0));

  std::cout << "data type: " << Comparator<TYPE>::type_str() << std::endl;
  std::cout << "matrix size: " << size << "x" << size << std::endl;

  uint32_t o_points = size * size;
  uint32_t i_points = (size + 2) * (size + 2);
  uint32_t w_points = 3 * 3;

  // Allocate device memory
  std::cout << "allocate device memory" << std::endl;
  size_t i_nbytes = i_points * sizeof(TYPE);
  size_t w_nbytes = w_points * sizeof(TYPE);
  size_t o_nbytes = o_points * sizeof(TYPE);

  TYPE* d_I = nullptr;
  TYPE* d_W = nullptr;
  TYPE* d_O = nullptr;

  HIP_CHECK(hipMalloc(&d_I, i_nbytes));
  HIP_CHECK(hipMalloc(&d_W, w_nbytes));
  HIP_CHECK(hipMalloc(&d_O, o_nbytes));

  std::cout << "dev_I=0x" << std::hex << (uintptr_t)d_I << std::endl;
  std::cout << "dev_W=0x" << std::hex << (uintptr_t)d_W << std::endl;
  std::cout << "dev_O=0x" << std::hex << (uintptr_t)d_O << std::endl;

  // Generate input values
  std::vector<TYPE> h_I(i_points);
  std::vector<TYPE> h_W(w_points);
  std::vector<TYPE> h_O(o_points);

  for (int32_t y = -1; y < size + 1; ++y) {
    for (int32_t x = -1; x < size + 1; ++x) {
      if (x >= 0 && x < size && y >= 0 && y < size) {
        h_I[(y + 1) * (size + 2) + (x + 1)] = static_cast<TYPE>(rand()) / RAND_MAX;
      } else {
        h_I[(y + 1) * (size + 2) + (x + 1)] = 0;
      }
    }
  }

  for (uint32_t i = 0; i < w_points; ++i) {
    h_W[i] = static_cast<TYPE>(rand()) / RAND_MAX;
  }

  // Upload input buffer
  {
    std::cout << "upload source buffer" << std::endl;
    HIP_CHECK(hipMemcpy(d_I, h_I.data(), i_nbytes, hipMemcpyHostToDevice));
  }

  // Upload weight buffer
  {
    std::cout << "upload weight buffer" << std::endl;
    HIP_CHECK(hipMemcpy(d_W, h_W.data(), w_nbytes, hipMemcpyHostToDevice));
  }

  auto time_start = std::chrono::high_resolution_clock::now();

  // Launch kernel
  std::cout << "launch kernel" << std::endl;
  int blockSize = 16;
  int gridSize = (size + blockSize - 1) / blockSize;
  hipLaunchKernelGGL(conv3_kernel, dim3(gridSize, gridSize), dim3(blockSize, blockSize),
                     0, 0, d_I, d_W, d_O, size, use_lmem);
  HIP_CHECK(hipGetLastError());

  // Synchronize
  std::cout << "wait for completion" << std::endl;
  HIP_CHECK(hipDeviceSynchronize());

  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  // Download destination buffer
  std::cout << "download destination buffer" << std::endl;
  HIP_CHECK(hipMemcpy(h_O.data(), d_O, o_nbytes, hipMemcpyDeviceToHost));

  // Verify result
  std::cout << "verify result" << std::endl;
  int errors = 0;
  {
    std::vector<TYPE> h_ref(o_points);
    convolution_cpu(h_ref.data(), h_I.data(), h_W.data(), size, size);

    for (uint32_t i = 0; i < h_ref.size(); ++i) {
      auto ref = h_ref[i];
      auto cur = h_O[i];
      if (!Comparator<TYPE>::compare(cur, ref, i, errors)) {
        ++errors;
      }
    }
  }

  // Cleanup
  std::cout << "cleanup" << std::endl;
  HIP_CHECK(hipFree(d_I));
  HIP_CHECK(hipFree(d_W));
  HIP_CHECK(hipFree(d_O));

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  std::cout << "PASSED!" << std::endl;

  return 0;
}

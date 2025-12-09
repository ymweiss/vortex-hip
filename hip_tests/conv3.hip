// 3D Convolution - HIP version
// Converted from Vortex regression test: vortex/tests/regression/conv3

#include <hip/hip_runtime.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef TYPE
#define TYPE float
#endif

#define FLOAT_ULP 6

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

// CPU reference implementation
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

// HIP kernel for 3x3 convolution
__global__ void conv3_kernel(TYPE* I, TYPE* W, TYPE* O, uint32_t width) {
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
    float sum = 0.0f;

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

int main(int argc, char *argv[]) {
  uint32_t size = 32;

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      size = atoi(argv[i + 1]);
      i++;
    }
  }

  std::srand(50);

  std::cout << "data type: " << Comparator<TYPE>::type_str() << std::endl;
  std::cout << "matrix size: " << size << "x" << size << std::endl;

  uint32_t o_points = size * size;
  uint32_t i_points = (size + 2) * (size + 2);
  uint32_t w_points = 3 * 3;

  size_t i_nbytes = i_points * sizeof(TYPE);
  size_t w_nbytes = w_points * sizeof(TYPE);
  size_t o_nbytes = o_points * sizeof(TYPE);

  std::cout << "input buffer size: " << i_nbytes << " bytes" << std::endl;
  std::cout << "weight buffer size: " << w_nbytes << " bytes" << std::endl;
  std::cout << "output buffer size: " << o_nbytes << " bytes" << std::endl;

  // Allocate host buffers
  printf("allocate host buffers\n");
  std::vector<TYPE> h_I(i_points);
  std::vector<TYPE> h_W(w_points);
  std::vector<TYPE> h_O(o_points);

  // Generate input values
  for (int32_t y = -1; y < (int32_t)size + 1; ++y) {
    for (int32_t x = -1; x < (int32_t)size + 1; ++x) {
      if (x >= 0 && x < (int32_t)size && y >= 0 && y < (int32_t)size) {
        h_I[(y + 1) * (size + 2) + (x + 1)] = static_cast<TYPE>(rand()) / RAND_MAX;
      } else {
        h_I[(y + 1) * (size + 2) + (x + 1)] = 0;
      }
    }
  }

  // Generate weight values
  for (uint32_t i = 0; i < w_points; ++i) {
    h_W[i] = static_cast<TYPE>(rand()) / RAND_MAX;
  }

  // Allocate device memory
  printf("allocate device memory\n");
  TYPE *d_I, *d_W, *d_O;
  HIP_CHECK(hipMalloc(&d_I, i_nbytes));
  HIP_CHECK(hipMalloc(&d_W, w_nbytes));
  HIP_CHECK(hipMalloc(&d_O, o_nbytes));

  // Upload input buffer
  printf("upload source buffer\n");
  HIP_CHECK(hipMemcpy(d_I, h_I.data(), i_nbytes, hipMemcpyHostToDevice));

  // Upload weight buffer
  printf("upload weight buffer\n");
  HIP_CHECK(hipMemcpy(d_W, h_W.data(), w_nbytes, hipMemcpyHostToDevice));

  // Launch kernel
  printf("launch kernel\n");
  dim3 threads_per_block(16, 16);
  dim3 num_blocks((size + threads_per_block.x - 1) / threads_per_block.x,
                  (size + threads_per_block.y - 1) / threads_per_block.y);

  hipLaunchKernelGGL(conv3_kernel, num_blocks, threads_per_block, 0, 0,
                     d_I, d_W, d_O, size);

  // Wait for completion
  printf("wait for completion\n");
  HIP_CHECK(hipDeviceSynchronize());

  // Download destination buffer
  printf("download destination buffer\n");
  HIP_CHECK(hipMemcpy(h_O.data(), d_O, o_nbytes, hipMemcpyDeviceToHost));

  // Verify result
  printf("verify result\n");
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
  printf("cleanup\n");
  HIP_CHECK(hipFree(d_I));
  HIP_CHECK(hipFree(d_W));
  HIP_CHECK(hipFree(d_O));

  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    printf("FAILED!\n");
    return 1;
  }

  printf("PASSED!\n");
  return 0;
}

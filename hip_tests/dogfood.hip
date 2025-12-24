// Dogfood Test - HIP version (Simplified)
// Comprehensive arithmetic operations test
// Converted from Vortex regression test: vortex/tests/regression/dogfood
// Simplified to avoid templates and lambdas that cause Polygeist issues

#include <hip/hip_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>

#define FLOAT_ULP 6

#define HIP_CHECK(call) \
  do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
      printf("HIP error in %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
      exit(1); \
    } \
  } while (0)

///////////////////////////////////////////////////////////////////////////////
// HIP Kernels

// Integer addition
__global__ void kernel_iadd(int32_t* src0, int32_t* src1, int32_t* dst, uint32_t num_points) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_points) {
    dst[idx] = src0[idx] + src1[idx];
  }
}

// Integer multiplication
__global__ void kernel_imul(int32_t* src0, int32_t* src1, int32_t* dst, uint32_t num_points) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_points) {
    dst[idx] = src0[idx] * src1[idx];
  }
}

// Float addition
__global__ void kernel_fadd(float* src0, float* src1, float* dst, uint32_t num_points) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_points) {
    dst[idx] = src0[idx] + src1[idx];
  }
}

// Float multiplication
__global__ void kernel_fmul(float* src0, float* src1, float* dst, uint32_t num_points) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_points) {
    dst[idx] = src0[idx] * src1[idx];
  }
}

// Float division
__global__ void kernel_fdiv(float* src0, float* src1, float* dst, uint32_t num_points) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_points) {
    dst[idx] = src0[idx] / src1[idx];
  }
}

// Float square root
__global__ void kernel_fsqrt(float* src0, float* src1, float* dst, uint32_t num_points) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_points) {
    dst[idx] = sqrtf(src0[idx] * src1[idx]);
  }
}

// Float to int conversion
__global__ void kernel_ftoi(float* src0, float* src1, int32_t* dst, uint32_t num_points) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_points) {
    dst[idx] = (int32_t)(src0[idx] + src1[idx]);
  }
}

// Int to float conversion
__global__ void kernel_itof(int32_t* src0, int32_t* src1, float* dst, uint32_t num_points) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_points) {
    dst[idx] = (float)(src0[idx] + src1[idx]);
  }
}

///////////////////////////////////////////////////////////////////////////////
// Test helpers

static inline bool almost_equal(float a, float b) {
  if (a == b) return true;
  union { float f; int32_t i; } fa, fb;
  fa.f = a;
  fb.f = b;
  int32_t diff = fa.i - fb.i;
  if (diff < 0) diff = -diff;
  return diff <= FLOAT_ULP;
}

///////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
  uint32_t num_points = 1024;
  // Note: Vortex supports max 16 threads per block
  int threads_per_block = 16;

  // Parse command line
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      num_points = atoi(argv[i + 1]);
      i++;
    }
  }

  printf("Dogfood Arithmetic Operations Test\n");
  printf("Number of points: %u\n", num_points);

  int num_blocks = (num_points + threads_per_block - 1) / threads_per_block;
  size_t buf_size = num_points * sizeof(float);

  // Allocate host buffers
  int32_t* h_isrc0 = (int32_t*)malloc(buf_size);
  int32_t* h_isrc1 = (int32_t*)malloc(buf_size);
  int32_t* h_idst = (int32_t*)malloc(buf_size);
  float* h_fsrc0 = (float*)malloc(buf_size);
  float* h_fsrc1 = (float*)malloc(buf_size);
  float* h_fdst = (float*)malloc(buf_size);

  // Initialize test data
  for (uint32_t i = 0; i < num_points; ++i) {
    h_isrc0[i] = num_points / 2 - i;
    h_isrc1[i] = num_points / 2 + i;
    h_fsrc0[i] = (float)(num_points - i) / num_points;
    h_fsrc1[i] = (float)(num_points + i) / num_points;
  }

  // Allocate device memory
  int32_t *d_isrc0, *d_isrc1, *d_idst;
  float *d_fsrc0, *d_fsrc1, *d_fdst;
  HIP_CHECK(hipMalloc(&d_isrc0, buf_size));
  HIP_CHECK(hipMalloc(&d_isrc1, buf_size));
  HIP_CHECK(hipMalloc(&d_idst, buf_size));
  HIP_CHECK(hipMalloc(&d_fsrc0, buf_size));
  HIP_CHECK(hipMalloc(&d_fsrc1, buf_size));
  HIP_CHECK(hipMalloc(&d_fdst, buf_size));

  int total_errors = 0;
  int tests_passed = 0;
  int tests_run = 0;

  // Test 1: Integer addition
  printf("\n=== Test: iadd ===\n");
  tests_run++;
  HIP_CHECK(hipMemcpy(d_isrc0, h_isrc0, buf_size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_isrc1, h_isrc1, buf_size, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(kernel_iadd, dim3(num_blocks), dim3(threads_per_block), 0, 0,
                     d_isrc0, d_isrc1, d_idst, num_points);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(h_idst, d_idst, buf_size, hipMemcpyDeviceToHost));
  {
    int errors = 0;
    for (uint32_t i = 0; i < num_points; ++i) {
      int32_t expected = h_isrc0[i] + h_isrc1[i];
      if (h_idst[i] != expected) {
        if (errors < 10) printf("  error[%d]: expected=%d, actual=%d\n", i, expected, h_idst[i]);
        errors++;
      }
    }
    if (errors == 0) { printf("Test iadd: PASSED\n"); tests_passed++; }
    else { printf("Test iadd: FAILED (%d errors)\n", errors); total_errors += errors; }
  }

  // Test 2: Integer multiplication
  printf("\n=== Test: imul ===\n");
  tests_run++;
  hipLaunchKernelGGL(kernel_imul, dim3(num_blocks), dim3(threads_per_block), 0, 0,
                     d_isrc0, d_isrc1, d_idst, num_points);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(h_idst, d_idst, buf_size, hipMemcpyDeviceToHost));
  {
    int errors = 0;
    for (uint32_t i = 0; i < num_points; ++i) {
      int32_t expected = h_isrc0[i] * h_isrc1[i];
      if (h_idst[i] != expected) {
        if (errors < 10) printf("  error[%d]: expected=%d, actual=%d\n", i, expected, h_idst[i]);
        errors++;
      }
    }
    if (errors == 0) { printf("Test imul: PASSED\n"); tests_passed++; }
    else { printf("Test imul: FAILED (%d errors)\n", errors); total_errors += errors; }
  }

  // Test 3: Float addition
  printf("\n=== Test: fadd ===\n");
  tests_run++;
  HIP_CHECK(hipMemcpy(d_fsrc0, h_fsrc0, buf_size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_fsrc1, h_fsrc1, buf_size, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(kernel_fadd, dim3(num_blocks), dim3(threads_per_block), 0, 0,
                     d_fsrc0, d_fsrc1, d_fdst, num_points);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(h_fdst, d_fdst, buf_size, hipMemcpyDeviceToHost));
  {
    int errors = 0;
    for (uint32_t i = 0; i < num_points; ++i) {
      float expected = h_fsrc0[i] + h_fsrc1[i];
      if (!almost_equal(h_fdst[i], expected)) {
        if (errors < 10) printf("  error[%d]: expected=%f, actual=%f\n", i, (double)expected, (double)h_fdst[i]);
        errors++;
      }
    }
    if (errors == 0) { printf("Test fadd: PASSED\n"); tests_passed++; }
    else { printf("Test fadd: FAILED (%d errors)\n", errors); total_errors += errors; }
  }

  // Test 4: Float multiplication
  printf("\n=== Test: fmul ===\n");
  tests_run++;
  hipLaunchKernelGGL(kernel_fmul, dim3(num_blocks), dim3(threads_per_block), 0, 0,
                     d_fsrc0, d_fsrc1, d_fdst, num_points);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(h_fdst, d_fdst, buf_size, hipMemcpyDeviceToHost));
  {
    int errors = 0;
    for (uint32_t i = 0; i < num_points; ++i) {
      float expected = h_fsrc0[i] * h_fsrc1[i];
      if (!almost_equal(h_fdst[i], expected)) {
        if (errors < 10) printf("  error[%d]: expected=%f, actual=%f\n", i, (double)expected, (double)h_fdst[i]);
        errors++;
      }
    }
    if (errors == 0) { printf("Test fmul: PASSED\n"); tests_passed++; }
    else { printf("Test fmul: FAILED (%d errors)\n", errors); total_errors += errors; }
  }

  // Test 5: Float division
  printf("\n=== Test: fdiv ===\n");
  tests_run++;
  hipLaunchKernelGGL(kernel_fdiv, dim3(num_blocks), dim3(threads_per_block), 0, 0,
                     d_fsrc0, d_fsrc1, d_fdst, num_points);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(h_fdst, d_fdst, buf_size, hipMemcpyDeviceToHost));
  {
    int errors = 0;
    for (uint32_t i = 0; i < num_points; ++i) {
      float expected = h_fsrc0[i] / h_fsrc1[i];
      if (!almost_equal(h_fdst[i], expected)) {
        if (errors < 10) printf("  error[%d]: expected=%f, actual=%f\n", i, (double)expected, (double)h_fdst[i]);
        errors++;
      }
    }
    if (errors == 0) { printf("Test fdiv: PASSED\n"); tests_passed++; }
    else { printf("Test fdiv: FAILED (%d errors)\n", errors); total_errors += errors; }
  }

  // Test 6: Float square root
  printf("\n=== Test: fsqrt ===\n");
  tests_run++;
  // Use positive values for sqrt
  for (uint32_t i = 0; i < num_points; ++i) {
    h_fsrc0[i] = 1.0f + (i % 64);
    h_fsrc1[i] = 1.0f + (i % 64);
  }
  HIP_CHECK(hipMemcpy(d_fsrc0, h_fsrc0, buf_size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_fsrc1, h_fsrc1, buf_size, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(kernel_fsqrt, dim3(num_blocks), dim3(threads_per_block), 0, 0,
                     d_fsrc0, d_fsrc1, d_fdst, num_points);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(h_fdst, d_fdst, buf_size, hipMemcpyDeviceToHost));
  {
    int errors = 0;
    for (uint32_t i = 0; i < num_points; ++i) {
      float expected = sqrtf(h_fsrc0[i] * h_fsrc1[i]);
      if (!almost_equal(h_fdst[i], expected)) {
        if (errors < 10) printf("  error[%d]: expected=%f, actual=%f\n", i, (double)expected, (double)h_fdst[i]);
        errors++;
      }
    }
    if (errors == 0) { printf("Test fsqrt: PASSED\n"); tests_passed++; }
    else { printf("Test fsqrt: FAILED (%d errors)\n", errors); total_errors += errors; }
  }

  // Test 7: Float to int conversion
  printf("\n=== Test: ftoi ===\n");
  tests_run++;
  for (uint32_t i = 0; i < num_points; ++i) {
    h_fsrc0[i] = (float)(num_points / 2 - i);
    h_fsrc1[i] = (float)(num_points / 2 - i);
  }
  HIP_CHECK(hipMemcpy(d_fsrc0, h_fsrc0, buf_size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_fsrc1, h_fsrc1, buf_size, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(kernel_ftoi, dim3(num_blocks), dim3(threads_per_block), 0, 0,
                     d_fsrc0, d_fsrc1, d_idst, num_points);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(h_idst, d_idst, buf_size, hipMemcpyDeviceToHost));
  {
    int errors = 0;
    for (uint32_t i = 0; i < num_points; ++i) {
      int32_t expected = (int32_t)(h_fsrc0[i] + h_fsrc1[i]);
      if (h_idst[i] != expected) {
        if (errors < 10) printf("  error[%d]: expected=%d, actual=%d\n", i, expected, h_idst[i]);
        errors++;
      }
    }
    if (errors == 0) { printf("Test ftoi: PASSED\n"); tests_passed++; }
    else { printf("Test ftoi: FAILED (%d errors)\n", errors); total_errors += errors; }
  }

  // Test 8: Int to float conversion
  printf("\n=== Test: itof ===\n");
  tests_run++;
  hipLaunchKernelGGL(kernel_itof, dim3(num_blocks), dim3(threads_per_block), 0, 0,
                     d_isrc0, d_isrc1, d_fdst, num_points);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(h_fdst, d_fdst, buf_size, hipMemcpyDeviceToHost));
  {
    int errors = 0;
    for (uint32_t i = 0; i < num_points; ++i) {
      float expected = (float)(h_isrc0[i] + h_isrc1[i]);
      if (!almost_equal(h_fdst[i], expected)) {
        if (errors < 10) printf("  error[%d]: expected=%f, actual=%f\n", i, (double)expected, (double)h_fdst[i]);
        errors++;
      }
    }
    if (errors == 0) { printf("Test itof: PASSED\n"); tests_passed++; }
    else { printf("Test itof: FAILED (%d errors)\n", errors); total_errors += errors; }
  }

  // Print summary
  printf("\n========================================\n");
  printf("Summary: %d/%d tests passed\n", tests_passed, tests_run);
  printf("========================================\n");

  // Cleanup
  HIP_CHECK(hipFree(d_isrc0));
  HIP_CHECK(hipFree(d_isrc1));
  HIP_CHECK(hipFree(d_idst));
  HIP_CHECK(hipFree(d_fsrc0));
  HIP_CHECK(hipFree(d_fsrc1));
  HIP_CHECK(hipFree(d_fdst));
  free(h_isrc0);
  free(h_isrc1);
  free(h_idst);
  free(h_fsrc0);
  free(h_fsrc1);
  free(h_fdst);

  if (total_errors == 0) {
    printf("ALL TESTS PASSED!\n");
    return 0;
  } else {
    printf("FOUND %d TOTAL ERRORS!\n", total_errors);
    return 1;
  }
}

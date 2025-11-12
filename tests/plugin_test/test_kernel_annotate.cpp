// Test kernel using CUDA attribute annotation
// The plugin looks for CUDAGlobalAttr which is applied by __global__

// In CUDA mode, __global__ is automatically recognized
// Let's declare functions that will be recognized as CUDA kernels

extern "C" {

void vecadd(float* a, float* b, float* c, int n) __attribute__((annotate("cuda_global")));

void vecadd(float* a, float* b, float* c, int n) {
  // Dummy implementation
}

void mixed_types(
    char c, short s, int i, long l,
    float* ptr1, double* ptr2, int final
) __attribute__((annotate("cuda_global")));

void mixed_types(
    char c, short s, int i, long l,
    float* ptr1, double* ptr2, int final
) {
  // Dummy implementation
}

}

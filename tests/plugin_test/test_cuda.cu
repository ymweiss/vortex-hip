// CUDA test kernel - must be compiled in CUDA mode
// .cu extension tells Clang this is CUDA code

__global__ void vecadd(float* a, float* b, float* c, int n) {
  // Dummy body
}

__global__ void mixed_types(
    char c,
    short s,
    int i,
    long l,
    float* ptr1,
    double* ptr2,
    int final
) {
  // Dummy body
}

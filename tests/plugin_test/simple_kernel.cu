// Minimal CUDA kernel to see LLVM IR representation
// Compile: clang++ -x cuda --cuda-device-only -S -emit-llvm simple_kernel.cu

extern "C" __global__ void vecadd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

extern "C" __global__ void mixed_types(
    char c,
    short s,
    int i,
    long l,
    float* ptr1,
    double* ptr2,
    int final
) {
    // Test alignment handling
}

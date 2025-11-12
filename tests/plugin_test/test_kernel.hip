// Test kernel for HIPMetadataExtractor plugin
// This kernel has various parameter types to test offset calculation

__global__ void vecadd(float* a, float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

__global__ void mixed_types(
    char c,           // 1 byte, align 1
    short s,          // 2 bytes, align 2
    int i,            // 4 bytes, align 4
    long l,           // 8 bytes, align 8
    float* ptr1,      // 4 bytes, align 4
    double* ptr2,     // 4 bytes, align 4
    int final         // 4 bytes, align 4
) {
  // Test kernel with mixed types to verify alignment handling
}

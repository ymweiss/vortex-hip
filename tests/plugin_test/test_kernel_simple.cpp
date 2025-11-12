// Simple test kernel with __global__ defined as CUDA attribute
// This is for testing the HIPMetadataExtractor plugin only

#define __global__ __attribute__((global))

extern "C" {

__global__ void vecadd(float* a, float* b, float* c, int n) {
  // Dummy function body - we only care about metadata extraction
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
  // Dummy function body
}

}

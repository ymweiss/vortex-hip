// Simple test kernel for metadata generation
// Compile with: clang++ -target riscv64 -g simple_kernel.cpp -o simple_kernel.elf

__attribute__((noinline))
void vectorAdd(float* a, float* b, float* c, int n) {
    // Simple vector addition kernel
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

__attribute__((noinline))
void saxpy(float* y, float* x, float a, int n) {
    // SAXPY: y = a*x + y
    for (int i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

__attribute__((noinline))
void dotProduct(float* a, float* b, float* result, int n) {
    // Dot product kernel
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    *result = sum;
}

int main() {
    // Dummy main to make it a valid executable
    return 0;
}

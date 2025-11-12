#include <stdio.h>

extern "C" {
    extern const unsigned char kernel_vxbin[];
    extern const unsigned char kernel_vxbin_end[];
}

int main() {
    size_t size1 = (size_t)(&kernel_vxbin_end[0]) - (size_t)(&kernel_vxbin[0]);
    size_t size2 = (size_t)kernel_vxbin_end - (size_t)kernel_vxbin;
    
    printf("kernel_vxbin address: %p\n", (void*)kernel_vxbin);
    printf("kernel_vxbin_end address: %p\n", (void*)kernel_vxbin_end);
    printf("Size method 1 (&end[0] - &start[0]): %zu\n", size1);
    printf("Size method 2 (end - start): %zu\n", size2);
    printf("Size in hex: 0x%zx\n", size2);
    
    return 0;
}

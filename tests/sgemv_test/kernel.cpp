// Vortex kernel for sgemv
// TODO: Adapt from ../../../tests/sgemv.cpp

#include <vx_spawn.h>
#include <stdint.h>

// TODO: Define kernel argument structure
// struct SgemvArgs {
//     uint32_t grid_dim[3];
//     uint32_t block_dim[3];
//     uint64_t shared_mem;
//     // TODO: Add user arguments
// } __attribute__((packed));

// TODO: Implement kernel_body function
// void kernel_body(SgemvArgs* __UNIFORM__ args) {
//     // TODO: Implement kernel logic
// }

// Device main - spawns threads
int main() {
    // TODO: Get args from CSR
    // TODO: Calculate num_threads
    // TODO: Call vx_spawn_threads
    return 0;
}

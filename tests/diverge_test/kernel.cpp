// Vortex kernel for control flow divergence test
// Adapted from tests/diverge.cpp
// Tests various divergent control flow patterns

#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <stdint.h>

// Kernel argument structure
struct DivergeArgs {
    // Runtime-provided fields
    uint32_t grid_dim[3];
    uint32_t block_dim[3];
    uint64_t shared_mem;

    // User arguments
    int32_t* src;
    int32_t* dst;
    uint32_t num_points;
    uint32_t dummy;  // Padding for 4-arg metadata pattern
} __attribute__((packed));

// Helper structure for device function
struct key_t {
    uint32_t user;
};

// Device function - called from kernel
static void hacker(key_t* key, uint32_t task_id) {
    key->user = task_id;
}

// Inline min/max functions
static inline int32_t min_val(int32_t a, int32_t b) {
    return (a < b) ? a : b;
}

static inline int32_t max_val(int32_t a, int32_t b) {
    return (a > b) ? a : b;
}

// Kernel body - tests various control flow divergence patterns
void kernel_body(DivergeArgs* __UNIFORM__ args) {
    uint32_t task_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (task_id >= args->num_points) return;

    int32_t value = args->src[task_id];

    // Pattern 1: Loop with device function call
    // Tests divergent loop iterations and function calls
    key_t key;
    key.user = 0;
    uint32_t samples = args->num_points;
    while (samples--) {
        hacker(&key, task_id);
        if ((key.user & 0x1) == 0) {
            value += 1;
        }
    }

    // Pattern 2: None taken branch
    // All threads take same path (else)
    if (task_id >= 0x7fffffff) {
        value = 0;
    } else {
        value += 2;
    }

    // Pattern 3: Divergent nested if/else
    // Different threads take different paths
    if (task_id > 1) {
        if (task_id > 2) {
            value += 6;
        } else {
            value += 5;
        }
    } else {
        if (task_id > 0) {
            value += 4;
        } else {
            value += 3;
        }
    }

    // Pattern 4: All taken branch
    // All threads take same path (if)
    if (task_id >= 0) {
        value += 7;
    } else {
        value = 0;
    }

    // Pattern 5: Divergent loop
    // Each thread executes different number of iterations
    for (uint32_t i = 0; i < task_id; ++i) {
        value += args->src[i];
    }

    // Pattern 6: Switch statement
    // Tests divergent multi-way branch
    switch (task_id) {
    case 0:
        value += 1;
        break;
    case 1:
        value -= 1;
        break;
    case 2:
        value *= 3;
        break;
    case 3:
        value *= 5;
        break;
    default:
        break;
    }

    // Pattern 7: Nested ternary operators (select)
    // Tests complex conditional expressions
    value += (task_id >= 0) ?
             ((task_id > 5) ? args->src[0] : (int32_t)task_id) :
             ((task_id < 5) ? args->src[1] : -(int32_t)task_id);

    // Pattern 8: Min/max operations
    // Tests conditional data-dependent operations
    value += min_val(args->src[task_id], value);
    value += max_val(args->src[task_id], value);

    args->dst[task_id] = value;
}

// Device main - spawns threads
int main() {
    DivergeArgs* args = (DivergeArgs*)csr_read(VX_CSR_MSCRATCH);

    // 1D grid for element-wise operations
    return vx_spawn_threads(1, args->grid_dim, args->block_dim,
                           (vx_kernel_func_cb)kernel_body, args);
}

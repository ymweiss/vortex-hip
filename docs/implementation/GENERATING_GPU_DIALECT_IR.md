# Generating GPU Dialect IR from HIP Code

## Status: Option A - Part 1 Complete

Successfully configured Polygeist to generate GPU dialect IR from HIP source files without requiring CUDA toolkit installation.

## Working Configuration

### Script Location
`scripts/polygeist/hip-to-gpu-dialect.sh`

### Command Template
```bash
cgeist <input.hip> \
    --cuda-gpu-arch=sm_60 \
    -nocudalib \
    -nocudainc \
    -resource-dir=Polygeist/llvm-project/build/lib/clang/18 \
    -Ihip/include \
    --function=* \
    --emit-cuda \
    -S \
    -o <output.mlir>
```

### Key Findings

1. **Use Official HIP Headers**: The HIP repository headers at `hip/include` provide complete HIP API definitions

2. **CUDA Built-ins via Clang**: Clang's resource directory provides `__clang_cuda_builtin_vars.h` which defines threadIdx, blockIdx, etc.

3. **Syntax-Only Mode Works**: With `POLYGEIST_ENABLE_CUDA_SYNTAX_ONLY=ON`, Polygeist successfully parses HIP/CUDA syntax without CUDA toolkit

4. **Host Function Required**: To generate GPU module IR, you need a host function that launches the kernel with `<<<>>>` syntax

## Example Output

For this input:
```cpp
#include <hip/hip_runtime.h>

__global__ void basic_kernel(int32_t* src, int32_t* dst, uint32_t count) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < count) {
    dst[tid] = src[tid];
  }
}

void launch_basic(int32_t* d_src, int32_t* d_dst, uint32_t count, int tpb) {
  int num_blocks = (count + tpb - 1) / tpb;
  basic_kernel<<<num_blocks, tpb>>>(d_src, d_dst, count);
}
```

Polygeist generates GPU dialect IR with:
- `gpu.module` containing the kernel
- `gpu.func` with kernel attribute
- `gpu.thread_id`, `gpu.block_id`, `gpu.block_dim` operations
- `gpu.launch_func` for kernel invocation

## Next Steps (Option A - Part 2)

Create `gpu-to-vortex` translation tool that:
1. Parses GPU dialect MLIR
2. Maps GPU operations to Vortex ISA
3. Generates Vortex assembly/binary

## Test Files

Located in `hip_tests/`:
- `basic.hip` - Simple memory copy
- `vecadd.hip` - Vector addition
- `dotproduct.hip` - Dot product
- `sgemm.hip` - Matrix multiplication

All use `#include <hip/hip_runtime.h>` and can be processed by Polygeist.

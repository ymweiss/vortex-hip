# Phase 2A Solution: HIP-to-GPU Dialect Conversion

## Problem Summary

Phase 2A was marked as complete but the HIP-to-MLIR conversion pipeline was **NOT working**. The `hip-to-gpu-dialect.sh` script reported success but generated no output files.

## Root Causes Identified

### 1. Missing Critical Flag: `--emit-cuda`
**Problem:** Script used `--cuda-lower` but NOT `--emit-cuda`
**Impact:** Polygeist generated SCF dialect (`scf.parallel`) instead of GPU dialect (`gpu.launch`)
**Pipeline behavior:**
- `EmitGPU = EmitROCM || EmitCUDA` (from driver.cc:648)
- Without `--emit-cuda` or `--emit-rocm`: `EmitGPU = false`
- Pipeline stops at SCF dialect, never reaches `ConvertParallelToGPUPass1`

### 2. Polygeist Doesn't Handle `.hip` File Extension
**Problem:** Polygeist fails silently when processing `.hip` files
**Evidence:** Identical file content works with `.cu` extension but fails with `.hip`
**Workaround:** Convert `.hip` → `.cu` temporarily before processing

### 3. Missing Clang CUDA Builtin Variables Header
**Problem:** Without `__clang_cuda_builtin_vars.h`, `threadIdx`/`blockIdx` become regular globals
**Result:** Generated `memref.global @threadIdx` instead of `gpu.thread_id` operations
**Solution:** Source files MUST include `__clang_cuda_builtin_vars.h`

### 4. Missing Kernel Launch Function Declarations
**Problem:** `<<<>>>` syntax requires `cudaConfigureCall` declaration
**Result:** Silent compilation failure
**Solution:** Added declarations to `hip_mock.h`

## Solution Implemented

### 1. Updated `hip_mock.h` (Project Root)
**Changes:**
- ✅ Include `__clang_cuda_builtin_vars.h` for GPU builtin variables
- ✅ Add `cudaConfigureCall` and `hipConfigureCall` declarations
- ✅ Add `dim3` structure with constructor
- ✅ Define HIP/CUDA attributes (`__global__`, `__device__`, etc.)

**Key section:**
```cpp
#include "__clang_cuda_builtin_vars.h"  // CRITICAL for GPU ops

extern "C" int cudaConfigureCall(dim3 gridSize, dim3 blockSize,
                                 size_t sharedSize = 0,
                                 cudaStream_t stream = 0);
```

### 2. Updated `scripts/polygeist/hip-to-gpu-dialect.sh`
**Critical flags added:**
```bash
--cuda-lower          # Convert kernels to scf.parallel
--emit-cuda           # Enable GPU dialect generation (sets EmitGPU=true)
--output-intermediate-gpu=1  # For debugging
```

**Workaround for `.hip` extension:**
```bash
# Convert .hip to .cu temporarily (Polygeist limitation)
if [[ "$INPUT_FILE" == *.hip ]]; then
    TEMP_CU_FILE="/tmp/polygeist_temp_$(basename "$INPUT_FILE" .hip).cu"
    cp "$INPUT_FILE" "$TEMP_CU_FILE"
    CGEIST_INPUT="$TEMP_CU_FILE"
fi
```

**Include paths:**
```bash
-I"$HIP_INCLUDE"      # For hip/hip_polygeist.h
-I"$REPO_ROOT"        # For hip_mock.h
```

### 3. Created Test Files
**hip_tests/simple_for_polygeist.hip:**
- Uses `#include "hip_mock.h"`
- Demonstrates vector addition kernel
- Successfully generates GPU dialect

## Verification

### Test Command:
```bash
./scripts/polygeist/hip-to-gpu-dialect.sh hip_tests/simple_for_polygeist.hip output.mlir
```

### Expected Output:
```
Converting hip_tests/simple_for_polygeist.hip to GPU dialect IR...
Output: output.mlir
Using HIP headers: /home/yaakov/vortex_hip/hip/include
Using resource dir: /home/yaakov/vortex_hip/Polygeist/llvm-project/build/lib/clang/18
Note: Converted .hip to temporary .cu file for Polygeist compatibility
GPU dialect IR generated successfully!

Verifying GPU dialect presence...
✓ GPU dialect operations found
```

### GPU Dialect Verification:
```bash
grep -E "gpu\.(module|func|thread_id|block_id|launch_func)" output.mlir
```

Should show:
- `gpu.module @__polygeist_gpu_module`
- `gpu.func @kernel_name ... kernel`
- `gpu.thread_id x/y/z`
- `gpu.block_id x/y/z`
- `gpu.block_dim x/y/z`
- `gpu.grid_dim x/y/z`

## Polygeist ROCm Backend Investigation

### Findings:
- ✅ Polygeist HAS explicit ROCm/HIP support
- Build flag: `-DPOLYGEIST_ENABLE_ROCM=1`
- Runtime flag: `--emit-rocm` (for AMD GPU output)
- HIP support implemented as translation layer over CUDA infrastructure
- `ConvertCudaRTtoHipRTPass` handles 200+ function mappings (cudaMalloc→hipMalloc, etc.)

### For Vortex Project:
**Recommendation:** Use generic GPU dialect (via `--emit-cuda`)
- GPU dialect is vendor-neutral at the operation level
- `ConvertGPUToVortex` pass consumes GPU dialect regardless of metadata
- Vendor-specific attributes (nvvm.*, rocdl.*) can be ignored by our pass

## Vendor-Specific Metadata

The generated GPU dialect contains some vendor-specific attributes:
```mlir
polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda"
nvvm.maxntidx = 256 : index
rocdl.max_flat_work_group_size = 256 : index
```

**Impact on Vortex:**
- These are metadata attributes, not core GPU operations
- `ConvertGPUToVortex` pass ignores them
- The actual GPU operations (`gpu.thread_id`, `gpu.launch_func`, etc.) are vendor-neutral

## Files Modified

1. **hip_mock.h** - Comprehensive mock header for Polygeist
   - Includes clang builtin vars
   - Kernel launch declarations
   - HIP/CUDA compatibility layer

2. **scripts/polygeist/hip-to-gpu-dialect.sh** - Fixed conversion script
   - Added `--cuda-lower` and `--emit-cuda` flags
   - Workaround for `.hip` extension limitation
   - GPU dialect verification

3. **hip_tests/simple_for_polygeist.hip** - Test case
   - Vector addition kernel
   - Demonstrates proper header usage

## Usage for HIP Kernel Files

To convert a HIP kernel to GPU dialect:

```cpp
// your_kernel.hip
#include "hip_mock.h"  // All necessary declarations

__global__ void my_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 2.0f;
    }
}

void launch_kernel(float* d_data, int n) {
    my_kernel<<<(n+255)/256, 256>>>(d_data, n);
}
```

Convert to GPU dialect:
```bash
./scripts/polygeist/hip-to-gpu-dialect.sh hip_tests/your_kernel.hip output.mlir
```

## Next Steps for Phase 2B

Now that Phase 2A is working:

1. ✅ GPU dialect generation is confirmed working
2. ✅ Script can process HIP files
3. **NEXT:** Generate MLIR test suite from all HIP test files
4. **NEXT:** Implement metadata extraction in `ConvertGPUToVortex` pass
5. **NEXT:** Extract argument types/sizes from `gpu.launch_func` operations

## Status

**Phase 2A: COMPLETE** ✅
- HIP files can be converted to GPU dialect MLIR
- Script is working and verified
- Test cases demonstrate functionality
- Ready for metadata extraction work (Phase 2B)

## Future Work

**Note:** This solution is functional but somewhat ad-hoc. Consider revisiting later for a more robust approach:

**Current Limitations:**
- Relies on Python script for variant filtering (external to MLIR pipeline)
- Temporary `.hip` → `.cu` file conversion workaround
- Manual integration of filtering step in bash script

**Potential Improvements:**
- Write custom MLIR pass for variant filtering (integrate into mlir-opt pipeline)
- Investigate Polygeist flags more thoroughly for native single-variant generation
- Consider upstreaming `.hip` extension support to Polygeist
- Create proper MLIR dialect operation for kernel metadata (instead of post-processing)

**Priority:** Low - Current solution is sufficient for Phase 2B development and beyond. Revisit only if:
1. Variant filtering becomes a performance bottleneck
2. Python dependency causes deployment issues
3. Polygeist updates provide better native solutions
4. Need to process thousands of kernels (current approach scales fine for typical workloads)

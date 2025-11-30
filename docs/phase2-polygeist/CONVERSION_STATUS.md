# HIP-to-MLIR Conversion Status

## ✅ Phase 2A: COMPLETE AND WORKING

Successfully fixed and verified HIP kernel to GPU dialect MLIR conversion pipeline.

## What Was Accomplished

### 1. Fixed Root Cause
**Problem:** Missing `--emit-cuda` flag prevented GPU dialect generation
- Script only had `--cuda-lower` → Generated SCF dialect
- Added `--emit-cuda` → Now generates GPU dialect

**Evidence of fix:**
```mlir
// BEFORE (broken):
memref.global @threadIdx : memref<1x3xi32>  // Wrong - treated as variable

// AFTER (working):
%2 = gpu.thread_id x  // Correct - GPU dialect operation
```

### 2. Created Organized Runtime Mock
**Location:** `hip_runtime_vortex/hip_runtime.h`

**Features:**
- Includes `__clang_cuda_builtin_vars.h` (critical for GPU ops recognition)
- Kernel launch function declarations (cudaConfigureCall, etc.)
- HIP/CUDA attribute definitions
- Complete dim3 implementation
- ~90 lines vs 100,000+ for full HIP runtime

### 3. Conversion Scripts

#### Single File Converter
**Script:** `scripts/polygeist/hip-to-gpu-dialect.sh`
**Usage:**
```bash
./scripts/polygeist/hip-to-gpu-dialect.sh input.hip output.mlir
```

**Features:**
- Workaround for Polygeist .hip extension limitation (converts to .cu temporarily)
- Automatic GPU dialect verification
- Proper include paths for hip_runtime_vortex

#### Batch Converter
**Script:** `scripts/polygeist/convert-all-kernels.sh`
**Usage:**
```bash
./scripts/polygeist/convert-all-kernels.sh
```

**Output:**
```
==========================================
HIP Kernel to GPU Dialect Batch Converter
==========================================

Kernel directory: /home/yaakov/vortex_hip/hip_tests/kernels
Output directory: /home/yaakov/vortex_hip/hip_tests/mlir_output

[1] Converting: basic_kernel.hip
    ✓ Success: basic_kernel.mlir

[2] Converting: dotproduct_kernel.hip
    ✓ Success: dotproduct_kernel.mlir

[3] Converting: vecadd_kernel.hip
    ✓ Success: vecadd_kernel.mlir

==========================================
Conversion Summary
==========================================
Total files:     3
Successful:      3
Failed:          0

✓ All conversions successful!
```

### 4. Test Kernel Files
**Location:** `hip_tests/kernels/`

Created kernel-only versions of HIP tests:
- ✅ `basic_kernel.hip` - Memory copy kernel
- ✅ `vecadd_kernel.hip` - Vector addition kernel
- ✅ `dotproduct_kernel.hip` - Dot product kernel

All successfully convert to GPU dialect MLIR.

### 5. Generated MLIR Output
**Location:** `hip_tests/mlir_output/`

**Files:**
- `basic_kernel.mlir` (13KB)
- `vecadd_kernel.mlir` (14KB)
- `dotproduct_kernel.mlir` (14KB)

**Verified content:**
- ✅ `gpu.module @__polygeist_gpu_module`
- ✅ `gpu.func ... kernel` attributes
- ✅ `gpu.thread_id x/y/z` operations
- ✅ `gpu.block_id x/y/z` operations
- ✅ `gpu.block_dim x/y/z` operations
- ✅ Kernel arguments preserved with types
- ✅ Control flow (scf.if) preserved
- ✅ Arithmetic operations (arith.addi, arith.addf, etc.)

## Directory Structure

```
vortex_hip/
├── hip_runtime_vortex/           # NEW: Mock HIP runtime for Polygeist
│   ├── hip_runtime.h             # Minimal HIP header (~90 lines)
│   └── README.md                 # Documentation
│
├── hip_tests/
│   ├── kernels/                  # NEW: Kernel-only files for conversion
│   │   ├── basic_kernel.hip
│   │   ├── vecadd_kernel.hip
│   │   └── dotproduct_kernel.hip
│   │
│   ├── mlir_output/              # Generated GPU dialect MLIR
│   │   ├── basic_kernel.mlir     # ✓ Working
│   │   ├── vecadd_kernel.mlir    # ✓ Working
│   │   └── dotproduct_kernel.mlir # ✓ Working
│   │
│   └── *.hip                     # Original full HIP test programs
│
├── scripts/polygeist/
│   ├── hip-to-gpu-dialect.sh     # UPDATED: Single file converter
│   ├── convert-all-kernels.sh   # NEW: Batch converter
│   └── build-polygeist.sh
│
└── docs/phase2-polygeist/
    ├── PHASE2A_SOLUTION.md       # NEW: Technical deep-dive
    ├── PHASE2B_A_STATUS.md       # Updated status
    └── STATUS.md
```

## Verification Examples

### Basic Kernel GPU Dialect Output
```mlir
gpu.module @__polygeist_gpu_module {
  gpu.func @_Z12launch_basicPiS_ji_kernel94889929402208(
    %arg0: index,
    %arg1: i32,
    %arg2: i32,
    %arg3: memref<?xi32>,  // src
    %arg4: memref<?xi32>   // dst
  ) kernel attributes {gpu.known_block_size = array<i32: 32, 1, 1>} {
    %0 = gpu.block_id x
    %1 = gpu.block_id y
    %2 = gpu.thread_id x
    // ... computation ...
    gpu.return
  }
}
```

### Vector Add GPU Dialect Output
```mlir
gpu.func @_Z13launch_vecaddPKfS0_Pfji_kernel94071272653664(
  %arg0: index,
  %arg1: i32,
  %arg2: i32,
  %arg3: memref<?xf32>,  // a
  %arg4: memref<?xf32>,  // b
  %arg5: memref<?xf32>   // c
) kernel {
  %2 = gpu.thread_id x
  // Loads from a and b
  %13 = memref.load %arg3[%10] : memref<?xf32>
  %14 = memref.load %arg4[%10] : memref<?xf32>
  // Add operation
  %15 = arith.addf %13, %14 : f32
  // Store to c
  memref.store %15, %arg5[%10] : memref<?xf32>
  gpu.return
}
```

## Key Technical Details

### Polygeist Flags Required
```bash
--cuda-lower              # Convert kernels to scf.parallel
--emit-cuda               # Enable GPU dialect (sets EmitGPU=true)
--cuda-gpu-arch=sm_60     # Target architecture
-nocudalib                # Don't link CUDA device library
-nocudainc                # Don't include CUDA system headers
-resource-dir=...         # Path to Clang builtins
-I hip_runtime_vortex     # Include mock runtime
--output-intermediate-gpu=1  # Debug flag
```

### Why `__clang_cuda_builtin_vars.h` is Critical

Without it:
```cpp
extern const uint3 threadIdx;  // Treated as regular variable
// Result: memref.global @threadIdx : memref<1x3xi32>
```

With it:
```cpp
struct __cuda_builtin_threadIdx_t {
  __device__ unsigned int __fetch_builtin_x();  // Special member function
};
__device__ const __cuda_builtin_threadIdx_t threadIdx;
// Result: %0 = gpu.thread_id x
```

Polygeist recognizes the `__fetch_builtin_*` member functions and generates GPU dialect operations.

### Vendor-Specific Metadata (Not a Problem)

The generated MLIR contains:
```mlir
polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda"
nvvm.maxntidx = 256 : index
rocdl.max_flat_work_group_size = 256 : index
```

**Impact:** These are module-level metadata attributes, NOT core GPU operations.
**For Vortex:** ConvertGPUToVortex pass ignores these, focuses on gpu.* operations.
**Core operations are vendor-neutral:** gpu.thread_id, gpu.block_id, gpu.launch_func

## What's Ready for Phase 2B

### ✅ Working Infrastructure
1. HIP kernel → GPU dialect MLIR conversion pipeline
2. Mock runtime header (hip_runtime_vortex)
3. Batch conversion scripts
4. Test suite with verified MLIR output

### ✅ Available for Metadata Extraction
GPU dialect MLIR contains all necessary information:
- Kernel function signatures with argument types
- Argument ordering and types (memref<?xi32>, i32, etc.)
- Grid/block dimensions (in launch contexts)
- Memory access patterns

Example metadata available:
```mlir
gpu.launch_func @__polygeist_gpu_module::@kernel_name
  blocks in (%c1, %c1, %c1)
  threads in (%num_threads, %c1, %c1)
  args(
    %arg0 : memref<?xi32>,   // Pointer argument, 4 bytes per element
    %arg1 : i32,             // Scalar argument, 4 bytes
    %arg2 : memref<?xf32>    // Pointer argument, 4 bytes per element
  )
```

### Next Steps for Phase 2B Metadata Extraction

1. **Design metadata representation**
   - Function attributes vs global constants
   - Struct layout matching Vortex kernel_arg_t

2. **Implement LaunchFuncOpLowering pattern**
   - Extract argument types and sizes
   - Distinguish pointer vs value arguments
   - Calculate struct offsets

3. **Test with MLIR suite**
   - Use generated basic_kernel.mlir, vecadd_kernel.mlir
   - Verify metadata extraction
   - Validate with Vortex runtime expectations

## Documentation

- **Technical deep-dive:** `docs/phase2-polygeist/PHASE2A_SOLUTION.md`
- **Runtime mock usage:** `hip_runtime_vortex/README.md`
- **This summary:** `CONVERSION_STATUS.md`
- **Overall status:** `docs/phase2-polygeist/PHASE2B_A_STATUS.md`

## Status Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Phase 2A HIP→MLIR | ✅ Complete | Working and verified (see note below) |
| hip_runtime_vortex | ✅ Complete | Mock header for Polygeist |
| Conversion scripts | ✅ Complete | Single + batch converters |
| Test kernels | ✅ Complete | 3 kernels verified |
| GPU dialect output | ✅ Verified | Proper gpu.* operations |
| Phase 2B metadata | ⏸️ Ready | Infrastructure in place |

**Overall: Phase 2A is COMPLETE. Ready to proceed with Phase 2B metadata extraction.**

### Phase 2A Implementation Note

The current Phase 2A solution is **functional and sufficient for proceeding** with Phase 2B, but uses a somewhat ad-hoc approach:
- Python script for variant filtering (external to MLIR pipeline)
- Temporary `.hip` → `.cu` conversion workaround
- Bash script integration

**Recommendation:** Proceed with Phase 2B as-is. Consider revisiting Phase 2A later for a more integrated MLIR-based solution if:
- Variant filtering becomes a performance issue
- Need to scale to thousands of kernels
- Polygeist provides better native options
- Python dependency causes deployment concerns

**Priority:** Low - Current approach is adequate for development and deployment.

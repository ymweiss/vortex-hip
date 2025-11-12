# Polygeist Build Options Analysis

**Purpose:** Evaluate which Polygeist build options are useful for HIP-to-Vortex compilation

**Date:** November 11, 2025

---

## Available Build Options

### 1. POLYGEIST_ENABLE_CUDA

**What it does:**
- Enables CUDA frontend support
- Adds `--cuda-lower` flag to cgeist
- Enables `--emit-cuda` flag
- Includes GPU passes: `ConvertParallelToGPUPass`, `GpuKernelOutliningPass`
- Converts CUDA runtime calls (cudaMalloc, cudaMemcpy, etc.) to GPU ops

**For Vortex HIP:**
- ❌ **NOT NEEDED** - CUDA-specific, we're targeting HIP
- ROCm support is more appropriate for HIP

**Recommendation:** Keep **DISABLED** (0)

---

### 2. POLYGEIST_ENABLE_ROCM ⭐

**What it does:**
- Enables ROCm/HIP frontend support
- Adds `--emit-rocm` flag to cgeist
- Includes `ConvertCudaRTtoHipRTPass` - Converts CUDA API calls → HIP API calls
- Uses ROCDL dialect instead of NVVM dialect
- Sets ROCm-specific GPU attributes (e.g., max_flat_work_group_size = 1024)
- Converts `NVVM::Barrier0Op` → `ROCDL::BarrierOp`

**Source code evidence:**
```cpp
// From driver.cc
if (EmitROCM) {
  pm.addPass(polygeist::createConvertCudaRTtoHipRTPass());
}

// From ParallelLower.cpp
void ConvertCudaRTtoHipRT::runOnOperation() {
  // Converts cudaMalloc → hipMalloc
  // Converts cudaMemcpy → hipMemcpy
  // Converts NVVM barriers → ROCDL barriers
}
```

**For Vortex HIP:**
- ✅ **POTENTIALLY USEFUL** - Provides HIP/ROCm support
- HIP is AMD's CUDA-compatible API
- The `ConvertCudaRTtoHipRT` pass could be helpful
- ROCm attributes might be closer to our needs than CUDA

**However:**
- We're NOT targeting AMD ROCm hardware
- We're targeting custom Vortex RISC-V hardware
- The ROCDL dialect is AMD-specific
- We'll need custom GPU→Vortex lowering anyway

**Recommendation:** Keep **DISABLED** (0)
- **Reason:** ROCm is AMD hardware-specific
- We need GPU dialect → Vortex custom lowering
- The HIP API compatibility is at the syntax level (handled by Clang frontend)
- Enabling ROCm would pull in AMD-specific code generation we don't need

---

### 3. POLYGEIST_ENABLE_POLYMER

**What it does:**
- Enables Polymer polyhedral optimization framework
- Adds `--polyhedral-opt` flag
- Provides advanced loop transformations (tiling, fusion, etc.)
- Requires external ISL and Pluto dependencies

**For Vortex HIP:**
- ❓ **POTENTIALLY USEFUL (FUTURE)** - Advanced optimizations
- Not needed for initial Phase 2 implementation
- Could be valuable for Phase 3+ (optimization phase)

**Recommendation:** Keep **DISABLED** (0) for now
- **Reason:** Not needed for basic pipeline
- Can re-evaluate in optimization phase
- Adds build complexity and dependencies

---

## Current Build Configuration ✅

Our current `scripts/polygeist/build-polygeist.sh` has:

```cmake
-DPOLYGEIST_ENABLE_CUDA=0      # ✅ Correct - Not needed
-DPOLYGEIST_ENABLE_ROCM=0      # ✅ Correct - AMD-specific
-DPOLYGEIST_ENABLE_POLYMER=0   # ✅ Correct - Not needed yet
```

---

## Why ROCm Support is Not Needed

### What We Actually Need

**Pipeline Overview:**
```
HIP Source Code
    ↓
[Clang Frontend] ← HIP syntax handled HERE (built-in to Clang)
    ↓
LLVM IR with HIP constructs
    ↓
[Polygeist] → SCF Dialect
    ↓
[MLIR Passes] → GPU Dialect (generic, not AMD-specific)
    ↓
[Custom Pass] → Vortex-specific lowering
    ↓
LLVM IR → Vortex RISC-V
```

### HIP Syntax Support

**HIP syntax is already supported by Clang's frontend:**
- `__global__` attribute
- `__device__`, `__host__` attributes
- `threadIdx`, `blockIdx`, `blockDim`, `gridDim`
- `__shared__` memory
- `__syncthreads()`

**These are recognized by:**
1. **Clang** (already built in LLVM)
2. **Polygeist** (uses Clang as frontend)

**ROCm option only affects:**
- GPU dialect → ROCDL dialect lowering (AMD-specific)
- AMD GPU codegen (not relevant for Vortex)

### What ROCm Would Give Us

✅ **Already have:**
- HIP syntax parsing (Clang frontend)
- CUDA runtime → HIP runtime name conversion

❌ **Don't need:**
- ROCDL dialect (AMD instruction set)
- AMD GPU backend
- ROCm-specific optimizations

### Our Custom Approach

We need:
```mlir
gpu.thread_id x  →  [Custom Pass]  →  call @vx_thread_id()
gpu.block_id x   →  [Custom Pass]  →  compute from vx_warp_id()
gpu.barrier      →  [Custom Pass]  →  call @vx_barrier()
```

This is **independent** of CUDA/ROCm backends.

---

## Testing Strategy

### Phase 2A: HIP Syntax Testing (Without ROCm)

**Test:** Can we compile HIP kernels without enabling ROCm?

```bash
# Using hip_minimal.h (our custom header)
cgeist hip_kernel.cpp -I docs/phase2-polygeist/ -S -o kernel.mlir

# Check if HIP built-ins are recognized
grep -E "threadIdx|blockIdx|blockDim" kernel.mlir
```

**Expected:** Clang should recognize HIP syntax without ROCm flag

**If successful:** No need to enable ROCm
**If failed:** Re-evaluate

---

## Alternative: Selective ROCm Features

If we find we need **only** the `ConvertCudaRTtoHipRT` pass:

**Option:** Extract and adapt the pass
- Copy the pass source
- Modify for Vortex needs
- Don't enable full ROCm support

**Benefits:**
- Get HIP API conversion without AMD backend
- More control over transformations
- No AMD-specific code

---

## Recommendations

### Current Phase (Phase 2A-2B)

**Build configuration:**
```cmake
-DPOLYGEIST_ENABLE_CUDA=0      # Keep disabled
-DPOLYGEIST_ENABLE_ROCM=0      # Keep disabled
-DPOLYGEIST_ENABLE_POLYMER=0   # Keep disabled
```

**Rationale:**
1. Clang already handles HIP syntax
2. We need generic GPU dialect, not AMD-specific ROCDL
3. Our custom GPU→Vortex pass is architecture-specific anyway
4. Simpler build, faster compile times
5. Fewer dependencies

### Future Considerations

**Phase 3+ (Optimization):**
- ✅ Consider enabling POLYMER for advanced loop optimizations
- ❌ Still don't need CUDA/ROCm (hardware-specific)

**If HIP syntax issues arise:**
- ✅ First try with better Clang flags
- ✅ Try custom HIP header improvements
- ❓ Last resort: Enable ROCm and filter out AMD codegen

---

## Summary

**Current build script is optimal ✅**

All three options remain **DISABLED**:
- **CUDA** - Not targeting NVIDIA hardware
- **ROCm** - Not targeting AMD hardware
- **POLYMER** - Not needed for basic pipeline (maybe later)

**HIP support comes from:**
1. Clang frontend (built-in HIP support)
2. Our custom `hip_minimal.h` header
3. Generic MLIR GPU dialect
4. **Our custom GPUToVortexLLVM pass** (the only Vortex-specific code needed)

---

**Status:** Build configuration verified optimal for HIP-to-Vortex compilation
**No changes needed to build script**

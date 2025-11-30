# Handling Polygeist Kernel Variants in ConvertGPUToVortex

## Problem

Polygeist generates multiple specialized versions of each kernel (6 variants with different block sizes: 32, 64, 128, 256, 512, 1024).

**For Vortex:** These variants are NOT needed - Vortex handles dynamic block sizes at runtime.

## Attempted Solutions

### ❌ Flag: `--use-original-gpu-block-size`
**Problem:** Causes cgeist to fail - this flag appears to be incompatible with `--emit-cuda`

### ❌ Flag: `--polygeist-alternatives-mode=static`
**Problem:** Doesn't prevent variant generation, just changes runtime selection strategy

### ❌ MLIR Pass to Remove Variants
**Problem:** No built-in pass exists to remove alternatives

## ✅ Recommended Solution: Ignore Variants in ConvertGPUToVortex

**Approach:** Process only the variants that are actually used by `gpu.launch_func` operations.

### Implementation Strategy

The Polygeist output has this structure:

```mlir
gpu.module @__polygeist_gpu_module {
  // 6 kernel variants
  gpu.func @kernel_variant1(...) kernel attributes {gpu.known_block_size = array<i32: 32, 1, 1>}
  gpu.func @kernel_variant2(...) kernel attributes {gpu.known_block_size = array<i32: 64, 1, 1>}
  // ... 4 more variants ...
}

func.func @launch_function(...) {
  // Runtime selection of variant
  scf.if %cond1 {
    gpu.launch_func @__polygeist_gpu_module::@kernel_variant1 ...
  } else {
    scf.if %cond2 {
      gpu.launch_func @__polygeist_gpu_module::@kernel_variant2 ...
    }
    // ... etc
  }
}
```

### Option 1: Process Only Called Variants (Recommended)

In `ConvertGPUToVortex.cpp`:

```cpp
// Step 1: Find all gpu.launch_func operations
SmallVector<gpu.LaunchFuncOp> launchOps;
module.walk([&](gpu.LaunchFuncOp op) {
  launchOps.push_back(op);
});

// Step 2: Get list of actually-called kernel names
DenseSet<StringRef> calledKernels;
for (auto launchOp : launchOps) {
  calledKernels.insert(launchOp.getKernelName());
}

// Step 3: Process only kernels that are actually called
for (auto gpuModule : module.getOps<gpu.GPUModuleOp>()) {
  for (auto gpuFunc : gpuModule.getOps<gpu.GPUFuncOp>()) {
    if (!gpuFunc->hasAttr("kernel")) continue;

    StringRef funcName = gpuFunc.getName();
    if (calledKernels.contains(funcName)) {
      // Convert this kernel
      convertKernelToVortex(gpuFunc);
    }
    // Skip uncalled variants
  }
}
```

**Result:** Only processes kernels that are actually launched, typically 1 per source kernel.

**Note:** This works correctly even with multiple source kernels, as each kernel's variants will have unique names.

### Option 2: Process First Variant Only (Simpler - For Single Kernel Files)

```cpp
// WARNING: This simplified approach assumes single kernel per file
// For multiple kernels, need to track per-kernel basis (see Option 1 or Option 3)

bool foundKernel = false;
for (auto gpuModule : module.getOps<gpu.GPUModuleOp>()) {
  for (auto gpuFunc : gpuModule.getOps<gpu.GPUFuncOp>()) {
    if (!gpuFunc->hasAttr("kernel")) continue;

    if (!foundKernel) {
      // Process first kernel variant only
      convertKernelToVortex(gpuFunc);
      foundKernel = true;
      break;  // Skip remaining variants
    }
  }
}
```

**Limitation:** If source has multiple kernels (e.g., `kernel_a` and `kernel_b`), this would only process `kernel_a` and skip `kernel_b` entirely.

**TODO:** Add robust tracking for multiple kernels (see Option 3).

### Option 3: Process First Variant Per Kernel (Better for Multiple Kernels)

```cpp
// For each unique kernel base name, process only the first variant
StringMap<bool> processedKernels;

for (auto gpuModule : module.getOps<gpu.GPUModuleOp>()) {
  for (auto gpuFunc : gpuModule.getOps<gpu.GPUFuncOp>()) {
    if (!gpuFunc->hasAttr("kernel")) continue;

    // Extract base kernel name (remove variant suffix like _kernel94...)
    StringRef funcName = gpuFunc.getName();
    StringRef baseName = extractBaseName(funcName);

    if (processedKernels.contains(baseName)) {
      // Already processed a variant of this kernel, skip
      continue;
    }

    // Process first variant of this kernel
    convertKernelToVortex(gpuFunc);
    processedKernels[baseName] = true;
  }
}

// Helper function to extract base kernel name
StringRef extractBaseName(StringRef mangledName) {
  // Polygeist adds suffix like "_kernel94565344022848"
  // Strip everything after "_kernel" pattern
  size_t pos = mangledName.find("_kernel");
  if (pos != StringRef::npos) {
    return mangledName.substr(0, pos);
  }
  return mangledName;
}
```

**Result:** Processes exactly one variant per source kernel, works correctly with multiple kernels in same file.

**Note:** This is more robust than Option 2 for files with multiple kernels.

## Recommended for Phase 2B

**Use Option 1 (Process Only Called Variants)** - Most Robust

**Rationale:**
- Works correctly with single or multiple kernels
- Respects Polygeist's intent (only process what's actually used)
- Automatically handles edge cases
- Based on actual program semantics, not name heuristics

**Alternative: Use Option 3** - If you want compile-time filtering without runtime analysis

**Rationale:**
- Simpler than Option 1
- Still handles multiple kernels correctly
- Uses name-based heuristic (less robust but practical)

**Avoid Option 2** - Too simplistic for production use

## Multiple Kernels Example

**Source file with multiple kernels:**
```cpp
__global__ void kernel_a(int* data) { /* ... */ }
__global__ void kernel_b(float* data) { /* ... */ }

void launch_both(int* d_a, float* d_b) {
    kernel_a<<<1, 256>>>(d_a);
    kernel_b<<<1, 128>>>(d_b);
}
```

**Polygeist output:**
```mlir
gpu.module @__polygeist_gpu_module {
  // kernel_a variants (6 versions)
  gpu.func @_Z8kernel_aPi_kernel...848(...) kernel
  gpu.func @_Z8kernel_aPi_kernel...704(...) kernel
  // ... 4 more kernel_a variants

  // kernel_b variants (6 versions)
  gpu.func @_Z8kernel_bPf_kernel...123(...) kernel
  gpu.func @_Z8kernel_bPf_kernel...456(...) kernel
  // ... 4 more kernel_b variants
}
```

**With Option 1:** Processes 2 kernels (one variant of kernel_a, one variant of kernel_b)
**With Option 2:** Processes 1 kernel (only kernel_a, misses kernel_b entirely) ❌
**With Option 3:** Processes 2 kernels (first variant of each kernel) ✅

## Testing

To verify correct handling with multiple kernels:

```bash
# Create test with 2 kernels
cat > test_multi.hip << 'EOF'
#include "hip_runtime_vortex/hip_runtime.h"

__global__ void kernel_a(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= 2;
}

__global__ void kernel_b(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= 3.0f;
}

void launch_both(int* d_a, float* d_b, int n) {
    kernel_a<<<(n+255)/256, 256>>>(d_a, n);
    kernel_b<<<(n+255)/256, 256>>>(d_b, n);
}
EOF

# Generate MLIR
./scripts/polygeist/hip-to-gpu-dialect.sh test_multi.hip test_multi.mlir

# Count total variants
grep -c "gpu.func.*kernel" test_multi.mlir
# Expected: 12 (6 variants × 2 kernels)

# Run ConvertGPUToVortex pass
mlir-opt test_multi.mlir --convert-gpu-to-vortex -o vortex_multi.mlir

# Count Vortex kernels after conversion
grep -c "vortex.kernel" vortex_multi.mlir
# Expected: 2 (one variant per source kernel)
```

## Implementation TODO

**Phase 2B Initial Implementation:**
- Use Option 1 (process only called variants)
- Add assertion to catch if we miss any kernel
- Log which variants are being processed vs skipped

**Future Enhancement:**
- Add flag to control behavior (process all, first-only, called-only)
- Add metrics to track variant usage
- Consider variant specialization if performance benefits

**Robust Checks to Add:**
```cpp
// Validation: Ensure we don't miss any distinct source kernels
void validateAllKernelsCovered(ModuleOp module) {
  // 1. Count unique kernel base names in GPU module
  StringSet<> uniqueKernelBases;
  for (auto gpuModule : module.getOps<gpu.GPUModuleOp>()) {
    for (auto gpuFunc : gpuModule.getOps<gpu.GPUFuncOp>()) {
      if (!gpuFunc->hasAttr("kernel")) continue;
      uniqueKernelBases.insert(extractBaseName(gpuFunc.getName()));
    }
  }

  // 2. Count unique kernels actually converted
  StringSet<> convertedKernels;
  for (auto vortexKernel : module.getOps<vortex.KernelOp>()) {
    convertedKernels.insert(extractBaseName(vortexKernel.getName()));
  }

  // 3. Assert they match
  if (uniqueKernelBases.size() != convertedKernels.size()) {
    llvm::errs() << "Warning: Found " << uniqueKernelBases.size()
                 << " unique source kernels but only converted "
                 << convertedKernels.size() << " kernels\n";
    // List missing kernels for debugging
  }
}
```

## Summary

✅ **Don't try to prevent Polygeist from generating variants** - it's deeply integrated
✅ **Instead: Filter variants in ConvertGPUToVortex pass**
✅ **Use Option 1 (called variants) or Option 3 (first per kernel)** - both handle multiple kernels
❌ **Don't use Option 2 (first variant only)** - breaks with multiple kernels
✅ **Add validation checks** - ensure no kernels are accidentally skipped
✅ **All variants have same logic** - just different optimizations
✅ **Vortex handles dynamic sizes** - doesn't need compile-time specialization

**For Phase 2B:** Start with Option 1, add validation, extend later if needed.

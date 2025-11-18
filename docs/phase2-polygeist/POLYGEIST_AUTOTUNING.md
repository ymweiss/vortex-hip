# Polygeist Auto-Tuning Feature

## Observation

Polygeist generates **multiple specialized versions** of each kernel with different block sizes in the GPU dialect output.

## Example from basic_kernel.mlir

```mlir
gpu.module @__polygeist_gpu_module {
  // Version 1: 32 threads
  gpu.func @...kernel94565344022848(...) kernel
    attributes {gpu.known_block_size = array<i32: 32, 1, 1>}

  // Version 2: 64 threads
  gpu.func @...kernel94565344022704(...) kernel
    attributes {gpu.known_block_size = array<i32: 64, 1, 1>}

  // Version 3: 128 threads
  gpu.func @...kernel94565343835264(...) kernel
    attributes {gpu.known_block_size = array<i32: 128, 1, 1>}

  // Versions 4-6: 256, 512, 1024 threads
  // ...
}
```

## Launch Function Selection

The host launch function uses control flow to select the appropriate kernel variant:

```mlir
func.func @_Z12launch_basicPiS_ji(...) {
  // Runtime checks for block size
  %cond = arith.cmpi eq, %block_size, %c32
  scf.if %cond {
    gpu.launch_func @...kernel..._32threads ...
  } else {
    %cond2 = arith.cmpi eq, %block_size, %c64
    scf.if %cond2 {
      gpu.launch_func @...kernel..._64threads ...
    } // ... etc
  }
}
```

## Alternatives Metadata

Polygeist adds metadata describing the variants:

```mlir
alternatives.descs = [
  "block_size=32,blockDims=x:32;y:1;z:1;,intOps=4:64;8:64;,...",
  "block_size=64,blockDims=x:64;y:1;z:1;,intOps=4:128;8:128;,...",
  "block_size=128,...",
  "block_size=256,...",
  "block_size=512,...",
  "block_size=1024,..."
]
```

This includes:
- Block dimensions
- Operation counts (int ops, float ops)
- Memory access patterns (loads, stores)

## Why Polygeist Does This

**Performance optimization through specialization:**
- Each block size may benefit from different code generation strategies
- Compile-time known block sizes enable better optimization
- Allows LLVM to unroll loops, eliminate conditionals, etc.

**Common GPU optimization technique:**
- Similar to CUDA's `__launch_bounds__` attribute
- Enables better register allocation
- Improves occupancy calculation

## Impact on Vortex ConvertGPUToVortex Pass

### Option 1: Process All Variants
**Approach:** Lower all kernel variants to Vortex IR
```cpp
// In ConvertGPUToVortex.cpp
for (auto gpuFunc : gpuModule.getOps<gpu.GPUFuncOp>()) {
  if (!gpuFunc->getAttr("kernel")) continue;
  // Convert each variant
  convertKernelToVortex(gpuFunc);
}
```

**Pros:**
- Preserves optimization opportunities
- Runtime can select best variant

**Cons:**
- More code generation
- Larger binary size

### Option 2: Select Single Variant
**Approach:** Choose one variant (e.g., 256 threads - common default)
```cpp
// Select variant with specific block size
for (auto gpuFunc : gpuModule.getOps<gpu.GPUFuncOp>()) {
  auto blockSize = gpuFunc->getAttrOfType<ArrayAttr>("gpu.known_block_size");
  if (blockSize && blockSize[0].cast<IntegerAttr>().getInt() == 256) {
    // Convert only this variant
    convertKernelToVortex(gpuFunc);
  }
}
```

**Pros:**
- Simpler
- Smaller binary

**Cons:**
- Less flexible
- May not be optimal for all use cases

### Option 3: MLIR Pass to Simplify
**Approach:** Add pre-processing pass to remove alternatives
```bash
mlir-opt input.mlir \
  --gpu-kernel-outlining \
  --convert-alternatives-to-single-kernel \
  -o simplified.mlir
```

Then convert simplified IR.

**Pros:**
- Keeps conversion pass simple
- Reusable for other projects

**Cons:**
- Extra pass to maintain

## Recommendation for Vortex

**Start with Option 2 (Select Single Variant):**

1. **For Phase 2B metadata extraction:**
   - Focus on one variant to prove the concept
   - Choose 256 threads (common GPU default)
   - Simpler to debug and verify

2. **For future optimization:**
   - Can extend to support multiple variants
   - Vortex runtime could select based on workload
   - Or compile multiple variants and let linker choose

## Implementation Strategy

### Current (Phase 2B):
```cpp
// In LaunchFuncOpLowering::matchAndRewrite()
void matchAndRewrite(gpu.LaunchFuncOp launchOp, ...) {
  auto callee = launchOp.getKernel();
  // Get the kernel function
  auto gpuFunc = symbolTable.lookup<gpu.GPUFuncOp>(callee);

  // For now, just process whatever variant is called
  // Don't worry about which variant - just extract metadata
  extractMetadata(gpuFunc);
}
```

### Future Enhancement:
```cpp
// Could add variant selection logic
auto blockSize = getBlockSizeFromLaunch(launchOp);
auto preferredVariant = selectVariant(blockSize);
extractMetadata(preferredVariant);
```

## Example: Selecting 256-thread Variant

```cpp
// In ConvertGPUToVortex.cpp
LogicalResult processGPUModule(gpu.GPUModuleOp gpuModule) {
  for (auto gpuFunc : gpuModule.getOps<gpu.GPUFuncOp>()) {
    // Check if this is a kernel
    if (!gpuFunc->hasAttr("kernel")) continue;

    // Get block size attribute
    auto blockSizeAttr = gpuFunc->getAttrOfType<ArrayAttr>("gpu.known_block_size");
    if (!blockSizeAttr) {
      // No known block size - process anyway
      convertKernel(gpuFunc);
      continue;
    }

    // Extract x dimension of block size
    int xBlockSize = blockSizeAttr[0].cast<IntegerAttr>().getInt();

    // Only process 256-thread variant (or first variant if none match)
    if (xBlockSize == 256 || /* is first variant */) {
      convertKernel(gpuFunc);
    }
  }
}
```

## Testing Approach

**Verify with different variants:**
```bash
# Extract just the 32-thread variant
mlir-opt basic_kernel.mlir --gpu-extract-variant="block-size=32" -o variant_32.mlir

# Extract just the 256-thread variant
mlir-opt basic_kernel.mlir --gpu-extract-variant="block-size=256" -o variant_256.mlir

# Test ConvertGPUToVortex on each
mlir-opt variant_256.mlir --convert-gpu-to-vortex -o vortex_256.mlir
```

## Conclusion

Polygeist's auto-tuning is a **feature, not a bug**. It generates multiple optimized variants of each kernel.

**For Phase 2B:**
- Start simple: process whichever variant is called by `gpu.launch_func`
- Don't special-case variant selection yet
- Focus on metadata extraction mechanics

**For Future:**
- Can add variant selection logic
- Or support all variants in Vortex runtime
- Or pre-process MLIR to remove alternatives

**Bottom line:** This doesn't block Phase 2B progress. The metadata extraction logic is the same regardless of which variant is processed.

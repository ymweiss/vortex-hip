# Phase 2B: Metadata Extraction Specification

## Test Case: metadata_test.mlir

### Source Kernel
```cpp
__global__ void metadata_kernel(
    int32_t* input,      // Pointer arg: 4 bytes (RV32 address)
    int32_t* output,     // Pointer arg: 4 bytes (RV32 address)
    int32_t count,       // Scalar arg: 4 bytes (i32)
    float scale          // Scalar arg: 4 bytes (f32)
)
```

### GPU Dialect IR (Line 3)
```mlir
gpu.func @_Z22launch_metadata_kernelPiS_if_kernel94738444747728(
  %arg0: i32,              // count (scalar)
  %arg1: memref<?xi32>,    // input (pointer)
  %arg2: f32,              // scale (scalar)
  %arg3: memref<?xi32>     // output (pointer)
) kernel
```

### Launch Function (Line 48)
```mlir
gpu.launch_func @__polygeist_gpu_module::@_Z22launch_metadata_kernelPiS_if_kernel94738444747728
  blocks in (%2, %c8, %c1)
  threads in (%c32, %c1, %c1)
  args(%arg2 : i32, %arg0 : memref<?xi32>, %arg3 : f32, %arg1 : memref<?xi32>)
```

## Metadata to Extract

### 1. Argument Information

| Position | Type | Category | Size | Notes |
|----------|------|----------|------|-------|
| 0 | i32 | Scalar | 4 bytes | count parameter |
| 1 | memref<?xi32> | Pointer | 4 bytes | input pointer (RV32 address) |
| 2 | f32 | Scalar | 4 bytes | scale parameter |
| 3 | memref<?xi32> | Pointer | 4 bytes | output pointer (RV32 address) |

**Total struct size:** 16 bytes (4 + 4 + 4 + 4)

### 2. Vortex Kernel Argument Struct

Expected layout matching Vortex's `kernel_arg_t` pattern (RV32 - 4-byte pointers):

```c
typedef struct {
  int32_t count;      // Offset 0, size 4
  uint32_t input;     // Offset 4, size 4 (RV32 pointer)
  float scale;        // Offset 8, size 4
  uint32_t output;    // Offset 12, size 4 (RV32 pointer)
} metadata_kernel_args_t;
// Total: 16 bytes (no padding needed - all fields are 4 bytes)
```

**Note:** For RV32 (32-bit RISC-V), pointers are 4 bytes. The struct layout is naturally aligned without requiring padding since all fields are 4 bytes.

### 3. Metadata Representation in MLIR

**Option A: Function Attribute**
```mlir
vortex.kernel @metadata_kernel(...) {
  vortex.metadata = {
    arg_offsets = [0, 4, 8, 12],
    arg_sizes = [4, 4, 4, 4],
    arg_types = ["scalar", "pointer", "scalar", "pointer"],
    total_size = 16
  }
}
```

**Option B: Separate Metadata Op**
```mlir
vortex.kernel_metadata @metadata_kernel {
  vortex.arg_layout {
    vortex.arg index=0, offset=0, size=4, type="scalar"
    vortex.arg index=1, offset=4, size=4, type="pointer"
    vortex.arg index=2, offset=8, size=4, type="scalar"
    vortex.arg index=3, offset=12, size=4, type="pointer"
  }
  vortex.total_size = 16
}
```

**Option C: Module-Level Global (Recommended)**
```mlir
// At module level
vortex.kernel_args @metadata_kernel_args : !vortex.kernel_args<
  struct_size = 16,
  args = [
    <index=0, offset=0, size=4, type=i32>,
    <index=1, offset=4, size=4, type=!llvm.ptr>,
    <index=2, offset=8, size=4, type=f32>,
    <index=3, offset=12, size=4, type=!llvm.ptr>
  ]
>

// Kernel references it
vortex.kernel @metadata_kernel(...) {
  vortex.kernel_args = @metadata_kernel_args
}
```

## Extraction Algorithm

### Step 1: Parse gpu.launch_func Operation
```cpp
void extractMetadata(gpu::LaunchFuncOp launchOp) {
  // Get kernel name
  StringRef kernelName = launchOp.getKernelName();

  // Get kernel function
  auto gpuModule = /* find gpu.module */;
  auto gpuFunc = gpuModule.lookupSymbol<gpu::GPUFuncOp>(kernelName);

  // Get arguments from launch
  auto launchArgs = launchOp.getKernelOperands();

  // Process each argument
  for (auto [idx, arg] : llvm::enumerate(launchArgs)) {
    Type argType = arg.getType();
    processArgument(idx, argType);
  }
}
```

### Step 2: Determine Argument Properties
```cpp
struct ArgMetadata {
  size_t index;
  size_t offset;
  size_t size;
  bool isPointer;
  Type mlirType;
};

ArgMetadata analyzeArgument(size_t index, Type type) {
  ArgMetadata meta;
  meta.index = index;

  if (auto memrefType = type.dyn_cast<MemRefType>()) {
    // Pointer argument (RV32 uses 4-byte pointers)
    meta.isPointer = true;
    meta.size = 4;  // 32-bit pointer for RV32
  } else if (type.isIntOrFloat()) {
    // Scalar argument
    meta.isPointer = false;
    meta.size = type.getIntOrFloatBitWidth() / 8;
  }

  return meta;
}
```

### Step 3: Calculate Offsets with Alignment
```cpp
std::vector<ArgMetadata> calculateLayout(std::vector<ArgMetadata> args) {
  size_t currentOffset = 0;

  for (auto& arg : args) {
    // Align to natural alignment
    size_t alignment = arg.size;
    if (alignment > 4) alignment = 4;  // Max 4-byte alignment for RV32

    // Add padding if needed
    size_t remainder = currentOffset % alignment;
    if (remainder != 0) {
      currentOffset += alignment - remainder;
    }

    arg.offset = currentOffset;
    currentOffset += arg.size;
  }

  // Align total size to 4 bytes (word-aligned for RV32)
  if (currentOffset % 4 != 0) {
    currentOffset += 4 - (currentOffset % 4);
  }

  return args;
}
```

### Step 4: Generate Metadata
```cpp
void emitMetadata(std::vector<ArgMetadata> layout, StringRef kernelName) {
  // Create module-level metadata operation
  auto metadataOp = builder.create<vortex::KernelArgsOp>(
    loc, kernelName + "_args");

  for (const auto& arg : layout) {
    builder.create<vortex::ArgOp>(
      metadataOp.getLoc(),
      arg.index,
      arg.offset,
      arg.size,
      arg.isPointer ? "pointer" : "scalar",
      arg.mlirType
    );
  }

  metadataOp.setTotalSize(layout.back().offset + layout.back().size);
}
```

## Expected Output for Test Case

### Metadata Structure
```
Kernel: metadata_kernel
Total size: 16 bytes (RV32)

Arguments:
  [0] count   : offset=0,  size=4, type=scalar (i32)
  [1] input   : offset=4,  size=4, type=pointer (memref<?xi32>)
  [2] scale   : offset=8,  size=4, type=scalar (f32)
  [3] output  : offset=12, size=4, type=pointer (memref<?xi32>)

Padding: None needed (all fields are 4 bytes, naturally aligned)
```

### Vortex Runtime Usage
```c
// Runtime prepares argument struct (RV32 - 4-byte pointers)
typedef struct {
  int32_t count;      // Value passed directly
  uint32_t input;     // Device pointer address (32-bit)
  float scale;        // Value passed directly
  uint32_t output;    // Device pointer address (32-bit)
} kernel_args;

kernel_args args = {
  .count = 1024,
  .input = input_device_addr,
  .scale = 2.5f,
  .output = output_device_addr
};

// Upload to device
uint32_t args_buffer;  // RV32 uses 32-bit addresses
vx_upload_bytes(device, &args, sizeof(args), &args_buffer);

// Launch kernel
vx_start(device, kernel_buffer, args_buffer);
```

## Validation Tests

### Test 1: Argument Count
```cpp
assert(extractedArgs.size() == 4);
```

### Test 2: Argument Types
```cpp
assert(extractedArgs[0].isPointer == false);  // count is scalar
assert(extractedArgs[1].isPointer == true);   // input is pointer
assert(extractedArgs[2].isPointer == false);  // scale is scalar
assert(extractedArgs[3].isPointer == true);   // output is pointer
```

### Test 3: Argument Sizes
```cpp
assert(extractedArgs[0].size == 4);  // i32
assert(extractedArgs[1].size == 4);  // ptr (RV32)
assert(extractedArgs[2].size == 4);  // f32
assert(extractedArgs[3].size == 4);  // ptr (RV32)
```

### Test 4: Offsets with Alignment
```cpp
assert(extractedArgs[0].offset == 0);   // count at offset 0
assert(extractedArgs[1].offset == 4);   // input at offset 4
assert(extractedArgs[2].offset == 8);   // scale at offset 8
assert(extractedArgs[3].offset == 12);  // output at offset 12
```

### Test 5: Total Size
```cpp
assert(totalStructSize == 16);  // 4 + 4 + 4 + 4 (no padding needed)
```

## Next Steps

1. **Implement extraction in ConvertGPUToVortex.cpp**
   - Add `LaunchFuncOpLowering` pattern
   - Implement argument analysis
   - Calculate offsets with alignment

2. **Define Vortex dialect ops**
   - `vortex.kernel_args` - Metadata container
   - `vortex.arg` - Individual argument metadata
   - Or use attributes on existing ops

3. **Test with metadata_test.mlir**
   - Run pass on test case
   - Verify extracted metadata
   - Compare with expected values

4. **Extend to other test cases**
   - basic_kernel.mlir (2 pointer args + 1 scalar)
   - vecadd_kernel.mlir (3 pointer args + 1 scalar)
   - dotproduct_kernel.mlir (3 pointer args + 1 scalar)

## Files Created

- `hip_tests/kernels/metadata_test.hip` - Test kernel source
- `hip_tests/mlir_output/metadata_test.mlir` - Generated GPU dialect
- `docs/phase2-polygeist/METADATA_EXTRACTION_SPEC.md` - This document

## Ready for Implementation

All prerequisites are in place:
- ✅ Clean GPU dialect with single kernel variant
- ✅ Clear argument types in `gpu.launch_func`
- ✅ Test case with mixed scalar/pointer arguments
- ✅ Specification for metadata structure
- ✅ Algorithm for extraction and layout calculation

Next: Implement in ConvertGPUToVortex pass.

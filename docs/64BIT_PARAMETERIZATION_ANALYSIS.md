# Polygeist 64-bit Parameterization Analysis

**Goal:** Parameterize all Vortex-related Polygeist passes to support both RV32 and RV64 targets.

**Key Constraint:** A 64-bit kernel cannot run on a 32-bit Vortex device. The target pointer width must be specified at compile time via a pass option.

---

## Pass Option Strategy

Each pass will receive a `pointerWidth` option (32 or 64) that determines:
- Pointer load/store types (i32 vs i64)
- Argument offset calculations (4-byte vs 8-byte stride for pointers)
- CSR read result interpretation
- Device address type width

This option will be set by the build system based on the Vortex target (`XLEN=32` or `XLEN=64`).

---

## Passes to Modify

| Pass | File | Status |
|------|------|--------|
| ConvertGPUToVortex | `ConvertGPUToVortex.cpp` | Needs 64-bit parameterization |
| GenerateVortexMain | `GenerateVortexMain.cpp` | Needs 64-bit parameterization |
| ConvertGPULaunchToHostCall | `ConvertGPULaunchToHostCall.cpp` | Needs 64-bit parameterization |
| InsertVortexDivergence | `InsertVortexDivergence.cpp` | **DEPRECATED** - Not used, remove |

**Note on InsertVortexDivergence:** Branch divergence is handled at the LLVM backend level via `--vortex-branch-divergence=1` flag passed to llc. The MLIR-level pass is never invoked in the compilation pipeline.

---

## File-by-File Analysis

### 1. GenerateVortexMain.cpp

**Purpose:** Generates the Vortex `main()` entry point and `kernel_body` wrapper that unpacks arguments from the args buffer.

| Line | Code | Issue | Fix |
|------|------|-------|-----|
| 133-138 | `auto i32Type = IntegerType::get(ctx, 32);` | vx_spawn_threads signature uses i32 | Keep i32 for dimension arg (always 32-bit); pointers stay as `ptr` |
| 219-220 | `// uint32_t grid_dim[3]; // 12 bytes` | Header layout comment assumes 32-bit | Update comments to note XLEN dependency |
| 229-230 | `constexpr BLOCK_DIM_OFFSET = 12; USER_ARGS_OFFSET = 24;` | Fixed offsets assume i32 grid/block dims | Grid/block dims remain i32 (per Vortex ABI), so these are correct |
| 309, 413 | `argOffset = USER_ARGS_OFFSET + hostIdx * 4;` | 4-byte stride for all args | **CRITICAL:** Pointers need XLEN/8 bytes, scalars vary |
| 321-322 | `LoadOp(loc, i32Type, argBytePtr); IntToPtrOp(loc, ptrType, rawPtr)` | Pointer load as i32 | **CRITICAL:** Use iXLEN type |
| 342 | `currentOffset += 4; // Single pointer` | 4-byte pointer assumption | **CRITICAL:** Use XLEN/8 |
| 425-428 | `LoadOp(loc, i32Type, argBytePtr); IntToPtrOp(loc, ptrType, rawPtr)` | Scalar pointer load as i32 | **CRITICAL:** Use iXLEN type |
| 496, 516-518 | `InlineAsmOp(... i32Type ...)` | CSR read returns i32 | **CRITICAL:** CSR read returns XLEN-bit value |
| 531 | `IntToPtrOp(loc, ptrType, argsRaw)` | Args pointer from CSR (i32) | **CRITICAL:** argsRaw must be iXLEN |

**Changes Required:**
```cpp
// Add pass option
unsigned pointerWidth = 32;  // or 64, from option

// Helper function
Type getXLENType(MLIRContext *ctx, unsigned xlen) {
  return IntegerType::get(ctx, xlen);
}

// Usage
auto ptrIntType = getXLENType(ctx, pointerWidth);
unsigned ptrSize = pointerWidth / 8;  // 4 or 8 bytes

// For pointer loads:
auto rawPtr = builder.create<LLVM::LoadOp>(loc, ptrIntType, argBytePtr);
auto devicePtr = builder.create<LLVM::IntToPtrOp>(loc, ptrType, rawPtr);
currentOffset += ptrSize;
```

---

### 2. ConvertGPUToVortex.cpp

**Purpose:** Lowers GPU dialect operations to LLVM with Vortex intrinsics.

| Line | Code | Issue | Fix |
|------|------|-------|-----|
| 220-222 | `auto i32Type = rewriter.getI32Type(); auto dim3Type = LLVM::LLVMStructType::getLiteral(context, {i32Type, i32Type, i32Type});` | dim3_t struct uses i32 | Correct - Vortex dim3 is always 32-bit |
| 235 | `rewriter.create<LLVM::LoadOp>(loc, i32Type, gep)` | Load dim3 field as i32 | Correct - dim3 fields are always 32-bit |
| 900-951 | Shared memory address calculations use i32 | Address arithmetic as i32 | **CRITICAL:** Use iXLEN for addresses |
| 911-923 | `csrr $0, VX_CSR_LOCAL_MEM_BASE` returns i32 | CSR read result type | **CRITICAL:** CSR returns XLEN-bit value |
| 940-951 | `MulOp/AddOp(loc, i32Type, ...)` | Address calculations | **CRITICAL:** Use iXLEN type |
| 954 | `IntToPtrOp(loc, ptrType, finalAddr)` | Convert address to pointer | Need iXLEN input |
| 987-989 | `numArgs * 4` for RV32 args size | 4-byte arg assumption | Pointers need XLEN/8 |
| 1058-1066 | `getTypeSizeRV32()` function | Hardcoded RV32 sizes | **CRITICAL:** Parameterize for XLEN |
| 1071 | `if (metaType == "ptr") return "uint32_t";` | Pointer C type | **CRITICAL:** Return `uint64_t` for RV64 |

**Changes Required:**
```cpp
// Add helper functions
unsigned getTypeSizeForXLEN(Type type, unsigned xlen) {
  if (type.isa<MemRefType>() || type.isa<LLVM::LLVMPointerType>())
    return xlen / 8;  // 4 or 8 bytes
  // ... rest of type checking
}

std::string getCTypeStringForXLEN(const std::string &metaType, unsigned xlen) {
  if (metaType == "ptr")
    return (xlen == 64) ? "uint64_t" : "uint32_t";
  // ... rest
}

// For shared memory:
Type addrType = getXLENType(context, pointerWidth);
auto csrRead = rewriter.create<LLVM::InlineAsmOp>(loc, addrType, ...);
// Use addrType for all address arithmetic
```

---

### 3. ConvertGPULaunchToHostCall.cpp

**Purpose:** Converts gpu.launch_func to host-side runtime calls.

| Line | Code | Issue | Fix |
|------|------|-------|-----|
| 51-52 | `getOrInsertFunc("hip_ptr_to_device_addr", i32Type, {ptrType})` | Device address as i32 | **CRITICAL:** Return type should be iXLEN |
| 115-117 | `headerSize = 6 * 4; argsSize = numArgs * 4;` | 4-byte args | Pointers need XLEN/8 |
| 141-175 | Kernel argument marshalling with 4-byte strides | All args as 4 bytes | **CRITICAL:** Pointers need XLEN/8 |
| 159-161 | `TruncOp(loc, i32Type, arg)` for i64 | Truncate to 32-bit | Keep for scalars, but not for pointers |
| 441-443 | Same calculations for buffer size | 4-byte stride | **CRITICAL:** Parameterize |

**Changes Required:**
```cpp
// Calculate correct size per argument
unsigned getArgDeviceSize(Type argType, unsigned xlen) {
  if (argType.isa<MemRefType>() || argType.isa<LLVM::LLVMPointerType>())
    return xlen / 8;
  if (argType.isInteger(64) || argType.isF64())
    return 8;
  return 4;  // i32, f32, index (converted to i32)
}

// For device address function:
auto deviceAddrType = getXLENType(context, pointerWidth);
getOrInsertFunc("hip_ptr_to_device_addr", deviceAddrType, {ptrType});

// For storing pointer args:
if (argType.isa<MemRefType>()) {
  // ... get device address ...
  storeAtOffset(argVal, argOffset, deviceAddrType);  // Use proper type
  argOffset += pointerWidth / 8;
}
```

---

### 4. InsertVortexDivergence.cpp - **TO BE REMOVED**

**Status:** This pass is **NOT USED** in the compilation pipeline.

**Reason:** Branch divergence is handled at the LLVM backend level via `--vortex-branch-divergence=1` flag passed to llc (line 489 of compile_hip_v2.sh).

**Action:** Remove this pass from the codebase.

---

## Summary of Critical Changes

| Category | Files Affected | Change Description |
|----------|---------------|-------------------|
| Pointer loads | GenerateVortexMain.cpp | Load pointers as iXLEN, not i32 |
| Pointer stores | ConvertGPULaunchToHostCall.cpp | Store device addresses as XLEN bytes |
| Arg offset calc | All 3 passes | Pointers take XLEN/8 bytes, not 4 |
| CSR read type | GenerateVortexMain.cpp, ConvertGPUToVortex.cpp | CSR values are XLEN-width |
| Address arith | ConvertGPUToVortex.cpp | Shared memory addresses use iXLEN |
| Device addr | ConvertGPULaunchToHostCall.cpp | hip_ptr_to_device_addr returns iXLEN |
| Metadata | ConvertGPUToVortex.cpp | Emit correct C types for target |

---

## Implementation Order

1. **Remove InsertVortexDivergence.cpp** - Unused pass
2. **Add pass option infrastructure** to Passes.td for XLEN (pointerWidth parameter)
3. **Update GenerateVortexMain.cpp** - Most critical, handles arg unpacking
4. **Update ConvertGPUToVortex.cpp** - Shared memory and metadata
5. **Update ConvertGPULaunchToHostCall.cpp** - Host-side arg marshalling
6. **Update compile script** to pass XLEN option to passes
7. **Test on RV64 simulator**

---

## Unchanged Items

The following use i32 intentionally and should NOT change:

| Item | Reason |
|------|--------|
| Grid/block dimensions | Vortex ABI uses 32-bit dim values |
| vx_spawn_threads dimension arg | Always 32-bit per Vortex API |
| Barrier ID | Always 32-bit |
| dim3_t struct | Vortex uses 32-bit dim3 |
| i32/f32 scalar args | Size is intrinsic to type |

---

## Host Runtime Notes

Per user clarification: "host-side should require no changes."

The host runtime (`vortex_hip_runtime.cpp`) runs on a 64-bit host and handles:
- Host pointer → device address conversion (already works for both)
- Argument buffer construction (packs based on metadata)

The metadata files (`.meta.json`) will specify device sizes, which differ between RV32 and RV64 for pointers.

---

## Pass Option Definition (Passes.td)

```tablegen
def GenerateVortexMain : Pass<"generate-vortex-main", "ModuleOp"> {
  let summary = "Generate Vortex main() entry point";
  let options = [
    Option<"pointerWidth", "pointer-width", "unsigned", "32",
           "Target pointer width in bits (32 or 64)">
  ];
}
```

Similar options added to:
- `ConvertGPUToVortex`
- `ConvertGPULaunchToHostCall`

---

## Compile Script Updates

Add parameter for 64-bit mode:

```bash
# At the top of compile_hip_v2.sh
XLEN="${XLEN:-32}"  # Default to 32-bit

# In convert_to_vortex():
run_cmd "$POLYGEIST_OPT" "$GPU_MLIR" \
    --convert-gpu-to-vortex="pointer-width=$XLEN" \
    --strip-host-only-functions \
    -o "$VORTEX_MLIR"

# In mlir_to_llvm():
run_cmd "$POLYGEIST_OPT" "$LLVM_DIALECT" \
    --generate-vortex-main="pointer-width=$XLEN" \
    -o "$MAIN_MLIR"

# In compile_kernel_binary():
if [ "$XLEN" = "64" ]; then
    MARCH_FLAG="-march=riscv64"
    MATTR_FLAG="-mattr=+m,+a,+f,+d,+vortex"
    TARGET="riscv64-unknown-unknown-elf"
else
    MARCH_FLAG="-march=riscv32"
    MATTR_FLAG="-mattr=+m,+a,+f,+vortex"
    TARGET="riscv32-unknown-unknown-elf"
fi
```

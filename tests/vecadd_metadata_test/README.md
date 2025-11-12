# Vector Addition Metadata Generation Test

## Overview

This test demonstrates the **complete Phase 1 metadata generation workflow** for HIP kernels on Vortex, requiring **zero compiler modifications**.

## What This Tests

✅ **Kernel Compilation** - Vortex kernel with debug info
✅ **Metadata Extraction** - Python script parses DWARF/ELF
✅ **Stub Generation** - Auto-generated registration code
✅ **HIP Runtime** - Metadata-driven argument marshaling
✅ **End-to-End** - Full workflow from source to execution

## Build Flow (No Compiler Changes!)

```
[1] kernel.cpp
     ↓ (Vortex Clang with -g flag)
[2] kernel.elf (contains DWARF debug info)
     ↓ (hip_metadata_gen.py - Phase 1)
[3] kernel_metadata.cpp (auto-generated)
     ↓ (g++)
[4] kernel_metadata.o

[2] kernel.elf
     ↓ (vxbin.py)
[5] kernel.vxbin (device binary)
     ↓ (ld -r -b binary)
[6] kernel_vxbin.o

[7] main.cpp
     ↓ (g++)
[8] main.o

[4] + [6] + [8]
     ↓ (g++ link with libhip_vortex.so)
[9] vecadd_test (final executable)
```

## Prerequisites

1. **Vortex built and configured**
   ```bash
   cd ~/vortex
   make -j
   ```

2. **Vortex HIP runtime built**
   ```bash
   cd ~/vortex_hip/runtime
   ./build.sh
   ```

3. **Environment variables set**
   ```bash
   export VORTEX_HOME=$HOME/vortex
   source $VORTEX_HOME/ci/toolchain_env.sh
   ```

## Building

```bash
# Show build configuration
make info

# Build everything
make

# Or build step-by-step to see each phase
make kernel.elf              # Compile kernel with debug info
make kernel_metadata.cpp     # Generate metadata (Phase 1)
make kernel.vxbin            # Convert to Vortex binary
make vecadd_test             # Link final application
```

## Running

```bash
# Run with default size (256 elements)
make run

# Or run directly with custom size
./vecadd_test 1024
```

## Expected Output

```
=== HIP Vector Addition with Metadata Test ===
Vector size: 256 elements

Initializing HIP device...
Device: Vortex

Allocating host memory...
Initializing input data...
Allocating device memory...
  d_a = 0x...
  d_b = 0x...
  d_c = 0x...

Launch configuration:
  Grid:  (4, 1, 1)
  Block: (64, 1, 1)

Preparing kernel arguments...
Arguments (HIP array-of-pointers style):
  args[0] = &d_a (pointer to float*, value=0x...)
  args[1] = &d_b (pointer to float*, value=0x...)
  args[2] = &d_c (pointer to float*, value=0x...)
  args[3] = &n   (pointer to uint32_t, value=256)

Expected metadata marshaling:
  Runtime will use metadata to pack arguments:
    arg[0]: size=8, align=8, pointer=1 -> copy 8 bytes from &d_a
    arg[1]: size=8, align=8, pointer=1 -> copy 8 bytes from &d_b
    arg[2]: size=8, align=8, pointer=1 -> copy 8 bytes from &d_c
    arg[3]: size=4, align=4, pointer=0 -> copy 4 bytes from &n

Launching kernel...
Waiting for kernel completion...
Copying results back to host...
Verifying results...
Cleaning up...

=== Test Results ===
✓ PASSED! All 256 elements computed correctly.

This confirms:
  ✓ Metadata was generated correctly
  ✓ Runtime marshaled arguments using metadata
  ✓ Kernel received properly packed arguments
  ✓ Computation completed successfully
```

## Inspecting Generated Metadata

```bash
# Generate and view the metadata stub
make kernel_metadata.cpp
make show-metadata
```

Expected metadata structure:
```cpp
// Auto-generated metadata for vecadd
static const hipKernelArgumentMetadata vecadd_metadata[] = {
    {.offset = 0,  .size = 8, .alignment = 8, .is_pointer = 1},  // float* a
    {.offset = 8,  .size = 8, .alignment = 8, .is_pointer = 1},  // float* b
    {.offset = 16, .size = 8, .alignment = 8, .is_pointer = 1},  // float* c
    {.offset = 24, .size = 4, .alignment = 4, .is_pointer = 0}   // uint32_t n
};
```

## Files

- **`kernel.cpp`** - HIP-style vector addition kernel for Vortex
  - Uses Vortex threading model (`blockIdx`, `threadIdx`)
  - Argument struct matches HIP runtime expectations
  - Device-side code

- **`main.cpp`** - HIP-style host application
  - Uses HIP API (`hipMalloc`, `hipMemcpy`, `hipLaunchKernel`)
  - Demonstrates array-of-pointers argument passing
  - Verifies results

- **`Makefile`** - Complete build system
  - Compiles kernel with `-g` debug info
  - Runs Python metadata generator
  - Links everything together

- **`kernel_metadata.cpp`** - **Auto-generated** (do not edit!)
  - Created by `hip_metadata_gen.py`
  - Contains metadata array
  - Registration function

## Key Features

### 1. No Compiler Modifications

This test uses the **existing Vortex compiler** with just the `-g` flag to include debug info. The metadata is extracted post-compilation using a Python script.

### 2. Real Metadata Extraction

The Python script actually parses the ELF file structure (currently using heuristics, will parse DWARF in production).

### 3. HIP API Compatibility

The host code uses standard HIP APIs, demonstrating that the runtime correctly translates HIP calls to Vortex operations using metadata.

### 4. Argument Marshaling Validation

The test explicitly shows:
- HIP's array-of-pointers input format
- Expected metadata-driven marshaling
- Vortex's packed struct format
- Correct execution results

## Debugging

### View kernel disassembly
```bash
make kernel.dump
less kernel.dump
```

### Check symbol table
```bash
nm -C kernel.elf
```

### Inspect ELF structure
```bash
readelf -a kernel.elf
```

### Debug DWARF info
```bash
llvm-dwarfdump --debug-info kernel.elf | less
```

### Verbose build
```bash
make clean
make -n  # Dry run to see commands
```

## Troubleshooting

### "Metadata generator not found"

**Solution:**
```bash
ls $VORTEX_HOME/scripts/hip_metadata_gen.py
# If not found, check installation
```

### "LLVM_VORTEX not set"

**Solution:**
```bash
source $VORTEX_HOME/ci/toolchain_env.sh
```

### "libhip_vortex.so not found"

**Solution:**
```bash
cd $VORTEX_HIP_HOME/runtime
./build.sh
```

### Kernel runs but produces wrong results

This likely indicates a marshaling issue:
1. Check generated metadata: `make show-metadata`
2. Verify argument sizes match expectations
3. Check if debug info was included: `objdump -g kernel.elf`

## Next Steps

After validating Phase 1:

1. **Test with more complex kernels**
   - Multiple argument types
   - Struct-by-value arguments
   - Mixed pointer/scalar arguments

2. **Improve metadata extraction**
   - Full DWARF parsing
   - Better type detection
   - Struct layout handling

3. **Begin Phase 2**
   - Design LLVM pass
   - Integrate with compiler
   - Automatic metadata generation

## Related Documentation

- `$VORTEX_HOME/scripts/README_metadata_gen.md` - Metadata generator docs
- `$VORTEX_HIP_HOME/runtime/ARGUMENT_MARSHALING.md` - Runtime implementation
- `$VORTEX_HIP_HOME/docs/implementation/COMPILER_METADATA_GENERATION.md` - Full strategy

---

**Status:** Phase 1 Complete - Ready for Testing
**Last Updated:** 2025-11-06

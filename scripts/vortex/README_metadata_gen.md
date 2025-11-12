# HIP Kernel Metadata Generator

## Overview

`hip_metadata_gen.py` is a Python script that extracts kernel argument information from compiled ELF files and generates C++ registration code with metadata. This is **Phase 1** of the metadata generation strategy - a prototype for testing before implementing the full LLVM pass.

## Purpose

The script bridges the gap between HIP's array-of-pointers argument model and Vortex's struct-based model by generating metadata that tells the runtime:
- Size of each argument
- Alignment requirements
- Whether it's a pointer or scalar

## Usage

### Basic Usage

```bash
# Generate metadata stub from compiled kernel
python3 hip_metadata_gen.py kernel.elf > kernel_stub.cpp

# Compile the stub
g++ -c kernel_stub.cpp -I$VORTEX_HIP_HOME/runtime/include

# Link with your application
g++ main.o kernel_stub.o -lhip_vortex
```

### Options

- `--arch=rv32` - Force 32-bit RISC-V metadata (auto-detected by default)
- `--arch=rv64` - Force 64-bit RISC-V metadata
- `--makefile` - Generate Makefile snippet instead of C++ code

### Example

```bash
# Compile kernel with debug info
clang++ -target riscv64 -g vectorAdd.cpp -o vectorAdd.elf

# Generate metadata
python3 hip_metadata_gen.py vectorAdd.elf > vectorAdd_stub.cpp

# Expected output:
# // Auto-generated metadata for vectorAdd
# #include "vortex_hip_runtime.h"
#
# static const hipKernelArgumentMetadata vectorAdd_metadata[] = {
#     {.offset = 0,  .size = 8, .alignment = 8, .is_pointer = 1},
#     {.offset = 8,  .size = 8, .alignment = 8, .is_pointer = 1},
#     {.offset = 16, .size = 8, .alignment = 8, .is_pointer = 1},
#     {.offset = 24, .size = 4, .alignment = 4, .is_pointer = 0}
# };
```

## Generated Output Format

The script generates:

1. **Metadata Array** - Argument layout information
2. **Registration Function** - Called at program startup
3. **Launcher Wrapper** - Optional convenience function

Example output structure:

```cpp
// Auto-generated metadata for vectorAdd
#include "vortex_hip_runtime.h"

extern const uint8_t vectorAdd_vxbin[];
extern const size_t vectorAdd_vxbin_size;

static void* vectorAdd_handle = nullptr;

static const hipKernelArgumentMetadata vectorAdd_metadata[] = {
    {.offset = 0,  .size = 8, .alignment = 8, .is_pointer = 1},  // float* a
    {.offset = 8,  .size = 8, .alignment = 8, .is_pointer = 1},  // float* b
    {.offset = 16, .size = 8, .alignment = 8, .is_pointer = 1},  // float* c
    {.offset = 24, .size = 4, .alignment = 4, .is_pointer = 0}   // int n
};

__attribute__((constructor))
static void register_vectorAdd() {
    hipError_t err = __hipRegisterFunctionWithMetadata(
        &vectorAdd_handle,
        "vectorAdd",
        vectorAdd_vxbin,
        vectorAdd_vxbin_size,
        4,
        vectorAdd_metadata
    );

    if (err != hipSuccess) {
        fprintf(stderr, "Failed to register kernel vectorAdd: %s\n",
                hipGetErrorString(err));
    }
}
```

## Build System Integration

### Makefile Example

```makefile
# Define paths
VORTEX_HOME ?= $(HOME)/vortex
VORTEX_HIP_HOME ?= $(HOME)/vortex_hip
METADATA_GEN = $(VORTEX_HOME)/scripts/hip_metadata_gen.py

# Compile kernel to ELF
kernel.elf: kernel.cpp
	$(RISCV_CXX) -target riscv64 -g $< -o $@

# Generate metadata stub
kernel_stub.cpp: kernel.elf
	python3 $(METADATA_GEN) $< > $@

# Compile stub
kernel_stub.o: kernel_stub.cpp
	$(CXX) -c $< -o $@ -I$(VORTEX_HIP_HOME)/runtime/include

# Link application
app: main.o kernel_stub.o kernel.vxbin
	$(CXX) main.o kernel_stub.o -o $@ -lhip_vortex
```

### CMake Example

```cmake
# Find metadata generator
find_program(METADATA_GEN hip_metadata_gen.py
    PATHS ${VORTEX_HOME}/scripts
    REQUIRED
)

# Custom command to generate metadata
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kernel_stub.cpp
    COMMAND ${PYTHON_EXECUTABLE} ${METADATA_GEN}
            ${CMAKE_CURRENT_BINARY_DIR}/kernel.elf
            > ${CMAKE_CURRENT_BINARY_DIR}/kernel_stub.cpp
    DEPENDS kernel.elf
    COMMENT "Generating kernel metadata"
)

# Add stub to application
add_executable(app
    main.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/kernel_stub.cpp
)

target_link_libraries(app hip_vortex)
```

## Current Limitations (Phase 1)

This is a **prototype** implementation with the following limitations:

1. **Limited Type Detection**
   - Uses heuristics and symbol table instead of full DWARF parsing
   - May not detect all kernel functions automatically
   - Assumes common patterns (vectorAdd, saxpy, etc.)

2. **Simplified Type System**
   - Only handles basic types (int, float, double, pointers)
   - Struct-by-value requires manual handling
   - No support for complex types yet

3. **Manual Binary Linking**
   - Generated stub expects `kernel_vxbin[]` and `kernel_vxbin_size` symbols
   - Must manually link kernel binary (converted by vxbin.py)

4. **Debug Info Required**
   - Must compile with `-g` flag for DWARF info
   - Without debug info, falls back to heuristics

## Testing

Run the test suite:

```bash
cd $VORTEX_HIP_HOME/tests/test_metadata_gen
./test_script.sh
```

Test with manual metadata:

```bash
cd $VORTEX_HIP_HOME/tests/test_metadata_gen
g++ test_manual_metadata.cpp \
    -I$VORTEX_HIP_HOME/runtime/include \
    -L$VORTEX_HIP_HOME/runtime/build \
    -lhip_vortex \
    -Wl,-rpath,$VORTEX_HIP_HOME/runtime/build \
    -o test_manual_metadata

./test_manual_metadata
```

Expected output:
```
=== Manual Metadata Test ===
Expected metadata for vectorAdd(float* a, float* b, float* c, int n):
  arg[0]: offset=0,  size=8, align=8, pointer=1 (float* a)
  arg[1]: offset=8,  size=8, align=8, pointer=1 (float* b)
  arg[2]: offset=16, size=8, align=8, pointer=1 (float* c)
  arg[3]: offset=24, size=4, align=4, pointer=0 (int n)

✓ Kernel vectorAdd registered successfully with metadata
```

## Type Size Reference

### RV32 (32-bit RISC-V)

| Type       | Size | Alignment | is_pointer |
|------------|------|-----------|------------|
| char       | 1    | 1         | 0          |
| short      | 2    | 2         | 0          |
| int        | 4    | 4         | 0          |
| long       | 4    | 4         | 0          |
| long long  | 8    | 8         | 0          |
| float      | 4    | 4         | 0          |
| double     | 8    | 8         | 0          |
| pointer    | 4    | 4         | 1          |

### RV64 (64-bit RISC-V)

| Type       | Size | Alignment | is_pointer |
|------------|------|-----------|------------|
| char       | 1    | 1         | 0          |
| short      | 2    | 2         | 0          |
| int        | 4    | 4         | 0          |
| long       | 8    | 8         | 0          |
| long long  | 8    | 8         | 0          |
| float      | 4    | 4         | 0          |
| double     | 8    | 8         | 0          |
| pointer    | 8    | 8         | 1          |

## Future Work (Phase 2: LLVM Pass)

The ultimate goal is a proper LLVM pass that:
- Has access to full type information
- Integrates directly with compilation
- Handles all C++ types correctly
- Generates metadata automatically

See `docs/implementation/COMPILER_METADATA_GENERATION.md` for the full roadmap.

## Troubleshooting

### "No HIP kernels detected"

**Cause:** Script couldn't find kernel functions in the ELF

**Solutions:**
1. Compile with debug info: `-g` flag
2. Check that functions are not stripped
3. Verify function names match expected patterns
4. Use `nm -C kernel.elf` to list symbols

### "File not found" error

**Cause:** Invalid ELF path

**Solution:** Use absolute path or verify file exists

### Metadata doesn't match actual arguments

**Cause:** Phase 1 uses heuristics, not actual type analysis

**Solution:** This is expected - Phase 1 is for testing. For production, use Phase 2 (LLVM pass) when available, or manually verify/edit the generated metadata.

## Related Documentation

- `$VORTEX_HIP_HOME/runtime/ARGUMENT_MARSHALING.md` - Runtime implementation
- `$VORTEX_HIP_HOME/docs/implementation/COMPILER_METADATA_GENERATION.md` - Full strategy
- `$VORTEX_HIP_HOME/tests/test_metadata_gen/` - Test examples

## Status

✅ **Phase 1 Complete** - Prototype working and tested

**Next Steps:**
1. Test with real compiled Vortex kernels
2. Integrate with existing Vortex build system
3. Begin Phase 2 (LLVM pass) implementation

---

**Last Updated:** 2025-11-06
**Status:** Phase 1 Prototype Complete

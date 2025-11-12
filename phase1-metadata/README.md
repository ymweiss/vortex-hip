# Phase 1: Metadata Generation

**Status:** ✅ COMPLETE (40/40 tests passing)

## Overview

This phase implements automatic metadata generation for HIP kernel arguments by parsing DWARF debug information. The metadata describes argument layout (offsets, sizes, alignments, pointer flags) needed to marshal HIP's array-of-pointers calling convention to Vortex's packed struct format.

## Components

### Python Script
- **Location:** Integrated with Vortex at `vortex/scripts/hip_metadata_gen.py`
- **Purpose:** Extract kernel signatures from ELF files and generate C++ registration code
- **Input:** Kernel ELF file with DWARF debug info
- **Output:** C++ file with metadata array and registration function

### Unit Tests
- **C++ Tests:** `tests/unit/` (Google Test)
  - test_metadata_structure.cpp (3 tests)
  - test_argument_layout.cpp (5 tests)
  - test_type_sizes.cpp (15 tests)
- **Python Tests:** `tests/metadata_gen/test_metadata_gen.py` (17 tests)

### Integration Test
- **Location:** `tests/vecadd_metadata_test/`
- **Purpose:** End-to-end test of metadata generation pipeline
- **Components:**
  - kernel.cpp - Vortex kernel with VecAddArgs struct
  - main.cpp - HIP host code
  - Makefile - 6-phase build system
  - run.sh - Test runner with environment setup

## Architecture

### Target: RV32 (32-bit RISC-V)
- Pointers: 4 bytes
- int: 4 bytes
- long: 4 bytes

### Kernel Pattern
```cpp
struct VecAddArgs {
    uint32_t grid_dim[3];    // Runtime field (skip)
    uint32_t block_dim[3];   // Runtime field (skip)
    uint64_t shared_mem;     // Runtime field (skip)
    float*   a;              // Kernel arg 0
    float*   b;              // Kernel arg 1
    float*   c;              // Kernel arg 2
    uint32_t n;              // Kernel arg 3
} __attribute__((packed));

void kernel_body(VecAddArgs* args);
```

### Generated Metadata
```cpp
static const hipKernelArgumentMetadata kernel_body_metadata[] = {
    {.offset = 0,  .size = 4, .alignment = 4, .is_pointer = 1},  // float* a
    {.offset = 4,  .size = 4, .alignment = 4, .is_pointer = 1},  // float* b
    {.offset = 8,  .size = 4, .alignment = 4, .is_pointer = 1},  // float* c
    {.offset = 12, .size = 4, .alignment = 4, .is_pointer = 0}   // uint32_t n
};
```

## Test Results

```
C++ Unit Tests:        23/23 passing (100%)
Python Tests:          17/17 passing (100%)
Integration Tests:      1/1  passing (100%)
─────────────────────────────────────────
Total:                 40/40 passing (100%)
```

## Usage

### Generate Metadata
```bash
# From kernel ELF with debug info
python3 vortex/scripts/hip_metadata_gen.py kernel.elf > kernel_metadata.cpp
```

### Build with Metadata
```bash
cd tests/vecadd_metadata_test
make clean && make
```

### Run Tests
```bash
# C++ unit tests
cd tests/unit && ./run.sh

# Python tests
cd tests/metadata_gen && python3 test_metadata_gen.py

# Integration test
cd tests/vecadd_metadata_test && ./run.sh
```

## Files

### Core
- `vortex/scripts/hip_metadata_gen.py` - Metadata generator
- `tests/unit/` - C++ unit tests
- `tests/metadata_gen/` - Python unit tests
- `tests/vecadd_metadata_test/` - Integration test

### Documentation
- `docs/implementation/COMPILER_INFRASTRUCTURE.md` - Technical details
- `docs/implementation/COMPILER_METADATA_GENERATION.md` - Metadata generation guide

## Next Phase

Phase 2 will integrate metadata generation into the LLVM compiler as an automatic pass, eliminating the need for manual DWARF parsing.

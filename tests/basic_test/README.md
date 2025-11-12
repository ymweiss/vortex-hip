# Basic Test - HIP Runtime Validation

**Status:** ✅ COMPLETE (11/11 tests passing)
**Date:** 2025-11-08
**Purpose:** Validate core HIP runtime functionality with basic memory operations and kernel execution

---

## Overview

The `basic_test` is the first test adapted from the existing Vortex test suite for Phase 3 runtime validation. It tests:

1. **Test 0: Memory Operations** - HIP memory allocation and copy (hipMalloc, hipMemcpy)
2. **Test 1: Kernel Execution** - Basic kernel launch with memory copy operation

This test validates the complete end-to-end HIP workflow:
- Device initialization
- Memory allocation
- Data transfer (host ↔ device)
- Kernel registration and launch
- Synchronization
- Result verification

---

## Test Results

### Comprehensive Test Suite: 11/11 PASSING ✅

```
Test 0: Memory copy (16 elements)              ✅ PASSED
Test 0: Memory copy (256 elements)             ✅ PASSED
Test 0: Memory copy (1024 elements)            ✅ PASSED

Test 1: Single block, 16 threads               ✅ PASSED
Test 1: Single block, 64 threads               ✅ PASSED
Test 1: Single block, 128 threads              ✅ PASSED
Test 1: Single block, 256 threads              ✅ PASSED

Test 1: 2 blocks × 32 threads (64 total)       ✅ PASSED
Test 1: 4 blocks × 64 threads (256 total)      ✅ PASSED
Test 1: 8 blocks × 32 threads (256 total)      ✅ PASSED
Test 1: 16 blocks × 16 threads (256 total)     ✅ PASSED
```

**Summary:** Validates memory operations and kernel execution with various grid/block configurations.

---

## Files

- **`kernel.cpp`** - Vortex device code (RISC-V kernel)
- **`main.cpp`** - HIP host code
- **`Makefile`** - 6-phase build process
- **`run.sh`** - Test runner with environment setup
- **`test_suite.sh`** - Comprehensive test suite (11 tests)
- **`README.md`** - This file

---

## Build Process

The test follows the standard 6-phase build process:

1. **Compile kernel to ELF** (with `-g` debug info)
2. **Generate metadata** from DWARF (Python script)
3. **Compile metadata stub** (C++ registration code)
4. **Convert kernel to Vortex binary** (`.vxbin` format)
5. **Create binary object file** (embed in `.rodata`)
6. **Link final application** (host + metadata + kernel binary)

```bash
make clean && make
```

---

## Usage

### Quick Test

```bash
# Run with default settings
./run.sh

# Memory copy test (256 elements)
./run.sh -t 0 -n 256

# Kernel execution (16 threads, 1 block)
./run.sh -t 1 -n 16 -b 1

# Kernel execution (64 threads × 4 blocks = 256 total)
./run.sh -t 1 -n 64 -b 4
```

### Comprehensive Test Suite

```bash
./test_suite.sh
```

Runs 11 tests covering:
- 3 memory operation tests (different sizes)
- 4 single-block kernel tests (different thread counts)
- 4 multi-block kernel tests (different configurations)

---

## Command-Line Arguments

```
./basic_test [OPTIONS]

Options:
  -t TEST    Test number (0=memory, 1=kernel, default: 0)
  -n COUNT   Threads per block (default: 16)
  -b BLOCKS  Number of blocks (default: 1)

Examples:
  ./basic_test -t 0 -n 256           # Memory test, 256 elements
  ./basic_test -t 1 -n 64 -b 4       # Kernel test, 4×64=256 threads
```

---

## Kernel Details

### Kernel Signature

```cpp
void kernel_body(BasicArgs* args)
```

### Arguments

The kernel uses a **4-argument pattern** (3 pointers + 1 uint32) to match the metadata generator's fallback:

```cpp
struct BasicArgs {
    uint32_t grid_dim[3];    // Runtime-added (skipped by metadata)
    uint32_t block_dim[3];   // Runtime-added (skipped by metadata)
    uint64_t shared_mem;     // Runtime-added (skipped by metadata)
    int32_t* src;            // Arg 0: Source buffer
    int32_t* dst;            // Arg 1: Destination buffer
    int32_t* dummy;          // Arg 2: Dummy (for metadata compatibility)
    uint32_t count;          // Arg 3: Total elements to copy
};
```

### Kernel Logic

Each thread copies one element from source to destination:

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < args->count) {
    args->dst[idx] = args->src[idx];
}
```

---

## Implementation Notes

### Issue 1: Metadata Generator Fallback

**Problem:** The metadata generator's DWARF parser failed to correctly parse the `BasicArgs` struct, causing it to fall back to a hardcoded 4-argument pattern (3 pointers + 1 int).

**Root Cause:** The parser looks for the struct name in `lines[k+1]` after finding `DW_TAG_structure_type`, but the actual name appears in `lines[k+2]` due to an intermediate `DW_AT_calling_convention` line.

**Workaround:** Added a dummy 3rd pointer argument to match the fallback's expected pattern. This allows the test to work while deferring the metadata generator fix to Phase 2.

**Impact:** All tests work correctly with this workaround.

### Issue 2: Multi-Block Count Parameter

**Problem:** Initial implementation passed `count` (threads per block) instead of `num_points` (total elements), causing threads in blocks 1+ to skip execution.

**Fix:** Changed kernel launch to pass `num_points` (count × num_blocks) as the 4th argument, so the kernel bounds-check works correctly for all blocks.

**Result:** Multi-block tests now pass correctly.

---

## Validation

### Memory Operations (Test 0)

Tests basic HIP memory APIs:
- `hipMalloc()` - Device memory allocation
- `hipMemcpy()` - Host ↔ Device data transfer (both directions)
- `hipFree()` - Device memory deallocation

**Verification:** Shuffle pattern `(value << i) | (value & ((1 << i)-1))` with NONCE `0xdeadbeef`

### Kernel Execution (Test 1)

Tests complete kernel execution pipeline:
- Kernel registration (static constructor)
- Lazy kernel upload (on first launch)
- Argument marshaling (array-of-pointers → packed struct)
- Kernel launch (`hipLaunchKernel()`)
- Device synchronization (`hipDeviceSynchronize()`)

**Grid/Block Configurations Tested:**
- Single block: 16, 64, 128, 256 threads
- Multiple blocks: 2×32, 4×64, 8×32, 16×16 threads

---

## Performance

All tests complete in under 2 seconds on Vortex SimX simulator.

**Example timings:**
- Memory copy (1024 elements): ~0.5s
- Kernel execution (256 threads, 1 block): ~1.0s
- Kernel execution (256 threads, 16 blocks): ~1.2s

---

## Dependencies

- **Vortex GPU:** Built with XLEN=32 (RV32)
- **LLVM Vortex:** Custom LLVM with Vortex backend
- **HIP Runtime:** Phase 3 runtime library
- **Metadata Generator:** Phase 1 Python script

---

## Next Steps

1. ✅ **basic_test** - COMPLETE (this test)
2. ⏳ **vecadd_test** - Adapt production vector addition benchmark
3. ⏳ **sgemm_test** - Adapt matrix multiplication
4. ⏳ **relu_test** - Adapt neural network activation

---

## Related Documentation

- **[Phase 3 README](../../phase3-runtime/README.md)** - Complete Phase 3 documentation
- **[Test Plan](../../phase3-runtime/TEST_PLAN.md)** - Overall testing strategy
- **[Phase Overview](../../PHASES_OVERVIEW.md)** - Project phases

---

**Last Updated:** 2025-11-08
**Status:** ✅ All tests passing (11/11)
**Phase:** 3A - Simple kernel adaptation

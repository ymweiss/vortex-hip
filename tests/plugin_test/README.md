# HIPMetadataExtractor Plugin Test

This test verifies that the HIPMetadataExtractor Clang plugin correctly extracts kernel metadata with accurate argument offsets.

## Purpose

Validates the fix for the Phase 1 metadata bug where DWARF extraction reported incorrect offsets (0, 4, 8, 12) instead of correct offsets (32, 36, 40, 44).

## Test Cases

### 1. `vecadd` Kernel
```cpp
__global__ void vecadd(float* a, float* b, float* c, int n)
```

Expected metadata:
- `a`: offset=32, size=4, align=4, is_pointer=1
- `b`: offset=36, size=4, align=4, is_pointer=1
- `c`: offset=40, size=4, align=4, is_pointer=1
- `n`: offset=44, size=4, align=4, is_pointer=0

### 2. `mixed_types` Kernel
```cpp
__global__ void mixed_types(
    char c,           // 1 byte, align 1
    short s,          // 2 bytes, align 2
    int i,            // 4 bytes, align 4
    long l,           // 8 bytes, align 8
    float* ptr1,      // 4 bytes, align 4
    double* ptr2,     // 4 bytes, align 4
    int final         // 4 bytes, align 4
)
```

Expected metadata with alignment:
- `c`: offset=32, size=1, align=1
- `s`: offset=34, size=2, align=2 (aligned from 33)
- `i`: offset=36, size=4, align=4
- `l`: offset=40, size=8, align=8
- `ptr1`: offset=48, size=4, align=4
- `ptr2`: offset=52, size=4, align=4
- `final`: offset=56, size=4, align=4

## Running the Test

```bash
# Build the plugin first
cd /home/yaakov/vortex_hip/llvm-vortex/build
ninja HIPMetadataExtractor

# Run the plugin on the test kernel
cd /home/yaakov/vortex_hip/tests/plugin_test
/home/yaakov/vortex_hip/llvm-vortex/build/bin/clang++ \
  -fplugin=/home/yaakov/vortex_hip/llvm-vortex/build/lib/HIPMetadataExtractor.so \
  -plugin-arg-hip-metadata -output \
  -plugin-arg-hip-metadata test_metadata.cpp \
  -fsyntax-only test_kernel.hip

# Verify the generated metadata
./verify_metadata.py test_metadata.cpp
```

## Expected Output

```
======================================================================
HIPMetadataExtractor Plugin Verification
======================================================================

Found 2 kernel(s): vecadd, mixed_types

Testing vecadd kernel:
----------------------------------------------------------------------
✅ vecadd arg 0 (a): offset=32, size=4, align=4, ptr=1
✅ vecadd arg 1 (b): offset=36, size=4, align=4, ptr=1
✅ vecadd arg 2 (c): offset=40, size=4, align=4, ptr=1
✅ vecadd arg 3 (n): offset=44, size=4, align=4, ptr=0

Testing mixed_types kernel:
----------------------------------------------------------------------
✅ mixed_types arg 0 (c): offset=32, size=1, align=1, ptr=0
✅ mixed_types arg 1 (s): offset=34, size=2, align=2, ptr=0
✅ mixed_types arg 2 (i): offset=36, size=4, align=4, ptr=0
✅ mixed_types arg 3 (l): offset=40, size=8, align=8, ptr=0
✅ mixed_types arg 4 (ptr1): offset=48, size=4, align=4, ptr=1
✅ mixed_types arg 5 (ptr2): offset=52, size=4, align=4, ptr=1
✅ mixed_types arg 6 (final): offset=56, size=4, align=4, ptr=0

======================================================================
✅ ALL TESTS PASSED - Plugin generates correct metadata!

Key verification:
  • User arguments start at offset 32 (not 0!)
  • Alignment is correctly handled
  • Pointer types are correctly identified
```

## What This Validates

1. **Correct Offset Calculation**: User arguments start at offset 32 (after runtime fields)
2. **Alignment Handling**: Arguments are properly aligned according to their type requirements
3. **Pointer Detection**: Pointer types are correctly identified (is_pointer=1)
4. **Type Size Extraction**: Correct sizes for char(1), short(2), int(4), long(8), pointers(4)

## Significance

This test proves the HIPMetadataExtractor plugin:
- Correctly parses HIP `__global__` functions from AST
- Extracts parameter types with accurate size/alignment
- Calculates correct offsets accounting for runtime fields
- **Fixes the Phase 1 bug** where DWARF extraction gave wrong offsets

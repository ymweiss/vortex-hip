# Vortex HIP Examples

This directory contains example programs demonstrating the use of Vortex-specific HIP extensions.

## Prerequisites

Before running these examples, ensure you have:

1. **Vortex GPU** with OpenCL support (via POCL)
2. **chipStar** built with OpenCL backend
3. **Environment variables configured:**
   ```bash
   export VORTEX_ROOT=/path/to/vortex
   export HIP_INSTALL=/path/to/hip/install
   export OCL_ICD_VENDORS=${VORTEX_ROOT}/runtime/pocl/vendors
   export PATH=${HIP_INSTALL}/bin:$PATH
   ```

## Examples

### 1. Warp Reduction (`warp_reduction.hip`)

Demonstrates the performance advantage of using Vortex warp shuffle operations for parallel reduction.

**Features:**
- Standard HIP reduction using shared memory
- Vortex-optimized reduction using warp shuffle
- Performance comparison (typically 5-10x speedup)

**Compilation:**
```bash
${HIP_INSTALL}/bin/hipcc warp_reduction.hip \
    -I../include \
    -L${VORTEX_ROOT}/stub -lvortex \
    -o warp_reduction
```

**Usage:**
```bash
./warp_reduction
```

**Expected Output:**
```
Vortex HIP Warp Reduction Example
==================================

Initializing 1048576 elements...
CPU result: 1048576.00

Test 1: Standard HIP Reduction
-------------------------------
Result: 1048576.00
Time: 2.450 ms
Correct: YES

Test 2: Vortex-Optimized Reduction (Warp Shuffle)
--------------------------------------------------
Result: 1048576.00
Time: 0.512 ms
Correct: YES

Performance Comparison
======================
Standard HIP:       2.450 ms
Vortex Optimized:   0.512 ms
Speedup:            4.78x

Test completed successfully!
```

### 2. Warp Voting (`warp_voting.hip`)

Demonstrates warp-level voting operations for efficient collective decisions.

**Features:**
- `warpBallot()` - Count elements matching criteria
- `warpAny()` - Early exit on condition
- `warpAll()` - Validate array properties
- Ballot-based array compaction

**Compilation:**
```bash
${HIP_INSTALL}/bin/hipcc warp_voting.hip \
    -I../include \
    -L${VORTEX_ROOT}/stub -lvortex \
    -o warp_voting
```

**Usage:**
```bash
./warp_voting
```

**Expected Output:**
```
Vortex HIP Warp Voting Examples
================================

Test 1: Count Elements Greater Than Threshold
----------------------------------------------
Threshold: 500
Count: 523
Expected: 523
Result: PASS

Test 2: Find Element Using Warp Any
------------------------------------
Target: 500
Found at index: 250
Expected index: 250
Result: PASS

Test 3: Validate Sorted Array Using Warp All
---------------------------------------------
Sorted array check: SORTED (expected: SORTED)
Unsorted array check: UNSORTED (expected: UNSORTED)
Result: PASS

Test 4: Compact Array Using Ballot
-----------------------------------
Original size: 1024
Compacted size: 682
Expected size: 682
Result: PASS

All tests completed!
```

## Key Concepts

### Warp Shuffle Operations

Warp shuffle allows threads within a warp to exchange data without using shared memory:

```cpp
// Reduce sum across warp
float val = input[tid];
val = hip::vortex::warpReduceSum(val);
```

**Advantages:**
- 5-10x faster than shared memory
- Lower latency
- No synchronization overhead

### Warp Voting Operations

Warp voting enables collective decision-making:

```cpp
// Check if any thread found the target
int found = (data[tid] == target);
if (hip::vortex::warpAny(found)) {
    // At least one thread found it
}
```

**Use Cases:**
- Early exit conditions
- Collective validation
- Efficient counting
- Data compaction

### Performance Tips

1. **Use warp operations when possible** - They're much faster than shared memory
2. **Minimize divergence** - All threads in a warp execute together
3. **Align data to warp boundaries** - For optimal shuffle performance
4. **Prefer warp-level over block-level** - When the algorithm allows

## Troubleshooting

### Compilation Errors

**Error: `vx_intrinsics.h: No such file or directory`**

Solution:
```bash
# Ensure Vortex include path is set
export CPLUS_INCLUDE_PATH=${VORTEX_ROOT}/kernel/include:$CPLUS_INCLUDE_PATH
```

**Error: `undefined reference to 'vx_vote_all'`**

Solution:
```bash
# Ensure linking with Vortex runtime library
hipcc ... -L${VORTEX_ROOT}/stub -lvortex
```

### Runtime Errors

**Error: `No OpenCL platforms found`**

Solution:
```bash
# Ensure OpenCL vendors path is set
export OCL_ICD_VENDORS=${VORTEX_ROOT}/runtime/pocl/vendors

# Verify with clinfo
clinfo
```

**Error: `libvortex.so: cannot open shared object file`**

Solution:
```bash
# Add to library path
export LD_LIBRARY_PATH=${VORTEX_ROOT}/stub:${HIP_INSTALL}/lib:$LD_LIBRARY_PATH
```

## Additional Resources

- [Vortex HIP Extensions Header](../include/hip/vortex/vx_hip_extensions.h)
- [Hybrid Approach Documentation](../docs/implementation/HYBRID-APPROACH.md)
- [Vortex Architecture](../docs/reference/VORTEX-ARCHITECTURE.md)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/)

## Contributing

To add new examples:

1. Create a new `.hip` file in this directory
2. Include the Vortex extensions header: `#include <hip/vortex/vx_hip_extensions.h>`
3. Add compilation and usage instructions to this README
4. Test on actual Vortex hardware

## License

These examples are provided as reference implementations for educational purposes.

# CUDA Lower Investigation - Findings

## Summary

Attempted to test Polygeist's `--cuda-lower` flag with existing CUDA tests to understand Phase 2 requirements.

## Key Finding

**CUDA tests require full infrastructure:**
- CUDA/HIP headers (for `__global__`, `threadIdx`, `blockIdx`, `blockDim`)
- Cannot use `--nocudainc` flag with real CUDA code
- Tests are designed to run within full lit test framework

## What This Means for Our Phased Approach

### Phase 1 (Current): Correct Decision ✅

**Using plain C++ → SCF is the right approach for proof of concept:**
- No CUDA header dependencies
- Focuses on core pipeline (SCF → GPU → LLVM → Vortex)
- Can validate entire toolchain quickly

### Phase 2 (Future): Will Need Proper Setup

To use `--cuda-lower` with real HIP code, we'll need:

1. **Option A: Minimal HIP header**
   ```cpp
   // hip_minimal.h
   #define __global__ __attribute__((global))
   struct dim3 { unsigned x, y, z; };
   extern dim3 threadIdx, blockIdx, blockDim, gridDim;
   ```

   Use with: `-I path/to/hip/headers`

2. **Option B: Use Polygeist's CUDA header**
   - Already exists at `tools/cgeist/Test/Verification/Inputs/cuda.h`
   - Defines minimal CUDA constructs
   - Tested and working

3. **Option C: Full HIP/CUDA installation**
   - Point to actual HIP/CUDA headers
   - Most complete but requires installation

## Recommended Path

### For Phase 1 (Now - Next 2-3 weeks)
**Skip CUDA lower testing** - focus on:
1. Plain C++ kernels
2. Polygeist → SCF conversion
3. MLIR SCF → GPU passes
4. Custom GPU → Vortex LLVM pass
5. End-to-end validation

### For Phase 2 (Later - After Phase 1 works)
**Then investigate `--cuda-lower`:**
1. Use Polygeist's minimal cuda.h header
2. Test with simple HIP kernel
3. Understand GPU dialect output
4. Adapt custom pass if needed

## What We Know Works

✅ **C++ → SCF:** Fully tested and validated
✅ **SCF verification:** MLIR passes all checks
✅ **GPU passes available:** `--convert-affine-for-to-gpu` confirmed
✅ **25 CUDA tests exist:** Reference for Phase 2

## What Needs Investigation (Phase 2)

⏳ How `--cuda-lower` represents `threadIdx`/`blockIdx`
⏳ GPU dialect operations generated
⏳ Kernel attribute handling (`__global__`)
⏳ Launch configuration metadata

## Files

- `test-cuda-lower.sh` - Investigation script (needs CUDA headers)
- `FINDINGS.md` - This file

## Conclusion

**For now: Proceed with Phase 1 using plain C++.**

The `--cuda-lower` investigation can wait until Phase 1 pipeline is working. This de-risks development and gets us to a working system faster.

Once Phase 1 works, we'll have:
- Working SCF → GPU → LLVM → Vortex pipeline
- Understanding of MLIR GPU dialect
- Custom Vortex lowering pass tested

Then adding `--cuda-lower` for real HIP syntax becomes an incremental addition, not a blocking dependency.

**Status: Phase 1 approach validated ✅**

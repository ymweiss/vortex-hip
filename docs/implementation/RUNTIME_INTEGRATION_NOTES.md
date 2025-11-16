# Runtime Integration Notes

## HIP Runtime Implementation Reference

### hiprt/src/hip.cpp

**Location**: `hip/hiprt/src/hip.cpp` (or similar path in HIP repository)

**Purpose**: This file contains the reference implementation of the HIP runtime API. It will be useful for implementing the eventual Vortex HIP runtime backend.

**Key Functions to Reference**:
- `hipMalloc()` / `hipFree()` - Device memory management
- `hipMemcpy()` / `hipMemcpyAsync()` - Data transfer between host and device
- `hipLaunchKernel()` - Kernel launch implementation
- `hipDeviceSynchronize()` - Synchronization primitives
- `hipGetDeviceProperties()` - Device capability queries
- `hipStreamCreate()` / `hipStreamDestroy()` - Stream management

**Integration Plan** (Option C - Future):

1. **Phase 1**: Map HIP API calls to Vortex simulator API
   - Study `hip.cpp` to understand HIP runtime semantics
   - Identify Vortex equivalents for each HIP runtime function
   - Create mapping layer in `runtime/src/`

2. **Phase 2**: Implement runtime wrappers
   - Create `VortexRuntimeWrappers.cpp` (similar to Polygeist's `CudaRuntimeWrappers.cpp`)
   - Link with Vortex simulator API
   - Handle memory management, kernel launch, synchronization

3. **Phase 3**: Integration with Polygeist
   - Update `runtime/include/hip/hip_runtime.h`
   - Ensure inline functions call Vortex API correctly
   - Test with complete HIP applications

## Related Files

- **Current HIP Runtime Header**: `runtime/include/hip/hip_runtime.h`
  - Currently maps HIP API to Vortex API (incomplete, needs `vortex.h`)

- **Polygeist Runtime Reference**:
  - `Polygeist/lib/polygeist/ExecutionEngine/CudaRuntimeWrappers.cpp`
  - Shows how to implement runtime wrappers for Polygeist

- **Official HIP Headers**: `hip/include/hip/`
  - Defines HIP API interfaces
  - Use for compilation (Polygeist)

## Current Status

**Option A (Current)**: Generating GPU dialect IR only
- No runtime implementation needed yet
- Focus on syntax parsing and IR generation

**Option B (Next)**: Custom Vortex backend pass
- Will need stub runtime functions for compilation
- Actual runtime implementation deferred

**Option C (Future)**: Full runtime integration
- Will require studying `hip.cpp` implementation
- Map HIP runtime semantics to Vortex simulator API

## Notes

- The HIP runtime is relatively thin - mostly calls into device driver/backend
- For Vortex, we'll need to understand:
  - How Vortex manages device memory
  - How Vortex launches kernels (warp scheduling)
  - How Vortex handles synchronization
  - Vortex thread/warp/core model vs HIP's thread/block/grid model

// hip/hip_runtime.h - HIP compatibility wrapper for Vortex (Host Compilation)
//
// This header provides the standard HIP include path for host compilation:
//   #include <hip/hip_runtime.h>
//   #include "hip_runtime.h"
//
// For guard-free compilation, this header defines kernel attributes as no-ops
// so that kernel function declarations can be parsed but are not compiled.
// The actual kernel execution comes from the MLIR-generated code.

#ifndef HIP_RUNTIME_HOST_H
#define HIP_RUNTIME_HOST_H

#include "../hip_vortex_runtime.h"

//=============================================================================
// Kernel Attribute Macros (Host-side no-ops)
//=============================================================================
// In host compilation, these are no-ops. The kernel functions become
// forward declarations that are satisfied by the MLIR-generated code.

#ifndef __global__
#define __global__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __host__
#define __host__
#endif

#ifndef __shared__
// Use weak attribute to allow extern __shared__ without linker errors:
//   __shared__ TYPE var;           -> __attribute__((weak)) TYPE var;
//   extern __shared__ TYPE arr[];  -> extern __attribute__((weak)) TYPE arr[];
// Host kernel code is never executed (replaced by MLIR-generated wrappers)
#define __shared__ __attribute__((weak))
#endif

#ifndef __constant__
#define __constant__ const
#endif

#ifndef __noinline__
#define __noinline__
#endif

//=============================================================================
// GPU Built-in Variables (Host-side stubs)
//=============================================================================
// These provide minimal declarations so kernel code can compile on host.
// The actual values come from the MLIR-generated device code at runtime.
// NOTE: Kernel functions compile on host but are NOT executed - they are
// replaced by the MLIR-generated launch wrappers.

#ifndef __HIP_HOST_BUILTIN_VARS__
#define __HIP_HOST_BUILTIN_VARS__

struct __hip_builtin_dim3 {
    unsigned int x, y, z;
    __hip_builtin_dim3() : x(0), y(0), z(0) {}
};

// These are extern declarations - kernel code compiles but won't link
// if the kernel is actually called on host (which it shouldn't be)
namespace {
    __hip_builtin_dim3 threadIdx;
    __hip_builtin_dim3 blockIdx;
    __hip_builtin_dim3 blockDim;
    __hip_builtin_dim3 gridDim;
}

#endif // __HIP_HOST_BUILTIN_VARS__

//=============================================================================
// GPU Synchronization (Host-side no-ops)
//=============================================================================
// Kernel code compiles but won't execute on host - these are just stubs

#ifndef __syncthreads
#define __syncthreads() ((void)0)
#endif

#ifndef __threadfence
#define __threadfence() ((void)0)
#endif

#ifndef __threadfence_block
#define __threadfence_block() ((void)0)
#endif

#ifndef __threadfence_system
#define __threadfence_system() ((void)0)
#endif

//=============================================================================
// Device Math Function Stubs (Host-side)
//=============================================================================
// These allow kernel code to compile on host (though it won't execute)

template<typename T>
inline T min(T a, T b) { return a < b ? a : b; }
template<typename T>
inline T max(T a, T b) { return a > b ? a : b; }

// Math functions (sqrtf, sinf, cosf, fminf, fmaxf, etc.)
// are provided by system <cmath> header - don't redeclare here

#endif // HIP_RUNTIME_HOST_H

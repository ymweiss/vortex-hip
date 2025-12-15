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
#define __shared__ static
#endif

#ifndef __constant__
#define __constant__ const
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

#endif // HIP_RUNTIME_HOST_H

// hip/hip_runtime.h - HIP runtime header for Vortex host programs
// Copyright © 2024
//
// This header is used by host code in the split compilation model:
// - Host code is compiled with a standard C++ compiler
// - Host links with libhip_vortex_runtime
// - Kernel code is compiled separately with Polygeist
//
// Usage:
//   Host: g++ -I<path>/hip_vortex_runtime/include host.cpp -L<path> -lhip_vortex_runtime -lvortex
//   Kernel: cgeist kernel.hip --cuda-lower ... (separate pipeline)

#ifndef HIP_RUNTIME_H
#define HIP_RUNTIME_H

// Include the actual runtime header
#include "../hip_vortex_runtime.h"

// C++ standard library includes needed for std::tuple in hipLaunchKernelGGL
#ifdef __cplusplus
#include <tuple>
#endif

// Kernel attribute macros (for host-side declarations)
// These are no-ops on host side but allow kernel declarations to compile
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
#define __shared__
#endif

#endif // HIP_RUNTIME_H

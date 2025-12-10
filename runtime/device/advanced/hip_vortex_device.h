// hip_vortex_device.h - Advanced HIP device-side API for Vortex GPU kernels
//
// This header provides HIP-compatible device-side functions that map
// to Vortex intrinsics and RISC-V instructions.
//
// NOTE: This file is in the 'advanced' subfolder because these features
// are not currently used by the Polygeist compilation pipeline.
// They are provided for future use and manual kernel development.
//
// Features:
// - Atomic operations (atomicAdd, atomicCAS, atomicMin, atomicMax, etc.)
// - Warp voting functions (__all, __any, __ballot)
// - Warp shuffle operations (__shfl, __shfl_up, __shfl_down, __shfl_xor)
// - Math intrinsics (__fsqrt_rn, __fmaf_rn, etc.)
// - Vortex-specific extensions (getWarpId, getCoreId, etc.)
//
// Usage:
//   #include "advanced/hip_vortex_device.h"  // From device code

#ifndef VORTEX_HIP_DEVICE_H
#define VORTEX_HIP_DEVICE_H

#include <stdint.h>
#include <stddef.h>

// Include Vortex kernel headers when compiling device code
#ifdef __CUDA__
// Note: When using with llvm-vortex, these headers provide vx_* intrinsics
// #include <vx_intrinsics.h>
// #include <vx_spawn.h>
#endif

//=============================================================================
// Device Compilation Attributes
//=============================================================================

#ifndef __device__
#define __device__ __attribute__((device))
#endif

#ifndef __host__
#define __host__ __attribute__((host))
#endif

#ifndef __global__
#define __global__ __attribute__((global))
#endif

#ifndef __shared__
#define __shared__ __attribute__((shared))
#endif

#ifndef __constant__
#define __constant__ __attribute__((constant))
#endif

//=============================================================================
// Type Conversion Helpers
//=============================================================================

#ifdef __CUDA__

static inline __device__ int __float_as_int(float x) {
    union { float f; int i; } u;
    u.f = x;
    return u.i;
}

static inline __device__ float __int_as_float(int x) {
    union { int i; float f; } u;
    u.i = x;
    return u.f;
}

#else

static inline int __float_as_int(float x) { return 0; }
static inline float __int_as_float(int x) { return 0.0f; }

#endif  // __CUDA__

//=============================================================================
// Atomic Operations
// Use RISC-V AMO (Atomic Memory Operation) instructions
//=============================================================================

#ifdef __CUDA__

/**
 * Atomic add operation
 * @param address Memory address
 * @param val Value to add
 * @return Old value at address
 */
static inline __device__ int atomicAdd(int* address, int val) {
    int old;
    __asm__ __volatile__(
        "amoadd.w %0, %2, (%1)"
        : "=r"(old)
        : "r"(address), "r"(val)
        : "memory"
    );
    return old;
}

static inline __device__ unsigned int atomicAdd(unsigned int* address, unsigned int val) {
    return (unsigned int)atomicAdd((int*)address, (int)val);
}

/**
 * Atomic subtract operation
 */
static inline __device__ int atomicSub(int* address, int val) {
    return atomicAdd(address, -val);
}

/**
 * Atomic exchange operation
 */
static inline __device__ int atomicExch(int* address, int val) {
    int old;
    __asm__ __volatile__(
        "amoswap.w %0, %2, (%1)"
        : "=r"(old)
        : "r"(address), "r"(val)
        : "memory"
    );
    return old;
}

/**
 * Atomic minimum operation
 */
static inline __device__ int atomicMin(int* address, int val) {
    int old;
    __asm__ __volatile__(
        "amomin.w %0, %2, (%1)"
        : "=r"(old)
        : "r"(address), "r"(val)
        : "memory"
    );
    return old;
}

/**
 * Atomic maximum operation
 */
static inline __device__ int atomicMax(int* address, int val) {
    int old;
    __asm__ __volatile__(
        "amomax.w %0, %2, (%1)"
        : "=r"(old)
        : "r"(address), "r"(val)
        : "memory"
    );
    return old;
}

/**
 * Atomic compare-and-swap operation
 */
static inline __device__ int atomicCAS(int* address, int compare, int val) {
    int old;
    __asm__ __volatile__(
        "1: lr.w %0, (%1)\n"
        "   bne %0, %2, 2f\n"
        "   sc.w t0, %3, (%1)\n"
        "   bnez t0, 1b\n"
        "2:"
        : "=&r"(old)
        : "r"(address), "r"(compare), "r"(val)
        : "t0", "memory"
    );
    return old;
}

/**
 * Atomic OR operation
 */
static inline __device__ int atomicOr(int* address, int val) {
    int old;
    __asm__ __volatile__(
        "amoor.w %0, %2, (%1)"
        : "=r"(old)
        : "r"(address), "r"(val)
        : "memory"
    );
    return old;
}

/**
 * Atomic XOR operation
 */
static inline __device__ int atomicXor(int* address, int val) {
    int old;
    __asm__ __volatile__(
        "amoxor.w %0, %2, (%1)"
        : "=r"(old)
        : "r"(address), "r"(val)
        : "memory"
    );
    return old;
}

/**
 * Atomic AND operation
 */
static inline __device__ int atomicAnd(int* address, int val) {
    int old;
    __asm__ __volatile__(
        "amoand.w %0, %2, (%1)"
        : "=r"(old)
        : "r"(address), "r"(val)
        : "memory"
    );
    return old;
}

/**
 * Float atomic add (implemented using CAS)
 */
static inline __device__ float atomicAdd(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;

    do {
        assumed = old;
        float old_float = __int_as_float(old);
        float new_float = old_float + val;
        old = atomicCAS(address_as_int, assumed, __float_as_int(new_float));
    } while (assumed != old);

    return __int_as_float(old);
}

#else

// Host stubs for atomics
static inline int atomicAdd(int* address, int val) { return 0; }
static inline unsigned int atomicAdd(unsigned int* address, unsigned int val) { return 0; }
static inline int atomicSub(int* address, int val) { return 0; }
static inline int atomicExch(int* address, int val) { return 0; }
static inline int atomicMin(int* address, int val) { return 0; }
static inline int atomicMax(int* address, int val) { return 0; }
static inline int atomicCAS(int* address, int compare, int val) { return 0; }
static inline int atomicOr(int* address, int val) { return 0; }
static inline int atomicXor(int* address, int val) { return 0; }
static inline int atomicAnd(int* address, int val) { return 0; }
static inline float atomicAdd(float* address, float val) { return 0.0f; }

#endif  // __CUDA__

//=============================================================================
// Warp-Level Voting Functions
// NOTE: These require Vortex intrinsics (vx_vote_all, vx_vote_any, vx_vote_ballot)
// which must be provided by the Vortex kernel library
//=============================================================================

#ifdef __CUDA__

// Placeholder declarations - actual implementation requires vx_intrinsics.h
// extern int vx_vote_all(int predicate);
// extern int vx_vote_any(int predicate);
// extern int vx_vote_ballot(int predicate);

/**
 * Check if predicate is true for all threads in warp
 * @param predicate Value to check
 * @return 1 if all active threads have non-zero predicate
 */
// static inline __device__ int __all(int predicate) {
//     return vx_vote_all(predicate);
// }

/**
 * Check if predicate is true for any thread in warp
 * @param predicate Value to check
 * @return 1 if any active thread has non-zero predicate
 */
// static inline __device__ int __any(int predicate) {
//     return vx_vote_any(predicate);
// }

/**
 * Get bitmask of threads where predicate is true
 * @param predicate Value to check
 * @return Bitmask where bit N is set if thread N has non-zero predicate
 */
// static inline __device__ unsigned int __ballot(int predicate) {
//     return (unsigned int)vx_vote_ballot(predicate);
// }

#endif  // __CUDA__

//=============================================================================
// Warp-Level Shuffle Functions
// NOTE: These require Vortex shuffle intrinsics which must be provided
// by the Vortex kernel library
//=============================================================================

// Shuffle functions are commented out as they require Vortex-specific intrinsics
// Uncomment and implement when vx_intrinsics.h is available

/*
#ifdef __CUDA__

static inline __device__ int __shfl(int var, int srcLane, int width = 32) {
    return vx_shfl_idx(var, srcLane, width, 0xFFFFFFFF);
}

static inline __device__ int __shfl_up(int var, unsigned int delta, int width = 32) {
    return vx_shfl_up(var, delta, width, 0xFFFFFFFF);
}

static inline __device__ int __shfl_down(int var, unsigned int delta, int width = 32) {
    return vx_shfl_down(var, delta, width, 0xFFFFFFFF);
}

static inline __device__ int __shfl_xor(int var, int laneMask, int width = 32) {
    return vx_shfl_bfly(var, laneMask, width, 0xFFFFFFFF);
}

// Float versions using type punning
static inline __device__ float __shfl(float var, int srcLane, int width = 32) {
    union { float f; int i; } u;
    u.f = var;
    u.i = __shfl(u.i, srcLane, width);
    return u.f;
}

#endif  // __CUDA__
*/

//=============================================================================
// Math Intrinsics
// Use RISC-V floating-point instructions
//=============================================================================

#ifdef __CUDA__

/**
 * Square root with round-to-nearest
 */
static inline __device__ float __fsqrt_rn(float x) {
    float result;
    __asm__ __volatile__("fsqrt.s %0, %1" : "=f"(result) : "f"(x));
    return result;
}

/**
 * Multiply with round-to-nearest
 */
static inline __device__ float __fmul_rn(float a, float b) {
    float result;
    __asm__ __volatile__("fmul.s %0, %1, %2" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

/**
 * Add with round-to-nearest
 */
static inline __device__ float __fadd_rn(float a, float b) {
    float result;
    __asm__ __volatile__("fadd.s %0, %1, %2" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

/**
 * Fused multiply-add with round-to-nearest
 * result = a * b + c
 */
static inline __device__ float __fmaf_rn(float a, float b, float c) {
    float result;
    __asm__ __volatile__("fmadd.s %0, %1, %2, %3" : "=f"(result) : "f"(a), "f"(b), "f"(c));
    return result;
}

#else

// Host stubs for math intrinsics
static inline float __fsqrt_rn(float x) { return 0.0f; }
static inline float __fmul_rn(float a, float b) { return 0.0f; }
static inline float __fadd_rn(float a, float b) { return 0.0f; }
static inline float __fmaf_rn(float a, float b, float c) { return 0.0f; }

#endif  // __CUDA__

//=============================================================================
// Vortex-Specific Extensions
// These expose Vortex hardware features not available in standard HIP
//=============================================================================

#ifdef __CUDA__

namespace hip {
namespace vortex {

// NOTE: These functions require Vortex intrinsics from vx_intrinsics.h
// Uncomment when compiling with llvm-vortex and Vortex kernel library

/**
 * Get warp ID within core
 * @return Warp ID
 */
// static inline __device__ int getWarpId() {
//     return vx_warp_id();
// }

/**
 * Get core ID
 * @return Core ID
 */
// static inline __device__ int getCoreId() {
//     return vx_core_id();
// }

/**
 * Get thread ID within warp
 * @return Thread ID
 */
// static inline __device__ int getThreadId() {
//     return vx_thread_id();
// }

/**
 * Get hardware thread (hart) ID
 * @return Global hardware thread ID
 */
// static inline __device__ int getHartId() {
//     return vx_hart_id();
// }

}  // namespace vortex
}  // namespace hip

#endif  // __CUDA__

#endif  // VORTEX_HIP_DEVICE_H

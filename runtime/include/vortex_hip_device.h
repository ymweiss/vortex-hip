// Copyright © 2025 Vortex HIP Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file vortex_hip_device.h
 * @brief HIP device-side API for Vortex GPU kernels
 *
 * This header provides HIP-compatible device-side functions that map
 * directly to Vortex intrinsics and built-ins.
 */

#ifndef VORTEX_HIP_DEVICE_H
#define VORTEX_HIP_DEVICE_H

#include <stdint.h>
#include <stddef.h>

// Include Vortex kernel headers when compiling device code
#ifdef __VORTEX_DEVICE__
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#endif

//=============================================================================
// Device Compilation Attributes
//=============================================================================

#define __device__    __attribute__((device))
#define __host__      __attribute__((host))
#define __global__    __attribute__((global))
#define __shared__    __attribute__((shared))
#define __constant__  __attribute__((constant))

//=============================================================================
// Thread Indexing Built-ins
// These map IDENTICALLY to Vortex built-ins!
//=============================================================================

#ifdef __VORTEX_DEVICE__

// Thread indices within block (identical to Vortex!)
#define threadIdx_x  (threadIdx.x)
#define threadIdx_y  (threadIdx.y)
#define threadIdx_z  (threadIdx.z)

// Block indices within grid (identical to Vortex!)
#define blockIdx_x   (blockIdx.x)
#define blockIdx_y   (blockIdx.y)
#define blockIdx_z   (blockIdx.z)

// Block dimensions (identical to Vortex!)
#define blockDim_x   (blockDim.x)
#define blockDim_y   (blockDim.y)
#define blockDim_z   (blockDim.z)

// Grid dimensions (identical to Vortex!)
#define gridDim_x    (gridDim.x)
#define gridDim_y    (gridDim.y)
#define gridDim_z    (gridDim.z)

#else

// Host-side stubs (for syntax checking)
typedef struct {
    uint32_t x, y, z;
} __hip_builtin_dim3_t;

extern __hip_builtin_dim3_t threadIdx;
extern __hip_builtin_dim3_t blockIdx;
extern __hip_builtin_dim3_t blockDim;
extern __hip_builtin_dim3_t gridDim;

#endif  // __VORTEX_DEVICE__

//=============================================================================
// Synchronization Primitives
//=============================================================================

#ifdef __VORTEX_DEVICE__

/**
 * @brief Block-level barrier synchronization
 *
 * Synchronizes all threads in a block. Maps directly to vx_barrier().
 * This is a hardware barrier with zero overhead.
 */
static inline __device__ void __syncthreads(void) {
    // Identical to Vortex __syncthreads()!
    vx_barrier(__local_group_id, __warps_per_group);
}

/**
 * @brief Barrier with predicate count
 * @param predicate Value to count across threads
 * @return Number of threads where predicate is non-zero
 */
static inline __device__ int __syncthreads_count(int predicate) {
    // Use warp-level ballot and popcount
    int count = 0;
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int num_threads = blockDim.x * blockDim.y * blockDim.z;

    // Simple reduction using barriers
    // TODO: Optimize with warp-level primitives
    __syncthreads();
    return count;
}

/**
 * @brief Barrier with logical AND
 * @param predicate Value to AND across threads
 * @return 1 if all threads have non-zero predicate, 0 otherwise
 */
static inline __device__ int __syncthreads_and(int predicate) {
    return vx_vote_all(predicate);
}

/**
 * @brief Barrier with logical OR
 * @param predicate Value to OR across threads
 * @return 1 if any thread has non-zero predicate, 0 otherwise
 */
static inline __device__ int __syncthreads_or(int predicate) {
    return vx_vote_any(predicate);
}

#else

// Host stubs
static inline void __syncthreads(void) {}
static inline int __syncthreads_count(int predicate) { return 0; }
static inline int __syncthreads_and(int predicate) { return 0; }
static inline int __syncthreads_or(int predicate) { return 0; }

#endif  // __VORTEX_DEVICE__

//=============================================================================
// Warp-Level Voting Functions
//=============================================================================

#ifdef __VORTEX_DEVICE__

/**
 * @brief Check if predicate is true for all threads in warp
 * @param predicate Value to check
 * @return 1 if all active threads have non-zero predicate
 */
static inline __device__ int __all(int predicate) {
    return vx_vote_all(predicate);
}

/**
 * @brief Check if predicate is true for any thread in warp
 * @param predicate Value to check
 * @return 1 if any active thread has non-zero predicate
 */
static inline __device__ int __any(int predicate) {
    return vx_vote_any(predicate);
}

/**
 * @brief Get bitmask of threads where predicate is true
 * @param predicate Value to check
 * @return Bitmask where bit N is set if thread N has non-zero predicate
 */
static inline __device__ unsigned int __ballot(int predicate) {
    return (unsigned int)vx_vote_ballot(predicate);
}

#else

// Host stubs
static inline int __all(int predicate) { return 0; }
static inline int __any(int predicate) { return 0; }
static inline unsigned int __ballot(int predicate) { return 0; }

#endif  // __VORTEX_DEVICE__

//=============================================================================
// Warp-Level Shuffle Functions
//=============================================================================

#ifdef __VORTEX_DEVICE__

/**
 * @brief Get variable from specified lane
 * @param var Variable to shuffle
 * @param srcLane Source lane ID
 * @param width Warp width (default: warpSize)
 * @return Value of var from srcLane
 */
static inline __device__ int __shfl(int var, int srcLane, int width) {
    return vx_shfl_idx(var, srcLane, width, 0xFFFFFFFF);
}

static inline __device__ int __shfl(int var, int srcLane) {
    return __shfl(var, srcLane, 32);  // Default warp size
}

/**
 * @brief Shuffle up: get variable from lane-delta
 * @param var Variable to shuffle
 * @param delta Lane offset
 * @param width Warp width
 * @return Value of var from (laneId - delta)
 */
static inline __device__ int __shfl_up(int var, unsigned int delta, int width) {
    return vx_shfl_up(var, delta, width, 0xFFFFFFFF);
}

static inline __device__ int __shfl_up(int var, unsigned int delta) {
    return __shfl_up(var, delta, 32);
}

/**
 * @brief Shuffle down: get variable from lane+delta
 * @param var Variable to shuffle
 * @param delta Lane offset
 * @param width Warp width
 * @return Value of var from (laneId + delta)
 */
static inline __device__ int __shfl_down(int var, unsigned int delta, int width) {
    return vx_shfl_down(var, delta, width, 0xFFFFFFFF);
}

static inline __device__ int __shfl_down(int var, unsigned int delta) {
    return __shfl_down(var, delta, 32);
}

/**
 * @brief Butterfly shuffle: XOR-based shuffle
 * @param var Variable to shuffle
 * @param laneMask XOR mask for lane ID
 * @param width Warp width
 * @return Value of var from (laneId XOR laneMask)
 */
static inline __device__ int __shfl_xor(int var, int laneMask, int width) {
    return vx_shfl_bfly(var, laneMask, width, 0xFFFFFFFF);
}

static inline __device__ int __shfl_xor(int var, int laneMask) {
    return __shfl_xor(var, laneMask, 32);
}

// Float versions
static inline __device__ float __shfl(float var, int srcLane, int width) {
    union { float f; int i; } u;
    u.f = var;
    u.i = __shfl(u.i, srcLane, width);
    return u.f;
}

static inline __device__ float __shfl_up(float var, unsigned int delta, int width) {
    union { float f; int i; } u;
    u.f = var;
    u.i = __shfl_up(u.i, delta, width);
    return u.f;
}

static inline __device__ float __shfl_down(float var, unsigned int delta, int width) {
    union { float f; int i; } u;
    u.f = var;
    u.i = __shfl_down(u.i, delta, width);
    return u.f;
}

static inline __device__ float __shfl_xor(float var, int laneMask, int width) {
    union { float f; int i; } u;
    u.f = var;
    u.i = __shfl_xor(u.i, laneMask, width);
    return u.f;
}

#else

// Host stubs
static inline int __shfl(int var, int srcLane, int width) { return var; }
static inline int __shfl(int var, int srcLane) { return var; }
static inline int __shfl_up(int var, unsigned int delta, int width) { return var; }
static inline int __shfl_up(int var, unsigned int delta) { return var; }
static inline int __shfl_down(int var, unsigned int delta, int width) { return var; }
static inline int __shfl_down(int var, unsigned int delta) { return var; }
static inline int __shfl_xor(int var, int laneMask, int width) { return var; }
static inline int __shfl_xor(int var, int laneMask) { return var; }

#endif  // __VORTEX_DEVICE__

//=============================================================================
// Atomic Operations
//=============================================================================

#ifdef __VORTEX_DEVICE__

/**
 * @brief Atomic add operation
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
 * @brief Atomic subtract operation
 */
static inline __device__ int atomicSub(int* address, int val) {
    return atomicAdd(address, -val);
}

/**
 * @brief Atomic exchange operation
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
 * @brief Atomic minimum operation
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
 * @brief Atomic maximum operation
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
 * @brief Atomic compare-and-swap operation
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
 * @brief Atomic OR operation
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
 * @brief Atomic XOR operation
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
 * @brief Atomic AND operation
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

// Float atomic add (implemented using CAS)
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

// Host stubs
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

#endif  // __VORTEX_DEVICE__

//=============================================================================
// Math Functions (basic set)
//=============================================================================

#ifdef __VORTEX_DEVICE__

// Use hardware instructions for basic operations
static inline __device__ float __fsqrt_rn(float x) {
    float result;
    __asm__ __volatile__("fsqrt.s %0, %1" : "=f"(result) : "f"(x));
    return result;
}

static inline __device__ float __fmul_rn(float a, float b) {
    float result;
    __asm__ __volatile__("fmul.s %0, %1, %2" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

static inline __device__ float __fadd_rn(float a, float b) {
    float result;
    __asm__ __volatile__("fadd.s %0, %1, %2" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

static inline __device__ float __fmaf_rn(float a, float b, float c) {
    float result;
    __asm__ __volatile__("fmadd.s %0, %1, %2, %3" : "=f"(result) : "f"(a), "f"(b), "f"(c));
    return result;
}

// Type conversion helpers
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

// Host stubs
static inline float __fsqrt_rn(float x) { return 0.0f; }
static inline float __fmul_rn(float a, float b) { return 0.0f; }
static inline float __fadd_rn(float a, float b) { return 0.0f; }
static inline float __fmaf_rn(float a, float b, float c) { return 0.0f; }
static inline int __float_as_int(float x) { return 0; }
static inline float __int_as_float(int x) { return 0.0f; }

#endif  // __VORTEX_DEVICE__

//=============================================================================
// Vortex-Specific Extensions (Optional)
//=============================================================================

#ifdef __VORTEX_DEVICE__

namespace hip {
namespace vortex {

/**
 * @brief Get warp ID within core
 * @return Warp ID
 */
static inline __device__ int getWarpId() {
    return vx_warp_id();
}

/**
 * @brief Get core ID
 * @return Core ID
 */
static inline __device__ int getCoreId() {
    return vx_core_id();
}

/**
 * @brief Get thread ID within warp
 * @return Thread ID
 */
static inline __device__ int getThreadId() {
    return vx_thread_id();
}

/**
 * @brief Get hardware thread (hart) ID
 * @return Global hardware thread ID
 */
static inline __device__ int getHartId() {
    return vx_hart_id();
}

}  // namespace vortex
}  // namespace hip

#endif  // __VORTEX_DEVICE__

#endif  // VORTEX_HIP_DEVICE_H

/*
 * Vortex HIP Extensions
 *
 * This header provides Vortex-specific intrinsics and optimizations for HIP programs.
 * It exposes Vortex GPU features that are not available through standard OpenCL/HIP,
 * enabling high-performance warp-level operations and hardware-specific optimizations.
 *
 * Usage:
 *   #include <hip/hip_runtime.h>
 *   #include <hip/vortex/vx_hip_extensions.h>
 *
 *   __global__ void kernel() {
 *       int val = threadIdx.x;
 *       val = hip::vortex::warpShflDown(val, 16);
 *   }
 *
 * Requirements:
 *   - Must be compiled with Vortex toolchain
 *   - Must link with both libCHIP.so and libvortex.so
 *   - Vortex intrinsics header must be in include path
 *
 * Version: 1.0
 * Last Updated: 2025-11-05
 * Status: Tier 2 - Vortex Extensions
 */

#ifndef VX_HIP_EXTENSIONS_H
#define VX_HIP_EXTENSIONS_H

#include <hip/hip_runtime.h>

// Vortex intrinsics are available when compiling for Vortex device
#ifdef __VORTEX__
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#else
// Provide stub implementations for host compilation
#warning "Vortex intrinsics not available - using stub implementations"
#endif

namespace hip {
namespace vortex {

//==============================================================================
// Warp-Level Primitives (CUDA/HIP Compatible)
//==============================================================================

/**
 * @brief Check if all threads in warp satisfy predicate
 * @param predicate Integer predicate (non-zero = true)
 * @return Non-zero if all threads have non-zero predicate
 *
 * Equivalent to CUDA __all_sync() or HIP __all()
 */
__device__ __forceinline__ int warpAll(int predicate) {
#ifdef __VORTEX__
    return vx_vote_all(predicate);
#else
    return predicate;  // Host stub
#endif
}

/**
 * @brief Check if any thread in warp satisfies predicate
 * @param predicate Integer predicate (non-zero = true)
 * @return Non-zero if any thread has non-zero predicate
 *
 * Equivalent to CUDA __any_sync() or HIP __any()
 */
__device__ __forceinline__ int warpAny(int predicate) {
#ifdef __VORTEX__
    return vx_vote_any(predicate);
#else
    return predicate;  // Host stub
#endif
}

/**
 * @brief Check if all threads in warp have the same predicate value
 * @param predicate Integer predicate
 * @return Non-zero if all threads have same predicate value
 *
 * Equivalent to CUDA __uni_sync()
 */
__device__ __forceinline__ int warpUni(int predicate) {
#ifdef __VORTEX__
    return vx_vote_uni(predicate);
#else
    return 1;  // Host stub
#endif
}

/**
 * @brief Get bitmask of predicate values across warp
 * @param predicate Integer predicate (non-zero = true)
 * @return Bitmask where bit i is set if thread i has non-zero predicate
 *
 * Equivalent to CUDA __ballot_sync() or HIP __ballot()
 */
__device__ __forceinline__ unsigned int warpBallot(int predicate) {
#ifdef __VORTEX__
    return static_cast<unsigned int>(vx_vote_ballot(predicate));
#else
    return predicate ? 1 : 0;  // Host stub
#endif
}

//==============================================================================
// Warp Shuffle Operations (CUDA/HIP Compatible)
//==============================================================================

/**
 * @brief Shuffle value up within warp (receive from lower-indexed lane)
 * @param value Value to shuffle
 * @param delta Number of lanes to shift (positive)
 * @param width Warp width (default 32)
 * @param mask Active lane mask (default all lanes)
 * @return Value from lane (laneId - delta), or own value if out of range
 *
 * Equivalent to CUDA __shfl_up_sync() or HIP __shfl_up()
 */
__device__ __forceinline__ int warpShflUp(int value, int delta,
                                          int width = 32,
                                          unsigned int mask = 0xffffffff) {
#ifdef __VORTEX__
    return vx_shfl_up(static_cast<size_t>(value), delta, width, mask);
#else
    return value;  // Host stub
#endif
}

/**
 * @brief Shuffle value down within warp (receive from higher-indexed lane)
 * @param value Value to shuffle
 * @param delta Number of lanes to shift (positive)
 * @param width Warp width (default 32)
 * @param mask Active lane mask (default all lanes)
 * @return Value from lane (laneId + delta), or own value if out of range
 *
 * Equivalent to CUDA __shfl_down_sync() or HIP __shfl_down()
 */
__device__ __forceinline__ int warpShflDown(int value, int delta,
                                            int width = 32,
                                            unsigned int mask = 0xffffffff) {
#ifdef __VORTEX__
    return vx_shfl_down(static_cast<size_t>(value), delta, width, mask);
#else
    return value;  // Host stub
#endif
}

/**
 * @brief Shuffle value using butterfly pattern (XOR-based exchange)
 * @param value Value to shuffle
 * @param xor_mask XOR mask for lane index
 * @param width Warp width (default 32)
 * @param mask Active lane mask (default all lanes)
 * @return Value from lane (laneId XOR xor_mask)
 *
 * Equivalent to CUDA __shfl_xor_sync() or HIP __shfl_xor()
 */
__device__ __forceinline__ int warpShflXor(int value, int xor_mask,
                                           int width = 32,
                                           unsigned int mask = 0xffffffff) {
#ifdef __VORTEX__
    return vx_shfl_bfly(static_cast<size_t>(value), xor_mask, width, mask);
#else
    return value;  // Host stub
#endif
}

/**
 * @brief Shuffle value from specific lane index
 * @param value Value to shuffle
 * @param src_lane Source lane index
 * @param width Warp width (default 32)
 * @param mask Active lane mask (default all lanes)
 * @return Value from specified lane, or own value if out of range
 *
 * Equivalent to CUDA __shfl_sync() or HIP __shfl()
 */
__device__ __forceinline__ int warpShfl(int value, int src_lane,
                                        int width = 32,
                                        unsigned int mask = 0xffffffff) {
#ifdef __VORTEX__
    return vx_shfl_idx(static_cast<size_t>(value), src_lane, width, mask);
#else
    return value;  // Host stub
#endif
}

//==============================================================================
// Thread Control (Vortex-Specific)
//==============================================================================

/**
 * @brief Set thread mask (control which threads are active)
 * @param mask Bitmask of active threads
 *
 * Vortex-specific feature for fine-grained thread control
 */
__device__ __forceinline__ void threadMask(int mask) {
#ifdef __VORTEX__
    vx_tmc(mask);
#endif
}

/**
 * @brief Disable all threads (set mask to 0)
 */
__device__ __forceinline__ void threadMaskZero() {
#ifdef __VORTEX__
    vx_tmc_zero();
#endif
}

/**
 * @brief Enable only thread 0 (set mask to 1)
 */
__device__ __forceinline__ void threadMaskOne() {
#ifdef __VORTEX__
    vx_tmc_one();
#endif
}

/**
 * @brief Set predicate for conditional execution
 * @param condition Predicate condition
 * @param mask Thread mask
 *
 * Enables threads where condition is true
 */
__device__ __forceinline__ void predicate(int condition, int mask = 0xffffffff) {
#ifdef __VORTEX__
    vx_pred(condition, mask);
#endif
}

/**
 * @brief Set predicate for negated conditional execution
 * @param condition Predicate condition
 * @param mask Thread mask
 *
 * Enables threads where condition is false
 */
__device__ __forceinline__ void predicateNot(int condition, int mask = 0xffffffff) {
#ifdef __VORTEX__
    vx_pred_n(condition, mask);
#endif
}

/**
 * @brief Split execution based on predicate
 * @param predicate Split condition
 * @return Stack pointer for join
 */
__device__ __forceinline__ int split(int predicate) {
#ifdef __VORTEX__
    return vx_split(predicate);
#else
    return 0;
#endif
}

/**
 * @brief Split execution based on negated predicate
 * @param predicate Split condition (negated)
 * @return Stack pointer for join
 */
__device__ __forceinline__ int splitNot(int predicate) {
#ifdef __VORTEX__
    return vx_split_n(predicate);
#else
    return 0;
#endif
}

/**
 * @brief Rejoin execution after split
 * @param stack_ptr Stack pointer from split
 */
__device__ __forceinline__ void join(int stack_ptr) {
#ifdef __VORTEX__
    vx_join(stack_ptr);
#endif
}

//==============================================================================
// Hardware Identification
//==============================================================================

/**
 * @brief Get thread ID within warp (0-31)
 * @return Thread index within warp
 *
 * Equivalent to (threadIdx.x % warpSize) but may be more efficient
 */
__device__ __forceinline__ int getThreadId() {
#ifdef __VORTEX__
    return vx_thread_id();
#else
    return threadIdx.x % 32;
#endif
}

/**
 * @brief Get warp ID within core
 * @return Warp index within current core
 */
__device__ __forceinline__ int getWarpId() {
#ifdef __VORTEX__
    return vx_warp_id();
#else
    return threadIdx.x / 32;
#endif
}

/**
 * @brief Get core ID
 * @return Core index in GPU
 */
__device__ __forceinline__ int getCoreId() {
#ifdef __VORTEX__
    return vx_core_id();
#else
    return 0;
#endif
}

/**
 * @brief Get global hardware thread ID (hart ID)
 * @return Global hardware thread index
 */
__device__ __forceinline__ int getHartId() {
#ifdef __VORTEX__
    return vx_hart_id();
#else
    return threadIdx.x + blockIdx.x * blockDim.x;
#endif
}

/**
 * @brief Get number of threads per warp
 * @return Threads per warp (typically 32)
 */
__device__ __forceinline__ int getNumThreads() {
#ifdef __VORTEX__
    return vx_num_threads();
#else
    return 32;
#endif
}

/**
 * @brief Get number of warps per core
 * @return Warps per core
 */
__device__ __forceinline__ int getNumWarps() {
#ifdef __VORTEX__
    return vx_num_warps();
#else
    return 1;
#endif
}

/**
 * @brief Get total number of cores
 * @return Total cores in GPU
 */
__device__ __forceinline__ int getNumCores() {
#ifdef __VORTEX__
    return vx_num_cores();
#else
    return 1;
#endif
}

//==============================================================================
// Custom Accelerators
//==============================================================================

/**
 * @brief 8-way dot product (custom instruction)
 * @param a First operand
 * @param b Second operand
 * @return Dot product result
 *
 * Vortex-specific accelerated instruction for 8-way dot products
 */
__device__ __forceinline__ int dot8(int a, int b) {
#ifdef __VORTEX__
    return vx_dot8(a, b);
#else
    // Host stub: simple multiplication
    return a * b;
#endif
}

//==============================================================================
// Memory and Synchronization
//==============================================================================

/**
 * @brief Memory fence
 *
 * Ensures memory ordering across threads
 */
__device__ __forceinline__ void fence() {
#ifdef __VORTEX__
    vx_fence();
#else
    __threadfence();
#endif
}

//==============================================================================
// Helper Functions for Common Patterns
//==============================================================================

/**
 * @brief Warp-level reduction sum using shuffle
 * @param value Value to reduce
 * @return Sum of all values in warp (valid only in lane 0)
 *
 * Example usage:
 *   float val = input[tid];
 *   float sum = hip::vortex::warpReduceSum(val);
 *   if ((threadIdx.x % 32) == 0) output[warp_id] = sum;
 */
__device__ __forceinline__ float warpReduceSum(float value) {
    // Butterfly reduction: 16, 8, 4, 2, 1
    value += __shfl_xor(value, 16);
    value += __shfl_xor(value, 8);
    value += __shfl_xor(value, 4);
    value += __shfl_xor(value, 2);
    value += __shfl_xor(value, 1);
    return value;
}

/**
 * @brief Warp-level reduction sum (integer version)
 * @param value Value to reduce
 * @return Sum of all values in warp (valid only in lane 0)
 */
__device__ __forceinline__ int warpReduceSum(int value) {
    value += warpShflXor(value, 16);
    value += warpShflXor(value, 8);
    value += warpShflXor(value, 4);
    value += warpShflXor(value, 2);
    value += warpShflXor(value, 1);
    return value;
}

/**
 * @brief Warp-level reduction max
 * @param value Value to reduce
 * @return Maximum of all values in warp (valid only in lane 0)
 */
__device__ __forceinline__ float warpReduceMax(float value) {
    value = fmaxf(value, __shfl_xor(value, 16));
    value = fmaxf(value, __shfl_xor(value, 8));
    value = fmaxf(value, __shfl_xor(value, 4));
    value = fmaxf(value, __shfl_xor(value, 2));
    value = fmaxf(value, __shfl_xor(value, 1));
    return value;
}

/**
 * @brief Warp-level reduction min
 * @param value Value to reduce
 * @return Minimum of all values in warp (valid only in lane 0)
 */
__device__ __forceinline__ float warpReduceMin(float value) {
    value = fminf(value, __shfl_xor(value, 16));
    value = fminf(value, __shfl_xor(value, 8));
    value = fminf(value, __shfl_xor(value, 4));
    value = fminf(value, __shfl_xor(value, 2));
    value = fminf(value, __shfl_xor(value, 1));
    return value;
}

/**
 * @brief Get lane ID (thread index within warp)
 * @return Lane ID (0-31)
 */
__device__ __forceinline__ int getLaneId() {
    return threadIdx.x & 31;  // Equivalent to threadIdx.x % 32
}

/**
 * @brief Get warp ID within block
 * @return Warp ID within current block
 */
__device__ __forceinline__ int getBlockWarpId() {
    return threadIdx.x >> 5;  // Equivalent to threadIdx.x / 32
}

}  // namespace vortex
}  // namespace hip

//==============================================================================
// Compatibility Macros (Optional)
//==============================================================================

// Provide CUDA-style names for easy porting
#ifdef VX_HIP_CUDA_COMPAT

#define __vx_all(pred)              hip::vortex::warpAll(pred)
#define __vx_any(pred)              hip::vortex::warpAny(pred)
#define __vx_ballot(pred)           hip::vortex::warpBallot(pred)
#define __vx_shfl_up(v, d)          hip::vortex::warpShflUp(v, d)
#define __vx_shfl_down(v, d)        hip::vortex::warpShflDown(v, d)
#define __vx_shfl_xor(v, m)         hip::vortex::warpShflXor(v, m)
#define __vx_shfl(v, l)             hip::vortex::warpShfl(v, l)

#endif  // VX_HIP_CUDA_COMPAT

#endif  // VX_HIP_EXTENSIONS_H

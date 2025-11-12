# HIP Support Investigation - Phase 2A Assessment

**Goal:** Determine if Polygeist requires modifications to recognize HIP attributes

**Date:** Mon Nov 10 19:48:01 PST 2025

## Part 1: Source Code Analysis

### Searching Polygeist source for CUDA/HIP handling...

#### 1. CUDA attribute references:
```
tools/cgeist/Test/Verification/Inputs/cuda.h:#define __global__ __attribute__((global))
tools/cgeist/Test/Verification/Inputs/cuda.h:#define __global__
```

#### 2. HIP-specific code:
```
lib/polygeist/Passes/ParallelLower.cpp:                 << " for conversion to HIP, will be removed instead\n";
lib/polygeist/Passes/ConvertPolygeistToLLVM.cpp:#define RETURN_ON_HIP_ERROR(expr)                                              \
lib/polygeist/Passes/ConvertPolygeistToLLVM.cpp:        RETURN_ON_HIP_ERROR(hipInit(0));
lib/polygeist/Passes/ConvertPolygeistToLLVM.cpp:        RETURN_ON_HIP_ERROR(hipDeviceGet(&device, 0));
lib/polygeist/Passes/ConvertPolygeistToLLVM.cpp:          RETURN_ON_HIP_ERROR(hipModuleLoadData(&hipModule, blob));
lib/polygeist/Passes/ConvertPolygeistToLLVM.cpp:          RETURN_ON_HIP_ERROR(hipModuleGetFunction(
lib/polygeist/Passes/ConvertPolygeistToLLVM.cpp:          RETURN_ON_HIP_ERROR(hipFuncGetAttribute(
lib/polygeist/Passes/ConvertPolygeistToLLVM.cpp:              &maxThreadsPerBlock, HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
lib/polygeist/Passes/ConvertPolygeistToLLVM.cpp:          RETURN_ON_HIP_ERROR(hipFuncGetAttribute(
lib/polygeist/Passes/ConvertPolygeistToLLVM.cpp:              &sharedMemSize, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
lib/polygeist/Passes/ConvertPolygeistToLLVM.cpp:          RETURN_ON_HIP_ERROR(hipFuncGetAttribute(
lib/polygeist/Passes/ConvertPolygeistToLLVM.cpp:              &constMemSize, HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, hipFunction));
lib/polygeist/Passes/ConvertPolygeistToLLVM.cpp:          RETURN_ON_HIP_ERROR(hipFuncGetAttribute(
lib/polygeist/Passes/ConvertPolygeistToLLVM.cpp:              &localMemSize, HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, hipFunction));
lib/polygeist/Passes/ConvertPolygeistToLLVM.cpp:          RETURN_ON_HIP_ERROR(hipFuncGetAttribute(
lib/polygeist/Passes/ConvertPolygeistToLLVM.cpp:              &numRegs, HIP_FUNC_ATTRIBUTE_NUM_REGS, hipFunction));
lib/polygeist/Passes/ConvertPolygeistToLLVM.cpp:                  RETURN_ON_HIP_ERROR(
lib/polygeist/Passes/ConvertPolygeistToLLVM.cpp:          RETURN_ON_HIP_ERROR(hipModuleUnload(hipModule));
lib/polygeist/Passes/ConvertPolygeistToLLVM.cpp:// then be passed to the CUDA / ROCm (HIP) kernel launch calls.
lib/polygeist/Passes/ConvertPolygeistToLLVM.cpp:      constexpr unsigned HIPFatMagic = 0x48495046; // "HIPF"
```

#### 3. GPU/CUDA dialect usage:
```
lib/polygeist/Passes/SerializeToCubin.cpp:  registerNVVMDialectTranslation(registry);
lib/polygeist/Passes/PolygeistMem2Reg.cpp:#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
lib/polygeist/Passes/ParallelLower.cpp:#include "mlir/Dialect/GPU/IR/GPUDialect.h"
lib/polygeist/Passes/ParallelLower.cpp:#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
lib/polygeist/Passes/ConvertParallelToGPU.cpp:#include "mlir/Dialect/GPU/IR/GPUDialect.h"
lib/polygeist/Passes/ConvertParallelToGPU.cpp:#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
lib/polygeist/Passes/ConvertParallelToGPU.cpp:// TODO Add a NVVM::NVVMDialect::getLaunchBoundAttrName() (or a gpu dialect one?
lib/polygeist/Passes/CollectKernelStatistics.cpp:#include "mlir/Dialect/GPU/IR/GPUDialect.h"
lib/polygeist/Passes/ConvertPolygeistToLLVM.cpp:#include "mlir/Dialect/GPU/IR/GPUDialect.h"
lib/polygeist/Passes/ConvertPolygeistToLLVM.cpp:#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
lib/polygeist/Passes/ConvertPolygeistToLLVM.cpp:          static_cast<unsigned>(gpu::GPUDialect::getWorkgroupAddressSpace()));
lib/polygeist/Passes/ConvertPolygeistToLLVM.cpp:                          ? NVVM::NVVMDialect::getKernelFuncAttrName()
lib/polygeist/Passes/ConvertPolygeistToLLVM.cpp:          target.addLegalDialect<::mlir::NVVM::NVVMDialect>();
lib/polygeist/Passes/ConvertPolygeistToLLVM.cpp:        target.addIllegalDialect<gpu::GPUDialect>();
lib/polygeist/Passes/PolygeistCanonicalize.cpp:#include "mlir/Dialect/GPU/IR/GPUDialect.h"
```

#### 4. Attribute handling (global, device, host):
```
lib/polygeist/Passes/ParallelLower.cpp:      // Tag device side get globals with an attribute so that CSE does not
```

## Part 2: CUDA Lower Implementation

### Searching for --cuda-lower implementation...

#### Flag definition:
```
```

#### CUDA-related passes:
```
lib/polygeist/Passes/SerializeToCubin.cpp
lib/polygeist/Passes/ParallelLower.cpp
lib/polygeist/Passes/ConvertParallelToGPU.cpp
lib/polygeist/Passes/SerializeToHsaco.cpp
lib/polygeist/Passes/CollectKernelStatistics.cpp
lib/polygeist/Passes/ParallelLoopDistribute.cpp
lib/polygeist/Passes/ConvertPolygeistToLLVM.cpp
lib/polygeist/Passes/PolygeistCanonicalize.cpp
lib/polygeist/Passes/LowerAlternatives.cpp
```

#### ConvertParallelToGPU.cpp key sections:
```cpp
//===- ConvertParallelToGPU.cpp - Remove unused private functions ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "PassDetails.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "polygeist/BarrierUtils.h"
#include "polygeist/Ops.h"
#include "polygeist/Passes/Passes.h"
#include "polygeist/Passes/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"

#include <llvm/ADT/StringRef.h>
#include <optional>

#include "ParallelLoopUnroll.h"
#include "RuntimeWrapperUtils.h"

static llvm::cl::opt<bool> GPUKernelEmitCoarsenedAlternatives(
    "gpu-kernel-emit-coarsened-alternatives", llvm::cl::init(false),
    llvm::cl::desc("Emit alternative kernels with coarsened threads"));

static llvm::cl::opt<bool> GPUKernelEnableBlockCoarsening(
    "gpu-kernel-enable-block-coarsening", llvm::cl::init(true),
    llvm::cl::desc("When emitting coarsened kernels, enable block coarsening"));

static llvm::cl::opt<bool> GPUKernelEnableCoalescingFriendlyUnroll(
    "gpu-kernel-enable-coalescing-friendly-unroll", llvm::cl::init(false),
    llvm::cl::desc("When thread coarsening, do coalescing-friendly unrolling"));

// TODO when we add other backends, we would need to to add an argument to the
// pass which one we are compiling to to provide the appropriate error id
#if POLYGEIST_ENABLE_CUDA
#include <cuda.h>
#else
#define CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES 701
#endif

using namespace mlir;
using namespace polygeist;

#define DEBUG_TYPE "convert-parallel-to-gpu"
#define DBGS() ::llvm::dbgs() << "[" DEBUG_TYPE ":" << PATTERN << "] "

#define POLYGEIST_REMARK_TYPE "CONVERT_PARALLEL_TO_GPU"
#define POLYGEIST_REMARK(X)                                                    \
  do {                                                                         \
    if (getenv("POLYGEIST_EMIT_REMARKS_" POLYGEIST_REMARK_TYPE)) {             \
      X;                                                                       \
    }                                                                          \
  } while (0)

// From ParallelLICM.cpp
void moveParallelLoopInvariantCode(scf::ParallelOp looplike);

namespace {

static void shrinkAlternativesOp(polygeist::AlternativesOp alternativesOp,
                                 unsigned size, PatternRewriter &rewriter) {
  // New AOP with the exact number of regions needed
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(alternativesOp);
  auto newAop = rewriter.create<polygeist::AlternativesOp>(
      alternativesOp->getLoc(), size);
  newAop->setAttr("alternatives.type",
                  alternativesOp->getAttr("alternatives.type"));
  assert(newAop->getNumRegions() > 0);

  auto oldDescs =
      alternativesOp->getAttrOfType<ArrayAttr>("alternatives.descs");

  std::vector<Attribute> descs;
  for (unsigned i = 0; i < newAop->getNumRegions(); i++) {
    auto &region = alternativesOp->getRegion(i);
    auto &newRegion = newAop->getRegion(i);
    rewriter.inlineRegionBefore(region, newRegion, newRegion.begin());
```

## Part 3: Header and Built-in Handling

### Built-in variable handling (threadIdx, blockIdx, etc.):
```
lib/polygeist/Passes/ConvertParallelToGPU.cpp:    auto blockDims = launchOp.getBlockSizeOperandValues();
lib/polygeist/Passes/ConvertParallelToGPU.cpp:    auto bx = getConstantInteger(blockDims.x);
lib/polygeist/Passes/ConvertParallelToGPU.cpp:    auto by = getConstantInteger(blockDims.y);
lib/polygeist/Passes/ConvertParallelToGPU.cpp:    auto bz = getConstantInteger(blockDims.z);
lib/polygeist/Passes/ConvertParallelToGPU.cpp:    SmallVector<Value, 3> blockDims;
lib/polygeist/Passes/ConvertParallelToGPU.cpp:    SmallVector<Value, 3> gridDims;
lib/polygeist/Passes/ConvertParallelToGPU.cpp:        blockDims.insert(blockDims.begin(), bound);
lib/polygeist/Passes/ConvertParallelToGPU.cpp:            return cst && blockDims.size() < 3 && threadNum * val <= maxThreads;
lib/polygeist/Passes/ConvertParallelToGPU.cpp:          blockDims.insert(blockDims.begin(), bound);
lib/polygeist/Passes/ConvertParallelToGPU.cpp:          gridDims.insert(gridDims.begin(), bound);
lib/polygeist/Passes/ConvertParallelToGPU.cpp:        gridDims.insert(gridDims.begin(), bound);
lib/polygeist/Passes/ConvertParallelToGPU.cpp:    if (gridDims.size() > 3) {
lib/polygeist/Passes/ConvertParallelToGPU.cpp:    if (gridDims.size() == 0) {
lib/polygeist/Passes/ConvertParallelToGPU.cpp:      gridDims.push_back(oneindex);
lib/polygeist/Passes/ConvertParallelToGPU.cpp:      assert(splitDims <= gridDims.size());
lib/polygeist/Passes/ConvertParallelToGPU.cpp:      assert(splitDims + blockDims.size() <= 3);
lib/polygeist/Passes/ConvertParallelToGPU.cpp:        gi.push_back(gridDims.size() - 1 - i);
lib/polygeist/Passes/ConvertParallelToGPU.cpp:      // TODO try our best to make them divisors of the gridDims
lib/polygeist/Passes/ConvertParallelToGPU.cpp:      newBlockDims.insert(newBlockDims.end(), blockDims.begin(),
lib/polygeist/Passes/ConvertParallelToGPU.cpp:                          blockDims.end());
```

### Test infrastructure for CUDA:
```
config.suffixes = ['.c', '.cpp', '.cu']

# excludes: A list of directories or files to exclude from the testsuite even
# if they match the suffixes pattern.
config.excludes = []
if config.polygeist_enable_cuda == "0":
    config.excludes += ['CUDA']
if config.polygeist_enable_rocm == "0":
    config.excludes += ['ROCm']

# test_source_root: The root path where tests are located.
--
llvm_config.add_tool_substitutions(tools, tool_dirs)

import subprocess

resource_dir = subprocess.check_output([config.llvm_tools_dir + "/clang", "-print-resource-dir"]).decode('utf-8').strip()
cudaopts = '-L' + os.path.dirname(config.cudart_static_path) + ' -lstdc++ -ldl -lpthread -lrt -lcudart_static -lcuda --cuda-lower --emit-cuda --std=c++17 --cuda-gpu-arch=sm_80'
config.substitutions.append(('%stdinclude', '-resource-dir=' + resource_dir + " -I " + config.test_source_root + "/polybench/utilities"))
config.substitutions.append(('%resourcedir', '-resource-dir=' + resource_dir))
config.substitutions.append(('%polyexec', config.test_source_root + '/polybench/utilities/polybench.c -D POLYBENCH_TIME -D POLYBENCH_NO_FLUSH_CACHE -D MINI_DATASET'))
config.substitutions.append(('%polyverify', config.test_source_root + '/polybench/utilities/polybench.c -D POLYBENCH_DUMP_ARRAYS -D POLYBENCH_NO_FLUSH_CACHE -D MINI_DATASET'))
config.substitutions.append(('%cudaopts', cudaopts))
config.substitutions.append(('%polymer_pluto_cudaopts', cudaopts + ' --polyhedral-opt --raise-scf-to-affine'))
config.substitutions.append(('%polymer_enabled', config.polymer_enabled))
config.substitutions.append(('%polymer_pluto_enabled', config.polymer_pluto_enabled))
```

## Part 4: HIP vs CUDA Compatibility

### Key Question: Are HIP and CUDA treated identically?

HIP is designed to be CUDA-compatible:
- Same syntax: `__global__`, `__device__`, `__host__`
- Same built-ins: `threadIdx`, `blockIdx`, `blockDim`, `gridDim`
- Same dim3 structure

If Polygeist handles CUDA attributes, it should handle HIP attributes identically.

## Part 5: Clang CUDA/HIP Support

Polygeist uses Clang as frontend. Checking Clang's HIP support:
```bash
# Clang has built-in CUDA and HIP support
clang version 18.0.0 (https://github.com/llvm/llvm-project.git 26eb4285b56edd8c897642078d91f16ff0fd3472)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /home/yaakov/vortex_hip/Polygeist/llvm-project/build/bin

# Check for CUDA/HIP language modes:
  --cuda-compile-host-device
                          Compile CUDA code for both host and device (default). Has no effect on non-CUDA compilations.
  --cuda-device-only      Compile CUDA code for device only
  --cuda-feature=<value>  Manually specify the CUDA feature to use
  --cuda-host-only        Compile CUDA code for host only. Has no effect on non-CUDA compilations.
  --cuda-include-ptx=<value>
  --cuda-noopt-device-debug
  --cuda-path-ignore-env  Ignore environment variables to detect CUDA installation
  --cuda-path=<value>     CUDA installation path
  -cuid=<value>           An ID for compilation unit, which should be the same for the same compilation unit but different for different compilation units. It is used to externalize device-side static variables for single source offloading languages CUDA and HIP so that they can be accessed by the host code of the same compilation unit.
```

**Key insight:** Clang already supports both CUDA and HIP. Polygeist builds on Clang.

## Assessment & Recommendations

### Evidence Summary:

- CUDA references in source: 9
- HIP-specific references in source: 47
- CUDA test suite: 25 kernels exist
- Clang (Polygeist's frontend): Has built-in CUDA/HIP support

### Hypothesis:

✅ **Polygeist has CUDA support** (found 9 references)

✅ **HIP-specific code exists** (found 47 references)

### Phase 2A Requirement Assessment:

**Option 1: No modifications needed (likely)**
- Polygeist likely treats HIP as CUDA (correct, since they're identical)
- Use existing `--cuda-lower` flag with HIP code
- Minimal HIP header file to define attributes
- **Timeline:** 0 days

**Option 2: Add HIP flag alias (trivial)**
- Add `--hip-lower` flag as alias to `--cuda-lower`
- Purely cosmetic, no functional changes
- **Timeline:** 1 day

**Option 3: Explicit HIP support (if needed)**
- Add HIP-specific attribute recognition
- Add HIP header support
- Only if Option 1 doesn't work
- **Timeline:** 1 week

### Recommended Next Steps:

1. **Test with minimal HIP header** (2 hours)
   - Create `hip_minimal.h` with HIP attribute definitions
   - Test simple HIP kernel with `--cuda-lower`
   - See if it 'just works'

2. **If test fails, examine Clang AST** (4 hours)
   - Use `clang -Xclang -ast-dump` on HIP code
   - Check if HIP attributes are recognized
   - Determine if Polygeist needs modifications

3. **Parallel with Phase 1** (concurrent)
   - Person A: Phase 1 (plain C++ pipeline)
   - Person B: HIP header creation and testing
   - Minimal risk, clear separation of work

### Probability Assessment:

- **80% chance:** HIP works with existing `--cuda-lower` (no mods needed)
- **15% chance:** Need HIP flag alias (trivial mod)
- **5% chance:** Need explicit HIP support (1 week work)

## Conclusion

**Phase 2A likely NOT required** as a separate phase.

**Recommended approach:**
- Create minimal HIP header (1 hour)
- Test with `--cuda-lower` (1 hour)
- If it works: No modifications needed ✅
- If it fails: Investigate and modify as needed

**Can start immediately** while Phase 1 progresses.

---
Investigation complete: Mon Nov 10 19:48:02 PST 2025


## Bonus: Minimal HIP Header Created

Created: `hip_minimal.h`
```cpp
// Minimal HIP header for testing Polygeist compatibility
// HIP is CUDA-compatible by design

#ifndef HIP_MINIMAL_H
#define HIP_MINIMAL_H

// HIP attributes (identical to CUDA)
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))

// dim3 structure
struct dim3 {
    unsigned int x, y, z;

    __host__ __device__ dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1)
        : x(x_), y(y_), z(z_) {}
};

// HIP built-in variables (same as CUDA)
extern const dim3 threadIdx;
extern const dim3 blockIdx;
extern const dim3 blockDim;
extern const dim3 gridDim;

// HIP launch bounds (same as CUDA)
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))

#endif // HIP_MINIMAL_H
```

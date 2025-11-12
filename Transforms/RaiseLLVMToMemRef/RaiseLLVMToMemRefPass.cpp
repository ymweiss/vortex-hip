//===- RaiseLLVMToMemRefPass.cpp - Main pass implementation -----*- C++ -*-===//
//
// This pass converts LLVM dialect memory operations to memref dialect:
// - llvm.getelementptr + llvm.load → memref.load
// - llvm.getelementptr + llvm.store → memref.store
// - !llvm.ptr → memref<?xT> (unknown dimension for pointer args)
// - llvm.mlir.global (constant) → memref.global (constant)
// - llvm.mlir.addressof @global → memref.get_global @global
//
//===----------------------------------------------------------------------===//

#include "Common.h"

namespace mlir {
namespace dsa {

#define GEN_PASS_DEF_RAISELLVMTOMEMREF
#include "dsa/Transforms/Passes.h.inc"

namespace {

struct RaiseLLVMToMemRefPass
    : public impl::RaiseLLVMToMemRefBase<RaiseLLVMToMemRefPass> {
  using RaiseLLVMToMemRefBase::RaiseLLVMToMemRefBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    // Step 1: Convert global constants at module level
    llvm::DenseMap<StringRef, Type> globalSymbolToMemRefType;
    if (failed(convertGlobalConstants(module, globalSymbolToMemRefType, builder))) {
      signalPassFailure();
      return;
    }

    // Step 2: Collect func.func operations with dsa_optimize attribute
    // Note: RaiseLLVMToCF already converted llvm.func → func.func
    SmallVector<func::FuncOp> funcsToProcess;
    module.walk([&](func::FuncOp funcOp) {
      if (funcOp->hasAttr("dsa_optimize")) {
        funcsToProcess.push_back(funcOp);
      }
    });

    // Step 3: Convert memory operations in each function
    for (auto funcOp : funcsToProcess) {
      if (failed(convertLLVMMemoryOpsToMemRef(funcOp, globalSymbolToMemRefType, builder))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  LogicalResult convertLLVMMemoryOpsToMemRef(
      func::FuncOp funcOp,
      llvm::DenseMap<StringRef, Type> &globalSymbolToMemRefType,
      OpBuilder &builder) {
    // Step 1: Analyze argument usage patterns and classify them
    llvm::DenseMap<Value, ArgumentUsageInfo> usageInfo;
    Block &entryBlock = funcOp.getBody().front();

    for (BlockArgument arg : entryBlock.getArguments()) {
      if (isa<LLVM::LLVMPointerType>(arg.getType())) {
        usageInfo[arg] = analyzeArgumentUsage(arg);
      }
    }

    // Step 2: Build mapping from !llvm.ptr arguments to memref types
    llvm::DenseMap<Value, Type> ptrToMemRefType;
    if (failed(inferMemRefTypesForArgs(funcOp, ptrToMemRefType))) {
      return failure();
    }

    // Store inferred types in usageInfo for later use
    for (auto &[arg, info] : usageInfo) {
      info.inferredMemRefType = ptrToMemRefType.lookup(arg);
    }

    // Step 2.6: Infer types for llvm.alloca operations
    // IMPORTANT: This must happen BEFORE refineTypesFromMemcpyAlignment so that
    // alloca types are available for propagation through memcpy operations
    if (failed(inferTypesForAllocas(funcOp, ptrToMemRefType))) {
      return failure();
    }

    // Step 2.7: Infer types for heap allocations (C++ new[], malloc)
    // Also run before memcpy refinement for the same reason
    if (failed(inferTypesForHeapAllocations(funcOp, ptrToMemRefType))) {
      return failure();
    }

    // Step 2.5: Refine types from memcpy/memset alignment attributes
    // This must happen AFTER inferring alloca/heap types, because:
    // 1. inferMemRefTypesForArgs() may default to i8 for arguments only used in memcpy
    // 2. Alloca/heap allocations know their exact types (e.g., f32)
    // 3. We can propagate specific types from alloca to function args through memcpy
    // 4. This prevents ambiguous alignment-based inference (align 4 = i32 or f32?)
    // Note: We DON'T update BlockArgument types yet - LLVM operations still need !llvm.ptr
    if (failed(refineTypesFromMemcpyAlignment(funcOp, ptrToMemRefType))) {
      return failure();
    }

    // Step 3: Convert llvm.mlir.addressof by adding global memrefs as function arguments
    llvm::DenseMap<Value, Type> globalPtrToMemRefType;
    if (failed(convertAddressOfToArgs(funcOp, globalSymbolToMemRefType,
                                       globalPtrToMemRefType, builder))) {
      return failure();
    }

    // Merge global memrefs into ptrToMemRefType for unified handling
    for (auto &[val, type] : globalPtrToMemRefType) {
      ptrToMemRefType[val] = type;
    }

    // Step 4: Convert standalone GEP chains FIRST (before convertMemoryOperations)
    // This prevents use-after-free bugs where convertMemoryOperations deletes GEPs
    // that are part of multi-GEP chains being tracked by convertGEPChains.
    // Reason: multi-GEP chains (arg → GEP1 → GEP2 → load) have the last GEP (GEP2)
    // satisfy convertMemoryOperations' criteria (hasOneUse + user is load/store),
    // so convertMemoryOperations would delete GEP2, then convertGEPChains crashes
    // when trying to access the stored Operation* pointer to GEP2.
    if (failed(convertGEPChains(funcOp, usageInfo, ptrToMemRefType, builder))) {
      return failure();
    }

    // Step 5: Convert simple GEP+load and GEP+store pairs (remaining single-GEP patterns)
    // After convertGEPChains runs, this handles GEPs from other sources (alloca, heap, etc.)
    if (failed(convertMemoryOperations(funcOp, ptrToMemRefType, globalPtrToMemRefType, builder))) {
      return failure();
    }

    // Step 6: Convert direct load operations (without GEP)
    if (failed(convertDirectLoadOperations(funcOp, ptrToMemRefType, builder))) {
      return failure();
    }

    // Step 7: Convert direct store operations (without GEP)
    if (failed(convertDirectStoreOperations(funcOp, ptrToMemRefType, builder))) {
      return failure();
    }

    // Step 6.5: Convert GEPs with multiple uses
    // This catches GEPs missed by earlier steps (e.g., GEPs used by both load and store)
    if (failed(convertMultiUseGEPs(funcOp, ptrToMemRefType, builder))) {
      return failure();
    }

    // Step 7: Convert llvm.intr.memset to SCF loops
    if (failed(convertMemsetOperations(funcOp, ptrToMemRefType, builder))) {
      return failure();
    }

    // Step 8: Convert llvm.intr.memcpy to SCF loops
    if (failed(convertMemcpyOperations(funcOp, ptrToMemRefType, builder))) {
      return failure();
    }

    // Step 8.5: Convert GEP chains originating from allocas
    // This must happen BEFORE converting allocas to memref, otherwise we'll have
    // LLVM GEPs trying to use memref values
    if (failed(convertAllocaGEPChains(funcOp, ptrToMemRefType, builder))) {
      return failure();
    }

    // Step 9: Convert llvm.alloca to memref.alloca
    if (failed(convertAllocaToMemRefAlloca(funcOp, ptrToMemRefType, builder))) {
      return failure();
    }

    // Step 9.5: Convert heap allocations (C++ new[]) to memref.alloc
    if (failed(convertHeapAllocationsToMemRefAlloc(funcOp, ptrToMemRefType, builder))) {
      return failure();
    }

    // Step 9.6: Convert heap deallocations (C++ delete[]) to memref.dealloc
    if (failed(convertHeapDeallocations(funcOp, builder))) {
      return failure();
    }

    // Step 10: Update function signature to use memref types
    // This MUST be done last, after all LLVM operations are converted to memref
    if (failed(convertFunctionSignature(funcOp, ptrToMemRefType, builder))) {
      return failure();
    }

    return success();
  }
};

} // namespace

std::unique_ptr<Pass> createRaiseLLVMToMemRefPass() {
  return std::make_unique<RaiseLLVMToMemRefPass>();
}

} // namespace dsa
} // namespace mlir

//===- AllocaConversion.cpp - Alloca and heap conversion --------*- C++ -*-===//
//
// Allocation and deallocation conversion logic (stack and heap)
//
//===----------------------------------------------------------------------===//

#include "Common.h"

namespace mlir {
namespace dsa {

//===----------------------------------------------------------------------===//
// Heap Allocation Conversion
//===----------------------------------------------------------------------===//

/// Convert C++ new[]/new to memref.alloc
/// Handles: llvm.call @_Znam(size) → memref.alloc
LogicalResult convertHeapAllocationsToMemRefAlloc(
    func::FuncOp funcOp,
    llvm::DenseMap<Value, Type> &ptrToMemRefType,
    OpBuilder &builder) {

  SmallVector<LLVM::CallOp> heapAllocsToConvert;

  // Collect heap allocation calls
  funcOp.walk([&](LLVM::CallOp callOp) {
    auto callee = callOp.getCallee();
    if (!callee) return;

    StringRef calleeName = *callee;
    if (calleeName == "_Znam" || calleeName == "_Znwm") {
      heapAllocsToConvert.push_back(callOp);
    }
  });

  for (auto callOp : heapAllocsToConvert) {
    Value llvmPtr = callOp.getResult();

    // Skip if not in our mapping (shouldn't happen, but be safe)
    if (!ptrToMemRefType.count(llvmPtr))
      continue;

    Type memrefType = ptrToMemRefType[llvmPtr];
    auto memrefShapeType = cast<MemRefType>(memrefType);
    Type elementType = memrefShapeType.getElementType();

    // Get size parameter (first argument to _Znam)
    Value sizeInBytes = callOp.getOperand(0);
    Location loc = callOp.getLoc();

    builder.setInsertionPoint(callOp);

    // Calculate number of elements: sizeInBytes / sizeof(element)
    unsigned elementBitWidth = elementType.getIntOrFloatBitWidth();
    unsigned elementSizeBytes = (elementBitWidth + 7) / 8;

    // Convert sizeInBytes to i64 if needed
    Value size64 = sizeInBytes;
    if (sizeInBytes.getType() != builder.getI64Type()) {
      size64 = arith::IndexCastUIOp::create(builder, loc, builder.getI64Type(), sizeInBytes);
    }

    Value elemSize = arith::ConstantOp::create(builder, 
        loc, builder.getI64Type(), builder.getI64IntegerAttr(elementSizeBytes));
    Value numElements64 = arith::DivUIOp::create(builder, loc, size64, elemSize);

    // Convert to index type
    Value numElementsIndex = arith::IndexCastOp::create(builder, 
        loc, builder.getIndexType(), numElements64);

    // Create memref.alloc with dynamic size
    SmallVector<Value, 1> dynamicSizes = {numElementsIndex};
    SmallVector<Value, 0> symbolOperands = {};
    auto allocOp = memref::AllocOp::create(builder, 
        loc, cast<MemRefType>(memrefType), dynamicSizes, symbolOperands);

    Value memrefValue = allocOp.getResult();

    // Replace all uses of the LLVM pointer with the memref
    llvmPtr.replaceAllUsesWith(memrefValue);

    // Erase the heap allocation call
    callOp.erase();
  }

  return success();
}

/// Convert heap deallocation calls (delete[], free) to memref.dealloc
/// Handles: llvm.call @_ZdaPv(ptr) → memref.dealloc
LogicalResult convertHeapDeallocations(
    func::FuncOp funcOp,
    OpBuilder &builder) {
  SmallVector<LLVM::CallOp> deallocsToConvert;

  funcOp.walk([&](LLVM::CallOp callOp) {
    auto callee = callOp.getCallee();
    if (!callee) return;

    StringRef calleeName = *callee;
    // _ZdaPv = operator delete[](void*)
    // _ZdlPv = operator delete(void*)
    // free = free(void*)
    if (calleeName == "_ZdaPv" || calleeName == "_ZdlPv" || calleeName == "free") {
      deallocsToConvert.push_back(callOp);
    }
  });

  for (auto callOp : deallocsToConvert) {
    // Get the pointer being freed (first argument)
    Value ptrToFree = callOp.getOperand(0);
    Location loc = callOp.getLoc();

    // Create memref.dealloc
    builder.setInsertionPoint(callOp);
    memref::DeallocOp::create(builder, loc, ptrToFree);

    // Erase the original delete call
    callOp.erase();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Alloca Conversion
//===----------------------------------------------------------------------===//

LogicalResult convertAllocaToMemRefAlloca(
    func::FuncOp funcOp,
    llvm::DenseMap<Value, Type> &ptrToMemRefType,
    OpBuilder &builder) {

  SmallVector<LLVM::AllocaOp> allocasToConvert;
  funcOp.walk([&](LLVM::AllocaOp allocaOp) {
    allocasToConvert.push_back(allocaOp);
  });

  // Erase lifetime intrinsics for allocas we're about to convert
  // These are just optimization hints and don't affect correctness
  // After conversion to memref, these intrinsics become invalid
  for (auto allocaOp : allocasToConvert) {
    Value llvmPtr = allocaOp.getResult();

    // Collect lifetime intrinsics that use this alloca
    SmallVector<Operation *> lifetimeOpsToErase;
    for (Operation *user : llvmPtr.getUsers()) {
      if (isa<LLVM::LifetimeStartOp, LLVM::LifetimeEndOp>(user)) {
        lifetimeOpsToErase.push_back(user);
      }
    }

    // Erase them
    for (Operation *op : lifetimeOpsToErase) {
      op->erase();
    }
  }

  // Track allocations that need deallocation at function exit
  SmallVector<std::pair<Value, Location>> allocsNeedingDealloc;

  Block &entryBlock = funcOp.getBody().front();

  for (auto allocaOp : allocasToConvert) {
    Value llvmPtr = allocaOp.getResult();

    // Skip if not in our mapping (shouldn't happen, but be safe)
    if (!ptrToMemRefType.count(llvmPtr))
      continue;

    Type memrefType = ptrToMemRefType[llvmPtr];
    auto memrefShapeType = cast<MemRefType>(memrefType);
    Value arraySize = allocaOp.getArraySize();

    // Check if memref type has static or dynamic dimensions
    SmallVector<Value, 1> dynamicSizes;
    bool isDynamic = memrefShapeType.isDynamicDim(0);

    // Convert ALL llvm.alloca operations to memref.alloc (heap allocation)
    // This is necessary because memref.alloca requires AutomaticAllocationScope trait,
    // which handshake.func doesn't have. We insert implicit memref.dealloc at function exit.
    builder.setInsertionPoint(allocaOp);

    if (isDynamic) {
      // Dynamic size: need to pass arraySize as dynamicSizes parameter
      // Convert array size to index type
      Value arraySizeIndex = arraySize;
      if (arraySize.getType() != builder.getIndexType()) {
        arraySizeIndex = arith::IndexCastOp::create(builder, 
            allocaOp.getLoc(), builder.getIndexType(), arraySize);
      }
      dynamicSizes.push_back(arraySizeIndex);
    }
    // else: static size, dynamicSizes remains empty

    // Create memref.alloc with alignment hint
    SmallVector<Value, 0> symbolOperands = {};
    IntegerAttr alignment;
    if (auto alignValue = allocaOp.getAlignment()) {
      alignment = builder.getI64IntegerAttr(*alignValue);
    }
    auto memrefAlloc = memref::AllocOp::create(builder, 
        allocaOp.getLoc(), memrefType, dynamicSizes, symbolOperands, alignment);
    Value memrefValue = memrefAlloc.getResult();

    // Track this allocation for deallocation insertion at function exit
    allocsNeedingDealloc.push_back({memrefValue, allocaOp.getLoc()});

    // Replace all uses of llvm.ptr with memref
    llvmPtr.replaceAllUsesWith(memrefValue);

    // Erase old alloca
    allocaOp.erase();
  }

  // Insert implicit dealloc operations at function exit points
  // This implements automatic scope-based deallocation for stack allocations
  //
  // CRITICAL FIX: For functions with multiple returns, we must NOT insert
  // duplicate deallocations. In SSA form, each allocation should be freed
  // exactly once along each control flow path.
  //
  // Strategy: Defer dealloc insertion to SCFToHandshakeDSA pass, which has
  // better understanding of control flow and can insert synchronized deallocs.
  // For now, we track allocations but do NOT insert dealloc here.
  // The SCFToHandshakeDSA pass will handle dealloc insertion with proper
  // memory operation synchronization.
  if (!allocsNeedingDealloc.empty()) {
    // Mark allocations as requiring deallocation by adding an attribute
    // The SCFToHandshakeDSA pass will read this attribute and insert
    // properly synchronized deallocations
    for (auto [allocValue, loc] : allocsNeedingDealloc) {
      if (auto allocOp = allocValue.getDefiningOp<memref::AllocOp>()) {
        // Add marker attribute to indicate this allocation needs deallocation
        allocOp->setAttr("dsa.needs_dealloc", builder.getUnitAttr());
      }
    }

    // Note: We do NOT insert memref.dealloc here to avoid the multiple-dealloc
    // problem. The dealloc will be inserted in SCFToHandshakeDSA pass after
    // conversion to handshake dialect, where we can properly synchronize with
    // memory operations and handle control flow correctly.
  }

  return success();
}

} // namespace dsa
} // namespace mlir

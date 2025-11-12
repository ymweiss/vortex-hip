//===- IntrinsicOps.cpp - LLVM intrinsic conversion -------------*- C++ -*-===//
//
// LLVM intrinsic operation (memcpy/memset) conversion to SCF loops
//
//===----------------------------------------------------------------------===//

#include "Common.h"

namespace mlir {
namespace dsa {

//===----------------------------------------------------------------------===//
// Intrinsic Operations Conversion
//===----------------------------------------------------------------------===//

LogicalResult convertMemsetOperations(
    func::FuncOp funcOp, llvm::DenseMap<Value, Type> &ptrToMemRefType,
    OpBuilder &builder) {

  // Collect llvm.intr.memset operations on memref arguments
  SmallVector<Operation *> memsetsToConvert;
  funcOp.walk([&](Operation *op) {
    if (auto memsetOp = dyn_cast<LLVM::MemsetOp>(op)) {
      Value dest = memsetOp.getDst();
      // Check if destination is a converted memref argument
      if (ptrToMemRefType.count(dest)) {
        memsetsToConvert.push_back(op);
      }
    }
  });

  // Convert each memset to a loop
  for (Operation *op : memsetsToConvert) {
    auto memsetOp = cast<LLVM::MemsetOp>(op);
    Value memref = memsetOp.getDst();
    Value fillValue = memsetOp.getVal();
    Value sizeInBytes = memsetOp.getLen();

    Type memrefType = ptrToMemRefType[memref];
    Type elementType = cast<MemRefType>(memrefType).getElementType();

    builder.setInsertionPoint(op);
    Location loc = op->getLoc();

    // Calculate number of elements: size_bytes / sizeof(element)
    auto elementBitWidth = elementType.getIntOrFloatBitWidth();
    auto elementSizeBytes = (elementBitWidth + 7) / 8;  // Round up to bytes
    Value elementSize = arith::ConstantOp::create(
        builder, loc, builder.getI64Type(), builder.getI64IntegerAttr(elementSizeBytes));

    // Convert sizeInBytes to index type
    Value size64 = sizeInBytes;
    if (sizeInBytes.getType() != builder.getI64Type()) {
      size64 = arith::IndexCastUIOp::create(builder, loc, builder.getI64Type(), sizeInBytes);
    }

    Value numElements64 = arith::DivUIOp::create(builder, loc, size64, elementSize);
    Value numElements = arith::IndexCastOp::create(builder, loc, builder.getIndexType(), numElements64);

    // Convert fill value to element type
    Value fillElement;
    if (isa<IntegerType>(elementType)) {
      // Zero-extend i8 fill value to element width
      if (fillValue.getType() != elementType) {
        fillElement = arith::ExtUIOp::create(builder, loc, elementType, fillValue);
      } else {
        fillElement = fillValue;
      }
    } else if (isa<FloatType>(elementType)) {
      // For float types, treat fillValue as bitcast
      auto fillInt = arith::ExtUIOp::create(builder, loc, builder.getIntegerType(elementBitWidth), fillValue);
      fillElement = arith::BitcastOp::create(builder, loc, elementType, fillInt);
    } else {
      // Unsupported type, skip
      continue;
    }

    // Generate loop: for (i = 0; i < numElements; i++) { memref[i] = fillElement; }
    Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
    Value one = arith::ConstantIndexOp::create(builder, loc, 1);

    auto forOp = scf::ForOp::create(builder, loc, zero, numElements, one);
    builder.setInsertionPointToStart(forOp.getBody());
    Value loopVar = forOp.getInductionVar();
    memref::StoreOp::create(builder, loc, fillElement, memref, ValueRange{loopVar});

    // Erase old memset operation
    builder.setInsertionPointAfter(forOp);
    op->erase();
  }

  return success();
}

LogicalResult convertMemcpyOperations(
    func::FuncOp funcOp, llvm::DenseMap<Value, Type> &ptrToMemRefType,
    OpBuilder &builder) {

  // Collect llvm.intr.memcpy operations where both src and dst are converted memrefs
  SmallVector<Operation *> memcpysToConvert;
  funcOp.walk([&](Operation *op) {
    if (auto memcpyOp = dyn_cast<LLVM::MemcpyOp>(op)) {
      Value dst = memcpyOp.getDst();
      Value src = memcpyOp.getSrc();
      // Only convert if BOTH destination and source are converted memref arguments
      // This ensures we don't mix LLVM and memref dialects in the loop
      if (ptrToMemRefType.count(dst) && ptrToMemRefType.count(src)) {
        memcpysToConvert.push_back(op);
      }
    }
  });

  // Convert each memcpy to a loop
  for (Operation *op : memcpysToConvert) {
    auto memcpyOp = cast<LLVM::MemcpyOp>(op);
    Value dst = memcpyOp.getDst();
    Value src = memcpyOp.getSrc();
    Value sizeInBytes = memcpyOp.getLen();

    // Get element type from destination memref (already refined by refineTypesFromMemcpyAlignment)
    Type dstMemrefType = ptrToMemRefType[dst];
    Type elementType = cast<MemRefType>(dstMemrefType).getElementType();

    builder.setInsertionPoint(op);
    Location loc = op->getLoc();

    // Calculate number of elements: size_bytes / sizeof(element)
    auto elementBitWidth = elementType.getIntOrFloatBitWidth();
    auto elementSizeBytes = (elementBitWidth + 7) / 8;
    Value elementSize = arith::ConstantOp::create(builder, 
        loc, builder.getI64Type(), builder.getI64IntegerAttr(elementSizeBytes));

    // Convert sizeInBytes to i64 if needed
    Value size64 = sizeInBytes;
    if (sizeInBytes.getType() != builder.getI64Type()) {
      size64 = arith::IndexCastUIOp::create(builder, loc, builder.getI64Type(), sizeInBytes);
    }

    Value numElements64 = arith::DivUIOp::create(builder, loc, size64, elementSize);
    Value numElements = arith::IndexCastOp::create(builder, loc, builder.getIndexType(), numElements64);

    // Generate loop: for (i = 0; i < numElements; i++) { dst[i] = src[i]; }
    Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
    Value one = arith::ConstantIndexOp::create(builder, loc, 1);

    auto forOp = scf::ForOp::create(builder, loc, zero, numElements, one);
    builder.setInsertionPointToStart(forOp.getBody());
    Value loopVar = forOp.getInductionVar();

    // Load from source memref
    Value loadedValue = memref::LoadOp::create(builder, loc, elementType, src, ValueRange{loopVar});

    // Store to destination memref
    memref::StoreOp::create(builder, loc, loadedValue, dst, ValueRange{loopVar});

    // Erase old memcpy operation
    builder.setInsertionPointAfter(forOp);
    op->erase();
  }

  return success();
}

} // namespace dsa
} // namespace mlir

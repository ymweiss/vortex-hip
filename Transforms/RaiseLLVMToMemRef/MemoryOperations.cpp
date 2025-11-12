//===- MemoryOperations.cpp - Memory load/store conversion ------*- C++ -*-===//
//
// Memory operation (load/store) conversion logic
//
//===----------------------------------------------------------------------===//

#include "Common.h"

namespace mlir {
namespace dsa {

//===----------------------------------------------------------------------===//
// Memory Operation Conversion
//===----------------------------------------------------------------------===//

LogicalResult convertMemoryOperations(
    func::FuncOp funcOp,
    llvm::DenseMap<Value, Type> &ptrToMemRefType,
    llvm::DenseMap<Value, Type> &globalPtrToMemRefType,
    OpBuilder &builder) {

  // Collect GEP operations that need conversion
  SmallVector<LLVM::GEPOp> gepsToConvert;
  funcOp.walk([&](LLVM::GEPOp gepOp) {
    // Check if GEP is used by load or store
    if (gepOp->hasOneUse()) {
      Operation *user = *gepOp->getUsers().begin();
      if (isa<LLVM::LoadOp, LLVM::StoreOp>(user)) {
        gepsToConvert.push_back(gepOp);
      }
    }
  });

  // Convert each GEP+load/store pair
  for (auto gepOp : gepsToConvert) {
    Operation *user = *gepOp->getUsers().begin();

    if (auto loadOp = dyn_cast<LLVM::LoadOp>(user)) {
      if (failed(convertGEPLoadPair(gepOp, loadOp, ptrToMemRefType,
                                    globalPtrToMemRefType, builder))) {
        return failure();
      }
    } else if (auto storeOp = dyn_cast<LLVM::StoreOp>(user)) {
      if (failed(convertGEPStorePair(gepOp, storeOp, ptrToMemRefType,
                                     globalPtrToMemRefType, builder))) {
        return failure();
      }
    }
  }

  return success();
}

LogicalResult convertGEPLoadPair(
    LLVM::GEPOp gepOp, LLVM::LoadOp loadOp,
    llvm::DenseMap<Value, Type> &ptrToMemRefType,
    llvm::DenseMap<Value, Type> &globalPtrToMemRefType,
    OpBuilder &builder) {
  // Extract base pointer and index
  Value basePtr = gepOp.getBase();

  // Get memref type for base pointer (check both argument and global maps)
  Type memrefType = ptrToMemRefType.lookup(basePtr);
  if (!memrefType) {
    memrefType = globalPtrToMemRefType.lookup(basePtr);
  }
  if (!memrefType) {
    // Base is neither a function argument nor a global - skip
    return success();
  }

  // Extract index from GEP
  // For simple case: gep base[idx], the index is operand 1
  if (gepOp.getDynamicIndices().size() != 1) {
    // Complex GEP with multiple indices - not supported yet
    return success();
  }

  Value index = gepOp.getDynamicIndices()[0];

  // Convert index to index type if necessary - use separate variable
  builder.setInsertionPoint(gepOp);
  Type indexType = builder.getIndexType();
  Value indexAsIndex = index;
  if (index.getType() != indexType) {
    indexAsIndex = arith::IndexCastOp::create(builder,
        gepOp.getLoc(), indexType, index);
  }

  // Handle byte offset to element index conversion
  // If GEP uses i8 element type (byte addressing), the index is a byte offset
  // and needs to be divided by the memref element size
  Type gepElemType = gepOp.getElemType();
  Type memrefElemType = cast<MemRefType>(memrefType).getElementType();
  if (gepElemType.isInteger(8)) {  // i8 means byte offset
    unsigned elemBitWidth = memrefElemType.getIntOrFloatBitWidth();
    unsigned elemSizeBytes = (elemBitWidth + 7) / 8;

    if (elemSizeBytes > 1) {
      // Divide byte offset by element size to get element index
      Value elemSize = arith::ConstantIndexOp::create(builder, gepOp.getLoc(), elemSizeBytes);
      indexAsIndex = arith::DivUIOp::create(builder, gepOp.getLoc(), indexAsIndex, elemSize);
    }
  }

  // Create memref.load
  // Extract element type from memref type
  Type elementType = cast<MemRefType>(memrefType).getElementType();
  auto memrefLoad = memref::LoadOp::create(builder, 
      gepOp.getLoc(), elementType, basePtr, ValueRange{indexAsIndex});

  // Replace load result with memref.load result
  loadOp.getResult().replaceAllUsesWith(memrefLoad.getResult());

  // Erase old operations
  loadOp.erase();
  gepOp.erase();

  return success();
}

LogicalResult convertGEPStorePair(
    LLVM::GEPOp gepOp, LLVM::StoreOp storeOp,
    llvm::DenseMap<Value, Type> &ptrToMemRefType,
    llvm::DenseMap<Value, Type> &globalPtrToMemRefType,
    OpBuilder &builder) {
  // Extract base pointer and index
  Value basePtr = gepOp.getBase();

  // Get memref type for base pointer (check both argument and global maps)
  Type memrefType = ptrToMemRefType.lookup(basePtr);
  if (!memrefType) {
    memrefType = globalPtrToMemRefType.lookup(basePtr);
  }
  if (!memrefType) {
    // Base is neither a function argument nor a global - skip
    return success();
  }

  // Extract index from GEP
  if (gepOp.getDynamicIndices().size() != 1) {
    // Complex GEP with multiple indices - not supported yet
    return success();
  }

  Value index = gepOp.getDynamicIndices()[0];
  Value valueToStore = storeOp.getValue();

  // Convert index to index type if necessary - use separate variable
  builder.setInsertionPoint(storeOp);
  Type indexType = builder.getIndexType();
  Value indexAsIndex = index;
  if (index.getType() != indexType) {
    indexAsIndex = arith::IndexCastOp::create(builder,
        gepOp.getLoc(), indexType, index);
  }

  // Handle byte offset to element index conversion
  // If GEP uses i8 element type (byte addressing), the index is a byte offset
  // and needs to be divided by the memref element size
  Type gepElemType = gepOp.getElemType();
  Type memrefElemType = cast<MemRefType>(memrefType).getElementType();
  if (gepElemType.isInteger(8)) {  // i8 means byte offset
    unsigned elemBitWidth = memrefElemType.getIntOrFloatBitWidth();
    unsigned elemSizeBytes = (elemBitWidth + 7) / 8;

    if (elemSizeBytes > 1) {
      // Divide byte offset by element size to get element index
      Value elemSize = arith::ConstantIndexOp::create(builder, gepOp.getLoc(), elemSizeBytes);
      indexAsIndex = arith::DivUIOp::create(builder, gepOp.getLoc(), indexAsIndex, elemSize);
    }
  }

  // Create memref.store
  memref::StoreOp::create(builder, 
      gepOp.getLoc(), valueToStore, basePtr, ValueRange{indexAsIndex});

  // Erase old operations
  storeOp.erase();
  gepOp.erase();

  return success();
}

//===----------------------------------------------------------------------===//
// Direct Load/Store Conversion (without GEP)
//===----------------------------------------------------------------------===//

LogicalResult convertDirectLoadOperations(
    func::FuncOp funcOp, llvm::DenseMap<Value, Type> &ptrToMemRefType,
    OpBuilder &builder) {

  // Collect direct load operations on memref arguments
  SmallVector<LLVM::LoadOp> loadsToConvert;
  funcOp.walk([&](LLVM::LoadOp loadOp) {
    Value addr = loadOp.getAddr();
    // Check if this address is a converted memref argument
    if (ptrToMemRefType.count(addr)) {
      loadsToConvert.push_back(loadOp);
    }
  });

  // Convert each direct load
  for (auto loadOp : loadsToConvert) {
    Value memref = loadOp.getAddr();
    Type memrefType = ptrToMemRefType[memref];

    // Create constant zero index for base pointer access
    builder.setInsertionPoint(loadOp);
    Value zeroIndex = arith::ConstantIndexOp::create(builder, loadOp.getLoc(), 0);

    // Create memref.load with zero index
    Type elementType = cast<MemRefType>(memrefType).getElementType();
    auto memrefLoad = memref::LoadOp::create(builder, 
        loadOp.getLoc(), elementType, memref, ValueRange{zeroIndex});

    // Replace load result with memref.load result
    loadOp.getResult().replaceAllUsesWith(memrefLoad.getResult());

    // Erase old operation
    loadOp.erase();
  }

  return success();
}

LogicalResult convertDirectStoreOperations(
    func::FuncOp funcOp, llvm::DenseMap<Value, Type> &ptrToMemRefType,
    OpBuilder &builder) {

  // Collect direct store operations on memref arguments
  SmallVector<LLVM::StoreOp> storesToConvert;
  funcOp.walk([&](LLVM::StoreOp storeOp) {
    Value addr = storeOp.getAddr();
    // Check if this address is a converted memref argument
    if (ptrToMemRefType.count(addr)) {
      storesToConvert.push_back(storeOp);
    }
  });

  // Convert each direct store
  for (auto storeOp : storesToConvert) {
    Value memref = storeOp.getAddr();
    Value valueToStore = storeOp.getValue();

    // Create constant zero index for base pointer access
    builder.setInsertionPoint(storeOp);
    Value zeroIndex = arith::ConstantIndexOp::create(builder, storeOp.getLoc(), 0);

    // Create memref.store with zero index
    memref::StoreOp::create(builder, 
        storeOp.getLoc(), valueToStore, memref, ValueRange{zeroIndex});

    // Erase old operation
    storeOp.erase();
  }

  return success();
}

} // namespace dsa
} // namespace mlir

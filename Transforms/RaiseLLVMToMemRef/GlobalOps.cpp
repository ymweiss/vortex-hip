//===- GlobalOps.cpp - Global constants and function ops --------*- C++ -*-===//
//
// Global constant conversion and function signature update logic
//
//===----------------------------------------------------------------------===//

#include "Common.h"

namespace mlir {
namespace dsa {

//===----------------------------------------------------------------------===//
// Global Constant Conversion
//===----------------------------------------------------------------------===//

/// Convert llvm.mlir.global constants to memref.global
LogicalResult convertGlobalConstants(
    ModuleOp module,
    llvm::DenseMap<StringRef, Type> &globalSymbolToMemRefType,
    OpBuilder &builder) {

  // Collect all llvm.mlir.global operations that are constant arrays
  SmallVector<LLVM::GlobalOp> globalsToConvert;
  module.walk([&](LLVM::GlobalOp globalOp) {
    // Only convert constant globals with array types
    if (globalOp.getConstant() && isa<LLVM::LLVMArrayType>(globalOp.getType())) {
      globalsToConvert.push_back(globalOp);
    }
  });

  // Convert each global
  for (auto llvmGlobal : globalsToConvert) {
    StringRef symbolName = llvmGlobal.getSymName();
    auto arrayType = cast<LLVM::LLVMArrayType>(llvmGlobal.getType());

    // Convert LLVM array type to memref type
    MemRefType memrefType = convertLLVMArrayToMemRef(arrayType);

    // Get the initializer attribute
    Attribute initializer = llvmGlobal.getValueOrNull();
    if (!initializer) {
      // Skip globals without initializers
      continue;
    }

    // Create memref.global with ".memref" suffix to avoid symbol conflicts
    // (LLVM global is kept for non-dsa_optimize functions)
    std::string memrefSymbolName = (symbolName + ".memref").str();
    builder.setInsertionPoint(llvmGlobal);
    auto memrefGlobal = memref::GlobalOp::create(builder, 
        llvmGlobal.getLoc(),
        memrefSymbolName,
        builder.getStringAttr("private"),
        memrefType,
        initializer,
        /*constant=*/true,
        /*alignment=*/nullptr);

    // Track the LLVM symbol → memref type mapping
    // (We'll use the original LLVM symbol name as the key)
    globalSymbolToMemRefType[symbolName] = memrefType;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AddressOf Conversion - Convert to memref.get_global
//===----------------------------------------------------------------------===//

/// Convert llvm.mlir.addressof to memref.get_global operations
/// This preserves globals as module-level constants rather than function arguments
/// The simulator loads these during initialization (like OS loading .rodata section)
LogicalResult convertAddressOfToArgs(
    func::FuncOp funcOp,
    llvm::DenseMap<StringRef, Type> &globalSymbolToMemRefType,
    llvm::DenseMap<Value, Type> &globalPtrToMemRefType,
    OpBuilder &builder) {

  // Collect all llvm.mlir.addressof operations that reference globals
  SmallVector<LLVM::AddressOfOp> addressOfsToConvert;
  funcOp.walk([&](LLVM::AddressOfOp addressOfOp) {
    StringRef symbolName = addressOfOp.getGlobalName();
    if (globalSymbolToMemRefType.count(symbolName)) {
      addressOfsToConvert.push_back(addressOfOp);
    }
  });

  if (addressOfsToConvert.empty()) {
    return success();  // No globals referenced
  }

  // Replace each llvm.mlir.addressof with memref.get_global
  for (auto addressOfOp : addressOfsToConvert) {
    StringRef symbolName = addressOfOp.getGlobalName();
    Type memrefType = globalSymbolToMemRefType[symbolName];

    // Create memref.get_global operation
    // Use the ".memref" suffix to match the memref.global created earlier
    std::string memrefSymbolName = (symbolName + ".memref").str();
    builder.setInsertionPoint(addressOfOp);
    auto getGlobalOp = builder.create<memref::GetGlobalOp>(
        addressOfOp.getLoc(),
        cast<MemRefType>(memrefType),
        memrefSymbolName);

    // Replace all uses and erase the addressof
    addressOfOp.getResult().replaceAllUsesWith(getGlobalOp.getResult());
    addressOfOp.erase();

    // Track the mapping: get_global result → memref type
    globalPtrToMemRefType[getGlobalOp.getResult()] = memrefType;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Function Signature Conversion
//===----------------------------------------------------------------------===//

LogicalResult convertFunctionSignature(
    func::FuncOp funcOp, llvm::DenseMap<Value, Type> &ptrToMemRefType,
    OpBuilder &builder) {

  Block &entryBlock = funcOp.getBody().front();
  SmallVector<Type> newArgTypes;
  SmallVector<Type> resultTypes;  // Keep existing result types

  // Build new argument type list and update argument types in-place
  for (BlockArgument arg : entryBlock.getArguments()) {
    Type newType = ptrToMemRefType.lookup(arg);
    if (newType) {
      newArgTypes.push_back(newType);
      // Update argument type in-place
      arg.setType(newType);
    } else {
      newArgTypes.push_back(arg.getType());
    }
  }

  // Get existing result types
  FunctionType oldFuncType = funcOp.getFunctionType();
  resultTypes.append(oldFuncType.getResults().begin(),
                     oldFuncType.getResults().end());

  // Create new function type
  auto newFuncType = builder.getFunctionType(newArgTypes, resultTypes);
  funcOp.setFunctionType(newFuncType);

  return success();
}

} // namespace dsa
} // namespace mlir

//===- Common.h - Shared declarations for RaiseLLVMToMemRef ----*- C++ -*-===//
//
// Common types, forward declarations and utilities for RaiseLLVMToMemRef
//
//===----------------------------------------------------------------------===//

#ifndef LIB_DSA_TRANSFORMS_RAISELLVMTOMEMREF_COMMON_H
#define LIB_DSA_TRANSFORMS_RAISELLVMTOMEMREF_COMMON_H

#include "dsa/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace dsa {

//===----------------------------------------------------------------------===//
// GEP Chain Analysis Data Structures
//===----------------------------------------------------------------------===//

/// Information about a chain of GEP operations leading to a terminal operation
struct GEPChainInfo {
  SmallVector<Operation*, 4> chain;     // GEPs from root to leaf (as Operation*)
  Value rootMemref;                     // The function argument (before conversion)
  Operation *terminalOp;                // The load/store/etc that consumes the chain
  bool crossesBlocks;                   // GEP results used across block boundaries
  bool hasMultipleUses;                 // Any GEP in chain has multiple uses
  bool hasTypeChange;                   // Element type changes in the chain

  GEPChainInfo() : rootMemref(nullptr), terminalOp(nullptr),
                   crossesBlocks(false), hasMultipleUses(false),
                   hasTypeChange(false) {}
};

/// Classification of how a pointer argument is used
enum class ArgumentCategory {
  SIMPLE,           // Only direct ops or simple GEP+load/store pairs
  SIMPLE_CHAINS,    // Has GEP chains but all terminate in load/store
  COMPLEX_CHAINS,   // Has GEP chains with complex control flow
  UNCONVERTIBLE     // Used as block argument, returned, etc.
};

/// Complete usage information for a pointer argument
struct ArgumentUsageInfo {
  Value argument;
  Type inferredMemRefType;
  SmallVector<GEPChainInfo, 4> gepChains;
  SmallVector<LLVM::LoadOp, 4> directLoads;
  SmallVector<LLVM::StoreOp, 4> directStores;
  ArgumentCategory category;

  ArgumentUsageInfo() : argument(nullptr), inferredMemRefType(nullptr),
                        category(ArgumentCategory::UNCONVERTIBLE) {}
};

//===----------------------------------------------------------------------===//
// Forward Declarations - Type Conversion and Inference
//===----------------------------------------------------------------------===//

MemRefType convertLLVMArrayToMemRef(LLVM::LLVMArrayType arrayType);

LogicalResult inferMemRefTypesForArgs(
    func::FuncOp funcOp, llvm::DenseMap<Value, Type> &ptrToMemRefType);

LogicalResult refineTypesFromMemcpyAlignment(
    func::FuncOp funcOp, llvm::DenseMap<Value, Type> &ptrToMemRefType);

LogicalResult inferTypesForAllocas(
    func::FuncOp funcOp, llvm::DenseMap<Value, Type> &ptrToMemRefType);

LogicalResult inferTypesForHeapAllocations(
    func::FuncOp funcOp, llvm::DenseMap<Value, Type> &ptrToMemRefType);

//===----------------------------------------------------------------------===//
// Forward Declarations - GEP Chain Analysis and Conversion
//===----------------------------------------------------------------------===//

void analyzeGEPChain(LLVM::GEPOp rootGEP, Value argument,
                     GEPChainInfo &chainInfo);

ArgumentUsageInfo analyzeArgumentUsage(BlockArgument arg);

LogicalResult convertGEPChain(GEPChainInfo &chainInfo, Type memrefType,
                              OpBuilder &builder);

LogicalResult convertGEPChains(
    func::FuncOp funcOp,
    llvm::DenseMap<Value, ArgumentUsageInfo> &usageInfo,
    llvm::DenseMap<Value, Type> &ptrToMemRefType,
    OpBuilder &builder);

LogicalResult convertMultiUseGEPs(
    func::FuncOp funcOp,
    llvm::DenseMap<Value, Type> &ptrToMemRefType,
    OpBuilder &builder);

LogicalResult convertAllocaGEPChains(
    func::FuncOp funcOp,
    llvm::DenseMap<Value, Type> &ptrToMemRefType,
    OpBuilder &builder);

//===----------------------------------------------------------------------===//
// Forward Declarations - Memory Operations
//===----------------------------------------------------------------------===//

LogicalResult convertMemoryOperations(
    func::FuncOp funcOp,
    llvm::DenseMap<Value, Type> &ptrToMemRefType,
    llvm::DenseMap<Value, Type> &globalPtrToMemRefType,
    OpBuilder &builder);

LogicalResult convertGEPLoadPair(
    LLVM::GEPOp gepOp, LLVM::LoadOp loadOp,
    llvm::DenseMap<Value, Type> &ptrToMemRefType,
    llvm::DenseMap<Value, Type> &globalPtrToMemRefType,
    OpBuilder &builder);

LogicalResult convertGEPStorePair(
    LLVM::GEPOp gepOp, LLVM::StoreOp storeOp,
    llvm::DenseMap<Value, Type> &ptrToMemRefType,
    llvm::DenseMap<Value, Type> &globalPtrToMemRefType,
    OpBuilder &builder);

LogicalResult convertDirectLoadOperations(
    func::FuncOp funcOp, llvm::DenseMap<Value, Type> &ptrToMemRefType,
    OpBuilder &builder);

LogicalResult convertDirectStoreOperations(
    func::FuncOp funcOp, llvm::DenseMap<Value, Type> &ptrToMemRefType,
    OpBuilder &builder);

//===----------------------------------------------------------------------===//
// Forward Declarations - Intrinsic Operations
//===----------------------------------------------------------------------===//

LogicalResult convertMemsetOperations(
    func::FuncOp funcOp, llvm::DenseMap<Value, Type> &ptrToMemRefType,
    OpBuilder &builder);

LogicalResult convertMemcpyOperations(
    func::FuncOp funcOp, llvm::DenseMap<Value, Type> &ptrToMemRefType,
    OpBuilder &builder);

//===----------------------------------------------------------------------===//
// Forward Declarations - Alloca and Heap Allocations
//===----------------------------------------------------------------------===//

LogicalResult convertAllocaToMemRefAlloca(
    func::FuncOp funcOp,
    llvm::DenseMap<Value, Type> &ptrToMemRefType,
    OpBuilder &builder);

LogicalResult convertHeapAllocationsToMemRefAlloc(
    func::FuncOp funcOp,
    llvm::DenseMap<Value, Type> &ptrToMemRefType,
    OpBuilder &builder);

LogicalResult convertHeapDeallocations(
    func::FuncOp funcOp,
    OpBuilder &builder);

//===----------------------------------------------------------------------===//
// Forward Declarations - Global Constants and Function Signature
//===----------------------------------------------------------------------===//

LogicalResult convertGlobalConstants(
    ModuleOp module,
    llvm::DenseMap<StringRef, Type> &globalSymbolToMemRefType,
    OpBuilder &builder);

LogicalResult convertAddressOfToArgs(
    func::FuncOp funcOp,
    llvm::DenseMap<StringRef, Type> &globalSymbolToMemRefType,
    llvm::DenseMap<Value, Type> &globalPtrToMemRefType,
    OpBuilder &builder);

LogicalResult convertFunctionSignature(
    func::FuncOp funcOp, llvm::DenseMap<Value, Type> &ptrToMemRefType,
    OpBuilder &builder);

} // namespace dsa
} // namespace mlir

#endif // LIB_DSA_TRANSFORMS_RAISELLVMTOMEMREF_COMMON_H

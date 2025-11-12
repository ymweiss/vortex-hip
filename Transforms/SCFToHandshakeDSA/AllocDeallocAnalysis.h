//===- AllocDeallocAnalysis.h - Allocation/deallocation analysis -*- C++ -*-===//
//
// Analyzes memory allocations and their deallocation requirements
//
//===----------------------------------------------------------------------===//

#ifndef LIB_DSA_TRANSFORMS_SCFTOHANDSHAKEDSA_ALLOCDEALLOCANALYSIS_H
#define LIB_DSA_TRANSFORMS_SCFTOHANDSHAKEDSA_ALLOCDEALLOCANALYSIS_H

#include "Common.h"

namespace mlir {
namespace dsa {

//===----------------------------------------------------------------------===//
// AllocDeallocTracker
// Tracks heap and stack allocations and their deallocation requirements
//===----------------------------------------------------------------------===//

class AllocDeallocTracker {
public:
  // Data structures for tracking heap allocations and deallocations
  DenseMap<Value, memref::DeallocOp> allocToDeallocMap;
  DenseMap<memref::DeallocOp, Operation*> deallocToScopeMap;
  DenseMap<Value, bool> memrefHasAccess;

  // Track stack allocations that need deallocation (marked by RaiseLLVMToMemRef)
  // Store memref types instead of operation pointers (ops get invalidated during conversion)
  SmallVector<Type> stackAllocTypesNeedingDealloc;

  // Analyze alloc/dealloc pairs in the module (before SCF conversion)
  void analyzeAllocDealloc(ModuleOp module, OpBuilder &builder);

  // Clear all tracking data
  void clear();
};

} // namespace dsa
} // namespace mlir

#endif // LIB_DSA_TRANSFORMS_SCFTOHANDSHAKEDSA_ALLOCDEALLOCANALYSIS_H

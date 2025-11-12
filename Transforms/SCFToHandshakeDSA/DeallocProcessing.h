//===- DeallocProcessing.h - Deallocation processing -----------*- C++ -*-===//
//
// Functions for processing and inserting deallocations with control dependencies
//
//===----------------------------------------------------------------------===//

#ifndef LIB_DSA_TRANSFORMS_SCFTOHANDSHAKEDSA_DEALLOCPROCESSING_H
#define LIB_DSA_TRANSFORMS_SCFTOHANDSHAKEDSA_DEALLOCPROCESSING_H

#include "Common.h"
#include "AllocDeallocAnalysis.h"

namespace mlir {
namespace dsa {

//===----------------------------------------------------------------------===//
// Deallocation Processing Functions
//===----------------------------------------------------------------------===//

// Process dealloc operations in a handshake function (after conversion)
// Adds control dependencies to ensure deallocs wait for memory operations
LogicalResult processDeallocsInFunction(
    circt::handshake::FuncOp funcOp,
    OpBuilder &builder,
    AllocDeallocTracker &tracker);

// Insert deallocations for stack allocations at function exit
// This handles allocations marked by RaiseLLVMToMemRef pass
LogicalResult insertStackDeallocations(
    circt::handshake::FuncOp funcOp,
    OpBuilder &builder,
    AllocDeallocTracker &tracker);

} // namespace dsa
} // namespace mlir

#endif // LIB_DSA_TRANSFORMS_SCFTOHANDSHAKEDSA_DEALLOCPROCESSING_H

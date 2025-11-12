//===- LoopTraits.h - Loop Traits for Generic Patterns ---------*- C++ -*-===//
//
// Type traits to abstract loop-specific operations for generic patterns.
// Allows single patterns to handle both scf.while and scf.for loops.
//
//===----------------------------------------------------------------------===//

#ifndef DSA_TRANSFORMS_NORMALIZESCFLOOPS_LOOPTRAITS_H
#define DSA_TRANSFORMS_NORMALIZESCFLOOPS_LOOPTRAITS_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace dsa {

/// Type traits to abstract loop-specific operations for generic zero-trip
/// check removal. This allows a single pattern to handle both scf.while
/// and scf.for loops wrapped in redundant zero-trip check scf.if operations.
template <typename LoopOpType>
struct LoopTraits;

/// Traits for scf::WhileOp
template <>
struct LoopTraits<scf::WhileOp> {
  /// Get initial values for loop-carried values
  static SmallVector<Value> getInitValues(scf::WhileOp op) {
    SmallVector<Value> inits;
    for (Value init : op.getInits())
      inits.push_back(init);
    return inits;
  }

  /// Get loop results
  static ResultRange getResults(scf::WhileOp op) {
    return op.getResults();
  }

  /// Check if loop has results (always true for while)
  static bool hasResults(scf::WhileOp op) {
    return op.getNumResults() > 0;
  }

  /// Get the loop operation name for error messages
  static StringRef getLoopName() { return "scf.while"; }
};

/// Traits for scf::ForOp
template <>
struct LoopTraits<scf::ForOp> {
  /// Get initial values for loop-carried values (iter_args)
  static SmallVector<Value> getInitValues(scf::ForOp op) {
    SmallVector<Value> inits;
    for (Value init : op.getInitArgs())
      inits.push_back(init);
    return inits;
  }

  /// Get loop results
  static ResultRange getResults(scf::ForOp op) {
    return op.getResults();
  }

  /// Check if loop has results (true if has iter_args)
  static bool hasResults(scf::ForOp op) {
    return !op.getInitArgs().empty();
  }

  /// Get the loop operation name for error messages
  static StringRef getLoopName() { return "scf.for"; }
};

} // namespace dsa
} // namespace mlir

#endif // DSA_TRANSFORMS_NORMALIZESCFLOOPS_LOOPTRAITS_H

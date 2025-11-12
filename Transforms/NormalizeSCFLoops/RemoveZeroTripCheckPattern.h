//===- RemoveZeroTripCheckPattern.h - Remove Zero-Trip Check ---*- C++ -*-===//
//
// Generic pattern to remove redundant zero-trip check scf.if wrapping loops.
//
//===----------------------------------------------------------------------===//

#ifndef DSA_TRANSFORMS_NORMALIZESCFLOOPS_REMOVEZEROTRIP_CHECKPATTERN_H
#define DSA_TRANSFORMS_NORMALIZESCFLOOPS_REMOVEZEROTRIP_CHECKPATTERN_H

#include "LoopTraits.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace dsa {

/// Generic pattern to remove redundant zero-trip check scf.if that wraps loops.
/// Works for both scf.while and scf.for using LoopTraits for type-specific ops.
///
/// This is safe because loops already handle zero-trip cases:
/// - scf.while: condition is checked immediately in before region
/// - scf.for: when lb >= ub, body doesn't execute and iter_args init is returned
///
/// Before:
///   %result = scf.if %is_zero_trip -> (T) {
///     scf.yield %init : T
///   } else {
///     %r = scf.{while|for} ... iter_args(%init) {
///       ...
///     }
///     scf.yield %r : T
///   }
///
/// After:
///   %result = scf.{while|for} ... iter_args(%init) {
///     ...
///   }
template <typename LoopOpType>
struct RemoveZeroTripCheckAroundLoopPattern : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;
  using Traits = LoopTraits<LoopOpType>;

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    // Must have results (pattern is for value-returning ifs)
    if (ifOp.getNumResults() == 0)
      return failure();

    // Must have both then and else regions
    Block *thenBlock = ifOp.thenBlock();
    Block *elseBlock = ifOp.elseBlock();
    if (!thenBlock || !elseBlock)
      return failure();

    // Then block should be simple (just yield)
    if (thenBlock->getOperations().size() != 1)
      return failure();

    auto thenYield = dyn_cast<scf::YieldOp>(thenBlock->getTerminator());
    if (!thenYield)
      return failure();

    // Else block should contain a loop of the target type
    LoopOpType loopOp = nullptr;
    for (Operation &op : elseBlock->without_terminator()) {
      if (auto loopCandidate = dyn_cast<LoopOpType>(&op)) {
        if (loopOp)
          return failure();  // Multiple loops of this type
        loopOp = loopCandidate;
      }
    }

    if (!loopOp)
      return failure();

    // Check else block terminator
    auto elseYield = dyn_cast<scf::YieldOp>(elseBlock->getTerminator());
    if (!elseYield)
      return failure();

    // Verify the loop has results to return
    if (!Traits::hasResults(loopOp))
      return failure();

    // Check that then yields the same number of values as else
    if (thenYield.getNumOperands() != elseYield.getNumOperands())
      return failure();

    // Verify that else yields come from loop results
    // Build a mapping of loop results to indices
    llvm::DenseMap<Value, unsigned> loopResultToIdx;
    for (auto [idx, result] : llvm::enumerate(Traits::getResults(loopOp))) {
      loopResultToIdx[result] = idx;
    }

    // Check that else yields loop results in order
    for (auto [idx, yieldVal] : llvm::enumerate(elseYield.getOperands())) {
      auto it = loopResultToIdx.find(yieldVal);
      if (it == loopResultToIdx.end())
        return failure();  // Yields something other than loop results

      // Must be in the same order (for consistency)
      if (it->second != idx)
        return failure();
    }

    // Verify that then yields match loop's init values
    SmallVector<Value> initValues = Traits::getInitValues(loopOp);
    if (thenYield.getNumOperands() != initValues.size())
      return failure();

    for (auto [thenVal, initVal] : llvm::zip(thenYield.getOperands(), initValues)) {
      if (thenVal != initVal)
        return failure();
    }

    // All checks passed - the if is a redundant zero-trip check
    // Clone operations before the loop (type conversions, constants, etc.)
    rewriter.setInsertionPoint(ifOp);
    IRMapping mapper;
    for (Operation &op : elseBlock->without_terminator()) {
      if (&op == loopOp.getOperation())
        break;  // Stop before loop
      Operation *cloned = rewriter.clone(op, mapper);
      for (auto [orig, clonedRes] : llvm::zip(op.getResults(), cloned->getResults())) {
        mapper.map(orig, clonedRes);
      }
    }

    // Clone the loop itself
    auto newLoop = cast<LoopOpType>(rewriter.clone(*loopOp.getOperation(), mapper));

    // Map the loop results for the final replacement
    SmallVector<Value> replacementValues;
    for (Value yieldVal : elseYield.getOperands()) {
      // Find which loop result this is
      for (auto [idx, loopResult] : llvm::enumerate(Traits::getResults(loopOp))) {
        if (yieldVal == loopResult) {
          replacementValues.push_back(Traits::getResults(newLoop)[idx]);
          break;
        }
      }
    }

    // Replace the if with the cloned loop results
    rewriter.replaceOp(ifOp, replacementValues);

    return success();
  }
};

} // namespace dsa
} // namespace mlir

#endif // DSA_TRANSFORMS_NORMALIZESCFLOOPS_REMOVEZEROTRIP_CHECKPATTERN_H

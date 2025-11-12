//===- DetectParallelForPattern.h - Detect Parallel Loops ------*- C++ -*-===//
//
// Pattern to detect and annotate parallelizable scf.for loops.
//
//===----------------------------------------------------------------------===//

#ifndef DSA_TRANSFORMS_NORMALIZESCFLOOPS_DETECTPARALLELFORPATTERN_H
#define DSA_TRANSFORMS_NORMALIZESCFLOOPS_DETECTPARALLELFORPATTERN_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace dsa {

/// Analyzes scf.for loops to detect parallelizable patterns and annotates them.
/// This pattern does NOT modify the IR structure - it only adds attributes
/// that downstream passes (SCFToHandshakeDSA) can use to generate parallel
/// dataflow circuits.
///
/// Annotations added:
///   - "dsa.parallel_hint" (UnitAttr): Loop can be parallelized
///   - "dsa.parallel_degree" (IntegerAttr): Estimated parallelism degree
///
/// Detection strategy:
///   1. Check if parent function has "dsa.parallel_candidate" attribute
///   2. Perform dependency analysis via ParallelismAnalysis
///   3. Mark loops that pass both checks
struct DetectParallelForPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace dsa
} // namespace mlir

#endif // DSA_TRANSFORMS_NORMALIZESCFLOOPS_DETECTPARALLELFORPATTERN_H

//===- ForallToForPattern.h - Convert Forall to For ------------*- C++ -*-===//
//
// Pattern to convert scf.forall to scf.for for non-parallelizable cases.
//
//===----------------------------------------------------------------------===//

#ifndef DSA_TRANSFORMS_NORMALIZESCFLOOPS_FORALLTOFORPATTERN_H
#define DSA_TRANSFORMS_NORMALIZESCFLOOPS_FORALLTOFORPATTERN_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace dsa {

/// Converts scf.forall to scf.for when the forall cannot be parallelized.
/// This pattern should only match forall loops that:
/// 1. Are 1D (multi-dimensional requires more complex handling)
/// 2. Have reductions (scf.forall.in_parallel terminator with yielding ops)
///
/// For forall without reductions, the SCFToHandshakeDSA conversion will
/// handle them directly with parallel dataflow generation.
struct ForallToForPattern : public OpRewritePattern<scf::ForallOp> {
  using OpRewritePattern<scf::ForallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForallOp forallOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace dsa
} // namespace mlir

#endif // DSA_TRANSFORMS_NORMALIZESCFLOOPS_FORALLTOFORPATTERN_H

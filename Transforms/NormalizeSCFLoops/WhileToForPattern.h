//===- WhileToForPattern.h - Convert While to For --------------*- C++ -*-===//
//
// Pattern to convert counting-loop scf.while to scf.for.
//
//===----------------------------------------------------------------------===//

#ifndef DSA_TRANSFORMS_NORMALIZESCFLOOPS_WHILETOFORPATTERN_H
#define DSA_TRANSFORMS_NORMALIZESCFLOOPS_WHILETOFORPATTERN_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace dsa {

/// Pattern to convert counting-loop scf.while to scf.for.
///
/// Transforms while loops with induction variable patterns into
/// semantically equivalent for loops.
struct WhileToForPattern : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp whileOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace dsa
} // namespace mlir

#endif // DSA_TRANSFORMS_NORMALIZESCFLOOPS_WHILETOFORPATTERN_H

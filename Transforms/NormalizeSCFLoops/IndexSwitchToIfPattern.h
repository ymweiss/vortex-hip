//===- IndexSwitchToIfPattern.h - Convert IndexSwitch to If -----*- C++ -*-===//
//
// Pattern to convert scf.index_switch to nested scf.if operations.
//
//===----------------------------------------------------------------------===//

#ifndef DSA_TRANSFORMS_NORMALIZESCFLOOPS_INDEXSWITCHTOIFPATTERN_H
#define DSA_TRANSFORMS_NORMALIZESCFLOOPS_INDEXSWITCHTOIFPATTERN_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace dsa {

/// Convert scf.index_switch to nested scf.if operations.
///
/// For 2 cases (1 case + default), creates simple if-else:
///   scf.index_switch %arg -> T {
///     case 0 { yield %a }
///     default { yield %b }
///   }
/// becomes:
///   %cond = arith.cmpi eq, %arg, 0
///   scf.if %cond -> T { yield %a } else { yield %b }
///
/// For N cases (N-1 cases + default), creates nested if-else chain:
///   scf.index_switch %arg -> T {
///     case 0 { yield %a }
///     case 1 { yield %b }
///     default { yield %c }
///   }
/// becomes:
///   %cond0 = arith.cmpi eq, %arg, 0
///   scf.if %cond0 -> T {
///     yield %a
///   } else {
///     %cond1 = arith.cmpi eq, %arg, 1
///     scf.if %cond1 -> T { yield %b } else { yield %c }
///   }
struct IndexSwitchToIfPattern : public OpRewritePattern<scf::IndexSwitchOp> {
  using OpRewritePattern<scf::IndexSwitchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IndexSwitchOp switchOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace dsa
} // namespace mlir

#endif // DSA_TRANSFORMS_NORMALIZESCFLOOPS_INDEXSWITCHTOIFPATTERN_H

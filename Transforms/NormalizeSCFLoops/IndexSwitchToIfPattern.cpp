//===- IndexSwitchToIfPattern.cpp - Convert IndexSwitch to If ---*- C++ -*-===//
//
// Implementation of pattern to convert scf.index_switch to nested scf.if.
//
//===----------------------------------------------------------------------===//

#include "IndexSwitchToIfPattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

namespace mlir {
namespace dsa {

LogicalResult IndexSwitchToIfPattern::matchAndRewrite(
    scf::IndexSwitchOp switchOp, PatternRewriter &rewriter) const {

  Location loc = switchOp.getLoc();
  Value switchArg = switchOp.getArg();
  TypeRange resultTypes = switchOp.getResultTypes();

  // Get case values and regions
  ArrayRef<int64_t> caseValues = switchOp.getCases();
  MutableArrayRef<Region> caseRegions = switchOp.getCaseRegions();
  Region &defaultRegion = switchOp.getDefaultRegion();

  // Basic validation - ensure regions exist and are not empty
  for (Region &caseRegion : caseRegions) {
    if (caseRegion.empty() || caseRegion.front().empty())
      return failure();
  }
  if (defaultRegion.empty() || defaultRegion.front().empty())
    return failure();

  // Helper to clone a region's body into an scf.if branch
  auto cloneRegionToIfBranch = [&](Region &region, Block *targetBlock) {
    Block &sourceBlock = region.front();

    IRMapping mapping;
    // Get terminator (if it exists and is a yield)
    Operation *term = sourceBlock.getTerminator();
    scf::YieldOp yieldOp = term ? dyn_cast<scf::YieldOp>(term) : nullptr;

    // Clone all operations, excluding terminator only if it's a proper yield
    for (Operation &op : sourceBlock) {
      if (&op == term && yieldOp) {
        // Skip the yield terminator, we'll create a new one
        break;
      }
      rewriter.clone(op, mapping);
    }

    // Get yield values from source region
    SmallVector<Value> yieldValues;
    if (yieldOp) {
      for (Value operand : yieldOp.getOperands()) {
        yieldValues.push_back(mapping.lookupOrDefault(operand));
      }
    }

    return yieldValues;
  };

  // Build nested if-else chain recursively
  std::function<scf::IfOp(unsigned)> buildNestedIf;
  buildNestedIf = [&](unsigned caseIdx) -> scf::IfOp {
    if (caseIdx >= caseValues.size()) {
      // Base case: only default region remains
      // This shouldn't happen in the recursion, but handle it
      return nullptr;
    }

    // Create condition: switchArg == caseValue
    int64_t caseValue = caseValues[caseIdx];
    Value caseConst;
    if (switchArg.getType().isIndex()) {
      caseConst = rewriter.create<arith::ConstantIndexOp>(loc, caseValue);
    } else {
      // Assume integer type
      caseConst = rewriter.create<arith::ConstantIntOp>(
          loc, switchArg.getType(), caseValue);
    }
    Value condition = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, switchArg, caseConst);

    // Create if operation
    auto ifOp = rewriter.create<scf::IfOp>(loc, resultTypes, condition,
                                            /*withElseRegion=*/true);

    // Then branch: clone case region
    Block *thenBlock = &ifOp.getThenRegion().front();
    // Remove any auto-generated yield terminator
    if (!thenBlock->empty() && isa<scf::YieldOp>(thenBlock->back())) {
      rewriter.eraseOp(&thenBlock->back());
    }
    rewriter.setInsertionPointToStart(thenBlock);
    SmallVector<Value> thenYields = cloneRegionToIfBranch(caseRegions[caseIdx], thenBlock);
    // Ensure insertion point is at end of then block before creating yield
    rewriter.setInsertionPointToEnd(thenBlock);
    rewriter.create<scf::YieldOp>(loc, thenYields);

    // Else branch: recursively handle remaining cases or default
    Block *elseBlock = &ifOp.getElseRegion().front();
    // Remove any auto-generated yield terminator
    if (!elseBlock->empty() && isa<scf::YieldOp>(elseBlock->back())) {
      rewriter.eraseOp(&elseBlock->back());
    }
    rewriter.setInsertionPointToStart(elseBlock);

    if (caseIdx + 1 < caseValues.size()) {
      // More cases remain: recursively build nested if
      scf::IfOp nestedIf = buildNestedIf(caseIdx + 1);
      // Reset insertion point to end of else block before creating yield
      rewriter.setInsertionPointToEnd(elseBlock);
      rewriter.create<scf::YieldOp>(loc, nestedIf.getResults());
    } else {
      // Last case: else branch is default region
      SmallVector<Value> defaultYields = cloneRegionToIfBranch(defaultRegion, elseBlock);
      // Insertion point should already be at end, but ensure it
      rewriter.setInsertionPointToEnd(elseBlock);
      rewriter.create<scf::YieldOp>(loc, defaultYields);
    }

    return ifOp;
  };

  // Special case: single case + default (most common in break patterns)
  if (caseValues.size() == 1) {
    // Create condition: switchArg == caseValue
    int64_t caseValue = caseValues[0];
    Value caseConst;
    if (switchArg.getType().isIndex()) {
      caseConst = rewriter.create<arith::ConstantIndexOp>(loc, caseValue);
    } else {
      caseConst = rewriter.create<arith::ConstantIntOp>(
          loc, switchArg.getType(), caseValue);
    }
    Value condition = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, switchArg, caseConst);

    // Create if-else
    auto ifOp = rewriter.create<scf::IfOp>(loc, resultTypes, condition,
                                            /*withElseRegion=*/true);

    // Then branch: clone case 0 region
    Block *thenBlock = &ifOp.getThenRegion().front();
    // Remove any auto-generated yield terminator
    if (!thenBlock->empty() && isa<scf::YieldOp>(thenBlock->back())) {
      rewriter.eraseOp(&thenBlock->back());
    }
    rewriter.setInsertionPointToStart(thenBlock);
    SmallVector<Value> thenYields = cloneRegionToIfBranch(caseRegions[0], thenBlock);
    rewriter.setInsertionPointToEnd(thenBlock);
    rewriter.create<scf::YieldOp>(loc, thenYields);

    // Else branch: clone default region
    Block *elseBlock = &ifOp.getElseRegion().front();
    // Remove any auto-generated yield terminator
    if (!elseBlock->empty() && isa<scf::YieldOp>(elseBlock->back())) {
      rewriter.eraseOp(&elseBlock->back());
    }
    rewriter.setInsertionPointToStart(elseBlock);
    SmallVector<Value> elseYields = cloneRegionToIfBranch(defaultRegion, elseBlock);
    rewriter.setInsertionPointToEnd(elseBlock);
    rewriter.create<scf::YieldOp>(loc, elseYields);

    // Replace switch with if
    rewriter.replaceOp(switchOp, ifOp.getResults());
    return success();
  }

  // Multiple cases: build nested if-else chain
  scf::IfOp rootIf = buildNestedIf(0);
  rewriter.replaceOp(switchOp, rootIf.getResults());

  return success();
}

} // namespace dsa
} // namespace mlir

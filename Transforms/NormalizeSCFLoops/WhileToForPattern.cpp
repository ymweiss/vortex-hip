//===- WhileToForPattern.cpp - Convert While to For ------------*- C++ -*-===//
//
// Implementation of pattern to convert counting-loop scf.while to scf.for.
//
//===----------------------------------------------------------------------===//

#include "WhileToForPattern.h"
#include "IVPatternHelpers.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {
namespace dsa {

LogicalResult WhileToForPattern::matchAndRewrite(
    scf::WhileOp whileOp, PatternRewriter &rewriter) const {
  // Must have at least one result (the induction variable)
  if (whileOp.getNumResults() < 1)
    return failure();

  // Allow transformation of while loops nested in scf.if
  // (Previously disabled, but safe after Phase 2 IV detection fixes)
  // Still disallow nested while loops (too complex)
  Operation *parentOp = whileOp->getParentOp();
  while (parentOp && !isa<func::FuncOp>(parentOp)) {
    // Allow scf.if nesting, but not for/while nesting
    if (isa<scf::ForOp, scf::WhileOp>(parentOp))
      return failure();
    parentOp = parentOp->getParentOp();
  }

  // Check that do region passes through all values unchanged
  // This is required for converting to scf.for with iter_args
  Block *doBlock = whileOp.getAfterBody();
  if (!doBlock)
    return failure();

  // Should only have yield operation
  if (doBlock->getOperations().size() != 1)
    return failure();

  auto doYield = dyn_cast<scf::YieldOp>(&doBlock->front());
  if (!doYield)
    return failure();

  // All yielded values should be block arguments unchanged
  if (doYield.getNumOperands() != doBlock->getNumArguments())
    return failure();

  for (auto [yieldVal, blockArg] : llvm::zip(doYield.getOperands(), doBlock->getArguments())) {
    if (yieldVal != blockArg)
      return failure();
  }

  // Extract IV pattern at ANY position (not just position 0!)
  IVPattern pattern = extractIVPatternAtAnyPosition(whileOp);
  if (!pattern.valid)
    return failure();

  // Get the IV index from the pattern
  unsigned ivIdx = pattern.ivIndex;

  // Only transform simple loops without nested control flow
  // Nested loops/conditionals make the transformation complex and error-prone
  // Need to walk all nested regions, not just top-level operations
  Block *beforeBlock = whileOp.getBeforeBody();
  bool hasNestedControlFlow = false;
  beforeBlock->walk([&](Operation *op) {
    if (isa<scf::WhileOp, scf::ForOp, scf::IfOp>(op)) {
      hasNestedControlFlow = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (hasNestedControlFlow)
    return failure();

  // Ensure bound and step are loop-invariant (defined outside the while loop)
  // If they're defined inside, we can't safely use them after erasing the loop
  auto isDefinedInWhile = [&](Value val) -> bool {
    Operation *defOp = val.getDefiningOp();
    if (!defOp)
      return false; // Block argument, safe to use

    // Walk up the region hierarchy to check if the operation is anywhere
    // within the while loop's regions (including nested regions)
    Region *defRegion = defOp->getParentRegion();
    while (defRegion) {
      if (defRegion == &whileOp.getBefore() || defRegion == &whileOp.getAfter())
        return true;

      // Move to parent region
      Operation *parentOp = defRegion->getParentOp();
      if (!parentOp || parentOp == whileOp.getOperation())
        break;
      defRegion = parentOp->getParentRegion();
    }
    return false;
  };

  if (isDefinedInWhile(pattern.init) || isDefinedInWhile(pattern.bound) || isDefinedInWhile(pattern.step))
    return failure();

  // Only handle specific predicates for now
  // ne: while (%i != %N) - most common from LLVM
  // ult/slt: while (%i < %N)
  // ule/sle: while (%i <= %N)
  if (pattern.predicate != arith::CmpIPredicate::ne &&
      pattern.predicate != arith::CmpIPredicate::ult &&
      pattern.predicate != arith::CmpIPredicate::slt &&
      pattern.predicate != arith::CmpIPredicate::ule &&
      pattern.predicate != arith::CmpIPredicate::sle)
    return failure();

  Location loc = whileOp.getLoc();
  Type ivType = whileOp.getResult(ivIdx).getType();

  // Set insertion point BEFORE the while loop to create index casts
  rewriter.setInsertionPoint(whileOp);

  // Convert values to index type for scf.for
  Value initIdx, stepIdx, boundIdx;

  if (ivType.isIndex()) {
    initIdx = pattern.init;
    stepIdx = pattern.step;
    boundIdx = pattern.bound;
  } else {
    // Cast to index (these will be created before the while loop)
    initIdx = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), pattern.init);
    stepIdx = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), pattern.step);
    boundIdx = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), pattern.bound);
  }

  // Adjust bounds based on predicate
  // ne/ult/slt: for %i = init to bound (exclusive upper bound)
  // ule/sle: for %i = init to (bound+1) (inclusive -> exclusive)
  if (pattern.predicate == arith::CmpIPredicate::ule ||
      pattern.predicate == arith::CmpIPredicate::sle) {
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    boundIdx = arith::AddIOp::create(rewriter, loc, boundIdx, one);
  }

  // Collect iter_args (all values except the induction variable at ivIdx)
  SmallVector<Value> iterArgs;
  SmallVector<Type> iterArgTypes;
  for (unsigned i = 0; i < whileOp.getNumResults(); ++i) {
    if (i != ivIdx) {  // Skip IV position
      iterArgs.push_back(whileOp.getInits()[i]);
      iterArgTypes.push_back(whileOp.getResultTypes()[i]);
    }
  }

  // Create scf.for with iter_args for non-IV values
  auto forOp = scf::ForOp::create(rewriter, loc, initIdx, boundIdx, stepIdx, iterArgs);

  // Mark as converted from while loop for downstream analysis
  forOp->setAttr("dsa.converted_from_while", rewriter.getUnitAttr());

  Block *forBody = forOp.getBody();

  // Set insertion point to start of for body
  rewriter.setInsertionPointToStart(forBody);

  // If original type was not index, cast the IV
  Value forIV = forOp.getInductionVar();
  if (!ivType.isIndex()) {
    forIV = arith::IndexCastOp::create(rewriter, loc, ivType, forOp.getInductionVar());
  }

  // Move operations from while's before region to for body
  // (beforeBlock already retrieved earlier for validation)
  BlockArgument whileIV = beforeBlock->getArgument(ivIdx);

  // Get the condition operation
  auto conditionOp = cast<scf::ConditionOp>(beforeBlock->getTerminator());
  Value nextIV = conditionOp.getArgs()[ivIdx];
  Value condition = conditionOp.getCondition();

  // Identify operations to skip (loop control operations)
  llvm::SmallPtrSet<Operation *, 4> opsToSkip;

  // Mark the operation that computes the condition
  Operation *condDefOp = condition.getDefiningOp();
  if (condDefOp) {
    opsToSkip.insert(condDefOp);
  }

  // Mark the operation that computes nextIV, BUT ONLY if it's not used by other operations
  // Example: integrate_trapz uses nextIV (%5 = i+1) to access arrays at index i+1
  // In that case, we must clone the nextIV operation into the for body
  if (Operation *nextIVOp = nextIV.getDefiningOp()) {
    bool onlyUsedByControlFlow = true;
    for (Operation *user : nextIV.getUsers()) {
      // Allow uses by: condition definition op, scf.condition terminator
      if (user != condDefOp && user != conditionOp.getOperation()) {
        onlyUsedByControlFlow = false;
        break;
      }
    }

    // Only skip if exclusively used for control flow
    if (onlyUsedByControlFlow) {
      opsToSkip.insert(nextIVOp);
    }
  }

  // Clone operations from before region (except terminator and control ops)
  IRMapping mapping;
  mapping.map(whileIV, forIV);

  // Map the while loop's iter_args to for loop's iter_args
  // Note: forBody->getArgument(0) is the induction variable
  //       forBody->getArgument(1..) are the iter_args
  // Need to map correctly, skipping the IV position
  unsigned forIterArgIdx = 1;  // Start at 1, 0 is the induction variable
  for (unsigned i = 0; i < whileOp.getNumResults(); ++i) {
    if (i != ivIdx) {
      mapping.map(beforeBlock->getArgument(i), forBody->getArgument(forIterArgIdx));
      forIterArgIdx++;
    }
  }

  for (Operation &op : beforeBlock->without_terminator()) {
    // Skip loop control operations
    if (opsToSkip.contains(&op))
      continue;

    // Clone the operation and update mapping with cloned results
    Operation *clonedOp = rewriter.clone(op, mapping);

    // Map original results to cloned results for subsequent operations
    for (auto [originalResult, clonedResult] : llvm::zip(op.getResults(), clonedOp->getResults())) {
      mapping.map(originalResult, clonedResult);
    }
  }

  // Create scf.yield with mapped values (excluding the IV which is implicit)
  // The scf.for builder may have created an empty yield terminator
  // If so, we need to replace it with our yield that has the correct operands
  SmallVector<Value> yieldValues;
  for (unsigned i = 0; i < conditionOp.getArgs().size(); ++i) {
    if (i != ivIdx) {  // Skip IV position
      Value mappedValue = mapping.lookupOrDefault(conditionOp.getArgs()[i]);
      yieldValues.push_back(mappedValue);
    }
  }

  // Check if there's already a yield terminator
  // The scf.for builder creates an empty yield when there are iter_args
  if (forBody->mightHaveTerminator()) {
    Operation *terminator = forBody->getTerminator();
    if (terminator && isa<scf::YieldOp>(terminator)) {
      // Replace the existing yield with our new one
      rewriter.setInsertionPoint(terminator);
      scf::YieldOp::create(rewriter, loc, yieldValues);
      rewriter.eraseOp(terminator);
    } else if (terminator) {
      // Unexpected terminator type
      return failure();
    } else {
      // No terminator, add yield at end
      rewriter.setInsertionPointToEnd(forBody);
      scf::YieldOp::create(rewriter, loc, yieldValues);
    }
  } else {
    // No terminator possible, add yield at end
    rewriter.setInsertionPointToEnd(forBody);
    scf::YieldOp::create(rewriter, loc, yieldValues);
  }

  // Replace while op results
  // Need to insert results in correct positions, with IV at ivIdx
  SmallVector<Value> replacementValues(whileOp.getNumResults());

  // Compute final IV value (use upper bound)
  Value finalIV = boundIdx;
  if (!ivType.isIndex()) {
    rewriter.setInsertionPointAfter(forOp);
    finalIV = arith::IndexCastOp::create(rewriter, loc, ivType, boundIdx);
  }

  // Place IV at correct position
  replacementValues[ivIdx] = finalIV;

  // Place iter_arg results at other positions
  unsigned forResultIdx = 0;
  for (unsigned i = 0; i < whileOp.getNumResults(); ++i) {
    if (i != ivIdx) {
      replacementValues[i] = forOp.getResults()[forResultIdx];
      forResultIdx++;
    }
  }

  rewriter.replaceOp(whileOp, replacementValues);

  return success();
}

} // namespace dsa
} // namespace mlir

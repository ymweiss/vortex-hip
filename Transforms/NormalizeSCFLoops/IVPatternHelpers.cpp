//===- IVPatternHelpers.cpp - IV Pattern Extraction Helpers ----*- C++ -*-===//
//
// Implementation of helper functions for extracting induction variable patterns
// from SCF while loops.
//
//===----------------------------------------------------------------------===//

#include "IVPatternHelpers.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
namespace dsa {

/// Extract induction variable pattern from a while loop's before region
IVPattern extractIVPattern(scf::WhileOp whileOp) {
  IVPattern pattern;

  // Get the before region
  Block *beforeBlock = whileOp.getBeforeBody();
  if (!beforeBlock)
    return pattern;

  auto condOp = dyn_cast<scf::ConditionOp>(beforeBlock->getTerminator());
  if (!condOp)
    return pattern;

  // Check that we're passing through the updated IV
  if (condOp.getArgs().size() != 1)
    return pattern;

  Value nextIV = condOp.getArgs()[0];
  Value condition = condOp.getCondition();

  // Look for comparison operation
  auto cmpOp = condition.getDefiningOp<arith::CmpIOp>();
  if (!cmpOp)
    return pattern;

  // Look for addition operation that produces nextIV
  auto addOp = nextIV.getDefiningOp<arith::AddIOp>();
  if (!addOp)
    return pattern;

  // Get the block argument (IV)
  BlockArgument ivArg = whileOp.getBeforeArguments()[0];

  // Check that addition uses IV
  Value step;
  if (addOp.getLhs() == ivArg) {
    step = addOp.getRhs();
  } else if (addOp.getRhs() == ivArg) {
    step = addOp.getLhs();
  } else {
    return pattern;
  }

  // Determine bound based on comparison
  Value bound;
  arith::CmpIPredicate predicate = cmpOp.getPredicate();

  if (cmpOp.getLhs() == nextIV) {
    bound = cmpOp.getRhs();
  } else if (cmpOp.getRhs() == nextIV) {
    bound = cmpOp.getLhs();
    // Flip predicate if operands are swapped
    predicate = arith::invertPredicate(predicate);
  } else {
    return pattern;
  }

  // Store pattern
  pattern.init = whileOp.getInits()[0];
  pattern.step = step;
  pattern.bound = bound;
  pattern.predicate = predicate;
  pattern.valid = true;

  return pattern;
}

/// Check if the do region simply passes through the iteration variable
bool isDoRegionPassthrough(scf::WhileOp whileOp) {
  Block *doBlock = whileOp.getAfterBody();
  if (!doBlock)
    return false;

  // Should only have yield operation
  if (doBlock->getOperations().size() != 1)
    return false;

  auto yieldOp = dyn_cast<scf::YieldOp>(&doBlock->front());
  if (!yieldOp)
    return false;

  // Should yield the block argument unchanged
  return yieldOp.getOperands().size() == 1 &&
         yieldOp.getOperands()[0] == doBlock->getArgument(0);
}

/// Extract induction variable pattern from a canonical while loop
IVPattern extractCanonicalIVPattern(scf::WhileOp whileOp) {
  IVPattern pattern;

  // Get the before and after regions
  Block *beforeBlock = whileOp.getBeforeBody();
  Block *afterBlock = whileOp.getAfterBody();
  if (!beforeBlock || !afterBlock)
    return pattern;

  // Get condition operation from before region
  auto condOp = dyn_cast<scf::ConditionOp>(beforeBlock->getTerminator());
  if (!condOp || condOp.getArgs().empty())
    return pattern;

  // The first argument should be the IV (passed through)
  Value currentIV = condOp.getArgs()[0];

  // Check that currentIV is the first block argument
  BlockArgument ivArg = whileOp.getBeforeArguments()[0];
  if (currentIV != ivArg)
    return pattern;

  // Look for comparison operation in before region
  Value condition = condOp.getCondition();
  auto cmpOp = condition.getDefiningOp<arith::CmpIOp>();
  if (!cmpOp)
    return pattern;

  // Extract bound and predicate from comparison
  // The comparison should use the current IV (not incremented yet)
  Value bound;
  arith::CmpIPredicate predicate = cmpOp.getPredicate();

  if (cmpOp.getLhs() == currentIV) {
    bound = cmpOp.getRhs();
  } else if (cmpOp.getRhs() == currentIV) {
    bound = cmpOp.getLhs();
    // Flip predicate if operands are swapped
    predicate = arith::invertPredicate(predicate);
  } else {
    return pattern;
  }

  // Now find the increment in the after region
  // Look at the yield operation
  auto yieldOp = dyn_cast<scf::YieldOp>(afterBlock->getTerminator());
  if (!yieldOp || yieldOp.getNumOperands() == 0)
    return pattern;

  // The first yielded value should be the updated IV
  Value nextIV = yieldOp.getOperands()[0];

  // Look for the addition operation that produces nextIV
  auto addOp = nextIV.getDefiningOp<arith::AddIOp>();
  if (!addOp)
    return pattern;

  // Check that the addition uses the after block's IV argument
  BlockArgument afterIVArg = afterBlock->getArgument(0);
  Value step;

  if (addOp.getLhs() == afterIVArg) {
    step = addOp.getRhs();
  } else if (addOp.getRhs() == afterIVArg) {
    step = addOp.getLhs();
  } else {
    return pattern;
  }

  // Store the extracted pattern
  pattern.init = whileOp.getInits()[0];
  pattern.step = step;
  pattern.bound = bound;
  pattern.predicate = predicate;
  pattern.valid = true;

  return pattern;
}

/// Try to extract IV pattern at a specific argument position
IVPattern tryExtractIVAtPosition(scf::WhileOp whileOp, unsigned ivIdx) {
  IVPattern pattern;
  pattern.ivIndex = ivIdx;

  Block *beforeBlock = whileOp.getBeforeBody();
  Block *afterBlock = whileOp.getAfterBody();

  if (!beforeBlock || !afterBlock)
    return pattern;

  auto condOp = dyn_cast<scf::ConditionOp>(beforeBlock->getTerminator());
  if (!condOp || ivIdx >= condOp.getArgs().size())
    return pattern;

  Value trailingArg = condOp.getArgs()[ivIdx];
  BlockArgument ivArg = beforeBlock->getArgument(ivIdx);

  // Pattern 1: Incremented IV in before region
  // scf.condition(%cond) ..., %next_iv, ...
  // where %next_iv = arith.addi %iv, %step
  if (auto addOp = trailingArg.getDefiningOp<arith::AddIOp>()) {
    Value step;
    if (addOp.getLhs() == ivArg) {
      step = addOp.getRhs();
    } else if (addOp.getRhs() == ivArg) {
      step = addOp.getLhs();
    } else {
      // Not incrementing this argument
      return pattern;
    }

    // Find comparison using next_iv
    Value condition = condOp.getCondition();
    auto cmpOp = condition.getDefiningOp<arith::CmpIOp>();
    if (!cmpOp)
      return pattern;

    Value bound;
    arith::CmpIPredicate pred = cmpOp.getPredicate();

    if (cmpOp.getLhs() == trailingArg) {
      bound = cmpOp.getRhs();
    } else if (cmpOp.getRhs() == trailingArg) {
      bound = cmpOp.getLhs();
      pred = arith::invertPredicate(pred);
    } else {
      return pattern;
    }

    pattern.init = whileOp.getInits()[ivIdx];
    pattern.bound = bound;
    pattern.step = step;
    pattern.predicate = pred;
    pattern.valid = true;
    return pattern;
  }

  // Pattern 2: Non-incremented IV (canonical form)
  // scf.condition(%cond) ..., %iv, ...
  // Increment happens in after region
  if (trailingArg == ivArg) {
    // Look for comparison using current IV
    Value condition = condOp.getCondition();
    auto cmpOp = condition.getDefiningOp<arith::CmpIOp>();
    if (!cmpOp)
      return pattern;

    Value bound;
    arith::CmpIPredicate pred = cmpOp.getPredicate();

    if (cmpOp.getLhs() == ivArg) {
      bound = cmpOp.getRhs();
    } else if (cmpOp.getRhs() == ivArg) {
      bound = cmpOp.getLhs();
      pred = arith::invertPredicate(pred);
    } else {
      return pattern;
    }

    // Find increment in after region
    auto yieldOp = dyn_cast<scf::YieldOp>(afterBlock->getTerminator());
    if (!yieldOp || ivIdx >= yieldOp.getNumOperands())
      return pattern;

    Value nextIV = yieldOp.getOperands()[ivIdx];
    auto addOp = nextIV.getDefiningOp<arith::AddIOp>();
    if (!addOp)
      return pattern;

    BlockArgument afterIVArg = afterBlock->getArgument(ivIdx);
    Value step;

    if (addOp.getLhs() == afterIVArg) {
      step = addOp.getRhs();
    } else if (addOp.getRhs() == afterIVArg) {
      step = addOp.getLhs();
    } else {
      return pattern;
    }

    pattern.init = whileOp.getInits()[ivIdx];
    pattern.bound = bound;
    pattern.step = step;
    pattern.predicate = pred;
    pattern.valid = true;
    return pattern;
  }

  return pattern;
}

/// Extract induction variable pattern at ANY position
IVPattern extractIVPatternAtAnyPosition(scf::WhileOp whileOp) {
  // Try each argument position as potential IV
  for (unsigned ivIdx = 0; ivIdx < whileOp.getNumResults(); ++ivIdx) {
    IVPattern pattern = tryExtractIVAtPosition(whileOp, ivIdx);
    if (pattern.valid)
      return pattern;
  }

  // No IV found at any position
  return IVPattern{.valid = false};
}

} // namespace dsa
} // namespace mlir

//===- ForallToForPattern.cpp - Convert Forall to For ----------*- C++ -*-===//
//
// Implementation of pattern to convert scf.forall to scf.for.
//
//===----------------------------------------------------------------------===//

#include "ForallToForPattern.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

namespace mlir {
namespace dsa {

LogicalResult ForallToForPattern::matchAndRewrite(
    scf::ForallOp forallOp, PatternRewriter &rewriter) const {
  // Only handle 1D forall for now
  if (forallOp.getRank() != 1)
    return failure();

  // Handle both forall with and without reductions
  bool hasReductions = !forallOp.getOutputs().empty();

  // Check if there are actual reduction operations
  if (hasReductions) {
    auto inParallelOp = forallOp.getTerminator();
    if (!inParallelOp || inParallelOp.getYieldingOps().empty())
      hasReductions = false;  // No actual reductions, treat as parallel
  }

  Location loc = forallOp.getLoc();

  // Get loop bounds (only 1D supported)
  // Use getLowerBound/getUpperBound/getStep which materialize constants for static bounds
  SmallVector<Value> lbs = forallOp.getLowerBound(rewriter);
  SmallVector<Value> ubs = forallOp.getUpperBound(rewriter);
  SmallVector<Value> steps = forallOp.getStep(rewriter);

  Value lowerBound = lbs[0];
  Value upperBound = ubs[0];
  Value step = steps[0];

  // Create scf.for (with or without iter_args depending on reductions)
  scf::ForOp forOp;
  if (hasReductions) {
    // Get shared outputs (reduction accumulators)
    ValueRange outputs = forallOp.getOutputs();
    forOp = scf::ForOp::create(rewriter, loc, lowerBound, upperBound, step, outputs);
  } else {
    // No reductions - create simple scf.for
    forOp = scf::ForOp::create(rewriter, loc, lowerBound, upperBound, step);
  }

  Block *forBody = forOp.getBody();

  // Map forall iteration variable to for induction variable
  IRMapping mapper;
  Value forallIV = forallOp.getInductionVars()[0];
  Value forIV = forOp.getInductionVar();
  mapper.map(forallIV, forIV);

  // Map shared outputs to iter_args (if any)
  if (hasReductions) {
    for (auto [output, iterArg] :
         llvm::zip(forallOp.getRegionIterArgs(), forOp.getRegionIterArgs())) {
      mapper.map(output, iterArg);
    }
  }

  // Clone forall body into for body (before existing terminator)
  Block *forallBody = forallOp.getBody();
  rewriter.setInsertionPoint(forBody, forBody->begin());
  for (Operation &op : forallBody->without_terminator()) {
    rewriter.clone(op, mapper);
  }

  // Remove the auto-generated yield and create our own
  Operation *terminator = forBody->getTerminator();
  if (terminator && isa<scf::YieldOp>(terminator)) {
    rewriter.eraseOp(terminator);
  }

  // Set insertion point to end of block before creating yield
  rewriter.setInsertionPointToEnd(forBody);

  // Create yield operation
  if (hasReductions) {
    // Handle in_parallel terminator - convert to scf.yield
    // For now, conservatively pass through iter_args
    // TODO: Properly handle tensor.parallel_insert_slice operations
    SmallVector<Value> yieldValues;
    for (Value iterArg : forOp.getRegionIterArgs()) {
      yieldValues.push_back(iterArg);
    }
    scf::YieldOp::create(rewriter, loc, yieldValues);
  } else {
    // No reductions - just create empty yield
    scf::YieldOp::create(rewriter, loc);
  }

  // Add parallelism annotations if no reductions
  if (!hasReductions) {
    forOp->setAttr("dsa.parallel_hint", rewriter.getUnitAttr());
    // Estimate parallel degree as trip count (simplified)
    forOp->setAttr("dsa.parallel_degree", rewriter.getI64IntegerAttr(16));
    forOp->setAttr("dsa.original_forall", rewriter.getUnitAttr());
  }

  // Replace forall with for results
  rewriter.replaceOp(forallOp, forOp.getResults());

  return success();
}

} // namespace dsa
} // namespace mlir

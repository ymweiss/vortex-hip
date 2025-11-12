//===- ParallelLoops.cpp - Parallel loop conversion -------------*- C++ -*-===//
//
// Conversion patterns for parallel SCF constructs (scf.forall, scf.parallel)
// to Handshake+DSA dataflow IR with fork-based parallelism.
//
//===----------------------------------------------------------------------===//

#include "Common.h"
#include "ParallelismAnalysis.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "parallel-loops"

namespace mlir {
namespace dsa {

//===----------------------------------------------------------------------===//
// scf.forall Conversion Pattern
//===----------------------------------------------------------------------===//

/// Convert scf.forall to parallel Handshake+DSA IR.
///
/// STRATEGY:
/// Since Handshake doesn't have native parallel loop constructs, we use
/// fork-based parallelism:
///
/// 1. For simple 1D forall loops without reductions:
///    - Convert to scf.for (sequential iteration space generation)
///    - Add dsa.parallel_hint annotation for downstream optimization
///
/// 2. For forall with independent iterations and no shared outputs:
///    - Create multiple parallel dataflow instances using fork
///    - Each instance processes a subset of iterations
///
/// 3. For forall with reductions (in_parallel terminator):
///    - Convert to sequential scf.for (reductions create dependencies)
///
/// CURRENT IMPLEMENTATION:
/// We implement strategy #1 (convert to annotated scf.for) as the base case.
/// Future work can implement strategies #2 and #3 for more sophisticated
/// parallelization.
///
struct ConvertForallOp : public OpConversionPattern<scf::ForallOp> {
  using OpConversionPattern<scf::ForallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForallOp forallOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = forallOp.getLoc();

    // Only handle 1D forall for now
    // Multi-dimensional forall requires more complex handling
    if (forallOp.getRank() != 1) {
      return forallOp.emitError(
          "scf.forall conversion only supports 1D loops currently");
    }

    // Check if forall has reductions (shared outputs with in_parallel)
    bool hasReductions = !forallOp.getOutputs().empty();
    if (hasReductions) {
      auto inParallelOp = forallOp.getTerminator();
      if (inParallelOp && !inParallelOp.getYieldingOps().empty()) {
        // Has reductions - must be sequential
        LLVM_DEBUG(llvm::dbgs() << "Converting scf.forall with reductions to "
                                   "sequential scf.for\n");
        return convertForallWithReductionsToFor(forallOp, adaptor, rewriter);
      }
    }

    // No reductions - can be parallelized
    LLVM_DEBUG(llvm::dbgs() << "Converting scf.forall to parallel scf.for\n");
    return convertForallToParallelFor(forallOp, adaptor, rewriter);
  }

private:
  /// Convert scf.forall with reductions to sequential scf.for
  LogicalResult convertForallWithReductionsToFor(
      scf::ForallOp forallOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const {

    // For forall with reductions, we need to convert to scf.for with iter_args
    // This is a sequential execution since reductions create dependencies

    Location loc = forallOp.getLoc();

    // Get loop bounds (only 1D supported)
    // Use getLowerBound/getUpperBound/getStep which materialize constants for static bounds
    SmallVector<Value> lbs = forallOp.getLowerBound(rewriter);
    SmallVector<Value> ubs = forallOp.getUpperBound(rewriter);
    SmallVector<Value> steps = forallOp.getStep(rewriter);

    Value lowerBound = lbs[0];
    Value upperBound = ubs[0];
    Value step = steps[0];

    // Get shared outputs (reduction accumulators)
    ValueRange initOutputs = adaptor.getOutputs();

    // Create scf.for with iter_args for reductions
    auto forOp = scf::ForOp::create(rewriter, 
        loc, lowerBound, upperBound, step, initOutputs);

    Block *forBody = forOp.getBody();
    rewriter.setInsertionPointToStart(forBody);

    // Map forall iteration variable to for induction variable
    IRMapping mapper;
    Value forallIV = forallOp.getInductionVars()[0];
    Value forIV = forOp.getInductionVar();
    mapper.map(forallIV, forIV);

    // Map shared outputs to iter_args
    for (auto [output, iterArg] :
         llvm::zip(forallOp.getRegionIterArgs(), forOp.getRegionIterArgs())) {
      mapper.map(output, iterArg);
    }

    // Clone forall body into for body
    Block *forallBody = forallOp.getBody();
    for (Operation &op : forallBody->without_terminator()) {
      rewriter.clone(op, mapper);
    }

    // Handle in_parallel terminator - convert to scf.yield
    auto inParallelOp = forallOp.getTerminator();
    SmallVector<Value> yieldValues;

    for (Operation &yieldingOp : inParallelOp.getYieldingOps()) {
      // Extract the yielded values from parallel_insert_slice ops
      // For now, we just pass through the iter_args (conservative)
      // TODO: Handle tensor.parallel_insert_slice properly
    }

    // If no specific yield values, pass through iter_args
    if (yieldValues.empty()) {
      for (Value iterArg : forOp.getRegionIterArgs()) {
        yieldValues.push_back(iterArg);
      }
    }

    scf::YieldOp::create(rewriter, loc, yieldValues);

    // Replace forall with for results
    rewriter.replaceOp(forallOp, forOp.getResults());

    return success();
  }

  /// Convert scf.forall without reductions to parallel scf.for
  LogicalResult convertForallToParallelFor(
      scf::ForallOp forallOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const {

    Location loc = forallOp.getLoc();

    // Get loop bounds (only 1D supported)
    // Use getLowerBound/getUpperBound/getStep which materialize constants for static bounds
    SmallVector<Value> lbs = forallOp.getLowerBound(rewriter);
    SmallVector<Value> ubs = forallOp.getUpperBound(rewriter);
    SmallVector<Value> steps = forallOp.getStep(rewriter);

    Value lowerBound = lbs[0];
    Value upperBound = ubs[0];
    Value step = steps[0];

    // Create scf.for without iter_args (map-like parallel loop)
    auto forOp = scf::ForOp::create(rewriter, 
        loc, lowerBound, upperBound, step);

    Block *forBody = forOp.getBody();
    rewriter.setInsertionPointToStart(forBody);

    // Map forall iteration variable to for induction variable
    IRMapping mapper;
    Value forallIV = forallOp.getInductionVars()[0];
    Value forIV = forOp.getInductionVar();
    mapper.map(forallIV, forIV);

    // Clone forall body into for body
    Block *forallBody = forallOp.getBody();
    for (Operation &op : forallBody->without_terminator()) {
      rewriter.clone(op, mapper);
    }

    // Analyze parallelism to determine degree
    ParallelismAnalysis analysis(forallOp.getOperation());
    unsigned parallelDegree = analysis.estimateParallelismDegree();

    // Annotate the scf.for with parallel hints
    // This tells downstream passes (SCFToHandshakeDSA) that this loop
    // can be parallelized using fork-based dataflow
    forOp->setAttr("dsa.parallel_hint", rewriter.getUnitAttr());
    forOp->setAttr("dsa.parallel_degree",
                   rewriter.getI64IntegerAttr(parallelDegree));
    forOp->setAttr("dsa.original_forall", rewriter.getUnitAttr());

    LLVM_DEBUG({
      llvm::dbgs() << "Converted scf.forall to parallel scf.for\n";
      llvm::dbgs() << "  Parallel degree: " << parallelDegree << "\n";
    });

    // Replace forall with for
    rewriter.replaceOp(forallOp, forOp.getResults());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void populateParallelLoopConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertForallOp>(patterns.getContext());
}

} // namespace dsa
} // namespace mlir

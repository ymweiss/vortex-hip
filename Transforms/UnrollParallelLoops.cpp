//===- UnrollParallelLoops.cpp - Unroll parallel loops for spatial exec -*-===//
//
// This pass unrolls loops marked with `dsa.parallel_hint` to create multiple
// parallel execution instances for spatial hardware implementation.
//
// Supports two unrolling modes:
// - Interleave mode: Round-robin distribution (i, i+U, i+2U, ...)
// - Block mode: Contiguous blocks ([k*B, (k+1)*B))
//
// Uses predication for remainder handling when trip count is not divisible
// by the unroll factor.
//
//===----------------------------------------------------------------------===//

#include "dsa/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "unroll-parallel-loops"

namespace mlir {
namespace dsa {

#define GEN_PASS_DEF_UNROLLPARALLELLOOPS
#include "dsa/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Helper: Loop Analysis and Metadata Extraction
//===----------------------------------------------------------------------===//

/// Get the unroll factor for a loop.
/// Priority: pass option > loop attribute > default (4)
static unsigned getUnrollFactor(scf::ForOp forOp, unsigned passOption) {
  // Check pass option first
  if (passOption != 0)
    return passOption;

  // Check loop attribute
  if (auto degreeAttr = forOp->getAttrOfType<IntegerAttr>("dsa.parallel_degree"))
    return degreeAttr.getInt();

  // Default
  return 4;
}

/// Check if a value is a compile-time constant
static std::optional<int64_t> getConstantValue(Value value) {
  if (auto constOp = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
      return intAttr.getInt();
    }
  }
  return std::nullopt;
}

/// Determine if the loop has statically known bounds
static bool hasStaticBounds(scf::ForOp forOp) {
  return getConstantValue(forOp.getLowerBound()).has_value() &&
         getConstantValue(forOp.getUpperBound()).has_value() &&
         getConstantValue(forOp.getStep()).has_value();
}

//===----------------------------------------------------------------------===//
// Helper: Index Remapping for Loop Body
//===----------------------------------------------------------------------===//

/// Represents a mapping from original induction variable to unrolled instance index
struct IndexMapping {
  Value baseIndex;      // Base loop index (from outer loop)
  Value instanceOffset; // Offset for this instance (0, 1, 2, ...)
  Value actualIndex;    // Actual index = base + offset
  Value predicate;      // Valid flag (actualIndex < upperBound)
};

/// Create an index mapping for an unrolled instance
static IndexMapping createIndexMapping(OpBuilder &builder, Location loc,
                                        Value baseIndex, unsigned instanceId,
                                        unsigned unrollFactor, Value upperBound,
                                        bool isInterleaveMode) {
  IndexMapping mapping;

  // Create constant for instance offset
  Value offsetConst = arith::ConstantIndexOp::create(builder, loc, instanceId);
  mapping.instanceOffset = offsetConst;

  // Calculate actual index based on mode
  if (isInterleaveMode) {
    // Interleave: actualIndex = baseIndex + instanceId
    mapping.actualIndex = arith::AddIOp::create(builder, loc, baseIndex, offsetConst);
  } else {
    // Block mode: actualIndex = baseIndex * unrollFactor + instanceId
    Value factorConst = arith::ConstantIndexOp::create(builder, loc, unrollFactor);
    Value scaledBase = arith::MulIOp::create(builder, loc, baseIndex, factorConst);
    mapping.actualIndex = arith::AddIOp::create(builder, loc, scaledBase, offsetConst);
  }

  // Create predicate: actualIndex < upperBound
  mapping.predicate = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ult, mapping.actualIndex, upperBound);

  mapping.baseIndex = baseIndex;
  return mapping;
}

//===----------------------------------------------------------------------===//
// Helper: Store Operation Predication
//===----------------------------------------------------------------------===//

/// Wrap a store operation with predication
/// Returns the new if operation containing the predicated store
/// Preserves insertion point after the created if operation
static scf::IfOp predicateStoreOp(PatternRewriter &rewriter, memref::StoreOp storeOp,
                                   Value predicate) {
  Location loc = storeOp.getLoc();

  // Save insertion point
  OpBuilder::InsertPoint savePoint = rewriter.saveInsertionPoint();

  // Create scf.if for conditional store
  rewriter.setInsertionPoint(storeOp);
  auto ifOp = scf::IfOp::create(rewriter, loc, predicate, /*withElseRegion=*/false);

  // Move store into if.then block
  rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
  memref::StoreOp::create(
      rewriter, loc, storeOp.getValue(), storeOp.getMemRef(), storeOp.getIndices());

  // Erase original store using the rewriter
  rewriter.eraseOp(storeOp);

  // Restore insertion point to after the if operation
  rewriter.setInsertionPointAfter(ifOp);

  return ifOp;
}

//===----------------------------------------------------------------------===//
// Interleave Mode Unrolling
//===----------------------------------------------------------------------===//

/// Unroll loop using interleave mode: instance k processes k, k+U, k+2U, ...
static LogicalResult unrollInterleaveMode(scf::ForOp forOp, unsigned unrollFactor,
                                           PatternRewriter &rewriter) {
  Location loc = forOp.getLoc();
  Value lowerBound = forOp.getLowerBound();
  Value upperBound = forOp.getUpperBound();
  Value step = forOp.getStep();

  // New step = oldStep * unrollFactor
  Value factorConst = arith::ConstantIndexOp::create(rewriter, loc, unrollFactor);
  Value newStep = arith::MulIOp::create(rewriter, loc, step, factorConst);

  LLVM_DEBUG(llvm::dbgs() << "Unrolling loop with interleave mode, factor="
                          << unrollFactor << "\n");

  // Create new outer loop with multiplied step
  rewriter.setInsertionPoint(forOp);
  auto outerLoop = scf::ForOp::create(rewriter, loc, lowerBound, upperBound, newStep);
  Value baseIndex = outerLoop.getInductionVar();

  Block *outerBody = outerLoop.getBody();
  rewriter.setInsertionPointToStart(outerBody);

  // Create unroll factor instances
  for (unsigned instance = 0; instance < unrollFactor; ++instance) {
    LLVM_DEBUG(llvm::dbgs() << "  Creating instance " << instance << "\n");

    // Create index mapping for this instance
    IndexMapping mapping = createIndexMapping(
        rewriter, loc, baseIndex, instance, unrollFactor, upperBound,
        /*isInterleaveMode=*/true);

    // Clone the original loop body for this instance
    IRMapping valueMapping;
    valueMapping.map(forOp.getInductionVar(), mapping.actualIndex);

    // Clone all operations from original loop body and apply predication
    Block *originalBody = forOp.getBody();
    Block *outerBody = outerLoop.getBody();

    for (Operation &op : originalBody->without_terminator()) {
      // Always reset insertion point to before the outer loop terminator
      rewriter.setInsertionPoint(outerBody->getTerminator());

      // Check if this is a store operation that needs predication
      if (auto originalStoreOp = dyn_cast<memref::StoreOp>(&op)) {
        // Clone the store
        auto clonedStore = cast<memref::StoreOp>(rewriter.clone(op, valueMapping));

        // Immediately predicate it (this changes insertion point)
        predicateStoreOp(rewriter, clonedStore, mapping.predicate);
      } else {
        // Clone non-store operations normally
        Operation *clonedOp = rewriter.clone(op, valueMapping);

        // Update value mapping with cloned results
        for (auto [origResult, clonedResult] :
             llvm::zip(op.getResults(), clonedOp->getResults())) {
          valueMapping.map(origResult, clonedResult);
        }
      }
    }
  }

  // Mark outer loop with unroll metadata
  outerLoop->setAttr("dsa.unrolled_loop", rewriter.getUnitAttr());
  outerLoop->setAttr("dsa.unroll_factor",
                     rewriter.getI64IntegerAttr(unrollFactor));
  outerLoop->setAttr("dsa.unroll_mode", rewriter.getStringAttr("interleave"));

  // Remove original loop
  rewriter.eraseOp(forOp);

  return success();
}

//===----------------------------------------------------------------------===//
// Block Mode Unrolling
//===----------------------------------------------------------------------===//

/// Unroll loop using block mode: instance k processes [k*B, (k+1)*B)
static LogicalResult unrollBlockMode(scf::ForOp forOp, unsigned unrollFactor,
                                      PatternRewriter &rewriter) {
  Location loc = forOp.getLoc();
  Value lowerBound = forOp.getLowerBound();
  Value upperBound = forOp.getUpperBound();
  Value step = forOp.getStep();

  LLVM_DEBUG(llvm::dbgs() << "Unrolling loop with block mode, factor="
                          << unrollFactor << "\n");

  // Compute block size: blockSize = ceil(tripCount / unrollFactor)
  // For dynamic bounds, we compute: blockSize = (upperBound - lowerBound + step * unrollFactor - 1) / (step * unrollFactor)
  Value range = arith::SubIOp::create(rewriter, loc, upperBound, lowerBound);
  Value factorConst = arith::ConstantIndexOp::create(rewriter, loc, unrollFactor);
  Value scaledStep = arith::MulIOp::create(rewriter, loc, step, factorConst);

  // Add (step * unrollFactor - 1) for ceiling division
  Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
  Value adjustment = arith::SubIOp::create(rewriter, loc, scaledStep, one);
  Value adjustedRange = arith::AddIOp::create(rewriter, loc, range, adjustment);
  Value blockSize = arith::DivUIOp::create(rewriter, loc, adjustedRange, scaledStep);

  // New upper bound = lowerBound + blockSize * step
  Value blockSizeScaled = arith::MulIOp::create(rewriter, loc, blockSize, step);
  Value newUpperBound = arith::AddIOp::create(rewriter, loc, lowerBound, blockSizeScaled);

  // Create outer loop: for offset = lowerBound to lowerBound + blockSize step step
  rewriter.setInsertionPoint(forOp);
  auto outerLoop = scf::ForOp::create(rewriter, loc, lowerBound, newUpperBound, step);
  Value offsetIndex = outerLoop.getInductionVar();

  Block *outerBody = outerLoop.getBody();
  rewriter.setInsertionPointToStart(outerBody);

  // Create unroll factor instances
  for (unsigned instance = 0; instance < unrollFactor; ++instance) {
    LLVM_DEBUG(llvm::dbgs() << "  Creating block instance " << instance << "\n");

    // Calculate actual index for this instance:
    // actualIndex = lowerBound + instance * blockSize * step + (offset - lowerBound)
    Value instanceConst = arith::ConstantIndexOp::create(rewriter, loc, instance);
    Value instanceOffset = arith::MulIOp::create(rewriter, loc, instanceConst, blockSizeScaled);
    Value baseForInstance = arith::AddIOp::create(rewriter, loc, lowerBound, instanceOffset);

    Value offsetFromBase = arith::SubIOp::create(rewriter, loc, offsetIndex, lowerBound);
    Value actualIndex = arith::AddIOp::create(rewriter, loc, baseForInstance, offsetFromBase);

    // Create predicate: actualIndex < upperBound
    Value predicate = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, actualIndex, upperBound);

    // Clone the original loop body for this instance
    IRMapping valueMapping;
    valueMapping.map(forOp.getInductionVar(), actualIndex);

    // Clone all operations from original loop body and apply predication
    Block *originalBody = forOp.getBody();

    for (Operation &op : originalBody->without_terminator()) {
      // Always reset insertion point to before the outer loop terminator
      rewriter.setInsertionPoint(outerBody->getTerminator());

      // Check if this is a store operation that needs predication
      if (auto originalStoreOp = dyn_cast<memref::StoreOp>(&op)) {
        // Clone the store
        auto clonedStore = cast<memref::StoreOp>(rewriter.clone(op, valueMapping));

        // Immediately predicate it
        predicateStoreOp(rewriter, clonedStore, predicate);
      } else {
        // Clone non-store operations normally
        Operation *clonedOp = rewriter.clone(op, valueMapping);

        // Update value mapping with cloned results
        for (auto [origResult, clonedResult] :
             llvm::zip(op.getResults(), clonedOp->getResults())) {
          valueMapping.map(origResult, clonedResult);
        }
      }
    }
  }

  // Mark outer loop with unroll metadata
  outerLoop->setAttr("dsa.unrolled_loop", rewriter.getUnitAttr());
  outerLoop->setAttr("dsa.unroll_factor",
                     rewriter.getI64IntegerAttr(unrollFactor));
  outerLoop->setAttr("dsa.unroll_mode", rewriter.getStringAttr("block"));

  // Remove original loop
  rewriter.eraseOp(forOp);

  return success();
}

//===----------------------------------------------------------------------===//
// Pattern: Unroll Parallel For Loops
//===----------------------------------------------------------------------===//

struct UnrollParallelForPattern : public OpRewritePattern<scf::ForOp> {
  unsigned passUnrollFactor;
  std::string passUnrollMode;

  UnrollParallelForPattern(MLIRContext *context, unsigned unrollFactor,
                            std::string unrollMode)
      : OpRewritePattern<scf::ForOp>(context),
        passUnrollFactor(unrollFactor),
        passUnrollMode(std::move(unrollMode)) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    // Only unroll loops marked with parallel hint
    if (!forOp->hasAttr("dsa.parallel_hint"))
      return failure();

    // Skip if already unrolled
    if (forOp->hasAttr("dsa.unrolled_loop"))
      return failure();

    // Only support loops without iter_args for now
    if (!forOp.getInitArgs().empty()) {
      LLVM_DEBUG(llvm::dbgs() << "Skipping loop with iter_args\n");
      return failure();
    }

    // Get unroll factor
    unsigned unrollFactor = getUnrollFactor(forOp, passUnrollFactor);

    LLVM_DEBUG({
      llvm::dbgs() << "Found parallelizable loop to unroll:\n";
      llvm::dbgs() << "  Unroll factor: " << unrollFactor << "\n";
      llvm::dbgs() << "  Unroll mode: " << passUnrollMode << "\n";
      llvm::dbgs() << "  Static bounds: " << hasStaticBounds(forOp) << "\n";
    });

    // Perform unrolling based on mode
    if (passUnrollMode == "interleave") {
      return unrollInterleaveMode(forOp, unrollFactor, rewriter);
    } else if (passUnrollMode == "block") {
      return unrollBlockMode(forOp, unrollFactor, rewriter);
    }

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct UnrollParallelLoopsPass
    : public impl::UnrollParallelLoopsBase<UnrollParallelLoopsPass> {
  using UnrollParallelLoopsBase::UnrollParallelLoopsBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Only process functions marked for DSA optimization
    if (!func->hasAttr("dsa_optimize"))
      return;

    LLVM_DEBUG(llvm::dbgs() << "Processing function: " << func.getName() << "\n");

    // Apply unrolling patterns
    RewritePatternSet patterns(&getContext());
    patterns.add<UnrollParallelForPattern>(
        &getContext(), unrollFactor, unrollMode);

    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<Pass> createUnrollParallelLoopsPass() {
  return std::make_unique<UnrollParallelLoopsPass>();
}

} // namespace dsa
} // namespace mlir

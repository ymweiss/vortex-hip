//===- NormalizeSCFLoopsPass.cpp - Normalize SCF loops pass ----*- C++ -*-===//
//
// Main pass implementation for normalizing SCF control flow.
//
//===----------------------------------------------------------------------===//

#include "dsa/Transforms/Passes.h"
#include "DetectParallelForPattern.h"
#include "ForallToForPattern.h"
#include "IndexSwitchToIfPattern.h"
#include "IVPatternHelpers.h"
#include "LoopTraits.h"
#include "RemoveZeroTripCheckPattern.h"
#include "WhileToForPattern.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "normalize-scf-loops"

namespace mlir {
namespace dsa {

#define GEN_PASS_DEF_NORMALIZESCFLOOPS
#include "dsa/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Pattern: Convert parallel scf.for to scf.forall
//===----------------------------------------------------------------------===//

struct ConvertParallelForToForall : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    // Only convert if marked as parallel
    if (!forOp->hasAttr("dsa.parallel_hint"))
      return failure();

    // Only convert if no iter_args (no loop-carried dependencies)
    if (!forOp.getInitArgs().empty())
      return failure();

    Location loc = forOp.getLoc();

    // Get loop bounds
    Value lowerBound = forOp.getLowerBound();
    Value upperBound = forOp.getUpperBound();
    Value step = forOp.getStep();

    // Convert Values to OpFoldResult for scf.forall builder
    SmallVector<OpFoldResult> lbs{lowerBound};
    SmallVector<OpFoldResult> ubs{upperBound};
    SmallVector<OpFoldResult> steps{step};

    // Create scf.forall with dynamic bounds
    auto forallOp = scf::ForallOp::create(
        rewriter, loc,
        /*lowerBounds=*/lbs,
        /*upperBounds=*/ubs,
        /*steps=*/steps,
        /*outputs=*/ValueRange{},
        /*mapping=*/std::nullopt);

    // Move loop body
    Block *forBody = forOp.getBody();
    Block *forallBody = forallOp.getBody();

    // Map induction variable
    IRMapping mapper;
    mapper.map(forOp.getInductionVar(), forallOp.getInductionVars()[0]);

    // Clone body operations (without terminator)
    rewriter.setInsertionPointToStart(forallBody);
    for (Operation &op : forBody->without_terminator()) {
      rewriter.clone(op, mapper);
    }

    // Replace for with forall
    rewriter.eraseOp(forOp);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: Restructure inverted scf.while loops
//===----------------------------------------------------------------------===//

struct RestructureInvertedWhilePattern : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
    Block *beforeBlock = whileOp.getBeforeBody();
    Block *doBlock = whileOp.getAfterBody();

    if (!beforeBlock || !doBlock)
      return failure();

    // Check that do region is a pure passthrough
    if (doBlock->getOperations().size() != 1)
      return failure();

    auto doYield = dyn_cast<scf::YieldOp>(&doBlock->front());
    if (!doYield)
      return failure();

    // Verify all yielded values are block arguments unchanged
    if (doYield.getNumOperands() != doBlock->getNumArguments())
      return failure();

    for (auto [yieldVal, blockArg] : llvm::zip(doYield.getOperands(), doBlock->getArguments())) {
      if (yieldVal != blockArg)
        return failure();
    }

    // Check if this loop would be handled by WhileToForPattern
    IVPattern ivPattern = extractIVPattern(whileOp);
    if (ivPattern.valid) {
      // This loop has a simple IV pattern and can be converted to scf.for
      // Don't restructure it - let WhileToForPattern handle it
      return failure();
    }

    // Extract the condition and IV pattern from before region
    auto condOp = dyn_cast<scf::ConditionOp>(beforeBlock->getTerminator());
    if (!condOp)
      return failure();

    // Look for: %cond = arith.cmpi pred, %next_iv, %bound
    Value condition = condOp.getCondition();
    auto cmpOp = condition.getDefiningOp<arith::CmpIOp>();
    if (!cmpOp)
      return failure();

    // Must have at least one iter arg (the IV)
    if (condOp.getArgs().empty())
      return failure();

    // The first argument should be the IV (induction variable)
    Value nextIV = condOp.getArgs()[0];
    auto addOp = nextIV.getDefiningOp<arith::AddIOp>();
    if (!addOp)
      return failure();

    // Check that this addition increments the IV (first block argument)
    BlockArgument ivArg = whileOp.getBeforeArguments()[0];
    Value step;
    if (addOp.getLhs() == ivArg) {
      step = addOp.getRhs();
    } else if (addOp.getRhs() == ivArg) {
      step = addOp.getLhs();
    } else {
      return failure();
    }

    // Verify the condition uses the incremented IV
    if (cmpOp.getLhs() != nextIV && cmpOp.getRhs() != nextIV)
      return failure();

    // Only restructure if there are substantial operations beyond just IV/condition
    size_t bodyOpCount = 0;
    for (Operation &op : beforeBlock->without_terminator()) {
      if (&op != addOp.getOperation() && &op != cmpOp.getOperation())
        bodyOpCount++;
    }

    // Need at least one body operation to be worth restructuring
    if (bodyOpCount == 0)
      return failure();

    // Don't restructure if there's nested control flow
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

    Location loc = whileOp.getLoc();

    // Convert the condition to use the current IV instead of next IV
    arith::CmpIPredicate originalPred = cmpOp.getPredicate();
    arith::CmpIPredicate newPredicate;
    Value bound;

    // Determine which operand is the next_iv and extract bound
    bool ivOnLeft = (cmpOp.getLhs() == nextIV);
    if (ivOnLeft) {
      bound = cmpOp.getRhs();
    } else if (cmpOp.getRhs() == nextIV) {
      bound = cmpOp.getLhs();
      // Invert predicate because operands are swapped
      originalPred = arith::invertPredicate(originalPred);
    } else {
      return failure();
    }

    // Verify bound and step are loop-invariant
    auto isDefinedInWhile = [&](Value val) -> bool {
      if (!val)
        return false;

      Operation *defOp = val.getDefiningOp();
      if (!defOp)
        return false; // Block argument, safe to use

      Region *defRegion = defOp->getParentRegion();
      while (defRegion) {
        if (defRegion == &whileOp.getBefore() || defRegion == &whileOp.getAfter())
          return true;

        Operation *parentOp = defRegion->getParentOp();
        if (!parentOp || parentOp == whileOp.getOperation())
          break;
        defRegion = parentOp->getParentRegion();
      }
      return false;
    };

    if (isDefinedInWhile(bound) || isDefinedInWhile(step))
      return failure();

    // Convert predicate from next_iv to current iv
    switch (originalPred) {
      case arith::CmpIPredicate::ne:
        newPredicate = arith::CmpIPredicate::slt;
        break;
      case arith::CmpIPredicate::ult:
        newPredicate = arith::CmpIPredicate::ult;
        break;
      case arith::CmpIPredicate::slt:
        newPredicate = arith::CmpIPredicate::slt;
        break;
      case arith::CmpIPredicate::ule:
        newPredicate = arith::CmpIPredicate::ule;
        break;
      case arith::CmpIPredicate::sle:
        newPredicate = arith::CmpIPredicate::sle;
        break;
      default:
        return failure();
    }

    // Restructure using a builder callback
    rewriter.setInsertionPoint(whileOp);

    auto newWhile = scf::WhileOp::create(
        rewriter, loc, whileOp.getResultTypes(), whileOp.getInits(),
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          // Build before region - just the condition
          IRMapping beforeMapping;
          for (auto [oldArg, newArg] : llvm::zip(beforeBlock->getArguments(), args)) {
            beforeMapping.map(oldArg, newArg);
          }

          Value currentIV = args[0];
          Value newCond = arith::CmpIOp::create(builder, loc, newPredicate, currentIV, bound);

          SmallVector<Value> condArgs(args.begin(), args.end());
          scf::ConditionOp::create(builder, loc, newCond, condArgs);
        },
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          // Build do region - the loop body
          IRMapping doMapping;
          for (auto [oldArg, newArg] : llvm::zip(beforeBlock->getArguments(), args)) {
            doMapping.map(oldArg, newArg);
          }

          // Clone body operations
          for (Operation &op : beforeBlock->without_terminator()) {
            if (&op == cmpOp.getOperation())
              continue;  // Skip old condition
            Operation *cloned = builder.clone(op, doMapping);
            for (auto [orig, clonedRes] : llvm::zip(op.getResults(), cloned->getResults())) {
              doMapping.map(orig, clonedRes);
            }
          }

          // Yield updated values
          SmallVector<Value> yieldArgs;
          for (Value arg : condOp.getArgs()) {
            yieldArgs.push_back(doMapping.lookupOrDefault(arg));
          }
          scf::YieldOp::create(builder, loc, yieldArgs);
        });

    rewriter.replaceOp(whileOp, newWhile.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: Convert canonical scf.while to scf.for
//===----------------------------------------------------------------------===//

struct ConvertCanonicalWhileToForPattern : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
    // Must have at least one result (the induction variable)
    if (whileOp.getNumResults() < 1)
      return failure();

    // Extract canonical IV pattern
    IVPattern pattern = extractCanonicalIVPattern(whileOp);
    if (!pattern.valid)
      return failure();

    // Get blocks
    Block *beforeBlock = whileOp.getBeforeBody();
    Block *afterBlock = whileOp.getAfterBody();
    if (!beforeBlock || !afterBlock)
      return failure();

    // Check that before region has simple structure (just condition check)
    bool hasOnlyCmp = true;
    for (Operation &op : beforeBlock->without_terminator()) {
      if (!isa<arith::CmpIOp>(&op)) {
        hasOnlyCmp = false;
        break;
      }
    }
    if (!hasOnlyCmp)
      return failure();

    // Should have exactly one operation (the comparison)
    if (std::distance(beforeBlock->without_terminator().begin(),
                      beforeBlock->without_terminator().end()) != 1)
      return failure();

    // Only transform simple loops without nested control flow
    bool hasNestedControlFlow = false;
    afterBlock->walk([&](Operation *op) {
      if (isa<scf::WhileOp, scf::ForOp, scf::IfOp>(op)) {
        hasNestedControlFlow = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (hasNestedControlFlow)
      return failure();

    // Ensure bound and step are loop-invariant
    auto isDefinedInWhile = [&](Value val) -> bool {
      Operation *defOp = val.getDefiningOp();
      if (!defOp)
        return false;

      Region *defRegion = defOp->getParentRegion();
      while (defRegion) {
        if (defRegion == &whileOp.getBefore() || defRegion == &whileOp.getAfter())
          return true;

        Operation *parentOp = defRegion->getParentOp();
        if (!parentOp || parentOp == whileOp.getOperation())
          break;
        defRegion = parentOp->getParentRegion();
      }
      return false;
    };

    if (isDefinedInWhile(pattern.init) || isDefinedInWhile(pattern.bound) ||
        isDefinedInWhile(pattern.step))
      return failure();

    // Only handle specific predicates
    if (pattern.predicate != arith::CmpIPredicate::ult &&
        pattern.predicate != arith::CmpIPredicate::slt &&
        pattern.predicate != arith::CmpIPredicate::ule &&
        pattern.predicate != arith::CmpIPredicate::sle)
      return failure();

    Location loc = whileOp.getLoc();
    Type ivType = whileOp.getResult(0).getType();

    // Set insertion point before while loop
    rewriter.setInsertionPoint(whileOp);

    // Convert values to index type for scf.for
    Value initIdx, stepIdx, boundIdx;

    if (ivType.isIndex()) {
      initIdx = pattern.init;
      stepIdx = pattern.step;
      boundIdx = pattern.bound;
    } else {
      // Cast to index
      initIdx = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), pattern.init);
      stepIdx = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), pattern.step);
      boundIdx = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), pattern.bound);
    }

    // Adjust bounds for inclusive predicates
    if (pattern.predicate == arith::CmpIPredicate::ule ||
        pattern.predicate == arith::CmpIPredicate::sle) {
      Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
      boundIdx = arith::AddIOp::create(rewriter, loc, boundIdx, one);
    }

    // Collect iter_args (all values except the IV)
    SmallVector<Value> iterArgs;
    for (unsigned i = 1; i < whileOp.getNumResults(); ++i) {
      iterArgs.push_back(whileOp.getInits()[i]);
    }

    // Create scf.for
    auto forOp = scf::ForOp::create(rewriter, loc, initIdx, boundIdx, stepIdx, iterArgs);

    // Mark as converted from while loop
    forOp->setAttr("dsa.converted_from_while", rewriter.getUnitAttr());

    Block *forBody = forOp.getBody();

    // Set insertion point to start of for body
    rewriter.setInsertionPointToStart(forBody);

    // Cast IV if needed
    Value forIV = forOp.getInductionVar();
    if (!ivType.isIndex()) {
      forIV = arith::IndexCastOp::create(rewriter, loc, ivType, forOp.getInductionVar());
    }

    // Create mapping for block arguments
    IRMapping mapping;
    mapping.map(afterBlock->getArgument(0), forIV);

    // Map iter_args
    for (unsigned i = 1; i < whileOp.getNumResults(); ++i) {
      mapping.map(afterBlock->getArgument(i), forBody->getArgument(i));
    }

    // Find the increment operation
    auto yieldOp = cast<scf::YieldOp>(afterBlock->getTerminator());
    Value nextIV = yieldOp.getOperands()[0];
    Operation *incrementOp = nextIV.getDefiningOp();

    if (!incrementOp)
      return failure();

    // Clone ALL operations from after region (including increment)
    for (Operation &op : afterBlock->without_terminator()) {
      Operation *clonedOp = rewriter.clone(op, mapping);

      // Update mapping with cloned results
      for (auto [originalResult, clonedResult] :
           llvm::zip(op.getResults(), clonedOp->getResults())) {
        mapping.map(originalResult, clonedResult);
      }
    }

    // Create scf.yield with iter_args (excluding IV)
    SmallVector<Value> yieldValues;
    for (unsigned i = 1; i < yieldOp.getNumOperands(); ++i) {
      Value mappedValue = mapping.lookupOrDefault(yieldOp.getOperands()[i]);
      yieldValues.push_back(mappedValue);
    }

    // Handle the existing yield terminator
    if (forBody->mightHaveTerminator()) {
      Operation *terminator = forBody->getTerminator();
      if (terminator && isa<scf::YieldOp>(terminator)) {
        rewriter.setInsertionPoint(terminator);
        scf::YieldOp::create(rewriter, loc, yieldValues);
        rewriter.eraseOp(terminator);
      } else {
        rewriter.setInsertionPointToEnd(forBody);
        scf::YieldOp::create(rewriter, loc, yieldValues);
      }
    } else {
      rewriter.setInsertionPointToEnd(forBody);
      scf::YieldOp::create(rewriter, loc, yieldValues);
    }

    // Replace while op results
    SmallVector<Value> replacementValues;

    // First result: final IV value (use upper bound)
    Value finalIV = boundIdx;
    if (!ivType.isIndex()) {
      rewriter.setInsertionPointAfter(forOp);
      finalIV = arith::IndexCastOp::create(rewriter, loc, ivType, boundIdx);
    }
    replacementValues.push_back(finalIV);

    // Remaining results: for loop results
    for (Value result : forOp.getResults()) {
      replacementValues.push_back(result);
    }

    rewriter.replaceOp(whileOp, replacementValues);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: Convert scf.for to scf.while (for --no-use-stream)
//===----------------------------------------------------------------------===//

struct ForLoopLoweringPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    // Generate type signature for the loop-carried values
    SmallVector<Type> lcvTypes;
    SmallVector<Location> lcvLocs;
    lcvTypes.push_back(forOp.getInductionVar().getType());
    lcvLocs.push_back(forOp.getInductionVar().getLoc());
    for (Value value : forOp.getInitArgs()) {
      lcvTypes.push_back(value.getType());
      lcvLocs.push_back(value.getLoc());
    }

    // Build scf.WhileOp
    SmallVector<Value> initArgs;
    initArgs.push_back(forOp.getLowerBound());
    llvm::append_range(initArgs, forOp.getInitArgs());
    auto whileOp = scf::WhileOp::create(rewriter, forOp.getLoc(), lcvTypes,
                                        initArgs, forOp->getAttrs());

    // Build 'before' region
    auto *beforeBlock = rewriter.createBlock(
        &whileOp.getBefore(), whileOp.getBefore().begin(), lcvTypes, lcvLocs);
    rewriter.setInsertionPointToStart(whileOp.getBeforeBody());
    arith::CmpIPredicate predicate = forOp.getUnsignedCmp()
                                         ? arith::CmpIPredicate::ult
                                         : arith::CmpIPredicate::slt;
    auto cmpOp = arith::CmpIOp::create(rewriter, whileOp.getLoc(), predicate,
                                       beforeBlock->getArgument(0),
                                       forOp.getUpperBound());
    scf::ConditionOp::create(rewriter, whileOp.getLoc(), cmpOp.getResult(),
                             beforeBlock->getArguments());

    // Build 'after' region
    auto *afterBlock = rewriter.createBlock(
        &whileOp.getAfter(), whileOp.getAfter().begin(), lcvTypes, lcvLocs);

    // Add induction variable incrementation
    rewriter.setInsertionPointToEnd(afterBlock);
    auto ivIncOp =
        arith::AddIOp::create(rewriter, whileOp.getLoc(),
                              afterBlock->getArgument(0), forOp.getStep());

    // Rewrite uses of the for-loop block arguments to the new while-loop arguments
    for (const auto &barg : llvm::enumerate(forOp.getBody(0)->getArguments()))
      rewriter.replaceAllUsesWith(barg.value(),
                                  afterBlock->getArgument(barg.index()));

    // Inline for-loop body operations into 'after' region
    for (auto &arg : llvm::make_early_inc_range(*forOp.getBody()))
      rewriter.moveOpBefore(&arg, afterBlock, afterBlock->end());

    // Add incremented IV to yield operations
    for (auto yieldOp : afterBlock->getOps<scf::YieldOp>()) {
      SmallVector<Value> yieldOperands = yieldOp.getOperands();
      yieldOperands.insert(yieldOperands.begin(), ivIncOp.getResult());
      rewriter.modifyOpInPlace(yieldOp,
                               [&]() { yieldOp->setOperands(yieldOperands); });
    }

    // Replace for op results
    for (const auto &arg : llvm::enumerate(forOp.getResults()))
      rewriter.replaceAllUsesWith(arg.value(),
                                  whileOp.getResult(arg.index() + 1));

    rewriter.eraseOp(forOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct NormalizeSCFLoopsPass
    : public impl::NormalizeSCFLoopsBase<NormalizeSCFLoopsPass> {

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Phase 1: Convert scf.index_switch to nested scf.if
    // This must happen early since index_switch has no Handshake lowering
    {
      RewritePatternSet patterns(&getContext());
      patterns.add<IndexSwitchToIfPattern>(&getContext());
      if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }

    // Phase 2: Detect parallelizable loops (annotation only)
    {
      RewritePatternSet patterns(&getContext());
      patterns.add<DetectParallelForPattern>(&getContext());
      if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }

    // Phase 3: Remove redundant zero-trip checks (ONLY for scf.for)
    // NOTE: The following patterns are intentionally NOT used because they are unsafe:
    //
    // 1. RemoveEmptyIfAroundLoopPattern - REMOVED
    //    Problem: Hoists memory operations from else region before checking condition
    //    Example: scf.if %N_eq_0 {} else { load[0]; while ... }
    //    Result: load[0] executes even when N=0 → buffer overflow!
    //
    // 2. RemoveZeroTripCheckAroundLoopPattern<scf::WhileOp> - REMOVED
    //    Problem: Assumes while loop checks condition before executing body
    //    Reality: LLVM-generated while loops execute body in before region
    //    Result: Zero-trip loops still execute body once → buffer overflow!
    //
    // Only RemoveZeroTripCheck for scf.for is safe (condition checked before body):
    {
      RewritePatternSet patterns(&getContext());
      patterns.add<RemoveZeroTripCheckAroundLoopPattern<scf::ForOp>>(&getContext());
      if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }

    // Phase 4: Loop canonicalization (depends on --no-use-stream option)
    {
      RewritePatternSet patterns(&getContext());
      if (!noUseStream) {
        // Normal mode: Convert while→for to enable stream+gate pattern
        patterns.add<ConvertCanonicalWhileToForPattern>(&getContext());
        patterns.add<WhileToForPattern>(&getContext());
        patterns.add<RestructureInvertedWhilePattern>(&getContext());
      } else {
        // --no-use-stream mode: Convert for→while to force carry+invariant pattern
        patterns.add<ForLoopLoweringPattern>(&getContext());
      }
      patterns.add<ForallToForPattern>(&getContext());

      if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createNormalizeSCFLoopsPass() {
  return std::make_unique<NormalizeSCFLoopsPass>();
}

} // namespace dsa
} // namespace mlir

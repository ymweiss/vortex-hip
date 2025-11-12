//===- SCFLoopConversion.cpp - SCF loop conversion patterns -----*- C++ -*-===//
//
// Conversion patterns for SCF loop operations (ForOp, WhileOp)
//
//===----------------------------------------------------------------------===//

#include "SCFConversionPatterns.h"
#include <functional>

namespace mlir {
namespace dsa {

//===----------------------------------------------------------------------===//
// SCF ForOp Conversion Pattern
//===----------------------------------------------------------------------===//

LogicalResult ConvertForOp::matchAndRewrite(scf::ForOp forOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const {
    Location loc = forOp.getLoc();

    // Check if this operation has a valueMap pointer attribute from nested conversion
    DenseMap<Value, Value> *parentValueMap = nullptr;
    if (auto attr = forOp->getAttr("__dsa_value_map_ptr")) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        parentValueMap = reinterpret_cast<DenseMap<Value, Value> *>(
            intAttr.getValue().getZExtValue());
        // Remove the temporary attribute
        forOp->removeAttr("__dsa_value_map_ptr");
      }
    }

    rewriter.setInsertionPoint(forOp);

    Value lowerBound = adaptor.getLowerBound();
    Value upperBound = adaptor.getUpperBound();
    Value step = adaptor.getStep();

    // Stream generates condition computation flow (before region, N+1 outputs)
    // In scf.while terms: stream implements before region's condition check
    auto streamOp = dsa::StreamOp::create(rewriter,
        loc, lowerBound, step, upperBound);
    Value idx_raw = streamOp.getResult(0);          // Trailing args (N+1)
    Value willContinue_raw = streamOp.getResult(1); // scf.condition (N+1)

    // Gate transforms before region outputs (N+1) to after region inputs (N)
    // This is fundamental: condition evaluation happens N+1 times, but loop
    // body executes only N times. Gate bridges this inherent asymmetry.
    //
    // scf.while analogy:
    // - Before region: Executes N+1 times to compute condition + trailing args
    // - Gate: Adapts N+1 condition outputs to N loop body inputs
    // - After region: Executes N times using gated inputs
    //
    // Example: for i in [0,5):
    //   stream (before):  idx=[0,1,2,3,4,5], willContinue=[T,T,T,T,T,F] (N+1=6)
    //   gate (adapter):   trims head cond and tail index
    //   after (body):     idx=[0,1,2,3,4], willContinue=[T,T,T,T,F] (N=5)
    auto gateOp = dsa::GateOp::create(rewriter, loc, idx_raw, willContinue_raw);
    Value idx = gateOp.getIndex();              // After region inputs (N times)
    Value willContinue = gateOp.getCond();      // After region execution control (N times)

    // Mark willContinue (gated) as controlling "for.body" region
    markControlValue(willContinue, "for", "body", forOp, rewriter);

    Block *body = forOp.getBody();

    // CRITICAL: Mark all memref operations in the loop body BEFORE any processing
    // NOTE: Memory operations are already marked with complete paths during preprocessing
    // No need to mark them again here

    llvm::SetVector<Value> invariantValues;

    // CRITICAL: Use iterative work queue to collect ALL external values used anywhere
    // in the loop body, including those deep inside nested SCF structures
    SmallVector<Operation *> workList;
    llvm::DenseSet<Operation *> visited;

    // Initialize with top-level operations (skip terminator)
    for (Operation &op : body->without_terminator()) {
      workList.push_back(&op);
    }

    while (!workList.empty()) {
      Operation *op = workList.pop_back_val();

      // Skip if already visited
      if (!visited.insert(op).second) {
        continue;
      }

      // Check all operands of this operation
      for (Value operand : op->getOperands()) {
        // This correctly handles values used in nested SCF operations
        if (forOp.isDefinedOutsideOfLoop(operand)) {
          invariantValues.insert(operand);
        }
      }

      // Add operations from nested regions to work list
      // IMPORTANT: Skip terminators in nested regions (scf.yield, scf.condition, etc.)
      // Their operands have special semantics (loop-carried values, not invariants)
      for (Region &region : op->getRegions()) {
        if (region.empty()) continue;
        for (Block &block : region) {
          for (Operation &nestedOp : block.without_terminator()) {
            workList.push_back(&nestedOp);
          }
          // CRITICAL: For nested scf.if/switch operations, also check their yield operands
          // These represent values passed through the SCF operation and need routing
          Operation *terminator = block.getTerminator();
          if (auto yieldOp = dyn_cast<scf::YieldOp>(terminator)) {
            for (Value yieldVal : yieldOp.getOperands()) {
              if (forOp.isDefinedOutsideOfLoop(yieldVal)) {
                invariantValues.insert(yieldVal);
              }
            }
          }
        }
      }
    }

    IRMapping mapper;

    // Map gated index to induction variable (after region receives N values)
    mapper.map(forOp.getInductionVar(), idx);

    // Create carry operations for loop-carried values
    // In scf.while terms:
    // - carry.d (willContinue): scf.condition operand (controls before→after)
    // - carry.a (initVal): initial values entering before region from outside
    // - carry.b (feedback): loop-carried values from after region to before region
    //
    // CRITICAL: Use willContinue_raw (N+1 times from stream) for carry control
    // - Carry outputs N+1 times: init_value, then N feedback values
    // - Loop body executes N times using willContinue from gate
    //
    // NEW SEMANTICS: Carry output must be filtered through cond_br
    // - carry output (N+1) → cond_br(willContinue_raw)
    // - true branch (N): used in loop body, feedback to carry.b
    // - false branch (1): final result for loop exit
    SmallVector<dsa::CarryOp> carryOps;
    SmallVector<Value> carryTrueValues;  // Values used in loop body
    SmallVector<Value> carryFalseValues; // Values for loop exit

    // Fork willContinue_raw for multiple carry operations
    SmallVector<Value> willContinueForks;
    if (adaptor.getInitArgs().size() > 1) {
      auto willContinueFork = circt::handshake::ForkOp::create(rewriter,
          loc, willContinue_raw, adaptor.getInitArgs().size());
      for (unsigned i = 0; i < adaptor.getInitArgs().size(); ++i) {
        willContinueForks.push_back(willContinueFork->getResult(i));
      }
    } else if (adaptor.getInitArgs().size() == 1) {
      willContinueForks.push_back(willContinue_raw);
    }

    for (auto [initVal, regionArg, willCont] :
         llvm::zip(adaptor.getInitArgs(), forOp.getRegionIterArgs(), willContinueForks)) {
      // Create carry with placeholder for feedback (will be set later)
      auto carryOp = dsa::CarryOp::create(rewriter,
          loc, regionArg.getType(), willCont, initVal, initVal);
      carryOps.push_back(carryOp);

      // Filter carry output through cond_br based on willContinue_raw
      // true branch: enter loop body (N times)
      // false branch: exit loop (1 time)
      auto condBrOp = circt::handshake::ConditionalBranchOp::create(rewriter,
          loc,
          carryOp.getResult().getType(),  // trueResult type
          carryOp.getResult().getType(),  // falseResult type
          willCont,                        // condition operand
          carryOp.getResult());            // data operand

      // Map the true branch (loop body value) to regionArg
      mapper.map(regionArg, condBrOp.getTrueResult());
      carryTrueValues.push_back(condBrOp.getTrueResult());
      carryFalseValues.push_back(condBrOp.getFalseResult());
    }

    // Create invariant operations for loop-invariant values
    // In scf.while terms:
    // - invariant.a: external values used in before/after region but not loop-carried
    // - invariant.d (willContinue): same control signal as carry (loop continuation)
    SmallVector<dsa::InvariantOp> invariantOps;
    for (Value invariant : invariantValues) {
      // Skip memref types - they're handled by memory operations
      if (isa<MemRefType>(invariant.getType()))
        continue;

      Value mappedInvariant = mapper.lookupOrDefault(invariant);
      auto invariantOp = dsa::InvariantOp::create(rewriter,
          loc, invariant.getType(), willContinue, mappedInvariant);
      mapper.map(invariant, invariantOp.getResult());
      invariantOps.push_back(invariantOp);
    }

    SmallVector<Operation *> clonedOps;
    for (Operation &op : body->without_terminator()) {
      Operation *clonedOp = rewriter.clone(op, mapper);
      clonedOps.push_back(clonedOp);
    }

    // Create a valueMap for nested SCF operations
    DenseMap<Value, Value> nestedValueMap;
    if (failed(convertNestedSCFOps(clonedOps, rewriter, nestedValueMap)))
      return failure();

    auto yieldOp = cast<scf::YieldOp>(body->getTerminator());
    SmallVector<Value> mappedYieldValues;
    for (Value yieldVal : yieldOp.getOperands()) {
      // First try nestedValueMap (for values from nested SCF conversions)
      // Then fall back to mapper (for values from current level cloning)
      auto it = nestedValueMap.find(yieldVal);
      Value mappedVal = (it != nestedValueMap.end()) ? it->second : mapper.lookupOrDefault(yieldVal);
      mappedYieldValues.push_back(mappedVal);
    }

    for (auto [carryOp, yieldVal] :
         llvm::zip(carryOps, mappedYieldValues)) {
      carryOp.setOperand(2, yieldVal);
    }

    if (failed(verifyZeroTripSemantics(streamOp, carryOps))) {
      return failure();
    }

    // Result extraction: use carryFalseValues (already filtered by cond_br above)
    // - carryTrueValues (N times): used in loop body, feedback to carry.b
    // - carryFalseValues (1 time): final result for loop exit
    //
    // No need for additional cond_br here since we already filtered at carry output
    SmallVector<Value> results;

    if (!carryOps.empty()) {
      // Use the false branch values (final loop results) directly
      results = carryFalseValues;
    } else {
      // No iter_args - just need to ensure loop executes and completes
      // Loop body operations were already cloned and nested SCF converted above
      // No results to return
    }

    // If this is a nested conversion, populate the parent's valueMap
    if (parentValueMap) {
      for (unsigned i = 0; i < forOp->getNumResults(); ++i) {
        (*parentValueMap)[forOp->getResult(i)] = results[i];
      }
    }

    // Replace the for operation
    rewriter.replaceOp(forOp, results);

    return success();
}

//===----------------------------------------------------------------------===//
// SCF WhileOp Conversion Pattern
//===----------------------------------------------------------------------===//

LogicalResult ConvertWhileOp::matchAndRewrite(scf::WhileOp whileOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const {
    Location loc = whileOp.getLoc();

    rewriter.setInsertionPoint(whileOp);

    Block *beforeBlock = &whileOp.getBefore().front();
    Block *afterBlock = &whileOp.getAfter().front();

    // NOTE: Memory operations are already marked with complete paths during preprocessing
    // No need to mark them again here

    // Check if this operation has a valueMap pointer attribute from nested conversion
    DenseMap<Value, Value> *parentValueMap = nullptr;
    if (auto attr = whileOp->getAttr("__dsa_value_map_ptr")) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        parentValueMap = reinterpret_cast<DenseMap<Value, Value> *>(
            intAttr.getValue().getZExtValue());
        // Remove the temporary attribute
        whileOp->removeAttr("__dsa_value_map_ptr");
      }
    }

    IRMapping mapper;

    Value placeholderCond = arith::ConstantOp::create(rewriter,
        loc, rewriter.getI1Type(), rewriter.getBoolAttr(true));

    size_t numResults = whileOp.getNumResults();
    size_t numInits = adaptor.getInits().size();

    SmallVector<dsa::CarryOp> carryOps;
    for (size_t i = 0; i < numResults; i++) {
      Value initVal;
      Type resultType;

      if (i < numInits) {
        initVal = adaptor.getInits()[i];
        resultType = whileOp.getBeforeArguments()[i].getType();

        auto carryOp = dsa::CarryOp::create(rewriter,
            loc, resultType, placeholderCond, initVal, initVal);

        mapper.map(whileOp.getBeforeArguments()[i], carryOp.getResult());
        mapper.map(whileOp.getAfterArguments()[i], carryOp.getResult());
        // CRITICAL: Map the result as well, so nested operations can reference it
        mapper.map(whileOp.getResult(i), carryOp.getResult());

        carryOps.push_back(carryOp);
      } else {

        resultType = whileOp.getResultTypes()[i];

        TypedAttr zeroAttr;
        if (isa<IntegerType>(resultType)) {
          zeroAttr = rewriter.getIntegerAttr(resultType, 0);
        } else if (isa<FloatType>(resultType)) {
          zeroAttr = rewriter.getFloatAttr(resultType, 0.0);
        } else if (isa<IndexType>(resultType)) {
          zeroAttr = rewriter.getIntegerAttr(rewriter.getIndexType(), 0);
        } else {
          return whileOp.emitError("Unsupported type for exit-only value in scf.while");
        }

        initVal = arith::ConstantOp::create(rewriter, loc, zeroAttr);

        auto carryOp = dsa::CarryOp::create(rewriter,
            loc, resultType, placeholderCond, initVal, initVal);

        mapper.map(whileOp.getAfterArguments()[i], carryOp.getResult());
        // CRITICAL: Map the result as well, so nested operations can reference it
        mapper.map(whileOp.getResult(i), carryOp.getResult());

        carryOps.push_back(carryOp);
      }
    }

    // Collect invariants separately for before and after regions
    // Before region invariants: used to compute condition (N+1 executions)
    // After region invariants: used in loop body (N executions)
    llvm::SetVector<Value> beforeInvariants;
    llvm::SetVector<Value> afterInvariants;

    // Helper to check if a value is defined outside this while loop
    auto isDefinedOutsideWhile = [&](Value operand, ArrayRef<BlockArgument> loopArgs) -> bool {
      // Check if it's a loop argument
      for (BlockArgument arg : loopArgs) {
        if (operand == arg) {
          return false;
        }
      }

      // Check if operand is from outside the while loop
      if (auto *defOp = operand.getDefiningOp()) {
        // Walk up parent operations to check if defOp is inside this while loop
        Operation *parentOp = defOp->getParentOp();
        while (parentOp) {
          if (parentOp == whileOp.getOperation()) {
            return false; // Defined inside
          }
          parentOp = parentOp->getParentOp();
        }
        return true; // Defined outside
      } else if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
        // Walk up from the block argument's owner block to check if it's inside this while loop
        // This correctly handles block arguments from nested SCF operations
        Operation *parentOp = blockArg.getOwner()->getParentOp();
        while (parentOp) {
          if (parentOp == whileOp.getOperation()) {
            return false; // Defined inside
          }
          parentOp = parentOp->getParentOp();
        }
        return true; // External block arg
      }
      return true;
    };

    // CRITICAL: Use iterative work queue to collect ALL external values used anywhere
    // in the region, including those deep inside nested SCF structures
    auto collectInvariantsIterative = [&](Block *startBlock, llvm::SetVector<Value> &invariants,
                                          ArrayRef<BlockArgument> loopArgs) {
      SmallVector<Operation *> workList;
      llvm::DenseSet<Operation *> visited;

      // Initialize with top-level operations (skip terminator)
      for (Operation &op : startBlock->without_terminator()) {
        workList.push_back(&op);
      }

      while (!workList.empty()) {
        Operation *op = workList.pop_back_val();

        // Skip if already visited
        if (!visited.insert(op).second) {
          continue;
        }

        // Check all operands of this operation
        for (Value operand : op->getOperands()) {
          if (isDefinedOutsideWhile(operand, loopArgs)) {
            invariants.insert(operand);
          }
        }

        // Add operations from nested regions to work list
        // IMPORTANT: Skip terminators in nested regions (scf.yield, scf.condition, etc.)
        // Their operands have special semantics (loop-carried values, not invariants)
        for (Region &region : op->getRegions()) {
          if (region.empty()) continue;
          for (Block &block : region) {
            for (Operation &nestedOp : block.without_terminator()) {
              workList.push_back(&nestedOp);
            }
            // CRITICAL: For nested scf.if/switch operations, also check their yield operands
            // These represent values passed through the SCF operation and need routing
            Operation *terminator = block.getTerminator();
            if (auto yieldOp = dyn_cast<scf::YieldOp>(terminator)) {
              for (Value yieldVal : yieldOp.getOperands()) {
                if (isDefinedOutsideWhile(yieldVal, loopArgs)) {
                  invariants.insert(yieldVal);
                }
              }
            }
          }
        }
      }
    };

    // Collect invariants for before and after regions
    collectInvariantsIterative(beforeBlock, beforeInvariants, whileOp.getBeforeArguments());
    collectInvariantsIterative(afterBlock, afterInvariants, whileOp.getAfterArguments());


    // Step 1: Create before region invariants (N+1 executions)
    // These use placeholder willContinue, will be updated to raw willContinue later
    SmallVector<dsa::InvariantOp> beforeInvariantOps;
    for (Value invariant : beforeInvariants) {
      if (isa<MemRefType>(invariant.getType()))
        continue;

      Value mappedInvariant = mapper.lookupOrDefault(invariant);
      auto invOp = dsa::InvariantOp::create(rewriter,
          loc, invariant.getType(), placeholderCond, mappedInvariant);
      mapper.map(invariant, invOp.getResult());
      beforeInvariantOps.push_back(invOp);
    }

    // Step 2: Clone before region operations to compute condition (INCLUDING conditionOp!)
    SmallVector<Operation *> clonedBeforeOps;
    for (Operation &op : beforeBlock->without_terminator()) {
      Operation *clonedOp = rewriter.clone(op, mapper);
      clonedBeforeOps.push_back(clonedOp);
    }

    // CRITICAL: Clone the conditionOp BEFORE converting nested SCF ops!
    // When we clone it with the mapper, it gets inserted at the current insertion point.
    // Later, when convertNestedSCFOps replaces scf.while ops with carry ops,
    // MLIR automatically updates ALL uses of the replaced values, including
    // this cloned condition's operands!
    auto originalCondition = cast<scf::ConditionOp>(beforeBlock->getTerminator());
    auto clonedCondition = cast<scf::ConditionOp>(rewriter.clone(*originalCondition, mapper));

    DSA_DEBUG_STREAM << "[DEBUG ConvertWhileOp] BEFORE convertNestedSCFOps:\n";
    DSA_DEBUG_STREAM << "  clonedCondition has " << clonedCondition.getArgs().size() << " args:\n";
    for (unsigned i = 0; i < clonedCondition.getArgs().size(); ++i) {
      Value arg = clonedCondition.getArgs()[i];
      DSA_DEBUG_STREAM << "    arg[" << i << "]: ";
      if (!arg) {
        DSA_DEBUG_STREAM << "NULL\n";
      } else if (auto defOp = arg.getDefiningOp()) {
        DSA_DEBUG_STREAM << defOp->getName() << "\n";
      } else {
        DSA_DEBUG_STREAM << "BlockArgument\n";
      }
    }

    // Create a valueMap for nested SCF operations
    DenseMap<Value, Value> nestedValueMap;
    if (failed(convertNestedSCFOps(clonedBeforeOps, rewriter, nestedValueMap)))
      return failure();

    DSA_DEBUG_STREAM << "[DEBUG ConvertWhileOp] AFTER convertNestedSCFOps:\n";
    DSA_DEBUG_STREAM << "  clonedCondition has " << clonedCondition.getArgs().size() << " args:\n";
    for (unsigned i = 0; i < clonedCondition.getArgs().size(); ++i) {
      Value arg = clonedCondition.getArgs()[i];
      DSA_DEBUG_STREAM << "    arg[" << i << "]: ";
      if (!arg) {
        DSA_DEBUG_STREAM << "NULL\n";
      } else if (auto defOp = arg.getDefiningOp()) {
        DSA_DEBUG_STREAM << defOp->getName() << "\n";
      } else {
        DSA_DEBUG_STREAM << "BlockArgument\n";
      }
    }

    // Get the condition value and args from the cloned condition
    // These should now point to the converted operations (carry ops) thanks to MLIR's automatic update
    Value condition = clonedCondition.getCondition();

    // Step 3: The condition computed from before region IS the raw willContinue (N+1 length)
    // No need for a separate condition carry - condition is computed directly from carry outputs
    // This matches user's explanation: carry outputs → arith.cmpi → condition (raw willContinue)
    Value rawWillContinue = condition;

    // Mark rawWillContinue as controlling "while.before" region
    markControlValue(rawWillContinue, "while", "before", whileOp, rewriter);

    // Step 4: Update before region invariants to use raw willContinue (N+1)
    for (auto invOp : beforeInvariantOps) {
      invOp.getOperation()->setOperand(0, rawWillContinue);
    }

    // Step 5: Carry outputs are used ONLY in before region, no need to fork for exit
    // The while loop's exit values come from conditionArgExitValues (condition args' false branch)
    // Carry outputs should be used directly in before region (already mapped to beforeArguments above)

    // Step 6: Create gate to adapt raw willContinue (N+1) → gated willContinue (N)
    // Gate needs a dummy index input repeated N+1 times (matching rawWillContinue frequency)
    // Create a constant index 0 and use invariant to repeat it N+1 times
    Value dummyIndexConst = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);

    // Use dsa.invariant to repeat the dummy index N+1 times (controlled by rawWillContinue)
    auto dummyInvariantOp = dsa::InvariantOp::create(rewriter,
        loc, rewriter.getIndexType(), rawWillContinue, dummyIndexConst);
    Value dummyIndex = dummyInvariantOp.getResult();

    // Gate consumes dummy index and rawWillContinue (both N+1 times)
    // - When condition=TRUE: outputs dummy index (which we sink) and cond=TRUE
    // - When condition=FALSE: discards dummy, outputs only cond=FALSE
    auto gateOp = dsa::GateOp::create(rewriter, loc, dummyIndex, rawWillContinue);
    Value gatedIndex = gateOp.getIndex();        // N-length (gate filtered)
    Value gatedWillContinue = gateOp.getCond();  // N-length

    // Mark gatedWillContinue as controlling "while.after" region
    markControlValue(gatedWillContinue, "while", "after", whileOp, rewriter);

    // Sink the gated dummy index (not needed, we use actual carry outputs for after region)
    circt::handshake::SinkOp::create(rewriter, loc, gatedIndex);

    // Step 7: Route condition args through cond_br to after region
    // CRITICAL: According to scf.while semantics, trailing args from scf.condition should only
    // enter after region when condition=true. This means condition args (N+1 times) must go through
    // cond_br, and only the TRUE branch (N times) enters after region.
    //
    // Data flow: before region → condition args (N+1) → cond_br → true branch (N) → after region
    //
    // CRITICAL: After convertNestedSCFOps, we need to use the nestedValueMap to get the converted values!
    // The clonedCondition's args may reference operations inside nested SCF regions that were
    // converted. For example, if a nested while's before region computes a value used in the
    // condition, after conversion that value is produced by a carry op, not the original operation.
    // We must look up the mapped value in the nestedValueMap!
    SmallVector<Value> mappedConditionArgs;
    for (Value condArg : clonedCondition.getArgs()) {
      // First try nestedValueMap (for values from nested SCF conversions)
      // Then fall back to mapper (for values from current level cloning)
      auto it = nestedValueMap.find(condArg);
      Value mappedArg = (it != nestedValueMap.end()) ? it->second : mapper.lookupOrDefault(condArg);
      mappedConditionArgs.push_back(mappedArg);
    }

    // Route each condition arg through cond_br to control entry into after region
    // Fork rawWillContinue if multiple condition args need it
    // IMPORTANT: The FALSE branch (condition=false) of condition args is the while loop's EXIT VALUE
    // This is the core semantics of scf.while: when condition becomes false, trailing args are returned
    SmallVector<Value> conditionArgGatedValues;  // TRUE branch: N times, enters after region
    SmallVector<Value> conditionArgExitValues;   // FALSE branch: 1 time, while loop results

    if (!mappedConditionArgs.empty()) {
      SmallVector<Value> rawWillContinueForks;
      if (mappedConditionArgs.size() > 1) {
        auto forkOp = circt::handshake::ForkOp::create(rewriter,
            loc, rawWillContinue, mappedConditionArgs.size());
        for (unsigned i = 0; i < mappedConditionArgs.size(); ++i) {
          rawWillContinueForks.push_back(forkOp->getResult(i));
        }
      } else {
        rawWillContinueForks.push_back(rawWillContinue);
      }

      // Route each condition arg through cond_br
      for (auto [condArg, willCont] : llvm::zip(mappedConditionArgs, rawWillContinueForks)) {
        // Use cond_br: when condition=true, pass condArg to after region
        // when condition=false, condArg becomes the while loop's exit value
        auto condBrOp = circt::handshake::ConditionalBranchOp::create(rewriter,
            loc, condArg.getType(), condArg.getType(),
            willCont, condArg);

        // TRUE branch (N times): goes to after region
        conditionArgGatedValues.push_back(condBrOp.getTrueResult());

        // FALSE branch (1 time): while loop's exit value
        // DO NOT SINK! This is the result of scf.while when condition becomes false
        conditionArgExitValues.push_back(condBrOp.getFalseResult());
      }
    }

    // Map after region args to gated condition args (N times, controlled by condition)
    for (auto [afterArg, gatedCondArg] :
         llvm::zip(whileOp.getAfterArguments(), conditionArgGatedValues)) {
      mapper.map(afterArg, gatedCondArg);
    }

    // Erase the cloned condition since we don't need it in the final IR
    rewriter.eraseOp(clonedCondition);

    // Step 8: Create after region invariants (N executions)
    // These use gated willContinue (N-length)
    SmallVector<dsa::InvariantOp> afterInvariantOps;
    for (Value invariant : afterInvariants) {
      if (isa<MemRefType>(invariant.getType()))
        continue;

      Value mappedInvariant = mapper.lookupOrDefault(invariant);
      auto invOp = dsa::InvariantOp::create(rewriter,
          loc, invariant.getType(), gatedWillContinue, mappedInvariant);
      mapper.map(invariant, invOp.getResult());
      afterInvariantOps.push_back(invOp);
    }

    // Step 9: Clone after region operations (INCLUDING the yield!)
    SmallVector<Operation *> clonedAfterOps;
    for (Operation &op : afterBlock->without_terminator()) {
      Operation *clonedOp = rewriter.clone(op, mapper);
      clonedAfterOps.push_back(clonedOp);
    }

    // CRITICAL: Also clone the yield operation!
    // The yield's operands will be automatically updated by MLIR when we convert nested SCF ops
    auto originalYield = cast<scf::YieldOp>(afterBlock->getTerminator());
    auto clonedYield = cast<scf::YieldOp>(rewriter.clone(*originalYield, mapper));

    if (failed(convertNestedSCFOps(clonedAfterOps, rewriter, nestedValueMap)))
      return failure();

    // CRITICAL: Get yield operands from the CLONED yield and use nestedValueMap + MAPPER!
    // After convertNestedSCFOps, the cloned yield's operands may reference operations inside
    // nested SCF regions that were converted. We must first check nestedValueMap (for values
    // from nested SCF conversions), then fall back to mapper (for values from current level cloning).
    SmallVector<Value> mappedYieldValues;
    DSA_DEBUG_STREAM << "[DEBUG ConvertWhileOp] Processing yield with " << clonedYield.getNumOperands() << " operands\n";
    for (unsigned i = 0; i < clonedYield.getNumOperands(); ++i) {
      Value yieldVal = clonedYield.getOperand(i);
      DSA_DEBUG_STREAM << "  clonedYield.operand[" << i << "]: ";
      if (!yieldVal) {
        DSA_DEBUG_STREAM << "NULL\n";
      } else if (auto defOp = yieldVal.getDefiningOp()) {
        DSA_DEBUG_STREAM << defOp->getName() << "\n";
      } else {
        DSA_DEBUG_STREAM << "BlockArgument\n";
      }

      // CRITICAL: First try nestedValueMap (for values from nested SCF conversions)
      // Then fall back to mapper (for values from current level cloning)
      auto it = nestedValueMap.find(yieldVal);
      Value mappedYieldVal = (it != nestedValueMap.end()) ? it->second : mapper.lookupOrDefault(yieldVal);
      DSA_DEBUG_STREAM << "  mapped to: ";
      if (!mappedYieldVal) {
        DSA_DEBUG_STREAM << "NULL\n";
      } else if (auto defOp = mappedYieldVal.getDefiningOp()) {
        DSA_DEBUG_STREAM << defOp->getName() << "\n";
      } else {
        DSA_DEBUG_STREAM << "BlockArgument\n";
      }

      mappedYieldValues.push_back(mappedYieldVal);
    }

    // Erase the cloned yield since we don't need it in the final IR
    rewriter.eraseOp(clonedYield);

    // Step 10: Update loop-carry carries with raw willContinue and feedback
    // Loop-carry carries use raw willContinue (N+1) just like in scf.for
    // CRITICAL: All feedback values now come from after region's yield (for loop-carried values)
    // or from gated condition args (for exit-only values).
    // Since we routed condition args through cond_br in Step 7, the after region's yield
    // already contains properly gated values (N times, only when condition=true).
    //
    // For exit-only values (i >= numInits), there's NO corresponding yield value!
    // Their feedback should come from the gated condition args (conditionArgGatedValues).
    // These are already gated by cond_br.true in Step 7, so they execute N times correctly.
    for (size_t i = 0; i < carryOps.size(); i++) {
      auto carryOp = carryOps[i];
      Value feedback;
      if (i < numInits) {
        // Loop-carried value: feedback from after region's yield
        // The yield values are already properly gated (N times) because after region
        // receives gated condition args from Step 7
        feedback = mappedYieldValues[i];
      } else {
        // Exit-only value: feedback from gated condition args (N times)
        // These were gated by cond_br.true in Step 7
        feedback = conditionArgGatedValues[i];
      }

      // DEBUG: Print information about the CarryOp before updating
      DSA_DEBUG_STREAM << "[DEBUG ConvertWhileOp] Updating CarryOp #" << i << "\n";
      DSA_DEBUG_STREAM << "  Current operands:\n";
      for (unsigned j = 0; j < carryOp.getNumOperands(); ++j) {
        Value operand = carryOp.getOperand(j);
        if (!operand) {
          DSA_DEBUG_STREAM << "    operand[" << j << "]: NULL\n";
        } else if (auto defOp = operand.getDefiningOp()) {
          DSA_DEBUG_STREAM << "    operand[" << j << "]: " << defOp->getName() << "\n";
        } else {
          DSA_DEBUG_STREAM << "    operand[" << j << "]: BlockArgument\n";
        }
      }

      // Verify rawWillContinue
      if (!rawWillContinue) {
        DSA_DEBUG_STREAM << "[ERROR] rawWillContinue is NULL!\n";
        return failure();
      }
      if (!rawWillContinue.getType().isInteger(1)) {
        DSA_DEBUG_STREAM << "[ERROR] rawWillContinue is not i1!\n";
        return failure();
      }

      // Verify feedback
      if (!feedback) {
        DSA_DEBUG_STREAM << "[ERROR] feedback is NULL!\n";
        return failure();
      }

      DSA_DEBUG_STREAM << "  Will update to:\n";
      if (auto defOp = rawWillContinue.getDefiningOp()) {
        DSA_DEBUG_STREAM << "    operand[0] (rawWillContinue): " << defOp->getName() << "\n";
      } else {
        DSA_DEBUG_STREAM << "    operand[0] (rawWillContinue): BlockArgument\n";
      }
      if (auto defOp = feedback.getDefiningOp()) {
        DSA_DEBUG_STREAM << "    operand[2] (feedback): " << defOp->getName() << "\n";
      } else {
        DSA_DEBUG_STREAM << "    operand[2] (feedback): BlockArgument\n";
      }

      // Update willContinue to raw (N+1) and feedback from after region (N times, properly gated)
      carryOp.getOperation()->setOperand(0, rawWillContinue);
      carryOp.getOperation()->setOperand(2, feedback);

      DSA_DEBUG_STREAM << "[DEBUG] CarryOp #" << i << " updated successfully\n";
    }

    // Step 11: Extract final results from condition args' exit values
    // CRITICAL: scf.while results come from condition trailing args when condition=false
    // These are already extracted in conditionArgExitValues (false branch of condition args' cond_br)
    SmallVector<Value> results;

    // The while loop results are directly the conditionArgExitValues
    // These are the FALSE branch outputs from condition args routing (Step 7)
    // They represent the values when condition becomes false (loop exits)
    results = conditionArgExitValues;

    // Clean up placeholder
    if (auto placeholderOp = placeholderCond.getDefiningOp()) {
      if (placeholderOp->use_empty()) {
        rewriter.eraseOp(placeholderOp);
      }
    }

    // If this is a nested conversion, populate the parent's valueMap
    if (parentValueMap) {
      for (unsigned i = 0; i < whileOp->getNumResults(); ++i) {
        (*parentValueMap)[whileOp->getResult(i)] = results[i];
      }
    }

    rewriter.replaceOp(whileOp, results);
    return success();
}

//===----------------------------------------------------------------------===//
// Pattern Population Function
//===----------------------------------------------------------------------===//

void populateLoopConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertForOp, ConvertWhileOp>(patterns.getContext());
}

} // namespace dsa
} // namespace mlir

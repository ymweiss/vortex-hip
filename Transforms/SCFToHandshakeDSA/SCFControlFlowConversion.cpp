//===- SCFControlFlowConversion.cpp - SCF control flow conversion -*- C++ -*-===//
//
// Conversion patterns for SCF control flow operations (IfOp, IndexSwitchOp)
//
//===----------------------------------------------------------------------===//

#include "SCFConversionPatterns.h"

namespace mlir {
namespace dsa {

//===----------------------------------------------------------------------===//
// SCF IfOp Conversion Pattern
//===----------------------------------------------------------------------===//

LogicalResult ConvertIfOp::matchAndRewrite(scf::IfOp ifOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const {
    Location loc = ifOp.getLoc();

    // Check if this operation has a valueMap pointer attribute from nested conversion
    DenseMap<Value, Value> *parentValueMap = nullptr;
    if (auto attr = ifOp->getAttr("__dsa_value_map_ptr")) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        parentValueMap = reinterpret_cast<DenseMap<Value, Value> *>(
            intAttr.getValue().getZExtValue());
        // Remove the temporary attribute
        ifOp->removeAttr("__dsa_value_map_ptr");
      }
    }

    rewriter.setInsertionPoint(ifOp);

    Value condition = adaptor.getCondition();

    Block *thenBlock = &ifOp.getThenRegion().front();
    Block *elseBlock = ifOp.getElseRegion().empty()
                           ? nullptr
                           : &ifOp.getElseRegion().front();

    // NOTE: We'll fork the condition AFTER collecting external values
    // to know how many fork outputs we need

    // NOTE: Memory operations are already marked with complete paths during preprocessing
    // No need to mark them again here

    llvm::SetVector<Value> externalValues;

    // Helper to check if a value is defined outside this if operation
    auto isDefinedOutsideIf = [&](Value operand) -> bool {
      if (auto *defOp = operand.getDefiningOp()) {
        // Walk up parent operations to check if defOp is inside this if
        Operation *parentOp = defOp->getParentOp();
        while (parentOp) {
          if (parentOp == ifOp.getOperation()) {
            return false; // Defined inside
          }
          parentOp = parentOp->getParentOp();
        }
        return true; // Defined outside
      } else if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
        // Block arguments are internal if they're from this if's branches or from nested SCF operations
        Operation *blockParentOp = blockArg.getOwner()->getParentOp();
        while (blockParentOp) {
          if (blockParentOp == ifOp.getOperation()) {
            return false; // Internal block arg
          }
          blockParentOp = blockParentOp->getParentOp();
        }
        return true; // External block arg
      }
      return true;
    };

    // CRITICAL: Use iterative work queue to collect ALL external values used anywhere
    // in the branches, including those deep inside nested SCF structures
    auto collectExternalValuesIterative = [&](Block *startBlock) {
      if (!startBlock) return;

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
          if (isDefinedOutsideIf(operand)) {
            externalValues.insert(operand);
          }
        }

        // Add operations from nested regions to work list
        // IMPORTANT: Skip terminators in nested regions (scf.yield, scf.condition, etc.)
        // Their operands have special semantics (loop-carried values, not external values)
        for (Region &region : op->getRegions()) {
          if (region.empty()) continue;
          for (Block &block : region) {
            for (Operation &nestedOp : block.without_terminator()) {
              workList.push_back(&nestedOp);
            }
            // CRITICAL: For nested scf.if/switch operations, also check their yield operands
            // These represent values passed through the nested SCF and need routing
            Operation *terminator = block.getTerminator();
            if (auto yieldOp = dyn_cast<scf::YieldOp>(terminator)) {
              for (Value yieldVal : yieldOp.getOperands()) {
                if (isDefinedOutsideIf(yieldVal)) {
                  externalValues.insert(yieldVal);
                }
              }
            }
          }
        }
      }
    };

    // Collect external values from both branches
    collectExternalValuesIterative(thenBlock);
    collectExternalValuesIterative(elseBlock);

    // CRITICAL: Also check yield operands for external values
    // For scf.if, yield operands are results, not loop-carried values
    // If they're external, they need to be routed through cond_br
    if (thenBlock) {
      auto thenYield = cast<scf::YieldOp>(thenBlock->getTerminator());
      for (Value yieldVal : thenYield.getOperands()) {
        if (isDefinedOutsideIf(yieldVal)) {
          externalValues.insert(yieldVal);
        }
      }
    }
    if (elseBlock) {
      auto elseYield = cast<scf::YieldOp>(elseBlock->getTerminator());
      for (Value yieldVal : elseYield.getOperands()) {
        if (isDefinedOutsideIf(yieldVal)) {
          externalValues.insert(yieldVal);
        }
      }
    }

    // CRITICAL: Fork condition signal for:
    // - N cond_br operations (one per external value to route to branches)
    // - 1 mux select signal (for deterministic result selection)
    // If no external values, no fork needed (condition used directly for mux)
    Value muxSelectCondition;
    unsigned conditionIdx = 0;
    Operation *conditionForkOp = nullptr;  // Track the fork op for marking

    // Setup: Create mappers and fork condition if needed
    IRMapping thenMapper, elseMapper;
    SmallVector<Value> thenYieldValues;
    SmallVector<Value> elseYieldValues;

    if (!externalValues.empty()) {
      // Fork condition: N outputs for cond_br + 1 for mux
      unsigned numConditionUses = externalValues.size() + 1;
      auto conditionFork = circt::handshake::ForkOp::create(rewriter,
          loc, condition, numConditionUses);
      conditionForkOp = conditionFork.getOperation();
      muxSelectCondition = conditionFork->getResult(externalValues.size());

      // Create cond_br for each external value
      for (Value externalValue : externalValues) {
        if (isa<MemRefType>(externalValue.getType())) {
          thenMapper.map(externalValue, externalValue);
          if (elseBlock) {
            elseMapper.map(externalValue, externalValue);
          }
          continue;
        }

        // CRITICAL: When this if is nested inside a while/for loop, the external value
        // should have already been routed through cond_br/invariant/carry by parent SCF
        // The parent SCF's clone operation should have remapped all operands in this if's regions
        // So externalValue should already be the properly routed value (e.g., invariant output)
        // Use forked condition for this cond_br
        Value forkedCondition = conditionFork->getResult(conditionIdx);
        conditionIdx++;
        auto condBr = circt::handshake::ConditionalBranchOp::create(rewriter,
            loc, externalValue.getType(), externalValue.getType(),
            forkedCondition, externalValue);
        thenMapper.map(externalValue, condBr.getTrueResult());
        if (elseBlock) {
          elseMapper.map(externalValue, condBr.getFalseResult());
        }
      }
    } else {
      // No external values to route, use condition directly for mux
      muxSelectCondition = condition;
    }

    // Clone operations in then block
    SmallVector<Operation *> clonedThenOps;
    for (Operation &op : thenBlock->without_terminator()) {
      Operation *clonedOp = rewriter.clone(op, thenMapper);
      clonedThenOps.push_back(clonedOp);
    }

    // Mark the condition value as controlling "if.then" and "if.else" regions
    // When external values exist, conditionForkOp was created and should be marked
    if (conditionForkOp) {
      // Mark the fork operation that was created for routing external values
      markControlValue(conditionForkOp->getResult(0), "if", "then", ifOp, rewriter);
      markControlValue(conditionForkOp->getResult(0), "if", "else", ifOp, rewriter);
    } else {
      // No fork created (no external values)
      // Mark the condition value directly if it has a defining operation
      markControlValue(condition, "if", "then", ifOp, rewriter);
      markControlValue(condition, "if", "else", ifOp, rewriter);
    }

    // Create a valueMap for nested SCF operations
    DenseMap<Value, Value> nestedValueMap;
    if (failed(convertNestedSCFOps(clonedThenOps, rewriter, nestedValueMap)))
      return failure();

    auto thenYield = cast<scf::YieldOp>(thenBlock->getTerminator());
    for (Value yieldVal : thenYield.getOperands()) {
      // First try nestedValueMap (for values from nested SCF conversions)
      // Then fall back to thenMapper (for values from current level cloning)
      auto it = nestedValueMap.find(yieldVal);
      Value mappedVal = (it != nestedValueMap.end()) ? it->second : thenMapper.lookupOrDefault(yieldVal);
      thenYieldValues.push_back(mappedVal);
    }

    // Clone operations in else block if it exists
    if (elseBlock) {
      SmallVector<Operation *> clonedElseOps;
      for (Operation &op : elseBlock->without_terminator()) {
        Operation *clonedOp = rewriter.clone(op, elseMapper);
        clonedElseOps.push_back(clonedOp);
      }

      // Reuse nestedValueMap for else branch - it will accumulate mappings from both branches
      if (failed(convertNestedSCFOps(clonedElseOps, rewriter, nestedValueMap)))
        return failure();

      auto elseYield = cast<scf::YieldOp>(elseBlock->getTerminator());
      for (Value yieldVal : elseYield.getOperands()) {
        // First try nestedValueMap (for values from nested SCF conversions)
        // Then fall back to elseMapper (for values from current level cloning)
        auto it = nestedValueMap.find(yieldVal);
        Value mappedVal = (it != nestedValueMap.end()) ? it->second : elseMapper.lookupOrDefault(yieldVal);
        elseYieldValues.push_back(mappedVal);
      }
    } else {
      // No else branch - duplicate then values
      elseYieldValues = thenYieldValues;
    }

    SmallVector<Value> results;

    // Use handshake.mux instead of merge for deterministic result selection
    // In loops, next iteration's else branch result may race with previous iteration's then branch result
    // Both values could arrive at merge simultaneously, causing nondeterministic behavior
    // Solution: Use mux with condition to deterministically select the correct branch result
    if (!thenYieldValues.empty()) {
      // Convert i1 condition to index type for mux select signal
      auto indexCast = arith::IndexCastOp::create(rewriter,
          loc, rewriter.getIndexType(), muxSelectCondition);

      for (size_t i = 0; i < thenYieldValues.size(); ++i) {
        Value thenVal = thenYieldValues[i];
        Value elseVal = elseYieldValues[i];

        // handshake.mux: deterministic selection based on condition
        // mux(select=0) -> elseVal, mux(select=1) -> thenVal
        // Operand order: {falseValue, trueValue}
        auto muxOp = circt::handshake::MuxOp::create(rewriter,
            loc, thenVal.getType(), indexCast.getResult(),
            ValueRange{elseVal, thenVal});

        results.push_back(muxOp.getResult());
      }
    }

    // If this is a nested conversion, populate the parent's valueMap
    if (parentValueMap) {
      for (unsigned i = 0; i < ifOp->getNumResults(); ++i) {
        (*parentValueMap)[ifOp->getResult(i)] = results[i];
      }
    }

    rewriter.replaceOp(ifOp, results);

    return success();
}

//===----------------------------------------------------------------------===//
// SCF IndexSwitchOp Conversion Pattern
//===----------------------------------------------------------------------===//

LogicalResult ConvertIndexSwitchOp::matchAndRewrite(scf::IndexSwitchOp switchOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const {
    Location loc = switchOp.getLoc();

    // Check if this operation has a valueMap pointer attribute from nested conversion
    DenseMap<Value, Value> *parentValueMap = nullptr;
    if (auto attr = switchOp->getAttr("__dsa_value_map_ptr")) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        parentValueMap = reinterpret_cast<DenseMap<Value, Value> *>(
            intAttr.getValue().getZExtValue());
        // Remove the temporary attribute
        switchOp->removeAttr("__dsa_value_map_ptr");
      }
    }

    // CRITICAL: Set insertion point BEFORE the scf.index_switch operation
    // This ensures all new operations are created as siblings, not children
    rewriter.setInsertionPoint(switchOp);

    // Get case values and regions
    auto caseValues = switchOp.getCases();
    unsigned numCases = caseValues.size();
    unsigned numRegions = numCases + 1; // cases + default

    // Collect all regions (cases + default)
    SmallVector<Region *> allRegions;
    for (Region &caseRegion : switchOp.getCaseRegions()) {
      allRegions.push_back(&caseRegion);
    }
    allRegions.push_back(&switchOp.getDefaultRegion());

    // Collect all values used in any region that are defined outside
    llvm::SetVector<Value> externalValues;
    for (Region *region : allRegions) {
      if (region->empty())
        continue;
      Block *block = &region->front();
      for (Operation &op : *block) {
        for (Value operand : op.getOperands()) {
          if (operand.getParentBlock() != block) {
            externalValues.insert(operand);
          }
        }
      }
    }

    // Clone all regions and collect yield values
    SmallVector<SmallVector<Value>> allYieldValues(numRegions);
    SmallVector<SmallVector<Operation *>> allClonedOps(numRegions);

    for (unsigned i = 0; i < numRegions; ++i) {
      Region *region = allRegions[i];
      if (region->empty())
        continue;

      Block *block = &region->front();
      IRMapping mapper;


      for (Value externalValue : externalValues) {
        if (isa<MemRefType>(externalValue.getType())) {
          mapper.map(externalValue, externalValue);
        } else {
          mapper.map(externalValue, externalValue);
        }
      }

      for (Operation &op : block->without_terminator()) {
        Operation *clonedOp = rewriter.clone(op, mapper);
        allClonedOps[i].push_back(clonedOp);
      }

      // Create a valueMap for nested SCF operations in this region
      DenseMap<Value, Value> nestedValueMap;
      if (failed(convertNestedSCFOps(allClonedOps[i], rewriter, nestedValueMap)))
        return failure();

      auto yieldOp = dyn_cast<scf::YieldOp>(block->getTerminator());
      if (!yieldOp)
        return switchOp.emitError("region must terminate with scf.yield");

      for (Value yieldVal : yieldOp.getOperands()) {
        // First try nestedValueMap (for values from nested SCF conversions)
        // Then fall back to mapper (for values from current level cloning)
        auto it = nestedValueMap.find(yieldVal);
        Value mappedVal = (it != nestedValueMap.end()) ? it->second : mapper.lookupOrDefault(yieldVal);
        allYieldValues[i].push_back(mappedVal);
      }
    }

    if (switchOp.getNumResults() == 0) {
      rewriter.eraseOp(switchOp);
      return success();
    }

    Value switchArg = adaptor.getArg();

    Value selectIdx = switchArg;

    bool needsRemapping = false;
    for (size_t i = 0; i < caseValues.size(); ++i) {
      if (caseValues[i] != static_cast<int64_t>(i)) {
        needsRemapping = true;
        break;
      }
    }

    if (needsRemapping && !caseValues.empty()) {

      rewriter.setInsertionPoint(switchOp);
      Value defaultIdx = arith::ConstantIndexOp::create(rewriter, loc, numRegions - 1);
      selectIdx = defaultIdx;

      for (int64_t i = caseValues.size() - 1; i >= 0; --i) {
        Value caseValue = arith::ConstantIndexOp::create(rewriter, loc, caseValues[i]);
        Value caseIdx = arith::ConstantIndexOp::create(rewriter, loc, i);
        Value cmp = arith::CmpIOp::create(rewriter,
            loc, arith::CmpIPredicate::eq, switchArg, caseValue);
        selectIdx = arith::SelectOp::create(rewriter, loc, cmp, caseIdx, selectIdx);
      }
    }

    SmallVector<Value> results;
    unsigned numResults = switchOp.getNumResults();

    // MLIR's SSA form naturally supports multiple uses of the same value
    // No need to fork select index for multiple mux operations
    for (unsigned resIdx = 0; resIdx < numResults; ++resIdx) {
      SmallVector<Value> yieldValues;
      for (unsigned regionIdx = 0; regionIdx < numRegions; ++regionIdx) {
        yieldValues.push_back(allYieldValues[regionIdx][resIdx]);
      }

      auto muxOp = circt::handshake::MuxOp::create(rewriter,
          loc, switchOp.getResult(resIdx).getType(),
          selectIdx, yieldValues);

      results.push_back(muxOp.getResult());
    }

    // If this is a nested conversion, populate the parent's valueMap
    if (parentValueMap) {
      for (unsigned i = 0; i < switchOp->getNumResults(); ++i) {
        (*parentValueMap)[switchOp->getResult(i)] = results[i];
      }
    }

    rewriter.replaceOp(switchOp, results);

    return success();
}

//===----------------------------------------------------------------------===//
// Pattern Population Function
//===----------------------------------------------------------------------===//

void populateControlFlowConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertIfOp, ConvertIndexSwitchOp>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Registry Initialization
//===----------------------------------------------------------------------===//

void initializeSCFConversionRegistry(MLIRContext *context) {
  SCFConversionRegistry &registry = getSCFConversionRegistry(context);

  // Register loop conversion patterns
  registry.registerPattern<scf::ForOp, ConvertForOp>(context);
  registry.registerPattern<scf::WhileOp, ConvertWhileOp>(context);

  // Register control flow conversion patterns
  registry.registerPattern<scf::IfOp, ConvertIfOp>(context);
  registry.registerPattern<scf::IndexSwitchOp, ConvertIndexSwitchOp>(context);
}

} // namespace dsa
} // namespace mlir

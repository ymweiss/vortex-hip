//===- PassHelpers.cpp - Helper functions for SCFToHandshakeDSA -*- C++ -*-===//
//
// Pass helper functions for SCFToHandshakeDSA conversion
//
//===----------------------------------------------------------------------===//

#include "Common.h"

namespace mlir {
namespace dsa {

// NOTE: Constant conversion to handshake.constant has been removed.
// We keep arith.constant operations as-is for the following reasons:
//   1. The simulator natively supports arith.constant (no ExecutableOpInterface needed)
//   2. Operations with 0 operands execute immediately in simulation
//   3. This avoids false control dependencies on function arguments
//   4. Constants are semantically always available in dataflow
// If hardware synthesis is needed, a separate pass can convert them later.

//===----------------------------------------------------------------------===//
// Sink Unconsumed Values
//===----------------------------------------------------------------------===//

// Sink unconsumed operation results to satisfy handshake single-use requirement
// According to handshake semantics (handshake.txt:229), each Value must have exactly one use.
// ANY operation (handshake, dsacc, arith, etc.) may produce results that are not consumed by any
// downstream operations. For these unconsumed results, insert handshake.sink to discard them.
// This handles all multi-output operations uniformly: control_merge, fork, cond_br, dsacc.stream, etc.
template <typename RewriterT>
LogicalResult sinkUnconsumedValues(circt::handshake::FuncOp funcOp,
                                    RewriterT &rewriter) {
  // Collect all operations with unconsumed results
  SmallVector<std::pair<Operation *, unsigned>> unconsumedOps;

  funcOp.walk([&](Operation *op) {
    // Skip operations that should never have their outputs sinked:
    // - Function boundaries and terminators
    // - Operations explicitly designed to discard values
    if (isa<circt::handshake::FuncOp,
            circt::handshake::ReturnOp,
            circt::handshake::SinkOp>(op))
      return;

    // Check each result of the operation
    for (auto result : op->getResults()) {
      // Count uses
      unsigned useCount = std::distance(result.use_begin(), result.use_end());
      if (useCount == 0) {
        // Found an unconsumed result - need to sink it
        unsigned resultIdx = result.getResultNumber();
        unconsumedOps.push_back({op, resultIdx});
      }
    }
  });

  // Insert sink for each unconsumed result
  for (auto [op, resultIdx] : unconsumedOps) {
    Value unconsumedResult = op->getResult(resultIdx);

    // Set insertion point after the operation producing the unconsumed result
    rewriter.setInsertionPointAfter(op);

    // Create handshake.sink to consume the unused value
    // handshake.sink discards data that arrives at its input
    circt::handshake::SinkOp::create(
        rewriter, op->getLoc(), unconsumedResult);
  }

  return success();
}

// Explicit template instantiation

template LogicalResult sinkUnconsumedValues<OpBuilder>(
    circt::handshake::FuncOp, OpBuilder &);

//===----------------------------------------------------------------------===//
// Greedy Dead Code Elimination
//===----------------------------------------------------------------------===//

// Remove operations whose outputs are completely unused or only connected to sinks
// Iterates until no more dead operations can be removed (greedy approach)
//
// An operation is dead if ALL of its results are either:
// 1. Have zero uses (completely unused), OR
// 2. Are ONLY used by handshake.sink operations
//
// This is important for cascading cleanup: if we remove arith.xori, the constant
// that feeds it may become dead, and its control input may become dead, etc.
//
// Exceptions (operations that should NOT be removed even if outputs unused):
// - Memory operations (loads/stores/extmemory) - have side effects
// - Operations with zero results (return, sink, dealloc) - structural/side effects
// - Function arguments (BlockArguments) - can't be removed
template <typename RewriterT>
LogicalResult eliminateDeadCode(circt::handshake::FuncOp funcOp,
                                 RewriterT &rewriter) {
  bool madeProgress = true;
  int iterationCount = 0;

  while (madeProgress) {
    madeProgress = false;
    iterationCount++;

    // Collect operations that are dead (all results unused or only sunk)
    SmallVector<Operation *> deadOps;

    funcOp.walk([&](Operation *op) {
      // First check: Skip operations with zero results (by definition)
      // Operations with zero results have side effects and their existence matters.
      // This automatically excludes: memref::DeallocOp, handshake::SinkOp, handshake::ReturnOp
      // without needing to manually list them.
      if (op->getNumResults() == 0)
        return;

      // Second check: Skip specific operations that should never be removed
      // even though they have results (loads/stores/memory have side effects)
      //
      // NOTE: dsacc.carry and dsacc.invariant are NOT excluded from DCE.
      // If ALL outputs of a carry/invariant are unused or only sunk, they are dead code
      // and should be removed along with their upstream producers (cascading cleanup).
      //
      // NOTE: handshake.source is NOT excluded from DCE (removed from list).
      // We no longer use source operations in our dataflow model - execution is
      // triggered by availability of function arguments, not by continuous sources.
      //
      // NOTE: memref.get_global is excluded from DCE because it loads module-level
      // constants that are needed by handshake.memory operations (via index/address
      // calculations). These are preserved even if they appear unused after conversion.
      if (isa<circt::handshake::LoadOp,
              circt::handshake::StoreOp,
              circt::handshake::ExternalMemoryOp,
              circt::handshake::MemoryOp,
              memref::GetGlobalOp>(op)) {
        return;
      }

      // Check if ALL results are either unused or only used by sinks
      bool allResultsDead = true;
      for (Value result : op->getResults()) {
        bool thisResultDead = true;

        // Check all uses of this result
        for (OpOperand &use : result.getUses()) {
          Operation *user = use.getOwner();

          // If used by anything OTHER than sink, this result is alive
          if (!isa<circt::handshake::SinkOp>(user)) {
            thisResultDead = false;
            break;
          }
        }

        // If ANY result is alive, the whole operation is alive
        if (!thisResultDead) {
          allResultsDead = false;
          break;
        }
      }

      // If all results are dead, mark this operation for removal
      if (allResultsDead) {
        deadOps.push_back(op);
      }
    });

    // Remove dead operations
    for (Operation *op : deadOps) {
      // First, remove all sink operations that consume this op's results
      for (Value result : op->getResults()) {
        for (OpOperand &use : llvm::make_early_inc_range(result.getUses())) {
          if (auto sinkOp = dyn_cast<circt::handshake::SinkOp>(use.getOwner())) {
            sinkOp->erase();
          }
        }
      }

      // Then remove the operation itself
      op->erase();
      madeProgress = true;
    }
  }

  return success();
}

// Explicit template instantiation
template LogicalResult eliminateDeadCode<OpBuilder>(
    circt::handshake::FuncOp, OpBuilder &);

//===----------------------------------------------------------------------===//
// Fork Operation Optimization
//===----------------------------------------------------------------------===//

// Optimize fork operations by:
// 1. Removing fork outputs that only connect to sinks and reducing fanout
// 2. Replacing 1-input-1-output forks with direct pass-through connections
//
// This optimization reduces unnecessary fork operations and prepares the IR
// for more effective dead code elimination. It runs iteratively until no more
// optimizations can be made.
//
// Example transformations:
//   Before: %0:3 = fork [3] %in    // where %0#2 -> sink
//           sink %0#2
//   After:  %0:2 = fork [2] %in    // fanout reduced from 3 to 2
//
//   Before: %0 = fork [1] %in      // 1-to-1 fork is redundant
//   After:  (direct use of %in)    // replaced with pass-through
template <typename RewriterT>
LogicalResult optimizeForkOperations(circt::handshake::FuncOp funcOp,
                                      RewriterT &rewriter) {
  bool madeProgress = true;

  while (madeProgress) {
    madeProgress = false;

    SmallVector<circt::handshake::ForkOp> forksToOptimize;

    // Collect all fork operations
    funcOp.walk([&](circt::handshake::ForkOp forkOp) {
      forksToOptimize.push_back(forkOp);
    });

    for (auto forkOp : forksToOptimize) {
      // Analyze which outputs are live (used by non-sink operations)
      // and which only connect to sinks
      SmallVector<Value> liveOutputs;
      SmallVector<unsigned> sinkOnlyOutputs;

      for (auto result : forkOp.getResults()) {
        bool onlyUsedBySinks = false;
        bool hasUses = false;

        // Check all uses of this result
        for (auto user : result.getUsers()) {
          hasUses = true;
          if (!isa<circt::handshake::SinkOp>(user)) {
            // This output is used by a non-sink operation
            liveOutputs.push_back(result);
            onlyUsedBySinks = false;
            break;
          }
          onlyUsedBySinks = true;
        }

        // If this output only connects to sinks, mark it for removal
        if (hasUses && onlyUsedBySinks) {
          sinkOnlyOutputs.push_back(result.getResultNumber());
        }
        // If this output has NO uses at all (!hasUses), don't add it to liveOutputs
        // This allows the fork fanout to be reduced when the new fork is created
      }

      // Case 1: All outputs go to sinks (or are unused)
      // Skip this fork - DCE will handle removing it entirely
      if (liveOutputs.empty())
        continue;

      // Case 2: Some outputs go to sinks OR are completely unused - reduce fork fanout
      // Check if fanout needs to be reduced (either sink-only outputs or unused outputs)
      if (!sinkOnlyOutputs.empty() || liveOutputs.size() < forkOp.getNumResults()) {
        // First, remove all sink operations consuming fork-to-sink paths
        for (unsigned idx : sinkOnlyOutputs) {
          Value result = forkOp.getResult()[idx];
          for (auto user : llvm::make_early_inc_range(result.getUsers())) {
            if (auto sinkOp = dyn_cast<circt::handshake::SinkOp>(user)) {
              sinkOp->erase();
            }
          }
        }

        // Create new fork with reduced fanout
        unsigned newFanout = liveOutputs.size();
        Value input = forkOp.getOperand();

        if (newFanout == 1) {
          // Special case: reduced to 1-to-1 fork, replace with direct connection
          liveOutputs[0].replaceAllUsesWith(input);
        } else {
          // Create new fork with reduced fanout
          rewriter.setInsertionPoint(forkOp);
          auto newFork = circt::handshake::ForkOp::create(
              rewriter, forkOp.getLoc(), input, newFanout);

          // Replace uses of live outputs with new fork outputs
          for (unsigned i = 0; i < newFanout; ++i) {
            liveOutputs[i].replaceAllUsesWith(newFork.getResult()[i]);
          }
        }

        forkOp->erase();
        madeProgress = true;
        continue;
      }

      // Case 3: Check if this is a 1-to-1 fork (no sinks, just one output)
      // Replace with direct pass-through connection
      if (forkOp.getNumResults() == 1) {
        Value input = forkOp.getOperand();
        Value output = forkOp.getResult()[0];

        output.replaceAllUsesWith(input);
        forkOp->erase();
        madeProgress = true;
      }
    }
  }

  return success();
}

// Explicit template instantiation
template LogicalResult optimizeForkOperations<OpBuilder>(
    circt::handshake::FuncOp, OpBuilder &);

//===----------------------------------------------------------------------===//
// Entry Token Coordination for Function Arguments
//===----------------------------------------------------------------------===//

// Coordinates function arguments with the entry token before they can be used.
// In Handshake semantics, function data arguments need to be synchronized with
// the entry control token before they can participate in dataflow execution.
//
// This prevents deadlocks where forks on raw arguments never execute because
// arguments aren't activated by the entry token.
//
// Example transformation:
//   Before:
//   handshake.func @foo(%arg0: i32, %arg1: i32, %argCtrl: none) {
//     %0 = arith.addi %arg0, %arg1
//
//   After:
//   handshake.func @foo(%arg0: i32, %arg1: i32, %argCtrl: none) {
//     %0:2 = fork [2] %argCtrl : none  // Fork entry token for each arg
//     %arg0_ready = join %arg0, %0#0 : i32, none  // Coordinate with entry
//     %arg1_ready = join %arg1, %0#1 : i32, none
//     %1 = arith.addi %arg0_ready, %arg1_ready
template <typename RewriterT>
LogicalResult coordinateArgsWithEntryToken(circt::handshake::FuncOp funcOp,
                                             RewriterT &rewriter) {
  Block &entryBlock = funcOp.getBody().front();
  BlockArgument entryToken = nullptr;

  // Find the entry token: the last argument with type 'none'
  for (BlockArgument arg : entryBlock.getArguments()) {
    if (isa<NoneType>(arg.getType())) {
      entryToken = arg;
    }
  }

  // If no entry token found, nothing to coordinate
  if (!entryToken)
    return success();

  // Check which arguments have uses and need coordination
  SmallVector<BlockArgument> argsNeedingCoordination;

  for (BlockArgument arg : entryBlock.getArguments()) {
    // Skip the entry token itself
    if (arg == entryToken)
      continue;

    // Skip memref arguments - they're handled by memory operations
    if (isa<MemRefType>(arg.getType()))
      continue;

    // Check if this argument has any uses
    if (!arg.use_empty()) {
      argsNeedingCoordination.push_back(arg);
    }
  }

  // If no arguments need coordination, we're done
  if (argsNeedingCoordination.empty())
    return success();

  // Fork the entry token once for all arguments that need coordination
  rewriter.setInsertionPointToStart(&entryBlock);
  unsigned numArgs = argsNeedingCoordination.size();
  auto entryFork = circt::handshake::ForkOp::create(
      rewriter, entryToken.getLoc(), entryToken, numArgs);

  // Coordinate each argument with its forked entry token
  for (unsigned i = 0; i < numArgs; ++i) {
    BlockArgument arg = argsNeedingCoordination[i];
    Value entryCtrl = entryFork.getResult()[i];

    // CRITICAL: Collect uses BEFORE creating the join operation
    // Otherwise we'll include the join's own input use and create a cycle
    SmallVector<OpOperand *> uses;
    for (OpOperand &use : arg.getUses()) {
      uses.push_back(&use);
    }

    // Create join to coordinate argument with entry control
    rewriter.setInsertionPointToStart(&entryBlock);
    auto joinOp = circt::handshake::JoinOp::create(
        rewriter, arg.getLoc(), ValueRange{arg, entryCtrl});
    Value coordinated = joinOp.getResult();

    // Replace all uses of the argument with the coordinated value
    // (excluding the join's own input, which we didn't collect above)
    for (OpOperand *use : uses) {
      use->set(coordinated);
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Fork Insertion for Multiple Uses
//===----------------------------------------------------------------------===//

// Insert handshake.fork operations for values with multiple uses
// According to Handshake + DSA dialect semantics, if one variable needs to be
// used by multiple places, it should be explicitly "fanned out" via handshake.fork.
//
// This pass transforms:
//   %a = <some ops>
//   ... = <some ops> %a
//   ... = <some ops> %a
//
// Into:
//   %a = <some ops>
//   %a_0, %a_1 = handshake.fork [2] %a
//   ... = <some ops> %a_0
//   ... = <some ops> %a_1
//
// Implementation strategy:
// 1. Walk through all operations and find values with multiple uses
// 2. Create a fork operation with the appropriate number of outputs
// 3. Replace all uses with the forked outputs
// 4. Skip certain operations that should not have forks inserted:
//    - Block arguments (function arguments)
//    - Operations that are already forks
//    - Values with zero or one use (no fork needed)
template <typename RewriterT>
LogicalResult insertForkOperations(circt::handshake::FuncOp funcOp,
                                     RewriterT &rewriter) {
  // Collect all values that need fork operations
  // Map: Value -> number of uses
  DenseMap<Value, SmallVector<OpOperand *>> valuesToFork;

  funcOp.walk([&](Operation *op) {
    // Check each result of the operation
    for (Value result : op->getResults()) {
      // Collect all uses of this result
      SmallVector<OpOperand *> uses;
      for (OpOperand &use : result.getUses()) {
        uses.push_back(&use);
      }

      // If this value has more than one use, it needs a fork
      if (uses.size() > 1) {
        valuesToFork[result] = uses;
      }
    }
  });

  // Also check block arguments (function arguments and block parameters)
  for (Block &block : funcOp.getBody()) {
    for (Value arg : block.getArguments()) {
      SmallVector<OpOperand *> uses;
      for (OpOperand &use : arg.getUses()) {
        uses.push_back(&use);
      }

      if (uses.size() > 1) {
        valuesToFork[arg] = uses;
      }
    }
  }

  // Insert fork operations for each value that needs it
  for (auto &[value, uses] : valuesToFork) {
    unsigned numUses = uses.size();

    // Determine insertion point
    // For operation results, insert after the defining operation
    // For block arguments, insert at the start of the block
    if (auto defOp = value.getDefiningOp()) {
      rewriter.setInsertionPointAfter(defOp);
    } else if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      rewriter.setInsertionPointToStart(blockArg.getOwner());
    } else {
      continue; // Skip if we can't determine insertion point
    }

    // Create fork operation with numUses outputs
    auto forkOp = circt::handshake::ForkOp::create(
        rewriter, value.getLoc(), value, numUses);

    // Replace each use with the corresponding fork output
    for (unsigned i = 0; i < numUses; ++i) {
      uses[i]->set(forkOp.getResult()[i]);
    }
  }

  return success();
}

// Explicit template instantiation
template LogicalResult coordinateArgsWithEntryToken<OpBuilder>(
    circt::handshake::FuncOp, OpBuilder &);

template LogicalResult insertForkOperations<OpBuilder>(
    circt::handshake::FuncOp, OpBuilder &);

//===----------------------------------------------------------------------===//
// Heap Allocation/Deallocation Analysis Helpers
//===----------------------------------------------------------------------===//

// Helper function to trace back through dsacc.invariant and fork to find original memref
// This is an extended version of getOriginalMemRef that also handles fork operations
Value traceMemRefOrigin(Value memref) {
  while (auto defOp = memref.getDefiningOp()) {
    if (auto invariantOp = dyn_cast<dsa::InvariantOp>(defOp)) {
      memref = invariantOp.getA();
    } else if (auto forkOp = dyn_cast<circt::handshake::ForkOp>(defOp)) {
      memref = forkOp.getOperand();
    } else {
      break;
    }
  }
  return memref;
}

// Helper function to find loop exit control signal for a given context operation
// Returns the nearest enclosing loop's exit control signal (dsacc.stream's last indicator)
//
// CRITICAL FIX (Issue 5A): For nested loops, we must find the NEAREST enclosing loop,
// not the first/outermost loop. This ensures deallocations in nested loops wait for
// the correct loop's memory operations to complete.
//
// Example:
//   for (i = 0; i < N; i++) {        // Outer loop
//     int *ptr = new int[100];
//     for (j = 0; j < 100; j++) {    // Inner loop - THIS is the nearest
//       ptr[j] = i + j;
//     }
//     delete[] ptr;  // Must wait for INNER loop's memory ops, not outer!
//   }
Value findLoopExitControl(circt::handshake::FuncOp funcOp, Operation *contextOp) {
  if (!contextOp)
    return nullptr;

  // Walk up the parent chain to find the nearest enclosing dsa.stream operation
  // dsa.stream represents converted scf.for loops in the handshake IR
  Operation *current = contextOp;

  while (current && current != funcOp.getOperation()) {
    // Check if this operation's parent region contains a dsa.stream
    // We need to search in the same region as the current operation
    Block *parentBlock = current->getBlock();

    if (parentBlock) {
      // Search backward from current operation to find the most recent stream
      // in the same block (this would be the loop initialization)
      for (Operation &op : llvm::reverse(*parentBlock)) {
        if (auto streamOp = dyn_cast<dsa::StreamOp>(&op)) {
          // Found the nearest stream - return its last indicator (result 1)
          return streamOp.getResult(1);
        }

        // Stop if we've gone past the context operation
        if (&op == current)
          break;
      }
    }

    // Move up to parent operation
    current = current->getParentOp();
  }

  // No enclosing loop found - search for any loop in the function
  // This handles the case where dealloc is at function scope
  Value lastIndicator = nullptr;
  funcOp.walk([&](dsa::StreamOp streamOp) {
    if (!lastIndicator) {
      lastIndicator = streamOp.getResult(1);
    }
    return WalkResult::advance();
  });

  return lastIndicator;
}

// Legacy version: Find any loop exit control (for backward compatibility)
// Returns the first dsacc.stream's last indicator found in the function
Value findLoopExitControl(circt::handshake::FuncOp funcOp) {
  // Use the context-aware version with null context to get first stream
  return findLoopExitControl(funcOp, nullptr);
}

// Find loop iteration control signal for a given memory operation context
// This is CRITICAL for memory operations in loops to avoid single-use control token deadlocks.
//
// PROBLEM: Memory operations need a control token to fire. In loops, using a single-use
// function entry token causes deadlock after the first iteration because the token is consumed.
//
// SOLUTION: Use per-iteration control signals that fire on EVERY loop iteration:
// - For scf.for loops (converted to dsa.stream): use last signal directly
// - For scf.while loops: use isLast = !condition signal
//
// Returns the is_last signal:
// - For scf.for loops: dsa.stream result(1) - the last indicator
// - For scf.while loops: arith.xori result (isLast = !condition)
// - nullptr if not in a loop
//
// Control semantics: is_last=false (continue), is_last=true (exit)
Value findLoopIterationControl(circt::handshake::FuncOp funcOp, Operation *contextOp) {
  if (!contextOp)
    return nullptr;

  // Strategy: Search the function for loop control signals (is_last)
  // CRITICAL: All DSA primitives (carry, invariant) use is_last signal
  // Control semantics: is_last=false (continue), is_last=true (exit)
  //
  // 1. For scf.for (dsa.stream): use %last directly from dsa.stream result(1)
  // 2. For scf.while: use isLast = !condition (arith.xori)

  Block *block = contextOp->getBlock();
  if (!block)
    return nullptr;

  // First, check if we're in a scf.for loop (has dsa.stream + dsa.gate)
  // CRITICAL: Must use dsa.gate's output (N), NOT dsa.stream's raw output (N+1)
  // Per DSAOps.td lines 119-165: stream produces N+1 outputs
  // Per DSAOps.td lines 193-312: gate transforms N+1 to N outputs for loop body
  // dsa.carry and dsa.invariant MUST use the N-version (gated signal)
  Value willContinue = nullptr;
  block->walk([&](dsa::GateOp gateOp) {
    // Use the gated willContinue signal (result 1) - this is the N-version
    // that should be used by carry/invariant operations
    willContinue = gateOp.getCond();
    return WalkResult::interrupt();
  });

  if (willContinue)
    return willContinue;

  // Second, check if we're in a scf.while loop
  // Look for willContinue signal that's used by dsa.carry operations
  // willContinue can be:
  // - arith.xori result (isLast = !condition)
  // - arith.cmpi result (for simple while loops)
  // - Or any i1 value used as ctrl by dsa.carry/invariant
  block->walk([&](dsa::CarryOp carryOp) {
    // Get the ctrl operand (operand 0) - this is the willContinue signal
    Value ctrl = carryOp.getOperand(0);

    // Trace back through forks to find the original signal
    Value current = ctrl;
    while (current) {
      if (auto defOp = current.getDefiningOp()) {
        if (isa<arith::XOrIOp, arith::CmpIOp>(defOp)) {
          // Found the willContinue signal! Use the forked version (ctrl)
          willContinue = ctrl;
          return WalkResult::interrupt();
        } else if (auto forkOp = dyn_cast<circt::handshake::ForkOp>(defOp)) {
          // Trace back through fork
          current = forkOp.getOperand();
          continue;
        }
      }
      break;
    }
    return WalkResult::advance();
  });

  return willContinue;
}

} // namespace dsa
} // namespace mlir

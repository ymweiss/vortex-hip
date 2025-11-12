//===- HandshakeOptimizations.cpp - Handshake+DSA Optimizations -*- C++ -*-===//
//
// Greedy optimization patterns for Handshake+DSA dialect.
// Includes:
// 1. Collapse fork chains into single fork
// 2. Simplify index cast chains with same start/end types
//
// NOTE: Single-input joins are NOT optimized away because they serve an
// important purpose: converting any type to 'none' type (control token).
// This is essential for memory operations and control flow.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

namespace mlir {
namespace dsa {

//===----------------------------------------------------------------------===//
// Pattern 1: Simplify Index Cast Chains
//===----------------------------------------------------------------------===//

// Eliminates or shortens chains of index_cast operations where the final
// output type matches an intermediate or source type.
//
// Before: %0 = arith.index_cast %in : index to i32
//         %1 = arith.index_cast %0 : i32 to i64
//         %2 = arith.index_cast %1 : i64 to index
// After:  (use %in directly)
//
// This optimization:
// - Removes redundant type conversions
// - Simplifies dataflow graph
// - May improve performance by eliminating unnecessary operations
struct SimplifyIndexCastChain : public OpRewritePattern<arith::IndexCastOp> {
  using OpRewritePattern<arith::IndexCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::IndexCastOp castOp,
                                 PatternRewriter &rewriter) const override {
    // Trace back through the chain to find the original source
    Value current = castOp.getIn();
    Value source = current;

    // Follow the chain of index_cast operations backwards
    while (auto prevCast = source.getDefiningOp<arith::IndexCastOp>()) {
      source = prevCast.getIn();
    }

    // Check if we traced back at all (i.e., there's a chain)
    if (source == current) {
      // No chain, but check for degenerate cast (input type == output type)
      if (current.getType() == castOp.getType()) {
        rewriter.replaceOp(castOp, current);
        return success();
      }
      return failure();
    }

    // We have a chain - check if source type matches final output type
    if (source.getType() == castOp.getType()) {
      // Entire chain can be eliminated!
      rewriter.replaceOp(castOp, source);
      return success();
    }

    // Chain exists but types don't match at endpoints
    // Check if we can still shorten the chain by creating a single cast
    // from source to output (if it's shorter than the current chain)
    if (source != current) {
      // Count the chain length to decide if it's worth optimizing
      int chainLength = 0;
      Value temp = current;
      while (auto prevCast = temp.getDefiningOp<arith::IndexCastOp>()) {
        chainLength++;
        temp = prevCast.getIn();
        if (temp == source) break;
      }

      // If chain has 2+ casts, replace with single cast from source
      if (chainLength >= 2) {
        rewriter.setInsertionPoint(castOp);
        Value newCast = arith::IndexCastOp::create(rewriter, 
            castOp.getLoc(), castOp.getType(), source);
        rewriter.replaceOp(castOp, newCast);
        return success();
      }
    }

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pattern 2: Collapse Fork Chains
//===----------------------------------------------------------------------===//

// Collapses chains of fork operations into a single fork with the total fanout.
// This only applies when ALL outputs of a fork are consumed ONLY by other forks.
//
// Before: %0:2 = fork [2] %input
//         %1:3 = fork [3] %0#0
//         %2:2 = fork [2] %0#1
//         (consumers use %1#0, %1#1, %1#2, %2#0, %2#1)
//
// After:  %new:5 = fork [5] %input
//         (consumers use %new#0, %new#1, %new#2, %new#3, %new#4)
//
// This optimization:
// - Reduces the number of fork operations
// - Simplifies the dataflow graph
// - May improve hardware synthesis (single fanout buffer vs. nested buffers)
//
// IMPORTANT: We only optimize when ALL uses of ALL fork outputs are other forks.
// Mixed usage (some outputs to forks, some to other ops) is NOT optimized to
// preserve the original dataflow structure.
struct CollapseForkChains : public OpRewritePattern<circt::handshake::ForkOp> {
  using OpRewritePattern<circt::handshake::ForkOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(circt::handshake::ForkOp forkOp,
                                 PatternRewriter &rewriter) const override {
    // Check if ALL outputs are consumed ONLY by forks
    // Collect downstream forks and leaf consumers
    SmallVector<circt::handshake::ForkOp> downstreamForks;
    SmallVector<std::pair<Value, unsigned>> leafConsumers; // (new_fork_result, output_index)

    bool allOutputsAreForksOnly = true;

    for (OpResult result : forkOp.getResult()) {
      bool thisOutputHasOnlyForks = true;

      // Check all uses of this output
      for (OpOperand &use : result.getUses()) {
        if (auto downstreamFork = dyn_cast<circt::handshake::ForkOp>(use.getOwner())) {
          // This use is a fork - collect it
          if (std::find(downstreamForks.begin(), downstreamForks.end(), downstreamFork)
              == downstreamForks.end()) {
            downstreamForks.push_back(downstreamFork);
          }
        } else {
          // This use is NOT a fork - can't optimize
          thisOutputHasOnlyForks = false;
          break;
        }
      }

      if (!thisOutputHasOnlyForks) {
        allOutputsAreForksOnly = false;
        break;
      }
    }

    // If not all outputs are fork-only, we can't optimize
    if (!allOutputsAreForksOnly || downstreamForks.empty()) {
      return failure();
    }

    // Calculate total fanout needed
    unsigned totalFanout = 0;

    // Collect all leaf consumers (non-fork uses from downstream forks)
    for (auto downstreamFork : downstreamForks) {
      for (OpResult downstreamResult : downstreamFork.getResult()) {
        // Check if this downstream fork result is used by another fork
        bool usedByFork = false;
        for (OpOperand &use : downstreamResult.getUses()) {
          if (isa<circt::handshake::ForkOp>(use.getOwner())) {
            usedByFork = true;
            break;
          }
        }

        // If used by fork, this will be handled recursively
        // For now, just count immediate fanout
        if (!usedByFork) {
          totalFanout += std::distance(downstreamResult.use_begin(),
                                        downstreamResult.use_end());
        }
      }
    }

    // If total fanout is not greater than original, no benefit
    if (totalFanout <= forkOp.getResult().size()) {
      return failure();
    }

    // Create new fork with total fanout
    rewriter.setInsertionPoint(forkOp);
    auto newFork = circt::handshake::ForkOp::create(rewriter, 
        forkOp.getLoc(), forkOp.getOperand(), totalFanout);

    // Remap all leaf consumers to new fork outputs
    unsigned outputIndex = 0;
    for (auto downstreamFork : downstreamForks) {
      for (OpResult downstreamResult : downstreamFork.getResult()) {
        // Replace all uses of this downstream result with new fork outputs
        SmallVector<OpOperand*> usesToUpdate;
        for (OpOperand &use : downstreamResult.getUses()) {
          usesToUpdate.push_back(&use);
        }

        for (OpOperand *use : usesToUpdate) {
          if (outputIndex < totalFanout) {
            use->set(newFork.getResult()[outputIndex++]);
          }
        }
      }
    }

    // Erase downstream forks (they're now unused)
    for (auto downstreamFork : downstreamForks) {
      rewriter.eraseOp(downstreamFork);
    }

    // Erase original fork
    rewriter.eraseOp(forkOp);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Main Optimization Driver
//===----------------------------------------------------------------------===//

// Applies greedy optimization patterns to a handshake function.
// Runs patterns iteratively until a fixed point is reached.
//
// Pattern application order:
// 1. SimplifyIndexCastChain - removes redundant type conversions
// 2. CollapseForkChains - combines nested forks
//
// NOTE: Fork-to-sink optimization and single-output fork removal are handled
// by the separate optimizeForkOperations() pass in PassHelpers.cpp, which runs
// AFTER sinkUnconsumedValues() to ensure all unused outputs have explicit sinks.
//
// The greedy rewriter applies patterns repeatedly until no more changes occur
// or the maximum iteration limit is reached.
template <typename RewriterT>
LogicalResult optimizeHandshakeDSA(circt::handshake::FuncOp funcOp,
                                    RewriterT &rewriter) {
  MLIRContext *context = funcOp.getContext();

  // Populate optimization patterns
  RewritePatternSet patterns(context);
  patterns.add<SimplifyIndexCastChain>(context);
  patterns.add<CollapseForkChains>(context);

  // Apply patterns greedily until fixed point
  // The greedy rewriter will iterate until no more changes occur (up to default max iterations)
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    return failure();
  }

  return success();
}

// Explicit template instantiation for OpBuilder
template LogicalResult optimizeHandshakeDSA<OpBuilder>(
    circt::handshake::FuncOp, OpBuilder &);

} // namespace dsa
} // namespace mlir

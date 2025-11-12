//===- DeallocProcessing.cpp - Deallocation processing ---------*- C++ -*-===//
//
// Implementation of deallocation processing and insertion
//
//===----------------------------------------------------------------------===//

#include "DeallocProcessing.h"

namespace mlir {
namespace dsa {

//===----------------------------------------------------------------------===//
// Process Deallocations in Function
//===----------------------------------------------------------------------===//

LogicalResult processDeallocsInFunction(
    circt::handshake::FuncOp funcOp,
    OpBuilder &builder,
    AllocDeallocTracker &tracker) {

  // Collect deallocations to process
  SmallVector<std::pair<Value, memref::DeallocOp>> deallocsToProcess;

  funcOp.walk([&](memref::DeallocOp deallocOp) {
    Value memref = deallocOp.getMemref();

    // Trace back to find the original alloc
    Value originalMemref = traceMemRefOrigin(memref);

    // Check if this is a heap allocation we tracked
    if (tracker.allocToDeallocMap.count(originalMemref)) {
      deallocsToProcess.push_back({originalMemref, deallocOp});
    }
  });

  // Process each dealloc
  for (auto &pair : deallocsToProcess) {
    Value originalMemref = pair.first;
    memref::DeallocOp deallocOp = pair.second;
    Value memref = deallocOp.getMemref();
    Location loc = deallocOp.getLoc();

    // Check if this memref has memory accesses
    bool hasAccess = tracker.memrefHasAccess.lookup(originalMemref);

    builder.setInsertionPoint(deallocOp);

    // CRITICAL FIX (Issue 5A): Find the NEAREST enclosing loop's exit control
    // This ensures nested loops wait for the correct loop's memory operations
    Value loopExitControl = findLoopExitControl(funcOp, deallocOp.getOperation());

    Value i1Control;

    if (!hasAccess) {
      // Case 1: No memory accesses - use loop exit control if available
      if (loopExitControl) {
        i1Control = loopExitControl;
      } else {
        // No loop and no memory accesses - no control dependency needed
        // In pure dataflow, dealloc naturally executes after all memref uses
        // Skip adding control dependency
        continue;
      }

    } else {
      // Case 2: Has memory accesses - wait for completion

      // Find the memory operation (either internal or external) for this memref
      Operation *memOp = nullptr;
      int ldCount = 0;

      // Try external memory first (function arguments, dynamic allocations)
      funcOp.walk([&](circt::handshake::ExternalMemoryOp op) {
        if (traceMemRefOrigin(op.getMemref()) == originalMemref) {
          memOp = op.getOperation();
          ldCount = op.getLdCount();
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });

      // If not found, try internal memory (static allocations)
      // MemoryOp doesn't store the memref value, only its type as an attribute
      // So we match by comparing the memref type
      if (!memOp) {
        auto originalMemrefType = originalMemref.getType();
        funcOp.walk([&](circt::handshake::MemoryOp op) {
          if (op.getMemRefType() == originalMemrefType) {
            memOp = op.getOperation();
            ldCount = op.getLdCount();
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
      }

      if (!memOp) {
        return deallocOp.emitError("could not find memory operation for memref");
      }

      // Collect completion tokens (none-type results after load data outputs)
      // Works for both ExternalMemoryOp and MemoryOp (same result layout)
      SmallVector<Value> completionTokens;
      for (int i = ldCount; i < memOp->getNumResults(); ++i) {
        completionTokens.push_back(memOp->getResult(i));
      }

      if (completionTokens.empty()) {
        return deallocOp.emitError("memory operation has no completion tokens");
      }

      // Join all completion tokens
      Value completionToken;
      if (completionTokens.size() == 1) {
        completionToken = completionTokens[0];
      } else {
        auto joinOp = circt::handshake::JoinOp::create(builder,
            loc, completionTokens);
        completionToken = joinOp.getResult();
      }

      // Sync with loop exit control if available
      if (loopExitControl) {
        // Sync i1 control signal with none completion token
        auto syncOp = circt::handshake::SyncOp::create(builder,
            loc, ValueRange{loopExitControl, completionToken});
        i1Control = syncOp.getResults()[0];  // First result is i1
      } else {
        // No loop control - use completion token to trigger constant i1 true
        // The constant fires when memory operations complete
        Value trueConstant = circt::handshake::ConstantOp::create(builder,
            loc, builder.getI1Type(),
            builder.getIntegerAttr(builder.getI1Type(), 1),
            completionToken);
        i1Control = trueConstant;
      }
    }

    // Create dsacc.invariant to add control dependency to the memref
    // The invariant ensures dealloc waits for i1Control to be available
    auto invariantOp = dsa::InvariantOp::create(builder,
        loc, memref.getType(), i1Control, memref);

    // Replace the memref operand of deallocOp with the controlled version
    // This makes the dealloc wait for the control signal before executing
    deallocOp->setOperand(0, invariantOp.getO());

    // Keep the deallocOp in place (don't erase it)
    // Both memref.alloc and memref.dealloc should remain in handshake IR
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Insert Stack Deallocations
//===----------------------------------------------------------------------===//

LogicalResult insertStackDeallocations(
    circt::handshake::FuncOp funcOp,
    OpBuilder &builder,
    AllocDeallocTracker &tracker) {

  if (tracker.stackAllocTypesNeedingDealloc.empty()) {
    return success();
  }

  // Find the handshake.return operation (insertion point for deallocations)
  circt::handshake::ReturnOp returnOp = nullptr;
  funcOp.walk([&](circt::handshake::ReturnOp op) {
    returnOp = op;
    return WalkResult::interrupt();
  });

  if (!returnOp) {
    return funcOp.emitError("function has no handshake.return operation");
  }

  // Process each stack allocation type
  for (auto allocType : tracker.stackAllocTypesNeedingDealloc) {
    // Find the allocation operation in the handshake function
    memref::AllocOp allocOp = nullptr;
    funcOp.walk([&](memref::AllocOp op) {
      if (op.getResult().getType() == allocType) {
        allocOp = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (!allocOp) {
      // Allocation might have been optimized away, skip it
      continue;
    }

    Value memref = allocOp.getResult();
    Location loc = allocOp.getLoc();
    Value originalMemref = traceMemRefOrigin(memref);

    // Check if this memref has any memory accesses
    bool hasAccess = tracker.memrefHasAccess.lookup(memref);

    // Find the memory operation (handshake.memory or handshake.extmemory)
    Operation *memOp = nullptr;
    int ldCount = 0;

    // First, try to find ExternalMemoryOp (for memrefs passed as arguments or dynamic allocs)
    funcOp.walk([&](circt::handshake::ExternalMemoryOp op) {
      Value opMemref = op.getMemref();
      if (!opMemref) {
        return WalkResult::advance();
      }
      Value traced = traceMemRefOrigin(opMemref);
      if (!traced) {
        return WalkResult::advance();
      }
      if (traced == originalMemref || traced == memref || opMemref == memref) {
        memOp = op.getOperation();
        ldCount = op.getLdCount();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    // If not found, try MemoryOp (for static allocations)
    // MemoryOp doesn't store the memref value, only its type
    if (!memOp) {
      auto memrefType = memref.getType();
      funcOp.walk([&](circt::handshake::MemoryOp op) {
        if (op.getMemRefType() == memrefType) {
          // Additional check: verify this MemoryOp is actually used by our memref
          // Count how many MemoryOps have this type
          int countWithType = 0;
          funcOp.walk([&](circt::handshake::MemoryOp checkOp) {
            if (checkOp.getMemRefType() == memrefType) {
              countWithType++;
            }
          });

          // If this is the only MemoryOp with this type, use it
          if (countWithType == 1) {
            memOp = op.getOperation();
            ldCount = op.getLdCount();
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
    }

    // Prepare control signal for deallocation
    Value deallocControl;
    builder.setInsertionPoint(returnOp);

    if (!hasAccess || !memOp) {
      // Case 1: No memory accesses or no memory operation found
      // In pure dataflow, dealloc should execute when function returns
      // Use non-memref function arguments to derive entry token
      // IMPORTANT: Memref arguments MUST be used directly by extmemory operations
      // per CIRCT handshake dialect requirements
      SmallVector<Value> nonMemrefArgs;
      for (auto arg : funcOp.getArguments()) {
        if (!isa<MemRefType>(arg.getType())) {
          nonMemrefArgs.push_back(arg);
        }
      }

      if (nonMemrefArgs.empty()) {
        // No non-memref args - skip dealloc control (dealloc will execute naturally)
        continue;
      }

      // Join non-memref function arguments to create a "function entry" token
      builder.setInsertionPointToStart(&funcOp.getBody().front());
      auto joinOp = circt::handshake::JoinOp::create(builder,
          funcOp.getLoc(), nonMemrefArgs);
      Value entryToken = joinOp.getResult();
      builder.setInsertionPoint(returnOp);

      // Create constant true triggered by function entry
      deallocControl = circt::handshake::ConstantOp::create(builder,
          loc, builder.getI1Type(),
          builder.getIntegerAttr(builder.getI1Type(), 1),
          entryToken);

    } else {
      // Case 2: Has memory accesses - must synchronize with memory operations

      // Step 1: Collect completion tokens from memory operation
      SmallVector<Value> completionTokens;
      for (int i = ldCount; i < memOp->getNumResults(); ++i) {
        completionTokens.push_back(memOp->getResult(i));
      }

      if (completionTokens.empty()) {
        return allocOp.emitError("memory operation has no completion tokens");
      }

      // Step 2: Join all completion tokens into a single token
      Value completionToken;
      if (completionTokens.size() == 1) {
        completionToken = completionTokens[0];
      } else {
        auto joinOp = circt::handshake::JoinOp::create(builder,
            loc, completionTokens);
        completionToken = joinOp.getResult();
      }

      // Step 3: Find loop exit control signal (if in a loop)
      Value loopExitControl = findLoopExitControl(funcOp, allocOp.getOperation());

      if (loopExitControl) {
        // Step 4a: In a loop - synchronize loop exit with completion token
        auto syncOp = circt::handshake::SyncOp::create(builder,
            loc, ValueRange{loopExitControl, completionToken});
        deallocControl = syncOp.getResults()[0];  // First result has same type as loopExitControl (i1)
      } else {
        // Step 4b: Not in a loop - use completion token to trigger constant i1 true
        Value trueConstant = circt::handshake::ConstantOp::create(builder,
            loc, builder.getI1Type(),
            builder.getIntegerAttr(builder.getI1Type(), 1),
            completionToken);
        deallocControl = trueConstant;
      }
    }

    // Step 5: Create the dealloc operation with control dependency
    auto invariantOp = dsa::InvariantOp::create(builder,
        loc, memref.getType(), deallocControl, memref);

    // Step 6: Insert memref.dealloc before return
    auto deallocOp = memref::DeallocOp::create(builder,
        loc, invariantOp.getO());
  }

  return success();
}

} // namespace dsa
} // namespace mlir

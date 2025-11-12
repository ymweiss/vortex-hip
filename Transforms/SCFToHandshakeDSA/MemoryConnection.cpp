//===- MemoryConnection.cpp - Memory operation connection -------*- C++ -*-===//
//
// Functions for connecting memory operations to memory interfaces
// and managing control flow
//
//===----------------------------------------------------------------------===//

#include "Common.h"
#include "MemoryOpsHelpers.h"
#include "SerialRecursiveMemCtrl.h"
#include <algorithm>
#include <set>

namespace mlir {
namespace dsa {

//===----------------------------------------------------------------------===//
// Connect Return Control
//===----------------------------------------------------------------------===//

template <typename FuncOpT, typename RewriterT>
void connectReturnControl(FuncOpT funcOp, RewriterT &rewriter,
                          Value finalDone, Value entryToken) {

  DSA_DEBUG_STREAM << "\n========================================\n";
  DSA_DEBUG_STREAM << "[SCF MEM CTRL] Connecting Return Control\n";
  DSA_DEBUG_STREAM << "========================================\n";

  // Find return operation
  circt::handshake::ReturnOp returnOp = nullptr;
  funcOp.walk([&](circt::handshake::ReturnOp op) {
    returnOp = op;
    return WalkResult::interrupt();
  });

  if (!returnOp) {
    DSA_DEBUG_STREAM << "[SCF MEM CTRL] No return operation found\n";
    return;
  }

  // In serial-recursive algorithm, finalDone is already the correct token
  Value returnCtrl = finalDone ? finalDone : entryToken;

  DSA_DEBUG_STREAM << "[SCF MEM CTRL] Using "
                   << (finalDone ? "final done token" : "entry token (no ops)")
                   << " for return\n";

  // Connect to return
  SmallVector<Value> returnOperands(returnOp.getOperands());
  returnOperands.push_back(returnCtrl);
  returnOp->setOperands(returnOperands);

  // Update function signature
  auto currentFuncType = funcOp.getFunctionType();
  SmallVector<Type> newResultTypes(currentFuncType.getResults());
  newResultTypes.push_back(rewriter.getNoneType());

  auto newFuncType = rewriter.getFunctionType(
      currentFuncType.getInputs(), newResultTypes);
  funcOp.setFunctionTypeAttr(TypeAttr::get(newFuncType));

  // Update resNames and resAttrs
  SmallVector<Attribute> resNames, resAttrs;
  if (auto resNamesAttr = funcOp->template getAttrOfType<ArrayAttr>("resNames")) {
    resNames.append(resNamesAttr.begin(), resNamesAttr.end());
  }
  if (auto resAttrsAttr = funcOp.getResAttrsAttr()) {
    resAttrs.append(resAttrsAttr.begin(), resAttrsAttr.end());
  }

  resNames.push_back(rewriter.getStringAttr("ctrl"));
  NamedAttrList ctrlAttrs;
  ctrlAttrs.append("hw.name", rewriter.getStringAttr("ctrl"));
  resAttrs.push_back(rewriter.getDictionaryAttr(ctrlAttrs));

  funcOp->setAttr("resNames", rewriter.getArrayAttr(resNames));
  funcOp.setResAttrsAttr(rewriter.getArrayAttr(resAttrs));

  DSA_DEBUG_STREAM << "[SCF MEM CTRL] Return control connected\n";
  DSA_DEBUG_STREAM << "========================================\n\n";
}

//===----------------------------------------------------------------------===//
// Replace memref.load/store with handshake equivalents
//===----------------------------------------------------------------------===//

template <typename FuncOpType, typename RewriterT>
LogicalResult replaceMemoryOps(FuncOpType funcOp,
                                RewriterT &rewriter,
                                MemRefToMemoryAccessOp &memRefOps) {
  std::vector<Operation *> opsToErase;

  // Enrich memRefOps with function arguments
  for (auto arg : funcOp.getArguments()) {
    auto memrefType = dyn_cast<MemRefType>(arg.getType());
    if (!memrefType) continue;
    if (failed(isValidMemrefType(arg.getLoc(), memrefType)))
      return failure();
    memRefOps.insert(std::make_pair(arg, std::vector<Operation *>()));
  }

  // Replace load and store ops
  for (Operation &op : funcOp.getBody().getOps()) {
    if (!isMemoryOp(&op)) continue;

    rewriter.setInsertionPoint(&op);
    Value memref;
    if (getOpMemRef(&op, memref).failed())
      return failure();

    Operation *newOp = nullptr;

    if (auto loadOp = dyn_cast<memref::LoadOp>(&op)) {
      SmallVector<Value, 8> indices(loadOp.getIndices());
      newOp = circt::handshake::LoadOp::create(rewriter, op.getLoc(), memref, indices);
      op.getResult(0).replaceAllUsesWith(newOp->getResult(0));
    } else if (auto storeOp = dyn_cast<memref::StoreOp>(&op)) {
      SmallVector<Value, 8> indices(storeOp.getIndices());
      newOp = circt::handshake::StoreOp::create(
          rewriter, op.getLoc(), storeOp.getValueToStore(), indices);
    }

    if (newOp) {
      // Copy SCF metadata
      if (auto scfPathAttr = op.getAttrOfType<ArrayAttr>("dsa.scf_path")) {
        newOp->setAttr("dsa.scf_path", scfPathAttr);
      }
      if (auto globalSeqAttr = op.getAttrOfType<IntegerAttr>("dsa.global_seq")) {
        newOp->setAttr("dsa.global_seq", globalSeqAttr);
      }
      if (auto topSeqAttr = op.getAttrOfType<IntegerAttr>("dsa.top_seq")) {
        newOp->setAttr("dsa.top_seq", topSeqAttr);
      }
      if (auto localSeqAttr = op.getAttrOfType<IntegerAttr>("dsa.local_seq")) {
        newOp->setAttr("dsa.local_seq", localSeqAttr);
      }
      memRefOps[memref].push_back(newOp);
      opsToErase.push_back(&op);
    }
  }

  // Erase old memory ops
  for (auto *op : opsToErase) {
    for (int j = 0, e = op->getNumOperands(); j < e; ++j)
      op->eraseOperand(0);
    rewriter.eraseOp(op);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Helper: Set load data inputs from memory outputs
//===----------------------------------------------------------------------===//

void setLoadDataInputs(ArrayRef<Operation *> memOps, Operation *memOp) {
  int ldCount = 0;
  for (auto *op : memOps) {
    if (isa<circt::handshake::LoadOp>(op)) {
      auto loadOp = cast<circt::handshake::LoadOp>(op);
      SmallVector<Value> operands(loadOp.getOperands());
      operands.push_back(memOp->getResult(ldCount++));
      loadOp->setOperands(operands);
    }
  }
}

//===----------------------------------------------------------------------===//
// Connect memory operations to extmemory/memory interface
//===----------------------------------------------------------------------===//

template <typename FuncOpType, typename RewriterT>
LogicalResult connectToMemory(FuncOpType funcOp, RewriterT &rewriter,
                               MemRefToMemoryAccessOp &memRefOps) {
  int memCount = 0;
  Value cachedEntryToken;

  // Track memrefs with operations for later parallel processing
  SmallVector<std::pair<Value, SmallVector<Operation*>>> memrefsWithOps;

  // Step 1: Create all memory/extmemory operations and wire data connections
  for (auto &[memrefOperand, memOps] : memRefOps) {
    if (memOps.empty()) continue;

    // Sort by global_seq within this memref
    std::sort(memOps.begin(), memOps.end(), [](Operation *a, Operation *b) {
      auto aSeq = a->getAttrOfType<IntegerAttr>("dsa.global_seq");
      auto bSeq = b->getAttrOfType<IntegerAttr>("dsa.global_seq");
      if (!aSeq || !bSeq) return false;
      return aSeq.getInt() < bSeq.getInt();
    });

    Value originalMemRef = getOriginalMemRef(memrefOperand);
    auto memrefType = cast<MemRefType>(memrefOperand.getType());

    if (failed(isValidMemrefType(memrefOperand.getLoc(), memrefType)))
      return failure();

    bool isExternalMemory = isa<BlockArgument>(originalMemRef) ||
                            hasDynamicDimensions(memrefType);

    // Collect operands and record port indices
    SmallVector<Value> operands;
    SmallVector<int> indexMap(memOps.size(), 0);

    int idx = 0, ldCount = 0, stCount = 0;

    // Stores first
    for (int i = 0, e = memOps.size(); i < e; ++i) {
      auto *op = memOps[i];
      if (isa<circt::handshake::StoreOp>(op)) {
        SmallVector<Value> results = getResultsToMemory(op);
        operands.insert(operands.end(), results.begin(), results.end());
        indexMap[i] = idx++;
        op->setAttr("dsa.store_port_idx", rewriter.getI32IntegerAttr(stCount));
        stCount++;
      }
    }

    // Loads second
    for (int i = 0, e = memOps.size(); i < e; ++i) {
      auto *op = memOps[i];
      if (isa<circt::handshake::LoadOp>(op)) {
        SmallVector<Value> results = getResultsToMemory(op);
        operands.insert(operands.end(), results.begin(), results.end());
        indexMap[i] = idx++;
        op->setAttr("dsa.load_port_idx", rewriter.getI32IntegerAttr(ldCount));
        ldCount++;
      }
    }

    // Create memory operation
    rewriter.setInsertionPointToStart(&funcOp.getBody().front());
    Location loc = funcOp.getLoc();

    int control_outputs = ldCount + stCount;
    Operation *memOp = nullptr;

    if (isExternalMemory) {
      memOp = circt::handshake::ExternalMemoryOp::create(
          rewriter, loc, originalMemRef, operands, ldCount, stCount, memCount++);
    } else {
      memOp = circt::handshake::MemoryOp::create(
          rewriter, loc, operands, ldCount, control_outputs, false, memCount++, originalMemRef);
    }

    // If the memref comes from memref.get_global, store the global name as an attribute
    // so the simulator knows which module-level global to load
    if (auto getGlobalOp = originalMemRef.getDefiningOp<memref::GetGlobalOp>()) {
      memOp->setAttr("dsa.global_memref", getGlobalOp.getNameAttr());
    }

    // Wire up load data inputs BEFORE control inputs
    setLoadDataInputs(memOps, memOp);

    // Track this memref for control processing
    SmallVector<Operation*> opsVec(memOps.begin(), memOps.end());
    memrefsWithOps.push_back(std::make_pair(memrefOperand, opsVec));
  }

  // Create dummy extmemory for unused memref arguments
  for (auto arg : funcOp.getArguments()) {
    auto memrefType = dyn_cast<MemRefType>(arg.getType());
    if (!memrefType) continue;

    if (memRefOps.find(arg) != memRefOps.end() && !memRefOps[arg].empty())
      continue;

    rewriter.setInsertionPointToStart(&funcOp.getBody().front());
    SmallVector<Value> emptyOperands;
    circt::handshake::ExternalMemoryOp::create(
        rewriter, arg.getLoc(), arg, emptyOperands, 0, 0, memCount++);
  }

  // Remove unused alloc operations
  std::vector<Operation *> opsToDelete;
  for (auto &op : funcOp.getBody().getOps()) {
    if (isAllocOp(&op) && op.getResult(0).use_empty())
      opsToDelete.push_back(&op);
  }
  for (auto *op : opsToDelete) {
    rewriter.eraseOp(op);
  }

  // Step 2: Process control inputs for all memrefs in parallel
  if (memrefsWithOps.empty()) {
    // No memory operations: connect entry token directly to return
    if (!cachedEntryToken) {
      circt::handshake::JoinOp markedJoin = nullptr;
      funcOp.walk([&](circt::handshake::JoinOp joinOp) {
        if (joinOp->hasAttr("dsa.mem_entry_token")) {
          markedJoin = joinOp;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });

      if (markedJoin) {
        cachedEntryToken = markedJoin.getResult();
      }
    }

    connectReturnControl(funcOp, rewriter, cachedEntryToken, cachedEntryToken);
    return success();
  }

  // Get entry token
  if (!cachedEntryToken) {
    circt::handshake::JoinOp markedJoin = nullptr;
    funcOp.walk([&](circt::handshake::JoinOp joinOp) {
      if (joinOp->hasAttr("dsa.mem_entry_token")) {
        markedJoin = joinOp;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (markedJoin) {
      cachedEntryToken = markedJoin.getResult();
    }
  }

  if (!cachedEntryToken) {
    funcOp.emitError("No FUNC_ENTRY token found");
    return failure();
  }

  // Find return operation for insertion point
  Operation *returnOp = nullptr;
  funcOp.walk([&](circt::handshake::ReturnOp retOp) {
    returnOp = retOp.getOperation();
    return WalkResult::interrupt();
  });

  if (returnOp) {
    rewriter.setInsertionPoint(returnOp);
  }

  SmallVector<Value> memrefDoneTokens;

  if (memrefsWithOps.size() == 1) {
    // Single memref: no fork needed
    DSA_DEBUG_STREAM << "[SCF MEM CTRL] Single memref, processing directly\n";

    auto &[memref, memOps] = memrefsWithOps[0];

    // Process this memref with entry token
    SerialRecursiveMemCtrl<FuncOpType, RewriterT> processor(funcOp, rewriter, memOps);
    Value memrefDone = processor.processTopLevel(cachedEntryToken);

    // Connect to return
    connectReturnControl(funcOp, rewriter, memrefDone, cachedEntryToken);

  } else {
    // Multiple memrefs: fork entry token to each memref
    DSA_DEBUG_STREAM << "[SCF MEM CTRL] Multiple memrefs (" << memrefsWithOps.size()
                     << "), forking entry token\n";

    auto forkOp = circt::handshake::ForkOp::create(
      rewriter,
      funcOp.getLoc(),
      cachedEntryToken,
      memrefsWithOps.size()
    );

    // Process each memref independently
    for (size_t i = 0; i < memrefsWithOps.size(); ++i) {
      auto &[memref, memOps] = memrefsWithOps[i];
      Value forkedToken = forkOp.getResults()[i];

      DSA_DEBUG_STREAM << "[SCF MEM CTRL]   Processing memref #" << i
                       << " with " << memOps.size() << " operations\n";

      // Run serial-recursive algorithm for this memref
      SerialRecursiveMemCtrl<FuncOpType, RewriterT> processor(funcOp, rewriter, memOps);
      Value memrefDone = processor.processTopLevel(forkedToken);

      memrefDoneTokens.push_back(memrefDone);
    }

    // Join all memref done tokens
    DSA_DEBUG_STREAM << "[SCF MEM CTRL] Joining " << memrefDoneTokens.size()
                     << " memref done tokens\n";

    Value finalDone;
    if (memrefDoneTokens.size() == 1) {
      finalDone = memrefDoneTokens[0];
    } else {
      rewriter.setInsertionPoint(returnOp);
      auto joinOp = circt::handshake::JoinOp::create(
        rewriter, funcOp.getLoc(), memrefDoneTokens);
      finalDone = joinOp.getResult();
    }

    // Connect final joined done to return
    connectReturnControl(funcOp, rewriter, finalDone, cachedEntryToken);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Template Instantiations
//===----------------------------------------------------------------------===//

template void connectReturnControl<func::FuncOp, ConversionPatternRewriter>(
    func::FuncOp, ConversionPatternRewriter &, Value, Value);
template void connectReturnControl<circt::handshake::FuncOp, ConversionPatternRewriter>(
    circt::handshake::FuncOp, ConversionPatternRewriter &, Value, Value);

template LogicalResult replaceMemoryOps<func::FuncOp, ConversionPatternRewriter>(
    func::FuncOp, ConversionPatternRewriter &, MemRefToMemoryAccessOp &);
template LogicalResult replaceMemoryOps<circt::handshake::FuncOp, ConversionPatternRewriter>(
    circt::handshake::FuncOp, ConversionPatternRewriter &, MemRefToMemoryAccessOp &);

template LogicalResult connectToMemory<func::FuncOp, ConversionPatternRewriter>(
    func::FuncOp, ConversionPatternRewriter &, MemRefToMemoryAccessOp &);
template LogicalResult connectToMemory<circt::handshake::FuncOp, ConversionPatternRewriter>(
    circt::handshake::FuncOp, ConversionPatternRewriter &, MemRefToMemoryAccessOp &);

} // namespace dsa
} // namespace mlir

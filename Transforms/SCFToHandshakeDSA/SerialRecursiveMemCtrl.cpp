//===- SerialRecursiveMemCtrl.cpp - Serial-recursive algorithm -*- C++ -*-===//
//
// COMPLETE REWRITE: Serial-Recursive Algorithm
// Each SCF block abstracted as: entry -> done
// Utilizes RAR parallelism with fork/join
//
//===----------------------------------------------------------------------===//

#include "SerialRecursiveMemCtrl.h"

namespace mlir {
namespace dsa {

//===----------------------------------------------------------------------===//
// SerialRecursiveMemCtrl Implementation
//===----------------------------------------------------------------------===//

template <typename FuncOpT, typename RewriterT>
SerialRecursiveMemCtrl<FuncOpT, RewriterT>::SerialRecursiveMemCtrl(
    FuncOpT funcOp, RewriterT &rewriter, ArrayRef<Operation *> memOps)
    : funcOp(funcOp), rewriter(rewriter), allMemOps(memOps),
      currentOpIdx(0) {}

template <typename FuncOpT, typename RewriterT>
Value SerialRecursiveMemCtrl<FuncOpT, RewriterT>::processTopLevel(
    Value entryToken) {
  DSA_DEBUG_STREAM << "\n========================================\n";
  DSA_DEBUG_STREAM << "[SCF MEM CTRL] Starting serial-recursive processing\n";
  DSA_DEBUG_STREAM << "[SCF MEM CTRL] Total operations: " << allMemOps.size() << "\n";
  DSA_DEBUG_STREAM << "========================================\n";

  SmallVector<std::string> topPath; // empty path = top level
  return processLevel(topPath, entryToken);
}

template <typename FuncOpT, typename RewriterT>
bool SerialRecursiveMemCtrl<FuncOpT, RewriterT>::hasOpsInPath(
    ArrayRef<std::string> targetPath) const {
  for (Operation *op : allMemOps) {
    auto opPath = getScfPath(op);
    if (pathMatchesPrefix(opPath, targetPath)) {
      return true;
    }
  }
  return false;
}

template <typename FuncOpT, typename RewriterT>
Value SerialRecursiveMemCtrl<FuncOpT, RewriterT>::processLevel(
    ArrayRef<std::string> currentPath, Value currentToken) {
  DSA_DEBUG_STREAM << "\n[SCF MEM CTRL] === Processing level: depth="
                   << currentPath.size() << " ===\n";

  while (currentOpIdx < allMemOps.size()) {
    Operation *op = allMemOps[currentOpIdx];
    auto opPath = getScfPath(op);

    DSA_DEBUG_STREAM << "[SCF MEM CTRL] Current op #" << (currentOpIdx + 1)
                     << " (" << (isa<circt::handshake::LoadOp>(op) ? "LOAD" : "STORE")
                     << "), op_path depth=" << opPath.size()
                     << ", current_level depth=" << currentPath.size() << "\n";

    // Case 1: Op is at deeper level - need to enter child SCF block
    if (opPath.size() > currentPath.size() &&
        pathMatchesPrefix(opPath, currentPath)) {
      DSA_DEBUG_STREAM << "[SCF MEM CTRL]   -> Entering child SCF block\n";
      currentToken = enterAndProcessChild(currentPath, currentToken);
      continue;
    }

    // Case 2: Op is at different branch or parent level - exit this level
    if (!pathMatchesPrefix(opPath, currentPath)) {
      DSA_DEBUG_STREAM << "[SCF MEM CTRL]   -> Op in different branch/parent, exiting level\n";
      break;
    }

    // Case 3: Op is at same level - process it
    DSA_DEBUG_STREAM << "[SCF MEM CTRL]   -> Op at same level, processing\n";

    // Check for consecutive loads (RAR parallelism)
    // IMPORTANT: Only loads at EXACTLY the same level can be parallelized
    if (isa<circt::handshake::LoadOp>(op)) {
      SmallVector<size_t> consecutiveLoads;
      consecutiveLoads.push_back(currentOpIdx);

      // Look ahead for more loads at EXACTLY same level
      size_t lookAhead = currentOpIdx + 1;
      while (lookAhead < allMemOps.size()) {
        Operation *nextOp = allMemOps[lookAhead];
        auto nextPath = getScfPath(nextOp);

        // Must be load and at EXACTLY same level
        if (isa<circt::handshake::LoadOp>(nextOp) &&
            nextPath.size() == currentPath.size() &&
            pathMatchesPrefix(nextPath, currentPath)) {
          consecutiveLoads.push_back(lookAhead);
          lookAhead++;
        } else {
          break;
        }
      }

      if (consecutiveLoads.size() > 1) {
        DSA_DEBUG_STREAM << "[SCF MEM CTRL]   Found " << consecutiveLoads.size()
                         << " consecutive loads, using RAR parallelism\n";
        currentToken = processConsecutiveLoads(currentPath, currentToken,
                                                consecutiveLoads);
        currentOpIdx = lookAhead; // Skip processed loads
        continue;
      }
    }

    // Single operation: connect control input, get done token
    currentToken = processSingleOp(op, currentToken);
    currentOpIdx++;
  }

  DSA_DEBUG_STREAM << "[SCF MEM CTRL] === Level done (depth="
                   << currentPath.size() << ") ===\n";
  return currentToken;
}

template <typename FuncOpT, typename RewriterT>
Value SerialRecursiveMemCtrl<FuncOpT, RewriterT>::enterAndProcessChild(
    ArrayRef<std::string> parentPath, Value parentToken) {
  // Determine which child level to enter
  Operation *firstOp = allMemOps[currentOpIdx];
  auto firstOpPath = getScfPath(firstOp);

  if (firstOpPath.size() <= parentPath.size()) {
    DSA_DEBUG_STREAM << "[SCF MEM CTRL]   ERROR: Invalid child path\n";
    return parentToken;
  }

  std::string childLevelStr = firstOpPath[parentPath.size()];
  ScfLevel childLevel = ScfLevel::parse(childLevelStr);

  if (!childLevel.valid) {
    DSA_DEBUG_STREAM << "[SCF MEM CTRL]   ERROR: Invalid level format: "
                     << childLevelStr << "\n";
    return parentToken;
  }

  DSA_DEBUG_STREAM << "[SCF MEM CTRL]   Entering " << childLevel.type << "."
                   << childLevel.id << "." << childLevel.branch << "\n";

  // Get insertion point (before return)
  Operation *returnOp = nullptr;
  funcOp.walk([&](circt::handshake::ReturnOp retOp) {
    returnOp = retOp.getOperation();
    return WalkResult::interrupt();
  });
  if (returnOp) {
    rewriter.setInsertionPoint(returnOp);
  } else {
    rewriter.setInsertionPointToEnd(&funcOp.getBody().front());
  }

  Value childDone;

  if (childLevel.type == "while" || childLevel.type == "for") {
    childDone = processLoopBlock(parentPath, parentToken, childLevel);
  } else if (childLevel.type == "if") {
    childDone = processIfBlock(parentPath, parentToken, childLevel);
  } else {
    DSA_DEBUG_STREAM << "[SCF MEM CTRL]   ERROR: Unknown SCF type: "
                     << childLevel.type << "\n";
    return parentToken;
  }

  return childDone;
}

template <typename FuncOpT, typename RewriterT>
Value SerialRecursiveMemCtrl<FuncOpT, RewriterT>::processLoopBlock(
    ArrayRef<std::string> parentPath, Value parentToken,
    const ScfLevel &level) {
  DSA_DEBUG_STREAM << "[SCF MEM CTRL]   Processing " << level.type
                   << " block: " << level.fullRegion << "\n";

  // Find control signal
  Value control = findControlValue(funcOp, parentPath, level.fullRegion);
  if (!control) {
    DSA_DEBUG_STREAM << "[SCF MEM CTRL]   ERROR: No control signal\n";
    return parentToken;
  }

  // Check if we already created carry for this loop
  std::string loopKey = level.region;
  auto carryIt = loopCarryCache.find(loopKey);
  Value carryOutput;
  dsa::CarryOp carryOp;

  if (carryIt == loopCarryCache.end()) {
    // Create new carry operation
    carryOp = dsa::CarryOp::create(
      rewriter,
      funcOp.getLoc(),
      parentToken.getType(),
      control,      // d
      parentToken,  // a
      parentToken   // b (placeholder, will be updated)
    );
    carryOutput = carryOp.getResult();
    loopCarryCache[loopKey] = carryOutput;

    DSA_DEBUG_STREAM << "[SCF MEM CTRL]     Created dsa.carry for "
                     << level.region << "\n";
  } else {
    carryOutput = carryIt->second;
    carryOp = cast<dsa::CarryOp>(carryOutput.getDefiningOp());
    DSA_DEBUG_STREAM << "[SCF MEM CTRL]     Reusing dsa.carry for "
                     << level.region << "\n";
  }

  // Enter loop body: process child level
  SmallVector<std::string> childPath(parentPath.begin(), parentPath.end());
  childPath.push_back(level.fullRegion);

  Value loopBodyDone = processLevel(childPath, carryOutput);

  // Connect loop body done to carry's b input
  carryOp->setOperand(2, loopBodyDone); // b端 = loop body done

  // Create cond_br to get loop exit signal
  auto condBr = circt::handshake::ConditionalBranchOp::create(
    rewriter,
    funcOp.getLoc(),
    loopBodyDone.getType(),
    loopBodyDone.getType(),
    control,
    loopBodyDone
  );

  // Update carry's b input to use cond_br's true output
  carryOp->setOperand(2, condBr.getTrueResult());

  DSA_DEBUG_STREAM << "[SCF MEM CTRL]     " << level.type
                   << " done signal = cond_br.false\n";

  // Loop's done signal is cond_br's false output
  return condBr.getFalseResult();
}

template <typename FuncOpT, typename RewriterT>
Value SerialRecursiveMemCtrl<FuncOpT, RewriterT>::processIfBlock(
    ArrayRef<std::string> parentPath, Value parentToken,
    const ScfLevel &level) {
  DSA_DEBUG_STREAM << "[SCF MEM CTRL]   Processing if block: " << level.region << "\n";

  // Find control signal for if condition
  Value control = findControlValue(funcOp, parentPath, level.region);
  if (!control) {
    DSA_DEBUG_STREAM << "[SCF MEM CTRL]   ERROR: No control signal for if\n";
    return parentToken;
  }

  // Create cond_br to route entry token to then/else
  auto entryCondBr = circt::handshake::ConditionalBranchOp::create(
    rewriter,
    funcOp.getLoc(),
    parentToken.getType(),
    parentToken.getType(),
    control,
    parentToken
  );

  Value thenEntry = entryCondBr.getTrueResult();
  Value elseEntry = entryCondBr.getFalseResult();

  DSA_DEBUG_STREAM << "[SCF MEM CTRL]     Created entry cond_br for if."
                   << level.id << "\n";

  // Check which branches have operations
  SmallVector<std::string> thenPath(parentPath.begin(), parentPath.end());
  thenPath.push_back(level.region + ".then");
  bool hasThenOps = hasOpsInPath(thenPath);

  SmallVector<std::string> elsePath(parentPath.begin(), parentPath.end());
  elsePath.push_back(level.region + ".else");
  bool hasElseOps = hasOpsInPath(elsePath);

  DSA_DEBUG_STREAM << "[SCF MEM CTRL]     Then branch: " << (hasThenOps ? "has ops" : "empty")
                   << ", Else branch: " << (hasElseOps ? "has ops" : "empty") << "\n";

  // Process then branch
  Value thenDone;
  if (hasThenOps) {
    thenDone = processLevel(thenPath, thenEntry);
    DSA_DEBUG_STREAM << "[SCF MEM CTRL]     Then branch processed\n";
  } else {
    thenDone = thenEntry;  // Empty branch: entry = done
    DSA_DEBUG_STREAM << "[SCF MEM CTRL]     Then branch empty, entry=done\n";
  }

  // Process else branch
  Value elseDone;
  if (hasElseOps) {
    // Find first else operation
    size_t savedIdx = currentOpIdx;
    for (size_t i = 0; i < allMemOps.size(); ++i) {
      auto opPath = getScfPath(allMemOps[i]);
      if (pathMatchesPrefix(opPath, elsePath)) {
        currentOpIdx = i;
        break;
      }
    }

    elseDone = processLevel(elsePath, elseEntry);
    DSA_DEBUG_STREAM << "[SCF MEM CTRL]     Else branch processed\n";

    // Advance past remaining ops in both branches
    while (currentOpIdx < allMemOps.size()) {
      auto opPath = getScfPath(allMemOps[currentOpIdx]);
      if (pathMatchesPrefix(opPath, thenPath) ||
          pathMatchesPrefix(opPath, elsePath)) {
        currentOpIdx++;
      } else {
        break;
      }
    }
  } else {
    elseDone = elseEntry;  // Empty branch: entry = done
    DSA_DEBUG_STREAM << "[SCF MEM CTRL]     Else branch empty, entry=done\n";
  }

  // Create MuxOp to merge then/else done signals
  auto muxOp = circt::handshake::MuxOp::create(
    rewriter,
    funcOp.getLoc(),
    thenDone.getType(),
    control,
    ValueRange{elseDone, thenDone} // [false=else, true=then]
  );

  DSA_DEBUG_STREAM << "[SCF MEM CTRL]     Created mux for if." << level.id
                   << " done signal\n";

  return muxOp.getResult();
}

template <typename FuncOpT, typename RewriterT>
Value SerialRecursiveMemCtrl<FuncOpT, RewriterT>::processSingleOp(
    Operation *op, Value entryToken) {
  bool isLoad = isa<circt::handshake::LoadOp>(op);
  auto globalSeq = op->getAttrOfType<IntegerAttr>("dsa.global_seq");
  int seq = globalSeq ? globalSeq.getInt() : -1;

  DSA_DEBUG_STREAM << "[SCF MEM CTRL]     Connecting "
                   << (isLoad ? "LOAD" : "STORE") << " #" << seq
                   << " control input\n";

  // Connect entry token to control input
  addValueToOperands(op, entryToken);

  // Get done token from extmemory
  Value doneToken = getExtmemoryDoneToken(op);
  if (!doneToken) {
    DSA_DEBUG_STREAM << "[SCF MEM CTRL]     ERROR: No done token for op #"
                     << seq << "\n";
    return entryToken; // Fallback
  }

  DSA_DEBUG_STREAM << "[SCF MEM CTRL]     Got done token from extmemory\n";
  return doneToken;
}

template <typename FuncOpT, typename RewriterT>
Value SerialRecursiveMemCtrl<FuncOpT, RewriterT>::processConsecutiveLoads(
    ArrayRef<std::string> currentPath, Value entryToken,
    ArrayRef<size_t> loadIndices) {
  DSA_DEBUG_STREAM << "[SCF MEM CTRL]     RAR Parallelism: forking entry token to "
                   << loadIndices.size() << " loads\n";

  // Get insertion point
  Operation *returnOp = nullptr;
  funcOp.walk([&](circt::handshake::ReturnOp retOp) {
    returnOp = retOp.getOperation();
    return WalkResult::interrupt();
  });
  if (returnOp) {
    rewriter.setInsertionPoint(returnOp);
  } else {
    rewriter.setInsertionPointToEnd(&funcOp.getBody().front());
  }

  // Fork entry token
  SmallVector<Value> forkedTokens;
  if (loadIndices.size() == 1) {
    forkedTokens.push_back(entryToken);
  } else {
    auto forkOp = circt::handshake::ForkOp::create(
      rewriter,
      funcOp.getLoc(),
      entryToken,
      loadIndices.size()
    );
    for (unsigned i = 0; i < loadIndices.size(); ++i) {
      forkedTokens.push_back(forkOp.getResults()[i]);
    }
    DSA_DEBUG_STREAM << "[SCF MEM CTRL]       Created fork with "
                     << loadIndices.size() << " outputs\n";
  }

  // Process each load
  SmallVector<Value> doneTokens;
  for (size_t i = 0; i < loadIndices.size(); ++i) {
    size_t opIdx = loadIndices[i];
    Operation *loadOp = allMemOps[opIdx];
    Value tokenForLoad = forkedTokens[i];

    auto globalSeq = loadOp->getAttrOfType<IntegerAttr>("dsa.global_seq");
    int seq = globalSeq ? globalSeq.getInt() : -1;

    // Connect token to load's control input
    addValueToOperands(loadOp, tokenForLoad);
    DSA_DEBUG_STREAM << "[SCF MEM CTRL]       Load #" << seq
                     << " control connected\n";

    // Get done token
    Value doneToken = getExtmemoryDoneToken(loadOp);
    if (doneToken) {
      doneTokens.push_back(doneToken);
      DSA_DEBUG_STREAM << "[SCF MEM CTRL]       Load #" << seq
                       << " done token collected\n";
    }
  }

  // Join done tokens
  if (doneTokens.empty()) {
    DSA_DEBUG_STREAM << "[SCF MEM CTRL]       ERROR: No done tokens collected\n";
    return entryToken;
  }

  if (doneTokens.size() == 1) {
    DSA_DEBUG_STREAM << "[SCF MEM CTRL]       Single done token, no join needed\n";
    return doneTokens[0];
  }

  auto joinOp = circt::handshake::JoinOp::create(
    rewriter, funcOp.getLoc(), doneTokens);

  DSA_DEBUG_STREAM << "[SCF MEM CTRL]       Joined " << doneTokens.size()
                   << " done tokens\n";

  return joinOp.getResult();
}

// Template instantiations
template class SerialRecursiveMemCtrl<func::FuncOp, ConversionPatternRewriter>;
template class SerialRecursiveMemCtrl<circt::handshake::FuncOp, ConversionPatternRewriter>;

} // namespace dsa
} // namespace mlir

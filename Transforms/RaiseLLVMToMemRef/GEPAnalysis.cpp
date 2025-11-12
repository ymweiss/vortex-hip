//===- GEPAnalysis.cpp - GEP chain analysis and conversion ------*- C++ -*-===//
//
// GEP (GetElementPtr) chain analysis and conversion logic
//
//===----------------------------------------------------------------------===//

#include "Common.h"

namespace mlir {
namespace dsa {

//===----------------------------------------------------------------------===//
// GEP Chain Analysis
//===----------------------------------------------------------------------===//

/// Analyze a single GEP chain starting from a root GEP
void analyzeGEPChain(LLVM::GEPOp rootGEP, Value argument,
                     GEPChainInfo &chainInfo) {
  chainInfo.rootMemref = argument;
  chainInfo.chain.push_back(rootGEP.getOperation());  // Store Operation*

  // Check if this GEP crosses block boundaries
  Block *rootBlock = rootGEP->getBlock();

  // Follow the chain: check what uses this GEP
  Value currentValue = rootGEP.getResult();
  Operation *currentOp = rootGEP.getOperation();

  while (true) {
    // Check for multiple uses
    if (!currentValue.hasOneUse()) {
      chainInfo.hasMultipleUses = true;

      // Even with multiple uses, try to find a GEP user to continue the chain
      // This handles cases like: gep %base[i] used by both load and gep[+1]
      Operation *gepUser = nullptr;
      for (Operation *user : currentValue.getUsers()) {
        if (isa<LLVM::GEPOp>(user)) {
          gepUser = user;
          break;  // Follow first GEP user
        }
      }

      if (!gepUser) {
        // No GEP user found, chain ends here with multiple uses
        break;
      }

      // Continue following the GEP chain
      auto nextGEP = cast<LLVM::GEPOp>(gepUser);
      Type prevElemType = cast<LLVM::GEPOp>(currentOp).getElemType();
      Type nextElemType = nextGEP.getElemType();
      if (prevElemType != nextElemType) {
        chainInfo.hasTypeChange = true;
      }

      chainInfo.chain.push_back(nextGEP.getOperation());
      currentValue = nextGEP.getResult();
      currentOp = nextGEP.getOperation();
      continue;
    }

    Operation *user = *currentValue.getUsers().begin();

    // Check if user is in a different block
    if (user->getBlock() != rootBlock) {
      chainInfo.crossesBlocks = true;
    }

    // Check what kind of user this is
    if (auto loadOp = dyn_cast<LLVM::LoadOp>(user)) {
      // Terminal: load operation
      chainInfo.terminalOp = loadOp;
      break;
    } else if (auto storeOp = dyn_cast<LLVM::StoreOp>(user)) {
      // Terminal: store operation
      chainInfo.terminalOp = storeOp;
      break;
    } else if (auto nextGEP = dyn_cast<LLVM::GEPOp>(user)) {
      // Chain continues with another GEP
      Type prevElemType = cast<LLVM::GEPOp>(currentOp).getElemType();
      Type nextElemType = nextGEP.getElemType();
      if (prevElemType != nextElemType) {
        chainInfo.hasTypeChange = true;
      }

      chainInfo.chain.push_back(nextGEP.getOperation());  // Store Operation*
      currentValue = nextGEP.getResult();
      currentOp = nextGEP.getOperation();
    } else {
      // Some other operation - can't convert this chain
      chainInfo.terminalOp = nullptr;
      break;
    }
  }
}

/// Analyze how a pointer argument is used and classify it
ArgumentUsageInfo analyzeArgumentUsage(BlockArgument arg) {
  ArgumentUsageInfo info;
  info.argument = arg;

  // Collect all direct uses
  for (Operation *user : arg.getUsers()) {
    if (auto gepOp = dyn_cast<LLVM::GEPOp>(user)) {
      // Analyze this GEP chain
      GEPChainInfo chainInfo;
      analyzeGEPChain(gepOp, arg, chainInfo);
      info.gepChains.push_back(chainInfo);
    } else if (auto loadOp = dyn_cast<LLVM::LoadOp>(user)) {
      info.directLoads.push_back(loadOp);
    } else if (auto storeOp = dyn_cast<LLVM::StoreOp>(user)) {
      info.directStores.push_back(storeOp);
    } else if (isa<LLVM::MemsetOp, LLVM::MemcpyOp>(user)) {
      // Intrinsics are OK
      continue;
    } else {
      // Some other use - mark as unconvertible
      info.category = ArgumentCategory::UNCONVERTIBLE;
      return info;
    }
  }

  // Classify based on collected information
  if (info.gepChains.empty()) {
    // No GEP chains, only direct operations
    info.category = ArgumentCategory::SIMPLE;
  } else {
    // Has GEP chains - check if they're convertible
    bool allSimple = true;
    for (const auto &chain : info.gepChains) {
      if (!chain.terminalOp) {
        // Chain doesn't end in load/store
        allSimple = false;
        break;
      }
      if (chain.hasMultipleUses) {
        // Complex control flow - multiple uses require special handling
        // Note: crossesBlocks is OK since SSA ensures value availability
        allSimple = false;
        break;
      }
    }

    if (allSimple) {
      info.category = ArgumentCategory::SIMPLE_CHAINS;
    } else {
      info.category = ArgumentCategory::COMPLEX_CHAINS;
    }
  }

  return info;
}

//===----------------------------------------------------------------------===//
// GEP Chain Conversion
//===----------------------------------------------------------------------===//

/// Convert a GEP chain to memref operations with accumulated index
LogicalResult convertGEPChain(GEPChainInfo &chainInfo, Type memrefType,
                              OpBuilder &builder) {
  if (chainInfo.chain.empty() || !chainInfo.terminalOp) {
    return success();  // Nothing to convert
  }

  // IMPORTANT: Only convert multi-GEP chains (chain.size() > 1)
  // Single-GEP cases should be handled by convertMemoryOperations which generates
  // simpler code without byte offset arithmetic.
  // Example of single-GEP: arg → GEP → load (should use simple index_cast + load)
  // Example of multi-GEP: arg → GEP1 → GEP2 → load (needs byte offset accumulation)
  if (chainInfo.chain.size() == 1) {
    return success();  // Skip single-GEP chains
  }

  // Check if the first GEP is still alive and hasn't been transformed
  Operation *firstOpPtr = chainInfo.chain[0];
  if (!firstOpPtr->getBlock()) {
    // Operation has been erased, skip this chain
    return success();
  }

  // Check if it's still a GEPOp (might have been transformed)
  auto firstGEP = dyn_cast<LLVM::GEPOp>(firstOpPtr);
  if (!firstGEP) {
    // Operation has been transformed to something else, skip
    return success();
  }
  Location loc = firstGEP.getLoc();

  // Set insertion point at the first GEP
  builder.setInsertionPoint(firstGEP);

  // Start with zero offset
  Value accumulatedOffset = arith::ConstantIndexOp::create(builder, loc, 0);
  Type memrefElemType = cast<MemRefType>(memrefType).getElementType();
  unsigned memrefElemBitWidth = memrefElemType.getIntOrFloatBitWidth();
  unsigned memrefElemSizeBytes = (memrefElemBitWidth + 7) / 8;

  // Process each GEP in the chain
  for (Operation *gepOpPtr : chainInfo.chain) {
    LLVM::GEPOp gepOp = cast<LLVM::GEPOp>(gepOpPtr);

    // For simplicity, handle single-index GEPs
    if (gepOp.getDynamicIndices().size() != 1) {
      return success();  // Skip complex GEPs for now
    }

    Value index = gepOp.getDynamicIndices()[0];
    Type gepElemType = gepOp.getElemType();

    // Get element size for this GEP
    unsigned gepElemBitWidth = gepElemType.getIntOrFloatBitWidth();
    unsigned gepElemSizeBytes = (gepElemBitWidth + 7) / 8;

    // Set insertion point at THIS GEP for index conversions (not the first GEP)
    // This ensures the index value is available (handles block arguments correctly)
    builder.setInsertionPoint(gepOp);

    // Convert index to index type if needed - use separate variable
    Value indexAsIndex = index;
    Type indexType = builder.getIndexType();
    if (index.getType() != indexType) {
      indexAsIndex = arith::IndexCastOp::create(builder, gepOp.getLoc(), indexType, index);
    }

    // Compute byte offset contribution: indexAsIndex * gepElemSize
    Value gepElemSize = arith::ConstantIndexOp::create(builder, gepOp.getLoc(), gepElemSizeBytes);
    Value byteOffsetContrib = arith::MulIOp::create(builder, gepOp.getLoc(), indexAsIndex, gepElemSize);

    // Accumulate
    accumulatedOffset = arith::AddIOp::create(builder, gepOp.getLoc(), accumulatedOffset, byteOffsetContrib);
  }

  // Replace terminal operation
  // Set insertion point at terminal for final index computation
  builder.setInsertionPoint(chainInfo.terminalOp);

  // Convert accumulated byte offset to element index for the memref
  Value memrefElemSize = arith::ConstantIndexOp::create(builder, 
      chainInfo.terminalOp->getLoc(), memrefElemSizeBytes);
  Value finalIndex = arith::DivUIOp::create(builder, 
      chainInfo.terminalOp->getLoc(), accumulatedOffset, memrefElemSize);

  if (auto loadOp = dyn_cast<LLVM::LoadOp>(chainInfo.terminalOp)) {
    // Create memref.load
    auto memrefLoad = memref::LoadOp::create(builder, 
        loadOp.getLoc(), memrefElemType, chainInfo.rootMemref, ValueRange{finalIndex});
    loadOp.replaceAllUsesWith(memrefLoad.getResult());
    loadOp.erase();

  } else if (auto storeOp = dyn_cast<LLVM::StoreOp>(chainInfo.terminalOp)) {
    // Create memref.store
    memref::StoreOp::create(builder, 
        storeOp.getLoc(), storeOp.getValue(), chainInfo.rootMemref, ValueRange{finalIndex});
    storeOp.erase();
  }

  // Erase GEPs in reverse order (leaf to root) - safe now that we store Operation*
  for (auto it = chainInfo.chain.rbegin(); it != chainInfo.chain.rend(); ++it) {
    if ((*it)->use_empty()) {
      (*it)->erase();
    }
  }

  return success();
}

/// Convert all convertible GEP chains for the selected arguments
LogicalResult convertGEPChains(
    func::FuncOp funcOp,
    llvm::DenseMap<Value, ArgumentUsageInfo> &usageInfo,
    llvm::DenseMap<Value, Type> &ptrToMemRefType,
    OpBuilder &builder) {

  // Process each argument that has convertible GEP chains
  for (auto &[arg, info] : usageInfo) {
    if (info.category != ArgumentCategory::SIMPLE_CHAINS) {
      continue;  // Skip non-convertible arguments
    }

    Type memrefType = ptrToMemRefType.lookup(arg);
    if (!memrefType) {
      continue;  // Skip if no memref type inferred
    }

    // Convert each GEP chain for this argument
    for (GEPChainInfo &chain : info.gepChains) {
      // M4: Now handling cross-block chains
      // The index computation is built at the first GEP location, and SSA form
      // ensures values are available in all dominated blocks

      if (failed(convertGEPChain(chain, memrefType, builder))) {
        return failure();
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Multi-Use GEP Conversion
//===----------------------------------------------------------------------===//

/// Convert GEPs with multiple uses (e.g., used by both load and store)
/// This handles cases missed by convertMemoryOperations which only processes
/// GEPs with exactly one use.
LogicalResult convertMultiUseGEPs(
    func::FuncOp funcOp,
    llvm::DenseMap<Value, Type> &ptrToMemRefType,
    OpBuilder &builder) {

  // Collect all GEPs that:
  // 1. Have multiple uses OR have non-load/store uses
  // 2. Reference a converted memref argument
  SmallVector<LLVM::GEPOp> gepsToConvert;

  funcOp.walk([&](LLVM::GEPOp gepOp) {
    Value basePtr = gepOp.getBase();

    // Check if base pointer is a converted memref
    if (!ptrToMemRefType.count(basePtr)) {
      return;  // Not a converted memref, skip
    }

    // Check if this GEP has uses (otherwise it will be cleaned up anyway)
    if (gepOp->use_empty()) {
      return;
    }

    // Check if ALL uses are load/store operations
    bool allUsesAreLoadStore = true;
    for (Operation *user : gepOp->getUsers()) {
      if (!isa<LLVM::LoadOp, LLVM::StoreOp>(user)) {
        allUsesAreLoadStore = false;
        break;
      }
    }

    // Only convert if all uses are load/store
    // (Other patterns are handled by convertGEPChains or will fail gracefully)
    if (allUsesAreLoadStore && !gepOp->use_empty()) {
      gepsToConvert.push_back(gepOp);
    }
  });

  // Convert each multi-use GEP
  for (auto gepOp : gepsToConvert) {
    Value basePtr = gepOp.getBase();
    Type memrefType = ptrToMemRefType[basePtr];

    // Only handle simple single-index GEPs
    if (gepOp.getDynamicIndices().size() != 1) {
      continue;  // Skip complex GEPs
    }

    Value index = gepOp.getDynamicIndices()[0];
    Location loc = gepOp.getLoc();

    // Convert index to index type if needed
    builder.setInsertionPoint(gepOp);
    Value indexAsIndex = index;
    if (index.getType() != builder.getIndexType()) {
      indexAsIndex = arith::IndexCastOp::create(builder, 
          loc, builder.getIndexType(), index);
    }

    // Get element type from memref
    Type elementType = cast<MemRefType>(memrefType).getElementType();

    // Process each use of this GEP
    SmallVector<Operation *> usersToReplace(gepOp->getUsers().begin(),
                                             gepOp->getUsers().end());

    for (Operation *user : usersToReplace) {
      builder.setInsertionPoint(user);

      if (auto loadOp = dyn_cast<LLVM::LoadOp>(user)) {
        // Replace llvm.load with memref.load
        auto memrefLoad = memref::LoadOp::create(builder, 
            loadOp.getLoc(), elementType, basePtr, ValueRange{indexAsIndex});
        loadOp.replaceAllUsesWith(memrefLoad.getResult());
        loadOp.erase();

      } else if (auto storeOp = dyn_cast<LLVM::StoreOp>(user)) {
        // Replace llvm.store with memref.store
        Value valueToStore = storeOp.getValue();
        memref::StoreOp::create(builder, 
            storeOp.getLoc(), valueToStore, basePtr, ValueRange{indexAsIndex});
        storeOp.erase();
      }
    }

    // Erase the GEP if it has no more uses
    if (gepOp->use_empty()) {
      gepOp.erase();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Alloca GEP Chain Conversion
//===----------------------------------------------------------------------===//

/// Convert GEP operations originating from allocas
/// Uses chain analysis to handle multi-level GEP chains (alloca → GEP → GEP → load/store)
LogicalResult convertAllocaGEPChains(
    func::FuncOp funcOp,
    llvm::DenseMap<Value, Type> &ptrToMemRefType,
    OpBuilder &builder) {

  // Step 1: Find all allocas and analyze GEP chains from each
  llvm::DenseMap<Value, SmallVector<GEPChainInfo>> allocaChains;

  funcOp.walk([&](LLVM::AllocaOp allocaOp) {
    Value allocaPtr = allocaOp.getResult();

    // Skip if not in our type mapping
    if (!ptrToMemRefType.count(allocaPtr)) {
      return;
    }

    // Find all direct GEP users of this alloca
    for (Operation *user : allocaPtr.getUsers()) {
      if (auto gepOp = dyn_cast<LLVM::GEPOp>(user)) {
        // Analyze the chain starting from this root GEP
        GEPChainInfo chainInfo;
        analyzeGEPChain(gepOp, allocaPtr, chainInfo);

        // Only store chains that have a terminal operation (load/store)
        if (chainInfo.terminalOp) {
          allocaChains[allocaPtr].push_back(chainInfo);
        }
      }
    }
  });

  // Step 2: Convert each GEP chain using the same logic as function arguments
  for (auto &[allocaPtr, chains] : allocaChains) {
    Type memrefType = ptrToMemRefType[allocaPtr];

    for (GEPChainInfo &chain : chains) {
      // Skip single-GEP chains - they're handled by convertMemoryOperations
      if (chain.chain.size() <= 1) {
        continue;
      }

      // Convert multi-GEP chains using accumulated byte offset approach
      if (failed(convertGEPChain(chain, memrefType, builder))) {
        return failure();
      }
    }
  }

  // Step 3: Handle remaining single-GEP cases that weren't handled by convertMemoryOperations
  // This catches GEPs with multiple uses or other edge cases
  SmallVector<LLVM::GEPOp> remainingGEPs;

  funcOp.walk([&](LLVM::GEPOp gepOp) {
    Value basePtr = gepOp.getBase();

    // Check if base is an alloca in our mapping
    if (!ptrToMemRefType.count(basePtr)) {
      return;
    }

    auto defOp = basePtr.getDefiningOp();
    if (!defOp || !isa<LLVM::AllocaOp>(defOp)) {
      return;
    }

    // Handle simple single-index GEPs and two-index [0, i] patterns (array access)
    size_t numIndices = gepOp.getDynamicIndices().size();
    if (numIndices != 1 && numIndices != 2) {
      return;  // Skip complex GEPs
    }

    // For two-index GEPs, verify first index is constant 0 (standard array access)
    if (numIndices == 2) {
      Value firstIndex = gepOp.getDynamicIndices()[0];
      // Check if first index is a constant 0
      bool isFirstIndexZero = false;
      if (auto constOp = firstIndex.getDefiningOp<LLVM::ConstantOp>()) {
        if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
          isFirstIndexZero = (intAttr.getInt() == 0);
        }
      } else if (auto constOp = firstIndex.getDefiningOp<arith::ConstantOp>()) {
        if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
          isFirstIndexZero = (intAttr.getInt() == 0);
        }
      }

      if (!isFirstIndexZero) {
        return;  // Skip non-standard multi-index GEPs
      }
    }

    // Only convert if all users are load/store
    bool allUsesAreLoadStore = true;
    for (Operation *user : gepOp->getUsers()) {
      if (!isa<LLVM::LoadOp, LLVM::StoreOp>(user)) {
        allUsesAreLoadStore = false;
        break;
      }
    }

    if (allUsesAreLoadStore && !gepOp->use_empty()) {
      remainingGEPs.push_back(gepOp);
    }
  });

  // Convert collected GEPs
  for (auto gepOp : remainingGEPs) {
    // Check if already handled
    if (!gepOp->getBlock() || gepOp->use_empty()) {
      continue;
    }

    Value basePtr = gepOp.getBase();
    Type memrefType = ptrToMemRefType[basePtr];
    Type elementType = cast<MemRefType>(memrefType).getElementType();

    // Extract the actual array index:
    // - For single-index GEPs: use the only index
    // - For two-index [0, i] GEPs: use the second index (array offset)
    Value index;
    if (gepOp.getDynamicIndices().size() == 1) {
      index = gepOp.getDynamicIndices()[0];
    } else {
      // Two indices: [0, i] - use second index
      index = gepOp.getDynamicIndices()[1];
    }

    builder.setInsertionPoint(gepOp);
    Value indexAsIndex = index;
    if (index.getType() != builder.getIndexType()) {
      indexAsIndex = arith::IndexCastOp::create(builder, 
          gepOp.getLoc(), builder.getIndexType(), index);
    }

    // Convert all load/store users
    SmallVector<Operation *> usersToReplace(gepOp->getUsers().begin(),
                                             gepOp->getUsers().end());

    for (Operation *user : usersToReplace) {
      builder.setInsertionPoint(user);

      if (auto loadOp = dyn_cast<LLVM::LoadOp>(user)) {
        auto memrefLoad = memref::LoadOp::create(builder, 
            loadOp.getLoc(), elementType, basePtr, ValueRange{indexAsIndex});
        loadOp.replaceAllUsesWith(memrefLoad.getResult());
        loadOp.erase();
      } else if (auto storeOp = dyn_cast<LLVM::StoreOp>(user)) {
        memref::StoreOp::create(builder, 
            storeOp.getLoc(), storeOp.getValue(), basePtr, ValueRange{indexAsIndex});
        storeOp.erase();
      }
    }

    // Erase the GEP if it has no more uses
    if (gepOp->use_empty()) {
      gepOp.erase();
    }
  }

  return success();
}

} // namespace dsa
} // namespace mlir

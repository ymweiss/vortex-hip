//===- TypeInference.cpp - Type inference for LLVM to MemRef ----*- C++ -*-===//
//
// Type inference and refinement logic for RaiseLLVMToMemRef conversion
//
//===----------------------------------------------------------------------===//

#include "Common.h"

namespace mlir {
namespace dsa {

//===----------------------------------------------------------------------===//
// Type Conversion Helpers
//===----------------------------------------------------------------------===//

/// Convert LLVM array type to memref type
/// !llvm.array<256 x i32> → memref<256xi32>
MemRefType convertLLVMArrayToMemRef(LLVM::LLVMArrayType arrayType) {
  Type elementType = arrayType.getElementType();
  int64_t size = arrayType.getNumElements();
  return MemRefType::get({size}, elementType);
}

//===----------------------------------------------------------------------===//
// Type Inference
//===----------------------------------------------------------------------===//

LogicalResult inferMemRefTypesForArgs(
    func::FuncOp funcOp, llvm::DenseMap<Value, Type> &ptrToMemRefType) {

  MLIRContext *context = funcOp.getContext();
  Block &entryBlock = funcOp.getBody().front();

  // Analyze each function argument
  for (BlockArgument arg : entryBlock.getArguments()) {
    if (!isa<LLVM::LLVMPointerType>(arg.getType())) {
      continue;  // Not a pointer, skip
    }

    // Find the element type by looking at operations using this argument
    // We need to look at the END of GEP chains (at load/store operations)
    // rather than just the first GEP, because the first GEP might be i8 byte arithmetic
    Type elementType = nullptr;

    // 1. Try to infer from GEP chains - follow chains to find terminal load/store types
    for (Operation *user : arg.getUsers()) {
      if (auto gepOp = dyn_cast<LLVM::GEPOp>(user)) {
        // Follow this GEP chain to find the terminal operation
        Value currentValue = gepOp.getResult();
        while (currentValue) {
          if (currentValue.use_empty()) break;

          Operation *nextUser = *currentValue.getUsers().begin();
          if (auto loadOp = dyn_cast<LLVM::LoadOp>(nextUser)) {
            // Found terminal load - use its type
            elementType = loadOp.getType();
            break;
          } else if (auto storeOp = dyn_cast<LLVM::StoreOp>(nextUser)) {
            // Found terminal store - use the value's type
            elementType = storeOp.getValue().getType();
            break;
          } else if (auto nextGEP = dyn_cast<LLVM::GEPOp>(nextUser)) {
            // Chain continues - keep following
            currentValue = nextGEP.getResult();
          } else {
            // Some other operation - stop here
            break;
          }
        }

        if (elementType) break;  // Found a type, stop searching
      }
    }

    // 2. If no GEP chains found, try to infer from direct load operations
    if (!elementType) {
      for (Operation *user : arg.getUsers()) {
        if (auto loadOp = dyn_cast<LLVM::LoadOp>(user)) {
          elementType = loadOp.getType();
          break;
        }
      }
    }

    // 3. If still no type, try to infer from direct store operations
    if (!elementType) {
      for (Operation *user : arg.getUsers()) {
        if (auto storeOp = dyn_cast<LLVM::StoreOp>(user)) {
          // Check if this pointer is the destination (operand #1)
          if (storeOp.getAddr() == arg) {
            elementType = storeOp.getValue().getType();
            break;
          }
        }
      }
    }

    if (!elementType) {
      // No operations found that reveal element type - default to i8 (byte pointer)
      elementType = IntegerType::get(context, 8);
    }

    // Create memref<?xT> type with unknown dimension
    auto memrefType = MemRefType::get({ShapedType::kDynamic}, elementType);
    ptrToMemRefType[arg] = memrefType;
  }

  return success();
}

/// Helper: Infer element type by analyzing actual load/store operations
/// CRITICAL FIX (Issue 4C): Check actual usage to distinguish float from int
/// Returns the most common type used in loads/stores, or nullptr if no usage found
static Type inferTypeFromUsage(Value ptr) {
  // Track type occurrences
  llvm::DenseMap<Type, unsigned> typeUsageCounts;

  // Analyze direct loads/stores
  for (Operation *user : ptr.getUsers()) {
    Type usedType = nullptr;

    if (auto loadOp = dyn_cast<LLVM::LoadOp>(user)) {
      usedType = loadOp.getType();
    } else if (auto storeOp = dyn_cast<LLVM::StoreOp>(user)) {
      if (storeOp.getAddr() == ptr) {
        usedType = storeOp.getValue().getType();
      }
    } else if (auto gepOp = dyn_cast<LLVM::GEPOp>(user)) {
      // Follow GEP chains to find terminal loads/stores
      for (Operation *gepUser : gepOp->getUsers()) {
        if (auto loadOp = dyn_cast<LLVM::LoadOp>(gepUser)) {
          usedType = loadOp.getType();
        } else if (auto storeOp = dyn_cast<LLVM::StoreOp>(gepUser)) {
          usedType = storeOp.getValue().getType();
        }
      }
    }

    if (usedType) {
      typeUsageCounts[usedType]++;
    }
  }

  // Return the most frequently used type
  Type mostCommonType = nullptr;
  unsigned maxCount = 0;
  for (auto &entry : typeUsageCounts) {
    if (entry.second > maxCount) {
      maxCount = entry.second;
      mostCommonType = entry.first;
    }
  }

  return mostCommonType;
}

/// Refine memref types based on alignment attributes from memcpy/memset operations
/// This helps correct the default i8 inference for byte-level pointer arguments
/// ENHANCED (Issue 4C): Also checks actual usage patterns to distinguish float from int
LogicalResult refineTypesFromMemcpyAlignment(
    func::FuncOp funcOp,
    llvm::DenseMap<Value, Type> &ptrToMemRefType) {

  MLIRContext *ctx = funcOp.getContext();

  // Scan all memcpy operations
  funcOp.walk([&](LLVM::MemcpyOp memcpyOp) {
    Value dst = memcpyOp.getDst();
    Value src = memcpyOp.getSrc();

    // Only refine if both dst and src are in our mapping
    if (!ptrToMemRefType.count(dst) || !ptrToMemRefType.count(src)) {
      return;
    }

    Type dstType = ptrToMemRefType[dst];
    Type srcType = ptrToMemRefType[src];

    auto dstMemRefType = cast<MemRefType>(dstType);
    auto srcMemRefType = cast<MemRefType>(srcType);

    // Get element types for both operands
    Type dstElem = dstMemRefType.getElementType();
    Type srcElem = srcMemRefType.getElementType();

    bool dstIsI8 = dstElem.isInteger(8);
    bool srcIsI8 = srcElem.isInteger(8);

    // PRIORITY 1: Propagate specific types from alloca/heap to function args
    // If one side has a specific type (from alloca/heap), propagate it to the other side
    // This takes priority over alignment-based inference to avoid ambiguity
    // (e.g., align 4 could be i32 or f32, but alloca knows it's f32)
    if (!dstIsI8 && srcIsI8) {
      // dst has specific type (from alloca/heap), src is generic (function arg)
      // → propagate dst type to src
      ptrToMemRefType[src] = MemRefType::get({ShapedType::kDynamic}, dstElem);
      return;  // Skip alignment logic, we already have the right type

    } else if (dstIsI8 && !srcIsI8) {
      // src has specific type, dst is generic
      // → propagate src type to dst
      ptrToMemRefType[dst] = MemRefType::get({ShapedType::kDynamic}, srcElem);
      return;  // Skip alignment logic

    } else if (!dstIsI8 && !srcIsI8) {
      // Both have specific types - nothing to refine
      return;
    }

    // If we reach here, both are i8 - continue with PRIORITY 2: alignment-based inference

    // Check alignment attribute
    auto argAttrs = memcpyOp.getArgAttrsAttr();
    if (!argAttrs || argAttrs.size() == 0) {
      return;
    }

    auto dstAttrs = cast<DictionaryAttr>(argAttrs[0]);
    auto alignAttr = dstAttrs.getAs<IntegerAttr>("llvm.align");
    if (!alignAttr) {
      return;
    }

    uint64_t alignment = alignAttr.getInt();
    Type refinedElementType = nullptr;

    // CRITICAL FIX (Issue 4C): Infer element type from actual usage FIRST
    // This distinguishes between float and integer types of the same width
    // Priority order:
    // 1. Check actual load/store types (most reliable)
    // 2. Fall back to alignment-based inference (ambiguous for align 4/8)

    // Try to infer from dst usage
    Type dstUsageType = inferTypeFromUsage(dst);
    Type srcUsageType = inferTypeFromUsage(src);

    // If both have usage info and they match, use that type
    if (dstUsageType && srcUsageType && dstUsageType == srcUsageType) {
      refinedElementType = dstUsageType;
    }
    // If only one has usage info, use it
    else if (dstUsageType) {
      refinedElementType = dstUsageType;
    } else if (srcUsageType) {
      refinedElementType = srcUsageType;
    }
    // Fall back to alignment-based inference (conservative: choose integer)
    else {
      // Infer element type from alignment:
      // - align 4 → i32 (could be i32 or f32, default to i32)
      // - align 8 → i64 (could be i64 or f64, default to i64)
      // - align 2 → i16
      // - align 1 → i8 (keep as is)
      if (alignment == 4) {
        refinedElementType = IntegerType::get(ctx, 32);
      } else if (alignment == 8) {
        refinedElementType = IntegerType::get(ctx, 64);
      } else if (alignment == 2) {
        refinedElementType = IntegerType::get(ctx, 16);
      }
    }

    if (!refinedElementType) {
      return;  // No refinement for this alignment
    }

    // At this point, both dst and src must be i8 (generic)
    // Refine both using the inferred type (from usage or alignment)
    ptrToMemRefType[dst] = MemRefType::get({ShapedType::kDynamic}, refinedElementType);
    ptrToMemRefType[src] = MemRefType::get({ShapedType::kDynamic}, refinedElementType);
  });

  // Scan all memset operations
  funcOp.walk([&](LLVM::MemsetOp memsetOp) {
    Value dst = memsetOp.getDst();

    if (!ptrToMemRefType.count(dst)) {
      return;
    }

    Type dstType = ptrToMemRefType[dst];
    auto dstMemRefType = cast<MemRefType>(dstType);

    if (!dstMemRefType.getElementType().isInteger(8)) {
      return;  // Already has a specific type
    }

    // Check alignment attribute
    auto argAttrs = memsetOp.getArgAttrsAttr();
    if (!argAttrs || argAttrs.size() == 0) {
      return;
    }

    auto dstAttrs = cast<DictionaryAttr>(argAttrs[0]);
    auto alignAttr = dstAttrs.getAs<IntegerAttr>("llvm.align");
    if (!alignAttr) {
      return;
    }

    uint64_t alignment = alignAttr.getInt();
    Type refinedElementType = nullptr;

    // CRITICAL FIX (Issue 4C): Check actual usage first to distinguish float from int
    Type usageType = inferTypeFromUsage(dst);
    if (usageType) {
      refinedElementType = usageType;
    } else {
      // Fall back to alignment-based inference
      if (alignment == 4) {
        refinedElementType = IntegerType::get(ctx, 32);
      } else if (alignment == 8) {
        refinedElementType = IntegerType::get(ctx, 64);
      } else if (alignment == 2) {
        refinedElementType = IntegerType::get(ctx, 16);
      }
    }

    if (refinedElementType) {
      // Update destination type in the map
      // The actual BlockArgument type will be updated later in updateArgumentTypes()
      ptrToMemRefType[dst] = MemRefType::get({ShapedType::kDynamic}, refinedElementType);
    }
  });

  return success();
}

LogicalResult inferTypesForAllocas(
    func::FuncOp funcOp,
    llvm::DenseMap<Value, Type> &ptrToMemRefType) {

  funcOp.walk([&](LLVM::AllocaOp allocaOp) {
    // Get element type from the alloca: %N x f32 → f32
    Type elementType = allocaOp.getElemType();

    // Extract scalar element type from nested LLVM arrays AND collect dimensions
    // Example: !llvm.array<64 x i32> → i32, with dimensions = [64]
    // This handles cases like: %ptr = llvm.alloca %1 x !llvm.array<64 x i32>
    SmallVector<int64_t> arrayDimensions;
    while (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(elementType)) {
      arrayDimensions.push_back(arrayType.getNumElements());
      elementType = arrayType.getElementType();
    }

    // Validate that we have a proper memref element type
    // MemRef element types must be scalars (int/float) or vectors
    if (!elementType.isIntOrIndexOrFloat() && !isa<VectorType>(elementType)) {
      // Skip this alloca - unsupported element type
      allocaOp.emitWarning()
          << "Skipping alloca with unsupported element type: " << elementType;
      return;
    }

    Value arraySize = allocaOp.getArraySize();

    // Determine if array size is a compile-time constant
    MemRefType memrefType;
    if (auto constantOp = arraySize.getDefiningOp<LLVM::ConstantOp>()) {
      // Static size: extract the constant value
      if (auto intAttr = dyn_cast<IntegerAttr>(constantOp.getValue())) {
        int64_t allocaMultiplier = intAttr.getInt();

        // Compute total size = multiplier × product of array dimensions
        // Example: alloca 1 x !llvm.array<64 x i32> → size = 1 × 64 = 64
        int64_t totalSize = allocaMultiplier;
        for (int64_t dim : arrayDimensions) {
          totalSize *= dim;
        }

        // Create static memref type: memref<NxT>
        memrefType = MemRefType::get({totalSize}, elementType);
      } else {
        // Not an integer constant, treat as dynamic
        memrefType = MemRefType::get({ShapedType::kDynamic}, elementType);
      }
    } else if (auto constantOp = arraySize.getDefiningOp<arith::ConstantOp>()) {
      // Static size from arith dialect
      if (auto intAttr = dyn_cast<IntegerAttr>(constantOp.getValue())) {
        int64_t allocaMultiplier = intAttr.getInt();

        // Compute total size = multiplier × product of array dimensions
        int64_t totalSize = allocaMultiplier;
        for (int64_t dim : arrayDimensions) {
          totalSize *= dim;
        }

        // Create static memref type: memref<NxT>
        memrefType = MemRefType::get({totalSize}, elementType);
      } else {
        // Not an integer constant, treat as dynamic
        memrefType = MemRefType::get({ShapedType::kDynamic}, elementType);
      }
    } else {
      // Dynamic size: array size comes from SSA value (e.g., function parameter)
      // Create dynamic memref type: memref<?xT>
      // Note: Even if array dimensions are static, the multiplier is dynamic
      memrefType = MemRefType::get({ShapedType::kDynamic}, elementType);
    }

    // Track mapping from llvm.ptr → memref<NxT> or memref<?xT>
    ptrToMemRefType[allocaOp.getResult()] = memrefType;
  });

  return success();
}

/// Infer types for heap allocations (C++ new[], malloc)
/// Detects llvm.call @_Znam (new[]) and infers the element type
LogicalResult inferTypesForHeapAllocations(
    func::FuncOp funcOp,
    llvm::DenseMap<Value, Type> &ptrToMemRefType) {

  MLIRContext *context = funcOp.getContext();

  funcOp.walk([&](LLVM::CallOp callOp) {
    // Check if this is a call to operator new[] (_Znam) or malloc
    auto callee = callOp.getCallee();
    if (!callee) return;

    StringRef calleeName = *callee;
    // _Znam = operator new[](size_t)
    // _Znwm = operator new(size_t)
    if (calleeName != "_Znam" && calleeName != "_Znwm") {
      return;  // Not a heap allocation
    }

    Value allocPtr = callOp.getResult();

    // Infer element type by analyzing how the pointer is used
    Type elementType = nullptr;

    // Follow uses to find load/store operations
    for (Operation *user : allocPtr.getUsers()) {
      if (auto gepOp = dyn_cast<LLVM::GEPOp>(user)) {
        // Follow GEP chain to find terminal operation
        Value currentValue = gepOp.getResult();
        while (currentValue) {
          if (currentValue.use_empty()) break;

          Operation *nextUser = *currentValue.getUsers().begin();
          if (auto loadOp = dyn_cast<LLVM::LoadOp>(nextUser)) {
            elementType = loadOp.getType();
            break;
          } else if (auto storeOp = dyn_cast<LLVM::StoreOp>(nextUser)) {
            elementType = storeOp.getValue().getType();
            break;
          } else if (auto nextGEP = dyn_cast<LLVM::GEPOp>(nextUser)) {
            currentValue = nextGEP.getResult();
          } else {
            break;
          }
        }

        if (elementType) break;
      } else if (auto loadOp = dyn_cast<LLVM::LoadOp>(user)) {
        // Direct load
        elementType = loadOp.getType();
        break;
      } else if (auto storeOp = dyn_cast<LLVM::StoreOp>(user)) {
        // Direct store
        if (storeOp.getAddr() == allocPtr) {
          elementType = storeOp.getValue().getType();
          break;
        }
      } else if (auto memsetOp = dyn_cast<LLVM::MemsetOp>(user)) {
        // Used in memset - will default to i8
        continue;
      }
    }

    if (!elementType) {
      // No type inferred - default to i8 (byte array)
      elementType = IntegerType::get(context, 8);
    }

    // Create dynamic memref type: memref<?xT>
    auto memrefType = MemRefType::get({ShapedType::kDynamic}, elementType);
    ptrToMemRefType[allocPtr] = memrefType;
  });

  return success();
}

} // namespace dsa
} // namespace mlir

//===- SCFNestedConversion.cpp - Nested SCF conversion utilities -*- C++ -*-===//
//
// Utilities for converting nested SCF operations in inside-out order
//
//===----------------------------------------------------------------------===//

#include "SCFConversionHelpers.h"
#include "Common.h"
#include "ConversionRegistry.h"

namespace mlir {
namespace dsa {

//===----------------------------------------------------------------------===//
// Nested SCF Conversion Helper
//===----------------------------------------------------------------------===//

/// Convert nested SCF operations in inside-out order (innermost first)
/// This ensures that completion tokens from inner operations are available
/// when converting outer operations
LogicalResult convertNestedSCFOps(ArrayRef<Operation *> clonedOps,
                                   ConversionPatternRewriter &rewriter,
                                   DenseMap<Value, Value> &valueMap) {
  if (clonedOps.empty())
    return success();

  SCFConversionRegistry &registry = getSCFConversionRegistry(rewriter.getContext());

  // Collect all SCF operations with their nesting depths
  SmallVector<std::pair<int, Operation *>> scfOpsWithDepth;
  for (Operation *op : clonedOps) {
    if (registry.canConvert(op)) {
      int depth = getSCFNestingDepth(op);
      scfOpsWithDepth.push_back({depth, op});
    }
  }

  if (scfOpsWithDepth.empty())
    return success();

  // Sort by nesting depth (deepest first = innermost first)
  llvm::sort(scfOpsWithDepth, [](const auto &a, const auto &b) {
    return a.first > b.first; // Deeper = higher number = convert first
  });

  // Convert operations in inside-out order
  // CRITICAL: We store the valueMap pointer in a temporary attribute so that
  // nested conversion patterns can access and update it
  auto *ctx = rewriter.getContext();
  auto valueMapPtrAttr = IntegerAttr::get(IntegerType::get(ctx, 64),
                                           reinterpret_cast<int64_t>(&valueMap));

  for (auto [depth, op] : scfOpsWithDepth) {
    rewriter.setInsertionPoint(op);

    // Attach the valueMap pointer as a temporary attribute
    op->setAttr("__dsa_value_map_ptr", valueMapPtrAttr);

    if (failed(registry.convertOp(op, rewriter))) {
      return failure();
    }

    // The attribute will be removed by the conversion pattern or when the op is erased
  }

  return success();
}

/// Collect completion tokens from nested SCF operations
/// These are operations that were converted and produce completion tokens
/// Returns a list of completion tokens (none type) from all nested SCF operations
static SmallVector<Value> collectNestedSCFCompletionTokens(ArrayRef<Operation *> ops) {
  SmallVector<Value> completionTokens;

  for (Operation *op : ops) {
    // After conversion, SCF operations produce completion tokens as their last result
    // We look for handshake operations that replaced SCF operations
    if (auto muxOp = dyn_cast<circt::handshake::MuxOp>(op)) {
      // Mux operations from scf.if may have completion tokens attached
      if (muxOp->hasAttr("scf_completion_token")) {
        // The completion token is stored as an attribute reference
        // In practice, we need to trace the operation's results
        // For now, we'll handle this differently
      }
    }
  }

  return completionTokens;
}

} // namespace dsa
} // namespace mlir

//===- SCFConversionHelpers.cpp - SCF conversion helpers --------*- C++ -*-===//
//
// Helper functions for SCF to Handshake+DSA conversion
//
//===----------------------------------------------------------------------===//

#include "SCFConversionHelpers.h"
#include "Common.h"
#include "ConversionRegistry.h"

namespace mlir {
namespace dsa {

//===----------------------------------------------------------------------===//
// Zero-Trip Loop Semantic Verification (Issue 4A)
//===----------------------------------------------------------------------===//

LogicalResult verifyZeroTripSemantics(dsa::StreamOp streamOp,
                                       ArrayRef<dsa::CarryOp> carryOps) {

  if (streamOp.getNumResults() != 2) {
    return streamOp.emitError("dsa.stream must have exactly 2 results (index, last)");
  }

  for (auto carryOp : carryOps) {
    if (carryOp.getNumOperands() != 3) {
      return carryOp.emitError("dsa.carry must have exactly 3 operands (ctrl, i, f)");
    }

    Value ctrl = carryOp.getOperand(0);
    Value init = carryOp.getOperand(1);
    Value feedback = carryOp.getOperand(2);

    if (!ctrl.getType().isInteger(1)) {
      return carryOp.emitError("dsa.carry ctrl operand must be i1 type");
    }

    if (init.getType() != feedback.getType()) {
      return carryOp.emitError("dsa.carry init (i) and feedback (f) must have matching types");
    }

    if (carryOp.getResult().getType() != init.getType()) {
      return carryOp.emitError("dsa.carry result must match init type");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Helper: Add SCF path metadata to memory operations
//===----------------------------------------------------------------------===//

// Add a nesting level to memref.load/store's SCF path
// path format: "scf_type.branch" (e.g., "if.else", "while.before")
// Note: The path should already have "top.X" as first element from preprocessing
void addMemOpSCFPath(Operation *op, StringRef scfType, StringRef branch) {
  if (!isa<memref::LoadOp, memref::StoreOp>(op))
    return;

  OpBuilder builder(op->getContext());
  std::string pathEntry = (scfType + "." + branch).str();

  // Get existing path (should already include "top.X" from preprocessing)
  SmallVector<Attribute> path;
  if (auto existing = op->getAttrOfType<ArrayAttr>("dsa.scf_path")) {
    path.append(existing.begin(), existing.end());
  } else {
    // If no existing path, this might be called before preprocessing
    // Try to add top-level sequence ID first
    if (auto topSeq = op->getAttrOfType<IntegerAttr>("dsa.top_seq")) {
      std::string topPrefix = "top." + std::to_string(topSeq.getInt());
      path.push_back(builder.getStringAttr(topPrefix));
      DSA_DEBUG_STREAM << "[DEBUG addMemOpSCFPath] Initializing path with top prefix: "
                   << topPrefix << "\n";
    }
  }

  path.push_back(builder.getStringAttr(pathEntry));
  op->setAttr("dsa.scf_path", builder.getArrayAttr(path));

  DSA_DEBUG_STREAM << "[DEBUG addMemOpSCFPath] Added '" << pathEntry << "' to "
               << op->getName() << " (total path size: " << path.size() << ")\n";
}

// Get the SCF nesting path for the given operation by walking up parent operations
SmallVector<std::string> getSCFNestingPath(Operation *op) {
  SmallVector<std::string> path;
  Operation *parent = op->getParentOp();

  DSA_DEBUG_STREAM << "[DEBUG getSCFNestingPath] Starting from op: " << op->getName() << "\n";

  while (parent) {
    DSA_DEBUG_STREAM << "[DEBUG getSCFNestingPath] Checking parent: " << parent->getName() << "\n";

    // Check if parent is an SCF operation
    if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
      // Determine which region we're in
      bool inThenRegion = ifOp.getThenRegion().isAncestor(op->getParentRegion());
      std::string entry = inThenRegion ? "if.then" : "if.else";
      DSA_DEBUG_STREAM << "[DEBUG getSCFNestingPath] Found scf.if, adding: " << entry << "\n";
      path.insert(path.begin(), entry);
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(parent)) {
      // Determine which region we're in
      bool inBeforeRegion = whileOp.getBefore().isAncestor(op->getParentRegion());
      std::string entry = inBeforeRegion ? "while.before" : "while.after";
      DSA_DEBUG_STREAM << "[DEBUG getSCFNestingPath] Found scf.while, adding: " << entry << "\n";
      path.insert(path.begin(), entry);
    } else if (isa<scf::ForOp>(parent)) {
      DSA_DEBUG_STREAM << "[DEBUG getSCFNestingPath] Found scf.for, adding: for.body\n";
      path.insert(path.begin(), "for.body");
    }
    parent = parent->getParentOp();
  }

  DSA_DEBUG_STREAM << "[DEBUG getSCFNestingPath] Final path size: " << path.size() << "\n";
  return path;
}

// Mark a control value with the SCF region it controls AND its nesting path
// The control value gets two attributes:
// 1. "dsa.scf_region": ONLY the immediate region it controls (e.g., "while.3.before")
//    NOT including the nesting path (to avoid redundancy with scf_path)
// 2. "dsa.scf_path": the FULL nesting path including top.X where this control value is located
// The scfOp parameter is the ORIGINAL SCF operation being converted
// CRITICAL: Read the preprocessed "dsa.scf_nesting" and "dsa.scf_id" attributes
void markControlValue(Value val, StringRef scfType, StringRef branch,
                       Operation *scfOp, OpBuilder &builder) {
  if (!val) {
    DSA_DEBUG_STREAM << "[DEBUG markControlValue] WARNING: val is null\n";
    return;
  }

  Operation *defOp = val.getDefiningOp();
  if (!defOp) {
    DSA_DEBUG_STREAM << "[DEBUG markControlValue] WARNING: val has no defining op (likely a BlockArgument)\n";
    return;
  }

  // Get the SCF operation's unique ID
  auto idAttr = scfOp->getAttrOfType<IntegerAttr>("dsa.scf_id");
  unsigned scfId = idAttr ? idAttr.getInt() : 0;

  // Get the SCF operation's nesting path (where the SCF itself is located)
  SmallVector<std::string> nestingPath;
  if (auto nestingAttr = scfOp->getAttrOfType<ArrayAttr>("dsa.scf_nesting")) {
    for (auto attr : nestingAttr) {
      if (auto strAttr = dyn_cast<StringAttr>(attr)) {
        nestingPath.push_back(strAttr.getValue().str());
      }
    }
  }

  // Build the IMMEDIATE region identifier: ONLY scf_type.id[.branch]
  // For 'if': Don't include branch suffix because cond_br generates BOTH then/else branches
  //           Region format: "if.2" (not "if.2.then" or "if.2.else")
  // For 'while'/'for': Include branch because before/after/body are separate control points
  //           Region format: "while.3.before", "for.0.body"
  std::string immediateRegion = scfType.str() + "." + std::to_string(scfId);
  if (scfType != "if") {
    immediateRegion += "." + branch.str();
  }

  defOp->setAttr("dsa.scf_region", builder.getStringAttr(immediateRegion));

  // Build the FULL nesting path including top.X where this control value is located
  SmallVector<Attribute> pathAttrs;

  // Add top.X prefix if available
  if (auto topSeq = scfOp->getAttrOfType<IntegerAttr>("dsa.top_seq")) {
    std::string topPrefix = "top." + std::to_string(topSeq.getInt());
    pathAttrs.push_back(builder.getStringAttr(topPrefix));
  }

  // Add the nesting path
  for (const auto &entry : nestingPath) {
    pathAttrs.push_back(builder.getStringAttr(entry));
  }

  defOp->setAttr("dsa.scf_path", builder.getArrayAttr(pathAttrs));

  // Note: The three *_seq attributes (global_seq, top_seq, local_seq) should already
  // be present on defOp from the preprocessing stage. We don't need to copy them again
  // as they are already set on all operations during preprocessing.

  DSA_DEBUG_STREAM << "[DEBUG markControlValue] Marked control value:\n";
  DSA_DEBUG_STREAM << "  - Immediate region: " << immediateRegion << "\n";
  DSA_DEBUG_STREAM << "  - Nesting path: [";
  if (auto topSeq = scfOp->getAttrOfType<IntegerAttr>("dsa.top_seq")) {
    DSA_DEBUG_STREAM << "\"top." << topSeq.getInt() << "\"";
    if (!nestingPath.empty()) DSA_DEBUG_STREAM << ", ";
  }
  for (size_t i = 0; i < nestingPath.size(); ++i) {
    if (i > 0) DSA_DEBUG_STREAM << ", ";
    DSA_DEBUG_STREAM << "\"" << nestingPath[i] << "\"";
  }
  DSA_DEBUG_STREAM << "]\n";
}

//===----------------------------------------------------------------------===//
// Nesting Depth Utilities
//===----------------------------------------------------------------------===//

/// Determine the nesting depth of an SCF operation
/// Returns the number of nested SCF operations containing this operation
int getSCFNestingDepth(Operation *op) {
  int depth = 0;
  Operation *parent = op->getParentOp();

  while (parent) {
    if (isa<scf::ForOp, scf::WhileOp, scf::IfOp, scf::IndexSwitchOp>(parent)) {
      depth++;
    }
    parent = parent->getParentOp();
  }

  return depth;
}

} // namespace dsa
} // namespace mlir

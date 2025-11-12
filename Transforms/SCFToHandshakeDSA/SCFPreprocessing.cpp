//===- SCFPreprocessing.cpp - SCF metadata preprocessing -------*- C++ -*-===//
//
// Implementation of SCF preprocessing to compute nesting paths
//
//===----------------------------------------------------------------------===//

#include "SCFPreprocessing.h"

namespace mlir {
namespace dsa {

//===----------------------------------------------------------------------===//
// Helper: Compute nesting path for an operation
//===----------------------------------------------------------------------===//

static SmallVector<std::string> computeNestingPath(Operation *op) {
  SmallVector<std::string> path;
  Operation *parent = op->getParentOp();

  while (parent && !isa<func::FuncOp>(parent)) {
    if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
      // Get unique ID from parent SCF operation
      auto idAttr = ifOp->getAttrOfType<IntegerAttr>("dsa.scf_id");
      unsigned id = idAttr ? idAttr.getInt() : 0;

      Region *opRegion = op->getParentRegion();
      bool inThenRegion = ifOp.getThenRegion().isAncestor(opRegion);

      std::string entry = "if." + std::to_string(id) + (inThenRegion ? ".then" : ".else");
      path.insert(path.begin(), entry);
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(parent)) {
      auto idAttr = whileOp->getAttrOfType<IntegerAttr>("dsa.scf_id");
      unsigned id = idAttr ? idAttr.getInt() : 0;

      Region *opRegion = op->getParentRegion();
      bool inBeforeRegion = whileOp.getBefore().isAncestor(opRegion);

      std::string entry = "while." + std::to_string(id) + (inBeforeRegion ? ".before" : ".after");
      path.insert(path.begin(), entry);
    } else if (isa<scf::ForOp>(parent)) {
      auto idAttr = parent->getAttrOfType<IntegerAttr>("dsa.scf_id");
      unsigned id = idAttr ? idAttr.getInt() : 0;

      std::string entry = "for." + std::to_string(id) + ".body";
      path.insert(path.begin(), entry);
    }
    parent = parent->getParentOp();
  }

  return path;
}

//===----------------------------------------------------------------------===//
// Main Preprocessing Function
//===----------------------------------------------------------------------===//

void preprocessSCFMetadata(ModuleOp module, OpBuilder &builder) {
  module.walk([&](func::FuncOp funcOp) {
    // STEP 1: Assign unique IDs to all SCF operations at each nesting level
    // This allows us to distinguish between multiple SCF ops at the same level
    llvm::DenseMap<Operation*, unsigned> scfIdMap;
    unsigned nextId = 0;

    funcOp.walk([&](Operation *op) {
      if (isa<scf::IfOp, scf::WhileOp, scf::ForOp>(op)) {
        op->setAttr("dsa.scf_id", builder.getI32IntegerAttr(nextId));
        scfIdMap[op] = nextId;

        DSA_DEBUG_STREAM << "[DEBUG PreProcess] Assigned ID " << nextId
                         << " to " << op->getName() << "\n";
        nextId++;
      }
    });

    // STEP 1.5: Assign three types of sequence IDs
    // - top_seq: sequence within the entry block (top-level operations only)
    // - global_seq: global depth-first traversal order
    // - local_seq: sequence within the immediate parent block

    Block &entryBlock = funcOp.getBody().front();
    unsigned topLevelSeq = 0;
    unsigned globalSeq = 0;

    // First pass: assign top_seq and global_seq using depth-first traversal
    for (Operation &op : entryBlock) {
      // Assign top-level sequence number to this operation
      op.setAttr("dsa.top_seq", builder.getI32IntegerAttr(topLevelSeq));

      // Assign global sequence number
      op.setAttr("dsa.global_seq", builder.getI32IntegerAttr(globalSeq));
      globalSeq++;

      DSA_DEBUG_STREAM << "[DEBUG PreProcess] Assigned top_seq " << topLevelSeq
                       << ", global_seq " << (globalSeq - 1)
                       << " to top-level " << op.getName() << "\n";

      // Propagate the same top_seq to all nested operations (depth-first)
      // and assign global_seq to each nested operation
      op.walk([&](Operation *nestedOp) {
        if (nestedOp != &op) {
          if (!nestedOp->hasAttr("dsa.top_seq")) {
            nestedOp->setAttr("dsa.top_seq", builder.getI32IntegerAttr(topLevelSeq));
          }
          if (!nestedOp->hasAttr("dsa.global_seq")) {
            nestedOp->setAttr("dsa.global_seq", builder.getI32IntegerAttr(globalSeq));
            globalSeq++;
          }
        }
      });

      topLevelSeq++;
    }

    // Second pass: assign local_seq (sequence within immediate parent block)
    funcOp.walk([&](Block *block) {
      unsigned localSeq = 0;
      for (Operation &op : *block) {
        op.setAttr("dsa.local_seq", builder.getI32IntegerAttr(localSeq));
        localSeq++;
      }
    });

    // STEP 2: Mark all memref operations with their nesting path (using IDs)
    // Now includes top-level sequence as first element
    funcOp.walk([&](Operation *op) {
      if (isa<memref::LoadOp, memref::StoreOp>(op)) {
        SmallVector<std::string> nestingPath = computeNestingPath(op);
        SmallVector<Attribute> pathAttrs;

        // First, add the top-level sequence ID
        if (auto topSeq = op->getAttrOfType<IntegerAttr>("dsa.top_seq")) {
          std::string topPrefix = "top." + std::to_string(topSeq.getInt());
          pathAttrs.push_back(builder.getStringAttr(topPrefix));
          DSA_DEBUG_STREAM << "[DEBUG PreProcess] Adding top prefix: " << topPrefix << "\n";
        }

        // Then add the SCF nesting path
        for (const auto &entry : nestingPath) {
          pathAttrs.push_back(builder.getStringAttr(entry));
        }

        op->setAttr("dsa.scf_path", builder.getArrayAttr(pathAttrs));
      }
    });

    // STEP 3: Mark all SCF operations with their nesting path (using IDs)
    // This allows us to later identify which control value controls which nested SCF region
    funcOp.walk([&](Operation *op) {
      if (isa<scf::IfOp, scf::WhileOp, scf::ForOp>(op)) {
        SmallVector<std::string> nestingPath = computeNestingPath(op);
        SmallVector<Attribute> pathAttrs;
        for (const auto &entry : nestingPath) {
          pathAttrs.push_back(builder.getStringAttr(entry));
        }
        op->setAttr("dsa.scf_nesting", builder.getArrayAttr(pathAttrs));

        // Debug output
        DSA_DEBUG_STREAM << "[DEBUG PreProcess] Marked " << op->getName()
                         << " (ID=" << scfIdMap[op] << ") at path: [";
        for (size_t i = 0; i < nestingPath.size(); ++i) {
          if (i > 0) DSA_DEBUG_STREAM << ", ";
          DSA_DEBUG_STREAM << "\"" << nestingPath[i] << "\"";
        }
        DSA_DEBUG_STREAM << "]\n";
      }
    });
  });
}

} // namespace dsa
} // namespace mlir

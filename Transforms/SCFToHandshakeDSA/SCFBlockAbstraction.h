//===- SCFBlockAbstraction.h - SCF Block Abstraction -----------*- C++ -*-===//
//
// SCF block abstraction for memory dependency analysis
//
//===----------------------------------------------------------------------===//

#ifndef DSA_TRANSFORMS_SCFTOHANDSHAKE_SCFBLOCKABSTRACTION_H
#define DSA_TRANSFORMS_SCFTOHANDSHAKE_SCFBLOCKABSTRACTION_H

#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <memory>

namespace mlir {
namespace dsa {

/// Represents a memory operation or an abstracted SCF block at a given nesting level
struct MemoryElement {
  enum Kind {
    MEMORY_OP,    // Actual load/store operation
    SCF_BLOCK     // Abstracted SCF block (if/for/while)
  };

  Kind kind;

  // For MEMORY_OP
  Operation *memOp = nullptr;
  bool isLoad = false;
  bool isStore = false;

  // For SCF_BLOCK
  std::string scfRegion;        // e.g., "if.4.else.while.3.before.if.1"
  std::string scfType;          // "if", "for", or "while"
  Value condition = nullptr;    // Condition value for the SCF block
  bool hasReads = false;        // Does this block (recursively) contain reads?
  bool hasWrites = false;       // Does this block (recursively) contain writes?

  // Done token for this element
  // For MEMORY_OP: extmemory done token
  // For SCF_BLOCK: unconditional done token (from mux)
  Value doneToken = nullptr;

  // For SCF_BLOCK: done tokens from then/else branches before mux
  Value thenDoneToken = nullptr;
  Value elseDoneToken = nullptr;

  // Nesting path (for both kinds)
  SmallVector<std::string> nestingPath;

  // Constructor for memory operation
  static MemoryElement createMemoryOp(Operation *op, bool isLoad,
                                       ArrayRef<std::string> path) {
    MemoryElement elem;
    elem.kind = MEMORY_OP;
    elem.memOp = op;
    elem.isLoad = isLoad;
    elem.isStore = !isLoad;
    elem.nestingPath.assign(path.begin(), path.end());
    return elem;
  }

  // Constructor for SCF block
  static MemoryElement createSCFBlock(StringRef region, StringRef type,
                                       Value cond, bool reads, bool writes,
                                       ArrayRef<std::string> path) {
    MemoryElement elem;
    elem.kind = SCF_BLOCK;
    elem.scfRegion = region.str();
    elem.scfType = type.str();
    elem.condition = cond;
    elem.hasReads = reads;
    elem.hasWrites = writes;
    elem.nestingPath.assign(path.begin(), path.end());
    return elem;
  }

  bool isMemoryOp() const { return kind == MEMORY_OP; }
  bool isSCFBlock() const { return kind == SCF_BLOCK; }

  bool canRead() const {
    return (isMemoryOp() && isLoad) || (isSCFBlock() && hasReads);
  }

  bool canWrite() const {
    return (isMemoryOp() && isStore) || (isSCFBlock() && hasWrites);
  }

  std::string getDescription() const {
    if (isMemoryOp()) {
      return isLoad ? "LOAD" : "STORE";
    } else {
      std::string attrs = hasReads ? "R" : "-";
      attrs += hasWrites ? "W" : "-";
      return "SCF." + scfType + "{" + attrs + "}";
    }
  }
};

/// Represents a level in the SCF nesting hierarchy
struct NestingLevel {
  SmallVector<std::string> path;  // Path to this level
  SmallVector<MemoryElement> elements;  // Operations and abstracted sub-blocks

  std::string getPathString() const {
    std::string result;
    for (const auto &p : path) {
      if (!result.empty()) result += ".";
      result += p;
    }
    return result;
  }
};

} // namespace dsa
} // namespace mlir

#endif // DSA_TRANSFORMS_SCFTOHANDSHAKE_SCFBLOCKABSTRACTION_H

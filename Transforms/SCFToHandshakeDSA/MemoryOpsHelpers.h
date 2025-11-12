//===- MemoryOpsHelpers.h - Memory operation helper functions -*- C++ -*-===//
//
// Helper functions and utilities for memory operation conversion
//
//===----------------------------------------------------------------------===//

#ifndef LIB_DSA_TRANSFORMS_SCFTOHANDSHAKEDSA_MEMORYOPSHELPERS_H
#define LIB_DSA_TRANSFORMS_SCFTOHANDSHAKEDSA_MEMORYOPSHELPERS_H

#include "Common.h"

namespace mlir {
namespace dsa {

//===----------------------------------------------------------------------===//
// Basic Memory Operation Helpers
//===----------------------------------------------------------------------===//

bool isMemoryOp(Operation *op);
bool isAllocOp(Operation *op);
LogicalResult getOpMemRef(Operation *op, Value &out);
LogicalResult isValidMemrefType(Location loc, MemRefType type);
SmallVector<Value> getResultsToMemory(Operation *op);
Value getOriginalMemRef(Value memref);
bool hasDynamicDimensions(MemRefType memrefType);
void addValueToOperands(Operation *op, Value val);

//===----------------------------------------------------------------------===//
// SCF Path Helpers
//===----------------------------------------------------------------------===//

SmallVector<std::string> getScfPath(Operation *op);

struct ScfLevel {
  std::string type;       // "if", "while", "for"
  std::string id;         // "1", "2", etc.
  std::string branch;     // "then", "else", "before", "after"
  std::string region;     // "if.1", "while.2", etc.
  std::string fullRegion; // "if.1.then", "while.2.before", etc.
  bool valid = false;

  static ScfLevel parse(const std::string &levelStr);
};

//===----------------------------------------------------------------------===//
// Control Value Finding
//===----------------------------------------------------------------------===//

template <typename FuncOpT>
Value findControlValue(FuncOpT funcOp, ArrayRef<std::string> parentPath,
                       StringRef scfRegion);

//===----------------------------------------------------------------------===//
// Extmemory Token Helpers
//===----------------------------------------------------------------------===//

Value getExtmemoryDoneToken(Operation *memOp);

//===----------------------------------------------------------------------===//
// Path Matching Helpers
//===----------------------------------------------------------------------===//

bool pathMatchesPrefix(ArrayRef<std::string> path,
                       ArrayRef<std::string> prefix);

} // namespace dsa
} // namespace mlir

#endif // LIB_DSA_TRANSFORMS_SCFTOHANDSHAKEDSA_MEMORYOPSHELPERS_H

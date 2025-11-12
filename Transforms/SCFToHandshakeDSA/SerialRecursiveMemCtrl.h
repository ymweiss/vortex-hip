//===- SerialRecursiveMemCtrl.h - Serial-recursive memory control -*- C++ -*-===//
//
// Serial-recursive algorithm for memory operation control flow
//
//===----------------------------------------------------------------------===//

#ifndef LIB_DSA_TRANSFORMS_SCFTOHANDSHAKEDSA_SERIALRECURSIVEMEMCTRL_H
#define LIB_DSA_TRANSFORMS_SCFTOHANDSHAKEDSA_SERIALRECURSIVEMEMCTRL_H

#include "Common.h"
#include "MemoryOpsHelpers.h"
#include <map>

namespace mlir {
namespace dsa {

//===----------------------------------------------------------------------===//
// Serial-Recursive Memory Control Algorithm
//
// Recursively processes memory operations level by level:
// - Each level receives an entry token, returns a done token
// - SCF blocks are abstracted as entry->done
// - RAR parallelism: consecutive loads can fork/join
//===----------------------------------------------------------------------===//

template <typename FuncOpT, typename RewriterT>
class SerialRecursiveMemCtrl {
public:
  SerialRecursiveMemCtrl(FuncOpT funcOp, RewriterT &rewriter,
                         ArrayRef<Operation *> memOps);

  // Main entry point: process all operations from top level
  Value processTopLevel(Value entryToken);

private:
  FuncOpT funcOp;
  RewriterT &rewriter;
  ArrayRef<Operation *> allMemOps;
  size_t currentOpIdx; // Current position in allMemOps

  // Cache for loop-carry tokens: loop_region -> carry_op
  std::map<std::string, Value> loopCarryCache;

  // Check if there are any operations in this path (or child paths)
  bool hasOpsInPath(ArrayRef<std::string> targetPath) const;

  // Process all operations at a given level
  // Returns the done token for this level
  Value processLevel(ArrayRef<std::string> currentPath, Value currentToken);

  // Enter a child SCF block and process it
  Value enterAndProcessChild(ArrayRef<std::string> parentPath,
                              Value parentToken);

  // Process a while/for loop block
  Value processLoopBlock(ArrayRef<std::string> parentPath,
                          Value parentToken,
                          const ScfLevel &level);

  // Process an if block
  Value processIfBlock(ArrayRef<std::string> parentPath,
                        Value parentToken,
                        const ScfLevel &level);

  // Process a single memory operation
  Value processSingleOp(Operation *op, Value entryToken);

  // Process consecutive loads with RAR parallelism (fork/join)
  Value processConsecutiveLoads(ArrayRef<std::string> currentPath,
                                 Value entryToken,
                                 ArrayRef<size_t> loadIndices);
};

} // namespace dsa
} // namespace mlir

#endif // LIB_DSA_TRANSFORMS_SCFTOHANDSHAKEDSA_SERIALRECURSIVEMEMCTRL_H

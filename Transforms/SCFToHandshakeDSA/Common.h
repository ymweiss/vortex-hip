//===- Common.h - Shared declarations for SCFToHandshakeDSA ----*- C++ -*-===//
//
// Common types, forward declarations and utilities for SCFToHandshakeDSA
//
//===----------------------------------------------------------------------===//

#ifndef LIB_DSA_TRANSFORMS_SCFTOHANDSHAKEDSA_COMMON_H
#define LIB_DSA_TRANSFORMS_SCFTOHANDSHAKEDSA_COMMON_H

#include "dsa/Analysis/OperationID.h"
#include "dsa/Dialect/DSA/DSADialect.h"
#include "dsa/Dialect/DSA/DSAOps.h"
#include "dsa/Transforms/Passes.h"

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <cstdlib>

//===----------------------------------------------------------------------===//
// Debug Utilities
//===----------------------------------------------------------------------===//

// Macro to wrap debug output with DSA_DEBUG environment variable check
// Usage: DSA_DEBUG_STREAM << "debug message" << value << "\n";
#define DSA_DEBUG_STREAM \
  if (const char *dsaDebugEnv = std::getenv("DSA_DEBUG"); \
      dsaDebugEnv && std::string(dsaDebugEnv) == "1") \
    llvm::errs()

namespace mlir {
namespace dsa {

//===----------------------------------------------------------------------===//
// Type Aliases
//===----------------------------------------------------------------------===//

// Map from memref values to their memory access operations (loads/stores)
// Operations are stored in program order
using MemRefToMemoryAccessOp = DenseMap<Value, std::vector<Operation *>>;

//===----------------------------------------------------------------------===//
// Forward Declarations - Memory Operations
//===----------------------------------------------------------------------===//

template <typename FuncOpType, typename RewriterT>
LogicalResult replaceMemoryOps(FuncOpType funcOp,
                                RewriterT &rewriter,
                                MemRefToMemoryAccessOp &memRefOps);

template <typename FuncOpType, typename RewriterT>
LogicalResult connectToMemory(FuncOpType funcOp, RewriterT &rewriter,
                               MemRefToMemoryAccessOp &memRefOps);

//===----------------------------------------------------------------------===//
// Forward Declarations - Pattern Population Functions
//===----------------------------------------------------------------------===//

void populateFuncConversionPatterns(RewritePatternSet &patterns);
void populateLoopConversionPatterns(RewritePatternSet &patterns);
void populateControlFlowConversionPatterns(RewritePatternSet &patterns);
void populateParallelLoopConversionPatterns(RewritePatternSet &patterns);

//===----------------------------------------------------------------------===//
// Forward Declarations - Nested SCF Operation Conversion
//===----------------------------------------------------------------------===//

LogicalResult convertNestedSCFOps(ArrayRef<Operation *> clonedOps,
                                   ConversionPatternRewriter &rewriter,
                                   DenseMap<Value, Value> &valueMap);

//===----------------------------------------------------------------------===//
// Forward Declarations - Conversion Registry
//===----------------------------------------------------------------------===//

class SCFConversionRegistry;
SCFConversionRegistry &getSCFConversionRegistry(MLIRContext *context);
void initializeSCFConversionRegistry(MLIRContext *context);

//===----------------------------------------------------------------------===//
// Forward Declarations - Pass Helper Functions
//===----------------------------------------------------------------------===//

template <typename RewriterT>
LogicalResult sinkUnconsumedValues(circt::handshake::FuncOp funcOp,
                                    RewriterT &rewriter);

template <typename RewriterT>
LogicalResult eliminateDeadCode(circt::handshake::FuncOp funcOp,
                                 RewriterT &rewriter);

template <typename RewriterT>
LogicalResult optimizeForkOperations(circt::handshake::FuncOp funcOp,
                                      RewriterT &rewriter);

template <typename RewriterT>
LogicalResult coordinateArgsWithEntryToken(circt::handshake::FuncOp funcOp,
                                             RewriterT &rewriter);

template <typename RewriterT>
LogicalResult insertForkOperations(circt::handshake::FuncOp funcOp,
                                     RewriterT &rewriter);

template <typename RewriterT>
LogicalResult optimizeHandshakeDSA(circt::handshake::FuncOp funcOp,
                                    RewriterT &rewriter);

Value traceMemRefOrigin(Value memref);

// Find loop exit control signal for the entire function (returns first loop found)
Value findLoopExitControl(circt::handshake::FuncOp funcOp);

// Find loop exit control signal for a specific context operation (returns nearest enclosing loop)
// CRITICAL FIX (Issue 5A): Use this version to properly handle nested loops
Value findLoopExitControl(circt::handshake::FuncOp funcOp, Operation *contextOp);

// Find loop iteration control signal for memory operations in loops
// CRITICAL FIX (Issue 6A): Memory operations in loops need per-iteration control tokens
// Returns the is_last signal (last for scf.for, isLast=!condition for scf.while)
// Control semantics: is_last=false (continue), is_last=true (exit)
// or nullptr if not in a loop
Value findLoopIterationControl(circt::handshake::FuncOp funcOp, Operation *contextOp);

// Get the extmemory completion token for a memory operation
// Returns {extmemory_operation, completion_token} or {nullptr, nullptr} if not found
std::pair<Operation*, Value> getExtmemoryCompletionToken(Operation *memOp);

// Route a done token from predecessor to current memory operation
// Handles routing through nested SCF structures
Value routeDoneTokenToMemOp(Value doneToken, Operation *predOp,
                             Operation *currOp, Operation *funcOp,
                             OpBuilder &rewriter);

// Route entry token to a memory operation
// Handles routing through nested SCF structures
Value routeEntryTokenToMemOp(Value entryToken, Operation *memOp,
                              Operation *funcOp, OpBuilder &rewriter);

} // namespace dsa
} // namespace mlir

#endif // LIB_DSA_TRANSFORMS_SCFTOHANDSHAKEDSA_COMMON_H

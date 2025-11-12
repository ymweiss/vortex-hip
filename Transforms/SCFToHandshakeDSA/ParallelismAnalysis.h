//===- ParallelismAnalysis.h - Loop parallelism analysis -------*- C++ -*-===//
//
// Analysis for detecting independent loop iterations and dataflow parallelism.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_DSA_TRANSFORMS_SCFTOHANDSHAKEDSA_PARALLELISMANALYSIS_H
#define LIB_DSA_TRANSFORMS_SCFTOHANDSHAKEDSA_PARALLELISMANALYSIS_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace dsa {

//===----------------------------------------------------------------------===//
// ParallelismAnalysis Class
//===----------------------------------------------------------------------===//

/// Analysis for detecting parallelizable loops.
///
/// This class analyzes SCF loops (for, while, forall) to determine:
/// - Whether iterations are independent (no loop-carried dependencies)
/// - Potential memory conflicts that prevent parallelization
/// - Estimated degree of parallelism
///
/// Usage:
///   ParallelismAnalysis analysis(forOp);
///   if (analysis.hasIndependentIterations()) {
///     unsigned degree = analysis.estimateParallelismDegree();
///     // Convert to parallel execution with 'degree' parallel instances
///   }
///
class ParallelismAnalysis {
public:
  /// Construct analysis for a loop operation.
  /// Supports scf.for, scf.while, and scf.forall.
  explicit ParallelismAnalysis(Operation *loopOp);

  /// Check if loop iterations are independent.
  /// Returns true if there are no loop-carried dependencies that prevent
  /// parallel execution.
  bool hasIndependentIterations() const { return independentIterations; }

  /// Get the set of values with loop-carried dependencies.
  /// These are values that are updated in one iteration and used in the next.
  const DenseSet<Value> &getLoopCarriedDependencies() const {
    return loopCarriedValues;
  }

  /// Estimate the degree of parallelism.
  /// Returns the number of parallel instances that could be created.
  /// Returns 1 if loop is not parallelizable.
  unsigned estimateParallelismDegree() const;

  /// Check if memory operations prevent parallelization.
  /// Returns true if there are potential memory conflicts (e.g., aliasing).
  bool hasMemoryConflicts() const { return memoryConflicts; }

  /// Check if the loop can exploit parallelism.
  /// This is a high-level check that combines independent iterations
  /// and absence of memory conflicts.
  bool canExploitParallelism() const {
    return independentIterations && !memoryConflicts;
  }

  /// Get detailed analysis results as a string (for debugging).
  std::string getAnalysisReport() const;

private:
  /// Analyze dataflow within the loop.
  void analyzeDataflow();

  /// Analyze memory accesses for potential conflicts.
  void analyzeMemoryAccesses();

  /// Build dependence graph between operations.
  void buildDependenceGraph();

  /// Check if a value is loop-invariant (defined outside, used inside).
  bool isLoopInvariant(Value value) const;

  /// Check if two memory operations may conflict.
  bool mayConflict(Operation *op1, Operation *op2) const;

  /// Get the trip count if statically known, otherwise returns nullopt.
  std::optional<int64_t> getStaticTripCount() const;

  //===--------------------------------------------------------------------===//
  // Analysis State
  //===--------------------------------------------------------------------===//

  /// The loop operation being analyzed
  Operation *loop;

  /// Whether iterations are independent
  bool independentIterations = true;

  /// Whether memory conflicts exist
  bool memoryConflicts = false;

  /// Set of values with loop-carried dependencies
  DenseSet<Value> loopCarriedValues;

  /// Map from values to their uses within the loop
  DenseMap<Value, SmallVector<Operation *>> defUseChains;

  /// Memory operations in the loop (loads, stores)
  SmallVector<Operation *> memoryOps;
};

} // namespace dsa
} // namespace mlir

#endif // LIB_DSA_TRANSFORMS_SCFTOHANDSHAKEDSA_PARALLELISMANALYSIS_H

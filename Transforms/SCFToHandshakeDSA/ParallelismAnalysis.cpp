//===- ParallelismAnalysis.cpp - Loop parallelism analysis -----*- C++ -*-===//
//
// Implementation of loop parallelism analysis.
//
//===----------------------------------------------------------------------===//

#include "ParallelismAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "parallelism-analysis"

namespace mlir {
namespace dsa {

//===----------------------------------------------------------------------===//
// Constructor and Main Analysis Entry Point
//===----------------------------------------------------------------------===//

ParallelismAnalysis::ParallelismAnalysis(Operation *loopOp) : loop(loopOp) {
  // Perform analysis
  analyzeDataflow();
  analyzeMemoryAccesses();
  buildDependenceGraph();
}

//===----------------------------------------------------------------------===//
// Dataflow Analysis
//===----------------------------------------------------------------------===//

void ParallelismAnalysis::analyzeDataflow() {
  // Get the loop body region
  Region *bodyRegion = nullptr;

  if (auto forOp = dyn_cast<scf::ForOp>(loop)) {
    bodyRegion = &forOp.getRegion();

    // Check for iter_args (loop-carried dependencies)
    if (!forOp.getInitArgs().empty()) {
      // Has iter_args - analyze if they create true dependencies
      for (auto [initArg, regionArg, result] :
           llvm::zip(forOp.getInitArgs(), forOp.getRegionIterArgs(), forOp.getResults())) {

        // If the region argument is used in a way that creates a dependency
        // across iterations, mark as loop-carried
        bool usedInBody = false;
        for (auto user : regionArg.getUsers()) {
          if (user->getParentRegion() == bodyRegion) {
            usedInBody = true;
            break;
          }
        }

        if (usedInBody) {
          loopCarriedValues.insert(regionArg);
          independentIterations = false;

          LLVM_DEBUG(llvm::dbgs() << "Found loop-carried dependency: "
                                  << regionArg << "\n");
        }
      }
    }
  } else if (auto whileOp = dyn_cast<scf::WhileOp>(loop)) {
    // While loops always have loop-carried dependencies by definition
    // (the condition depends on previous iteration)
    independentIterations = false;

    LLVM_DEBUG(llvm::dbgs() << "scf.while always has loop-carried deps\n");
  } else if (auto forallOp = dyn_cast<scf::ForallOp>(loop)) {
    bodyRegion = &forallOp.getRegion();

    // scf.forall is explicitly parallel unless it has shared outputs
    // with reductions (in_parallel terminator)
    if (!forallOp.getOutputs().empty()) {
      // Check if there are reductions
      auto inParallelOp = forallOp.getTerminator();
      if (inParallelOp && !inParallelOp.getYieldingOps().empty()) {
        // Has reductions - creates dependencies
        independentIterations = false;

        LLVM_DEBUG(llvm::dbgs() << "scf.forall has reductions\n");
      }
    }
  }

  // Build def-use chains for values in the loop
  if (bodyRegion) {
    bodyRegion->walk([&](Operation *op) {
      for (Value result : op->getResults()) {
        SmallVector<Operation *> users;
        for (Operation *user : result.getUsers()) {
          users.push_back(user);
        }
        defUseChains[result] = users;
      }
    });
  }
}

//===----------------------------------------------------------------------===//
// Memory Access Analysis
//===----------------------------------------------------------------------===//

void ParallelismAnalysis::analyzeMemoryAccesses() {
  // Get the loop body region
  Region *bodyRegion = nullptr;

  if (auto forOp = dyn_cast<scf::ForOp>(loop)) {
    bodyRegion = &forOp.getRegion();
  } else if (auto whileOp = dyn_cast<scf::WhileOp>(loop)) {
    bodyRegion = &whileOp.getBefore();
  } else if (auto forallOp = dyn_cast<scf::ForallOp>(loop)) {
    bodyRegion = &forallOp.getRegion();
  }

  if (!bodyRegion)
    return;

  // Collect all memory operations
  bodyRegion->walk([&](Operation *op) {
    if (isa<memref::LoadOp, memref::StoreOp>(op)) {
      memoryOps.push_back(op);
    }
  });

  // Analyze pairs of memory operations for potential conflicts
  for (size_t i = 0; i < memoryOps.size(); ++i) {
    for (size_t j = i + 1; j < memoryOps.size(); ++j) {
      if (mayConflict(memoryOps[i], memoryOps[j])) {
        memoryConflicts = true;

        LLVM_DEBUG(llvm::dbgs() << "Potential memory conflict between:\n";
                   llvm::dbgs() << "  " << *memoryOps[i] << "\n";
                   llvm::dbgs() << "  " << *memoryOps[j] << "\n");

        // For now, conservatively assume conflicts prevent parallelization
        // Future: more sophisticated alias analysis
        independentIterations = false;
        return;
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Dependence Graph Construction
//===----------------------------------------------------------------------===//

void ParallelismAnalysis::buildDependenceGraph() {
  // For each value in the loop, check if it creates a dependence cycle
  // This is a simplified analysis - more sophisticated versions would
  // build a full dependence graph and look for cycles

  for (auto &entry : defUseChains) {
    Value def = entry.first;
    const auto &uses = entry.second;

    // Check if any use feeds back to the definition
    // (simplified cycle detection)
    for (Operation *use : uses) {
      // If the use is in a yield operation that feeds back to the loop
      if (auto forOp = dyn_cast<scf::ForOp>(loop)) {
        if (auto yieldOp = dyn_cast<scf::YieldOp>(use)) {
          // Check if this yield operand corresponds to an iter_arg
          for (auto [yieldOperand, iterArg] :
               llvm::zip(yieldOp.getOperands(), forOp.getRegionIterArgs())) {
            if (yieldOperand == def) {
              loopCarriedValues.insert(iterArg);
              independentIterations = false;

              LLVM_DEBUG(llvm::dbgs() << "Found feedback cycle through yield\n");
            }
          }
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Helper Methods
//===----------------------------------------------------------------------===//

bool ParallelismAnalysis::isLoopInvariant(Value value) const {
  // A value is loop-invariant if it's defined outside the loop
  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return true; // Block argument, could be from outside

  Region *loopRegion = nullptr;
  if (auto forOp = dyn_cast<scf::ForOp>(loop)) {
    loopRegion = &forOp.getRegion();
  } else if (auto whileOp = dyn_cast<scf::WhileOp>(loop)) {
    loopRegion = &whileOp.getBefore();
  } else if (auto forallOp = dyn_cast<scf::ForallOp>(loop)) {
    loopRegion = &forallOp.getRegion();
  }

  if (!loopRegion)
    return false;

  // Check if definition is outside the loop region
  return defOp->getParentRegion() != loopRegion;
}

bool ParallelismAnalysis::mayConflict(Operation *op1, Operation *op2) const {
  // Check if two memory operations may conflict
  // Enhanced analysis: check memref, indices, and iteration-dependence

  // Get memrefs and indices being accessed
  Value memref1, memref2;
  SmallVector<Value> indices1, indices2;

  if (auto loadOp = dyn_cast<memref::LoadOp>(op1)) {
    memref1 = loadOp.getMemRef();
    indices1.append(loadOp.getIndices().begin(), loadOp.getIndices().end());
  } else if (auto storeOp = dyn_cast<memref::StoreOp>(op1)) {
    memref1 = storeOp.getMemRef();
    indices1.append(storeOp.getIndices().begin(), storeOp.getIndices().end());
  } else {
    return false; // Not a memory op
  }

  if (auto loadOp = dyn_cast<memref::LoadOp>(op2)) {
    memref2 = loadOp.getMemRef();
    indices2.append(loadOp.getIndices().begin(), loadOp.getIndices().end());
  } else if (auto storeOp = dyn_cast<memref::StoreOp>(op2)) {
    memref2 = storeOp.getMemRef();
    indices2.append(storeOp.getIndices().begin(), storeOp.getIndices().end());
  } else {
    return false; // Not a memory op
  }

  // If accessing different memrefs, no conflict
  if (memref1 != memref2)
    return false;

  // Same memref - check if indices are iteration-dependent
  // If both operations use the induction variable as their index,
  // different iterations will access different locations (no conflict)
  auto isInductionVarDependent = [&](Value index) -> bool {
    if (auto forOp = dyn_cast<scf::ForOp>(loop)) {
      Value iv = forOp.getInductionVar();

      // Check if index is the induction variable directly
      if (index == iv)
        return true;

      // Check if index is derived from induction variable
      // (e.g., through index_cast operations)
      Operation *defOp = index.getDefiningOp();
      while (defOp && isa<arith::IndexCastOp>(defOp)) {
        Value operand = defOp->getOperand(0);
        if (operand == iv)
          return true;
        defOp = operand.getDefiningOp();
      }
    }
    return false;
  };

  // If both memory operations use iteration-dependent indices,
  // different iterations access different locations
  if (indices1.size() == indices2.size() && !indices1.empty()) {
    bool allInductionDependent = true;
    for (size_t i = 0; i < indices1.size(); ++i) {
      if (!isInductionVarDependent(indices1[i]) ||
          !isInductionVarDependent(indices2[i])) {
        allInductionDependent = false;
        break;
      }
    }

    if (allInductionDependent) {
      // Indices are iteration-dependent - different iterations access
      // different locations, so no cross-iteration conflict
      LLVM_DEBUG(llvm::dbgs() << "Memory operations use induction-variable indices, "
                              << "no cross-iteration conflict\n");
      return false;
    }
  }

  // Conservative analysis for non-induction-dependent indices
  // Same memref - check operation types
  auto isStore = [](Operation *op) {
    return isa<memref::StoreOp>(op);
  };

  // Store-Store conflict on same memref
  if (isStore(op1) && isStore(op2))
    return true;

  // Store-Load conflict on same memref
  if ((isStore(op1) && isa<memref::LoadOp>(op2)) ||
      (isa<memref::LoadOp>(op1) && isStore(op2)))
    return true;

  // Load-Load: no conflict (read-only)
  return false;
}

std::optional<int64_t> ParallelismAnalysis::getStaticTripCount() const {
  if (auto forOp = dyn_cast<scf::ForOp>(loop)) {
    // Try to extract static trip count from constant bounds
    auto lbConst = forOp.getLowerBound().getDefiningOp<arith::ConstantOp>();
    auto ubConst = forOp.getUpperBound().getDefiningOp<arith::ConstantOp>();
    auto stepConst = forOp.getStep().getDefiningOp<arith::ConstantOp>();

    if (lbConst && ubConst && stepConst) {
      if (auto lbAttr = dyn_cast<IntegerAttr>(lbConst.getValue())) {
        if (auto ubAttr = dyn_cast<IntegerAttr>(ubConst.getValue())) {
          if (auto stepAttr = dyn_cast<IntegerAttr>(stepConst.getValue())) {
            int64_t lb = lbAttr.getInt();
            int64_t ub = ubAttr.getInt();
            int64_t step = stepAttr.getInt();

            if (step > 0 && ub > lb) {
              return (ub - lb + step - 1) / step;
            }
          }
        }
      }
    }
  }

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Parallelism Degree Estimation
//===----------------------------------------------------------------------===//

unsigned ParallelismAnalysis::estimateParallelismDegree() const {
  if (!independentIterations || memoryConflicts) {
    return 1; // Not parallelizable
  }

  // Try to get static trip count
  if (auto tripCount = getStaticTripCount()) {
    // Conservative: limit parallelism to avoid excessive hardware
    // For small loops, full parallelism; for large loops, cap at 16
    if (*tripCount <= 4)
      return *tripCount;
    else if (*tripCount <= 16)
      return 8;
    else
      return 16;
  }

  // Dynamic trip count: assume moderate parallelism
  return 4;
}

//===----------------------------------------------------------------------===//
// Debug and Reporting
//===----------------------------------------------------------------------===//

std::string ParallelismAnalysis::getAnalysisReport() const {
  std::string report;
  llvm::raw_string_ostream os(report);

  os << "Parallelism Analysis Report\n";
  os << "==========================\n";
  os << "Loop: " << loop->getName() << "\n";
  os << "Independent iterations: " << (independentIterations ? "YES" : "NO") << "\n";
  os << "Memory conflicts: " << (memoryConflicts ? "YES" : "NO") << "\n";
  os << "Can exploit parallelism: " << (canExploitParallelism() ? "YES" : "NO") << "\n";
  os << "Estimated parallelism degree: " << estimateParallelismDegree() << "\n";

  if (!loopCarriedValues.empty()) {
    os << "Loop-carried dependencies:\n";
    for (Value v : loopCarriedValues) {
      os << "  - " << v << "\n";
    }
  }

  os << "Memory operations: " << memoryOps.size() << "\n";

  return report;
}

} // namespace dsa
} // namespace mlir

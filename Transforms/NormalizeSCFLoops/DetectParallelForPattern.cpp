//===- DetectParallelForPattern.cpp - Detect Parallel Loops ----*- C++ -*-===//
//
// Implementation of pattern to detect and annotate parallelizable scf.for loops.
//
//===----------------------------------------------------------------------===//

#include "DetectParallelForPattern.h"
#include "../SCFToHandshakeDSA/ParallelismAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "normalize-scf-loops"

namespace mlir {
namespace dsa {

LogicalResult DetectParallelForPattern::matchAndRewrite(
    scf::ForOp forOp, PatternRewriter &rewriter) const {
  // Skip if already analyzed
  if (forOp->hasAttr("dsa.parallel_hint") ||
      forOp->hasAttr("dsa.parallel_degree")) {
    return failure();
  }

  // Check if parent function is marked as parallel candidate
  auto funcOp = forOp->getParentOfType<func::FuncOp>();
  if (!funcOp || !funcOp->hasAttr("dsa.parallel_candidate")) {
    // Function not marked for parallelization
    LLVM_DEBUG(llvm::dbgs() << "Loop parent function lacks dsa.parallel_candidate, "
                            << "skipping parallelization\n");
    return failure();
  }

  // Log if this loop was converted from a while loop
  bool convertedFromWhile = forOp->hasAttr("dsa.converted_from_while");
  LLVM_DEBUG({
    if (convertedFromWhile) {
      llvm::dbgs() << "Analyzing loop converted from scf.while\n";
    }
  });

  // Perform parallelism analysis
  ParallelismAnalysis analysis(forOp.getOperation());

  // Check if loop can exploit parallelism
  if (!analysis.canExploitParallelism()) {
    // Not parallelizable - no annotation needed
    LLVM_DEBUG(llvm::dbgs() << "Loop failed dependency analysis, "
                            << "cannot exploit parallelism\n");
    return failure();
  }

  // Annotate loop as parallelizable
  forOp->setAttr("dsa.parallel_hint", rewriter.getUnitAttr());

  // Store estimated parallelism degree
  unsigned degree = analysis.estimateParallelismDegree();
  forOp->setAttr("dsa.parallel_degree",
                 rewriter.getI64IntegerAttr(degree));

  // Log the analysis for debugging
  LLVM_DEBUG({
    llvm::dbgs() << "Detected parallelizable loop in function "
                 << funcOp.getName() << ":\n";
    llvm::dbgs() << analysis.getAnalysisReport() << "\n";
  });

  // Return success to indicate we added annotations
  return success();
}

} // namespace dsa
} // namespace mlir

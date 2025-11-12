//===- IVPatternHelpers.h - IV Pattern Extraction Helpers ------*- C++ -*-===//
//
// Helper structures and functions for extracting induction variable patterns
// from SCF while loops.
//
//===----------------------------------------------------------------------===//

#ifndef DSA_TRANSFORMS_NORMALIZESCFLOOPS_IVPATTERNHELPERS_H
#define DSA_TRANSFORMS_NORMALIZESCFLOOPS_IVPATTERNHELPERS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace dsa {

/// Represents a counting loop induction variable pattern extracted from a while loop
struct IVPattern {
  Value init;                      // Initial value
  Value step;                      // Step value
  Value bound;                     // Upper/lower bound
  arith::CmpIPredicate predicate;  // Comparison predicate
  unsigned ivIndex = 0;            // Position of IV in while loop arguments/results
  bool valid = false;              // Whether pattern was successfully extracted
};

/// Extract induction variable pattern from a while loop's before region
/// Looks for:
///   %next = arith.addi %iv, %step
///   %cond = arith.cmpi pred, %next, %bound (or %bound, %next)
///   scf.condition(%cond) %next
/// ASSUMES: IV is at position 0
IVPattern extractIVPattern(scf::WhileOp whileOp);

/// Extract induction variable pattern from a canonical while loop
/// Looks for the pattern:
///   scf.while (%iv = %init, ...) {
///     %cond = arith.cmpi pred, %iv, %bound
///     scf.condition(%cond) %iv, ...
///   } do {
///     // body operations
///     %next_iv = arith.addi %iv, %step
///     scf.yield %next_iv, ...
///   }
/// ASSUMES: IV is at position 0
IVPattern extractCanonicalIVPattern(scf::WhileOp whileOp);

/// Extract induction variable pattern at ANY position in the while loop
/// Tries each argument position until a valid IV pattern is found
/// Returns pattern with ivIndex set to the detected position
IVPattern extractIVPatternAtAnyPosition(scf::WhileOp whileOp);

/// Try to extract IV pattern at a specific argument position
/// Handles both incremented-IV (before region) and canonical (after region) forms
IVPattern tryExtractIVAtPosition(scf::WhileOp whileOp, unsigned ivIdx);

/// Check if the do region simply passes through the iteration variable
bool isDoRegionPassthrough(scf::WhileOp whileOp);

} // namespace dsa
} // namespace mlir

#endif // DSA_TRANSFORMS_NORMALIZESCFLOOPS_IVPATTERNHELPERS_H

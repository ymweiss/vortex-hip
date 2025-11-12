//===- SCFConversionHelpers.h - SCF conversion helper functions -*- C++ -*-===//
//
// Helper functions for SCF to Handshake+DSA conversion
//
//===----------------------------------------------------------------------===//

#ifndef LIB_DSA_TRANSFORMS_SCFTOHANDSHAKEDSA_SCFCONVERSIONHELPERS_H
#define LIB_DSA_TRANSFORMS_SCFTOHANDSHAKEDSA_SCFCONVERSIONHELPERS_H

#include "Common.h"

namespace mlir {
namespace dsa {

//===----------------------------------------------------------------------===//
// Zero-Trip Loop Semantic Verification
//===----------------------------------------------------------------------===//

LogicalResult verifyZeroTripSemantics(dsa::StreamOp streamOp,
                                       ArrayRef<dsa::CarryOp> carryOps);

//===----------------------------------------------------------------------===//
// SCF Path Metadata Helpers
//===----------------------------------------------------------------------===//

// Add a nesting level to memref.load/store's SCF path
// path format: "scf_type.branch" (e.g., "if.else", "while.before")
void addMemOpSCFPath(Operation *op, StringRef scfType, StringRef branch);

// Get the SCF nesting path for the given operation by walking up parent operations
SmallVector<std::string> getSCFNestingPath(Operation *op);

// Mark a control value with the SCF region it controls AND its nesting path
// The control value gets two attributes:
// 1. "dsa.scf_region": ONLY the immediate region it controls (e.g., "while.3.before")
// 2. "dsa.scf_path": the FULL nesting path including top.X where this control value is located
// The scfOp parameter is the ORIGINAL SCF operation being converted
void markControlValue(Value val, StringRef scfType, StringRef branch,
                       Operation *scfOp, OpBuilder &builder);

//===----------------------------------------------------------------------===//
// Nesting Depth Utilities
//===----------------------------------------------------------------------===//

// Determine the nesting depth of an SCF operation
// Returns the number of nested SCF operations containing this operation
int getSCFNestingDepth(Operation *op);

} // namespace dsa
} // namespace mlir

#endif // LIB_DSA_TRANSFORMS_SCFTOHANDSHAKEDSA_SCFCONVERSIONHELPERS_H

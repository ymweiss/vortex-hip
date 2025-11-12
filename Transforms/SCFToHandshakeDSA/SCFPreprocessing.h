//===- SCFPreprocessing.h - SCF metadata preprocessing ---------*- C++ -*-===//
//
// Compute and attach SCF nesting path metadata before conversion
//
//===----------------------------------------------------------------------===//

#ifndef LIB_DSA_TRANSFORMS_SCFTOHANDSHAKEDSA_SCFPREPROCESSING_H
#define LIB_DSA_TRANSFORMS_SCFTOHANDSHAKEDSA_SCFPREPROCESSING_H

#include "Common.h"

namespace mlir {
namespace dsa {

//===----------------------------------------------------------------------===//
// SCF Preprocessing
//
// CRITICAL: Must be done BEFORE conversion, as parent SCF structures
// will be destroyed during conversion, making path computation impossible later
//===----------------------------------------------------------------------===//

// Preprocess all functions in a module to attach SCF metadata
// This includes:
// - Assigning unique IDs to SCF operations
// - Computing sequence numbers (top_seq, global_seq, local_seq)
// - Attaching nesting paths to memory operations and SCF operations
void preprocessSCFMetadata(ModuleOp module, OpBuilder &builder);

} // namespace dsa
} // namespace mlir

#endif // LIB_DSA_TRANSFORMS_SCFTOHANDSHAKEDSA_SCFPREPROCESSING_H

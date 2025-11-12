//===- Passes.cpp - DSA Pass Registration --------------------*- C++ -*-===//
//
// DSA pass registration
//
//===----------------------------------------------------------------------===//

#include "dsa/Transforms/Passes.h"

namespace mlir {
namespace dsa {

#define GEN_PASS_REGISTRATION
#include "dsa/Transforms/Passes.h.inc"

void registerDSAPasses() {
  registerDSATransformsPasses();
}

} // namespace dsa
} // namespace mlir

//===- ConversionRegistry.cpp - SCF Converter Registry ---------*- C++ -*-===//
//
// Implementation of the SCF converter registry.
//
//===----------------------------------------------------------------------===//

#include "ConversionRegistry.h"
#include "Common.h"
#include "llvm/Support/Threading.h"

namespace mlir {
namespace dsa {

//===----------------------------------------------------------------------===//
// SCFConversionRegistry Implementation
//===----------------------------------------------------------------------===//

void SCFConversionRegistry::registerConverter(
    StringRef opName, std::unique_ptr<SCFOpConverter> converter) {
  converters[opName] = std::move(converter);
}

LogicalResult SCFConversionRegistry::convertOp(
    Operation *op, ConversionPatternRewriter &rewriter) {
  if (!op)
    return failure();

  StringRef opName = op->getName().getStringRef();
  auto it = converters.find(opName);
  if (it == converters.end()) {
    return op->emitError("No converter registered for operation: ") << opName;
  }

  return it->second->convert(op, rewriter);
}

bool SCFConversionRegistry::canConvert(Operation *op) const {
  if (!op)
    return false;

  StringRef opName = op->getName().getStringRef();
  return converters.find(opName) != converters.end();
}

SCFConversionRegistry &SCFConversionRegistry::getInstance(MLIRContext *context) {
  // Thread-safe per-context singleton using LLVM's once flag
  static llvm::sys::SmartMutex<true> mutex;
  static DenseMap<MLIRContext *, std::unique_ptr<SCFConversionRegistry>> instances;

  llvm::sys::SmartScopedLock<true> lock(mutex);

  auto it = instances.find(context);
  if (it == instances.end()) {
    auto registry = std::unique_ptr<SCFConversionRegistry>(
        new SCFConversionRegistry(context));
    it = instances.insert({context, std::move(registry)}).first;
  }

  return *it->second;
}

//===----------------------------------------------------------------------===//
// Global Helper Functions
//===----------------------------------------------------------------------===//

SCFConversionRegistry &getSCFConversionRegistry(MLIRContext *context) {
  return SCFConversionRegistry::getInstance(context);
}

// Note: initializeSCFConversionRegistry() is implemented in SCFConversion.cpp
// to avoid circular dependencies (it needs access to pattern types)

} // namespace dsa
} // namespace mlir

//===- ConversionRegistry.h - SCF Converter Registry -----------*- C++ -*-===//
//
// Registry for extensible SCF operation converters.
// Enables pluggable pattern matching without hardcoded dispatch.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_DSA_TRANSFORMS_SCFTOHANDSHAKEDSA_CONVERSIONREGISTRY_H
#define LIB_DSA_TRANSFORMS_SCFTOHANDSHAKEDSA_CONVERSIONREGISTRY_H

#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace mlir {
namespace dsa {

//===----------------------------------------------------------------------===//
// SCFOpConverter Interface
//===----------------------------------------------------------------------===//

/// Base interface for SCF operation converters.
/// Each SCF operation (for, while, if, etc.) implements this interface
/// to provide conversion logic to Handshake+DSA IR.
class SCFOpConverter {
public:
  virtual ~SCFOpConverter() = default;

  /// Convert the given SCF operation to Handshake+DSA IR.
  /// @param op The SCF operation to convert
  /// @param rewriter The pattern rewriter for creating new operations
  /// @return success() if conversion succeeded, failure() otherwise
  virtual LogicalResult convert(Operation *op,
                                ConversionPatternRewriter &rewriter) = 0;

  /// Get the name of the SCF operation this converter handles.
  /// @return Operation name (e.g., "scf.for", "scf.while")
  virtual StringRef getOperationName() const = 0;
};

//===----------------------------------------------------------------------===//
// Concrete Converter Template
//===----------------------------------------------------------------------===//

/// Template wrapper that adapts OpConversionPattern to SCFOpConverter interface.
/// This allows existing conversion patterns to be registered without modification.
///
/// Usage:
///   registry.registerPattern<scf::ForOp, ConvertForOp>();
///
template <typename OpTy, typename PatternT>
class TemplatedSCFOpConverter : public SCFOpConverter {
public:
  explicit TemplatedSCFOpConverter(MLIRContext *context)
      : pattern(context) {}

  LogicalResult convert(Operation *op,
                        ConversionPatternRewriter &rewriter) override {
    // Get the specific operation type
    auto typedOp = dyn_cast<OpTy>(op);
    if (!typedOp) {
      return failure();
    }

    // Create adaptor with operation's operands
    SmallVector<Value> operands(op->getOperands());
    typename PatternT::OpAdaptor adaptor(operands, op->getAttrDictionary());

    // Invoke the pattern's matchAndRewrite
    return pattern.matchAndRewrite(typedOp, adaptor, rewriter);
  }

  StringRef getOperationName() const override {
    return OpTy::getOperationName();
  }

private:
  PatternT pattern;
};

//===----------------------------------------------------------------------===//
// SCFConversionRegistry
//===----------------------------------------------------------------------===//

/// Registry for SCF operation converters.
/// Provides extensible mechanism for converting SCF operations to Handshake+DSA.
///
/// Example usage:
///   SCFConversionRegistry &registry = getSCFConversionRegistry();
///   registry.registerConverter<ConvertForOp>();
///   registry.registerConverter<ConvertWhileOp>();
///
///   // Later, in nested conversion:
///   if (registry.canConvert(op)) {
///     registry.convertOp(op, rewriter);
///   }
///
class SCFConversionRegistry {
public:
  /// Register a converter for a specific SCF operation.
  /// @param opName The name of the operation (e.g., "scf.for")
  /// @param converter Unique pointer to the converter implementation
  void registerConverter(StringRef opName,
                         std::unique_ptr<SCFOpConverter> converter);

  /// Register a converter using template pattern type.
  /// This is a convenience method that creates TemplatedSCFOpConverter.
  /// @tparam OpTy The operation type (e.g., scf::ForOp)
  /// @tparam PatternT The OpConversionPattern type (e.g., ConvertForOp)
  template <typename OpTy, typename PatternT>
  void registerPattern(MLIRContext *context) {
    auto converter = std::make_unique<TemplatedSCFOpConverter<OpTy, PatternT>>(context);
    StringRef opName = converter->getOperationName();
    registerConverter(opName, std::move(converter));
  }

  /// Convert an SCF operation using the registered converter.
  /// @param op The operation to convert
  /// @param rewriter The pattern rewriter
  /// @return success() if conversion succeeded, failure() otherwise
  LogicalResult convertOp(Operation *op,
                          ConversionPatternRewriter &rewriter);

  /// Check if the operation can be converted by this registry.
  /// @param op The operation to check
  /// @return true if a converter is registered for this operation
  bool canConvert(Operation *op) const;

  /// Get the singleton instance of the registry.
  /// Note: This is a per-context singleton for thread safety.
  static SCFConversionRegistry &getInstance(MLIRContext *context);

private:
  /// Map from operation name to converter
  DenseMap<StringRef, std::unique_ptr<SCFOpConverter>> converters;

  /// Context for this registry instance
  MLIRContext *context = nullptr;

  /// Private constructor for singleton pattern
  explicit SCFConversionRegistry(MLIRContext *ctx) : context(ctx) {}
};

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Get the global SCF conversion registry for the given context.
/// This function ensures thread-safe access to the registry.
SCFConversionRegistry &getSCFConversionRegistry(MLIRContext *context);

/// Initialize the registry with default SCF converters.
/// This should be called once during pass initialization.
void initializeSCFConversionRegistry(MLIRContext *context);

} // namespace dsa
} // namespace mlir

#endif // LIB_DSA_TRANSFORMS_SCFTOHANDSHAKEDSA_CONVERSIONREGISTRY_H

//===- FuncConversion.cpp - Function conversion patterns -------*- C++ -*-===//
//
// Conversion patterns for func.func and return operations
//
//===----------------------------------------------------------------------===//

#include "Common.h"

namespace mlir {
namespace dsa {

//===----------------------------------------------------------------------===//
// FuncOp to HandshakeFuncOp Conversion Pattern
//===----------------------------------------------------------------------===//

struct ConvertFuncOp : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check if this is a mixed LLVM+MLIR case or pure MLIR case
    // Mixed: Only convert functions with dsa_optimize attribute
    // Pure MLIR: Convert all functions
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    bool hasDSAAnnotation = false;

    // Check if ANY func.func in the module has dsa_optimize attribute
    moduleOp.walk([&](func::FuncOp func) {
      if (func->hasAttr("dsa_optimize")) {
        hasDSAAnnotation = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    // If there are DSA annotations in the module, only convert annotated functions
    // This handles the C++ case where CPU and DSA versions coexist
    if (hasDSAAnnotation && !funcOp->hasAttr("dsa_optimize"))
      return failure();

    // Create a new handshake.func with entry token added
    // Handshake functions need an additional 'none' type entry control argument
    FunctionType funcType = funcOp.getFunctionType();
    SmallVector<Type> inputTypes(funcType.getInputs().begin(), funcType.getInputs().end());
    SmallVector<Type> resultTypes(funcType.getResults().begin(), funcType.getResults().end());

    // Add entry control token as last input
    inputTypes.push_back(rewriter.getNoneType());

    // Note: Control output token is added later by connectMemoryCompletionToReturn

    FunctionType handshakeType = rewriter.getFunctionType(inputTypes, resultTypes);
    auto handshakeFuncOp = circt::handshake::FuncOp::create(rewriter, 
        funcOp.getLoc(), funcOp.getName(), handshakeType);

    // Set argNames, resNames, argAttrs and resAttrs attributes
    SmallVector<Attribute> argNames, resNames, argAttrs, resAttrs;
    for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
      argNames.push_back(rewriter.getStringAttr("arg" + std::to_string(i)));
      NamedAttrList attrs;
      attrs.append("hw.name", rewriter.getStringAttr("arg" + std::to_string(i)));
      argAttrs.push_back(rewriter.getDictionaryAttr(attrs));
    }
    // Add entry control token argument name and attrs
    argNames.push_back(rewriter.getStringAttr("entry_ctrl"));
    NamedAttrList entryAttrs;
    entryAttrs.append("hw.name", rewriter.getStringAttr("entry_ctrl"));
    argAttrs.push_back(rewriter.getDictionaryAttr(entryAttrs));

    for (unsigned i = 0; i < funcOp.getNumResults(); ++i) {
      resNames.push_back(rewriter.getStringAttr("res" + std::to_string(i)));
      NamedAttrList attrs;
      attrs.append("hw.name", rewriter.getStringAttr("res" + std::to_string(i)));
      resAttrs.push_back(rewriter.getDictionaryAttr(attrs));
    }
    // Note: Control output token attributes are added later by connectMemoryCompletionToReturn

    handshakeFuncOp->setAttr("argNames", rewriter.getArrayAttr(argNames));
    handshakeFuncOp->setAttr("resNames", rewriter.getArrayAttr(resNames));
    handshakeFuncOp.setArgAttrsAttr(rewriter.getArrayAttr(argAttrs));
    handshakeFuncOp.setResAttrsAttr(rewriter.getArrayAttr(resAttrs));

    // Copy the function body
    rewriter.inlineRegionBefore(funcOp.getBody(), handshakeFuncOp.getBody(),
                                 handshakeFuncOp.end());

    // Add entry control token as a block argument
    // This must be done AFTER inlining the region
    Block &entryBlock = handshakeFuncOp.getBody().front();
    BlockArgument entryCtrl = entryBlock.addArgument(rewriter.getNoneType(), funcOp.getLoc());

    // CRITICAL: Create entry token by joining non-memref arguments
    // This must be done during func conversion (BEFORE SCF conversion) so that
    // SCF conversion can route this token through nested structures.
    // Without this, memory operations in nested SCF structures would receive
    // unrouted control tokens, violating dataflow semantics.
    SmallVector<Value> nonMemrefArgs;
    for (auto arg : handshakeFuncOp.getArguments()) {
      if (!isa<MemRefType>(arg.getType())) {
        nonMemrefArgs.push_back(arg);
      }
    }

    if (!nonMemrefArgs.empty()) {
      rewriter.setInsertionPointToStart(&entryBlock);
      auto joinOp = circt::handshake::JoinOp::create(rewriter, 
          handshakeFuncOp.getLoc(), nonMemrefArgs);

      // Mark the join operation as the memory entry token
      // Memory conversion will find this join and create appropriate forks
      // SCF conversion will route this join output through nested structures
      joinOp->setAttr("dsa.mem_entry_token", rewriter.getUnitAttr());
    }

    // Erase the original func.func
    rewriter.eraseOp(funcOp);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// HandshakeFuncOp Return Conversion Pattern
//===----------------------------------------------------------------------===//

struct ConvertFuncReturn : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp returnOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert func.return to handshake.return
    // The dataflow semantics naturally ensure all operations complete before return
    rewriter.replaceOpWithNewOp<circt::handshake::ReturnOp>(
        returnOp, adaptor.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// LLVM Return Conversion Pattern
//===----------------------------------------------------------------------===//

// NOTE: This pattern should rarely match in normal flow.
// The RaiseLLVMToCF pass converts llvm.return → func.return,
// and then ConvertFuncReturn pattern converts func.return → handshake.return.
// This pattern is kept as a safety net for edge cases where llvm.return
// might still exist in a handshake.func (e.g., if RaiseLLVMToCF was skipped).
struct ConvertLLVMReturn : public OpConversionPattern<LLVM::ReturnOp> {
  using OpConversionPattern<LLVM::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::ReturnOp returnOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check if we're in a handshake.func
    auto funcOp = returnOp->getParentOfType<circt::handshake::FuncOp>();

    // Only convert returns that are in a handshake.func
    // Returns in llvm.func should stay as llvm.return (will be converted by RaiseLLVMToCF)
    if (!funcOp)
      return failure();

    // If the signature needs fixing, fix it
    if (!adaptor.getOperands().empty()) {
      // Get current function type
      auto currentFuncType = funcOp.getFunctionType();
      unsigned numResults = currentFuncType.getNumResults();
      unsigned numReturnOperands = adaptor.getOperands().size();

      // If the function signature doesn't match the return operands, fix it
      if (numResults != numReturnOperands) {
        // Build new result types from return operands
        SmallVector<Type> resultTypes;
        for (Value operand : adaptor.getOperands()) {
          resultTypes.push_back(operand.getType());
        }

        // Create new function type with updated results
        auto newFuncType = rewriter.getFunctionType(
            currentFuncType.getInputs(), resultTypes);

        // Update function type
        funcOp.setFunctionTypeAttr(TypeAttr::get(newFuncType));

        // Update resNames and resAttrs attributes to match
        SmallVector<Attribute> resNames, resAttrs;
        for (unsigned i = 0; i < resultTypes.size(); ++i) {
          resNames.push_back(rewriter.getStringAttr("res" + std::to_string(i)));
          NamedAttrList attrs;
          attrs.append("hw.name", rewriter.getStringAttr("res" + std::to_string(i)));
          resAttrs.push_back(rewriter.getDictionaryAttr(attrs));
        }
        funcOp->setAttr("resNames", rewriter.getArrayAttr(resNames));
        funcOp.setResAttrsAttr(rewriter.getArrayAttr(resAttrs));
      }
    }

    // Convert llvm.return to handshake.return
    // The dataflow semantics naturally ensure all operations complete before return
    rewriter.replaceOpWithNewOp<circt::handshake::ReturnOp>(
        returnOp, adaptor.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern Registration
//===----------------------------------------------------------------------===//

void populateFuncConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertFuncOp, ConvertFuncReturn, ConvertLLVMReturn>(
      patterns.getContext());
}

} // namespace dsa
} // namespace mlir

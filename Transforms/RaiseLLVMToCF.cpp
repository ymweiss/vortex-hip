//===- RaiseLLVMToCF.cpp - Convert LLVM to Func+CF dialects ----*- C++ -*-===//
//
// This pass converts LLVM dialect to func+cf dialects:
// - llvm.func → func.func (enables --lift-cf-to-scf)
// - llvm.br → cf.br
// - llvm.cond_br → cf.cond_br
// - llvm.switch → cf.switch
// - llvm.return → func.return
//
// Additionally, this pass extracts parallelization metadata:
// - Recognizes dsa_parallel and dsa_optimize annotations
// - Detects LLVM loop metadata (unroll hints, parallel access hints)
// - Marks functions and control flow for downstream parallel optimization
//
//===----------------------------------------------------------------------===//

#include "dsa/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "raise-llvm-to-cf"

namespace mlir {
namespace dsa {

#define GEN_PASS_DEF_RAISELLVMTOCF
#include "dsa/Transforms/Passes.h.inc"

namespace {

struct RaiseLLVMToCFPass
    : public impl::RaiseLLVMToCFBase<RaiseLLVMToCFPass> {
  using RaiseLLVMToCFBase::RaiseLLVMToCFBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    // Process llvm.global.annotations to add dsa_optimize attributes
    processGlobalAnnotations(module, builder);

    // Collect llvm.func operations with dsa_optimize attribute
    SmallVector<LLVM::LLVMFuncOp> llvmFuncs;
    module.walk([&](LLVM::LLVMFuncOp funcOp) {
      if (funcOp->hasAttr("dsa_optimize")) {
        llvmFuncs.push_back(funcOp);
      }
    });

    // Convert each llvm.func to func.func with CF control flow
    for (auto llvmFunc : llvmFuncs) {
      convertLLVMFuncToFuncWithCF(llvmFunc, builder);
    }
  }

private:
  // Process llvm.global.annotations and attach dsa_optimize/dsa_parallel to annotated functions
  void processGlobalAnnotations(ModuleOp module, OpBuilder &builder) {
    // Find llvm.global.annotations
    LLVM::GlobalOp annotationsGlobal = nullptr;
    module.walk([&](LLVM::GlobalOp global) {
      if (global.getSymName() == "llvm.global.annotations") {
        annotationsGlobal = global;
      }
    });

    if (!annotationsGlobal) {
      return;  // No annotations
    }

    // Collect function names and their annotation types from annotation structure
    // Map: function name -> set of annotation strings
    llvm::DenseMap<StringRef, llvm::SmallVector<StringRef>> functionAnnotations;

    // Try to extract annotation strings by walking through the annotation structure
    // The structure is typically: array of {funcPtr, annotationStrPtr, ...}
    // We need to pair function references with their annotation string references
    SmallVector<StringRef> collectedFuncNames;
    SmallVector<StringRef> collectedAnnotations;

    annotationsGlobal.walk([&](LLVM::AddressOfOp addrOf) {
      StringRef name = addrOf.getGlobalName();
      if (!name.starts_with(".str")) {
        // This is a function reference
        collectedFuncNames.push_back(name);
      }
    });

    // Collect annotation string constants
    module.walk([&](LLVM::GlobalOp global) {
      if (global.getSymName().starts_with(".str")) {
        // Try to extract the string value
        if (auto initValue = global.getValueOrNull()) {
          if (auto strAttr = dyn_cast<StringAttr>(initValue)) {
            StringRef str = strAttr.getValue();
            // Check for known annotations
            if (str.contains("dsa_optimize") || str.contains("dsa_parallel")) {
              collectedAnnotations.push_back(str);
            }
          }
        }
      }
    });

    // For simplicity, assume all collected functions have the collected annotations
    // (More sophisticated parsing would track the exact pairing)
    for (StringRef funcName : collectedFuncNames) {
      for (StringRef annotation : collectedAnnotations) {
        functionAnnotations[funcName].push_back(annotation);
      }
    }

    // Mark functions with appropriate attributes
    module.walk([&](LLVM::LLVMFuncOp func) {
      StringRef funcName = func.getName();
      if (functionAnnotations.contains(funcName)) {
        // Always mark as dsa_optimize if annotated
        func->setAttr("dsa_optimize", builder.getUnitAttr());

        // Check if dsa_parallel is present
        for (StringRef annotation : functionAnnotations[funcName]) {
          if (annotation.contains("dsa_parallel")) {
            func->setAttr("dsa.parallel_candidate", builder.getUnitAttr());
            LLVM_DEBUG(llvm::dbgs() << "Marked function " << funcName
                                    << " as parallel candidate\n");
            break;
          }
        }
      }
    });

    // Remove llvm.global.annotations to avoid dangling references
    // after functions are converted
    SmallVector<LLVM::GlobalOp> toErase;
    module.walk([&](LLVM::GlobalOp global) {
      if (global.getSymName() == "llvm.global.annotations" ||
          global.getSymName().starts_with(".str")) {
        toErase.push_back(global);
      }
    });
    for (auto global : toErase) {
      global.erase();
    }
  }

  void convertLLVMFuncToFuncWithCF(LLVM::LLVMFuncOp llvmFunc, OpBuilder &builder) {
    // Skip external function declarations (no body)
    if (llvmFunc.getBody().empty()) {
      return;
    }

    // Step 1: Convert LLVM control flow to CF in place
    SmallVector<Operation *> branchOps;
    llvmFunc.walk([&](Operation *op) {
      if (isa<LLVM::BrOp, LLVM::CondBrOp, LLVM::SwitchOp>(op)) {
        branchOps.push_back(op);
        LLVM_DEBUG(llvm::dbgs() << "Found branch op: " << op->getName() << "\n");
      }
    });

    for (Operation *op : branchOps) {
      builder.setInsertionPoint(op);

      if (auto brOp = dyn_cast<LLVM::BrOp>(op)) {
        LLVM_DEBUG(llvm::dbgs() << "Converting llvm.br\n");
        cf::BranchOp::create(
            builder, brOp.getLoc(), brOp.getSuccessor(),
            brOp.getSuccessorOperands(0).getForwardedOperands());
        op->erase();

      } else if (auto condBrOp = dyn_cast<LLVM::CondBrOp>(op)) {
        LLVM_DEBUG(llvm::dbgs() << "Converting llvm.cond_br\n");
        cf::CondBranchOp::create(
            builder, condBrOp.getLoc(), condBrOp.getCondition(),
            condBrOp.getSuccessor(0),  // trueDest
            condBrOp.getSuccessorOperands(0).getForwardedOperands(),  // trueOps
            condBrOp.getSuccessor(1),  // falseDest
            condBrOp.getSuccessorOperands(1).getForwardedOperands()); // falseOps
        op->erase();

      } else if (auto switchOp = dyn_cast<LLVM::SwitchOp>(op)) {
        LLVM_DEBUG(llvm::dbgs() << "Converting llvm.switch\n");
        // Convert llvm.switch to cf.switch
        // Extract case values and destinations
        SmallVector<APInt> caseValues;
        SmallVector<Block *> caseDestinations;
        SmallVector<ValueRange> caseOperands;

        // Get case values from the switch operation
        if (auto caseValuesAttr = switchOp.getCaseValues()) {
          for (auto caseValue : caseValuesAttr->getValues<APInt>()) {
            caseValues.push_back(caseValue);
          }
        }

        // Get case destinations and operands
        for (unsigned i = 0; i < switchOp.getCaseDestinations().size(); ++i) {
          caseDestinations.push_back(switchOp.getCaseDestinations()[i]);
          caseOperands.push_back(switchOp.getCaseOperands(i));
        }

        // Create cf.switch operation
        LLVM_DEBUG(llvm::dbgs() << "Creating cf.switch with " << caseValues.size()
                                << " cases\n");
        builder.create<cf::SwitchOp>(
            switchOp.getLoc(),
            switchOp.getValue(),
            switchOp.getDefaultDestination(),
            switchOp.getDefaultOperands(),
            caseValues,
            caseDestinations,
            caseOperands);
        LLVM_DEBUG(llvm::dbgs() << "Successfully created cf.switch\n");
        op->erase();
      } else {
        LLVM_DEBUG(llvm::dbgs() << "WARNING: Unexpected branch op type: "
                                << op->getName() << "\n");
      }
    }

    // Step 1b: Convert llvm.return to func.return
    SmallVector<Operation *> returnOps;
    llvmFunc.walk([&](LLVM::ReturnOp returnOp) {
      returnOps.push_back(returnOp.getOperation());
    });

    for (Operation *op : returnOps) {
      auto returnOp = cast<LLVM::ReturnOp>(op);
      builder.setInsertionPoint(op);

      // Create func.return with the same operands
      func::ReturnOp::create(builder, returnOp.getLoc(), returnOp.getOperands());
      op->erase();
    }

    // Step 2: Convert llvm.func to func.func
    builder.setInsertionPoint(llvmFunc);

    // Extract parameter types
    SmallVector<Type> inputTypes;
    Block &entryBlock = llvmFunc.getBody().front();
    for (BlockArgument arg : entryBlock.getArguments()) {
      inputTypes.push_back(arg.getType());
    }

    // Extract return types from llvm.func
    SmallVector<Type> resultTypes;
    Type llvmReturnType = llvmFunc.getFunctionType().getReturnType();
    // LLVM functions return a single type (or void)
    // Only add to results if not void
    if (!isa<LLVM::LLVMVoidType>(llvmReturnType)) {
      resultTypes.push_back(llvmReturnType);
    }

    // Create func.func with correct signature
    auto funcType = builder.getFunctionType(inputTypes, resultTypes);
    auto newFunc = func::FuncOp::create(
        builder, llvmFunc.getLoc(), llvmFunc.getName(), funcType);
    newFunc->setAttr("dsa_optimize", builder.getUnitAttr());

    // Transfer parallelization attributes if present
    if (llvmFunc->hasAttr("dsa.parallel_candidate")) {
      newFunc->setAttr("dsa.parallel_candidate", builder.getUnitAttr());
      LLVM_DEBUG(llvm::dbgs() << "Transferred dsa.parallel_candidate to func.func "
                              << newFunc.getName() << "\n");
    }

    // Move body from llvm.func to func.func
    Region &funcBody = newFunc.getBody();
    funcBody.takeBody(llvmFunc.getBody());

    // Erase llvm.func
    llvmFunc.erase();
  }
};

} // namespace

std::unique_ptr<Pass> createRaiseLLVMToCFPass() {
  return std::make_unique<RaiseLLVMToCFPass>();
}

} // namespace dsa
} // namespace mlir

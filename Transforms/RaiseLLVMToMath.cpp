//===- RaiseLLVMToMath.cpp - Convert LLVM calls/intrinsics to Math dialect -*- C++ -*-===//
//
// This pass converts LLVM dialect operations to math dialect:
// - llvm.call @sqrtf/@sqrt → math.sqrt
// - llvm.call @expf/@exp → math.exp
// - llvm.call @sinf/@sin → math.sin
// - And many more standard cmath functions...
//
// Also converts LLVM intrinsics:
// - llvm.intr.sqrt → math.sqrt
// - llvm.intr.fabs → math.absf
// - llvm.intr.fma → math.fma
// - llvm.intr.ceil → math.ceil
// - llvm.intr.floor → math.floor
// - llvm.intr.round → math.round
// - llvm.intr.trunc → math.trunc
// - llvm.intr.ctpop → math.ctpop
// - llvm.intr.ctlz → math.ctlz
// - llvm.intr.cttz → math.cttz
//
//===----------------------------------------------------------------------===//

#include "dsa/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
namespace dsa {

#define GEN_PASS_DEF_RAISELLVMTOMATH
#include "dsa/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// CMath Function Mapping
//===----------------------------------------------------------------------===//

/// Get the base math operation name for a cmath function name
/// Handles both float (sinf) and double (sin) variants
/// Returns empty string if not a recognized cmath function
std::string getMathOpNameForCMathFunc(StringRef funcName) {
  // Create a map from cmath function names to math dialect operations
  // We normalize by stripping the 'f' suffix for float variants
  static const llvm::StringMap<std::string> cmathToMathOp = {
      // Trigonometric
      {"sin", "sin"},     {"sinf", "sin"},
      {"cos", "cos"},     {"cosf", "cos"},
      {"tan", "tan"},     {"tanf", "tan"},
      {"asin", "asin"},   {"asinf", "asin"},
      {"acos", "acos"},   {"acosf", "acos"},
      {"atan", "atan"},   {"atanf", "atan"},
      {"atan2", "atan2"}, {"atan2f", "atan2"},

      // Hyperbolic
      {"sinh", "sinh"},   {"sinhf", "sinh"},
      {"cosh", "cosh"},   {"coshf", "cosh"},
      {"tanh", "tanh"},   {"tanhf", "tanh"},
      {"asinh", "asinh"}, {"asinhf", "asinh"},
      {"acosh", "acosh"}, {"acoshf", "acosh"},
      {"atanh", "atanh"}, {"atanhf", "atanh"},

      // Exponential/Logarithmic
      {"exp", "exp"},     {"expf", "exp"},
      {"exp2", "exp2"},   {"exp2f", "exp2"},
      {"expm1", "expm1"}, {"expm1f", "expm1"},
      {"log", "log"},     {"logf", "log"},
      {"log10", "log10"}, {"log10f", "log10"},
      {"log2", "log2"},   {"log2f", "log2"},
      {"log1p", "log1p"}, {"log1pf", "log1p"},

      // Power/Root
      {"pow", "powf"},    {"powf", "powf"},  // Note: math.powf for float power
      {"sqrt", "sqrt"},   {"sqrtf", "sqrt"},
      {"cbrt", "cbrt"},   {"cbrtf", "cbrt"},
      {"rsqrt", "rsqrt"}, {"rsqrtf", "rsqrt"},

      // Rounding
      {"ceil", "ceil"},   {"ceilf", "ceil"},
      {"floor", "floor"}, {"floorf", "floor"},
      {"round", "round"}, {"roundf", "round"},
      {"trunc", "trunc"}, {"truncf", "trunc"},
      {"roundeven", "roundeven"}, // No 'f' variant in cmath typically

      // Other
      {"fabs", "absf"},   {"fabsf", "absf"},
      {"copysign", "copysign"}, {"copysignf", "copysign"},
      {"fma", "fma"},     {"fmaf", "fma"},
      {"erf", "erf"},     {"erff", "erf"},
      {"erfc", "erfc"},   {"erfcf", "erfc"},
  };

  auto it = cmathToMathOp.find(funcName);
  if (it != cmathToMathOp.end())
    return it->second;

  return "";
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct RaiseLLVMToMathPass
    : public impl::RaiseLLVMToMathBase<RaiseLLVMToMathPass> {
  using RaiseLLVMToMathBase::RaiseLLVMToMathBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    // Collect func.func operations with dsa_optimize attribute
    SmallVector<func::FuncOp> funcsToProcess;
    module.walk([&](func::FuncOp funcOp) {
      if (funcOp->hasAttr("dsa_optimize")) {
        funcsToProcess.push_back(funcOp);
      }
    });

    // Convert LLVM call operations and intrinsics in each function
    for (auto funcOp : funcsToProcess) {
      if (failed(convertLLVMCallsToMath(funcOp, builder))) {
        signalPassFailure();
        return;
      }
      if (failed(convertLLVMIntrinsicsToMath(funcOp, builder))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  LogicalResult convertLLVMCallsToMath(func::FuncOp funcOp,
                                        OpBuilder &builder) {
    // Collect all LLVM call operations to convert
    SmallVector<LLVM::CallOp> callsToConvert;

    funcOp.walk([&](LLVM::CallOp callOp) {
      // Get the callee name
      auto callee = callOp.getCallee();
      if (!callee)
        return;

      StringRef calleeName = *callee;

      // Check if this is a cmath function we can convert
      if (!getMathOpNameForCMathFunc(calleeName).empty()) {
        callsToConvert.push_back(callOp);
      }
    });

    // Convert each call operation
    for (LLVM::CallOp callOp : callsToConvert) {
      builder.setInsertionPoint(callOp);

      auto callee = callOp.getCallee();
      if (!callee)
        continue;

      StringRef calleeName = *callee;
      std::string mathOpName = getMathOpNameForCMathFunc(calleeName);

      if (mathOpName.empty())
        continue;

      // Get operands
      auto operands = callOp.getArgOperands();
      Location loc = callOp.getLoc();

      // Create the appropriate math operation
      Value result;

      // Single-argument operations
      if (operands.size() == 1) {
        if (mathOpName == "sin") {
          result = math::SinOp::create(builder, loc, operands[0]);
        } else if (mathOpName == "cos") {
          result = math::CosOp::create(builder, loc, operands[0]);
        } else if (mathOpName == "tan") {
          result = math::TanOp::create(builder, loc, operands[0]);
        } else if (mathOpName == "asin") {
          result = math::AsinOp::create(builder, loc, operands[0]);
        } else if (mathOpName == "acos") {
          result = math::AcosOp::create(builder, loc, operands[0]);
        } else if (mathOpName == "atan") {
          result = math::AtanOp::create(builder, loc, operands[0]);
        } else if (mathOpName == "sinh") {
          result = math::SinhOp::create(builder, loc, operands[0]);
        } else if (mathOpName == "cosh") {
          result = math::CoshOp::create(builder, loc, operands[0]);
        } else if (mathOpName == "tanh") {
          result = math::TanhOp::create(builder, loc, operands[0]);
        } else if (mathOpName == "asinh") {
          result = math::AsinhOp::create(builder, loc, operands[0]);
        } else if (mathOpName == "acosh") {
          result = math::AcoshOp::create(builder, loc, operands[0]);
        } else if (mathOpName == "atanh") {
          result = math::AtanhOp::create(builder, loc, operands[0]);
        } else if (mathOpName == "exp") {
          result = math::ExpOp::create(builder, loc, operands[0]);
        } else if (mathOpName == "exp2") {
          result = math::Exp2Op::create(builder, loc, operands[0]);
        } else if (mathOpName == "expm1") {
          result = math::ExpM1Op::create(builder, loc, operands[0]);
        } else if (mathOpName == "log") {
          result = math::LogOp::create(builder, loc, operands[0]);
        } else if (mathOpName == "log10") {
          result = math::Log10Op::create(builder, loc, operands[0]);
        } else if (mathOpName == "log2") {
          result = math::Log2Op::create(builder, loc, operands[0]);
        } else if (mathOpName == "log1p") {
          result = math::Log1pOp::create(builder, loc, operands[0]);
        } else if (mathOpName == "sqrt") {
          result = math::SqrtOp::create(builder, loc, operands[0]);
        } else if (mathOpName == "cbrt") {
          result = math::CbrtOp::create(builder, loc, operands[0]);
        } else if (mathOpName == "rsqrt") {
          result = math::RsqrtOp::create(builder, loc, operands[0]);
        } else if (mathOpName == "ceil") {
          result = math::CeilOp::create(builder, loc, operands[0]);
        } else if (mathOpName == "floor") {
          result = math::FloorOp::create(builder, loc, operands[0]);
        } else if (mathOpName == "round") {
          result = math::RoundOp::create(builder, loc, operands[0]);
        } else if (mathOpName == "trunc") {
          result = math::TruncOp::create(builder, loc, operands[0]);
        } else if (mathOpName == "roundeven") {
          result = math::RoundEvenOp::create(builder, loc, operands[0]);
        } else if (mathOpName == "absf") {
          result = math::AbsFOp::create(builder, loc, operands[0]);
        } else if (mathOpName == "erf") {
          result = math::ErfOp::create(builder, loc, operands[0]);
        } else if (mathOpName == "erfc") {
          result = math::ErfcOp::create(builder, loc, operands[0]);
        }
      }
      // Two-argument operations
      else if (operands.size() == 2) {
        if (mathOpName == "atan2") {
          result = math::Atan2Op::create(builder, loc, operands[0], operands[1]);
        } else if (mathOpName == "powf") {
          result = math::PowFOp::create(builder, loc, operands[0], operands[1]);
        } else if (mathOpName == "copysign") {
          result = math::CopySignOp::create(builder, loc, operands[0], operands[1]);
        }
      }
      // Three-argument operations
      else if (operands.size() == 3) {
        if (mathOpName == "fma") {
          result = math::FmaOp::create(builder, loc, operands[0], operands[1], operands[2]);
        }
      }

      // If we successfully created a math operation, replace the call
      if (result) {
        callOp.getResult().replaceAllUsesWith(result);
        callOp.erase();
      }
    }

    return success();
  }

  LogicalResult convertLLVMIntrinsicsToMath(func::FuncOp funcOp,
                                             OpBuilder &builder) {
    // Collect all LLVM intrinsic operations to convert
    SmallVector<Operation *> intrinsicsToConvert;

    funcOp.walk([&](Operation *op) {
      if (isa<LLVM::SqrtOp, LLVM::FAbsOp, LLVM::FMAOp,
              LLVM::FCeilOp, LLVM::FFloorOp, LLVM::RoundOp, LLVM::FTruncOp,
              LLVM::CtPopOp, LLVM::CountLeadingZerosOp, LLVM::CountTrailingZerosOp>(op)) {
        intrinsicsToConvert.push_back(op);
      }
    });

    // Convert each intrinsic operation
    for (Operation *op : intrinsicsToConvert) {
      builder.setInsertionPoint(op);
      Location loc = op->getLoc();

      if (auto sqrtOp = dyn_cast<LLVM::SqrtOp>(op)) {
        // llvm.intr.sqrt → math.sqrt
        auto mathSqrt = builder.create<math::SqrtOp>(loc, sqrtOp.getIn());
        sqrtOp.getRes().replaceAllUsesWith(mathSqrt.getResult());
        op->erase();

      } else if (auto fabsOp = dyn_cast<LLVM::FAbsOp>(op)) {
        // llvm.intr.fabs → math.absf
        auto mathAbs = builder.create<math::AbsFOp>(loc, fabsOp.getIn());
        fabsOp.getRes().replaceAllUsesWith(mathAbs.getResult());
        op->erase();

      } else if (auto fmaOp = dyn_cast<LLVM::FMAOp>(op)) {
        // llvm.intr.fma → math.fma
        auto mathFma = builder.create<math::FmaOp>(
            loc, fmaOp.getA(), fmaOp.getB(), fmaOp.getC());
        fmaOp.getRes().replaceAllUsesWith(mathFma.getResult());
        op->erase();

      } else if (auto ceilOp = dyn_cast<LLVM::FCeilOp>(op)) {
        // llvm.intr.ceil → math.ceil
        auto mathCeil = builder.create<math::CeilOp>(loc, ceilOp.getIn());
        ceilOp.getRes().replaceAllUsesWith(mathCeil.getResult());
        op->erase();

      } else if (auto floorOp = dyn_cast<LLVM::FFloorOp>(op)) {
        // llvm.intr.floor → math.floor
        auto mathFloor = builder.create<math::FloorOp>(loc, floorOp.getIn());
        floorOp.getRes().replaceAllUsesWith(mathFloor.getResult());
        op->erase();

      } else if (auto roundOp = dyn_cast<LLVM::RoundOp>(op)) {
        // llvm.intr.round → math.round
        auto mathRound = builder.create<math::RoundOp>(loc, roundOp.getIn());
        roundOp.getRes().replaceAllUsesWith(mathRound.getResult());
        op->erase();

      } else if (auto truncOp = dyn_cast<LLVM::FTruncOp>(op)) {
        // llvm.intr.trunc → math.trunc
        auto mathTrunc = builder.create<math::TruncOp>(loc, truncOp.getIn());
        truncOp.getRes().replaceAllUsesWith(mathTrunc.getResult());
        op->erase();

      } else if (auto ctpopOp = dyn_cast<LLVM::CtPopOp>(op)) {
        // llvm.intr.ctpop → math.ctpop
        auto mathCtpop = builder.create<math::CtPopOp>(loc, ctpopOp.getIn());
        ctpopOp.getRes().replaceAllUsesWith(mathCtpop.getResult());
        op->erase();

      } else if (auto ctlzOp = dyn_cast<LLVM::CountLeadingZerosOp>(op)) {
        // llvm.intr.ctlz → math.ctlz
        auto mathCtlz = builder.create<math::CountLeadingZerosOp>(loc, ctlzOp.getIn());
        ctlzOp.getRes().replaceAllUsesWith(mathCtlz.getResult());
        op->erase();

      } else if (auto cttzOp = dyn_cast<LLVM::CountTrailingZerosOp>(op)) {
        // llvm.intr.cttz → math.cttz
        auto mathCttz = builder.create<math::CountTrailingZerosOp>(loc, cttzOp.getIn());
        cttzOp.getRes().replaceAllUsesWith(mathCttz.getResult());
        op->erase();
      }
    }

    return success();
  }
};

} // namespace

std::unique_ptr<Pass> createRaiseLLVMToMathPass() {
  return std::make_unique<RaiseLLVMToMathPass>();
}

} // namespace dsa
} // namespace mlir

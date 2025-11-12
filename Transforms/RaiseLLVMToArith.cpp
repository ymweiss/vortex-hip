//===- RaiseLLVMToArith.cpp - Convert LLVM to Arith dialect ----*- C++ -*-===//
//
// This pass converts LLVM dialect arithmetic operations to arith dialect:
// - llvm.mlir.constant → arith.constant
// - llvm.add → arith.addi
// - llvm.sub → arith.subi
// - llvm.mul → arith.muli
// - llvm.and → arith.andi
// - llvm.or → arith.ori
// - llvm.xor → arith.xori
// - llvm.shl → arith.shli
// - llvm.lshr → arith.shrui
// - llvm.ashr → arith.shrsi
// - llvm.udiv → arith.divui
// - llvm.sdiv → arith.divsi
// - llvm.urem → arith.remui
// - llvm.srem → arith.remsi
// - llvm.icmp → arith.cmpi
// - llvm.fcmp → arith.cmpf
// - llvm.fadd → arith.addf
// - llvm.fsub → arith.subf
// - llvm.fmul → arith.mulf
// - llvm.fdiv → arith.divf
// - llvm.frem → arith.remf
// - llvm.fneg → arith.negf
// - llvm.select → arith.select
// - llvm.zext → arith.extui
// - llvm.sext → arith.extsi
// - llvm.trunc → arith.trunci
// - llvm.uitofp → arith.uitofp
// - llvm.sitofp → arith.sitofp
// - llvm.fptoui → arith.fptoui
// - llvm.fptosi → arith.fptosi
// - llvm.fpext → arith.extf
// - llvm.fptrunc → arith.truncf
// - llvm.intr.umax → arith.maxui
// - llvm.intr.umin → arith.minui
// - llvm.intr.smax → arith.maxsi
// - llvm.intr.smin → arith.minsi
// - llvm.intr.fmuladd → arith.mulf + arith.addf
// - llvm.intr.maxnum → arith.maximumf
// - llvm.intr.minnum → arith.minimumf
// - llvm.intr.maximum → arith.maximumf
// - llvm.intr.minimum → arith.minimumf
// - llvm.intr.abs → arith.select + arith.subi + arith.cmpi (expansion)
//
//===----------------------------------------------------------------------===//

#include "dsa/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace dsa {

#define GEN_PASS_DEF_RAISELLVMTOARITH
#include "dsa/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Conversion Helpers
//===----------------------------------------------------------------------===//

/// Convert LLVM ICmp predicate to Arith CmpI predicate
arith::CmpIPredicate convertICmpPredicate(LLVM::ICmpPredicate llvmPred) {
  switch (llvmPred) {
  case LLVM::ICmpPredicate::eq:
    return arith::CmpIPredicate::eq;
  case LLVM::ICmpPredicate::ne:
    return arith::CmpIPredicate::ne;
  case LLVM::ICmpPredicate::slt:
    return arith::CmpIPredicate::slt;
  case LLVM::ICmpPredicate::sle:
    return arith::CmpIPredicate::sle;
  case LLVM::ICmpPredicate::sgt:
    return arith::CmpIPredicate::sgt;
  case LLVM::ICmpPredicate::sge:
    return arith::CmpIPredicate::sge;
  case LLVM::ICmpPredicate::ult:
    return arith::CmpIPredicate::ult;
  case LLVM::ICmpPredicate::ule:
    return arith::CmpIPredicate::ule;
  case LLVM::ICmpPredicate::ugt:
    return arith::CmpIPredicate::ugt;
  case LLVM::ICmpPredicate::uge:
    return arith::CmpIPredicate::uge;
  }
  llvm_unreachable("Unknown LLVM ICmp predicate");
}

/// Convert LLVM FCmp predicate to Arith CmpF predicate
arith::CmpFPredicate convertFCmpPredicate(LLVM::FCmpPredicate llvmPred) {
  switch (llvmPred) {
  case LLVM::FCmpPredicate::_false:
    return arith::CmpFPredicate::AlwaysFalse;
  case LLVM::FCmpPredicate::oeq:
    return arith::CmpFPredicate::OEQ;
  case LLVM::FCmpPredicate::ogt:
    return arith::CmpFPredicate::OGT;
  case LLVM::FCmpPredicate::oge:
    return arith::CmpFPredicate::OGE;
  case LLVM::FCmpPredicate::olt:
    return arith::CmpFPredicate::OLT;
  case LLVM::FCmpPredicate::ole:
    return arith::CmpFPredicate::OLE;
  case LLVM::FCmpPredicate::one:
    return arith::CmpFPredicate::ONE;
  case LLVM::FCmpPredicate::ord:
    return arith::CmpFPredicate::ORD;
  case LLVM::FCmpPredicate::ueq:
    return arith::CmpFPredicate::UEQ;
  case LLVM::FCmpPredicate::ugt:
    return arith::CmpFPredicate::UGT;
  case LLVM::FCmpPredicate::uge:
    return arith::CmpFPredicate::UGE;
  case LLVM::FCmpPredicate::ult:
    return arith::CmpFPredicate::ULT;
  case LLVM::FCmpPredicate::ule:
    return arith::CmpFPredicate::ULE;
  case LLVM::FCmpPredicate::une:
    return arith::CmpFPredicate::UNE;
  case LLVM::FCmpPredicate::uno:
    return arith::CmpFPredicate::UNO;
  case LLVM::FCmpPredicate::_true:
    return arith::CmpFPredicate::AlwaysTrue;
  }
  llvm_unreachable("Unknown LLVM FCmp predicate");
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct RaiseLLVMToArithPass
    : public impl::RaiseLLVMToArithBase<RaiseLLVMToArithPass> {
  using RaiseLLVMToArithBase::RaiseLLVMToArithBase;

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

    // Convert arithmetic operations in each function
    for (auto funcOp : funcsToProcess) {
      if (failed(convertLLVMArithOpsToArith(funcOp, builder))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  LogicalResult convertLLVMArithOpsToArith(func::FuncOp funcOp,
                                            OpBuilder &builder) {
    // Collect all operations to convert
    SmallVector<Operation *> opsToConvert;

    funcOp.walk([&](Operation *op) {
      if (isa<LLVM::ConstantOp, LLVM::AddOp, LLVM::SubOp, LLVM::MulOp,
              LLVM::AndOp, LLVM::OrOp, LLVM::XOrOp,
              LLVM::ShlOp, LLVM::LShrOp, LLVM::AShrOp, LLVM::UDivOp, LLVM::SDivOp, LLVM::URemOp, LLVM::SRemOp,
              LLVM::ICmpOp, LLVM::FCmpOp,
              LLVM::FAddOp, LLVM::FSubOp, LLVM::FMulOp, LLVM::FDivOp, LLVM::FRemOp, LLVM::FNegOp,
              LLVM::SelectOp,
              LLVM::ZExtOp, LLVM::SExtOp, LLVM::TruncOp,
              LLVM::UIToFPOp, LLVM::SIToFPOp, LLVM::FPToUIOp, LLVM::FPToSIOp,
              LLVM::FPExtOp, LLVM::FPTruncOp,
              LLVM::UMaxOp, LLVM::UMinOp, LLVM::SMaxOp, LLVM::SMinOp,
              LLVM::FMulAddOp,
              LLVM::MaxNumOp, LLVM::MinNumOp, LLVM::MaximumOp, LLVM::MinimumOp,
              LLVM::AbsOp>(op)) {
        opsToConvert.push_back(op);
      }
    });

    // Convert each operation
    for (Operation *op : opsToConvert) {
      builder.setInsertionPoint(op);

      if (auto constOp = dyn_cast<LLVM::ConstantOp>(op)) {
        // llvm.mlir.constant → arith.constant
        // Cast to TypedAttr as required by arith::ConstantOp builder
        auto arithConst = arith::ConstantOp::create(builder,
            constOp.getLoc(), cast<TypedAttr>(constOp.getValue()));
        constOp.getResult().replaceAllUsesWith(arithConst.getResult());
        op->erase();

      } else if (auto addOp = dyn_cast<LLVM::AddOp>(op)) {
        // llvm.add → arith.addi
        auto arithAdd = arith::AddIOp::create(builder, 
            addOp.getLoc(), addOp.getLhs(), addOp.getRhs());
        addOp.getResult().replaceAllUsesWith(arithAdd.getResult());
        op->erase();

      } else if (auto subOp = dyn_cast<LLVM::SubOp>(op)) {
        // llvm.sub → arith.subi
        auto arithSub = arith::SubIOp::create(builder, 
            subOp.getLoc(), subOp.getLhs(), subOp.getRhs());
        subOp.getResult().replaceAllUsesWith(arithSub.getResult());
        op->erase();

      } else if (auto mulOp = dyn_cast<LLVM::MulOp>(op)) {
        // llvm.mul → arith.muli
        auto arithMul = arith::MulIOp::create(builder, 
            mulOp.getLoc(), mulOp.getLhs(), mulOp.getRhs());
        mulOp.getResult().replaceAllUsesWith(arithMul.getResult());
        op->erase();

      } else if (auto icmpOp = dyn_cast<LLVM::ICmpOp>(op)) {
        // llvm.icmp → arith.cmpi
        arith::CmpIPredicate arithPred =
            convertICmpPredicate(icmpOp.getPredicate());
        auto arithCmp = arith::CmpIOp::create(builder, 
            icmpOp.getLoc(), arithPred, icmpOp.getLhs(), icmpOp.getRhs());
        icmpOp.getResult().replaceAllUsesWith(arithCmp.getResult());
        op->erase();

      } else if (auto zextOp = dyn_cast<LLVM::ZExtOp>(op)) {
        // llvm.zext → arith.extui
        auto arithExtUI = arith::ExtUIOp::create(builder, 
            zextOp.getLoc(), zextOp.getType(), zextOp.getArg());
        zextOp.getResult().replaceAllUsesWith(arithExtUI.getResult());
        op->erase();

      } else if (auto sextOp = dyn_cast<LLVM::SExtOp>(op)) {
        // llvm.sext → arith.extsi
        auto arithExtSI = arith::ExtSIOp::create(builder, 
            sextOp.getLoc(), sextOp.getType(), sextOp.getArg());
        sextOp.getResult().replaceAllUsesWith(arithExtSI.getResult());
        op->erase();

      } else if (auto truncOp = dyn_cast<LLVM::TruncOp>(op)) {
        // llvm.trunc → arith.trunci
        auto arithTrunc = arith::TruncIOp::create(builder, 
            truncOp.getLoc(), truncOp.getType(), truncOp.getArg());
        truncOp.getResult().replaceAllUsesWith(arithTrunc.getResult());
        op->erase();

      } else if (auto fcmpOp = dyn_cast<LLVM::FCmpOp>(op)) {
        // llvm.fcmp → arith.cmpf
        arith::CmpFPredicate arithPred =
            convertFCmpPredicate(fcmpOp.getPredicate());
        auto arithCmp = arith::CmpFOp::create(builder, 
            fcmpOp.getLoc(), arithPred, fcmpOp.getLhs(), fcmpOp.getRhs());
        fcmpOp.getResult().replaceAllUsesWith(arithCmp.getResult());
        op->erase();

      } else if (auto lshrOp = dyn_cast<LLVM::LShrOp>(op)) {
        // llvm.lshr → arith.shrui
        auto arithShrUI = arith::ShRUIOp::create(builder, 
            lshrOp.getLoc(), lshrOp.getLhs(), lshrOp.getRhs());
        lshrOp.getResult().replaceAllUsesWith(arithShrUI.getResult());
        op->erase();

      } else if (auto ashrOp = dyn_cast<LLVM::AShrOp>(op)) {
        // llvm.ashr → arith.shrsi
        auto arithShrSI = arith::ShRSIOp::create(builder, 
            ashrOp.getLoc(), ashrOp.getLhs(), ashrOp.getRhs());
        ashrOp.getResult().replaceAllUsesWith(arithShrSI.getResult());
        op->erase();

      } else if (auto selectOp = dyn_cast<LLVM::SelectOp>(op)) {
        // llvm.select → arith.select
        auto arithSelect = arith::SelectOp::create(builder, 
            selectOp.getLoc(), selectOp.getCondition(),
            selectOp.getTrueValue(), selectOp.getFalseValue());
        selectOp.getResult().replaceAllUsesWith(arithSelect.getResult());
        op->erase();

      } else if (auto andOp = dyn_cast<LLVM::AndOp>(op)) {
        // llvm.and → arith.andi
        auto arithAnd = arith::AndIOp::create(builder, 
            andOp.getLoc(), andOp.getLhs(), andOp.getRhs());
        andOp.getResult().replaceAllUsesWith(arithAnd.getResult());
        op->erase();

      } else if (auto orOp = dyn_cast<LLVM::OrOp>(op)) {
        // llvm.or → arith.ori
        auto arithOr = arith::OrIOp::create(builder, 
            orOp.getLoc(), orOp.getLhs(), orOp.getRhs());
        orOp.getResult().replaceAllUsesWith(arithOr.getResult());
        op->erase();

      } else if (auto xorOp = dyn_cast<LLVM::XOrOp>(op)) {
        // llvm.xor → arith.xori
        auto arithXor = arith::XOrIOp::create(builder, 
            xorOp.getLoc(), xorOp.getLhs(), xorOp.getRhs());
        xorOp.getResult().replaceAllUsesWith(arithXor.getResult());
        op->erase();

      } else if (auto shlOp = dyn_cast<LLVM::ShlOp>(op)) {
        // llvm.shl → arith.shli
        auto arithShl = arith::ShLIOp::create(builder, 
            shlOp.getLoc(), shlOp.getLhs(), shlOp.getRhs());
        shlOp.getResult().replaceAllUsesWith(arithShl.getResult());
        op->erase();

      } else if (auto udivOp = dyn_cast<LLVM::UDivOp>(op)) {
        // llvm.udiv → arith.divui
        auto arithDivUI = arith::DivUIOp::create(builder, 
            udivOp.getLoc(), udivOp.getLhs(), udivOp.getRhs());
        udivOp.getResult().replaceAllUsesWith(arithDivUI.getResult());
        op->erase();

      } else if (auto uremOp = dyn_cast<LLVM::URemOp>(op)) {
        // llvm.urem → arith.remui
        auto arithRemUI = arith::RemUIOp::create(builder, 
            uremOp.getLoc(), uremOp.getLhs(), uremOp.getRhs());
        uremOp.getResult().replaceAllUsesWith(arithRemUI.getResult());
        op->erase();

      } else if (auto sdivOp = dyn_cast<LLVM::SDivOp>(op)) {
        // llvm.sdiv → arith.divsi
        auto arithDivSI = arith::DivSIOp::create(builder, 
            sdivOp.getLoc(), sdivOp.getLhs(), sdivOp.getRhs());
        sdivOp.getResult().replaceAllUsesWith(arithDivSI.getResult());
        op->erase();

      } else if (auto sremOp = dyn_cast<LLVM::SRemOp>(op)) {
        // llvm.srem → arith.remsi
        auto arithRemSI = arith::RemSIOp::create(builder, 
            sremOp.getLoc(), sremOp.getLhs(), sremOp.getRhs());
        sremOp.getResult().replaceAllUsesWith(arithRemSI.getResult());
        op->erase();

      } else if (auto faddOp = dyn_cast<LLVM::FAddOp>(op)) {
        // llvm.fadd → arith.addf
        auto arithAddF = arith::AddFOp::create(builder, 
            faddOp.getLoc(), faddOp.getLhs(), faddOp.getRhs());
        faddOp.getResult().replaceAllUsesWith(arithAddF.getResult());
        op->erase();

      } else if (auto fsubOp = dyn_cast<LLVM::FSubOp>(op)) {
        // llvm.fsub → arith.subf
        auto arithSubF = arith::SubFOp::create(builder, 
            fsubOp.getLoc(), fsubOp.getLhs(), fsubOp.getRhs());
        fsubOp.getResult().replaceAllUsesWith(arithSubF.getResult());
        op->erase();

      } else if (auto fmulOp = dyn_cast<LLVM::FMulOp>(op)) {
        // llvm.fmul → arith.mulf
        auto arithMulF = arith::MulFOp::create(builder, 
            fmulOp.getLoc(), fmulOp.getLhs(), fmulOp.getRhs());
        fmulOp.getResult().replaceAllUsesWith(arithMulF.getResult());
        op->erase();

      } else if (auto fdivOp = dyn_cast<LLVM::FDivOp>(op)) {
        // llvm.fdiv → arith.divf
        auto arithDivF = arith::DivFOp::create(builder, 
            fdivOp.getLoc(), fdivOp.getLhs(), fdivOp.getRhs());
        fdivOp.getResult().replaceAllUsesWith(arithDivF.getResult());
        op->erase();

      } else if (auto fremOp = dyn_cast<LLVM::FRemOp>(op)) {
        // llvm.frem → arith.remf
        auto arithRemF = builder.create<arith::RemFOp>(
            fremOp.getLoc(), fremOp.getLhs(), fremOp.getRhs());
        fremOp.getResult().replaceAllUsesWith(arithRemF.getResult());
        op->erase();

      } else if (auto fnegOp = dyn_cast<LLVM::FNegOp>(op)) {
        // llvm.fneg → arith.negf
        auto arithNegF = arith::NegFOp::create(builder, 
            fnegOp.getLoc(), fnegOp.getOperand());
        fnegOp.getResult().replaceAllUsesWith(arithNegF.getResult());
        op->erase();

      } else if (auto uitofpOp = dyn_cast<LLVM::UIToFPOp>(op)) {
        // llvm.uitofp → arith.uitofp
        auto arithUIToFP = arith::UIToFPOp::create(builder, 
            uitofpOp.getLoc(), uitofpOp.getType(), uitofpOp.getArg());
        uitofpOp.getResult().replaceAllUsesWith(arithUIToFP.getResult());
        op->erase();

      } else if (auto fptouiOp = dyn_cast<LLVM::FPToUIOp>(op)) {
        // llvm.fptoui → arith.fptoui
        auto arithFPToUI = arith::FPToUIOp::create(builder, 
            fptouiOp.getLoc(), fptouiOp.getType(), fptouiOp.getArg());
        fptouiOp.getResult().replaceAllUsesWith(arithFPToUI.getResult());
        op->erase();

      } else if (auto sitofpOp = dyn_cast<LLVM::SIToFPOp>(op)) {
        // llvm.sitofp → arith.sitofp
        auto arithSIToFP = builder.create<arith::SIToFPOp>(
            sitofpOp.getLoc(), sitofpOp.getType(), sitofpOp.getArg());
        sitofpOp.getResult().replaceAllUsesWith(arithSIToFP.getResult());
        op->erase();

      } else if (auto fptosiOp = dyn_cast<LLVM::FPToSIOp>(op)) {
        // llvm.fptosi → arith.fptosi
        auto arithFPToSI = builder.create<arith::FPToSIOp>(
            fptosiOp.getLoc(), fptosiOp.getType(), fptosiOp.getArg());
        fptosiOp.getResult().replaceAllUsesWith(arithFPToSI.getResult());
        op->erase();

      } else if (auto fpextOp = dyn_cast<LLVM::FPExtOp>(op)) {
        // llvm.fpext → arith.extf
        auto arithExtF = builder.create<arith::ExtFOp>(
            fpextOp.getLoc(), fpextOp.getType(), fpextOp.getArg());
        fpextOp.getResult().replaceAllUsesWith(arithExtF.getResult());
        op->erase();

      } else if (auto fptruncOp = dyn_cast<LLVM::FPTruncOp>(op)) {
        // llvm.fptrunc → arith.truncf
        auto arithTruncF = builder.create<arith::TruncFOp>(
            fptruncOp.getLoc(), fptruncOp.getType(), fptruncOp.getArg());
        fptruncOp.getResult().replaceAllUsesWith(arithTruncF.getResult());
        op->erase();

      } else if (auto umaxOp = dyn_cast<LLVM::UMaxOp>(op)) {
        // llvm.intr.umax → arith.maxui
        auto arithMaxUI = arith::MaxUIOp::create(builder, 
            umaxOp.getLoc(), umaxOp.getA(), umaxOp.getB());
        umaxOp.getRes().replaceAllUsesWith(arithMaxUI.getResult());
        op->erase();

      } else if (auto uminOp = dyn_cast<LLVM::UMinOp>(op)) {
        // llvm.intr.umin → arith.minui
        auto arithMinUI = arith::MinUIOp::create(builder, 
            uminOp.getLoc(), uminOp.getA(), uminOp.getB());
        uminOp.getRes().replaceAllUsesWith(arithMinUI.getResult());
        op->erase();

      } else if (auto smaxOp = dyn_cast<LLVM::SMaxOp>(op)) {
        // llvm.intr.smax → arith.maxsi
        auto arithMaxSI = builder.create<arith::MaxSIOp>(
            smaxOp.getLoc(), smaxOp.getA(), smaxOp.getB());
        smaxOp.getRes().replaceAllUsesWith(arithMaxSI.getResult());
        op->erase();

      } else if (auto sminOp = dyn_cast<LLVM::SMinOp>(op)) {
        // llvm.intr.smin → arith.minsi
        auto arithMinSI = builder.create<arith::MinSIOp>(
            sminOp.getLoc(), sminOp.getA(), sminOp.getB());
        sminOp.getRes().replaceAllUsesWith(arithMinSI.getResult());
        op->erase();

      } else if (auto fmuladdOp = dyn_cast<LLVM::FMulAddOp>(op)) {
        // llvm.intr.fmuladd(a, b, c) → arith.mulf(a, b) + arith.addf(result, c)
        // Decompose fused multiply-add into separate operations
        auto arithMul = arith::MulFOp::create(builder, 
            fmuladdOp.getLoc(), fmuladdOp.getA(), fmuladdOp.getB());
        auto arithAdd = arith::AddFOp::create(builder, 
            fmuladdOp.getLoc(), arithMul.getResult(), fmuladdOp.getC());
        fmuladdOp.getRes().replaceAllUsesWith(arithAdd.getResult());
        op->erase();

      } else if (auto maxnumOp = dyn_cast<LLVM::MaxNumOp>(op)) {
        // llvm.intr.maxnum → arith.maximumf
        auto arithMaxF = builder.create<arith::MaximumFOp>(
            maxnumOp.getLoc(), maxnumOp.getA(), maxnumOp.getB());
        maxnumOp.getRes().replaceAllUsesWith(arithMaxF.getResult());
        op->erase();

      } else if (auto minnumOp = dyn_cast<LLVM::MinNumOp>(op)) {
        // llvm.intr.minnum → arith.minimumf
        auto arithMinF = builder.create<arith::MinimumFOp>(
            minnumOp.getLoc(), minnumOp.getA(), minnumOp.getB());
        minnumOp.getRes().replaceAllUsesWith(arithMinF.getResult());
        op->erase();

      } else if (auto maximumOp = dyn_cast<LLVM::MaximumOp>(op)) {
        // llvm.intr.maximum → arith.maximumf
        auto arithMaxF = builder.create<arith::MaximumFOp>(
            maximumOp.getLoc(), maximumOp.getA(), maximumOp.getB());
        maximumOp.getRes().replaceAllUsesWith(arithMaxF.getResult());
        op->erase();

      } else if (auto minimumOp = dyn_cast<LLVM::MinimumOp>(op)) {
        // llvm.intr.minimum → arith.minimumf
        auto arithMinF = builder.create<arith::MinimumFOp>(
            minimumOp.getLoc(), minimumOp.getA(), minimumOp.getB());
        minimumOp.getRes().replaceAllUsesWith(arithMinF.getResult());
        op->erase();

      } else if (auto absOp = dyn_cast<LLVM::AbsOp>(op)) {
        // llvm.intr.abs(x, is_int_min_poison) → select(x < 0, -x, x)
        // Expand to: abs(x) = x < 0 ? -x : x
        Type intType = absOp.getIn().getType();
        auto zero = builder.create<arith::ConstantOp>(
            absOp.getLoc(), intType, builder.getZeroAttr(intType));
        auto neg = builder.create<arith::SubIOp>(
            absOp.getLoc(), zero.getResult(), absOp.getIn());
        auto cmp = builder.create<arith::CmpIOp>(
            absOp.getLoc(), arith::CmpIPredicate::slt, absOp.getIn(), zero.getResult());
        auto select = builder.create<arith::SelectOp>(
            absOp.getLoc(), cmp.getResult(), neg.getResult(), absOp.getIn());
        absOp.getRes().replaceAllUsesWith(select.getResult());
        op->erase();
      }
    }

    return success();
  }
};

} // namespace

std::unique_ptr<Pass> createRaiseLLVMToArithPass() {
  return std::make_unique<RaiseLLVMToArithPass>();
}

} // namespace dsa
} // namespace mlir

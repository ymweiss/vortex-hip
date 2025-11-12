//===- SCFToHandshakeDSAPass.cpp - SCF to Handshake+DSA --------*- C++ -*-===//
//
// Converts SCF loops to Handshake+DSA dataflow representation
//
//===----------------------------------------------------------------------===//

#include "Common.h"
#include "AllocDeallocAnalysis.h"
#include "DeallocProcessing.h"
#include "SCFPreprocessing.h"

namespace mlir {
namespace dsa {

#define GEN_PASS_DEF_SCFTOHANDSHAKEDSA
#include "dsa/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Permissive Conversion Target and Pattern for Memory Operations
//===----------------------------------------------------------------------===//

// Custom conversion target that marks all operations as legal, similar to CIRCT's
// LowerRegionTarget. This allows creating incomplete handshake operations that
// get wired together later.
//
// NOTE: memref.get_global is explicitly marked as legal to prevent MLIR's
// conversion framework from erasing it when handshake.load no longer uses
// the memref as an operand (handshake loads connect to memory interfaces).
class MemoryConversionTarget : public ConversionTarget {
public:
  explicit MemoryConversionTarget(MLIRContext &context, Operation *funcOp)
      : ConversionTarget(context) {
    // Mark everything as legal EXCEPT the function we're converting and memref.get_global
    markUnknownOpDynamicallyLegal([funcOp](Operation *op) {
      if (op == funcOp)
        return funcConverted;
      // Special handling: always preserve memref.get_global
      if (isa<memref::GetGlobalOp>(op))
        return true;
      return true; // Everything else is legal
    });

    // Explicitly mark memref operations as legal (must come after dynamic legality)
    // NOTE: memref.get_global needs special preservation because handshake.load doesn't
    // use the memref as an operand (it connects to memory interfaces instead).
    // TODO: Long-term goal is to eliminate ALL memref operations after lowering.
    addLegalDialect<memref::MemRefDialect>();
  }
  static bool funcConverted;
};
bool MemoryConversionTarget::funcConverted = false;

// Pattern that converts memory operations in a handshake function
struct ConvertMemoryOps : public OpConversionPattern<circt::handshake::FuncOp> {
  using OpConversionPattern<circt::handshake::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(circt::handshake::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefToMemoryAccessOp memRefOps;

    // Replace memref.load/store with handshake.load/store
    if (failed(replaceMemoryOps(funcOp, rewriter, memRefOps))) {
      return failure();
    }

    // Wire up memory operations
    if (failed(connectToMemory(funcOp, rewriter, memRefOps))) {
      return failure();
    }

    MemoryConversionTarget::funcConverted = true;
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SCFToHandshakeDSA Pass Implementation
//===----------------------------------------------------------------------===//

struct SCFToHandshakeDSAPass
    : public impl::SCFToHandshakeDSABase<SCFToHandshakeDSAPass> {

  // Tracker for allocations and deallocations
  AllocDeallocTracker allocTracker;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    // Initialize the SCF conversion registry
    initializeSCFConversionRegistry(context);

    // Clear state from previous runs
    allocTracker.clear();

    // Phase 1: Analyze alloc/dealloc pairs before SCF conversion
    OpBuilder builder(context);
    allocTracker.analyzeAllocDealloc(module, builder);

    // Phase 2: Preprocess SCF metadata (CRITICAL: before conversion)
    preprocessSCFMetadata(module, builder);

    // Phase 3: Convert SCF to Handshake
    ConversionTarget target(*context);

    // Mark SCF dialect and func.func as illegal (we want to convert them)
    target.addIllegalDialect<scf::SCFDialect>();
    target.addIllegalOp<func::FuncOp, func::ReturnOp>();

    // Mark DSA, Arith, Handshake and other dialects as legal
    target.addLegalDialect<dsa::DSADialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<circt::handshake::HandshakeDialect>();
    target.addLegalDialect<math::MathDialect>();

    // Keep memref operations legal during pattern conversion
    target.addLegalDialect<memref::MemRefDialect>();
    // Keep LLVM dialect legal
    target.addLegalDialect<LLVM::LLVMDialect>();

    // llvm.return is illegal ONLY if it's in a handshake.func
    target.addDynamicallyLegalOp<LLVM::ReturnOp>([](LLVM::ReturnOp op) {
      return !op->getParentOfType<circt::handshake::FuncOp>();
    });

    // Convert func.func to handshake.func first, then convert SCF operations
    RewritePatternSet patterns(context);
    populateFuncConversionPatterns(patterns);
    populateLoopConversionPatterns(patterns);
    populateParallelLoopConversionPatterns(patterns);
    populateControlFlowConversionPatterns(patterns);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // Phase 4: Convert memory operations and post-process
    for (auto funcOp : module.getOps<circt::handshake::FuncOp>()) {
      MemoryConversionTarget::funcConverted = false;
      MemoryConversionTarget memTarget(*context, funcOp);
      RewritePatternSet memPatterns(context);
      memPatterns.add<ConvertMemoryOps>(context);

      if (failed(applyPartialConversion(funcOp, memTarget, std::move(memPatterns)))) {
        signalPassFailure();
        return;
      }

      // Insert fork operations for multi-use values
      OpBuilder funcBuilder(context);
      if (failed(insertForkOperations(funcOp, funcBuilder))) {
        signalPassFailure();
        return;
      }

      // Optimize handshake+DSA constructs
      if (failed(optimizeHandshakeDSA(funcOp, funcBuilder))) {
        signalPassFailure();
        return;
      }

      // Iterative optimization loop
      for (int iter = 0; iter < 5; ++iter) {
        if (failed(sinkUnconsumedValues(funcOp, funcBuilder))) {
          signalPassFailure();
          return;
        }

        if (failed(optimizeForkOperations(funcOp, funcBuilder))) {
          signalPassFailure();
          return;
        }

        if (failed(eliminateDeadCode(funcOp, funcBuilder))) {
          signalPassFailure();
          return;
        }
      }

      // Insert forks again after optimization
      if (failed(insertForkOperations(funcOp, funcBuilder))) {
        signalPassFailure();
        return;
      }

      // Final optimization pass
      if (failed(sinkUnconsumedValues(funcOp, funcBuilder))) {
        signalPassFailure();
        return;
      }
      if (failed(optimizeForkOperations(funcOp, funcBuilder))) {
        signalPassFailure();
        return;
      }

      // Phase 5: Process dealloc operations
      if (failed(processDeallocsInFunction(funcOp, funcBuilder, allocTracker))) {
        signalPassFailure();
        return;
      }

      // Phase 6: Insert stack deallocations
      if (failed(insertStackDeallocations(funcOp, funcBuilder, allocTracker))) {
        signalPassFailure();
        return;
      }
    }

    // Phase 7: Verify control values and print metadata
    verifyAndPrintMetadata(module);
  }

  //===----------------------------------------------------------------------===//
  // Verification and Debug Output
  //===----------------------------------------------------------------------===//

  void verifyAndPrintMetadata(ModuleOp module) {
    // Verify control value correctness
    DSA_DEBUG_STREAM << "\n========================================\n";
    DSA_DEBUG_STREAM << "[SCF HIERARCHY] Verifying control values\n";
    DSA_DEBUG_STREAM << "========================================\n";

    bool allControlValuesValid = true;
    module.walk([&](Operation *op) {
      if (!op->hasAttr("dsa.scf_region")) return;

      auto regionAttr = op->getAttrOfType<StringAttr>("dsa.scf_region");
      std::string region = regionAttr.getValue().str();

      // Parse region type
      size_t firstDot = region.find('.');
      if (firstDot == std::string::npos) {
        DSA_DEBUG_STREAM << "[ERROR] Invalid scf_region format: " << region << "\n";
        allControlValuesValid = false;
        return;
      }

      std::string scfType = region.substr(0, firstDot);
      std::string remaining = region.substr(firstDot + 1);

      // Determine branch type
      std::string branchType;
      if (scfType == "if") {
        branchType = "if";
      } else if (scfType == "while") {
        if (remaining.find(".before") != std::string::npos) {
          branchType = "while.before";
        } else if (remaining.find(".after") != std::string::npos) {
          branchType = "while.after";
        }
      } else if (scfType == "for") {
        branchType = "for.body";
      }

      DSA_DEBUG_STREAM << "[VERIFY] Checking control value for " << region
                       << " | op=" << op->getName() << "\n";

      // Verify based on type
      if (branchType == "if") {
        bool foundCondBr = false;
        for (auto user : op->getUsers()) {
          if (isa<circt::handshake::ForkOp>(user)) {
            for (auto forkUser : user->getUsers()) {
              if (auto condBr = dyn_cast<circt::handshake::ConditionalBranchOp>(forkUser)) {
                if (condBr.getConditionOperand() == user->getResult(0) ||
                    std::find(user->getResults().begin(), user->getResults().end(),
                             condBr.getConditionOperand()) != user->getResults().end()) {
                  foundCondBr = true;
                  break;
                }
              }
            }
          } else if (auto condBr = dyn_cast<circt::handshake::ConditionalBranchOp>(user)) {
            if (condBr.getConditionOperand() == op->getResult(0)) {
              foundCondBr = true;
            }
          }
          if (foundCondBr) break;
        }

        if (!foundCondBr) {
          DSA_DEBUG_STREAM << "[ERROR] scf.if control value does not drive cond_br: "
                           << region << "\n";
          allControlValuesValid = false;
        }

      } else if (branchType == "while.before") {
        bool foundInvariant = false;
        for (auto user : op->getUsers()) {
          if (isa<circt::handshake::ForkOp>(user)) {
            for (auto forkUser : user->getUsers()) {
              if (auto invOp = dyn_cast<dsa::InvariantOp>(forkUser)) {
                if (invOp.getD() == user->getResult(0) ||
                    std::find(user->getResults().begin(), user->getResults().end(),
                             invOp.getD()) != user->getResults().end()) {
                  foundInvariant = true;
                  break;
                }
              }
            }
          } else if (auto invOp = dyn_cast<dsa::InvariantOp>(user)) {
            if (invOp.getD() == op->getResult(0)) {
              foundInvariant = true;
            }
          }
          if (foundInvariant) break;
        }

        if (!foundInvariant) {
          DSA_DEBUG_STREAM << "[ERROR] while.before control value does not drive dsa.invariant: "
                           << region << "\n";
          allControlValuesValid = false;
        }

      } else if (branchType == "while.after") {
        if (auto gateOp = dyn_cast_or_null<dsa::GateOp>(op)) {
          if (op->getNumResults() < 2) {
            DSA_DEBUG_STREAM << "[ERROR] while.after control value's dsa.gate has < 2 outputs: "
                             << region << "\n";
            allControlValuesValid = false;
          }
        } else {
          DSA_DEBUG_STREAM << "[ERROR] while.after control value is not from dsa.gate: "
                           << region << " | op=" << op->getName() << "\n";
          allControlValuesValid = false;
        }

      } else if (branchType == "for.body") {
        if (auto gateOp = dyn_cast_or_null<dsa::GateOp>(op)) {
          if (op->getNumResults() < 2) {
            DSA_DEBUG_STREAM << "[ERROR] for.body control value's dsa.gate has < 2 outputs: "
                             << region << "\n";
            allControlValuesValid = false;
          } else {
            Value gateCondInput = gateOp.getOperand(1);
            bool foundStream = false;

            Value current = gateCondInput;
            for (int depth = 0; depth < 5 && current; ++depth) {
              if (auto defOp = current.getDefiningOp()) {
                if (auto streamOp = dyn_cast<dsa::StreamOp>(defOp)) {
                  if (current == streamOp.getResult(1)) {
                    foundStream = true;
                    break;
                  }
                } else if (auto forkOp = dyn_cast<circt::handshake::ForkOp>(defOp)) {
                  current = forkOp.getOperand();
                } else {
                  break;
                }
              } else {
                break;
              }
            }

            if (!foundStream) {
              DSA_DEBUG_STREAM << "[WARN] for.body control value's dsa.gate input may not be from dsa.stream: "
                               << region << "\n";
            }
          }
        } else {
          DSA_DEBUG_STREAM << "[ERROR] for.body control value is not from dsa.gate: "
                           << region << " | op=" << op->getName() << "\n";
          allControlValuesValid = false;
        }
      }
    });

    if (!allControlValuesValid) {
      DSA_DEBUG_STREAM << "[SCF HIERARCHY] Control value verification FAILED\n";
      DSA_DEBUG_STREAM << "========================================\n\n";
      signalPassFailure();
      return;
    }

    DSA_DEBUG_STREAM << "[SCF HIERARCHY] Control value verification PASSED\n";
    DSA_DEBUG_STREAM << "========================================\n\n";

    // Print SCF metadata
    printSCFMetadata(module);
  }

  void printSCFMetadata(ModuleOp module) {
    DSA_DEBUG_STREAM << "\n========================================\n";
    DSA_DEBUG_STREAM << "[SCF HIERARCHY] Post-conversion metadata dump\n";
    DSA_DEBUG_STREAM << "========================================\n";

    // Assign global memory operation indices
    DenseMap<Operation*, unsigned> memOpIndex;
    unsigned globalMemOpIdx = 1;

    module.walk([&](Operation *op) {
      if (isa<circt::handshake::LoadOp, circt::handshake::StoreOp>(op)) {
        memOpIndex[op] = globalMemOpIdx++;
      }
    });

    // Print metadata for handshake.load/store and control values
    module.walk([&](Operation *op) {
      bool isHandshakeMemOp = isa<circt::handshake::LoadOp, circt::handshake::StoreOp>(op);
      bool isControlValue = op->hasAttr("dsa.scf_region");

      if (!isHandshakeMemOp && !isControlValue) {
        return;
      }

      std::string opName = op->getName().getStringRef().str();
      const int opNameWidth = 25;
      const int seqSectionWidth = 50;
      const int typeSectionWidth = 25;

      std::string debugOutput = "[SCF HIERARCHY] op=";
      debugOutput += opName;

      if (opName.length() < opNameWidth) {
        debugOutput.append(opNameWidth - opName.length(), ' ');
      }

      debugOutput += " | ";

      // Build seq section
      std::string seqSection;

      if (auto globalSeqAttr = op->getAttrOfType<IntegerAttr>("dsa.global_seq")) {
        seqSection += "global_seq=" + std::to_string(globalSeqAttr.getInt()) + ", ";
      }

      if (auto topSeqAttr = op->getAttrOfType<IntegerAttr>("dsa.top_seq")) {
        seqSection += "top_seq=" + std::to_string(topSeqAttr.getInt()) + ", ";
      }

      if (auto localSeqAttr = op->getAttrOfType<IntegerAttr>("dsa.local_seq")) {
        seqSection += "local_seq=" + std::to_string(localSeqAttr.getInt());
      }

      debugOutput += seqSection;
      if (seqSection.length() < seqSectionWidth) {
        debugOutput.append(seqSectionWidth - seqSection.length(), ' ');
      }

      debugOutput += " | ";

      // Build type section
      std::string typeSection;

      if (isHandshakeMemOp) {
        if (isa<circt::handshake::LoadOp>(op)) {
          typeSection = "LOAD#" + std::to_string(memOpIndex[op]);
        } else if (isa<circt::handshake::StoreOp>(op)) {
          typeSection = "STORE#" + std::to_string(memOpIndex[op]);
        }
      } else if (isControlValue) {
        if (auto regionAttr = op->getAttrOfType<StringAttr>("dsa.scf_region")) {
          typeSection = regionAttr.getValue().str();
        }
      }

      debugOutput += typeSection;
      if (typeSection.length() < typeSectionWidth) {
        debugOutput.append(typeSectionWidth - typeSection.length(), ' ');
      }

      debugOutput += " | ";

      // Add scf_path
      if (auto pathAttr = op->getAttrOfType<ArrayAttr>("dsa.scf_path")) {
        debugOutput += "scf_path=[";
        bool first = true;
        for (auto attr : pathAttr) {
          if (auto strAttr = dyn_cast<StringAttr>(attr)) {
            if (!first) debugOutput += ", ";
            debugOutput += "\"" + strAttr.getValue().str() + "\"";
            first = false;
          }
        }
        debugOutput += "]";
      }

      DSA_DEBUG_STREAM << debugOutput << "\n";
    });

    DSA_DEBUG_STREAM << "========================================\n";
    DSA_DEBUG_STREAM << "[SCF HIERARCHY] Post-conversion dump complete\n";
    DSA_DEBUG_STREAM << "========================================\n\n";
  }
};

} // namespace

std::unique_ptr<Pass> createSCFToHandshakeDSAPass() {
  return std::make_unique<SCFToHandshakeDSAPass>();
}

} // namespace dsa
} // namespace mlir

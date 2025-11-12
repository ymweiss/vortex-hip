# DSA Transforms Library

# Create output directory for generated headers
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/include/dsa/Transforms)

set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/include/dsa/Transforms/Passes.td)
mlir_tablegen(${PROJECT_BINARY_DIR}/include/dsa/Transforms/Passes.h.inc -gen-pass-decls -name DSATransforms)
mlir_tablegen(${PROJECT_BINARY_DIR}/include/dsa/Transforms/Passes.capi.h.inc -gen-pass-capi-header --prefix DSA)
mlir_tablegen(${PROJECT_BINARY_DIR}/include/dsa/Transforms/Passes.capi.cpp.inc -gen-pass-capi-impl --prefix DSA)
add_public_tablegen_target(MLIRDSATransformsIncGen)

add_mlir_library(MLIRDSATransforms
  RaiseLLVMToCF.cpp
  RaiseLLVMToMemRef/RaiseLLVMToMemRefPass.cpp
  RaiseLLVMToMemRef/TypeInference.cpp
  RaiseLLVMToMemRef/GEPAnalysis.cpp
  RaiseLLVMToMemRef/MemoryOperations.cpp
  RaiseLLVMToMemRef/IntrinsicOps.cpp
  RaiseLLVMToMemRef/AllocaConversion.cpp
  RaiseLLVMToMemRef/GlobalOps.cpp
  RaiseLLVMToArith.cpp
  RaiseLLVMToMath.cpp
  NormalizeSCFLoops/IVPatternHelpers.cpp
  NormalizeSCFLoops/DetectParallelForPattern.cpp
  NormalizeSCFLoops/ForallToForPattern.cpp
  NormalizeSCFLoops/IndexSwitchToIfPattern.cpp
  NormalizeSCFLoops/WhileToForPattern.cpp
  NormalizeSCFLoops/NormalizeSCFLoopsPass.cpp
  UnrollParallelLoops.cpp
  SCFToHandshakeDSA/MemoryOpsHelpers.cpp
  SCFToHandshakeDSA/SerialRecursiveMemCtrl.cpp
  SCFToHandshakeDSA/MemoryConnection.cpp
  SCFToHandshakeDSA/PassHelpers.cpp
  SCFToHandshakeDSA/HandshakeOptimizations.cpp
  SCFToHandshakeDSA/FuncConversion.cpp
  SCFToHandshakeDSA/SCFConversionHelpers.cpp
  SCFToHandshakeDSA/SCFLoopConversion.cpp
  SCFToHandshakeDSA/SCFControlFlowConversion.cpp
  SCFToHandshakeDSA/SCFNestedConversion.cpp
  SCFToHandshakeDSA/ConversionRegistry.cpp
  SCFToHandshakeDSA/ParallelismAnalysis.cpp
  SCFToHandshakeDSA/ParallelLoops.cpp
  SCFToHandshakeDSA/AllocDeallocAnalysis.cpp
  SCFToHandshakeDSA/DeallocProcessing.cpp
  SCFToHandshakeDSA/SCFPreprocessing.cpp
  SCFToHandshakeDSA/SCFToHandshakeDSAPass.cpp
  Passes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/dsa/Transforms
  ${PROJECT_BINARY_DIR}/include/dsa/Transforms

  DEPENDS
  MLIRDSATransformsIncGen

  LINK_LIBS PUBLIC
  MLIRDSADialect
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRControlFlowDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRArithDialect
  MLIRMathDialect
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRSupport
  MLIRTransformUtils
  CIRCTHandshake
)

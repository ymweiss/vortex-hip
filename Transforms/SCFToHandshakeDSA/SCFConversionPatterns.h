//===- SCFConversionPatterns.h - SCF conversion pattern classes -*- C++ -*-===//
//
// Pattern class declarations for SCF to Handshake+DSA conversion
//
//===----------------------------------------------------------------------===//

#ifndef LIB_DSA_TRANSFORMS_SCFTOHANDSHAKEDSA_SCFCONVERSIONPATTERNS_H
#define LIB_DSA_TRANSFORMS_SCFTOHANDSHAKEDSA_SCFCONVERSIONPATTERNS_H

#include "Common.h"
#include "SCFConversionHelpers.h"
#include "ConversionRegistry.h"

namespace mlir {
namespace dsa {

//===----------------------------------------------------------------------===//
// SCF ForOp Conversion Pattern
//===----------------------------------------------------------------------===//

struct ConvertForOp : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// SCF WhileOp Conversion Pattern
//===----------------------------------------------------------------------===//

struct ConvertWhileOp : public OpConversionPattern<scf::WhileOp> {
  using OpConversionPattern<scf::WhileOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(scf::WhileOp whileOp, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// SCF IfOp Conversion Pattern
//===----------------------------------------------------------------------===//

struct ConvertIfOp : public OpConversionPattern<scf::IfOp> {
  using OpConversionPattern<scf::IfOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(scf::IfOp ifOp, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// SCF IndexSwitchOp Conversion Pattern
//===----------------------------------------------------------------------===//

struct ConvertIndexSwitchOp : public OpConversionPattern<scf::IndexSwitchOp> {
  using OpConversionPattern<scf::IndexSwitchOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(scf::IndexSwitchOp switchOp, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override;
};

} // namespace dsa
} // namespace mlir

#endif // LIB_DSA_TRANSFORMS_SCFTOHANDSHAKEDSA_SCFCONVERSIONPATTERNS_H

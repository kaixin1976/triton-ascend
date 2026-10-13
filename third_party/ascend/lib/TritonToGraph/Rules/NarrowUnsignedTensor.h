// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_ASCEND_GRAPH_NARROW_UNSIGNED_TENSOR_H
#define TRITON_ASCEND_GRAPH_NARROW_UNSIGNED_TENSOR_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include <cstdint>

namespace mlir::triton {
namespace narrow_unsigned_tensor {

inline bool constant(Value value, uint64_t &bits) {
  APInt integer;
  if (!matchPattern(value, m_ConstantInt(&integer)))
    return false;
  bits = integer.getZExtValue();
  return true;
}

// Only values proven to have zero upper bits may change signed right shift
// into unsigned right shift. In particular, an arbitrary i64 is NOT a u32.
inline bool isU32(Value value) {
  if (auto ext = value.getDefiningOp<arith::ExtUIOp>())
    return getElementTypeOrSelf(ext.getIn().getType()).isInteger(32);
  uint64_t bits;
  return constant(value, bits) && bits <= UINT32_MAX;
}

inline Type i32Type(Type type, RewriterBase &rewriter) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return tensor.clone(rewriter.getI32Type());
  return rewriter.getI32Type();
}

// Build a narrowed bitwise node while eliminating identities that generic
// arith folding does not currently recognize for dense tensor splats.
inline Value createBitwise(Operation *source, Value lhs, Value rhs,
                           RewriterBase &rewriter, Location location) {
  uint64_t lhsBits = 0;
  uint64_t rhsBits = 0;
  const bool lhsConstant = constant(lhs, lhsBits);
  const bool rhsConstant = constant(rhs, rhsBits);

  if (isa<arith::AndIOp>(source)) {
    if (lhsConstant && lhsBits == UINT32_MAX)
      return rhs;
    if (rhsConstant && rhsBits == UINT32_MAX)
      return lhs;
    if (lhsConstant && lhsBits == 0)
      return lhs;
    if (rhsConstant && rhsBits == 0)
      return rhs;
    return rewriter.createOrFold<arith::AndIOp>(location, lhs, rhs);
  }
  if (isa<arith::OrIOp>(source)) {
    if (lhsConstant && lhsBits == 0)
      return rhs;
    if (rhsConstant && rhsBits == 0)
      return lhs;
    if (lhsConstant && lhsBits == UINT32_MAX)
      return lhs;
    if (rhsConstant && rhsBits == UINT32_MAX)
      return rhs;
    return rewriter.createOrFold<arith::OrIOp>(location, lhs, rhs);
  }
  if (lhsConstant && lhsBits == 0)
    return rhs;
  if (rhsConstant && rhsBits == 0)
    return lhs;
  return rewriter.createOrFold<arith::XOrIOp>(location, lhs, rhs);
}

inline bool isTruncatableArithmetic(Operation *op) {
  // Sinking a truncation through a shared producer would duplicate arithmetic
  // while retaining the original i64 operation for its other users.
  if (!op || !op->hasOneUse())
    return false;
  if (auto add = dyn_cast<arith::AddIOp>(op))
    return add.getOverflowFlags() == arith::IntegerOverflowFlags::none;
  if (auto sub = dyn_cast<arith::SubIOp>(op))
    return sub.getOverflowFlags() == arith::IntegerOverflowFlags::none;
  return false;
}

// Materialize the low 32 bits at this value without distributing the
// truncation into its producer graph. This is important for TopK's bitonic
// network: expanding through its XOR chain replaces one truncation with many
// trunc/xor operations and regresses the generated kernel.
inline Value truncateHere(Value value, RewriterBase &rewriter) {
  Location loc = value.getLoc();
  Type type = i32Type(value.getType(), rewriter);
  if (value.getType() == type)
    return value;
  if (auto ext = value.getDefiningOp<arith::ExtUIOp>())
    if (ext.getIn().getType() == type)
      return ext.getIn();
  if (auto ext = value.getDefiningOp<arith::ExtSIOp>())
    if (ext.getIn().getType() == type)
      return ext.getIn();
  uint64_t bits;
  if (constant(value, bits)) {
    auto attr =
        rewriter.getIntegerAttr(rewriter.getI32Type(), bits & UINT32_MAX);
    if (auto tensor = dyn_cast<RankedTensorType>(type))
      return rewriter.create<arith::ConstantOp>(
          loc, type, DenseElementsAttr::get(tensor, attr));
    return rewriter.create<arith::ConstantOp>(loc, attr);
  }
  return rewriter.create<arith::TruncIOp>(loc, type, value);
}

// Compute the low 32 bits without materializing wide bitwise temporaries.
// Truncation commutes with these operations, but not with right shift.
inline Value lowBits(Value value, RewriterBase &rewriter, unsigned depth = 0) {
  Location loc = value.getLoc();
  Type type = i32Type(value.getType(), rewriter);
  if (value.getType() == type)
    return value;
  if (auto ext = value.getDefiningOp<arith::ExtUIOp>())
    if (ext.getIn().getType() == type)
      return ext.getIn();
  if (auto ext = value.getDefiningOp<arith::ExtSIOp>())
    if (ext.getIn().getType() == type)
      return ext.getIn();
  uint64_t bits;
  if (constant(value, bits)) {
    auto attr =
        rewriter.getIntegerAttr(rewriter.getI32Type(), bits & UINT32_MAX);
    if (auto tensor = dyn_cast<RankedTensorType>(type))
      return rewriter.create<arith::ConstantOp>(
          loc, type, DenseElementsAttr::get(tensor, attr));
    return rewriter.create<arith::ConstantOp>(loc, attr);
  }
  if (depth < 8) {
    if (auto op = value.getDefiningOp<SplatOp>())
      return rewriter.create<SplatOp>(
          loc, type, lowBits(op.getSrc(), rewriter, depth + 1));
    Operation *op = value.getDefiningOp();
    if (op && isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op)) {
      uint64_t lhsBits = 0;
      uint64_t rhsBits = 0;
      const bool lhsConstant = constant(op->getOperand(0), lhsBits);
      const bool rhsConstant = constant(op->getOperand(1), rhsBits);
      const uint64_t lhsLowBits = lhsBits & UINT32_MAX;
      const uint64_t rhsLowBits = rhsBits & UINT32_MAX;
      if (isa<arith::AndIOp>(op)) {
        if (lhsConstant && lhsLowBits == UINT32_MAX)
          return truncateHere(op->getOperand(1), rewriter);
        if (rhsConstant && rhsLowBits == UINT32_MAX)
          return truncateHere(op->getOperand(0), rewriter);
      } else if (isa<arith::OrIOp, arith::XOrIOp>(op)) {
        if (lhsConstant && lhsLowBits == 0)
          return lowBits(op->getOperand(1), rewriter, depth + 1);
        if (rhsConstant && rhsLowBits == 0)
          return lowBits(op->getOperand(0), rewriter, depth + 1);
      }
      Value lhs = lowBits(op->getOperand(0), rewriter, depth + 1);
      Value rhs = lowBits(op->getOperand(1), rewriter, depth + 1);
      return createBitwise(op, lhs, rhs, rewriter, loc);
    }
  }
  return rewriter.create<arith::TruncIOp>(loc, type, value);
}

// Return low bits only when they are already materialized as i32 (or are a
// constant). In particular, do not recursively reopen a bitwise DAG here:
// TopK's bitonic network is large enough that moving its truncation boundary
// by one level makes downstream layout analysis pathologically expensive.
inline bool canMaterializeLowBits(Value value) {
  if (getElementTypeOrSelf(value.getType()).isInteger(32))
    return true;
  if (auto ext = value.getDefiningOp<arith::ExtUIOp>())
    return getElementTypeOrSelf(ext.getIn().getType()).isInteger(32);
  if (auto ext = value.getDefiningOp<arith::ExtSIOp>())
    return getElementTypeOrSelf(ext.getIn().getType()).isInteger(32);
  uint64_t bits;
  return constant(value, bits);
}

inline Value materializedLowBits(Value value, RewriterBase &rewriter) {
  Type type = i32Type(value.getType(), rewriter);
  if (value.getType() == type)
    return value;
  if (auto ext = value.getDefiningOp<arith::ExtUIOp>())
    if (ext.getIn().getType() == type)
      return ext.getIn();
  if (auto ext = value.getDefiningOp<arith::ExtSIOp>())
    if (ext.getIn().getType() == type)
      return ext.getIn();

  uint64_t bits;
  if (!constant(value, bits))
    return {};
  auto attr =
      rewriter.getIntegerAttr(rewriter.getI32Type(), bits & UINT32_MAX);
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return rewriter.create<arith::ConstantOp>(
        value.getLoc(), type, DenseElementsAttr::get(tensor, attr));
  return rewriter.create<arith::ConstantOp>(value.getLoc(), attr);
}

struct Narrow : RewritePattern {
  explicit Narrow(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() < 2)
      return failure();
    auto type = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    // Select has a predicate operand first.
    if (isa<arith::SelectOp>(op))
      type = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!type || !type.getElementType().isInteger(64))
      return failure();
    Location loc = op->getLoc();
    Value result;
    if (auto select = dyn_cast<arith::SelectOp>(op)) {
      if (!isU32(select.getTrueValue()) || !isU32(select.getFalseValue()))
        return failure();
      result = rewriter.create<arith::SelectOp>(
          loc, select.getCondition(), lowBits(select.getTrueValue(), rewriter),
          lowBits(select.getFalseValue(), rewriter));
    } else if (isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op)) {
      Value lhs = op->getOperand(0), rhs = op->getOperand(1);
      uint64_t mask;
      bool masked = isa<arith::AndIOp>(op) &&
                    ((constant(lhs, mask) && mask <= UINT32_MAX) ||
                     (constant(rhs, mask) && mask <= UINT32_MAX));
      if (!masked && !(isU32(lhs) && isU32(rhs)))
        return failure();
      result = lowBits(op->getResult(0), rewriter);
    } else if (isa<arith::ShRSIOp, arith::ShRUIOp>(op)) {
      uint64_t shift;
      if (!isU32(op->getOperand(0)) || !constant(op->getOperand(1), shift) ||
          shift >= 32)
        return failure();
      result = rewriter.create<arith::ShRUIOp>(
          loc, lowBits(op->getOperand(0), rewriter),
          lowBits(op->getOperand(1), rewriter));
    } else if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
      if (cmp.getPredicate() != arith::CmpIPredicate::eq ||
          !isU32(cmp.getLhs()))
        return failure();
      if (isU32(cmp.getRhs())) {
        rewriter.replaceOpWithNewOp<arith::CmpIOp>(
            op, cmp.getPredicate(), lowBits(cmp.getLhs(), rewriter),
            lowBits(cmp.getRhs(), rewriter));
        return success();
      }
      // A scalar threshold may still be i64: keep its upper-bit check scalar,
      // instead of silently truncating or creating a tensor<i64> comparison.
      auto splat = cmp.getRhs().getDefiningOp<SplatOp>();
      if (!splat)
        return failure();
      Value limit = rewriter.create<arith::ConstantIntOp>(loc, UINT32_MAX, 64);
      Value fits = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ule, splat.getSrc(), limit);
      Value equal = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, lowBits(cmp.getLhs(), rewriter),
          lowBits(cmp.getRhs(), rewriter));
      Value fitsTensor = rewriter.create<SplatOp>(loc, equal.getType(), fits);
      rewriter.replaceOpWithNewOp<arith::AndIOp>(op, equal, fitsTensor);
      return success();
    } else {
      return failure();
    }
    rewriter.replaceOpWithNewOp<arith::ExtUIOp>(op, type, result);
    return success();
  }
};

// Keep a low-bit consumer in i32 instead of forcing the narrowed producer back
// to i64. This removes chains such as
//
//   i32 -> extui i64 -> subi i64 -> trunci i32
//
// while preserving the original modulo-2^32 result. The i64 arithmetic is
// retained when it has another consumer or carries an overflow promise.
struct SinkTruncation : OpRewritePattern<arith::TruncIOp> {
  explicit SinkTruncation(MLIRContext *context)
      : OpRewritePattern<arith::TruncIOp>(context, 2) {}

  LogicalResult matchAndRewrite(arith::TruncIOp op,
                                PatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getIn().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!inputType || !resultType ||
        !inputType.getElementType().isInteger(64) ||
        !resultType.getElementType().isInteger(32) ||
        inputType.getShape() != resultType.getShape() ||
        !isTruncatableArithmetic(op.getIn().getDefiningOp()))
      return failure();

    Operation *producer = op.getIn().getDefiningOp();
    if (!canMaterializeLowBits(producer->getOperand(0)) ||
        !canMaterializeLowBits(producer->getOperand(1)))
      return failure();
    Value lhs = materializedLowBits(producer->getOperand(0), rewriter);
    Value rhs = materializedLowBits(producer->getOperand(1), rewriter);
    assert(lhs && rhs && "prechecked low bits must materialize");

    if (isa<arith::AddIOp>(producer))
      rewriter.replaceOpWithNewOp<arith::AddIOp>(op, lhs, rhs);
    else
      rewriter.replaceOpWithNewOp<arith::SubIOp>(op, lhs, rhs);
    return success();
  }
};
} // namespace narrow_unsigned_tensor
} // namespace mlir::triton
#endif

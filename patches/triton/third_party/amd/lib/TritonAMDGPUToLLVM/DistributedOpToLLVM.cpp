/*
 * Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "Dialect/Distributed/IR/Dialect.h"

#include "Utility.h"
#include <string>

using namespace mlir;
using namespace mlir::triton;
using namespace std::literals;

namespace {

bool useROCSHMEMLibrary(StringRef libname){
  return libname == "librocshmem_device";
}

template <typename DistOp>
class GenericOpToROCSHMEMDevice : public ConvertOpToLLVMPattern<DistOp> {
public:
  using OpAdaptor = typename DistOp::Adaptor;

  GenericOpToROCSHMEMDevice(const LLVMTypeConverter &converter,
                            const PatternBenefit &benefit, StringRef calleeName,
                            StringRef libname = "", StringRef libpath = "")
      : ConvertOpToLLVMPattern<DistOp>(converter, benefit),
        calleeName(calleeName), libname(libname), libpath(libpath) {}

  LogicalResult
  matchAndRewrite(DistOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    if (op->getNumResults() > 1)
      return failure();
    LLVM::LLVMVoidType voidTy = void_ty(op->getContext());
    auto newOperands = adaptor.getOperands();
    Type retType =
        op->getNumResults() == 0
            ? voidTy
            : this->getTypeConverter()->convertType(op->getResult(0).getType());
    
    Type funcType = mlir::triton::gpu::getFunctionType(retType, newOperands);
    LLVM::LLVMFuncOp funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
        rewriter, op, calleeName, funcType, libname, libpath);

    auto newResult =
        LLVM::createLLVMCallOp(rewriter, loc, funcOp, newOperands).getResult();
    
    
    if (op->getNumResults() == 0) {
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, newResult);
    }

    return success();
  }

private:
  StringRef calleeName;
  StringRef libname;
  StringRef libpath;
};

class ExternCallConversion
    : public ConvertOpToLLVMPattern<triton::distributed::ExternCallOp> {
public:
  ExternCallConversion(const LLVMTypeConverter &converter,
                       const PatternBenefit &benefit)
      : ConvertOpToLLVMPattern<triton::distributed::ExternCallOp>(converter,
                                                                  benefit) {}

  LogicalResult
  matchAndRewrite(triton::distributed::ExternCallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    if (op->getNumResults() > 1) {
      llvm::errs() << "ExternCallConversion does not support multi outs.";
      return failure();
    }

    LLVM::LLVMVoidType voidTy = void_ty(op->getContext());
    auto newOperands = adaptor.getOperands();
    Type retType =
        op->getNumResults() == 0
            ? voidTy
            : this->getTypeConverter()->convertType(op->getResult(0).getType());
    std::string funcName = op.getSymbol().str();

    StringRef libname = op.getLibname();
    StringRef libpath = op.getLibpath();

    SmallVector<Value> llvmOpearands;

    for (auto val : newOperands) {
      if (auto ptrTy = llvm::dyn_cast<LLVM::LLVMPointerType>(val.getType())) {
        assert((ptrTy.getAddressSpace() == 0 || ptrTy.getAddressSpace() == 1) &&
               "wrong address space.");
        Value ptrAfterCast = val;
        ptrAfterCast = rewriter.create<LLVM::AddrSpaceCastOp>(
            loc, LLVM::LLVMPointerType::get(rewriter.getContext()), val);
        llvmOpearands.push_back(ptrAfterCast);
      } else {
        llvmOpearands.push_back(val);
      }
    }

    Type llvmRetType = retType;
    if (auto retPtrType = llvm::dyn_cast<LLVM::LLVMPointerType>(retType)) {
      assert((retPtrType.getAddressSpace() == 0 ||
              retPtrType.getAddressSpace() == 1) &&
             "wrong address space.");
      llvmRetType = LLVM::LLVMPointerType::get(rewriter.getContext());
    }
  
    Type funcType = mlir::triton::gpu::getFunctionType(llvmRetType, llvmOpearands);

    LLVM::LLVMFuncOp funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
        rewriter, op, funcName, funcType, libname, libpath);
    auto callOp =
        LLVM::createLLVMCallOp(rewriter, loc, funcOp, llvmOpearands);

    if (op->getNumResults() == 0) {
      rewriter.eraseOp(op);
    } else {
      if (retType == llvmRetType){
          auto newResult = callOp.getResult();
          rewriter.replaceOp(op, newResult);
      }
      else{
        
        auto castOp = rewriter.create<LLVM::AddrSpaceCastOp>(loc, retType, callOp.getResult())->getResult(0);
        rewriter.replaceOp(op, castOp);
      }    
    }

    return success();
  }
};

template <typename... Args>
void registerGenericOpToROCSHMEMDevice(RewritePatternSet &patterns,
                                       LLVMTypeConverter &typeConverter,
                                       PatternBenefit benefit,
                                       StringRef calleeName, StringRef libname,
                                       StringRef libpath) {
  patterns.add<GenericOpToROCSHMEMDevice<Args>...>(
      typeConverter, benefit, calleeName, libname, libpath);
}

struct WaitOpConversion
    : public ConvertOpToLLVMPattern<triton::distributed::WaitOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::distributed::WaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return failure();
  }
};

struct ConsumeTokenOpConversion
    : public ConvertOpToLLVMPattern<triton::distributed::ConsumeTokenOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::distributed::ConsumeTokenOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

} // namespace

void mlir::triton::AMD::populateDistributedOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit, const TargetInfo &targetInfo,
    std::string ROCSHMEMLibname, std::string ROCSHMEMLibpath) {
  patterns.add<WaitOpConversion, ConsumeTokenOpConversion>(typeConverter,
                                                           benefit);

  // convert to rocshmem device func call
  registerGenericOpToROCSHMEMDevice<triton::distributed::GetRankOp>(
      patterns, typeConverter, benefit, "rocshmem_my_pe_kernel", ROCSHMEMLibname,
      ROCSHMEMLibpath);
  registerGenericOpToROCSHMEMDevice<triton::distributed::GetNumRanksOp>(
      patterns, typeConverter, benefit, "rocshmem_n_pes_kernel", ROCSHMEMLibname,
      ROCSHMEMLibpath);
  registerGenericOpToROCSHMEMDevice<triton::distributed::SymmAtOp>(
      patterns, typeConverter, benefit, "rocshmem_ptr_kernel", ROCSHMEMLibname,
      ROCSHMEMLibpath);
 patterns.add<ExternCallConversion>(typeConverter, benefit);

}

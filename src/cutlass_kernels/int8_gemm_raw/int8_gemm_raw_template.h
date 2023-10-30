/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_base.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/cutlass.h"
#include "cutlass/functional.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cuda_fp16.h"
#include "cutlass/gemm/device/gemm_splitk_parallel.h"
#include "cutlass_gemm.h"

#pragma GCC diagnostic pop
#include <string>
#include "utils/cuda_utils.h"

namespace fastertransformer {

template<typename T, 
         typename arch,
         typename ThreadblockShape, 
         typename WarpShape, 
         int Stages,
         int splitK>
void generic_int8_gemm_raw_kernelLauncher(const int8_t*     A,
                                          const int8_t*     B,
                                          const float       alpha,
                                          const float       beta,
                                          T*                C,
                                          int               m,
                                          int               n,
                                          int               k,
                                          char*             workspace,
                                          size_t            workspace_bytes,
                                          cudaStream_t      stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    using ElementInput = int8_t;

    using ElementOutput_ =
        typename cutlass::platform::conditional<cutlass::platform::is_same<T, half>::value, cutlass::half_t, T>::type;
#ifdef ENABLE_BF16
    using ElementOutput =
        typename cutlass::platform::conditional<cutlass::platform::is_same<ElementOutput_, __nv_bfloat16>::value,
                                                cutlass::bfloat16_t,
                                                ElementOutput_>::type;
#else
    using ElementOutput = ElementOutput_;
#endif

    using ElementAccumulator = int32_t;
    using ElementCompute     = float;

    using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    // TODO(mseznec): put below types in traits class
    using OperatorClass   = cutlass::arch::OpClassTensorOp;
    using DefaultGemmConf = typename cutlass::gemm::device::
        DefaultGemmConfiguration<OperatorClass, arch, ElementInput, ElementInput, ElementOutput, ElementCompute>;
    using InstructionShape = typename DefaultGemmConf::InstructionShape;
    // using GemmOp           = typename DefaultGemmConf::Operator;
    using EpilogueOp       = cutlass::epilogue::thread::LinearCombination<ElementOutput,
                                                                         128 / cutlass::sizeof_bits<ElementOutput>::value,
                                                                         ElementAccumulator,
                                                                         ElementCompute>;

    using Gemm = cutlass::gemm::device::Gemm<ElementInput,                           // Element type for A matrix operand
                                            cutlass::layout::RowMajor,              // Layout type for A matrix operand
                                            ElementInput,                           // Element type for B matrix operand
                                            cutlass::layout::ColumnMajor,           // Layout type for B matrix operand
                                            ElementOutput,                          // Element type for C and D matrix operands
                                            cutlass::layout::RowMajor,              // Layout type for C and D matrix operands
                                            ElementAccumulator,                     // Element type for internal accumulation
                                            OperatorClass,                          // Operator class tag
                                            arch,                                   // Tag indicating architecture to tune for
                                            ThreadblockShape                        // Threadblock-level tile size (concept: GemmShape)
                                            WarpShape,                              // Warp-level tile size (concept: GemmShape)
                                            InstructionShape,                       // Instruction-level tile size (concept: GemmShape)
                                            EpilogueOp,                             // Epilogue output operator
                                            ThreadblockSwizzle,                     // Threadblock-level swizzling operator
                                            stages,                                 // Number of stages used in the pipelined mainloop
                                            DefaultGemmConf::kAlignmentA,           // Access granularity of A matrix in units of elements
                                            DefaultGemmConf::kAlignmentB,           // Access granularity of B matrix in units of elements
                                            true>;                                  // If true, kernel supports split-K with serial reduction


    typename Gemm::Arguments args{{m, n, k},
                                  {reinterpret_cast<ElementInput*>(const_cast<ElementInput*>(A)), k},
                                  {reinterpret_cast<ElementInput*>(const_cast<ElementInput*>(B)), k},
                                  {nullptr, 0},
                                  {reinterpret_cast<ElementOutput*>(C), n},
                                  {alpha,beta}
                                  splitK}; // TODO 确认split K从哪里传入

    Gemm gemm;

    // 申请workspace
    size_t workspace_size_ = Gemm::get_workspace_size(arguments);

    if (workspace){
        if (workspace_size_ > workspace_bytes){
            std::string err_msg = "workspace_bytes size too small to apply split-k algo,needed: "+std::string(workspace_size_)+"alloc:"+std::string(workspace_bytes);
        throw std::runtime_error("[FT Error][int8gemm Runner] " + err_msg);
        }
    }else{
        // Allocate workspace memory
        cutlass::device_memory::allocation <uint8_t> workspace(workspace_size_);
    }
    
    // TODO(mseznec): handle that
    if (gemm.get_workspace_size(args) > workspace_bytes) {
        FT_LOG_WARNING(
            "Requested split-k but workspace size insufficient. Falling back to non-split-k implementation.");
        // If requested split-k factor will require more workspace bytes, revert to standard gemm.
        args.batch_count = 1;
    }

    auto can_implement = gemm.can_implement(args);
    if (can_implement != cutlass::Status::kSuccess) {
        std::string err_msg = "int8gemm cutlass kernel will fail for params. Error: "
                              + std::string(cutlassGetStatusString(can_implement));
        throw std::runtime_error("[FT Error][int8gemm Runner] " + err_msg);
    }

    auto init_status = gemm.initialize(args, workspace, stream);
    if (init_status != cutlass::Status::kSuccess) {
        std::string err_msg =
            "Failed to initialize cutlass int8 gemm. Error: " + std::string(cutlassGetStatusString(init_status));
        throw std::runtime_error("[FT Error][int8gemm Runner] " + err_msg);
    }

    auto run_status = gemm.run(stream);
    if (run_status != cutlass::Status::kSuccess) {
        std::string err_msg =
            "Failed to run cutlass int8 gemm. Error: " + std::string(cutlassGetStatusString(run_status));
        throw std::runtime_error("[FT Error][int8gemm Runner] " + err_msg);
    }
}

template<typename T, 
         typename arch>
void dispatch_gemm_to_cutlass(const int8_t*     A,
                              const int8_t*     B,
                              const float       alpha,
                              const float       beta,
                              T*                C,
                              int               m,
                              int               n,
                              int               k,
                              char*             workspace,
                              size_t            workspace_bytes,
                              std::string       tile_config,
                              int               stages,
                              int               splitK,
                              cudaStream_t      stream)
{

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    // Note that SIMT configs are omitted here since they are not supported for int8.
    // We also only instantiate configs here where threadblockShapeM == warpShapeM since those usually perform the best
    // for mixed type gemms.
    // dispatch_gemm_config<T, arch, cutlass::gemm::GemmShape<32, 128, 64>, cutlass::gemm::GemmShape<32, 32, 64>>(
        // A, B, quant_mode, alpha_col, alpha_row, C, m, n, k, gemm_config, workspace, workspace_bytes, stream, occupancy);
    if (tile_config == "CtaShape128x256x128_WarpShape64x64x128"):{
            generic_int8_gemm_raw_kernelLauncher<T, 
                                                arch, 
                                                cutlass::gemm::GemmShape<128, 256, 128> 
                                                cutlass::gemm::GemmShape<64, 64, 128>,
                                                stages, splitK>( 
                A, B, alpha, beta, C, m, n, k, workspace, workspace_bytes, stream);
    } else if (tile_config == 'CtaShape256x128x128_WarpShape64x64x128'){
            generic_int8_gemm_raw_kernelLauncher<T, 
                                 arch, 
                                 cutlass::gemm::GemmShape<256, 128, 128>, 
                                 cutlass::gemm::GemmShape<64, 64, 128>,
                                 stages, splitK>( 
                A, B, alpha, beta, C, m, n, k, workspace, workspace_bytes, stream);
    }else if (tile_config == 'CtaShape128x128x128_WarpShape64x64x128'){
            generic_int8_gemm_raw_kernelLauncher<T, 
                                 arch, 
                                 cutlass::gemm::GemmShape<128, 128, 128>,
                                 cutlass::gemm::GemmShape<64, 64, 128>, 
                                 stages, splitK>( 
                A, B, alpha, beta, C, m, n, k, workspace, workspace_bytes, stream);
    }else if (tile_config == 'CtaShape256x64x128_WarpShape64x64x128'){
            generic_int8_gemm_raw_kernelLauncher<T, 
                                 arch, 
                                 cutlass::gemm::GemmShape<256, 64, 128>,
                                 cutlass::gemm::GemmShape<64, 64, 128>, 
                                 stages, splitK>( 
                A, B, alpha, beta, C, m, n, k, workspace, workspace_bytes, stream);
    }else if (tile_config== 'CtaShape64x256x128_WarpShape64x64x128'){
            generic_int8_gemm_raw_kernelLauncher<T, 
                                 arch, 
                                 cutlass::gemm::GemmShape<64, 256, 128>,
                                 cutlass::gemm::GemmShape<64, 64, 128>, 
                                 stages, splitK>( 
                A, B, alpha, beta, C, m, n, k, workspace, workspace_bytes, stream);
    }else if (tile_config == 'CtaShape64x128x128_WarpShape32x64x128'){
            generic_int8_gemm_raw_kernelLauncher<T, 
                                 arch, 
                                 cutlass::gemm::GemmShape<64, 128, 128>,
                                 cutlass::gemm::GemmShape<32, 64, 128>, 
                                 stages, splitK>( 
                A, B, alpha, beta, C, m, n, k, workspace, workspace_bytes, stream);
    }else if (tile_config =='CtaShape128x64x128_WarpShape64x32x128' ){
            generic_int8_gemm_raw_kernelLauncher<T, 
                                 arch, 
                                 cutlass::gemm::GemmShape<128, 64, 128>,
                                 cutlass::gemm::GemmShape<64, 32, 128>, 
                                 stages, splitK>( 
                A, B, alpha, beta, C, m, n, k, workspace, workspace_bytes, stream);
    }else if (tile_config == 'CtaShape64x64x128_WarpShape32x32x128'){
            generic_int8_gemm_raw_kernelLauncher<T, 
                                 arch, 
                                 cutlass::gemm::GemmShape<64, 64, 128>,
                                 cutlass::gemm::GemmShape<32, 32, 128>,
                                 stages, splitK>( 
                A, B, alpha, beta, C, m, n, k, workspace, workspace_bytes, stream);
    }else if (tile_config == 'CtaShape128x256x64_WarpShape64x64x64'){
            generic_int8_gemm_raw_kernelLauncher<T, 
                                 arch, 
                                 cutlass::gemm::GemmShape<128, 256, 64>,
                                 cutlass::gemm::GemmShape<64, 64, 64>,
                                 stages, splitK>( 
                A, B, alpha, beta, C, m, n, k, workspace, workspace_bytes, stream);
    }else if (tile_config == 'CtaShape256x128x64_WarpShape64x64x64'){
            generic_int8_gemm_raw_kernelLauncher<T, 
                                 arch, 
                                 cutlass::gemm::GemmShape<256, 128, 64>,
                                 cutlass::gemm::GemmShape<64, 64, 64>, 
                                 stages, splitK>( 
                A, B, alpha, beta, C, m, n, k, workspace, workspace_bytes, stream);
    }else if (tile_config == "CtaShape128x128x64_WarpShape64x64x64"){
            generic_int8_gemm_raw_kernelLauncher<T, 
                                 arch, 
                                 cutlass::gemm::GemmShape<128, 128, 64>,
                                 cutlass::gemm::GemmShape<64, 64, 64>,
                                 stages, splitK>( 
                A, B, alpha, beta, C, m, n, k, workspace, workspace_bytes, stream);
    }else if (tile_config == 'CtaShape256x64x64_WarpShape64x64x64'){
            generic_int8_gemm_raw_kernelLauncher<T, 
                                 arch, 
                                 cutlass::gemm::GemmShape<256, 64, 64>,
                                 cutlass::gemm::GemmShape<64, 64, 64>, 
                                 stages, splitK>( 
                A, B, alpha, beta, C, m, n, k, workspace, workspace_bytes, stream);
    }else if (tile_config == 'CtaShape64x256x64_WarpShape64x64x64'){
            generic_int8_gemm_raw_kernelLauncher<T, 
                                 arch, 
                                 cutlass::gemm::GemmShape<64, 256, 64>,
                                 cutlass::gemm::GemmShape<64, 64, 64>,
                                 stages, splitK>( 
                A, B, alpha, beta, C, m, n, k, workspace, workspace_bytes, stream);
    }else if (tile_config == 'CtaShape64x128x64_WarpShape32x64x64'){
            generic_int8_gemm_raw_kernelLauncher<T, 
                                 arch, 
                                 cutlass::gemm::GemmShape<64, 128, 64>,
                                 cutlass::gemm::GemmShape<32, 64, 64>,
                                 stages, splitK>( 
                A, B, alpha, beta, C, m, n, k, workspace, workspace_bytes, stream);
    }else if (tile_config == 'CtaShape128x64x64_WarpShape64x32x64'){
            generic_int8_gemm_raw_kernelLauncher<T, 
                                 arch, 
                                 cutlass::gemm::GemmShape<128, 64, 64>,
                                 cutlass::gemm::GemmShape<64, 32, 64>,
                                 stages, splitK>( 
                A, B, alpha, beta, C, m, n, k, workspace, workspace_bytes, stream);
    }else if (tile_config == 'CtaShape64x64x64_WarpShape32x32x64'){
            generic_int8_gemm_raw_kernelLauncher<T, 
                                 arch, 
                                 cutlass::gemm::GemmShape<64, 64, 64>,
                                 cutlass::gemm::GemmShape<32, 32, 64>,
                                 stages, splitK>( 
                A, B, alpha, beta, C, m, n, k, workspace, workspace_bytes, stream);
    }else{
            generic_int8_gemm_raw_kernelLauncher<T, 
                                 arch, 
                                 cutlass::gemm::GemmShape<128, 256, 128> 
                                 cutlass::gemm::GemmShape<64, 64, 128>,
                                 stages, splitK>( 
                A, B, alpha, beta, C, m, n, k, workspace, workspace_bytes, stream);
    }

}

template<typename T>
void cutlass_int8_gemm_per_tensor(const int8_t*     A,
                                  const int8_t*     B,
                                  const float       alpha,
                                  const float       beta,
                                  T*                C,
                                  int               m,
                                  int               n,
                                  int               k,
                                  std::string       tile_config,
                                  int               stages,
                                  int               splitK,
                                  char*             workspace_ptr,
                                  const size_t      workspace_bytes)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (sm_ >= 80 && sm_ < 90) {
        dispatch_gemm_to_cutlass<T, cutlass::arch::Sm80>(A,
                                                         B,
                                                         alpha,
                                                         beta,
                                                         C,
                                                         m,
                                                         n,
                                                         k,
                                                         workspace_ptr,
                                                         workspace_bytes,
                                                         tile_config,
                                                         stages,splitK,
                                                         stream);
    }
    else {
        throw std::runtime_error(
            "[FT Error][CutlassInt8GemmRunner][GEMM Dispatch] Arch unsupported for CUTLASS int8 GEMM");
    }
}




}  // namespace fastertransformer
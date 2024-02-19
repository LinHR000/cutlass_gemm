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
#include "cutlass/epilogue/thread/linear_combination_gelu.h"
#pragma GCC diagnostic pop
#include <string>
#include "cutlass_kernels/int8_gemm_raw/int8_gemm_raw.h"
#include "utils/cuda_utils.h"

namespace fastertransformer {

template<typename T, 
         typename arch,
         typename ThreadblockShape, 
         typename WarpShape,
         int stages>
void generic_int8_gemm_gelu_kernelLauncher(const int8_t*     A,
                                          const int8_t*     B,
                                          const T*          bias,
                                          const float       alpha,
                                          const float       beta,
                                          T*                C,
                                          int               m,
                                          int               n,
                                          int               k,
                                          char*             workspace,
                                          size_t            workspace_bytes,
                                          const int         splitK,
                                          cudaStream_t      stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    using ElementInput = int8_t;

    using ElementOutput_ =
        // typename cutlass::platform::conditional<cutlass::platform::is_same<T, half>::value, cutlass::half_t, T>::type;
        typename cutlass::platform::conditional<cutlass::platform::is_same<T, half>::value, __half, T>::type;
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
    // using EpilogueOp       = cutlass::epilogue::thread::LinearCombination<ElementOutput,
    //                                                                      128 / cutlass::sizeof_bits<ElementOutput>::value,
    //                                                                      ElementAccumulator,
    //                                                                      ElementCompute>;
    using EpilogueOp = cutlass::epilogue::thread::LinearCombinationGelu<
    ElementOutput,                                        // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,     // <- this is the number of elements per
                                                          // vectorized memory access. For half
                                                          // precision, it's 8 elements. This becomes
                                                          // the vector width of math instructions in
                                                          // epilogue too
    ElementAccumulator,                                   // <- data type of accumulator
    ElementComputeEpilogue,                               // <- data type for alpha in linear combination function
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>; // <- alpha x C + bias
    // using EpilogueOp       = typename DefaultGemmConf::EpilogueOutputOp;                                                                  

    using Gemm = cutlass::gemm::device::Gemm<ElementInput,                           // Element type for A matrix operand
                                            cutlass::layout::RowMajor,              // Layout type for A matrix operand
                                            ElementInput,                           // Element type for B matrix operand
                                            cutlass::layout::ColumnMajor,           // Layout type for B matrix operand
                                            ElementOutput,                          // Element type for C and D matrix operands
                                            cutlass::layout::RowMajor,              // Layout type for C and D matrix operands
                                            ElementAccumulator,                     // Element type for internal accumulation
                                            OperatorClass,                          // Operator class tag
                                            arch,                                   // Tag indicating architecture to tune for
                                            ThreadblockShape,                       // Threadblock-level tile size (concept: GemmShape)
                                            WarpShape,                              // Warp-level tile size (concept: GemmShape)
                                            InstructionShape,                       // Instruction-level tile size (concept: GemmShape)
                                            EpilogueOp,                             // Epilogue output operator
                                            ThreadblockSwizzle,                     // Threadblock-level swizzling operator
                                            stages,                                 // Number of stages used in the pipelined mainloop
                                            DefaultGemmConf::kAlignmentA,           // Access granularity of A matrix in units of elements
                                            DefaultGemmConf::kAlignmentB,           // Access granularity of B matrix in units of elements
                                            true>;                                  // If true, kernel supports split-K with serial reduction
    // using bias_ptr = bias ? bias : nullptr;
    cutlass::gemm::GemmCoord problem_size(m, n, k);
    typename Gemm::Arguments args{problem_size,
                                    {A, k},    // <- reference to matrix A on device
                                    {B, k},   // <- reference to matrix B on device
                                    {bias, 0},      // <- reference to matrix C on device
                                    {C, n},   
                                  {alpha,beta},
                                  splitK}; // TODO 确认split K从哪里传入

    Gemm gemm_op;

    // 申请workspace
    size_t workspace_size_ = Gemm::get_workspace_size(args);

    if (workspace){
        if (workspace_size_ > workspace_bytes){
            std::string err_msg = "workspace_bytes size too small to apply split-k algo";
        throw std::runtime_error("[FT Error][int8gemm Runner] " + err_msg);
        }
    }else{
        // Allocate workspace memory
        cutlass::device_memory::allocation <uint8_t> workspace(workspace_size_);
    }
    
        // Check the problem size is supported or not
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("cutlass cannot implement, status: " +
                                 std::to_string((int) status));
    }

    // Initialize CUTLASS kernel with arguments and workspace pointer
    auto init_status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("cutlass cannot initialize, status: " +
                                 std::to_string((int) status));
    }

    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("cutlass cannot run, status: " +
                                 std::to_string((int) status));
    }
}






template<typename T, typename arch, typename ThreadblockShape, typename WarpShape, int Stages, typename Enable = void>
struct dispatch_stages_gelu {
    static void dispatch(const int8_t*     A,
                        const int8_t*     B,
                        const T*          bias,
                        const float       alpha,
                        const float       beta,
                        T*                C,
                        int               m,
                        int               n,
                        int               k,
                        char*             workspace,
                        size_t            workspace_bytes,
                        const int         splitK,
                        cudaStream_t      stream)
    {

        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        std::string err_msg = "Cutlass int8 gemm. Not instantiates for arch "
                              + std::to_string(arch::kMinComputeCapability) + " with stages set to "
                              + std::to_string(Stages);
        throw std::runtime_error("[FT Error][dispatch_stages::dispatch] " + err_msg);
    }
};

template<typename T, typename arch, typename ThreadblockShape, typename WarpShape>
struct dispatch_stages_gelu<T, arch, ThreadblockShape, WarpShape, 2> {
    static void dispatch(const int8_t*     A,
                        const int8_t*     B,
                        const T*          bias,
                        const float       alpha,
                        const float       beta,
                        T*                C,
                        int               m,
                        int               n,
                        int               k,
                        char*             workspace,
                        size_t            workspace_bytes,
                        const int         splitK,
                        cudaStream_t      stream)
    {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        generic_int8_gemm_gelu_kernelLauncher<T, arch, ThreadblockShape, WarpShape, 2>(A, 
                                                                                  B, 
                                                                                  bias,
                                                                                  alpha,
                                                                                  beta, 
                                                                                  C, 
                                                                                  m, 
                                                                                  n, 
                                                                                  k, 
                                                                                  workspace, 
                                                                                  workspace_bytes, 
                                                                                  splitK, 
                                                                                  stream);
    }
};

template<typename T, typename ThreadblockShape, typename WarpShape, int Stages>
struct dispatch_stages_gelu<T,
                       cutlass::arch::Sm80,
                       ThreadblockShape,
                       WarpShape,
                       Stages,
                       typename std::enable_if<(Stages > 2)>::type> {
    static void dispatch(const int8_t*     A,
                        const int8_t*     B,
                        const T*          bias,
                        const float       alpha,
                        const float       beta,
                        T*                C,
                        int               m,
                        int               n,
                        int               k,
                        char*             workspace,
                        size_t            workspace_bytes,
                        const int         splitK,
                        cudaStream_t      stream)
    {

        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        generic_int8_gemm_gelu_kernelLauncher<T, cutlass::arch::Sm80, ThreadblockShape, WarpShape, Stages>(A, 
                                                                                                    B, 
                                                                                                    bias,
                                                                                                    alpha,
                                                                                                    beta, 
                                                                                                    C, 
                                                                                                    m, 
                                                                                                    n, 
                                                                                                    k, 
                                                                                                    workspace, 
                                                                                                    workspace_bytes, 
                                                                                                    splitK, 
                                                                                                    stream);
    }
};

template<typename T, typename arch, typename ThreadblockShape, typename WarpShape>
void dispatch_gemm_config_gelu(const int8_t*     A,
                        const int8_t*     B,
                        const T*          bias,
                        const float       alpha,
                        const float       beta,
                        T*                C,
                        int               m,
                        int               n,
                        int               k,
                        char*             workspace,
                        size_t            workspace_bytes,
                        const int         stages,
                        const int         splitK,
                        cudaStream_t      stream)
{

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    switch (stages) {
        case 2:
            using DispatcherStages2 = dispatch_stages_gelu<T, arch, ThreadblockShape, WarpShape, 2>;
            DispatcherStages2::dispatch(A, 
                                        B, 
                                        bias,
                                        alpha,
                                        beta, 
                                        C, 
                                        m, 
                                        n, 
                                        k, 
                                        workspace, 
                                        workspace_bytes, 
                                        splitK, 
                                        stream);
            break;
        case 3:
            using DispatcherStages3 = dispatch_stages_gelu<T, arch, ThreadblockShape, WarpShape, 3>;
            DispatcherStages3::dispatch(A, 
                                        B, 
                                        bias,
                                        alpha,
                                        beta, 
                                        C, 
                                        m, 
                                        n, 
                                        k, 
                                        workspace, 
                                        workspace_bytes, 
                                        splitK, 
                                        stream);
            break;
        case 4:
            using DispatcherStages4 = dispatch_stages_gelu<T, arch, ThreadblockShape, WarpShape, 4>;
            DispatcherStages4::dispatch(A, 
                                        B, 
                                        bias,
                                        alpha,
                                        beta, 
                                        C, 
                                        m, 
                                        n, 
                                        k, 
                                        workspace, 
                                        workspace_bytes, 
                                        splitK, 
                                        stream);
            break;
        default:
            std::string err_msg = "dispatch_gemm_config does not support stages " + std::to_string(stages);
            throw std::runtime_error("[FT Error][dispatch_gemm_config] " + err_msg);
            break;
    }
}

void dispatch_gemm_to_cutlass_fp16_gelu(const int8_t*     A,
                              const int8_t*     B,
                              const __half*          bias,
                              const float       alpha,
                              const float       beta,
                              __half*           C,
                              int               m,
                              int               n,
                              int               k,
                              char*             workspace,
                              size_t            workspace_bytes,
                              std::string       tile_config,
                              const int         stages,
                              const int         splitK,
                              cudaStream_t      stream)
{

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (tile_config == "CtaShape128x256x128_WarpShape64x64x128"){
            dispatch_gemm_config_gelu<__half, 
                                cutlass::arch::Sm80, 
                                cutlass::gemm::GemmShape<128, 256, 128>, 
                                cutlass::gemm::GemmShape<64, 64, 128>>( 
                A, B, bias,alpha, beta, C, m, n, k, workspace, workspace_bytes, stages, splitK, stream);
    }else if (tile_config == "CtaShape128x128x128_WarpShape64x64x128"){
            dispatch_gemm_config_gelu<__half, 
                                cutlass::arch::Sm80, 
                                cutlass::gemm::GemmShape<128, 128, 128>,
                                cutlass::gemm::GemmShape<64, 64, 128>>( 
                A, B, bias,alpha, beta, C, m, n, k, workspace, workspace_bytes, stages, splitK, stream);
    }else if (tile_config== "CtaShape64x256x128_WarpShape64x64x128"){
            dispatch_gemm_config_gelu<__half, 
                                cutlass::arch::Sm80, 
                                cutlass::gemm::GemmShape<64, 256, 128>,
                                cutlass::gemm::GemmShape<64, 64, 128>>( 
                A, B, bias,alpha, beta, C, m, n, k, workspace, workspace_bytes, stages, splitK, stream);
    }else if (tile_config == "CtaShape64x128x128_WarpShape32x64x128"){
            dispatch_gemm_config_gelu<__half, 
                                cutlass::arch::Sm80, 
                                cutlass::gemm::GemmShape<64, 128, 128>,
                                cutlass::gemm::GemmShape<32, 64, 128>>( 
                A, B, bias,alpha, beta, C, m, n, k, workspace, workspace_bytes, stages, splitK, stream);
    }else if (tile_config == "CtaShape256x128x64_WarpShape64x64x64"){
            dispatch_gemm_config_gelu<__half, 
                                cutlass::arch::Sm80, 
                                cutlass::gemm::GemmShape<256, 128, 64>,
                                cutlass::gemm::GemmShape<64, 64, 64>>( 
                A, B, bias,alpha, beta, C, m, n, k, workspace, workspace_bytes, stages, splitK, stream);
    }else if (tile_config == "CtaShape128x128x64_WarpShape64x64x64"){
            dispatch_gemm_config_gelu<__half, 
                                cutlass::arch::Sm80, 
                                cutlass::gemm::GemmShape<128, 128, 64>,
                                cutlass::gemm::GemmShape<64, 64, 64>>( 
                A, B, bias,alpha, beta, C, m, n, k, workspace, workspace_bytes, stages, splitK, stream);

    }else if (tile_config == "CtaShape64x256x64_WarpShape64x64x64"){
            dispatch_gemm_config_gelu<__half, 
                                cutlass::arch::Sm80, 
                                cutlass::gemm::GemmShape<64, 256, 64>,
                                cutlass::gemm::GemmShape<64, 64, 64>>( 
                A, B, bias,alpha, beta, C, m, n, k, workspace, workspace_bytes, stages, splitK, stream);
    }else if (tile_config == "CtaShape64x128x64_WarpShape32x64x64"){
            dispatch_gemm_config_gelu<__half, 
                                cutlass::arch::Sm80, 
                                cutlass::gemm::GemmShape<64, 128, 64>,
                                cutlass::gemm::GemmShape<32, 64, 64>>( 
                A, B, bias,alpha, beta, C, m, n, k, workspace, workspace_bytes, stages, splitK, stream);
    }else{
            dispatch_gemm_config_gelu<__half, 
                                cutlass::arch::Sm80, 
                                cutlass::gemm::GemmShape<128, 256, 128>, 
                                cutlass::gemm::GemmShape<64, 64, 128>>( 
                A, B, bias,alpha, beta, C, m, n, k, workspace, workspace_bytes, stages, splitK, stream);
    }


}


void cutlass_int8_fp16_gemm_per_tensor_gelu(const int8_t*     A,
                                    const int8_t*     B,
                                    const __half*           bias,
                                    const float       alpha,
                                    const float       beta,
                                    __half*           C,
                                    int               m,
                                    int               n,
                                    int               k,
                                    std::string       tile_config,
                                    const int         stages,
                                    const int         splitK,
                                    char*             workspace_ptr,
                                    const size_t      workspace_bytes,
                                    int               sm_,
                                    cudaStream_t      stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (sm_ >= 80 && sm_ < 90) {
        dispatch_gemm_to_cutlass_fp16_gelu(A,
                                    B,
                                    bias,
                                    alpha,
                                    beta,
                                    C,
                                    m,
                                    n,
                                    k,
                                    workspace_ptr,
                                    workspace_bytes,
                                    tile_config,
                                    stages,
                                    splitK,
                                    stream);
    }
    else {
        throw std::runtime_error(
            "[FT Error][CutlassInt8GemmRunner][GEMM Dispatch] Arch unsupported for CUTLASS int8 GEMM");
    }
}



}  // namespace fastertransformer
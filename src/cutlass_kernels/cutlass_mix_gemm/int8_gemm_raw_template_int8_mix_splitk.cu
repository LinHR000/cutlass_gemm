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

#pragma GCC diagnostic pop
#include <string>
#include "cutlass_kernels/cutlass_mix_gemm/mix_gemm.h"
#include "utils/cuda_utils.h"
#include "cutlass/cutlass.h"

#include "cutlass/gemm/device/gemm_universal.h"

namespace fastertransformer {

template<typename T, 
         typename arch,
         typename ThreadblockShape, 
         typename WarpShape,
         int stages>
void generic_int8_gemm_mix_splitk_kernelLauncher(const T*     A,
                                          const int8_t*     B,
                                          const float       alpha,
                                          const float       beta,
                                          T*                C,
                                          int               m,
                                          int               n,
                                          int               k,
                                          int               stridea,
                                          int               strideb,
                                          int               stridec,
                                          char*             workspace,
                                          size_t            workspace_bytes,
                                          const int         splitK,
                                          cudaStream_t      stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    using ElementInputB = int8_t;

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
    using ElementInputA = ElementOutput_;
#endif

    using ElementCompute     = float;
    using ElementAccumulator = float;

    using Gemm = cutlass::gemm::device::GemmUniversal<
    ElementOutput, 
    cutlass::layout::RowMajor, 
    ElementInputB,
    cutlass::layout::ColumnMajor, 
    ElementOutput, 
    cutlass::layout::RowMajor,
    ElementAccumulator, 
    cutlass::arch::OpClassTensorOp, 
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 
    stages,  // Stages
    8,  // AlignmentA
    16, // AlignmentB
    cutlass::arch::OpMultiplyAddMixedInputUpcast,
    cutlass::ComplexTransform::kNone,
    cutlass::ComplexTransform::kNone
  >;

    cutlass::gemm::GemmCoord problem_size(m, n, k);

    typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel,
      problem_size,
      splitK,
      {alpha, beta},
      {reinterpret_cast<ElementOutput*>(A),k},
      {B,k},
      {reinterpret_cast<ElementOutput*>(C),n},
      {reinterpret_cast<ElementOutput*>(C),n},
      problem_size.m() * problem_size.k(),
      problem_size.n() * problem_size.k(),
      problem_size.m() * problem_size.n(),
      problem_size.m() * problem_size.n(),
      stridea,
      strideb,
      stridec,
      stridec
    };


    Gemm gemm_op;

    // 申请workspace
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    
        // Check the problem size is supported or not
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("cutlass cannot implement, status: " +
                                 std::to_string((int) status));
    }

    // Initialize CUTLASS kernel with arguments and workspace pointer
    auto init_status = gemm_op.initialize(arguments, workspace.get());
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
struct dispatch_stages_splitk {
    static void dispatch(const T*     A,
                        const int8_t*     B,
                        const float       alpha,
                        const float       beta,
                        T*                C,
                        int               m,
                        int               n,
                        int               k,
                        int               stridea,
                        int               strideb,
                        int               stridec,
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
struct dispatch_stages_splitk<T, arch, ThreadblockShape, WarpShape, 2> {
    static void dispatch(const T*     A,
                        const int8_t*     B,
                        const float       alpha,
                        const float       beta,
                        T*                C,
                        int               m,
                        int               n,
                        int               k,
                        int               stridea,
                        int               strideb,
                        int               stridec,
                        char*             workspace,
                        size_t            workspace_bytes,
                        const int         splitK,
                        cudaStream_t      stream)
    {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        generic_int8_gemm_mix_splitk_kernelLauncher<T, arch, ThreadblockShape, WarpShape, 2>(A, 
                                                                                  B, 
                                                                                  alpha,
                                                                                  beta, 
                                                                                  C, 
                                                                                  m, 
                                                                                  n, 
                                                                                  k, 
                                                                                  stridea,
                                                                                  strideb,
                                                                                  stridec,
                                                                                  workspace, 
                                                                                  workspace_bytes, 
                                                                                  splitK, 
                                                                                  stream);
    }
};

template<typename T, typename ThreadblockShape, typename WarpShape, int Stages>
struct dispatch_stages_splitk<T,
                       cutlass::arch::Sm80,
                       ThreadblockShape,
                       WarpShape,
                       Stages,
                       typename std::enable_if<(Stages > 2)>::type> {
    static void dispatch(const T*     A,
                        const int8_t*     B,
                        const float       alpha,
                        const float       beta,
                        T*                C,
                        int               m,
                        int               n,
                        int               k,
                        int               stridea,
                        int               strideb,
                        int               stridec,
                        char*             workspace,
                        size_t            workspace_bytes,
                        const int         splitK,
                        cudaStream_t      stream)
    {

        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        generic_int8_gemm_mix_splitk_kernelLauncher<T, cutlass::arch::Sm80, ThreadblockShape, WarpShape, Stages>(A, 
                                                                                                    B, 
                                                                                                    alpha,
                                                                                                    beta, 
                                                                                                    C, 
                                                                                                    m, 
                                                                                                    n, 
                                                                                                    k, 
                                                                                                    stridea,
                                                                                                    strideb,
                                                                                                    stridec,
                                                                                                    workspace, 
                                                                                                    workspace_bytes, 
                                                                                                    splitK, 
                                                                                                    stream);
    }
};

template<typename T, typename arch, typename ThreadblockShape, typename WarpShape>
void dispatch_gemm_config_splitk(const T*     A,
                        const int8_t*     B,
                        const float       alpha,
                        const float       beta,
                        T*                C,
                        int               m,
                        int               n,
                        int               k,
                        int               stridea,
                        int               strideb,
                        int               stridec,
                        char*             workspace,
                        size_t            workspace_bytes,
                        const int         stages,
                        const int         splitK,
                        cudaStream_t      stream)
{

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    switch (stages) {
        case 2:
            using DispatcherStages2 = dispatch_stages_splitk<T, arch, ThreadblockShape, WarpShape, 2>;
            DispatcherStages2::dispatch(A, 
                                        B, 
                                        alpha,
                                        beta, 
                                        C, 
                                        m, 
                                        n, 
                                        k, 
                                        stridea,
                                        strideb,
                                        stridec,
                                        workspace, 
                                        workspace_bytes, 
                                        splitK, 
                                        stream);
            break;
        case 3:
            using DispatcherStages3 = dispatch_stages_splitk<T, arch, ThreadblockShape, WarpShape, 3>;
            DispatcherStages3::dispatch(A, 
                                        B, 
                                        alpha,
                                        beta, 
                                        C, 
                                        m, 
                                        n, 
                                        k, 
                                        stridea,
                                        strideb,
                                        stridec,
                                        workspace, 
                                        workspace_bytes, 
                                        splitK, 
                                        stream);
            break;
        case 4:
            using DispatcherStages4 = dispatch_stages_splitk<T, arch, ThreadblockShape, WarpShape, 4>;
            DispatcherStages4::dispatch(A, 
                                        B, 
                                        alpha,
                                        beta, 
                                        C, 
                                        m, 
                                        n, 
                                        k, 
                                        stridea,
                                        strideb,
                                        stridec,
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
};

template <typename T> 
void dispatch_gemm_to_cutlass_splitk(const T*     A,
                                const int8_t*     B,
                                const float       alpha,
                                const float       beta,
                                T*           C,
                                int               m,
                                int               n,
                                int               k,
                                int               stridea,
                                int               strideb,
                                int               stridec,
                                char*             workspace,
                                size_t            workspace_bytes,
                                std::string       tile_config,
                                const int         stages,
                                const int         splitK,
                                cudaStream_t      stream)
    {

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (tile_config == "CtaShape128x128x64_WarpShape64x64x64"){
            dispatch_gemm_config_splitk<T, 
                                cutlass::arch::Sm80, 
                                cutlass::gemm::GemmShape<128, 128, 64>,
                                cutlass::gemm::GemmShape<64, 64, 64>>( 
                A, B, alpha, beta, C, m, n, k,stridea,strideb,stridec,workspace, workspace_bytes, stages, splitK, stream);
    }
    else if (tile_config == "CtaShape128x128x32_WarpShape64x64x32"){
            dispatch_gemm_config_splitk<T, 
                                cutlass::arch::Sm80, 
                                cutlass::gemm::GemmShape<128, 128, 32>, 
                                cutlass::gemm::GemmShape<64, 64, 32>>( 
                A, B, alpha, beta, C, m, n, k,stridea,strideb,stridec,workspace, workspace_bytes, stages, splitK, stream);
    }
    else if (tile_config == "CtaShape64x128x32_WarpShape32x64x32"){
            dispatch_gemm_config_splitk<T, 
                                cutlass::arch::Sm80, 
                                cutlass::gemm::GemmShape<64, 128, 32>, 
                                cutlass::gemm::GemmShape<32, 64, 32>>( 
                A, B, alpha, beta, C, m, n, k,stridea,strideb,stridec,workspace, workspace_bytes, stages, splitK, stream);
    }else if (tile_config == "CtaShape64x64x32_WarpShape32x32x32"){
            dispatch_gemm_config_splitk<T, 
                                cutlass::arch::Sm80, 
                                cutlass::gemm::GemmShape<64, 64, 32>,
                                cutlass::gemm::GemmShape<32, 32, 32>>( 
                A, B, alpha, beta, C, m, n, k,stridea,strideb,stridec,workspace, workspace_bytes, stages, splitK, stream);
    }else if (tile_config == "CtaShape128x64x32_WarpShape64x32x32"){
            dispatch_gemm_config_splitk<T, 
                                cutlass::arch::Sm80, 
                                cutlass::gemm::GemmShape<128, 64, 32>,
                                cutlass::gemm::GemmShape<64, 32, 32>>( 
                A, B, alpha, beta, C, m, n, k,stridea,strideb,stridec,workspace, workspace_bytes, stages, splitK, stream);
    }else if (tile_config== "CtaShape128x64x32_WarpShape64x64x32"){
            dispatch_gemm_config_splitk<T, 
                                cutlass::arch::Sm80, 
                                cutlass::gemm::GemmShape<128, 64, 32>,
                                cutlass::gemm::GemmShape<64, 64, 32>>( 
                A, B, alpha, beta, C, m, n, k,stridea,strideb,stridec,workspace, workspace_bytes, stages, splitK, stream);
    }else if (tile_config == "CtaShape128x32x32_WarpShape64x32x32"){
            dispatch_gemm_config_splitk<T, 
                                cutlass::arch::Sm80, 
                                cutlass::gemm::GemmShape<128, 32, 32>,
                                cutlass::gemm::GemmShape<64, 32, 32>>( 
                A, B, alpha, beta, C, m, n, k,stridea,strideb,stridec,workspace, workspace_bytes, stages, splitK, stream); 

    }else{
           dispatch_gemm_config_splitk<T, 
                                cutlass::arch::Sm80, 
                                cutlass::gemm::GemmShape<128, 128, 64>,
                                cutlass::gemm::GemmShape<64, 64, 64>>( 
                A, B, alpha, beta, C, m, n, k,stridea,strideb,stridec,workspace, workspace_bytes, stages, splitK, stream);
    }
   

}


void cutlass_fp16_int8_gemm_per_tensor_splitk(const __half*     A,
                                    const int8_t*     B,
                                    const float       alpha,
                                    const float       beta,
                                    __half*           C,
                                    int               m,
                                    int               n,
                                    int               k,
                                    int               stridea,
                                    int               strideb,
                                    int               stridec,
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
        dispatch_gemm_to_cutlass_splitk<__half>(A,
                                                B,
                                                alpha,
                                                beta,
                                                C,
                                                m,
                                                n,
                                                k,
                                                stridea,
                                                strideb,
                                                stridec,
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

void cutlass_bf16_int8_gemm_per_tensor_splitk(const __nv_bfloat16*     A,
                                            const int8_t*     B,
                                            const float       alpha,
                                            const float       beta,
                                            __nv_bfloat16*           C,
                                            int               m,
                                            int               n,
                                            int               k,
                                            int               stridea,
                                            int               strideb,
                                            int               stridec,
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
        dispatch_gemm_to_cutlass_splitk<__nv_bfloat16>(A,
                                                    B,
                                                    alpha,
                                                    beta,
                                                    C,
                                                    m,
                                                    n,
                                                    k,
                                                    stridea,
                                                    strideb,
                                                    stridec,
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
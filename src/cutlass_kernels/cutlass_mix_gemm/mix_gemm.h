
#pragma once

#include "utils/activation_types.h"
#include "utils/allocator.h"
#include <cuda_runtime_api.h>
#include <string>
# include <map>
namespace fastertransformer {
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
                                            cudaStream_t      stream);

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
                                    cudaStream_t      stream);

void cutlass_bf16_int8_gemm_per_tensor(__nv_bfloat16*     A,
                                    int8_t*     B,
                                    __nv_bfloat16*    bias,
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
                                    cudaStream_t      stream);

void cutlass_fp16_int8_gemm_per_tensor(__half*     A,
                                    int8_t*     B,
                                    __half*    bias,
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
                                    cudaStream_t      stream);                                                                                                                
}  // namespace fastertransformer

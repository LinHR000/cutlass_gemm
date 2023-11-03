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

#pragma once

#include "utils/activation_types.h"
#include "utils/allocator.h"
#include <cuda_runtime_api.h>
#include <string>
# include <map>
namespace fastertransformer {


void cutlass_int8_int8_gemm_per_tensor(const int8_t*     A,
                                    const int8_t*     B,
                                    const float       alpha,
                                    const float       beta,
                                    int8_t*           C,
                                    int               m,
                                    int               n,
                                    int               k,
                                    std::string       tile_config,
                                    const int         stages,
                                    const int         splitK,
                                    char*             workspace_ptr,
                                    const size_t      workspace_bytes,
                                    int               sm_,
                                    cudaStream_t      stream);

void cutlass_int8_fp16_gemm_per_tensor(const int8_t*     A,
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
                                    cudaStream_t      stream);


void cutlass_int8_int32_gemm_per_tensor(const int8_t*     A,
                                    const int8_t*     B,
                                    const float       alpha,
                                    const float       beta,
                                    int32_t*          C,
                                    int               m,
                                    int               n,
                                    int               k,
                                    std::string       tile_config,
                                    const int         stages,
                                    const int         splitK,
                                    char*             workspace_ptr,
                                    const size_t      workspace_bytes,
                                    int               sm_,
                                    cudaStream_t      stream);
void cutlass_int8_fp16_gemm_per_tensor_splitk(const int8_t*     A,
                                            const int8_t*     B,
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
                                            cudaStream_t      stream);
void cutlass_int8_int32_gemm_per_tensor_splitk(const int8_t*     A,
                                    const int8_t*     B,
                                    const float       alpha,
                                    const float       beta,
                                    int32_t*          C,
                                    int               m,
                                    int               n,
                                    int               k,
                                    std::string       tile_config,
                                    const int         stages,
                                    const int         splitK,
                                    char*             workspace_ptr,
                                    const size_t      workspace_bytes,
                                    int               sm_,
                                    cudaStream_t      stream);

}  // namespace fastertransformer

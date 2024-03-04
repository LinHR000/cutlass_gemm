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

#include "tensorrt_llm/common/cudaBf16Wrapper.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.h"
#include "tensorrt_llm/thop/thUtils.h"
namespace torch_ext
{
    using torch::Tensor;
    std::vector<Tensor> _symmetric_quantize_last_axis_of_batched_matrix(Tensor weight, int quant_mode);
    Tensor preprocess_weights_for_mixed_gemm_(Tensor row_major_quantized_weight, int quant_mode);
    Tensor pack_int8_tensor_to_packed_int4(Tensor weight);
    Tensor unpack_int4_packed_tensor_to_int8(Tensor weight);
    Tensor permute_B_rows_for_mixed_gemm(Tensor quantized_tensor, torch::ScalarType quant_type, const int64_t arch_version);
}
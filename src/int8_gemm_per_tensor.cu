#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>

#include "torch/csrc/cuda/Stream.h"
#include <torch/custom_class.h>
#include <torch/script.h>

#include "cutlass_kernels/int8_gemm_raw/int8_gemm_raw.h"
#include "utils/th_utils.h"
#include "utils/cuda_bf16_wrapper.h"
#include "utils/logger.h"
#include <string>
#include "cutlass/numeric_types.h"

using torch::Tensor;
using torch_ext::get_ptr;

namespace ft = fastertransformer;
Tensor gemm_in8_w8_ofp16_per_token(Tensor         input,
                                Tensor            weight,
                                float             alpha, 
                                float             beta,
                                int64_t           m,
                                int64_t           n,
                                int64_t           k,
                                std::string       tile_config,
                                const int               stages,
                                const int               splitK,
                                char*             workspace_ptr,
                                const size_t      workspace_bytes){
    at::ScalarType output_data_type = at::ScalarType::Half;
    Tensor output = torch::zeros({m, n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int sm = 80;
    ft::cutlass_int8_gemm_per_tensor<half>(get_ptr<int8_t>(input),
                                        get_ptr<int8_t>(weight),
                                        alpha,
                                        beta,
                                        get_ptr<half>(output),
                                        m,
                                        n,
                                        k,
                                        tile_config,
                                        stages,
                                        splitK,
                                        workspace_ptr,
                                        workspace_bytes,
                                        sm,
                                        stream);
    return output;
}

Tensor gemm_in8_w8_o8_per_token(Tensor         input,
                                Tensor            weight,
                                float             alpha, 
                                float             beta,
                                int64_t           m,
                                int64_t           n,
                                int64_t           k,
                                std::string       tile_config,
                                const int               stages,
                                const int               splitK,
                                char*             workspace_ptr,
                                const size_t      workspace_bytes){
    at::ScalarType output_data_type = at::ScalarType::Char;
    Tensor output = torch::zeros({m, n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int sm = 80;
    ft::cutlass_int8_gemm_per_tensor<int8_t>(get_ptr<int8_t>(input),
                                        get_ptr<int8_t>(weight),
                                        alpha,
                                        beta,
                                        get_ptr<int8_t>(output),
                                        m,
                                        n,
                                        k,
                                        tile_config,
                                        stages,
                                        splitK,
                                        workspace_ptr,
                                        workspace_bytes,
                                        sm,
                                        stream);
    return output;
}
Tensor gemm_in8_w8_o32_per_token(Tensor         input,
                                Tensor            weight,
                                float             alpha, 
                                float             beta,
                                int64_t           m,
                                int64_t           n,
                                int64_t           k,
                                std::string       tile_config,
                                const int         stages,
                                const int         splitK,
                                char*             workspace_ptr,
                                const size_t      workspace_bytes){
    at::ScalarType output_data_type = at::ScalarType::Int;
    Tensor output = torch::zeros({m, n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int sm=80;
    ft::cutlass_int8_gemm_per_tensor<int32_t>(get_ptr<int8_t>(input),
                                        get_ptr<int8_t>(weight),
                                        alpha,
                                        beta,
                                        get_ptr<int32_t>(output),
                                        m,
                                        n,
                                        k,
                                        tile_config,
                                        stages,
                                        splitK,
                                        workspace_ptr,
                                        workspace_bytes,
                                        sm,
                                        stream);
    return output;
}

// TORCH_LIBRARY(gemm_dq_int8_ops, m)
// {
//     m.def("int8_gemm_dq", int8_gemm_dq);
// }

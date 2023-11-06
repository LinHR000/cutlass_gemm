#include <torch/extension.h>
#include <c10/util/Optional.h>
#include <ATen/cuda/CUDAContext.h>
#include "torch/csrc/cuda/Stream.h"
#include <torch/custom_class.h>
#include <torch/script.h>
#include "cutlass_kernels/int8_gemm_raw/int8_gemm_raw.h"
#include "utils/th_utils.h"
using torch::Tensor;
using torch_ext::get_ptr;

namespace ft = fastertransformer;
Tensor gemm_in8_w8_ofp16_per_tensor(Tensor&         input,
                                Tensor&            weight,
                                c10::optional<torch::Tensor>&            bias,
                                float             alpha, 
                                float             beta,
                                int64_t           m,
                                int64_t           n,
                                int64_t           k,
                                std::string       tile_config,
                                const int               stages,
                                const int               splitK){
    at::ScalarType output_data_type = at::ScalarType::Half;
    Tensor output = torch::zeros({m, n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int sm = 80;
    // auto bias_ptr = bias ? get_ptr<__half>(bias) : nullptr;
    // const __half* bias_ptr = bias ? get_ptr<__half>(bias):nullptr;
    const __half* bias_ptr = bias ?
    reinterpret_cast<const __half*>(bias.value().data_ptr())
    : nullptr;

    ft::cutlass_int8_fp16_gemm_per_tensor(get_ptr<int8_t>(input),
                                        get_ptr<int8_t>(weight),
                                        bias_ptr,
                                        alpha,
                                        beta,
                                        get_ptr<__half>(output),
                                        m,
                                        n,
                                        k,
                                        tile_config,
                                        stages,
                                        splitK,
                                        nullptr,
                                        0,
                                        sm,
                                        stream);
    return output;
}

Tensor gemm_in8_w8_ofp16_gelu_per_tensor(Tensor&         input,
                                Tensor&            weight,
                                c10::optional<torch::Tensor>&            bias,
                                float             alpha, 
                                float             beta,
                                int64_t           m,
                                int64_t           n,
                                int64_t           k,
                                std::string       tile_config,
                                const int               stages,
                                const int               splitK){
    at::ScalarType output_data_type = at::ScalarType::Half;
    Tensor output = torch::zeros({m, n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int sm = 80;
    // auto bias_ptr = bias ? get_ptr<__half>(bias) : nullptr;
    // const __half* bias_ptr = bias ? get_ptr<__half>(bias):nullptr;
    const __half* bias_ptr = bias ?
    reinterpret_cast<const __half*>(bias.value().data_ptr())
    : nullptr;

    ft::cutlass_int8_fp16_gemm_per_tensor_gelu(get_ptr<int8_t>(input),
                                        get_ptr<int8_t>(weight),
                                        bias_ptr,
                                        alpha,
                                        beta,
                                        get_ptr<__half>(output),
                                        m,
                                        n,
                                        k,
                                        tile_config,
                                        stages,
                                        splitK,
                                        nullptr,
                                        0,
                                        sm,
                                        stream);
    return output;
}


Tensor gemm_in8_w8_o8_per_tensor(Tensor         input,
                                Tensor            weight,
                                float             alpha, 
                                float             beta,
                                int64_t           m,
                                int64_t           n,
                                int64_t           k,
                                std::string       tile_config,
                                const int               stages,
                                const int               splitK){
    at::ScalarType output_data_type = at::ScalarType::Char;
    Tensor output = torch::zeros({m, n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int sm = 80;
    ft::cutlass_int8_int8_gemm_per_tensor(get_ptr<int8_t>(input),
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
                                        nullptr,
                                        0,
                                        sm,
                                        stream); 
    return output;                               
 }

Tensor gemm_in8_w8_o32_per_tensor(Tensor         input,
                                Tensor            weight,
                                float             alpha, 
                                float             beta,
                                int64_t           m,
                                int64_t           n,
                                int64_t           k,
                                std::string       tile_config,
                                const int               stages,
                                const int               splitK){
    at::ScalarType output_data_type = at::ScalarType::Int;
    Tensor output = torch::zeros({m, n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int sm=80;
    ft::cutlass_int8_int32_gemm_per_tensor(get_ptr<int8_t>(input),
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
                                        nullptr,
                                        0,
                                        sm,
                                        stream);
    return output;                                
}  

Tensor gemm_in8_w8_ofp16_per_tensor_splitk(Tensor         input,
                                        Tensor            weight,
                                        float             alpha, 
                                        float             beta,
                                        int64_t           m,
                                        int64_t           n,
                                        int64_t           k,
                                        std::string       tile_config,
                                        const int               stages,
                                        const int               splitK){
    at::ScalarType output_data_type = at::ScalarType::Half;
    Tensor output = torch::zeros({m, n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int sm = 80;
    ft::cutlass_int8_fp16_gemm_per_tensor_splitk(get_ptr<int8_t>(input),
                                        get_ptr<int8_t>(weight),
                                        alpha,
                                        beta,
                                        get_ptr<__half>(output),
                                        m,
                                        n,
                                        k,
                                        tile_config,
                                        stages,
                                        splitK,
                                        nullptr,
                                        0,
                                        sm,
                                        stream);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def(
    "gemm_in8_w8_ofp16_per_tensor",
    &gemm_in8_w8_ofp16_per_tensor,
    "Compute the attention between an input query and the cached key/value tensors");
m.def(
    "gemm_in8_w8_o8_per_tensor",
    &gemm_in8_w8_o8_per_tensor,
    "Compute the attention between an input query and the cached key/value tensors");
m.def(
    "gemm_in8_w8_o32_per_tensor",
    &gemm_in8_w8_o32_per_tensor,
    "Compute the attention between an input query and the cached key/value tensors");
m.def(
    "gemm_in8_w8_ofp16_per_tensor_splitk",
    &gemm_in8_w8_ofp16_per_tensor_splitk,
    "Compute the attention between an input query and the cached key/value tensors");
m.def(
    "gemm_in8_w8_ofp16_gelu_per_tensor",
    &gemm_in8_w8_ofp16_gelu_per_tensor,
    "Compute the attention between an input query and the cached key/value tensors");
}
#include <torch/extension.h>
#include <c10/util/Optional.h>
#include <ATen/cuda/CUDAContext.h>
#include "torch/csrc/cuda/Stream.h"
#include <torch/custom_class.h>
#include <torch/script.h>
#include "cutlass_kernels/cutlass_mix_gemm/mix_gemm.h"
#include "utils/th_utils.h"
using torch::Tensor;
using torch_ext::get_ptr;

namespace ft = fastertransformer;
Tensor fp16_int8_gemm_per_tensor(Tensor&         input,
                                Tensor&            weight,
                                c10::optional<torch::Tensor>&            bias,
                                float             alpha, 
                                float             beta,
                                int64_t           m,
                                int64_t           n,
                                int64_t           k,
                                std::string       tile_config,
                                const int               stages){
    at::ScalarType output_data_type = at::ScalarType::Half;
    Tensor output;
    int stridea;
    int strideb;
    int stridec;
    if (input.dim() == 2){
        output = torch::zeros({m, n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
        stridea =  input.stride(0);
        strideb =  weight.stride(0);
        stridec =  output.stride(0);

    }else if (input.dim() == 3){
        output = torch::empty({input.size(0),input.size(1), n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
        stridea =  input.stride(1);
        strideb =  weight.stride(0);
        stridec =  output.stride(1);
    }else{
        throw std::runtime_error("Invalid rank for activations");
    }
    // Tensor output = torch::zeros({m, n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int sm = 80;
    // auto bias_ptr = bias ? get_ptr<__half>(bias) : nullptr;
    // const __half* bias_ptr = bias ? get_ptr<__half>(bias):nullptr;
    __half* bias_ptr = bias ? reinterpret_cast<__half*>(bias.value().data_ptr()) : nullptr;

    ft::cutlass_fp16_int8_gemm_per_tensor(get_ptr<__half>(input),
                                        get_ptr<int8_t>(weight),
                                        bias_ptr,
                                        alpha,
                                        beta,
                                        get_ptr<__half>(output),
                                        m,
                                        n,
                                        k,
                                        stridea,
                                        strideb,
                                        stridec,
                                        tile_config,
                                        stages,
                                        1,
                                        nullptr,
                                        0,
                                        sm,
                                        stream);
    return output;
}

Tensor bf16_int8_gemm_per_tensor(Tensor&         input,
                                Tensor&            weight,
                                c10::optional<torch::Tensor>&            bias,
                                float             alpha, 
                                float             beta,
                                int64_t           m,
                                int64_t           n,
                                int64_t           k,
                                std::string       tile_config,
                                const int               stages){
    at::ScalarType output_data_type = at::ScalarType::BFloat16;
    Tensor output;
    int stridea;
    int strideb;
    int stridec;
    if (input.dim() == 2){
        output = torch::zeros({m, n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
        stridea =  input.stride(0);
        strideb =  weight.stride(0);
        stridec =  output.stride(0);

    }else if (input.dim() == 3){
        output = torch::empty({input.size(0),input.size(1), n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
        stridea =  input.stride(1);
        strideb =  weight.stride(0);
        stridec =  output.stride(1);
    }else{
        throw std::runtime_error("Invalid rank for activations");
    }
    // Tensor output = torch::zeros({m, n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int sm = 80;
    // auto bias_ptr = bias ? get_ptr<__half>(bias) : nullptr;
    // const __half* bias_ptr = bias ? get_ptr<__half>(bias):nullptr;
    __nv_bfloat16* bias_ptr = bias ? reinterpret_cast<__nv_bfloat16*>(bias.value().data_ptr()) : nullptr;

    ft::cutlass_bf16_int8_gemm_per_tensor(get_ptr<__nv_bfloat16>(input),
                                        get_ptr<int8_t>(weight),
                                        bias_ptr,
                                        alpha,
                                        beta,
                                        get_ptr<__nv_bfloat16>(output),
                                        m,
                                        n,
                                        k,
                                        stridea,
                                        strideb,
                                        stridec,
                                        tile_config,
                                        stages,
                                        1,
                                        nullptr,
                                        0,
                                        sm,
                                        stream);
    return output;
}


// Tensor fp16_int8_gemm_per_tensor_splitk(Tensor&         input,
//                                 Tensor&            weight,
//                                 c10::optional<torch::Tensor>&            bias,
//                                 float             alpha, 
//                                 float             beta,
//                                 int64_t           m,
//                                 int64_t           n,
//                                 int64_t           k,
//                                 std::string       tile_config,
//                                 const int               stages,
//                                 int                splitK){
//     at::ScalarType output_data_type = at::ScalarType::Half;
//     Tensor output;
//     int stridea;
//     int strideb;
//     int stridec;
//     if (input.dim() == 2){
//         output = torch::zeros({m, n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
//         stridea =  input.stride(0);
//         strideb =  weight.stride(0);
//         stridec =  output.stride(0);

//     }else if (input.dim() == 3){
//         output = torch::empty({input.size(0),input.size(1), n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
//         stridea =  input.stride(1);
//         strideb =  weight.stride(0);
//         stridec =  output.stride(1);
//     }else{
//         throw std::runtime_error("Invalid rank for activations");
//     }
//     // Tensor output = torch::zeros({m, n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
//     auto stream = at::cuda::getCurrentCUDAStream().stream();
//     int sm = 80;
//     // auto bias_ptr = bias ? get_ptr<__half>(bias) : nullptr;
//     // const __half* bias_ptr = bias ? get_ptr<__half>(bias):nullptr;
//     __half* bias_ptr = bias ? reinterpret_cast<__half*>(bias.value().data_ptr()) : nullptr;

//     ft::cutlass_fp16_int8_gemm_per_tensor_splitk(get_ptr<__half>(input),
//                                         get_ptr<int8_t>(weight),
//                                         alpha,
//                                         beta,
//                                         get_ptr<__half>(output),
//                                         m,
//                                         n,
//                                         k,
//                                         stridea,
//                                         strideb,
//                                         stridec,
//                                         tile_config,
//                                         stages,
//                                         splitK,
//                                         nullptr,
//                                         0,
//                                         sm,
//                                         stream);
//     return output;
// }



// Tensor bf16_int8_gemm_per_tensor_splitk(Tensor&         input,
//                                 Tensor&            weight,
//                                 c10::optional<torch::Tensor>&            bias,
//                                 float             alpha, 
//                                 float             beta,
//                                 int64_t           m,
//                                 int64_t           n,
//                                 int64_t           k,
//                                 std::string       tile_config,
//                                 const int               stages,
//                                 int                splitK){
//     at::ScalarType output_data_type = at::ScalarType::BFloat16;
//     Tensor output;
//     int stridea;
//     int strideb;
//     int stridec;
//     if (input.dim() == 2){
//         output = torch::zeros({m, n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
//         stridea =  input.stride(0);
//         strideb =  weight.stride(0);
//         stridec =  output.stride(0);

//     }else if (input.dim() == 3){
//         output = torch::empty({input.size(0),input.size(1), n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
//         stridea =  input.stride(1);
//         strideb =  weight.stride(0);
//         stridec =  output.stride(1);
//     }else{
//         throw std::runtime_error("Invalid rank for activations");
//     }
//     // Tensor output = torch::zeros({m, n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
//     auto stream = at::cuda::getCurrentCUDAStream().stream();
//     int sm = 80;
//     // auto bias_ptr = bias ? get_ptr<__half>(bias) : nullptr;
//     // const __half* bias_ptr = bias ? get_ptr<__half>(bias):nullptr;
//     __nv_bfloat16* bias_ptr = bias ? reinterpret_cast<__nv_bfloat16*>(bias.value().data_ptr()) : nullptr;

//     ft::cutlass_bf16_int8_gemm_per_tensor_splitk(get_ptr<__nv_bfloat16>(input),
//                                         get_ptr<int8_t>(weight),
//                                         alpha,
//                                         beta,
//                                         get_ptr<__nv_bfloat16>(output),
//                                         m,
//                                         n,
//                                         k,
//                                         stridea,
//                                         strideb,
//                                         stridec,
//                                         tile_config,
//                                         stages,
//                                         splitK,
//                                         nullptr,
//                                         0,
//                                         sm,
//                                         stream);
//     return output;
// }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def(
    "fp16_int8_gemm_per_tensor",
    &fp16_int8_gemm_per_tensor,
    "Compute the attention between an input query and the cached key/value tensors");
m.def(
    "bf16_int8_gemm_per_tensor",
    &bf16_int8_gemm_per_tensor,
    "Compute the attention between an input query and the cached key/value tensors");
// m.def(
//     "fp16_int8_gemm_per_tensor_splitk",
//     &fp16_int8_gemm_per_tensor_splitk,
//     "Compute the attention between an input query and the cached key/value tensors");
// m.def(
//     "bf16_int8_gemm_per_tensor_splitk",
//     &bf16_int8_gemm_per_tensor_splitk,
//     "Compute the attention between an input query and the cached key/value tensors");
}
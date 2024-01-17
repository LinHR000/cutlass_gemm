#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>

#include "torch/csrc/cuda/Stream.h"
#include <torch/custom_class.h>
#include <torch/script.h>

#include "cutlass_kernels/int8_gemm/int8_gemm.h"
#include "utils/th_utils.h"
#include "utils/cuda_bf16_wrapper.h"
#include "utils/logger.h"

#include "cutlass/numeric_types.h"

using torch::Tensor;
using torch_ext::get_ptr;

namespace ft = fastertransformer;
Tensor gemm_in8_w8_ofp16_pt(Tensor output,
                            Tensor input,
                            Tensor weight,
                            Tensor alpha_col, 
                            Tensor alpha_row,
                            int64_t m,
                            int64_t n,
                            int64_t k,
                            int     tile_config,
                            int     split_k_style,
                            int     split_k_factor,
                            int     stages){
    at::ScalarType output_data_type = at::ScalarType::Half;
    const at::ScalarType at_fp32  = at::ScalarType::Float;
    ft::CutlassInt8GemmRunner<half> cutlass_runner_half;
    // Tensor output;
    // if (input.dim() == 2){
    //     output = torch::zeros({m, n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
    // }else if (input.dim() == 3){
    //     output = torch::empty({input.size(0),input.size(1), n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
    // }else{
    //     throw std::runtime_error("Invalid rank for activations");
    // }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    QuantMode quant_mode = QuantMode::PerTokenQuant;
    cutlass_runner_half.gemm(get_ptr<int8_t>(input),
            get_ptr<int8_t>(weight),
            quant_mode,
            get_ptr<float>(alpha_col),
            get_ptr<float>(alpha_row),
            get_ptr<half>(output),
            m,
            n,
            k,
            tile_config,
            split_k_style,
            split_k_factor,
            stages,
            nullptr,
            0,
            stream);
    return output;
}

Tensor gemm_in8_w8_ofp16_pc(Tensor output,
                            Tensor input,
                            Tensor weight,
                            Tensor alpha_col, 
                            Tensor alpha_row,
                            int64_t m,
                            int64_t n,
                            int64_t k,
                            int     tile_config,
                            int     split_k_style,
                            int     split_k_factor,
                            int     stages){
    at::ScalarType output_data_type = at::ScalarType::Half;
    const at::ScalarType at_fp32  = at::ScalarType::Float;
    ft::CutlassInt8GemmRunner<half> cutlass_runner_half;
//     Tensor output = torch::zeros({m, n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
    // Tensor output;
    // if (input.dim() == 2){
    //     output = torch::zeros({m, n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
    // }else if (input.dim() == 3){
    //     output = torch::empty({input.size(0),input.size(1), n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
    // }else{
    //     throw std::runtime_error("Invalid rank for activations");
    // }
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    QuantMode quant_mode = QuantMode::PerChannelQuant;
    cutlass_runner_half.gemm(get_ptr<int8_t>(input),
            get_ptr<int8_t>(weight),
            quant_mode,
            get_ptr<float>(alpha_col),
            get_ptr<float>(alpha_row),
            get_ptr<half>(output),
            m,
            n,
            k,
            tile_config,
            split_k_style,
            split_k_factor,
            stages,
            nullptr,
            0,
            stream);
    return output;
}

Tensor gemm_in8_w8_ofp16_ptpc(Tensor output,
                            Tensor input,
                            Tensor weight,
                            Tensor alpha_col, 
                            Tensor alpha_row,
                            int64_t m,
                            int64_t n,
                            int64_t k,
                            int     tile_config,
                            int     split_k_style,
                            int     split_k_factor,
                            int     stages){
    at::ScalarType output_data_type = at::ScalarType::Half;
    const at::ScalarType at_fp32  = at::ScalarType::Float;
    ft::CutlassInt8GemmRunner<half> cutlass_runner_half;
//     Tensor output = torch::zeros({m, n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
    // Tensor output;
    // if (input.dim() == 2){
    //     output = torch::zeros({m, n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
    // }else if (input.dim() == 3){
    //     output = torch::empty({input.size(0),input.size(1), n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
    // }else{
    //     throw std::runtime_error("Invalid rank for activations");
    // }
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    QuantMode quant_mode = QuantMode::PerTokenChannelQuant;
    cutlass_runner_half.gemm(get_ptr<int8_t>(input),
            get_ptr<int8_t>(weight),
            quant_mode,
            get_ptr<float>(alpha_col),
            get_ptr<float>(alpha_row),
            get_ptr<half>(output),
            m,
            n,
            k,
            tile_config,
            split_k_style,
            split_k_factor,
            stages,
            nullptr,
            0,
            stream);
    return output;
}

std::vector<int> gemm_in8_w8_ofp16_config(Tensor output,
                            Tensor input,
                            Tensor weight,
                            Tensor alpha_col, 
                            Tensor alpha_row,
                            int64_t m,
                            int64_t n,
                            int64_t k){
    at::ScalarType output_data_type = at::ScalarType::Half;
    const at::ScalarType at_fp32  = at::ScalarType::Float;
    ft::CutlassInt8GemmRunner<half> cutlass_runner_half;
    // Tensor output;
    // if (input.dim() == 2){
    //     output = torch::zeros({m, n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
    // }else if (input.dim() == 3){
    //     output = torch::empty({input.size(0),input.size(1), n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
    // }else{
    //     throw std::runtime_error("Invalid rank for activations");
    // }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    QuantMode quant_mode = QuantMode::PerTokenQuant;
    
    std::vector<int> out = cutlass_runner_half.run_gemm_config(get_ptr<int8_t>(input),
            get_ptr<int8_t>(weight),
            quant_mode,
            get_ptr<float>(alpha_col),
            get_ptr<float>(alpha_row),
            get_ptr<half>(output),
            m,
            n,
            k,
            nullptr,
            0,
            stream);
    return out;
}

// TORCH_LIBRARY(gemm_dq_int8_ops, m)
// {
//     m.def("int8_gemm_dq", int8_gemm_dq);
// }

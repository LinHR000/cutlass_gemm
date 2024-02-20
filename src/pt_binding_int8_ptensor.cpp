#include <torch/extension.h>
#include <c10/util/Optional.h>
#include <ATen/cuda/CUDAContext.h>
#include "torch/csrc/cuda/Stream.h"
#include <torch/custom_class.h>
#include <torch/script.h>
#include "tensorrt_llm//kernels/moe_gemm/moe_gemm_kernels.h"
#include "utils/th_utils.h"
using torch::Tensor;
using torch_ext::get_ptr;
using tensorrt_llm;

Tensor moe_gemm(Tensor&         input,
                Tensor&         weight,
                c10::optional<torch::Tensor>&            out, // if out is None, allocate a new tensor
                c10::optional<torch::Tensor>&            bias,
                c10::optional<torch::Tensor>&            weight_scales,
                int64_t         total_rows_before_expert,
                int64_t         total_rows,
                int64_t         gemm_n,
                int64_t         gemm_k,
                int             num_experts,
                int             activation_type,
                int             tile_config, // if tie_config is -1, use default tile config
                int             split_k_style,
                int             split_k_factor,
                int             stages

){
    // if out is None, allocate a new tensor
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    bool allocate_out = out ? false : true;
    at::ScalarType output_data_type = input.scalar_type();
    if (input.dim() == 2){
        output = torch::zeros({input.size(0), n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
    }else if (input.dim() == 3){
        output = torch::empty({input.size(0),input.size(1), n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
    }else{
        throw std::runtime_error("Invalid rank for activations");
    }
    void *input_ptr, *weight_ptr, *weight_scales_ptr, *bias_ptr, *output_ptr;
    // according input data type, initialize the moe gemm runner and choose the corresponding gemm function
    bool use_bias = bias ? true : false;
    if (input.scalar_type() == at::ScalarType::Half){
        input_ptr = get_ptr<half>(input);
        weight_scales_ptr = get_ptr<half>(weight_scales);
        if (use_bias){
            bias_ptr = get_ptr<half>(bias);
        }
        output_ptr = get_ptr<half>(output);
        // according weight data type, choose the corresponding gemm function
        if (weight.scalar_type() == at::ScalarType::Half){
            MoeGemmRunner<half,half> cutlass_runner;
            weight_ptr = get_ptr<half>(weight);
        }else if (weight.scalar_type() == at::ScalarType::Char){
            MoeGemmRunner<half,uint8_t> cutlass_runner;
            weight_ptr = get_ptr<uint8_t>(weight);
        } else if (weight.scalar_type() == at::ScalarType::QInt4x2){
            MoeGemmRunner<half,cutlass::uint4b_t> cutlass_runner;
            weight_ptr = get_ptr<cutlass::uint4b_t>(weight);
        } else{
            throw std::runtime_error("unsupported data type");
        }
    }else if (input.scalar_type() == at::ScalarType::BFloat16){
        input_ptr = get_ptr<half>(input);
        weight_scales_ptr = get_ptr<__nv_bfloat16>(weight_scales);
        if (use_bias){
            bias_ptr = get_ptr<__nv_bfloat16>(bias);
        }
        output_ptr = get_ptr<__nv_bfloat16>(output);

        if (weight.scalar_type() == at::ScalarType::BFloat16){
            MoeGemmRunner<__nv_bfloat16,__nv_bfloat16> cutlass_runner;
            weight_ptr = get_ptr<half>(weight);
        }else if (weight.scalar_type() == at::ScalarType::Char){
            MoeGemmRunner<__nv_bfloat16,uint8_t> cutlass_runner;
            weight_ptr = get_ptr<uint8_t>(weight);
        } else if (weight.scalar_type() == at::ScalarType::QInt4x2){
            MoeGemmRunner<__nv_bfloat16,cutlass::uint4b_t> cutlass_runner;
            weight_ptr = get_ptr<cutlass::uint4b_t>(weight);
        } else{
            throw std::runtime_error("unsupported data type");
        }
    } else{
        throw std::runtime_error("unsupported data type");
    }

    // run the moe gemm
    if (use_bias){ // use bias api
        cutlass_runner.moeGemmBiasAct(input_ptr,
                                      weight_ptr,
                                      weight_scales_ptr,
                                      bias_ptr,
                                      output_ptr,
                                      &total_rows_before_expert,
                                      total_rows,
                                      gemm_n,
                                      gemm_k,
                                      num_experts,
                                      ActivationType(activation_type),
                                      tile_config,
                                      split_k_style,
                                      split_k_factor,
                                      stages
                                      stream);
    } else{
        cutlass_runner.moeGemm(input_ptr,
                               weight_ptr,
                               weight_scales_ptr,
                               bias_ptr,
                               output_ptr,
                              &total_rows_before_expert,
                              total_rows,
                              gemm_n,
                              gemm_k,
                              num_experts,
                              ActivationType(activation_type),
                              tile_config,
                              split_k_style,
                              split_k_factor,
                              stages
                              stream);
    }
    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moe_gemm", &moe_gemm, "moe gemm");}
#include <torch/extension.h>
#include <c10/util/Optional.h>
#include <ATen/cuda/CUDAContext.h>
#include "torch/csrc/cuda/Stream.h"
#include <torch/custom_class.h>
#include <torch/script.h>
#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels.h"
#include "tensorrt_llm/thop/thUtils.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
using torch::Tensor;
using torch_ext::get_ptr;
// using tensorrt_llm;

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
    Tensor output;
    if (allocate_out){
        if (input.dim() == 2){
            output = torch::zeros({input.size(0), gemm_n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
        }else if (input.dim() == 3){
            output = torch::empty({input.size(0),input.size(1), gemm_n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
        }else{
            throw std::runtime_error("Invalid rank for activations");
        }
    }else{
        output = *out;
    }

    tensorrt_llm::cutlass_extensions::CutlassGemmConfig* gemm_config;
    //get gemm config
    if (tile_config != -1) {
        gemm_config = new tensorrt_llm::cutlass_extensions::CutlassGemmConfig(tensorrt_llm::cutlass_extensions::CutlassTileConfig(tile_config), 
                                                                            tensorrt_llm::cutlass_extensions::SplitKStyle(split_k_style), 
                                                                            split_k_factor, stages);
    }else{
        gemm_config = nullptr;
    }

    // according input data type, initialize the moe gemm runner and choose the corresponding gemm function
    bool use_bias = bias ? true : false;
    bool use_weight_scales = weight_scales ? true : false;
    if (input.scalar_type() == at::ScalarType::Half){
        // according weight data type, choose the corresponding gemm function
        half* bias_ptr = bias ? reinterpret_cast<half*>(bias.value().data_ptr()) : nullptr;
        half* weight_scales_ptr = weight_scales ? reinterpret_cast<half*>(weight_scales.value().data_ptr()) : nullptr;
        if (weight.scalar_type() == at::ScalarType::Half){
            tensorrt_llm::MoeGemmRunner<half,half> cutlass_runner;
                // run the moe gemm
            if (use_bias){ // use bias api
                cutlass_runner.moeGemmBiasAct(get_ptr<half>(input),
                                            get_ptr<half>(weight),
                                            weight_scales_ptr,
                                            bias_ptr,
                                            get_ptr<half>(output),
                                            &total_rows_before_expert,
                                            total_rows,
                                            gemm_n,
                                            gemm_k,
                                            num_experts,
                                            tensorrt_llm::ActivationType(activation_type),
                                            stream);
            } 
            else{
                cutlass_runner.moeGemm(get_ptr<half>(input),
                                    get_ptr<half>(weight),
                                    weight_scales_ptr,
                                    get_ptr<half>(output),
                                    &total_rows_before_expert,
                                    total_rows,
                                    gemm_n,
                                    gemm_k,
                                    num_experts,
                                    stream);
            }
        }else if (weight.scalar_type() == at::ScalarType::Char){
            tensorrt_llm::MoeGemmRunner<half,uint8_t> cutlass_runner;
                // run the moe gemm
            if (use_bias){ // use bias api
                cutlass_runner.moeGemmBiasAct(get_ptr<half>(input),
                                            get_ptr<uint8_t>(weight),
                                            weight_scales_ptr,
                                            bias_ptr,
                                            get_ptr<half>(output),
                                            &total_rows_before_expert,
                                            total_rows,
                                            gemm_n,
                                            gemm_k,
                                            num_experts,
                                            tensorrt_llm::ActivationType(activation_type),
                                            stream);
            } 
            else{
                cutlass_runner.moeGemm(get_ptr<half>(input),
                                    get_ptr<uint8_t>(weight),
                                    weight_scales_ptr,
                                    get_ptr<half>(output),
                                    &total_rows_before_expert,
                                    total_rows,
                                    gemm_n,
                                    gemm_k,
                                    num_experts,
                                    stream);
            }
        } else if (weight.scalar_type() == at::ScalarType::QUInt4x2){
            tensorrt_llm::MoeGemmRunner<half,cutlass::uint4b_t> cutlass_runner;
                // run the moe gemm
            if (use_bias){ // use bias api
                cutlass_runner.moeGemmBiasAct(get_ptr<half>(input),
                                            get_ptr<cutlass::uint4b_t>(weight),
                                            weight_scales_ptr,
                                            bias_ptr,
                                            get_ptr<half>(output),
                                            &total_rows_before_expert,
                                            total_rows,
                                            gemm_n,
                                            gemm_k,
                                            num_experts,
                                            tensorrt_llm::ActivationType(activation_type),
                                            stream);
            } 
            else{
                cutlass_runner.moeGemm(get_ptr<half>(input),
                                    get_ptr<cutlass::uint4b_t>(weight),
                                    weight_scales_ptr,
                                    get_ptr<half>(output),
                                    &total_rows_before_expert,
                                    total_rows,
                                    gemm_n,
                                    gemm_k,
                                    num_experts,
                                    stream);
            }
        } else{
            throw std::runtime_error("unsupported data type");
        }
    }else if (input.scalar_type() == at::ScalarType::BFloat16){
        __nv_bfloat16* bias_ptr = bias ? reinterpret_cast<__nv_bfloat16*>(bias.value().data_ptr()) : nullptr;
        __nv_bfloat16* weight_scales_ptr = weight_scales ? reinterpret_cast<__nv_bfloat16*>(weight_scales.value().data_ptr()) : nullptr;
        if (weight.scalar_type() == at::ScalarType::BFloat16){
            tensorrt_llm::MoeGemmRunner<__nv_bfloat16,__nv_bfloat16> cutlass_runner;
            if (use_bias){ // use bias api
                cutlass_runner.moeGemmBiasAct(get_ptr<__nv_bfloat16>(input),
                                            get_ptr<__nv_bfloat16>(weight),
                                            weight_scales_ptr,
                                            bias_ptr,
                                            get_ptr<__nv_bfloat16>(output),
                                            &total_rows_before_expert,
                                            total_rows,
                                            gemm_n,
                                            gemm_k,
                                            num_experts,
                                            tensorrt_llm::ActivationType(activation_type),
                                            stream);
            } 
            else{
                cutlass_runner.moeGemm(get_ptr<__nv_bfloat16>(input),
                                    get_ptr<__nv_bfloat16>(weight),
                                    weight_scales_ptr,
                                    get_ptr<__nv_bfloat16>(output),
                                    &total_rows_before_expert,
                                    total_rows,
                                    gemm_n,
                                    gemm_k,
                                    num_experts,
                                    stream);
            }
        }else if (weight.scalar_type() == at::ScalarType::Char){
            tensorrt_llm::MoeGemmRunner<__nv_bfloat16,uint8_t> cutlass_runner;
                // run the moe gemm
            if (use_bias){ // use bias api
                cutlass_runner.moeGemmBiasAct(get_ptr<__nv_bfloat16>(input),
                                            get_ptr<uint8_t>(weight),
                                            weight_scales_ptr,
                                            bias_ptr,
                                            get_ptr<__nv_bfloat16>(output),
                                            &total_rows_before_expert,
                                            total_rows,
                                            gemm_n,
                                            gemm_k,
                                            num_experts,
                                            tensorrt_llm::ActivationType(activation_type),
                                            stream);
            }
            else{
                cutlass_runner.moeGemm(get_ptr<__nv_bfloat16>(input),
                                    get_ptr<uint8_t>(weight),
                                    weight_scales_ptr,
                                    get_ptr<__nv_bfloat16>(output),
                                    &total_rows_before_expert,
                                    total_rows,
                                    gemm_n,
                                    gemm_k,
                                    num_experts,
                                    stream);
            }
        } else if (weight.scalar_type() == at::ScalarType::QUInt4x2){
            tensorrt_llm::MoeGemmRunner<__nv_bfloat16,cutlass::uint4b_t> cutlass_runner;
                // run the moe gemm
            if (use_bias){ // use bias api
                cutlass_runner.moeGemmBiasAct(get_ptr<__nv_bfloat16>(input),
                                            get_ptr<cutlass::uint4b_t>(weight),
                                            weight_scales_ptr,
                                            bias_ptr,
                                            get_ptr<__nv_bfloat16>(output),
                                            &total_rows_before_expert,
                                            total_rows,
                                            gemm_n,
                                            gemm_k,
                                            num_experts,
                                            tensorrt_llm::ActivationType(activation_type),
                                            stream);
            } 
            else{
                cutlass_runner.moeGemm(get_ptr<__nv_bfloat16>(input),
                                    get_ptr<cutlass::uint4b_t>(weight),
                                    weight_scales_ptr,
                                    get_ptr<__nv_bfloat16>(output),
                                    &total_rows_before_expert,
                                    total_rows,
                                    gemm_n,
                                    gemm_k,
                                    num_experts,
                                    stream);
            }
        } else{
            throw std::runtime_error("unsupported data type");
        }
    } else{
        throw std::runtime_error("unsupported data type");
    }

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moe_gemm", &moe_gemm, "moe gemm");}
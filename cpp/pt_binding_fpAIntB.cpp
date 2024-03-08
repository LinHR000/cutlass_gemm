
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include "torch/csrc/cuda/Stream.h"
#include <torch/custom_class.h>
#include <torch/script.h>
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "tensorrt_llm/cutlass_extensions/include/cutlass_extensions/gemm_configs.h"
#include "tensorrt_llm/thop/thUtils.h"
#include "tensorrt_llm/common/cudaBf16Wrapper.h"
#include "cutlass/numeric_types.h"
#include <cstdlib>
#include <chrono>
#include <optional>

using torch::Tensor;
using torch_ext::get_ptr;

tensorrt_llm::cutlass_extensions::CutlassGemmConfig getCusConfig(std::string tile_config,std::string split_k_style, int split_k_factor,int stages)
{
    tensorrt_llm::cutlass_extensions::CutlassTileConfig choose_tile_config;
    tensorrt_llm::cutlass_extensions::SplitKStyle choose_split_k_style;
    if (tile_config == "CtaShape32x128x64_WarpShape32x32x64"){
        choose_tile_config = tensorrt_llm::cutlass_extensions::CutlassTileConfig::CtaShape128x128x8_WarpShape64x64x8;
    }else if (tile_config == "CtaShape32x128x64_WarpShape32x32x64"){
        choose_tile_config = tensorrt_llm::cutlass_extensions::CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64;
    }else if (tile_config == "CtaShape64x128x64_WarpShape32x64x64"){
        choose_tile_config = tensorrt_llm::cutlass_extensions::CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64;
    }else if (tile_config == "CtaShape64x64x128_WarpShape32x64x64"){
        choose_tile_config = tensorrt_llm::cutlass_extensions::CutlassTileConfig::CtaShape64x64x128_WarpShape32x64x64;
    }else if (tile_config == "CtaShape64x128x64_WarpShape64x32x64"){
        choose_tile_config = tensorrt_llm::cutlass_extensions::CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64;
    }else if (tile_config == "CtaShape128x64x64_WarpShape64x32x64"){
        choose_tile_config = tensorrt_llm::cutlass_extensions::CutlassTileConfig::CtaShape128x64x64_WarpShape64x32x64;
    }else if (tile_config == "CtaShape128x128x64_WarpShape64x32x64"){
        choose_tile_config = tensorrt_llm::cutlass_extensions::CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64;
    }else if (tile_config == "CtaShape128x128x64_WarpShape64x64x64"){
        choose_tile_config = tensorrt_llm::cutlass_extensions::CutlassTileConfig::CtaShape128x128x64_WarpShape64x64x64;
    }else if (tile_config == "CtaShape128x128x64_WarpShape128x32x64"){
        choose_tile_config = tensorrt_llm::cutlass_extensions::CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64;
    }else if (tile_config == "CtaShape128x256x64_WarpShape64x64x64"){
        choose_tile_config = tensorrt_llm::cutlass_extensions::CutlassTileConfig::CtaShape128x256x64_WarpShape64x64x64;
    }else if (tile_config == "CtaShape256x128x64_WarpShape64x64x64"){
        choose_tile_config = tensorrt_llm::cutlass_extensions::CutlassTileConfig::CtaShape256x128x64_WarpShape64x64x64;
    }else{
        std::cout << "TileConfig Type: " <<  tile_config << " not supported !";
    }

    if (split_k_style == "NO_SPLIT_K"){
        choose_split_k_style = tensorrt_llm::cutlass_extensions::SplitKStyle::NO_SPLIT_K;
    }else if (split_k_style == "SPLIT_K_SERIAL"){
        choose_split_k_style = tensorrt_llm::cutlass_extensions::SplitKStyle::SPLIT_K_SERIAL;
    }else{
        std::cout << "SplitKStyle Type: " <<  split_k_style << " not supported !";
    }
    return tensorrt_llm::cutlass_extensions::CutlassGemmConfig(choose_tile_config,choose_split_k_style,split_k_factor,stages);
}


template<typename T, typename WeightType>
Tensor fused_gemm_dq_helper(
    Tensor                                  output_tensor,
    Tensor                                  input_activations, 
    Tensor                                  weight, 
    Tensor                                  weight_scales,
    c10::optional<torch::Tensor>            biases,
    c10::optional<torch::Tensor>            weight_zero_points,
    std::optional<int>                      group_size,
    std::optional<float>                    alpha,
    cutlass::WeightOnlyQuantOp              quant_op,
    std::optional<tensorrt_llm::cutlass_extensions::CutlassGemmConfig> gemm_config
    )
{
    const at::ScalarType _st    = input_activations.scalar_type();
    int m_ = 1;
    
    if (input_activations.dim() == 2){
        m_ = input_activations.size(0);
    }else if (input_activations.dim() == 3){
        m_ = input_activations.size(0) * input_activations.size(1);
    }else{
        throw std::runtime_error("Invalid rank for activations");
    }

    const int            m      = m_;
    const int            n      = weight_scales.size(0);
    const int            k      = weight.size(0);
    auto                 stream = at::cuda::getCurrentCUDAStream().stream();
    bool                 use_alpha = alpha ? true:false;

    const T*          input_act_ptr = get_ptr<const T>(input_activations);
    const WeightType* weight_ptr    = get_ptr<const WeightType>(weight);
    const T*          scales_ptr    = get_ptr<const T>(weight_scales);
    T*   output_tensor_ptr = get_ptr<T>(output_tensor);
    T*   weight_zero_points_ptr = weight_zero_points ? get_ptr<T>(weight_zero_points.value()) : nullptr;
    T*   bias_ptr = biases ? get_ptr<T>(biases.value()) : nullptr;

    if (quant_op == cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY){
        tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<T, WeightType, cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY> fused_gemm_dq_runner;
        const int ws_bytes = fused_gemm_dq_runner.getWorkspaceSize(m, n, k);
        auto ws_tensor     = torch::empty({ws_bytes}, torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));
        char* ws_ptr            = get_ptr<char>(ws_tensor);
        
        if (use_alpha){
            fused_gemm_dq_runner.gemm(input_act_ptr, 
                                    weight_ptr, 
                                    scales_ptr, 
                                    alpha.value(),
                                    output_tensor_ptr, 
                                    m,
                                    n, 
                                    k, 
                                    gemm_config,
                                    ws_ptr, 
                                    ws_bytes, 
                                    stream);
        }else{
            fused_gemm_dq_runner.gemm(input_act_ptr, 
                                    weight_ptr, 
                                    scales_ptr, 
                                    output_tensor_ptr, 
                                    m,
                                    n, 
                                    k, 
                                    gemm_config,
                                    ws_ptr, 
                                    ws_bytes, 
                                    stream);
        }
        

    }else if (quant_op == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY){
        tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<T, WeightType, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> fused_gemm_dq_runner;
        const int ws_bytes = fused_gemm_dq_runner.getWorkspaceSize(m, n, k);
        auto ws_tensor     = torch::empty({ws_bytes}, torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));
        char* ws_ptr            = get_ptr<char>(ws_tensor);
        if (use_alpha){
            fused_gemm_dq_runner.gemm(input_act_ptr,
                            weight_ptr,
                            scales_ptr,
                            weight_zero_points_ptr,
                            bias_ptr,
                            alpha.value(),
                            output_tensor_ptr,
                            m,
                            n,
                            k,
                            group_size.value(),
                            gemm_config,
                            ws_ptr,
                            ws_bytes,
                            stream);
        }else{
            fused_gemm_dq_runner.gemm(input_act_ptr,
                            weight_ptr,
                            scales_ptr,
                            weight_zero_points_ptr,
                            bias_ptr,
                            output_tensor_ptr,
                            m,
                            n,
                            k,
                            group_size.value(),
                            gemm_config,
                            ws_ptr,
                            ws_bytes,
                            stream);

        }
        
    }else if (quant_op == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS){
        tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<T, WeightType, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS> fused_gemm_dq_runner;
        const int ws_bytes = fused_gemm_dq_runner.getWorkspaceSize(m, n, k);
        auto ws_tensor     = torch::empty({ws_bytes}, torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));
        char* ws_ptr            = get_ptr<char>(ws_tensor);
        if (use_alpha){
            fused_gemm_dq_runner.gemm(input_act_ptr,
                            weight_ptr,
                            scales_ptr,
                            weight_zero_points_ptr,
                            nullptr,
                            alpha.value(),
                            output_tensor_ptr,
                            m,
                            n,
                            k,
                            group_size.value(),
                            gemm_config,
                            ws_ptr,
                            ws_bytes,
                            stream);

        }else{
            fused_gemm_dq_runner.gemm(input_act_ptr,
                            weight_ptr,
                            scales_ptr,
                            weight_zero_points_ptr,
                            nullptr,
                            output_tensor_ptr,
                            m,
                            n,
                            k,
                            group_size.value(),
                            gemm_config,
                            ws_ptr,
                            ws_bytes,
                            stream);

        }
    }

    

   return output_tensor;
}

Tensor fpAIntB_gemm(
    Tensor                                  input_activations, 
    Tensor                                  weight, 
    Tensor                                  weight_scales,
    c10::optional<torch::Tensor>&           bias,
    c10::optional<torch::Tensor>&           out,
    c10::optional<torch::Tensor>&           weight_zero_points,
    std::optional<float>                    alpha,
    std::optional<int>                      group_size,
    std::optional<std::string>              tile_config, // if tie_config is -1, use default tile config
    std::optional<std::string>              split_k_style,
    std::optional<int>                      split_k_factor,
    std::optional<int>                      stages
    )
{
    const at::ScalarType _st = input_activations.scalar_type();
    CHECK_INPUT(weight_scales, _st);

    // TORCH_CHECK(input_activations.dim() == 2, "Invalid rank for activations");
    TORCH_CHECK(weight.dim() == 2, "Invalid rank for weight");
    // TORCH_CHECK(weight_scales.dim() == 1, "Invalid rank for scales");

    cutlass::WeightOnlyQuantOp quant_op;
    bool use_weight_zero_points = weight_zero_points ? true : false;
    bool use_group = group_size ? true : false;
    if (use_group){
        if (use_weight_zero_points){
            quant_op = cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS;
        }else{
            quant_op = cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY;
        }
    }else{
        quant_op = cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY;
    }

    const int n = weight_scales.size(-1);
    // const int n = 1024;

    TORCH_CHECK(input_activations.size(1) == weight.size(0), "dim 1 of act and dim 0 of weight must be equal");

    // We signal int4 by having the last weight dim be half the size of the scales.
    // This is because int4 elements are packed into a single byte.
    torch::ScalarType quant_type = weight.scalar_type();
    if (weight.size(-1) == weight_scales.size(-1) / 2) {
        quant_type = at::ScalarType::QUInt4x2;
    }
    // quant_type = at::ScalarType::QUInt4x2;
    else {
        TORCH_CHECK(weight.size(-1) == weight_scales.size(-1),
                    "Last dim of weight and scales must be equal for int8 "
                    "or last dim of scale must be 2x last dim of weight for int4.");
    }
    Tensor output_tensor;
    bool allocate_out = out ? false : true;
    if (allocate_out){
        if (input_activations.dim() == 2) {
            output_tensor = torch::zeros({input_activations.size(0), n}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
        } else if (input_activations.dim() == 3) {
            output_tensor = torch::empty({input_activations.size(0), input_activations.size(1), n},
                                torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
        } else {
            throw std::runtime_error("Invalid rank for activations");
        }
    }else{
        Tensor &output_tensor = *out;
    }

    std::optional<tensorrt_llm::cutlass_extensions::CutlassGemmConfig> gemm_config;
    bool set_config = tile_config ? true : false;
    if (set_config) {
        gemm_config = getCusConfig(tile_config.value(), split_k_style.value(), split_k_factor.value(), stages.value());
    }

    switch (_st) {
        case at::ScalarType::Half: {
            if (quant_type == torch::kInt8) {
                output_tensor =
                    fused_gemm_dq_helper<half, uint8_t>(output_tensor, input_activations, weight, weight_scales, bias, weight_zero_points, group_size, alpha, quant_op,gemm_config);
            }
            else if (quant_type == at::ScalarType::QUInt4x2) {
                output_tensor = fused_gemm_dq_helper<half, cutlass::uint4b_t>(
                    output_tensor, input_activations, weight, weight_scales, bias, weight_zero_points, group_size, alpha, quant_op,gemm_config);
            }
            else {
                std::string err_msg = "Unsupported weight type " + std::string(at::toString(quant_type));
                throw std::runtime_error(err_msg);
            }
            break;
        }
#ifdef ENABLE_BF16
        case at::ScalarType::BFloat16: {
            if (quant_type == torch::kInt8) {
                output_tensor = fused_gemm_dq_helper<__nv_bfloat16, uint8_t>(
                    output_tensor, input_activations, weight, weight_scales, bias, weight_zero_points, group_size, alpha, quant_op,gemm_config);
            }
            else if (quant_type == at::ScalarType::QUInt4x2) {
                output_tensor = fused_gemm_dq_helper<__nv_bfloat16, cutlass::uint4b_t>(
                    output_tensor, input_activations, weight, weight_scales, bias, weight_zero_points, group_size, alpha, quant_op,gemm_config);
            }
            else {
                std::string err_msg = "Unsupported weight type " + std::string(at::toString(quant_type));
                throw std::runtime_error(err_msg);
            }
            break;
        }
#endif
        default:
            throw std::runtime_error("Unsupported tensor type. Got " + std::string(at::toString(_st)));
    }
    return output_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fpAIntB_gemm", &fpAIntB_gemm, "fpAIntB_gemm");
    }
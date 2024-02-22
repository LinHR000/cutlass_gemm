#include <torch/extension.h>
#include <c10/util/Optional.h>
#include <ATen/cuda/CUDAContext.h>
#include "torch/csrc/cuda/Stream.h"
#include <torch/custom_class.h>
#include <torch/script.h>
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "tensorrt_llm/cutlass_extensions/include/cutlass_extensions/gemm_configs.h"
#include "tensorrt_llm/thop/thUtils.h"
#include <optional>
#include "cutlass/numeric_types.h"
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
//const void* A, const void* B, const void* weight_scales, void* C, int m, int n, int k,
//        tkc::CutlassGemmConfig gemmConfig, char* workspace_ptr, const size_t workspace_bytes,
//        cudaStream_t stream
template<typename T, typename WeightType>
Tensor fpAIntB_gemm_helper(Tensor&         input,
                    Tensor&         weight,
                    c10::optional<torch::Tensor>&            out, // if out is None, allocate a new tensor
                    c10::optional<torch::Tensor>&            bias,
                    c10::optional<torch::Tensor>&            weight_scales,
                    c10::optional<torch::Tensor>&            weight_zero_points,
                    std::optional<float>                     alpha,
                    std::optional<int64_t>                   group_size,
                    cutlass::WeightOnlyQuantOp               quant_op,
                    int             m,
                    int             n,
                    int             k,
                    std::optional<std::string>               tile_config, // if tie_config is -1, use default tile config
                    std::optional<std::string>               split_k_style,
                    int                                      split_k_factor,
                    int                                      stages

){
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    bool allocate_out = out ? false : true;
    at::ScalarType output_data_type = input.scalar_type();
    Tensor output;
    if (allocate_out){
        if (input.dim() == 2) {
            output = torch::zeros({input.size(0), n},
                                torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
        } else if (input.dim() == 3) {
            output = torch::empty({input.size(0), input.size(1), n},
                                torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
        } else {
            throw std::runtime_error("Invalid rank for activations");
        }
    }else{
        Tensor &output = *out;
    }
    
    bool use_alpha = alpha ? true:false;
    T* input_ptr = get_ptr<T>(input);
    T* bias_ptr = bias ? reinterpret_cast<T*>(bias.value().data_ptr()) : nullptr;
    T* weight_scales_ptr = weight_scales ? reinterpret_cast<T*>(weight_scales.value().data_ptr()) : nullptr;
    T* weight_zero_points_ptr = weight_zero_points ? reinterpret_cast<T*>(weight_zero_points.value().data_ptr()) : nullptr;
    T* output_ptr = allocate_out ? get_ptr<T>(output):  reinterpret_cast<T*>(out.value().data_ptr()) ;

    std::optional<tensorrt_llm::cutlass_extensions::CutlassGemmConfig> gemm_config;
    //get gemm config
    bool set_config = tile_config ? true : false;
    if (set_config) {
        gemm_config = getCusConfig(tile_config.value(), split_k_style.value(), split_k_factor, stages);
    }

    char* d_workspace;

    // run the moe gemm
    if (quant_op == cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY) {
        tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<T, WeightType, cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY> cutlass_runner;
         // get workspace
        size_t workspace_bytes = cutlass_runner.getWorkspaceSize(m,n,k);
        cudaMalloc(&d_workspace, workspace_bytes * sizeof(char));
        if (use_alpha){
            cutlass_runner.gemm(input_ptr,
                            get_ptr<WeightType>(weight),
                            weight_scales_ptr,
                            alpha.value(),
                            output_ptr,
                            m,
                            n,
                            k,
                            gemm_config,
                            d_workspace,
                            workspace_bytes,
                            stream);
        }else{
            cutlass_runner.gemm(input_ptr,
                            get_ptr<WeightType>(weight),
                            weight_scales_ptr,
                            output_ptr,
                            m,
                            n,
                            k,
                            gemm_config,
                            d_workspace,
                            workspace_bytes,
                            stream);

        }

    } else if (quant_op == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS) {
        tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<T, WeightType, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS> cutlass_runner;
        size_t workspace_bytes = cutlass_runner.getWorkspaceSize(m,n,k);
        char* d_workspace;
        cudaMalloc(&d_workspace, workspace_bytes * sizeof(char));
        if (use_alpha){
            cutlass_runner.gemm(input_ptr,
                            get_ptr<WeightType>(weight),
                            weight_scales_ptr,
                            weight_zero_points_ptr,
                            bias_ptr,
                            alpha.value(),
                            output_ptr,
                            m,
                            n,
                            k,
                            group_size.value(),
                            gemm_config,
                            d_workspace,
                            workspace_bytes,
                            stream);
        }else{
            cutlass_runner.gemm(input_ptr,
                            get_ptr<WeightType>(weight),
                            weight_scales_ptr,
                            weight_zero_points_ptr,
                            bias_ptr,
                            output_ptr,
                            m,
                            n,
                            k,
                            group_size.value(),
                            gemm_config,
                            d_workspace,
                            workspace_bytes,
                            stream);
        }
    } else if (quant_op == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY){
        tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<T, WeightType, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> cutlass_runner;
        size_t workspace_bytes = cutlass_runner.getWorkspaceSize(m,n,k);
        char* d_workspace;
        cudaMalloc(&d_workspace, workspace_bytes * sizeof(char));
        if (use_alpha){
            cutlass_runner.gemm(input_ptr,
                            get_ptr<WeightType>(weight),
                            weight_scales_ptr,
                            weight_zero_points_ptr,
                            bias_ptr,
                            alpha.value(),
                            output_ptr,
                            m,
                            n,
                            k,
                            group_size.value(),
                            gemm_config,
                            d_workspace,
                            workspace_bytes,
                            stream);
        }else{
            cutlass_runner.gemm(input_ptr,
                            get_ptr<WeightType>(weight),
                            weight_scales_ptr,
                            weight_zero_points_ptr,
                            bias_ptr,
                            output_ptr,
                            m,
                            n,
                            k,
                            group_size.value(),
                            gemm_config,
                            d_workspace,
                            workspace_bytes,
                            stream);
        }
        
    }
    else {
        throw std::runtime_error("unsupported quantization type");
    }
    cudaFree(d_workspace);
    return output;
}

Tensor fpAIntB_gemm(Tensor&         input,
                    Tensor&         weight,
                    c10::optional<torch::Tensor>&            out, // if out is None, allocate a new tensor
                    c10::optional<torch::Tensor>&            bias,
                    c10::optional<torch::Tensor>&            weight_scales,
                    c10::optional<torch::Tensor>&            weight_zero_points,
                    std::optional<float>                     alpha,
                    std::optional<int64_t>                   group_size,
                    std::string     quant_type,
                    int             m,
                    int             n,
                    int             k,
                    std::optional<std::string>               tile_config, // if tie_config is -1, use default tile config
                    std::optional<std::string>               split_k_style,
                    int                                      split_k_factor,
                    int                                      stages

) {
    Tensor output_tensor;
    cutlass::WeightOnlyQuantOp quant_op;
    bool use_weight_scales = weight_scales ? true: false;
    bool use_weight_zero_points = weight_zero_points ? true : false;
    bool use_group = group_size ? true : false;
    if (use_weight_scales){
        if (use_group){
            if (use_weight_zero_points){
                quant_op = cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS;
            }else{
                quant_op = cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY;
            }
        }else{
            quant_op = cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY;
        }
    }else{
        quant_op = cutlass::WeightOnlyQuantOp::UNDEFINED;
    }


    const at::ScalarType _st = input.scalar_type();
    if (_st == at::ScalarType::Half){
        if (quant_type == "auto" || quant_type == "W16A16"){
            output_tensor = fpAIntB_gemm_helper<half, half>(input,
                                                            weight,
                                                            out,
                                                            bias,
                                                            weight_scales,
                                                            weight_zero_points,
                                                            alpha,
                                                            group_size,
                                                            quant_op,
                                                            m,n,k,
                                                            tile_config,
                                                            split_k_style,
                                                            split_k_factor,
                                                            stages);
        }else if (quant_type == "W8A16"){
            output_tensor = fpAIntB_gemm_helper<half, uint8_t>(input,
                                                            weight,
                                                            out,
                                                            bias,
                                                            weight_scales,
                                                            weight_zero_points,
                                                            alpha,
                                                            group_size,
                                                            quant_op,
                                                            m,n,k,
                                                            tile_config,
                                                            split_k_style,
                                                            split_k_factor,
                                                            stages);
            
        }else if (quant_type == "W4A14"){
            output_tensor = fpAIntB_gemm_helper<half, cutlass::uint4b_t>(input,
                                                            weight,
                                                            out,
                                                            bias,
                                                            weight_scales,
                                                            weight_zero_points,
                                                            alpha,
                                                            group_size,
                                                            quant_op,
                                                            m,n,k,
                                                            tile_config,
                                                            split_k_style,
                                                            split_k_factor,
                                                            stages);

        }else{
            std::string err_msg = "Unsupported quant type " + std::string(at::toString(quant_type));
            throw std::runtime_error(err_msg);
        }

    }else if (_st == at::ScalarType::BFloat16){
        if (quant_type == "auto" || quant_type == "W16A16"){
            output_tensor = fpAIntB_gemm_helper<__nv_bfloat16, __nv_bfloat16>(input,
                                                            weight,
                                                            out,
                                                            bias,
                                                            weight_scales,
                                                            weight_zero_points,
                                                            alpha,
                                                            group_size,
                                                            quant_op,
                                                            m,n,k,
                                                            tile_config,
                                                            split_k_style,
                                                            split_k_factor,
                                                            stages);

        }else if (quant_type == "W8A16"){
            output_tensor = fpAIntB_gemm_helper<__nv_bfloat16, uint8_t>(input,
                                                            weight,
                                                            out,
                                                            bias,
                                                            weight_scales,
                                                            weight_zero_points,
                                                            alpha,
                                                            group_size,
                                                            quant_op,
                                                            m,n,k,
                                                            tile_config,
                                                            split_k_style,
                                                            split_k_factor,
                                                            stages);
            
        }else if (quant_type == "W4A14"){
            output_tensor = fpAIntB_gemm_helper<__nv_bfloat16, cutlass::uint4b_t>(input,
                                                            weight,
                                                            out,
                                                            bias,
                                                            weight_scales,
                                                            weight_zero_points,
                                                            alpha,
                                                            group_size,
                                                            quant_op,
                                                            m,n,k,
                                                            tile_config,
                                                            split_k_style,
                                                            split_k_factor,
                                                            stages);

        }else{
            std::string err_msg = "Unsupported quant type " + std::string(at::toString(quant_type));
            throw std::runtime_error(err_msg);
        }
    }else{
        std::string err_msg = "Unsupported weight type " + std::string(at::toString(_st));
        throw std::runtime_error(err_msg);
    }

    return output_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fpAIntB_gemm", &fpAIntB_gemm, "fpAIntB_gemm");
    }
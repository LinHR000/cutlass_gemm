#include <torch/extension.h>
#include <c10/util/Optional.h>
#include <ATen/cuda/CUDAContext.h>
#include "torch/csrc/cuda/Stream.h"
#include <torch/custom_class.h>
#include <torch/script.h>
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "tensorrt_llm/thop/thUtils.h"
using torch::Tensor;
using torch_ext::get_ptr;
using tensorrt_llm;
//const void* A, const void* B, const void* weight_scales, void* C, int m, int n, int k,
//        tkc::CutlassGemmConfig gemmConfig, char* workspace_ptr, const size_t workspace_bytes,
//        cudaStream_t stream
Tensor fpAIntB_gemm(Tensor&         input,
                Tensor&         weight,
                c10::optional<torch::Tensor>&            out, // if out is None, allocate a new tensor
                c10::optional<torch::Tensor>&            bias,
                c10::optional<torch::Tensor>&            weight_scales,
                c10::optional<torch::Tensor>&            weight_zero_points,
                std::optional<float>                     alpha
                int64_t         group_size,
                int64_t         quant_type,
                int             m,
                int             n,
                int             k,
                std::optional<int>                      tile_config, // if tie_config is -1, use default tile config
                std::optional<int                       split_k_style,
                std::optional<int>                      split_k_factor,
                std::optional<int>                      stages

) {
    // if out is None, allocate a new tensor
    WeightOnlyQuantOp quant_op = WeightOnlyQuantOp(quan_type);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    bool allocate_out = out ? false : true;
    at::ScalarType output_data_type = input.scalar_type();
    if (input.dim() == 2) {
        output = torch::zeros({input.size(0), n},
                              torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
    } else if (input.dim() == 3) {
        output = torch::empty({input.size(0), input.size(1), n},
                              torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
    } else {
        throw std::runtime_error("Invalid rank for activations");
    }

    void *input_ptr, *weight_ptr, *weight_scales_ptr, *bias_ptr, *output_ptr, *weight_zero_points_ptr;
    // according input data type, initialize the moe gemm runner and choose the corresponding gemm function
    bool use_bias = bias ? true : false;
    bool use_weight_zero_points = weight_zero_points ? true : false;
    bool use_alpha = alpha ? true : false;

    if (input.scalar_type() == at::ScalarType::Half) {
        input_ptr = get_ptr<half>(input);
        weight_scales_ptr = get_ptr<half>(weight_scales);
        if (use_bias) {
            bias_ptr = get_ptr<half>(bias);
        }
        if (use_weight_zero_points) {
            weight_zero_points_ptr = get_ptr<half>(weight_zero_points);
        }
        output_ptr = get_ptr<half>(output);

        // according weight data type, choose the corresponding gemm function
        if (weight.scalar_type() == at::ScalarType::Char) {
            CutlassFpAIntBGemmRunner <half, uint8_t, quant_op> cutlass_runner;
            weight_ptr = get_ptr<uint8_t>(weight);
        } else if (weight.scalar_type() == at::ScalarType::QInt4x2) {
            CutlassFpAIntBGemmRunner <half, cutlass::uint4b_t, quant_op> cutlass_runner;
            weight_ptr = get_ptr<cutlass::uint4b_t>(weight);
        } else {
            throw std::runtime_error("unsupported data type");
        }
    } else if (input.scalar_type() == at::ScalarType::BFloat16) {
        input_ptr = get_ptr<half>(input);
        weight_scales_ptr = get_ptr<__nv_bfloat16>(weight_scales);
        if (use_bias) {
            bias_ptr = get_ptr<__nv_bfloat16>(bias);
        }
        if (use_weight_zero_points) {
            weight_zero_points_ptr = get_ptr<__nv_bfloat16>(weight_zero_points);
        }
        output_ptr = get_ptr<__nv_bfloat16>(output);

        if (weight.scalar_type() == at::ScalarType::Char) {
            CutlassFpAIntBGemmRunner <__nv_bfloat16, uint8_t, quant_op> cutlass_runner;
            weight_ptr = get_ptr<uint8_t>(weight);
        } else if (weight.scalar_type() == at::ScalarType::QInt4x2) {
            CutlassFpAIntBGemmRunner <__nv_bfloat16, cutlass::uint4b_t, quant_op> cutlass_runner;
            weight_ptr = get_ptr<cutlass::uint4b_t>(weight);
        } else {
            throw std::runtime_error("unsupported data type");
        }
    } else {
        throw std::runtime_error("unsupported data type");
    }
    // get workspace
    size_t workspace_bytes = cutlass_runner.getWorkspaceSize(m,n,k);
    char* d_workspace;
    cudaMalloc(&d_workspace, workspace_bytes * sizeof(char));
    CutlassGemmConfig gemm_config;
    //get gemm config
    bool set_config = tile_config ? true : false;
    if (set_config) {
        gemm_config = CutlassGemmConfig(CutlassTileConfig(tile_config), SplitKStyle(split_k_style), split_k_factor, stages);
        CutlassFpAIntBGemmRunner.setGemmConfig(gemm_config);
    }
    // run the moe gemm
    if (quant_op == WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY) {
        if (use_alpha){
            cutlass_runner.gemm(input_ptr,
                                weight_ptr,
                                weight_scales_ptr,
                                output_ptr,
                                m,
                                n,
                                k,
                                gemm_config,
                                d_workspace,
                                workspace_bytes,
                                stream)
        }else{
            cutlass_runner.gemm(input_ptr,
                                weight_ptr,
                                weight_scales_ptr,
                                alpha,
                                output_ptr,
                                m,
                                n,
                                k,
                                gemm_config,
                                d_workspace,
                                workspace_bytes,
                                stream)
        }

    } else if (quant_op == WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY || quant_op == WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS) {
        if (use_alpha){
            cutlass_runner.gemm(input_ptr,
                                weight_ptr,
                                weight_scales_ptr,
                                weight_zero_points_ptr,
                                bias_ptr,
                                output_ptr,
                                m,
                                n,
                                k,
                                group_size,
                                gemm_config,
                                d_workspace,
                                workspace_bytes,
                                stream)
        }else{
            cutlass_runner.gemm(input_ptr,
                                weight_ptr,
                                weight_scales_ptr,
                                weight_zero_points_ptr,
                                bias_ptr,
                                alpha,
                                output_ptr,
                                m,
                                n,
                                k,
                                group_size,
                                gemm_config,
                                d_workspace,
                                workspace_bytes,
                                stream)
        }
    } else {
        throw std::runtime_error("unsupported quantization type");
    }
    cudaFree(d_workspace);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fpAIntB_gemm", &moe_gemm, "fpA intB gemm");}
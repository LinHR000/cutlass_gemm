#include <torch/extension.h>
#include <c10/util/Optional.h>
#include <ATen/cuda/CUDAContext.h>
#include "torch/csrc/cuda/Stream.h"
#include <torch/custom_class.h>
#include <torch/script.h>
#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels.h"
#include "tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.h"
#include "tensorrt_llm/cutlass_extensions/include/cutlass_extensions/gemm_configs.h"
#include "tensorrt_llm/thop/thUtils.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include <map>
#include <optional>
using torch::Tensor;
using torch_ext::get_ptr;
// using tensorrt_llm;

tensorrt_llm::ActivationType getActivationType(std::string activation_type_str)
{
    if (activation_type_str == "Gelu" || activation_type_str == "gelu") {
        return tensorrt_llm::ActivationType::Gelu;
    }
    else if (activation_type_str == "Relu" || activation_type_str == "relu") {
        return tensorrt_llm::ActivationType::Relu;
    }
    else if (activation_type_str == "Silu" || activation_type_str == "silu") {
        return tensorrt_llm::ActivationType::Silu;
    }
    else if (activation_type_str == "GeGLU" || activation_type_str == "geglu" || activation_type_str == "gated-gelu") {
        return tensorrt_llm::ActivationType::Geglu;
    }
    else if (activation_type_str == "Swiglu") {
        return tensorrt_llm::ActivationType::Swiglu;
    }
    else {
        std::cout << "Activation Type: " <<  activation_type_str << " not supported !";
    }
    return tensorrt_llm::ActivationType::InvalidType;
}

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
                std::string     activation_type,
                std::optional<std::string>             tile_config, // if tie_config is -1, use default tile config
                std::optional<std::string>             split_k_style,
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
    bool use_config = tile_config ? true : false;
    tensorrt_llm::cutlass_extensions::CutlassGemmConfig gemm_config;
    //get gemm config
    if (use_config) {
        // std::string 
        gemm_config = getCusConfig(tile_config.value(),split_k_style.value(),split_k_factor,stages);
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
            cutlass_runner.setBestConfig(gemm_config);
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
                                            getActivationType(activation_type),
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
            cutlass_runner.setBestConfig(gemm_config);
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
                                            getActivationType(activation_type),
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
            cutlass_runner.setBestConfig(gemm_config);
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
                                            getActivationType(activation_type),
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
            cutlass_runner.setBestConfig(gemm_config);
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
                                            getActivationType(activation_type),
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
            cutlass_runner.setBestConfig(gemm_config);
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
                                            getActivationType(activation_type),
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
            cutlass_runner.setBestConfig(gemm_config);
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
                                            getActivationType(activation_type),
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


template<typename T, typename WeightType>
Tensor run_moe_fc_helper(Tensor                            input_activations, //(num_tokens, hidden_size)
                         Tensor                            gating_output, //(num_tokens, num_experts)
                         Tensor                            fc1_expert_weights, //(num_experts, hidden_size, inter_size)
                         tensorrt_llm::ActivationType fc1_activation_type,
                         Tensor                            fc2_expert_weights, //(num_experts, inter_size, hidden_size)
                         c10::optional<Tensor>             fc1_expert_biases,
                         c10::optional<Tensor>             fc2_expert_biases,
                         c10::optional<Tensor>             fc1_expert_scales,
                         c10::optional<Tensor>             fc2_expert_scales,
                         c10::optional<Tensor>             output,
                         const int                         active_rows,
                         const int                         k,
                         std::optional<std::string>        tile_config, // if tie_config is -1, use default tile config
                         std::optional<std::string>        split_k_style,
                         std::optional<int>                split_k_factor,
                         std::optional<int>                stages)
{

    const int num_rows    = input_activations.size(0); //(num_tokens, hidden_size)
    const int hidden_size = input_activations.size(1);
    const int inter_size  = fc2_expert_weights.size(1); //(num_experts, inter_size, hidden_size)
    const int num_experts = gating_output.size(-1); //(num_tokens, num_experts)
    auto      stream      = at::cuda::getCurrentCUDAStream().stream();

    T* input_act_ptr     = get_ptr<T>(input_activations);
    T* gating_output_ptr = get_ptr<T>(gating_output);

    WeightType*           fc1_expert_weights_ptr = get_ptr<WeightType>(fc1_expert_weights);
    static constexpr bool is_fp16_or_fp32 =
        std::is_same<WeightType, float>::value || std::is_same<WeightType, half>::value;
#ifdef ENABLE_BF16
    static constexpr bool ignore_scales = is_fp16_or_fp32 || std::is_same<WeightType, __nv_bfloat16>::value;
#else
    static constexpr bool ignore_scales = is_fp16_or_fp32;
#endif

    T* fc1_scales_ptr        = ignore_scales ? nullptr : reinterpret_cast<T*>(fc1_expert_scales.value().data_ptr());
    T* fc1_expert_biases_ptr = fc1_expert_biases ? reinterpret_cast<T*>(fc1_expert_biases.value().data_ptr()) : nullptr; //如果需要bias可以添加

    WeightType* fc2_expert_weights_ptr = get_ptr<WeightType>(fc2_expert_weights);
    T*          fc2_scales_ptr         = ignore_scales ? nullptr : reinterpret_cast<T*>(fc2_expert_scales.value().data_ptr());
    T*          fc2_expert_biases_ptr  = fc2_expert_biases ? reinterpret_cast<T*>(fc2_expert_biases.value().data_ptr()) : nullptr;;

    // bool* finished_ptr   = get_ptr<bool>(finished);
    bool* finished_ptr = nullptr;

    tensorrt_llm::kernels::MOEParallelismConfig moe_parallel_config = tensorrt_llm::kernels::MOEParallelismConfig::TensorParallelism(1, 0);
    tensorrt_llm::kernels::CutlassMoeFCRunner<T, WeightType> moe_runner;

    // tile config有三种：CtaShape32x128x64_WarpShape32x32x64， CtaShape64x128x64_WarpShape32x64x64， CtaShape128x128x64_WarpShape64x32x64:
    // for fp16/bf16
    // tensorrt_llm::cutlass_extensions::CutlassGemmConfig gemm_config;
    // gemm_config.tile_config = tensorrt_llm::cutlass_extensions::CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64;
    // gemm_config.stages = 4;
    // for int8 and int4

    // tensorrt_llm::cutlass_extensions::CutlassGemmConfig gemm_config;
    // gemm_config.tile_config = tensorrt_llm::cutlass_extensions::CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64;
    // gemm_config.stages = 4;
    if (tile_config){
        tensorrt_llm::cutlass_extensions::CutlassGemmConfig gemm_config;
        gemm_config = getCusConfig(tile_config.value(),split_k_style.value(),split_k_factor.value(),stages.value());
        // gemm_config = tensorrt_llm::cutlass_extensions::CutlassGemmConfig(tensorrt_llm::cutlass_extensions::CutlassTileConfig(tile_config), 
        //                                                                   tensorrt_llm::cutlass_extensions::SplitKStyle(split_k_style), 
        //                                                                   split_k_factor, stages);
        moe_runner.setTactic(gemm_config);
    }

    long int bytes        = moe_runner.getWorkspaceSize(num_rows, hidden_size, inter_size, num_experts, k, fc1_activation_type, moe_parallel_config);

    auto workspace_tensor = torch::empty({bytes * 4}, torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));
    char* workspace_ptr   = get_ptr<char>(workspace_tensor);

    const at::ScalarType _st = input_activations.scalar_type();
    auto                 fc2_output =
        torch::empty({k * num_rows, hidden_size}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    T* fc2_output_ptr = get_ptr<T>(fc2_output);

    auto expert_scales     = torch::empty({num_rows, k}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    T*   expert_scales_ptr = get_ptr<T>(expert_scales);

    auto expanded_source_row_to_expanded_dest_row =
        torch::empty({num_rows, k}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    int* expanded_source_row_to_expanded_dest_row_ptr = get_ptr<int>(expanded_source_row_to_expanded_dest_row);

    auto expert_for_source_row =
        torch::empty({num_rows, k}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    int* expert_for_source_row_ptr = get_ptr<int>(expert_for_source_row);
    bool allocate_out  = output ? false : true;
    Tensor output_tensor ;
    if (allocate_out){
        output_tensor = torch::empty({num_rows, hidden_size}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    }else{
        Tensor &ououtput_tensortput = *output;
    }

    T* output_tensor_ptr = get_ptr<T>(output_tensor);

    moe_runner.runMoe(input_act_ptr,
                        gating_output_ptr,
                        fc1_expert_weights_ptr,
                        fc1_scales_ptr, // nullptr
                        fc1_expert_biases_ptr, // nullptr
                        fc1_activation_type,
                        fc2_expert_weights_ptr,
                        fc2_scales_ptr, // nullptr
                        fc2_expert_biases_ptr, // nullptr
                        num_rows,
                        hidden_size,
                        inter_size,
                        num_experts,
                        k,
                        workspace_ptr,
                        output_tensor_ptr,
                        fc2_output_ptr,
                        finished_ptr, // nullptr
                        active_rows, // original num_rows
                        expert_scales_ptr,
                        expanded_source_row_to_expanded_dest_row_ptr,
                        expert_for_source_row_ptr,
                        moe_parallel_config,
                        tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::NONE,
                        stream);

    return output_tensor;
}


Tensor run_moe_fc(Tensor      input_activations, //(num_tokens, hidden_size)
                  Tensor      gating_output, //(num_tokens, num_experts)
                  Tensor      fc1_expert_weights, //(num_experts, hidden_size, inter_size)
                  std::string fc1_activation_type_str,
                  Tensor      fc2_expert_weights, //(num_experts, inter_size, hidden_size)
                  c10::optional<Tensor>             fc1_expert_biases,
                  c10::optional<Tensor>             fc2_expert_biases,
                  c10::optional<Tensor>             fc1_expert_scales,
                  c10::optional<Tensor>             fc2_expert_scales,
                  c10::optional<Tensor>             output,
                  int64_t     active_rows,
                  int64_t     k,
                  std::optional<std::string>        tile_config, // if tie_config is -1, use default tile config
                  std::optional<std::string>        split_k_style,
                  std::optional<int>                split_k_factor,
                  std::optional<int>                stages)
{

    const at::ScalarType _st = input_activations.scalar_type();

    const int num_rows    = input_activations.size(0);
    const int hidden_size = input_activations.size(1);
    const int inter_size  = fc2_expert_weights.size(1);
    const int num_experts = gating_output.size(-1);

    // torch::ScalarType quant_type = fc2_expert_weights.scalar_type();

    CHECK_INPUT(input_activations, _st);
    TORCH_CHECK(input_activations.dim() == 2, "Invalid rank for activations");

    CHECK_INPUT(gating_output, _st);
    TORCH_CHECK(gating_output.dim() == 2, "Invalid rank for gating output");
    TORCH_CHECK(gating_output.size(0) == num_rows, "gating output and activations must have same number of rows");

    CHECK_TH_CUDA(fc1_expert_weights);
    CHECK_CONTIGUOUS(fc1_expert_weights);
    TORCH_CHECK(fc1_expert_weights.dim() == 3, "Invalid rank for fc1 weights");
    TORCH_CHECK(fc1_expert_weights.size(0) == num_experts, "Experts mismatch between gate outputs and fc1 weights");
    TORCH_CHECK(fc1_expert_weights.size(1) == hidden_size,
                "Activation last dim must equal size of dim 1 for fc1 weight");

    const int fc1_num_cols = fc1_expert_weights.size(-1);

    
    CHECK_TH_CUDA(fc2_expert_weights);
    CHECK_CONTIGUOUS(fc2_expert_weights);
    TORCH_CHECK(fc2_expert_weights.dim() == 3, "Invalid rank for fc2 weights");
    TORCH_CHECK(fc2_expert_weights.size(0) == gating_output.size(-1),
                "Experts mismatch between gate outputs and fc2 weights");
    // TORCH_CHECK(fc2_expert_weights.size(1) == fc1_num_cols, "fc1 weight last dim must equal dim 1 of fc2 weights"); 如果是 glu 类，该条件无法满足

    Tensor output_tensor;

    tensorrt_llm::ActivationType fc1_activation_type = tensorrt_llm::ActivationType::InvalidType;
    if (fc1_activation_type_str == "identity") {
        fc1_activation_type = tensorrt_llm::ActivationType::Identity;
    }
    else {
        fc1_activation_type = getActivationType(fc1_activation_type_str);
    }
    torch::ScalarType quant_type = fc1_expert_weights.scalar_type();
    if (fc1_expert_scales){
        if (fc1_expert_weights.size(-1) == fc1_expert_scales.value().size(-1) / 2) {
                quant_type = at::ScalarType::QUInt4x2;
            }
    }
    
    switch (_st) {
        case at::ScalarType::Float: {

            if (quant_type == _st) {
                output_tensor = run_moe_fc_helper<float, float>(input_activations,
                                                                gating_output,
                                                                fc1_expert_weights,
                                                                fc1_activation_type,
                                                                fc2_expert_weights,
                                                                fc1_expert_biases,
                                                                fc2_expert_biases,
                                                                fc1_expert_scales,
                                                                fc2_expert_scales,
                                                                output,
                                                                active_rows,
                                                                k,
                                                                tile_config,
                                                                split_k_style,
                                                                split_k_factor,
                                                                stages);
            }
            else {
                std::string err_msg = "Unsupported weight type " + std::string(at::toString(quant_type));
                throw std::runtime_error(err_msg);
            }
            break;
        }
        case at::ScalarType::Half: {

            if (quant_type == _st) {
                output_tensor = run_moe_fc_helper<half, half>(input_activations,
                                                              gating_output,
                                                              fc1_expert_weights,
                                                              fc1_activation_type,
                                                              fc2_expert_weights,
                                                              fc1_expert_biases,
                                                              fc2_expert_biases,
                                                              fc1_expert_scales,
                                                              fc2_expert_scales,
                                                              output,
                                                              active_rows,
                                                              k,
                                                              tile_config,
                                                                split_k_style,
                                                                split_k_factor,
                                                                stages);
            }else if (quant_type == torch::kInt8){
                output_tensor = run_moe_fc_helper<half, uint8_t>(input_activations,
                                                              gating_output,
                                                              fc1_expert_weights,
                                                              fc1_activation_type,
                                                              fc2_expert_weights,
                                                              fc1_expert_biases,
                                                              fc2_expert_biases,
                                                              fc1_expert_scales,
                                                              fc2_expert_scales,
                                                              output,
                                                              active_rows,
                                                              k,
                                                              tile_config,
                                                                split_k_style,
                                                                split_k_factor,
                                                                stages);
            }else if (quant_type == at::ScalarType::QUInt4x2){
                output_tensor = run_moe_fc_helper<half, cutlass::uint4b_t>(input_activations,
                                                              gating_output,
                                                              fc1_expert_weights,
                                                              fc1_activation_type,
                                                              fc2_expert_weights,
                                                              fc1_expert_biases,
                                                              fc2_expert_biases,
                                                              fc1_expert_scales,
                                                              fc2_expert_scales,
                                                              output,
                                                              active_rows,
                                                              k,
                                                              tile_config,
                                                                split_k_style,
                                                                split_k_factor,
                                                                stages);
            }
            else {
                std::string err_msg = "Unsupported weight type " + std::string(at::toString(quant_type));
                throw std::runtime_error(err_msg);
            }
            break;
        }
#ifdef ENABLE_BF16
        case at::ScalarType::BFloat16: {
            if (quant_type == _st) {
                output_tensor = run_moe_fc_helper<__nv_bfloat16, __nv_bfloat16>(input_activations,
                                                                                gating_output,
                                                                                fc1_expert_weights,
                                                                                fc1_activation_type,
                                                                                fc2_expert_weights,
                                                                                fc1_expert_biases,
                                                                                fc2_expert_biases,
                                                                                fc1_expert_scales,
                                                                                fc2_expert_scales,
                                                                                output,
                                                                                active_rows,
                                                                                k,
                                                                                tile_config,
                                                                split_k_style,
                                                                split_k_factor,
                                                                stages);
            }else if (quant_type == torch::kInt8){
                output_tensor = run_moe_fc_helper<__nv_bfloat16, uint8_t>(input_activations,
                                                              gating_output,
                                                              fc1_expert_weights,
                                                              fc1_activation_type,
                                                              fc2_expert_weights,
                                                              fc1_expert_biases,
                                                              fc2_expert_biases,
                                                              fc1_expert_scales,
                                                              fc2_expert_scales,
                                                              output,
                                                              active_rows,
                                                              k,
                                                              tile_config,
                                                                split_k_style,
                                                                split_k_factor,
                                                                stages);
            }else if (quant_type == at::ScalarType::QUInt4x2){
                output_tensor = run_moe_fc_helper<__nv_bfloat16, cutlass::uint4b_t>(input_activations,
                                                              gating_output,
                                                              fc1_expert_weights,
                                                              fc1_activation_type,
                                                              fc2_expert_weights,
                                                              fc1_expert_biases,
                                                              fc2_expert_biases,
                                                              fc1_expert_scales,
                                                              fc2_expert_scales,
                                                              output,
                                                              active_rows,
                                                              k,
                                                              tile_config,
                                                                split_k_style,
                                                                split_k_factor,
                                                                stages);
            }
            else {
                std::string err_msg = "Unsupported weight type " + std::string(at::toString(quant_type));
                throw std::runtime_error(err_msg);
            }
            break;
        }
#endif
        default:
            throw std::runtime_error("Wrong Tensor type.");
    }
    return output_tensor;
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moe_gemm", &moe_gemm, "moe gemm");
    m.def("run_moe_fc", &run_moe_fc, "moe fc");
    }
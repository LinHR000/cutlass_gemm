#include <torch/extension.h>
#include <c10/util/Optional.h>
#include "torch/csrc/cuda/Stream.h"
#include <torch/custom_class.h>
#include <torch/script.h>
#include "cutlass_kernels/cutlass_preprocessors.h"
#include "utils/th_utils.h"
using torch::Tensor;
using torch_ext::get_ptr;

namespace ft = fastertransformer;
enum class QuantType {
    INT8_WEIGHT_ONLY,
    PACKED_INT4_WEIGHT_ONLY
};
// using fastertransformer;
Tensor gemm_in8_w8_ofp16_pt(Tensor output, //int8 * int8 -> fp16 per token 量化
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
                            int     stages);
Tensor gemm_in8_w8_ofp16_pc(Tensor output,
                            Tensor input, // int8 * int8 -> fp16 per channel 量化
                            Tensor weight,
                            Tensor alpha_col, 
                            Tensor alpha_row,
                            int64_t m,
                            int64_t n,
                            int64_t k,
                            int     tile_config,
                            int     split_k_style,
                            int     split_k_factor,
                            int     stages);
Tensor gemm_in8_w8_ofp16_ptpc(Tensor output,
                            Tensor input,// int8 * int8 -> fp16 per token per channel 量化
                            Tensor weight,
                            Tensor alpha_col, 
                            Tensor alpha_row,
                            int64_t m,
                            int64_t n,
                            int64_t k,
                            int     tile_config,
                            int     split_k_style,
                            int     split_k_factor,
                            int     stages);
Tensor gemm_infp16_w8_ofp16(Tensor output_tensor,
                            Tensor input_activations, // int8 * fp16 -> fp16 weight only 量化
                            Tensor weight, 
                            Tensor scales,
                            int               tile_config,
                            int               split_k_style,
                            int               split_k_factor,
                            int               stages);
Tensor gemm_infp16_w8_ofp16_bias_act(Tensor output_tensor,
                            Tensor input_activations, 
                            Tensor weight,
                             Tensor scales, 
                             Tensor bias, 
                             std::string activation_type_str,
                             int               tile_config,
                            int               split_k_style,
                            int               split_k_factor,
                            int               stages);
std::vector<Tensor> _symmetric_quantize_last_axis_of_batched_matrix(Tensor weight,
                                                                    int quant_mode);
                                                                    
std::vector<int> choose_best_config_half(Tensor input_activations, Tensor weight, Tensor scales);    

std::vector<int> gemm_in8_w8_ofp16_config(Tensor output,
                            Tensor input,
                            Tensor weight,
                            Tensor alpha_col, 
                            Tensor alpha_row,
                            int64_t m,
                            int64_t n,
                            int64_t k);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "gemm_in8_w8_ofp16_pt",
    &gemm_in8_w8_ofp16_pt,
    "Compute the attention between an input query and the cached key/value tensors");
m.def(
    "gemm_in8_w8_ofp16_pc",
    &gemm_in8_w8_ofp16_pc,
    "Compute the attention between an input query and the cached key/value tensors");
m.def(
    "gemm_in8_w8_ofp16_ptpc",
    &gemm_in8_w8_ofp16_ptpc,
    "Compute the attention between an input query and the cached key/value tensors");
m.def(
    "gemm_infp16_w8_ofp16",
    &gemm_infp16_w8_ofp16,
    "Compute the attention between an input query and the cached key/value tensors");
m.def(
    "gemm_infp16_w8_ofp16_bias_act",
    &gemm_infp16_w8_ofp16_bias_act,
    "Compute the attention between an input query and the cached key/value tensors");
m.def(
    "preprocess_weights_for_mixed_gemm",
    &fastertransformer::preprocess_weights_for_mixed_gemm,
    "Compute the attention between an input query and the cached key/value tensors");
m.def(
    "symmetric_quantize_last_axis_of_batched_matrix",
    &_symmetric_quantize_last_axis_of_batched_matrix,
    "Compute the attention between an input query and the cached key/value tensors");
m.def(
    "choose_best_config_half",
    &choose_best_config_half,
    "Compute the attention between an input query and the cached key/value tensors");
m.def(
    "gemm_in8_w8_ofp16_config",
    &gemm_in8_w8_ofp16_config,
    "Compute the attention between an input query and the cached key/value tensors");
}
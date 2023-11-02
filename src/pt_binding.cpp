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
Tensor gemm_in8_w8_ofp16_pt(Tensor input, //int8 * int8 -> fp16 per token 量化
                            Tensor weight,
                            Tensor alpha_col, 
                            Tensor alpha_row,
                            int64_t m,
                            int64_t n,
                            int64_t k);
Tensor gemm_in8_w8_ofp16_pc(Tensor input, // int8 * int8 -> fp16 per channel 量化
                            Tensor weight,
                            Tensor alpha_col, 
                            Tensor alpha_row,
                            int64_t m,
                            int64_t n,
                            int64_t k);
Tensor gemm_in8_w8_ofp16_ptpc(Tensor input,// int8 * int8 -> fp16 per token per channel 量化
                            Tensor weight,
                            Tensor alpha_col, 
                            Tensor alpha_row,
                            int64_t m,
                            int64_t n,
                            int64_t k);
Tensor gemm_infp16_w8_ofp16(Tensor input_activations, // int8 * fp16 -> fp16 weight only 量化
                            Tensor weight, 
                            Tensor scales);
Tensor gemm_infp16_w8_ofp16_bias_act(
                            Tensor input_activations, 
                            Tensor weight,
                             Tensor scales, 
                             Tensor bias, 
                             std::string activation_type_str);
std::vector<Tensor> _symmetric_quantize_last_axis_of_batched_matrix(Tensor weight,
                                                                    int quant_mode);
                                                   

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
}
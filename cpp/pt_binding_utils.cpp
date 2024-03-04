#include <torch/extension.h>
#include <c10/util/Optional.h>
#include "torch/csrc/cuda/Stream.h"
#include <torch/custom_class.h>
#include <torch/script.h>
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.h"
#include "tensorrt_llm/thop/weightOnlyQuantOp.h"
#include "tensorrt_llm/thop/thUtils.h"
using torch::Tensor;
using torch_ext::get_ptr;

// std::vector<Tensor> _symmetric_quantize_last_axis_of_batched_matrix(Tensor weight,
//                                                                     int quant_mode);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def(
    "symmetric_quantize_last_axis_of_batched_matrix",
    &torch_ext::_symmetric_quantize_last_axis_of_batched_matrix,
    "Compute the attention between an input query and the cached key/value tensors");
m.def(
    "preprocess_weights_for_mixed_gemm",
    &torch_ext::preprocess_weights_for_mixed_gemm_,
    "Compute the attention between an input query and the cached key/value tensors");
m.def(
    "pack_int8_tensor_to_packed_int4",
    &torch_ext::pack_int8_tensor_to_packed_int4,
    "Compute the attention between an input query and the cached key/value tensors");
m.def(
    "unpack_int4_packed_tensor_to_int8",
    &torch_ext::unpack_int4_packed_tensor_to_int8,
    "Compute the attention between an input query and the cached key/value tensors");
m.def(
    "permute_B_rows_for_mixed_gemm",
    &torch_ext::permute_B_rows_for_mixed_gemm,
    "Compute the attention between an input query and the cached key/value tensors");
}
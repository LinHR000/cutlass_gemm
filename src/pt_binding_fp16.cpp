#include <torch/extension.h>
#include <c10/util/Optional.h>
#include "torch/csrc/cuda/Stream.h"
#include <torch/custom_class.h>
#include <torch/script.h>
#include "cutlass_kernels/fp16_gemm/fp16_gemm.h"
#include "utils/th_utils.h"
using torch::Tensor;
using torch_ext::get_ptr;

namespace ft = fastertransformer;
Tensor gemm_fp16_ofp16(Tensor         input,
                        Tensor            weight,
                        float             alpha, 
                        float             beta,
                        int64_t           m,
                        int64_t           n,
                        int64_t           k,
                        std::string       tile_config,
                        const int               stages,
                        const int               splitK){
    at::ScalarType output_data_type = at::ScalarType::Half;
    Tensor output = torch::zeros({m, n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int sm = 80;
    ft::cutlass_fp16_ofp16_gemm(get_ptr<__half>(input),
                                        get_ptr<__half>(weight),
                                        alpha,
                                        beta,
                                        get_ptr<__half>(output),
                                        m,
                                        n,
                                        k,
                                        tile_config,
                                        stages,
                                        splitK,
                                        nullptr,
                                        0,
                                        sm,
                                        stream);
    return output;
}           


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def(
    "gemm_fp16_ofp16",
    &gemm_fp16_ofp16,
    "Compute the attention between an input query and the cached key/value tensors");
}

#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include "torch/csrc/cuda/Stream.h"
#include <torch/custom_class.h>
#include <torch/script.h>
#include "tensorrt_llm/thop/thUtils.h"
#include "cutlass/numeric_types.h"
#include <cstdlib>
#include <optional>
#include "tensorrt_llm/kernels/fp8_cublaslt/cublasLt_fp8Matmul.h"


using torch::Tensor;
using torch_ext::get_ptr;


torch::Tensor get_workspace() {
    // Static workspace tensor
    static torch::Tensor cublas_workspace;

    // If workspace is not allocated, allocate it
    uint32_t size_w = 4194304; //4MB  33554432 for sm90
    cublas_workspace = torch::empty(size_w, torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));

    return cublas_workspace;
}

Tensor fp8_gemm(
    Tensor&                                 input_activations, 
    Tensor&                                 weight, 
    const float                             alpha,
    const float                             a_scale,
    const float                             b_scale,
    const float                             c_scale,
    const float                             d_scale,
    float                                   amax_d
    )
{   
    // TN 模式下应该采用w/a 输m_split入模式
    const at::ScalarType _st_out = at::ScalarType::Half;
    // const at::ScalarType _st_out = input_activations.scalar_type();
    TORCH_CHECK(weight.dim() == 2, "Invalid rank for weight");
    int m;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
    Tensor output_tensor;
    if (input_activations.dim() == 3){
        m = input_activations.size(0) * input_activations.size(1);
        k = input_activations.size(2);
        n = weight.size(0);
        lda = k;
        ldb = k;
        ldc = n;
        output_tensor = torch::zeros({input_activations.size(0),input_activations.size(1), n}, torch::dtype(_st_out).device(torch::kCUDA).requires_grad(false));

    }else if(input_activations.dim() == 2){
        m = input_activations.size(0);
        k = input_activations.size(1);
        n = weight.size(0);
        lda = k;
        ldb = k;
        ldc = n;
        output_tensor = torch::zeros({input_activations.size(0), n}, torch::dtype(_st_out).device(torch::kCUDA).requires_grad(false));

    }else{
        throw std::runtime_error("Invalid rank for activations, support 2 or 3");
    }
    size_t workspaceSize = 4194304;
    void *workspace;
    cudaMalloc(&workspace, workspaceSize);

    float* d_a_scale;
    cudaMalloc(&d_a_scale, sizeof(float));
    cudaMemcpy(d_a_scale, &a_scale, sizeof(float), cudaMemcpyHostToDevice);

    float* d_b_scale;
    cudaMalloc(&d_b_scale, sizeof(float));
    cudaMemcpy(d_b_scale, &b_scale, sizeof(float), cudaMemcpyHostToDevice);
    float* d_c_scale;
    cudaMalloc(&d_c_scale, sizeof(float));
    cudaMemcpy(d_c_scale, &c_scale, sizeof(float), cudaMemcpyHostToDevice);
    float* d_d_scale;
    cudaMalloc(&d_d_scale, sizeof(float));
    cudaMemcpy(d_d_scale, &d_scale, sizeof(float), cudaMemcpyHostToDevice);
    float* d_amax_d_scale;
    cudaMalloc(&d_amax_d_scale, sizeof(float));
    cudaMemcpy(d_amax_d_scale, &amax_d, sizeof(float), cudaMemcpyHostToDevice);


    LtFp8Matmul(n, m, k, // m,n,k
                &alpha, /* host pointer */
                d_b_scale, /* device pointer */
                get_ptr<const __nv_fp8_e4m3>(weight),
                ldb,
                d_a_scale, /* device pointer */
                get_ptr<const __nv_fp8_e4m3>(input_activations),
                lda,
                d_c_scale, /* device pointer */
                get_ptr<__half>(output_tensor),
                ldc,
                d_d_scale, /* device pointer */
                d_amax_d_scale, /* device pointer */
                workspace, //workspace
                workspaceSize); //size_t workspace
                // nullptr, //workspace
                // 0); //size_t workspace
    cudaFree(workspace);
    cudaFree(d_a_scale);
    cudaFree(d_b_scale);
    cudaFree(d_c_scale);
    cudaFree(d_d_scale);
    cudaFree(d_amax_d_scale);
    return output_tensor;
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp8_gemm", &fp8_gemm, "fp8_gemm");
    }
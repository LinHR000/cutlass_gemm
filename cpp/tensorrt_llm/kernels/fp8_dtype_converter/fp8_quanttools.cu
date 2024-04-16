#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"
#include "quant_utils.cuh"

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>
#include <string.h>
#ifdef USE_ROCM
  #include <hip/hip_bf16.h>
  typedef __hip_bfloat16 __nv_bfloat16;
#endif
#include "fp8_quanttools.h"


constexpr int FP8_E4M3 = 0;
constexpr int FP8_E5M2 = 1;

template<typename Tout>
__global__ void fake_quant_fp8_e5m2_kernel(
  const Tout* __restrict__ src_cache,
  Tout* __restrict__ dst_cache,
  int dtype,
  const int64_t block_stride) {
  const int64_t block_idx = blockIdx.x;
  if (dtype == 0) {
    for (int i = threadIdx.x; i < block_stride; i += blockDim.x) {
      int64_t idx = block_idx * block_stride + i;
      uint8_t tmp_val = fp8_e5m2_unscaled::vec_conversion<uint8_t,Tout, FP8_E5M2>(src_cache[idx]);
      dst_cache[idx] = fp8_e5m2_unscaled::vec_conversion<Tout, uint8_t, FP8_E5M2>(tmp_val);
    }
  }else if (dtype == 1) {
    for (int i = threadIdx.x; i < block_stride; i += blockDim.x) {
        int64_t idx = block_idx * block_stride + i;
        uint8_t tmp_val = fp8_e5m2_unscaled::vec_conversion<uint8_t,Tout, FP8_E4M3>(src_cache[idx]);
        dst_cache[idx] = fp8_e5m2_unscaled::vec_conversion<Tout, uint8_t, FP8_E4M3>(tmp_val);
    }
  }
  else{
    assert(false);
  }
}




#define CALL_FAKE_QUANT_FP8_E5M2(Tout)                                   \
  fake_quant_fp8_e5m2_kernel<Tout><<<grid, block, 0, stream>>>(    \
    reinterpret_cast<Tout*>(src_cache.data_ptr()),                       \
    reinterpret_cast<Tout*>(dst_cache.data_ptr()),                       \
    dtype,                                                               \
    block_stride);

torch::Tensor fake_quant_fp8(
  torch::Tensor& src_cache,
  torch::Tensor& dst_cache,
  std::string dtype_str)
{
  int64_t num_blocks = src_cache.size(0);
  int64_t block_stride = src_cache.stride(0);
  int dtype = -1;
  if (dtype_str == "fp8_e5m2") {
    dtype = 0;
  }else if (dtype_str == "fp8_e4m3") {
    dtype = 1;
  }else{
    TORCH_CHECK(false, "Unsupported fp8 data type:  ", dtype);
  }


  dim3 grid(num_blocks);
  dim3 block(std::min(block_stride, int64_t(512)));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (src_cache.dtype() == at::ScalarType::Half) {
    CALL_FAKE_QUANT_FP8_E5M2(uint16_t);
  }else{
    TORCH_CHECK(false, "Unsupported data type:  ", src_cache.dtype());
  }
  return dst_cache;
}

template<typename Tout, typename Tin>
__global__ void convert_fp8_e5m2_kernel(
  const Tin* __restrict__ src_cache,
  Tout* __restrict__ dst_cache,
  int dtype,
  const int64_t block_stride) {
  const int64_t block_idx = blockIdx.x;
  if (dtype == 0) {
    for (int i = threadIdx.x; i < block_stride; i += blockDim.x) {
    int64_t idx = block_idx * block_stride + i;
    dst_cache[idx] = fp8_e5m2_unscaled::vec_conversion<Tout, Tin,FP8_E5M2>(src_cache[idx]);

    }
  }else if (dtype == 1) {
    for (int i = threadIdx.x; i < block_stride; i += blockDim.x) {
    int64_t idx = block_idx * block_stride + i;
    dst_cache[idx] = fp8_e5m2_unscaled::vec_conversion<Tout, Tin,FP8_E4M3>(src_cache[idx]);

    }
  }else{
    assert(false);
  }

  
}

#define CALL_CONVERT_FP8_E5M2(Tout, Tin)                                 \
  convert_fp8_e5m2_kernel<Tout, Tin><<<grid, block, 0, stream>>>(  \
    reinterpret_cast<Tin*>(src_cache.data_ptr()),                        \
    reinterpret_cast<Tout*>(dst_cache.data_ptr()),                       \
    dtype,                                                               \
    block_stride);

torch::Tensor convert_fp8_e5m2(
  torch::Tensor& src_cache,
  torch::Tensor& dst_cache,
  std::string dtype_str)
{
  int dtype = -1;
  if (dtype_str == "fp8_e5m2") {
    dtype = 0;
  }else if (dtype_str == "fp8_e4m3") {
    dtype = 1;
  }else{
    TORCH_CHECK(false, "Unsupported fp8 data type:  ", dtype);
  }
  int64_t num_blocks = src_cache.size(0);
  int64_t block_stride = src_cache.stride(0);

  dim3 grid(num_blocks);
  dim3 block(std::min(block_stride, int64_t(512)));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (src_cache.dtype() == at::ScalarType::Half) {
    CALL_CONVERT_FP8_E5M2(uint8_t, uint16_t);
  } else if (dst_cache.dtype() == at::ScalarType::Half) {
    CALL_CONVERT_FP8_E5M2(uint16_t, uint8_t);
  } else{
    TORCH_CHECK(false, "Unsupported data type:  ", src_cache.dtype());
  }
  return dst_cache;
}

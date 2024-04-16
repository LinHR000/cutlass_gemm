#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>
#include <string.h>

torch::Tensor fake_quant_fp8(
  torch::Tensor& src_cache,
  torch::Tensor& dst_cache,
  std::string dtype_str);

torch::Tensor convert_fp8_e5m2(
  torch::Tensor& src_cache,
  torch::Tensor& dst_cache,
  std::string dtype_str);


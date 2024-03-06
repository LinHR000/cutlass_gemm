# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
import numpy as np
import unittest

def random_cuda_tensor(shape, dtype, mean=0, std=1):
    return torch.empty(shape, dtype=dtype, device="cuda").normal_(mean, std)


class TestFpAInt8(unittest.TestCase):

  def setUp(self) -> None:
    from gemm_op import gemm_op_fpAIntB
    # import moe_ops
    self.gemm_func = gemm_op_fpAIntB.fpAIntB_gemm
    torch.manual_seed(734876213)
  
  def generate_inputs(self, m, k, dtype, quant_type):
    inputs = dict()
    inputs["input"] = random_cuda_tensor([m, k], dtype, mean=0, std=0.02)
    return inputs
  
  def generate_weights(self, n, k, dtype, quant_type):
    weights = dict()
    quantize = quant_type == torch.int8 or quant_type == torch.quint4x2
    if quantize:
      from gemm_op import gemm_op_utils

    weights["weights_for_ref"] = random_cuda_tensor([k, n], dtype, mean=0, std=0.02)
    weights["weights_for_ft"] = weights["weights_for_ref"].t().contiguous()
    if quantize:
      x = weights["weights_for_ft"]
      x = x.view(-1,128)
      x_max = x.abs().amax(dim=1, keepdim=True)
      x_scale = x_max / 8
      x_int = (x / x_scale).round().clamp(-8,7)
      x_dequant = x_int * x_scale
      x_dequant = x_dequant.view(weights["weights_for_ft"].shape).t()
      x_scale = x_scale.view(weights["weights_for_ref"].shape[1],-1).contiguous()
      x_int4 = gemm_op_utils.pack_int8_tensor_to_packed_int4(x_int.char().cpu().view(weights["weights_for_ft"].shape).t().contiguous())
      x_int4 = gemm_op_utils.preprocess_weights_for_mixed_gemm(x_int4,1)
      weights["weights_for_ft_scale"] = x_scale.to(weights["weights_for_ft"].device)
      weights["weights_for_ft"] = x_int4.to(weights["weights_for_ft"].device)
      weights["weights_for_ref"] = x_dequant.to(weights["weights_for_ft"].device)
    else:
      weights["weights_for_ft_scale"] = None
    return weights
  
  def run_ft_gemm(self, input_dict, m,n,k,quant_type):
    output = self.gemm_func(input_dict["input"],
                            input_dict["weights_for_ft"], 
                            input_dict["weights_for_ft_scale"],
                            None,# bias
                            None,#out
                            None,#weight_zero_points
                            None,#alpha
                            128,# group_size
                            None,#tile config
                            None,# split_k_style
                            None,# split_k_factor
                            None) # stages
    return output
  
  def run_ref_gemm(self, input_dict):
    output = torch.matmul(input_dict["input"],input_dict['weights_for_ref'])
    return output

  def gemm_test_helper(self, dtype, quant_type, rtol, atol, hidden_sizes=[1024], inter_sizes=[4096]):
    torch.cuda.empty_cache() # Empty the cache here so a bad ordering does not cause OOM.
    rows = [16]

    quant_mode = 'W16A16'
    if quant_type == torch.int8:
      quant_mode = 'W8A16'
    if quant_type == torch.quint4x2:
      quant_mode = 'W4A16'

    for hidden_size in hidden_sizes:
      for inter_size in inter_sizes:
          weights = self.generate_weights(hidden_size, inter_size, dtype, quant_type)
          for row in rows:
                input_dict = self.generate_inputs(row, inter_size, dtype, quant_type)
                input_dict.update(weights)            
                act_output = self.run_ft_gemm(input_dict, row, hidden_size,inter_size,quant_mode)
                ref_output = self.run_ref_gemm(input_dict)

                msg = "gemm Failed"
                print(f"act_output: {act_output}")
                print(f"ref_output: {ref_output}")
                torch.testing.assert_close(act_output, ref_output, rtol=rtol, atol=atol, msg=msg, check_dtype=False)
  

  def test_fpAIntB_bf16_int8(self):
    self.gemm_test_helper(torch.bfloat16, torch.int8, rtol=1e-3, atol=0.05, \
                         hidden_sizes=[1024, 2048], \
                         inter_sizes=[4096])
  def a():
    m,n,k=16,1024, 4096
    x = torch.randn(m,k).cuda().to(torch.bfloat16)
    w = torch.randn(n,k).cuda().to(torch.bfloat16)
    wq = w.reshape(-1,128)
    max_val = wq.abs().amax(dim=1, keepdim=True)
    scales = max_val / 8
    wqq = (wq / scales).round().clamp(-8,7)
    wqq = wqq.view(w.shape)
    wqq = wqq.t().contiguous()
    from gemm_op import gemm_op_utils
    w_int4 = gemm_op_utils.pack_int8_tensor_to_packed_int4(wqq.char().cpu())
    w_int4 = gemm_op_utils.preprocess_weights_for_mixed_gemm(w_int4,1)
    scales = scales.view(1024,-1)
    from gemm_op import gemm_op_fpAIntB
    output = gemm_op_fpAIntB.fpAIntB_gemm(x,
                                w_int4, 
                                scales,
                                None,# bias
                                None,#out
                                None,#weight_zero_points
                                None,#alpha
                                128,# group_size
                                None,#tile config
                                None,# split_k_style
                                None,# split_k_factor
                                None) # stages


if __name__ == '__main__':
    unittest.main()


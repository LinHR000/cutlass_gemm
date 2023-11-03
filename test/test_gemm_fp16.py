import torch
# from cutlass_gemm import gemm_op,gemm_op_int8,gemm_op_fp16
from cutlass_gemm import CutlassGemm
import torch.nn as nn
from icecream import ic



m,n,k = 128,256,8192
input = torch.randint(low=-5, high=5, size=(m,k),dtype=torch.int8).cuda()
input_f = input.half()
weight = torch.randint(low=-5, high=5, size=(n,k),dtype=torch.int8).cuda()
weight_f = weight.half()
weight_t = weight.t().contiguous()
weight_t_half = weight_t.half()

cg_instance = CutlassGemm("/mnt/infra/haoran.lin2/cutlass_gemm/output/gemm_best_one.json")
output = cg_instance.gemm_in8_w8_ofp16_per_tensor(input, weight, 1.0, 0.0, m, n, k)
print(output)

output = cg_instance.gemm_in8_w8_ofp16_per_tensor_splitk(input, weight, 1.0, 0.0, m, n, k,splitk=2)
print(output)
import torch
# from cutlass_gemm import gemm_op,gemm_op_int8,gemm_op_fp16
from cutlass_gemm import CutlassGemm
import torch.nn as nn
from icecream import ic
import time
import torch.nn as nn

m,n,k = 1000,3840,1280
input = torch.randint(low=-5, high=5, size=(m,k),dtype=torch.int8).cuda()
input_f = input.half()
weight = torch.randint(low=-5, high=5, size=(n,k),dtype=torch.int8).cuda()
weight_f = weight.half()
weight_t = weight.t().contiguous()
weight_t_half = weight_t.half()

cg_instance = CutlassGemm("/mnt/infra/haoran.lin2/cutlass_gemm/output/gemm_best_one.json")
output = cg_instance.gemm_in8_w8_ofp16_per_tensor(input, weight, 1.0, 0.0, m, n, k)
total_time = 0
for i in range(1):
    torch.cuda.synchronize()
    time_start = time.time()
    output = cg_instance.gemm_in8_w8_ofp16_per_tensor(input, weight, 1.0, 0.0, m, n, k) 
    torch.cuda.synchronize()
    time_end = time.time()
    total_time+=(time_end-time_start)*1024
print(f"gemm per tensor time:{total_time/1}")

total_time_linear = 0
func = nn.Linear(k, n,bias=False,dtype=torch.float16).cuda()
func.weight.data.copy_(weight.half())
output = func(input_f)
for i in range(1):
    torch.cuda.synchronize()
    time_start = time.time()
    output = func(input_f)
    torch.cuda.synchronize()
    time_end = time.time()
    total_time_linear+=(time_end-time_start)*1000
print(f"gemm per tensor time:{total_time_linear/1}")

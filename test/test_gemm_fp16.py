import torch
from cutlass_gemm import gemm_op,gemm_op_int8,gemm_op_fp16
import torch.nn as nn
from icecream import ic

m,n,k = 16,1024,64
input = torch.randn(size=(m,k),dtype=torch.float16).cuda()
weight = torch.randint(size=(n,k),dtype=torch.float16).cuda()


# ref

func = nn.Linear(k, n,bias=False,dtype=torch.float16).cuda()
func.weight.data.copy_(weight)
ref_output = func(input)

# test int8 * int8 -> fp16 per tensor
tile_config = ""                             
cutlass_fp16_ofp16_gemm_reuslt = gemm_op_fp16.cutlass_fp16_ofp16_gemm(input,           # input
                                                                        weight,          # weight
                                                                        1.0,             # alpha
                                                                        0.0,             # beta
                                                                        m,               # m
                                                                        n,               # n
                                                                        k,               # k
                                                                        tile_config,     # tile config
                                                                        3,               # stages
                                                                        2)               # workspeace bytes
ic((ref_output - gemm_in8_w8_ofp16_per_tensor_result).pow(2).mean() / ref_output.pow(2).mean())

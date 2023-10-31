import torch
from cutlass_gemm import gemm_op
import torch.nn as nn
from icecream import ic
symmetric_quantizer = gemm_op.symmetric_quantize_last_axis_of_batched_matrix

m,n,k = 16,64,64
input = torch.randint(low=-5, high=5, size=(m,k),dtype=torch.int8).cuda()
input_f = input.half()
weight = torch.randint(low=-5, high=5, size=(n,k),dtype=torch.int8).cuda()
weight_f = weight.half()
weight_t = weight.t().contiguous()
weight_t_half = weight_t.half()
_,weight_mix,weight_mix_scale = symmetric_quantizer(weight_t_half.cpu(),0)
# symmetric_quantizer(weight_t_half.cpu(),0)
weight_mix  = weight_mix.cuda()
weight_mix_scale = weight_mix_scale.cuda()
bias_fp16 = torch.zeros(n,dtype=torch.float16).cuda()

alpha_row = torch.ones(m,1,dtype=torch.float32).cuda().contiguous()
alpha_col = torch.ones(1,n,dtype=torch.float32).cuda().contiguous()

# ref

func = nn.Linear(k, n,bias=False,dtype=torch.float16).cuda()
func.weight.data.copy_(weight_f)
ref_output = func(input_f)

# test int8 * int8 -> fp16 per tensor
tile_config = ""                             
gemm_in8_w8_ofp16_per_tensor_result = gemm_op.gemm_in8_w8_ofp16_per_tensor(input,           # input
                                                                        weight,          # weight
                                                                        1.0,             # alpha
                                                                        0.0,             # beta
                                                                        m,               # m
                                                                        n,               # n
                                                                        k,               # k
                                                                        tile_config,     # tile config
                                                                        3,               # stages
                                                                        1)               # workspeace bytes
ic((ref_output - gemm_in8_w8_ofp16_per_tensor_result).pow(2).mean() / ref_output.pow(2).mean())

# test gemm_in8_w8_ofp16 per token
gemm_in8_w8_ofp16_pt = gemm_op.gemm_in8_w8_ofp16_pt
gemm_in8_w8_ofp16_pt_result = gemm_in8_w8_ofp16_pt(input,weight,alpha_col,alpha_row,m,n,k)
ic((ref_output - gemm_in8_w8_ofp16_pt_result).pow(2).mean() / ref_output.pow(2).mean())

# test gemm_in8_w8_ofp16 per channel
gemm_in8_w8_ofp16_pc = gemm_op.gemm_in8_w8_ofp16_pc
gemm_in8_w8_ofp16_pc_reuslt = gemm_in8_w8_ofp16_pc(input,weight,alpha_col,alpha_row,m,n,k)
ic((ref_output - gemm_in8_w8_ofp16_pc_reuslt).pow(2).mean() / ref_output.pow(2).mean())

# test gemm_in8_w8_ofp16 per token per channel
gemm_in8_w8_ofp16_ptpc = gemm_op.gemm_in8_w8_ofp16_ptpc
gemm_in8_w8_ofp16_ptpc_reuslt = gemm_in8_w8_ofp16_ptpc(input,weight,alpha_col,alpha_row,m,n,k)
ic((ref_output - gemm_in8_w8_ofp16_ptpc_reuslt).pow(2).mean() / ref_output.pow(2).mean())

# test gemm_infp16_w8_ofp16 weight only
gemm_infp16_w8_ofp16 = gemm_op.gemm_infp16_w8_ofp16
gemm_infp16_w8_ofp16_result = gemm_infp16_w8_ofp16(input_f,weight_mix,weight_mix_scale)
ic((ref_output - gemm_infp16_w8_ofp16_result).pow(2).mean() / ref_output.pow(2).mean())

# test gemm_infp16_w8_ofp16_bias_act
gelu_func = torch.nn.GELU()
ref_gelu = gelu_func(ref_output)

gemm_infp16_w8_ofp16_bias_act = gemm_op.gemm_infp16_w8_ofp16_bias_act
gemm_infp16_w8_ofp16_bias_act_result = gemm_infp16_w8_ofp16_bias_act(input_f,weight_mix,weight_mix_scale,bias_fp16,'gelu')
ic((ref_gelu - gemm_infp16_w8_ofp16_bias_act_result).pow(2).mean() / ref_gelu.pow(2).mean())

                                                                  
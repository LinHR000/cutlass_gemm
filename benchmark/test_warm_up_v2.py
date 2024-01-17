import torch
import time
from cutlass_gemm import gemm_op,gemm_op_int8
symmetric_quantizer = gemm_op.symmetric_quantize_last_axis_of_batched_matrix

def per_channel_quant(x):
    weight_scale = torch.amax(x,dim=1).float() / 127.0
    weight_scale = weight_scale.view(-1,1)
    x_quant = (x / weight_scale).round().clamp(-128,127).to(torch.int8)
    weight_scale = weight_scale.view(1,-1)
    return x_quant,weight_scale

def per_token_quant(x):
    weight_scale = torch.amax(x,dim=2).float() / 127.0
    weight_scale = weight_scale.view(-1,1)
    x_quant = (x / weight_scale).round().clamp(-128,127).to(torch.int8)
    weight_scale = weight_scale.view(-1,1)
    return x_quant,weight_scale


# per tensor, ptpc交替执行
config_qkv = ["CtaShape64x128x128_WarpShape32x64x128", 0, 1, 4] # 0.05791187286376953
config_proj = [6, 1, 6, 3] # 0.029611587524414062
config_fc1 = ["CtaShape64x128x128_WarpShape32x64x128", 0, 1, 3] # 0.0866651535034179
config_fc2 = [6, 1, 3, 4] # 0.05619525909423828

bsz = 1
seq_len = 8
hidden_size = 4608
inter_dim = 12288
low=-1e-3
high=1e-3
input = torch.randint(low=-5, high=5, size=(bsz,seq_len,hidden_size),dtype=torch.int8).cuda()

weight_qkv = torch.empty(hidden_size,hidden_size*3,dtype=torch.float16).cuda()
weight_qkv.uniform_(low,high)
weight_qkv_scale = (weight_qkv.amax() / 127).view(1).float()
weight_qkv_quant = (weight_qkv / weight_qkv_scale).round().clamp(-127,128).to(torch.int8)

weight_proj = torch.randn(hidden_size,hidden_size,dtype=torch.float16).cuda()
weight_proj.uniform_(low,high)
weight_proj_mix,weight_proj_mix_scale = per_channel_quant(weight_proj)


weight_fc1 = torch.randn(hidden_size,inter_dim*2,dtype=torch.float16).cuda()
weight_fc1.uniform_(low,high)
weight_fc1_scale = (weight_fc1.amax() / 127).view(1).float()
weight_fc1_quant = (weight_fc1 / weight_fc1_scale).round().clamp(-127,128).to(torch.int8)


weight_fc2 = torch.randn(hidden_size,inter_dim,dtype=torch.float16).cuda()
weight_fc2.uniform_(low,high)
weight_fc2_mix,weight_fc2_mix_scale = per_channel_quant(weight_fc2)

output_qkv = torch.empty(bsz,seq_len,hidden_size*3,dtype=torch.float16).cuda()
output_proj = torch.empty(bsz,seq_len,hidden_size,dtype=torch.float16).cuda()
output_fc1 = torch.empty(bsz,seq_len,inter_dim*2,dtype=torch.float16).cuda()
output_fc2 = torch.empty(bsz,seq_len,hidden_size,dtype=torch.float16).cuda()

alpha_qkv = weight_qkv_scale.tolist()[0]
output_fc2_quant = input
m=1
n_qkv = hidden_size*3
k_qkv = hidden_size
n_fc1 = inter_dim*2
k_fc1 = hidden_size
n_proj = hidden_size
k_proj = hidden_size
n_fc2 = hidden_size
k_fc2 = inter_dim

def global_warm_up():
    bsz_ = 1
    seq_len_ = 8
    hidden_size_ = 4608
    inter_dim_ = 12288


    weight_qkv_ = torch.randn(hidden_size_,hidden_size_*3,dtype=torch.float16).cuda()
    weight_qkv_scale_ = (weight_qkv_.amax() / 127).view(1).float()
    weight_qkv_quant_ = (weight_qkv_ / weight_qkv_scale_).round().clamp(-127,127).to(torch.int8)

    weight_proj_ = torch.randn(hidden_size_,hidden_size_,dtype=torch.float16).cuda()
    weight_proj_mix_,weight_proj_mix_scale_ = per_channel_quant(weight_proj_)

    weight_fc1_ = torch.randn(hidden_size_,inter_dim_*2,dtype=torch.float16).cuda()
    weight_fc1_scale_ = (weight_fc1_.amax() / 127).view(1).float()
    weight_fc1_quant_ = (weight_fc1_ / weight_fc1_scale_).round().clamp(-127,127).to(torch.int8)


    weight_fc2_ = torch.randn(hidden_size_,inter_dim_,dtype=torch.float16).cuda()
    weight_fc2_mix_,weight_fc2_mix_scale_ = per_channel_quant(weight_fc2_)

    output_qkv_ = torch.empty(bsz_,seq_len_,hidden_size_*3,dtype=torch.float16).cuda()
    output_proj_ = torch.empty(bsz_,seq_len_,hidden_size_,dtype=torch.float16).cuda()
    output_fc1_ = torch.empty(bsz_,seq_len_,inter_dim_*2,dtype=torch.float16).cuda()
    output_fc2_ = torch.empty(bsz_,seq_len_,hidden_size_,dtype=torch.float16).cuda()

    m_=1
    n_qkv_ = hidden_size_*3
    k_qkv_ = hidden_size_
    n_proj_ = hidden_size_
    k_proj_ = hidden_size_
    n_fc1_ = inter_dim_*2
    k_fc1_ = hidden_size_
    n_fc2_ = hidden_size_
    k_fc2_ = inter_dim_

    input_qkv_ = torch.randint(low=-5, high=5, size=(bsz_,seq_len_,hidden_size_),dtype=torch.int8).cuda()
    alpha_row_ = torch.ones(m_,1,dtype=torch.float32).cuda().contiguous()
    input_fc2_ = torch.randint(low=-5, high=5, size=(bsz_,seq_len_,inter_dim_),dtype=torch.int8).cuda()


    gemm_op_int8.gemm_in8_w8_ofp16_per_tensor(output_qkv_,input_qkv_,weight_qkv_quant_,None,1.0,0.0,m_,n_qkv_,k_qkv_,config_qkv[0],config_qkv[3],config_qkv[2])
    gemm_op.gemm_in8_w8_ofp16_ptpc(output_proj_,input_qkv_,weight_proj_mix_,alpha_row_,weight_proj_mix_scale_,m_,n_proj_,k_proj_,config_proj[0],config_proj[1],config_proj[2],config_proj[3])
    gemm_op_int8.gemm_in8_w8_ofp16_per_tensor(output_fc1_,input_qkv_,weight_fc1_quant_,None,1.0,0.0,m_,n_fc1_,k_fc1_,config_fc1[0],config_fc1[3],config_fc1[2])
    gemm_op.gemm_in8_w8_ofp16_ptpc(output_proj_,input_fc2_,weight_fc2_mix_, alpha_row_, weight_fc2_mix_scale_,m_,n_fc2_,k_fc2_,config_proj[0],config_proj[1],config_proj[2],config_proj[3])
    pass

global_warm_up()

for i in range(10):
    print("#"*10+f"layer {i}"+ "#"*10)
    torch.cuda.synchronize()
    time_s = time.time()
    output_qkv =  gemm_op_int8.gemm_in8_w8_ofp16_per_tensor(output_qkv,output_fc2_quant,weight_qkv_quant,None,alpha_qkv,0.0,m,n_qkv,k_qkv,config_qkv[0],config_qkv[3],config_qkv[2])
    torch.cuda.synchronize()
    time_e = time.time()
    print("qkv cost time:",(time_e-time_s)*1000)
    output_qkv_split = output_qkv[:,:,:hidden_size].contiguous() 
    output_qkv_split_quant,output_qkv_qlpha = per_token_quant(output_qkv_split)

    torch.cuda.synchronize()
    time_s = time.time()
    output_proj = gemm_op.gemm_in8_w8_ofp16_ptpc(output_proj,output_qkv_split_quant,weight_proj_mix, output_qkv_qlpha,weight_proj_mix_scale,m,n_proj,k_proj,config_proj[0],config_proj[1],config_proj[2],config_proj[3])
    torch.cuda.synchronize()
    time_e = time.time()
    print("proj cost time:",(time_e-time_s)*1000)
    output_proj_sclae = (output_proj.amax() / 127).view(1).float()
    output_proj_quant = (output_proj / output_proj_sclae).round().clamp(-127,128).to(torch.int8)

    alpha_fc1 = output_proj_sclae * weight_fc1_scale
    alpha_fc1 = alpha_fc1.tolist()[0]
    torch.cuda.synchronize()
    time_s = time.time()
    output_fc1 =  gemm_op_int8.gemm_in8_w8_ofp16_per_tensor(output_fc1,output_proj_quant,weight_fc1_quant,None,alpha_fc1,0.0,m,n_fc1,k_fc1,config_fc1[0],config_fc1[3],config_fc1[2])
    torch.cuda.synchronize()
    time_e = time.time()
    print("fc1 cost time:",(time_e-time_s)*1000)
    output_fc1_split = output_fc1[:,:,:inter_dim].contiguous()
    output_fc1_split_quant,output_fc1_alpha = per_token_quant(output_fc1_split)

    torch.cuda.synchronize()
    time_s = time.time()
    output_fc2 = gemm_op.gemm_in8_w8_ofp16_ptpc(output_fc2,output_fc1_split_quant,weight_fc2_mix,output_fc1_alpha,weight_fc2_mix_scale,m,n_fc2,k_fc2,config_fc2[0],config_fc2[1],config_fc2[2],config_fc2[3])
    torch.cuda.synchronize()
    time_e = time.time()
    print("fc2 cost time:",(time_e-time_s)*1000)
    output_fc2_sclae = (output_fc2.amax() / 127).view(1).float()
    output_fc2_quant = (output_fc2 / output_fc2_sclae).round().clamp(-127,128).to(torch.int8)
    alpha_qkv = weight_qkv_scale * output_fc2_sclae
    alpha_qkv = alpha_qkv.tolist()[0]



# no warm  up
# qkv cost time: 0.06532669067382812
# proj cost time: 0.06031990051269531
# fc1 cost time: 0.09465217590332031
# fc2 cost time: 0.09131431579589844
# warm up
# qkv cost time: 0.06341934204101562
# proj cost time: 0.06318092346191406
# fc1 cost time: 0.0972747802734375
# fc2 cost time: 0.09083747863769531












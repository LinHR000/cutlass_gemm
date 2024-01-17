import torch
from cutlass_gemm import gemm_op,gemm_op_int8
import argparse
import time
import math
import os
import torch.nn as nn
import json
def main(args):
    # prepare data for test
    with open("/mnt/infra/haoran.lin2/cutlass_gemm/output/gemm_best_13b_per_tensor_1gpu_fc1.json",'r') as r:
        config_dict = json.load(r)

    with open('/mnt/infra/haoran.lin2/cutlass_gemm/output/gemm_wt_13b_weight_only_qkv_1gpu.json','r') as r:
        wt_config_dict_ = json.load(r)
    wt_config_dict = {}
    for key,value in wt_config_dict_.items():
        wt_config_dict[key] = [int(i) for i in value.split(" ")]


    symmetric_quantizer = gemm_op.symmetric_quantize_last_axis_of_batched_matrix
    # for m in range(8,args.m,32):
    for m in range(1024,2000,2000):
        # for key,config_dict_best in config_dict.items():
        #     if m < int(key):
        #         break
        config_dict_best =config_dict[str(int(math.ceil(m/8))*8)]
        config_dict_best = config_dict_best.split("#")
        tile_config,stages,splitk = config_dict_best[0],int(config_dict_best[1]),int(config_dict_best[2])
        tile_config = 'CtaShape64x128x128_WarpShape32x64x128'
        stages=4
        splitk=1
        n,k = args.n,args.k
        input = torch.randint(low=-127, high=127, size=(m,k),dtype=torch.int8).cuda()
        weight = torch.randint(low=-127, high=127, size=(n,k),dtype=torch.int8).cuda()
        input_f = input.half()
        weight_t = weight.t().contiguous()
        weight_t_half = weight_t.half()
        _,weight_mix,weight_mix_scale = symmetric_quantizer(weight_t_half.cpu(),0)
        # symmetric_quantizer(weight_t_half.cpu(),0)
        weight_mix  = weight_mix.cuda()
        weight_mix_scale = weight_mix_scale.cuda()

        bias_int8 = torch.ones(n,dtype=torch.int8).cuda()
        bias_int32 = torch.ones(n,dtype=torch.int32).cuda()
        bias_fp16 = torch.ones(n,dtype=torch.float16).cuda()

        output_int8 = torch.empty(m,n,dtype=torch.int8).cuda()
        output_int32 = torch.empty(m,n,dtype=torch.int32).cuda()
        output_fp16 = torch.empty(m,n,dtype=torch.float16).cuda()

        alpha_row = torch.ones(m,1,dtype=torch.float32).cuda().contiguous()
        alpha_col = torch.ones(1,n,dtype=torch.float32).cuda().contiguous()

        lda,ldb,ldc = k,k,n

        

        gemm_in8_w8_ofp16_per_tensor = gemm_op_int8.gemm_in8_w8_ofp16_per_tensor
        print(input.shape)
        print(weight.shape)
        gemm_in8_w8_ofp16_per_tensor(output_fp16,input,weight,bias_fp16, 1.0,0.0,m,n,k,tile_config,stages,splitk)
        gemm_in8_w8_ofp16_per_tensor_time = 0
        for i in range(args.num_iters):
            torch.cuda.synchronize()
            time_start = time.time()
            gemm_in8_w8_ofp16_per_tensor(output_fp16,input,weight,None,1.0,0.0,m,n,k,tile_config,stages,splitk)
            torch.cuda.synchronize()
            time_end = time.time()
            a = time_end-time_start
            # print(a)
            gemm_in8_w8_ofp16_per_tensor_time+=a
        gemm_in8_w8_ofp16_per_tensor_time = gemm_in8_w8_ofp16_per_tensor_time * 1000 / args.num_iters

        torch.cuda.synchronize()
        time_start = time.time()
        gemm_in8_w8_o8_per_tensor = gemm_op_int8.gemm_in8_w8_o8_per_tensor
        torch.cuda.synchronize()
        time_end = time.time()
        a = time_end-time_start
        print(a * 1000)
        print(input.shape)
        print(weight.shape)
        gemm_in8_w8_o8_per_tensor(output_int8,input,weight, 1.0,0.0,m,n,k,tile_config,stages,splitk)
        gemm_in8_w8_o8_per_tensor_time = 0
        for i in range(args.num_iters):
            torch.cuda.synchronize()
            time_start = time.time()
            gemm_in8_w8_o8_per_tensor(output_int8,input,weight,1.0,0.0,m,n,k,tile_config,stages,splitk)
            torch.cuda.synchronize()
            time_end = time.time()
            a = time_end-time_start
            # print(a)
            gemm_in8_w8_o8_per_tensor_time+=a
        gemm_in8_w8_o8_per_tensor_time = gemm_in8_w8_o8_per_tensor_time * 1000 / args.num_iters


        # test gemm_in8_w8_ofp16_pt
        gemm_in8_w8_ofp16_pt = gemm_op.gemm_in8_w8_ofp16_pt
        gemm_in8_w8_ofp16_pt(output_fp16,input,weight,alpha_col,alpha_row,m,n,k) + bias_fp16
        gemm_in8_w8_ofp16_pt_time = 0
        for i in range(args.num_iters):
            torch.cuda.synchronize()
            time_start = time.time()
            gemm_in8_w8_ofp16_pt(output_fp16,input,weight,alpha_col,alpha_row,m,n,k) 
            torch.cuda.synchronize()
            time_end = time.time()
            gemm_in8_w8_ofp16_pt_time+=time_end-time_start
        gemm_in8_w8_ofp16_pt_time = gemm_in8_w8_ofp16_pt_time * 1000 / args.num_iters

        # test gemm_in8_w8_ofp16_pc
        gemm_in8_w8_ofp16_pc = gemm_op.gemm_in8_w8_ofp16_pc
        gemm_in8_w8_ofp16_pc(output_fp16,input,weight,alpha_col,alpha_row,m,n,k) + bias_fp16
        gemm_in8_w8_ofp16_pc_time = 0
        for i in range(args.num_iters):
            torch.cuda.synchronize()
            time_start = time.time()
            gemm_in8_w8_ofp16_pc(output_fp16,input,weight,alpha_col,alpha_row,m,n,k)
            torch.cuda.synchronize()
            time_end = time.time()
            gemm_in8_w8_ofp16_pc_time+=time_end-time_start
        gemm_in8_w8_ofp16_pc_time = gemm_in8_w8_ofp16_pc_time * 1000 / args.num_iters

        # test gemm_in8_w8_ofp16_ptpc
        gemm_in8_w8_ofp16_ptpc = gemm_op.gemm_in8_w8_ofp16_ptpc
        gemm_in8_w8_ofp16_ptpc(output_fp16,input,weight,alpha_col,alpha_row,m,n,k) 
        gemm_in8_w8_ofp16_ptpc_time = 0
        for i in range(args.num_iters):
            torch.cuda.synchronize()
            time_start = time.time()
            gemm_in8_w8_ofp16_ptpc(output_fp16,input,weight,alpha_col,alpha_row,m,n,k) 
            torch.cuda.synchronize()
            time_end = time.time()
            gemm_in8_w8_ofp16_ptpc_time+=time_end-time_start
        gemm_in8_w8_ofp16_ptpc_time = gemm_in8_w8_ofp16_ptpc_time * 1000 / args.num_iters

        if m<256:
            wt_list = wt_config_dict[str(m)]
        elif m > 4080:
            wt_list = wt_config_dict[str(4080)]
        else:
            m_ = math.ceil(m / 8) * 8 
            wt_list = wt_config_dict[str(m_)]
        wt_list = [3, 1, 1, 4]
        # test gemm_infp16_w8_ofp16
        gemm_infp16_w8_ofp16 = gemm_op.gemm_infp16_w8_ofp16
        # gemm_infp16_w8_ofp16(output_fp16,input_f,weight_mix,weight_mix_scale,3,1,2,4)
        gemm_infp16_w8_ofp16(output_fp16,input_f,weight_mix,weight_mix_scale,wt_list[0],wt_list[1],wt_list[2],wt_list[3])
        gemm_infp16_w8_ofp16_time = 0
        for i in range(args.num_iters):
            torch.cuda.synchronize()
            time_start = time.time()
            gemm_infp16_w8_ofp16(output_fp16,input_f,weight_mix,weight_mix_scale,wt_list[0],wt_list[1],wt_list[2],wt_list[3]) 
            torch.cuda.synchronize()
            time_end = time.time()
            gemm_infp16_w8_ofp16_time+=time_end-time_start
        gemm_infp16_w8_ofp16_time = gemm_infp16_w8_ofp16_time * 1000 / args.num_iters
        # test gemm_infp16_w8_ofp16_bias_act

        
        print(input_f.shape)
        print(weight_mix.shape)
        # input_f = input_f.view(1,1,-1)
        gemm_infp16_w8_ofp16_bias_act = gemm_op.gemm_infp16_w8_ofp16_bias_act
        gemm_infp16_w8_ofp16_bias_act(output_fp16,input_f,weight_mix,weight_mix_scale,bias_fp16,'identity',wt_list[0],wt_list[1],wt_list[2],wt_list[3])
        gemm_infp16_w8_ofp16_bias_act_time = 0
        for i in range(args.num_iters):
            torch.cuda.synchronize()
            time_start = time.time()
            gemm_infp16_w8_ofp16_bias_act(output_fp16,input_f,weight_mix,weight_mix_scale,bias_fp16,'identity',wt_list[0],wt_list[1],wt_list[2],wt_list[3])
            torch.cuda.synchronize()
            time_end = time.time()
            gemm_infp16_w8_ofp16_bias_act_time+=time_end-time_start
        gemm_infp16_w8_ofp16_bias_act_time = gemm_infp16_w8_ofp16_bias_act_time * 1000 / args.num_iters

        

        # gemm_in8_w8_ofp16_per_tensor_splitk = gemm_op_int8.gemm_in8_w8_ofp16_per_tensor_splitk
        # gemm_in8_w8_ofp16_per_tensor_splitk(input,weight,1.0,0.0,m,n,k,tile_config,stages,splitk)
        # gemm_in8_w8_ofp16_per_tensor_splitk_time = 0
        # for i in range(args.num_iters):
        #     torch.cuda.synchronize()
        #     time_start = time.time()
        #     gemm_in8_w8_ofp16_per_tensor_splitk(input,weight,1.0,0.0,m,n,k,tile_config,stages,splitk)
        #     torch.cuda.synchronize()
        #     time_end = time.time()
        #     gemm_in8_w8_ofp16_per_tensor_splitk_time+=time_end-time_start
        # gemm_in8_w8_ofp16_per_tensor_splitk_time = gemm_in8_w8_ofp16_per_tensor_splitk_time * 1000 / args.num_iters

        # linear


        total_time_linear = 0
        func = nn.Linear(k, n,bias=False,dtype=torch.float16).cuda()
        func.weight.data.copy_(weight_t_half.t())
        output = func(input_f)
        for i in range(args.num_iters):
            torch.cuda.synchronize()
            time_start = time.time()
            output = func(input_f)
            torch.cuda.synchronize()
            time_end = time.time()
            total_time_linear+=time_end-time_start
        total_time_linear = total_time_linear * 1000 / args.num_iters



        print("="*10+"M={}".format(m)+"="*10)
        print("TIME INT8 * INT8 -> FP16 (per tensor):",gemm_in8_w8_ofp16_per_tensor_time)
        print("TIME INT8 * INT8 -> INT8 (per tensor):",gemm_in8_w8_o8_per_tensor_time)
        # print("TIME INT8 * INT8 -> FP16 (per tensor splitk):",gemm_in8_w8_ofp16_per_tensor_splitk_time)
        print("TIME INT8 * INT8 -> FP16 (per token):",gemm_in8_w8_ofp16_pt_time)
        print("TIME INT8 * INT8 -> FP16 (per channel)",gemm_in8_w8_ofp16_pc_time )
        print("TIME INT8 * INT8 -> FP16 (per token per channel):",gemm_in8_w8_ofp16_ptpc_time)
        print("TIME INT8 * FP16 -> Fp16 (WO bias):",gemm_infp16_w8_ofp16_time)
        print("TIME INT8 * FP16 -> Fp16 (WI bias):",gemm_infp16_w8_ofp16_bias_act_time)
        print("TIME Linear:",total_time_linear)

        print("Speed Up INT8 * INT8 -> FP16 (per tensor):{}%".format(round((total_time_linear-gemm_in8_w8_ofp16_per_tensor_time)/total_time_linear*100,2)))
        print("Speed Up INT8 * INT8 -> INT8 (per tensor):{}%".format(round((total_time_linear-gemm_in8_w8_o8_per_tensor_time)/total_time_linear*100,2)))
        # print("Speed Up INT8 * INT8 -> FP16 (per tensor splitk):{}%".format(round((total_time_linear-gemm_in8_w8_ofp16_per_tensor_splitk_time)/total_time_linear*100,2)))
        print("Speed Up INT8 * INT8 -> FP16 (per token):{}%".format(round((total_time_linear-gemm_in8_w8_ofp16_pt_time)/total_time_linear*100,2)))
        print("Speed Up INT8 * INT8 -> FP16 (per channel):{}%".format(round((total_time_linear-gemm_in8_w8_ofp16_pc_time)/total_time_linear*100,2)))
        print("Speed Up INT8 * INT8 -> FP16 (per token per channel):{}%".format(round((total_time_linear-gemm_in8_w8_ofp16_ptpc_time)/total_time_linear*100,2)))
        print("Speed Up INT8 * FP16 -> Fp16 (WO bias):{}%".format(round((total_time_linear-gemm_infp16_w8_ofp16_time)/total_time_linear*100,2)))
        print("Speed Up INT8 * FP16 -> Fp16 (WI bias):{}%".format(round((total_time_linear-gemm_infp16_w8_ofp16_bias_act_time)/total_time_linear*100,2)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark the latency of processing a single gemm.')
    parser.add_argument('--m', type=int, default=4096)
    # parser.add_argument('--n', type=int, default=int(12288*2))
    parser.add_argument('--n', type=int, default=int(4608*3))
    parser.add_argument('--k', type=int, default=int(4608))
    parser.add_argument('--num-iters', type=int, default=10,
                        help='Number of iterations to run.')
    args = parser.parse_args()
    print(args)
    main(args)
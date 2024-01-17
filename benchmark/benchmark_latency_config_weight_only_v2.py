import torch
from cutlass_gemm import gemm_op,gemm_op_int8
import argparse
import time
import torch.nn as nn
import json
import os
from tqdm import tqdm
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def main(args):
    def get_config():
        config_list = []
        for t in [3,5,7]:
            for s in [0,1]:
                if s == 0:
                    for sp in [1]:
                        for st in [2,3,4]:
                            config_list.append([t,s,sp,st])
                else:
                    for sp in [1,2,3,4,5,6,7]:
                        for st in [2,3,4]:
                            config_list.append([t,s,sp,st]) 
        return config_list

    symmetric_quantizer = gemm_op.symmetric_quantize_last_axis_of_batched_matrix
    # for m in range(8,args.m,32):
    config_list = get_config()
    result_dict = {}
    m_list = list([i for i in range(1,256)])
    m_list += list([i for i in range(256,4096,8)])
    n,k = args.n,args.k
    weight = torch.randint(low=-127, high=127, size=(n,k),dtype=torch.int8).cuda()
    weight_t = weight.t().contiguous()
    weight_t_half = weight_t.half()
    _,weight_mix,weight_mix_scale = symmetric_quantizer(weight_t_half.cpu(),0)
    # symmetric_quantizer(weight_t_half.cpu(),0)
    weight_mix  = weight_mix.cuda()
    weight_mix_scale = weight_mix_scale.cuda()
    for m in tqdm(m_list):
        
        
        input_f = torch.randn(m,k,dtype=torch.float16).cuda()
        

        output_fp16 = torch.empty(m,n,dtype=torch.float16).cuda()

        # alpha_row = torch.ones(m,1,dtype=torch.float32).cuda().contiguous()
        # alpha_col = torch.ones(1,n,dtype=torch.float32).cuda().contiguous()

        lda,ldb,ldc = k,k,n

        # config = gemm_op.choose_best_config_half(input_f,weight_mix,weight_mix_scale)
        # print("="*10+"M={}".format(m)+"="*10)


        tmp_dict = {}
        for config in config_list:
            gemm_infp16_w8_ofp16 = gemm_op.gemm_infp16_w8_ofp16
            try:
                for i in range(5):
                    gemm_infp16_w8_ofp16(output_fp16,input_f,weight_mix,weight_mix_scale,config[0],config[1],config[2],config[3])
            except Exception as e:
                continue
            gemm_infp16_w8_ofp16_bias_act_time = 0
            for i in range(args.num_iters):
                torch.cuda.synchronize()
                time_start = time.time()
                gemm_infp16_w8_ofp16(output_fp16,input_f,weight_mix,weight_mix_scale,config[0],config[1],config[2],config[3])
                torch.cuda.synchronize()
                time_end = time.time()
                gemm_infp16_w8_ofp16_bias_act_time+=time_end-time_start
            gemm_infp16_w8_ofp16_bias_act_time = gemm_infp16_w8_ofp16_bias_act_time * 1000 / args.num_iters
            # print(config,gemm_infp16_w8_ofp16_bias_act_time)
            tmp_dict[" ".join(map(lambda x: str(x), config))] = gemm_infp16_w8_ofp16_bias_act_time
            
        result_dict[m] = tmp_dict
    with open("/mnt/infra/haoran.lin2/cutlass_gemm/benchmark/gemm_wt_13b_fc1_2gpu_v2.json",'w') as w:
        json.dump(result_dict,w)               

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark the latency of processing a single gemm.')
    parser.add_argument('--m', type=int, default=4608)
    parser.add_argument('--n', type=int, default=int(12288))
    parser.add_argument('--k', type=int, default=int(4608))
    parser.add_argument('--num-iters', type=int, default=10,
                        help='Number of iterations to run.')
    args = parser.parse_args()
    print(args)
    main(args)
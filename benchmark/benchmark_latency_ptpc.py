import torch
from cutlass_gemm import gemm_op,gemm_op_int8
import argparse
import time
import torch.nn as nn
import json
import os
from tqdm import tqdm
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def main(args):
    def get_config():
        config_list = []
        for t in [3,4,6]:
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

    # prepare data for test

    config_list = get_config()
    result_dict = {}
    m_list = list([i for i in range(1,256)])
    m_list += list([i for i in range(256,4096,8)])
    for m in tqdm(m_list):

        n,k = args.n,args.k
        input = torch.randint(low=-127, high=127, size=(m,k),dtype=torch.int8).cuda()
        weight = torch.randint(low=-127, high=127, size=(n,k),dtype=torch.int8).cuda()

        output_fp16 = torch.empty(m,n,dtype=torch.float16).cuda()

        alpha_row = torch.ones(m,1,dtype=torch.float32).cuda().contiguous()
        alpha_col = torch.ones(1,n,dtype=torch.float32).cuda().contiguous()

        lda,ldb,ldc = k,k,n

        tmp_dict = {}
        for config in config_list:
            gemm_in8_w8_ofp16_ptpc = gemm_op.gemm_in8_w8_ofp16_ptpc
            try:
                for i in range(5):
                    gemm_in8_w8_ofp16_ptpc(output_fp16,input,weight,alpha_col,alpha_row,m,n,k,config[0],config[1],config[2],config[3])
            except Exception as e:
                continue
            gemm_in8_w8_ofp16_ptpc_time = 0
            for i in range(args.num_iters):
                torch.cuda.synchronize()
                time_start = time.time()
                gemm_in8_w8_ofp16_ptpc(output_fp16,input,weight,alpha_col,alpha_row,m,n,k,config[0],config[1],config[2],config[3])
                torch.cuda.synchronize()
                time_end = time.time()
                gemm_in8_w8_ofp16_ptpc_time+=time_end-time_start
            gemm_in8_w8_ofp16_ptpc_time = gemm_in8_w8_ofp16_ptpc_time * 1000 / args.num_iters
            # print(config,gemm_infp16_w8_ofp16_bias_act_time)
            tmp_dict[" ".join(map(lambda x: str(x), config))] = gemm_in8_w8_ofp16_ptpc_time
            
        result_dict[m] = tmp_dict
    with open(args.save_path,'w') as w:
        json.dump(result_dict,w)               

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark the latency of processing a single gemm.')
    parser.add_argument('--m', type=int, default=4608)
    parser.add_argument('--n', type=int, default=int(4608*3))
    parser.add_argument('--k', type=int, default=int(4608))
    parser.add_argument('--num-iters', type=int, default=10,
                        help='Number of iterations to run.')
    parser.add_argument('--save_path', type=str, default='/mnt/infra/haoran.lin2/cutlass_gemm/benchmark/gemm_ptpc_13b_proj_1gpu.json',
                        help='Number of iterations to run.')
            
    args = parser.parse_args()
    print(args)
    main(args)
    # CUDA_VISIBLE_DEVICES=3  python benchmark_latency_ptpc.py  --m  4096 --n 4608 --k 12288 --save_path /mnt/infra/haoran.lin2/cutlass_gemm/benchmark/gemm_ptpc_13b_fc2_1gpu.json  
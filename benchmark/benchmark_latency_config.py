import torch
from cutlass_gemm import gemm_op,gemm_op_int8
import argparse
import time
import torch.nn as nn
import json
def main(args):
    # prepare data for test
    with open("/mnt/infra/haoran.lin2/cutlass_gemm/output/gemm_best_choose.json",'r') as r:
        config_dict = json.load(r)
    symmetric_quantizer = gemm_op.symmetric_quantize_last_axis_of_batched_matrix
    # for m in range(8,args.m,32):
    for m in range(3000,args.m,32):
        for key,config_dict_best in config_dict.items():
            if m < int(key):
                break
        config_dict_best = config_dict_best.split("#")
        tile_config,stages,splitk = config_dict_best[0],int(config_dict_best[1]),int(config_dict_best[2])
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

       
        time_config = 0
        for i in range(args.num_iters):
            torch.cuda.synchronize()
            time_start = time.time()
            gemm_op.choose_best_config_half(input_f,weight_mix,weight_mix_scale)
            torch.cuda.synchronize()
            time_end = time.time()
            time_config+=time_end-time_start
        time_config = time_config * 1000 / args.num_iters



        print("="*10+"M={}".format(m)+"="*10)
        print("TIME config:",time_config)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark the latency of processing a single gemm.')
    parser.add_argument('--m', type=int, default=8192)
    parser.add_argument('--n', type=int, default=int(8192))
    parser.add_argument('--k', type=int, default=int(8192))
    parser.add_argument('--num-iters', type=int, default=10,
                        help='Number of iterations to run.')
    args = parser.parse_args()
    print(args)
    main(args)
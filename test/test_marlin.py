import sys

import numpy as np
import torch
import marlin
import torch.nn as nn
import time
from gemm_op import gemm_op_fp8,fp8_dtype_converter,fp8_gemm_v2

alpha = 1.0
a_scale=1.0
b_scale=1.0
c_scale=1.0
d_scale=1.0
amax_d=468.0

def benchmark(f, warmup=10, iter=10):
    for i in range(warmup + iter):
        f()
        # We do not synchronize here in order to hide the kernel launch overhead during benchmarkining as this will also
        # happen during realistic model inference as many launches are submitted to the kernel queue.
        if i == warmup - 1:
            torch.cuda.synchronize()
            tick = time.time()
    torch.cuda.synchronize()
    res = (time.time() - tick) / iter
    # Make sure there is enough to "cool down" the GPU in between benchmarks to avoid throttling for later runs when
    # we execute many benchmarks consecutively
    time.sleep(1.)
    return res * 1000

def get_problem(m, n, k, groupsize=-1):
    if groupsize == -1:
        groupsize = k
    dev = torch.device('cuda:0')
    A = torch.randn((m, k), dtype=torch.half, device=dev)
    B = torch.randint(low=-2**31, high=2**31, size=(k * n // 8,), device=dev)
    B_ref = torch.randn((k, n), dtype=torch.half, device=dev)
    C = torch.zeros((m, n), dtype=torch.half, device=dev)
    s = torch.zeros((k // groupsize, n), dtype=torch.half, device=dev)
    torch.cuda.synchronize()
    return A, B, C, B_ref, s

def benchmark_dense(A, B, C):
    res = benchmark(lambda: torch.matmul(A, B, out=C))
    return {
        's': res,
        'TFLOP/s': 2 * A.numel() * C.shape[1] / res / 10 ** 12,
        'GB/s': (2 * A.numel() + 2 * B.numel() + 2 * C.numel()) / res / 10 ** 9
    }

def benchmark_nnLinear(A, B, C):
    nn_linear = nn.Linear(B.shape[0],B.shape[1],bias=False).to(A.dtype).cuda()
    nn_linear.weight.data.copy_(B.t().contiguous())
    res = benchmark(lambda: nn_linear(A))
    return {
        's': res,
        'TFLOP/s': 2 * A.numel() * C.shape[1] / res / 10 ** 12,
        'GB/s': (2 * A.numel() + 2 * B.numel() + 2 * C.numel()) / res / 10 ** 9
    }

def benchmark_fp8(A, B, C):
    A_fp8 = torch.zeros_like(A,dtype=torch.int8)
    B_t = B.t().contiguous()
    B_fp8 = torch.zeros_like(B_t,dtype=torch.int8)
    fp8_dtype_converter.fp8_dtype_converter(A,A_fp8,'fp8_e4m3')
    fp8_dtype_converter.fp8_dtype_converter(B_t,B_fp8,'fp8_e4m3')
    res = benchmark(lambda: gemm_op_fp8.fp8_gemm(A_fp8,
                                                B_fp8,
                                                alpha,
                                                a_scale,
                                                b_scale,
                                                c_scale,
                                                d_scale,
                                                amax_d))
    return {
        's': res,
        'TFLOP/s': 2 * A.numel() * C.shape[1] / res / 10 ** 12,
        'GB/s': (2 * A.numel() + 2 * B.numel() + 2 * C.numel()) / res / 10 ** 9
    }

def benchmark_quant(A, B, C, s, thread_k, thread_n, sms):
    workspace = torch.zeros(C.shape[1] // 128 * 16, device=torch.device('cuda:0'))
    res = benchmark(lambda: marlin.mul(A, B, C, s, workspace, thread_k, thread_n, sms))
    return {
        's': res,
        'TFLOP/s': 2 * A.numel() * C.shape[1] / res / 10 ** 12,
        'GB/s': (2 * A.numel() + 4 * B.numel() + 2 * C.numel() + 2 * s.numel()) / res / 10 ** 9
    }

# Pass the SM count for known GPUs to avoid the kernel having to query this information (this is very minor)
gpu = torch.cuda.get_device_name(0)
if 'A100' in gpu:
    SMS = 108
elif 'A10' in gpu:
    SMS = 72
elif '3090' in gpu:
    SMS = 82
elif 'A6000' in gpu:
    SMS = 84
else:
    SMS = -1

MODELS = {
    # 'ideal': [
    #     (4 * 256 * SMS, 256 * SMS)
    # ],
    # 'Llama7B': [
    #     (4096, 3 * 4096),
    #     (4096, 4096),
    #     (4096, 2 * 10752),
    #     (10752, 4096)
    # ],
    # 'Llama13B': [
    #     (5120, 3 * 5120),
    #     (5120, 5120),
    #     (5120, 2 * 13568),
    #     (13568, 5120)
    # ],
    # 'Llama33B': [
    #     (6656, 3 * 6656),
    #     (6656, 6656),
    #     (6656, 2 * 17664),
    #     (17664, 6656)
    # ],
    'Qwen1.5-72B': [
        (8192, 3 * 8192),
        (8192, 8192),
        (8192, 2 * 24576),
        (24576, 8192)
    ],
    'Yi-34B': [
        (7168, 9216),
        (7168, 7168),
        (7168, 2 * 20480),
        (20480, 7168)
    ],
    # 'Llama65B': [
    #     (8192, 3 * 8192),
    #     (8192, 8192),
    #     (8192, 2 * 21760),
    #     (21760, 8192)
    # ],
    # 'Falcon180B': [
    #     # Note that parallel attention and FC allows layer fusions
    #     (14848, 14848 * 5 + 1024),
    #     (14848 * 5, 14848)
    # ]
}

# Set to true in order to run a more complete benchmark sweep; the default is reproduce README experiments
ALL = True

# for groupsize in [-1, 128] if ALL else [128]:
for groupsize in [128] if ALL else [128]:
    print('groupsize=%d' % groupsize)
    print()
    for model, layers in MODELS.items():
        print(model)
        if ALL:
            batchsizes =  [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        else:
            batchsizes = [1, 2, 4, 8, 16, 32, 64, 128]
        for batch in batchsizes:
            if not ALL and model != 'ideal' and batch != 16:
                continue
            tot_q = {'s_Marlin': 0,'s_Matmul': 0, 's_nnLinear': 0,'s_FP8': 0,'TFLOP/s': 0, 'GB/s': 0, 'speedup': 0,} 
            for layer in layers:
                A, B, C, B_ref, s = get_problem(batch, layer[1], layer[0], groupsize)
                res_d = benchmark_dense(A, B_ref, C)
                res_nn_linear = benchmark_nnLinear(A, B_ref, C)
                res_fp8 = benchmark_fp8(A, B_ref, C)

                if model == 'ideal' and batch == 16:
                    # This is a special case constructed to be optimal for a thread-shape different than the default one
                    res_q = benchmark_quant(A, B, C, s, 64, 256, SMS)
                else:
                    res_q = benchmark_quant(A, B, C, s, -1, -1, SMS)
                
                tot_q['s_Marlin'] += res_q['s']
                tot_q['s_Matmul'] += res_d['s']
                tot_q['s_nnLinear'] += res_nn_linear['s']
                tot_q['s_FP8'] += res_fp8['s']
            tot_q['speedup'] = tot_q['s_nnLinear'] / tot_q['s_Marlin']
            tot_q['speedup_fp8'] = tot_q['s_nnLinear'] / tot_q['s_FP8']
            print('batch=%04d: s_Matmul=%.5f, s_nnLinear=%.5f, s_Marlin=%.5f, speedup Marlin=%.2f, s_FP8=%.5f, speedup FP8=%.2f' % (
                batch,
                tot_q['s_Matmul'],
                tot_q['s_nnLinear'],
                tot_q['s_Marlin'],
                tot_q['speedup'],
                tot_q['s_FP8'],
                tot_q['speedup_fp8']
            ))
        print()

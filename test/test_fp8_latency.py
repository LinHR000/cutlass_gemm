import random
import time
import torch
import torch.nn as nn
from typing import Tuple
from gemm_op import gemm_op_fp8,fp8_dtype_converter
# import transformer_engine.pytorch as te
# from transformer_engine.common import recipe
from gemm_op import gemm_op_utils
import numpy as np
import marlin

DEV = torch.device('cuda:0')

def gen_quant4(m, n, groupsize=-1):
    tile = 16
    maxq = 2 ** 4 - 1
    w = torch.randn((m, n), dtype=torch.half, device=DEV)
    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq
    w = torch.round(w / s).int()
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)
    ref = (w - (maxq + 1) // 2).half() * s
    if groupsize != -1:
        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((m, n)).contiguous()
            return w
        ref = reshape(ref)
        w = reshape(w)
    s = s.reshape((-1, n)).contiguous()
    linear = nn.Linear(m, n)
    linear.weight.data = ref.t()
    # Workaround to test some special cases that are forbidden by the API
    layer = marlin.Layer(256, 256, groupsize=groupsize)
    if groupsize == -1:
        groupsize = m
    layer.k = m
    layer.n = n
    layer.groupsize = groupsize
    layer.B = torch.empty((m // 16, n * 16 // 8), dtype=torch.int, device=DEV)
    layer.s = torch.empty((m // groupsize, n), dtype=torch.half, device=DEV)
    layer.pack(linear, s.t())
    q = layer.B
    s = layer.s
    return ref, q, s




rtol=1e-3
atol=1e-5
thread_k = -1
thread_n = -1
groupsize = 128

shape_list = [[7168,5120],[5120,5120],[5120, 13824 * 2],[13824, 5120]] # 13B
shape_list = [[9216,7168],[7168,7168],[7168, 20480 * 2],[20480, 5120]] # 34B
# shape_list = [[8192*3,8192],[8192,8192],[8192, 24576 * 2],[24576, 8192]] # 72B
# input_len = [4, 8, 16,32,64,128,256,512,1024,2048,4096,8192]
input_len = [16,32,64,128,256,512,1024,2048,4096,8192]

def test_fp8_latency(
    m=16,
    n=4096,
    k=32,
    dtype=torch.half,
    seed=0,
    device='cuda:0',
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    alpha = 1.0
    a_scale=1.0
    b_scale=1.0
    c_scale=1.0
    d_scale=1.0
    amax_d=468.0
    repeate_num = 3

    # fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.E4M3)

    for m in input_len:
        total_time_torch = 0
        total_time_te = 0
        total_time_fp8 = 0
        total_time_marlin = 0
        for problem_size in shape_list:
            k,n = problem_size[0],problem_size[1]
            input = torch.randn(m,k,dtype=dtype).to(device)
            weight = torch.randn(n,k,dtype=dtype).to(device)

            input_fp8 = torch.zeros_like(input,dtype=torch.int8)
            weight_fp8 = torch.zeros_like(weight,dtype=torch.int8)

            fp8_dtype_converter.fp8_dtype_converter(input,input_fp8,'fp8_e4m3')
            fp8_dtype_converter.fp8_dtype_converter(weight,weight_fp8,'fp8_e4m3')

            A = torch.randn((m, k), dtype=torch.half, device=DEV)
            B_ref, B, s = gen_quant4(k, n, groupsize=groupsize)
            workspace = torch.zeros(n*10 // 128 * 16, device=DEV)


            nn_linear = nn.Linear(k, n, bias=False).half().cuda()
            # te_linear = te.Linear(k, n, bias=True).half().cuda()
            nn_linear.weight.data.copy_(weight)
            # te_linear.weight.data.copy_(weight)
            for i in range(10):
                nn_linear(input)
            time_tmp = 0
            for i in range(repeate_num):
                torch.cuda.synchronize()
                time_s = time.time()
                nn_linear(input)
                torch.cuda.synchronize()
                time_e = time.time()
                time_tmp +=(time_e - time_s) * 1000
            time_tmp /=repeate_num
            total_time_torch +=time_tmp
            if m >= 16:
                for i in range(10):
                    fp8_out = gemm_op_fp8.fp8_gemm(input_fp8,
                                    weight_fp8,
                                    alpha,
                                    a_scale,
                                    b_scale,
                                    c_scale,
                                    d_scale,
                                    amax_d)
                time_tmp = 0
                for i in range(repeate_num):
                    torch.cuda.synchronize()
                    time_s = time.time()
                    fp8_out = gemm_op_fp8.fp8_gemm(input_fp8,
                                    weight_fp8,
                                    alpha,
                                    a_scale,
                                    b_scale,
                                    c_scale,
                                    d_scale,
                                    amax_d)
                    torch.cuda.synchronize()
                    time_e = time.time()
                    time_tmp +=(time_e - time_s) * 1000
                time_tmp /=repeate_num
                total_time_fp8 +=time_tmp

                # for i in range(10):
                #     with torch.no_grad():
                #         with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                #             out_te = te_linear(input)
                # time_tmp = 0
                # for i in range(repeate_num):
                #     torch.cuda.synchronize()
                #     time_s = time.time()
                #     with torch.no_grad():
                #         with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                #             out_te = te_linear(input)
                #     torch.cuda.synchronize()
                #     time_e = time.time()
                #     time_tmp +=(time_e - time_s) * 1000
                # time_tmp /=repeate_num
                # total_time_te +=time_tmp
            # ====================
            # for i in range(10):
            #     marlin.mul(A, B, input, s, workspace, thread_k, thread_n, -1)
            # time_tmp = 0
            # for i in range(repeate_num):
            #     torch.cuda.synchronize()
            #     time_s = time.time()
            #     marlin.mul(A, B, input, s, workspace, thread_k, thread_n, -1)
            #     torch.cuda.synchronize()
            #     time_e = time.time()
            #     time_tmp +=(time_e - time_s) * 1000
            # time_tmp /=repeate_num
            # total_time_marlin +=time_tmp

        print("#"*20 + f"{m}" + "#"*20)
        print(f"torch Linear latency  : {round(total_time_torch,4)}")
        if m>=16:
            print(f"fp8 Linear latency    : {round(total_time_fp8,4)}, speed_up : {round((total_time_torch-total_time_fp8) / total_time_torch * 100, 2)}%")
            # print(f"TE Linear latency     : {round(total_time_te,4)}, speed_up : {round((total_time_torch-total_time_te) / total_time_torch * 100, 2)}%")
        # print(f"Marlin Linear latency : {round(total_time_marlin,4)}, speed_up : {round((total_time_torch-total_time_marlin) / total_time_torch * 100, 2)}%")

test_fp8_latency()



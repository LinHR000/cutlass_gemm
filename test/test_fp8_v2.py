import random

import torch
import torch.nn as nn
from typing import Tuple
from gemm_op import gemm_op_fp8,fp8_dtype_converter,fp8_gemm_v2
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

DTYPES = [torch.half]
M = [16,32,64,128,256,512,1024,2048]
N = [4090]
K = [4090]
SEEDS = [0]
CUDA_DEVICES = [f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)]
rtol=1e-3
atol=1e-5

def test_copy_blocks(
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

    input = torch.randn(m,k,dtype=dtype).to(device)
    weight = torch.randn(n,k,dtype=dtype).to(device)

    input_fp8 = torch.zeros_like(input,dtype=torch.int8)
    weight_fp8 = torch.zeros_like(weight,dtype=torch.int8)

    fp8_dtype_converter.fp8_dtype_converter(input,input_fp8,'fp8_e4m3')
    fp8_dtype_converter.fp8_dtype_converter(weight,weight_fp8,'fp8_e4m3')
    alpha = 1.0
    a_scale=1.0
    b_scale=1.0
    c_scale=1.0
    d_scale=1.0
    amax_d=468.0

    ref_out = torch.matmul(input,weight.t())
    fp8_out = gemm_op_fp8.fp8_gemm(input_fp8,
                                   weight_fp8,
                                   alpha,
                                   a_scale,
                                   b_scale,
                                   c_scale,
                                   d_scale,
                                   amax_d)

    fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.E4M3)
    model = te.Linear(k, n, bias=True).half().cuda()
    model.weight.data.copy_(weight)
    with torch.no_grad():
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            out_te = model(input)
    torch.testing.assert_close(ref_out, fp8_out, rtol=rtol, atol=atol, check_dtype=False)

test_copy_blocks()



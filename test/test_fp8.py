import random

import pytest
import torch
import torch.nn as nn
from typing import Tuple
from gemm_op import gemm_op_fp8,fp8_dtype_converter
DTYPES = [torch.half]
M = [16,32,64,128,256,512,1024,2048]
N = [4090]
K = [4090]
SEEDS = [0]
CUDA_DEVICES = [f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)]
rtol=1e-3, atol=1e-5

@pytest.mark.parametrize("m", M)
@pytest.mark.parametrize("n", N)
@pytest.mark.parametrize("k", K)
@pytest.mark.parametrize("dtyoe", DTYPES)
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_copy_blocks(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
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
    fp8_dtype_converter.fp8_dtype_converter(input,input_fp8,'fp8_e4m3')

    ref_out = torch.matmul(input,weight.t())
    fp8_out = gemm_op_fp8.fp8_gemm(input_fp8,
                                   weight_fp8,
                                   a_scale,
                                   b_scale,
                                   c_scale,
                                   d_scale,
                                   amax_d)
    torch.testing.assert_close(ref_out, fp8_out, rtol=rtol, atol=atol, check_dtype=False)



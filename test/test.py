import torch
from gemm_op import gemm_op_fpAIntB,gemm_op_fp8,fp8_dtype_converter
from gemm_op import gemm_op_fpAIntB
def a():
    m,n,k=16,1024, 4096
    x = torch.randn(m,k).cuda().to(torch.bfloat16)
    w = torch.randn(n,k).cuda().to(torch.bfloat16)
    wq = w.reshape(-1,128)
    max_val = wq.abs().amax(dim=1, keepdim=True)
    scales = max_val / 8
    wqq = (wq / scales).round().clamp(-8,7)
    wqq = wqq.view(w.shape)
    wqq = wqq.t().contiguous()
    from gemm_op import gemm_op_utils
    w_int4 = gemm_op_utils.pack_int8_tensor_to_packed_int4(wqq.char().cpu())
    w_int4 = gemm_op_utils.preprocess_weights_for_mixed_gemm(w_int4,1)
    scales = scales.view(1024,-1)
    from gemm_op import gemm_op_fpAIntB
    output = gemm_op_fpAIntB.fpAIntB_gemm(x,
                                w_int4, 
                                scales,
                                None,# bias
                                None,#out
                                None,#weight_zero_points
                                None,#alpha
                                128,# group_size
                                None,#tile config
                                None,# split_k_style
                                None,# split_k_factor
                                None) # stages
    print(output)

a()
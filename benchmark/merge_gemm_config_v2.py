import json
merge_dict = {}
total_dict = {}
with open("/mnt/infra/haoran.lin2/cutlass_gemm/benchmark/gemm_per_tensor_13b_fc1_1gpu.json",'r') as r:
    data_dict_fc1 = json.load(r)
with open("/mnt/infra/haoran.lin2/cutlass_gemm/benchmark/gemm_per_tensor_13b_fc2_1gpu.json",'r') as r:
    data_dict_fc2 = json.load(r)
with open("/mnt/infra/haoran.lin2/cutlass_gemm/benchmark/gemm_per_tensor_13b_proj_1gpu.json",'r') as r:
    data_dict_proj = json.load(r)
with open("/mnt/infra/haoran.lin2/cutlass_gemm/benchmark/gemm_per_tensor_13b_qkv_1gpu.json",'r') as r:
    data_dict_qkv = json.load(r)
data_dict = {}
for key,value in data_dict_fc1.items():
    tmp_dict = {}
    for m,cost_time in value.items():
        tmp_dict[m] = cost_time + data_dict_fc2[key][m]+data_dict_proj[key][m]+data_dict_qkv[key][m]
    data_dict[key] = tmp_dict
sorted_dict = {}
for key,value in data_dict.items():
    value = dict(sorted(value.items(), key=lambda item: item[1], reverse=False))
    sorted_dict[key] = list(value.keys())[0]

total_dict['fc1'] = sorted_dict
total_dict['fc2'] = sorted_dict
total_dict['proj'] = sorted_dict
total_dict['qkv'] = sorted_dict

total_dict_2 = {}
with open("/mnt/infra/haoran.lin2/cutlass_gemm/benchmark/gemm_wt_13b_fc1_1gpu_v2.json",'r') as r:
    data_dict_fc1 = json.load(r)
with open("/mnt/infra/haoran.lin2/cutlass_gemm/benchmark/gemm_wt_13b_fc2_1gpu_v2.json",'r') as r:
    data_dict_fc2 = json.load(r)
with open("/mnt/infra/haoran.lin2/cutlass_gemm/benchmark/gemm_wt_13b_proj_1gpu_v2.json",'r') as r:
    data_dict_proj = json.load(r)
with open("/mnt/infra/haoran.lin2/cutlass_gemm/benchmark/gemm_wt_13b_qkv_1gpu_v2.json",'r') as r:
    data_dict_qkv = json.load(r)
data_dict = {}
for key,value in data_dict_fc1.items():
    tmp_dict = {}
    for m,cost_time in value.items():
        tmp_dict[m] = cost_time + data_dict_fc2[key][m]+data_dict_proj[key][m]+data_dict_qkv[key][m]
    data_dict[key] = tmp_dict
sorted_dict = {}
for key,value in data_dict.items():
    value = dict(sorted(value.items(), key=lambda item: item[1], reverse=False))
    sorted_dict[key] = list(value.keys())[0]

total_dict_2['fc1'] = sorted_dict
total_dict_2['fc2'] = sorted_dict
total_dict_2['proj'] = sorted_dict
total_dict_2['qkv'] = sorted_dict


total_dict_3 = {}
with open("/mnt/infra/haoran.lin2/cutlass_gemm/benchmark/gemm_ptpc_13b_fc1_1gpu.json",'r') as r:
    data_dict_fc1 = json.load(r)
with open("/mnt/infra/haoran.lin2/cutlass_gemm/benchmark/gemm_ptpc_13b_fc2_1gpu.json",'r') as r:
    data_dict_fc2 = json.load(r)
with open("/mnt/infra/haoran.lin2/cutlass_gemm/benchmark/gemm_ptpc_13b_proj_1gpu.json",'r') as r:
    data_dict_proj = json.load(r)
with open("/mnt/infra/haoran.lin2/cutlass_gemm/benchmark/gemm_ptpc_13b_qkv_1gpu.json",'r') as r:
    data_dict_qkv = json.load(r)
data_dict = {}
for key,value in data_dict_fc1.items():
    tmp_dict = {}
    for m,cost_time in value.items():
        tmp_dict[m] = cost_time + data_dict_fc2[key][m]+data_dict_proj[key][m]+data_dict_qkv[key][m]
    data_dict[key] = tmp_dict
sorted_dict = {}
for key,value in data_dict.items():
    value = dict(sorted(value.items(), key=lambda item: item[1], reverse=False))
    sorted_dict[key] = list(value.keys())[0]

total_dict_3['fc1'] = sorted_dict
total_dict_3['fc2'] = sorted_dict
total_dict_3['proj'] = sorted_dict
total_dict_3['qkv'] = sorted_dict

merge_dict['per_tensor'] = total_dict
merge_dict['weight_only'] = total_dict_2
merge_dict['ptpc'] = total_dict_3
with open("/mnt/infra/haoran.lin2/cutlass_gemm/benchmark/gemm_13b_1gpu_4fuse1.json",'w') as w:
    json.dump(merge_dict,w,indent=4)



import json
# with open("/mnt/infra/haoran.lin2/cutlass_gemm/benchmark/gemm_wt1.json",'r') as r:
#     data = json.load(r)
# with open("/mnt/infra/haoran.lin2/cutlass_gemm/benchmark/gemm_wt2.json",'r') as r:
#     data2 = json.load(r)
# data.update(data2)

# sorted_dict = {}
# for key,value in data.items():
#     value = dict(sorted(value.items(), key=lambda item: item[1], reverse=False))
#     config = list(value.keys())[0]
#     config = config.split(" ")
#     config = [int(i) for i in config]
#     sorted_dict[key] = config
# with open("/mnt/infra/haoran.lin2/cutlass_gemm/output/gemm_best_weight_only.json",'w') as w:
#     json.dump(sorted_dict,w,indent=4)
# with open(file_path,'w') as w:
    # json.dump(sorted_dict_org,w,indent=4)
with open("/mnt/infra/haoran.lin2/cutlass_gemm/output/gemm_best_weight_only.json",'r') as r:
    data = json.load(r)
print()



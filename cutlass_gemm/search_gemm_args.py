import torch
from cutlass_gemm import gemm_op
from tqdm import tqdm
import time
import json

class GemmSearch(object):
    @staticmethod
    def gemm_search(max_m,n,k,iters=3,output_path = 'gemm_stat.json'):
        gemm_stat_dict = {}
        config_set = GemmSearch.get_gemm_config_set(mode='in8_w8_ofp16')

        weight = torch.randint(-127, 127, (n,k),dtype=torch.int8).cuda()
        total_time = 0
        for m in tqdm(range(1,max_m,15)):
            tmp_dict = {}
            input = torch.randint(-127, 127, (m,k),dtype=torch.int8).cuda()
            for stages in range(2,5,1):
                for splitk in range(1,5,1):
                    for config in config_set:
                        total_time = 0
                        try:
                            gemm_op.gemm_in8_w8_ofp16_per_tensor(input, weight, 1.0, 0.0, m, n, k, config, stages, splitk) 
                            
                            for iter in range(iters):
                                torch.cuda.synchronize()
                                time_start = time.time()
                                gemm_op.gemm_in8_w8_ofp16_per_tensor(input, weight, 1.0, 0.0, m, n, k, config, stages, splitk)  
                                torch.cuda.synchronize()
                                time_end = time.time()
                                total_time += time_end - time_start
                            key = config+"#"+str(stages) + "#" + str(splitk)
                            tmp_dict[key] = total_time * 1000 / iters
                        except Exception as e:
                            # print(e)
                            continue  
            gemm_stat_dict[m] = tmp_dict

        with open(output_path,'w') as w:
            json.dump(gemm_stat_dict,w,indent=4)
        pass

    # @staticmethod
    # def get_gemm_param(config):
    #     if config == 'CtaShape128x256x128_WarpShape64x64x128':
    #         return None
    #     elif config == 'CtaShape256x128x128_WarpShape64x64x128':
    #         return None
    #     elif config == 'CtaShape128x128x128_WarpShape64x64x128':
    #         return None
    #     elif config == 'CtaShape256x64x128_WarpShape64x64x128':
    #         return None
    #     elif config == 'CtaShape64x256x128_WarpShape64x64x128':
    #         return None
    #     elif config == 'CtaShape64x128x128_WarpShape32x64x128':
    #         pass
    #     elif config == 'CtaShape128x64x128_WarpShape64x32x128':
    #         pass
    #     elif config == 'CtaShape64x64x128_WarpShape32x32x128':
    #         pass
    #     elif config == 'CtaShape128x256x64_WarpShape64x64x64':
    #         pass
    #     elif config == 'CtaShape256x128x64_WarpShape64x64x64':
    #         pass
    #     elif config == 'CtaShape128x128x64_WarpShape64x64x64':
    #         pass
    #     elif config == 'CtaShape256x64x64_WarpShape64x64x64':
    #         pass
    #     elif config == 'CtaShape64x256x64_WarpShape64x64x64':
    #         pass
    #     elif config == 'CtaShape64x128x64_WarpShape32x64x64':
    #         pass
    #     elif config == 'CtaShape128x64x64_WarpShape64x32x64':
    #         pass
    #     elif config == 'CtaShape64x64x64_WarpShape32x32x64':
    #         pass
    #     pass
    @staticmethod
    def get_gemm_config_set(mode='in8_w8_ofp16'):
        if mode == 'in8_w8_ofp16':
            return ['CtaShape128x256x128_WarpShape64x64x128',
                    'CtaShape128x128x128_WarpShape64x64x128',
                    'CtaShape64x256x128_WarpShape64x64x128',
                    'CtaShape64x128x128_WarpShape32x64x128',
                    'CtaShape256x128x64_WarpShape64x64x64',
                    'CtaShape128x128x64_WarpShape64x64x64',
                    'CtaShape64x256x64_WarpShape64x64x64',
                    'CtaShape64x128x64_WarpShape32x64x64']
        elif mode == 'in8_w8_o32':
            return ['CtaShape128x256x128_WarpShape64x64x128',
                    'CtaShape256x128x128_WarpShape64x64x128',
                    'CtaShape128x128x128_WarpShape64x64x128',
                    'CtaShape256x64x128_WarpShape64x64x128',
                    'CtaShape64x256x128_WarpShape64x64x128',
                    'CtaShape64x128x128_WarpShape32x64x128',
                    'CtaShape128x64x128_WarpShape64x32x128',
                    'CtaShape64x64x128_WarpShape32x32x128',
                    'CtaShape128x256x64_WarpShape64x64x64',
                    'CtaShape256x128x64_WarpShape64x64x64',
                    'CtaShape128x128x64_WarpShape64x64x64',
                    'CtaShape256x64x64_WarpShape64x64x64',
                    'CtaShape64x256x64_WarpShape64x64x64',
                    'CtaShape64x128x64_WarpShape32x64x64',
                    'CtaShape128x64x64_WarpShape64x32x64',
                    'CtaShape64x64x64_WarpShape32x32x64']
        elif mode == 'in8_w8_o8':
            return ['CtaShape128x256x128_WarpShape64x64x128',
                    'CtaShape256x128x128_WarpShape64x64x128',
                    'CtaShape128x128x128_WarpShape64x64x128',
                    'CtaShape256x64x128_WarpShape64x64x128',
                    'CtaShape64x256x128_WarpShape64x64x128',
                    'CtaShape64x128x128_WarpShape32x64x128',
                    'CtaShape128x256x64_WarpShape64x64x64',
                    'CtaShape256x128x64_WarpShape64x64x64',
                    'CtaShape128x128x64_WarpShape64x64x64',
                    'CtaShape256x64x64_WarpShape64x64x64',
                    'CtaShape64x256x64_WarpShape64x64x64',
                    'CtaShape64x128x64_WarpShape32x64x64']
        else:
            raise ValueError()

    @staticmethod
    def get_best_config(file_path):
        with open(file_path,'r') as r:
            data_dict = json.load(r)
        sorted_dict = {}
        for key,value in data_dict.items():
            value = dict(sorted(value.items(), key=lambda item: item[1], reverse=False))
            sorted_dict[key] = list(value.keys())[:3]
        with open("/mnt/infra/haoran.lin2/cutlass_gemm/output/gemm_best.json",'w') as w:
            json.dump(sorted_dict,w,indent=4)
        result_dict = []
        for i in range(3):
            # 需要考虑长度惩罚，排序惩罚。
            pass

        pass

if __name__ == "__main__":
    # GemmSearch.gemm_search(4096, int(8192/4), 8192, iters=5, output_path='/mnt/infra/haoran.lin2/cutlass_gemm/output/gemm_stat.json')
    GemmSearch.get_best_config('/mnt/infra/haoran.lin2/cutlass_gemm/output/gemm_stat.json')

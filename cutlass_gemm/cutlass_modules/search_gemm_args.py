import torch
from tqdm import tqdm
import time
import json
from cutlass_gemm import gemm_op,gemm_op_int8
class GemmSearch(object):
    @staticmethod
    def gemm_search(max_m,n,k,iters=3,output_path = 'gemm_stat_splitk.json'):
        gemm_stat_dict = {}
        config_set = GemmSearch.get_gemm_config_set(mode='in8_w8_ofp16')

        weight = torch.randint(-1, 1, (n,k),dtype=torch.int8).cuda()
        bias = torch.randn(n,dtype=torch.float16).cuda()
        total_time = 0
        for m in tqdm(range(8,8192,8)):
            tmp_dict = {}
            input = torch.randint(-1, 1, (m,k),dtype=torch.int8).cuda()
            for stages in range(2,5,1):
                for splitk in range(1,5,1):
                    for config in config_set:
                        total_time = 0
                        try:
                            output = gemm_op_int8.gemm_in8_w8_ofp16_per_tensor(input, weight,bias, 1.0, 0.0, m, n, k, config, stages, splitk) 
                            gemm_op_int8.gemm_in8_w8_ofp16_per_tensor(input, weight, bias,1.0, 0.0, m, n, k, config, stages, splitk) 

                            
                            for iter in range(iters):
                                torch.cuda.synchronize()
                                time_start = time.time()
                                gemm_op_int8.gemm_in8_w8_ofp16_per_tensor(input, weight,bias,1.0, 0.0, m, n, k, config, stages, splitk) 
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
        sorted_dict_org = {}
        for key,value in data_dict.items():
            value = dict(sorted(value.items(), key=lambda item: item[1], reverse=False))
            sorted_dict[key] = list(value.keys())[0]
            sorted_dict_org[key] = value
        with open("/mnt/infra/haoran.lin2/cutlass_gemm/output/gemm_best_splitk_choose.json",'w') as w:
            json.dump(sorted_dict,w,indent=4)
        # with open(file_path,'w') as w:
            # json.dump(sorted_dict_org,w,indent=4)
        result_dict = []
        for i in range(3):
            # 需要考虑长度惩罚，排序惩罚。
            pass

        pass


class CutlassGemm(object):
    def __init__(self,config_path=None):
        self.config_dict = {}
        if config_path is not None:
            with open(config_path,'r') as r:
                data_total = json.load(r)
            for k_num,data in data_total.items():
                before_idx = 1
                tmp_dict = {}
                for key,value in data.items():
                    value_split = value.split("#")
                    tile_config,stages,splitk = value_split[0],int(value_split[1]),int(value_split[2])
                    for i in range(before_idx,int(key)+1):
                        tmp_dict[i] = {"tile_config":tile_config,"stages":stages,'splitk':splitk}
                    before_idx = int(key) + 1
                    self.max_index = before_idx
                self.config_dict[int(k_num)] = tmp_dict
                
        pass

    def get_config(self, m,n,k):
        # if k == 2048:
        #     if m in self.config_dict[2048]:
        #         return self.config_dict[2048][m]
        #     else:
        #         return self.config_dict[2048][self.max_index]
        # else:
        #     if n == 6144:
        #         if m in self.config_dict[6144]:
        #             return self.config_dict[6144][m]
        #         else:
        #             return self.config_dict[6144][self.max_index]
        #     elif n == 8192:
        #         if m in self.config_dict[8192]:
        #             return self.config_dict[8192][m]
        #         else:
        #             return self.config_dict[8192][self.max_index]
        if m in self.config_dict[8192]:
            return self.config_dict[8192][m]
        else:
            return self.config_dict[8192][self.max_index]
        # if m in self.config_dict:
        #     return self.config_dict[m]
        # else:
        #     return self.config_dict[self.max_index]

    def gemm_in8_w8_ofp16_per_tensor(self, input,weight,bias, alpha,beta,m,n,k,tile_config = '',stages=3,splitk=1):
        if len(self.config_dict) >0 :
            # config_dict_ = self.config_dict[self.max_index if m > self.max_index else m]
            time_start = time.time()
            config_dict_ = self.get_config(m, n, k)
            time_end = time.time()
            print((time_end-time_start)*1000)
            time_start = time.time()
            tile_config,stages,splitk = config_dict_['tile_config'],config_dict_['stages'],config_dict_['splitk']
            time_end = time.time()
            print((time_end-time_start)*1000)
        torch.cuda.synchronize()
        time_start = time.time()
        output = gemm_op_int8.gemm_in8_w8_ofp16_per_tensor(input,           # input
                                                            weight,          # weight
                                                            bias,
                                                            alpha,             # alpha
                                                            beta,             # beta
                                                            m,               # m
                                                            n,               # n
                                                            k,               # k
                                                            tile_config,     # tile config
                                                            stages,               # stages
                                                            splitk)               # workspeace bytes
        torch.cuda.synchronize()
        time_end = time.time()
        print((time_end-time_start)*1000)
        return output

    def gemm_in8_w8_obf16_per_tensor(self, input,weight,bias, alpha,beta,m,n,k,tile_config = '',stages=3,splitk=1):
        if len(self.config_dict) >0 :
            config_dict_ = self.get_config(m, n, k)
            # config_dict_ = self.config_dict[self.max_index if m > self.max_index else m]
            tile_config,stages,splitk = config_dict_['tile_config'],config_dict_['stages'],config_dict_['splitk']
        output = gemm_op_int8.gemm_in8_w8_obf16_per_tensor(input,           # input
                                                            weight,          # weight
                                                            bias,
                                                            alpha,             # alpha
                                                            beta,             # beta
                                                            m,               # m
                                                            n,               # n
                                                            k,               # k
                                                            tile_config,     # tile config
                                                            stages,               # stages
                                                            splitk)               # workspeace bytes
        return output

    # def gemm_in8_w8_ofp16_gelu_per_tensor(self, input,weight,bias, alpha,beta,m,n,k,tile_config = '',stages=3,splitk=1):
    #     if len(self.config_dict) >0 :
    #         config_dict_ = self.config_dict[m]
    #         tile_config,stages,splitk = config_dict_['tile_config'],config_dict_['stages'],config_dict_['splitk']
    #     output = gemm_op_int8.gemm_in8_w8_ofp16_gelu_per_tensor(input,           # input
    #                                                         weight,          # weight
    #                                                         bias,
    #                                                         alpha,             # alpha
    #                                                         beta,             # beta
    #                                                         m,               # m
    #                                                         n,               # n
    #                                                         k,               # k
    #                                                         tile_config,     # tile config
    #                                                         stages,               # stages
    #                                                         splitk)               # workspeace bytes
    #     return output

    def gemm_in8_w8_ofp16_per_tensor_splitk(self, input,weight,alpha,beta,m,n,k,tile_config = '',stages=3,splitk=1):
        if len(self.config_dict) >0 :
            # config_dict_ = self.config_dict[m]
            config_dict_ = self.get_config(m, n, k)
            tile_config,stages,splitk = config_dict_['tile_config'],config_dict_['stages'],config_dict_['splitk']
        output = gemm_op_int8.gemm_in8_w8_ofp16_per_tensor_splitk(input,           # input
                                                                    weight,          # weight
                                                                    alpha,             # alpha
                                                                    beta,             # beta
                                                                    m,               # m
                                                                    n,               # n
                                                                    k,               # k
                                                                    tile_config,     # tile config
                                                                    stages,               # stages
                                                                    splitk) 
        return output
    def gemm_in8_w8_ofp16_pt(self, input,weight,alpha_col,alpha_row,m,n,k):
        output = gemm_op.gemm_in8_w8_ofp16_pt(input,weight,alpha_col,alpha_row,m,n,k)
        return output
        
    def gemm_in8_w8_ofp16_pc(self, input,weight,alpha_col,alpha_row,m,n,k):
        output = gemm_op.gemm_in8_w8_ofp16_pc(input,weight,alpha_col,alpha_row,m,n,k)
        return output
    
    def gemm_in8_w8_ofp16_ptpc(self, input,weight,alpha_col,alpha_row,m,n,k):
        output = gemm_op.gemm_in8_w8_ofp16_ptpc(input,weight,alpha_col,alpha_row,m,n,k)
        return output
    
    def gemm_infp16_w8_ofp16(self, input,weight,weight_scale):
        output = gemm_op.gemm_infp16_w8_ofp16(input,weight,weight_scale)
        return output
    
    def gemm_infp16_w8_ofp16_bias_act(self, input,weight,weight_scale,bias_fp16,act_func='gelu'):
        output = gemm_op.gemm_infp16_w8_ofp16_bias_act(input,weight,weight_scale,bias_fp16,act_func)
        return output

    def symmetric_quantize(self, weight,quant_mode=0):
        """
        0 fot int8
        1 for int4
        """
        _,weight_mix,weight_mix_scale = gemm_op.symmetric_quantize_last_axis_of_batched_matrix(weight,quant_mode)
        return weight_mix,weight_mix_scale



if __name__ == "__main__":
    # GemmSearch.gemm_search(8192, int(8192), int(8192 /4), iters=5, output_path='/mnt/infra/haoran.lin2/cutlass_gemm/output/gemm_stat_76b_o_proj.json')
    # GemmSearch.get_best_config('/mnt/infra/haoran.lin2/cutlass_gemm/output/gemm_stat_splitk.json')
    with open('/mnt/infra/haoran.lin2/cutlass_gemm/output/gemm_stat_76b_o_proj.json','r') as r:
            data_dict_1 = json.load(r)
    with open('/mnt/infra/haoran.lin2/cutlass_gemm/output/gemm_stat_76b_qkv.json','r') as r:
            data_dict_2 = json.load(r)
    with open('/mnt/infra/haoran.lin2/cutlass_gemm/output/gemm_stat_76b_mlp.json','r') as r:
            data_dict_3 = json.load(r)

    sorted_dict_1 = {}
    sorted_dict_2 = {}
    sorted_dict_3 = {}

    for key,value in data_dict_1.items():
        value = dict(sorted(value.items(), key=lambda item: item[1], reverse=False))
        sorted_dict_1[key] = list(value.keys())[0]
    for key,value in data_dict_2.items():
        value = dict(sorted(value.items(), key=lambda item: item[1], reverse=False))
        sorted_dict_2[key] = list(value.keys())[0]
    for key,value in data_dict_3.items():
        value = dict(sorted(value.items(), key=lambda item: item[1], reverse=False))
        sorted_dict_3[key] = list(value.keys())[0]
    
    best_dict = {8192:sorted_dict_3,int(8192 * 3 /4):sorted_dict_2,int(8192/4):sorted_dict_1}

    with open("/mnt/infra/haoran.lin2/cutlass_gemm/output/gemm_best_76b_choose.json",'w') as w:
        json.dump(best_dict,w,indent=4)

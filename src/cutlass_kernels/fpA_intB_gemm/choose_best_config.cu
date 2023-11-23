#include "cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h"
#include <iostream>
#include <vector>
#include "cutlass_extensions/gemm/kernel/fpA_intB_gemm.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

#include "cutlass/gemm/device/gemm_universal_base.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass_extensions/compute_occupancy.h"

#include "cutlass_extensions/epilogue_helpers.h"
#include "cutlass_extensions/ft_gemm_configs.h"
#include "cutlass_extensions/gemm/kernel/default_fpA_intB_traits.h"
#include "cutlass_extensions/gemm/kernel/fpA_intB_gemm.h"
#include "cutlass_extensions/gemm/threadblock/default_mma.h"

#pragma GCC diagnostic pop

#include "cutlass_kernels/cutlass_heuristic.h"
#include "cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "utils/cuda_utils.h"
#include <cublas_v2.h>
namespace fastertransformer {

std::vector<int> choose_best_config(const half*          A,
                                    const uint8_t* B,
                                    const half*          weight_scales,
                                    const half*          biases,
                                    half*                C,
                                    int               m,
                                    int               n,
                                    int               k,
                                    char*             workspace_ptr,
                                    const size_t      workspace_bytes,
                                    cudaStream_t      stream)
{

    static constexpr bool          is_weight_only    = true;
    std::vector<CutlassGemmConfig> candidate_configs = get_candidate_configs(80, is_weight_only, false);
    std::vector<int>               occupancies(candidate_configs.size());

    for (size_t ii = 0; ii < candidate_configs.size(); ++ii) {
        dispatch_gemm_to_cutlass<half, uint8_t, cutlass::arch::Sm80, EpilogueOpNoBias>(
            A, B, weight_scales, biases, C, m, n, k, workspace_ptr, workspace_bytes, candidate_configs[ii], stream, &occupancies[ii]);
    }
    // Standard GEMM, so 1 "expert". We use the same function for MoE and regular FFN.
    static constexpr int num_experts   = 1;
    int multi_processor_count = get_multi_processor_count();
    CutlassGemmConfig    chosen_config = estimate_best_config_from_occupancies(candidate_configs,
                                                                            occupancies,
                                                                            m,
                                                                            n,
                                                                            k,
                                                                            num_experts,
                                                                            7,
                                                                            workspace_bytes,
                                                                            multi_processor_count,
                                                                            is_weight_only);
    return {int(chosen_config.tile_config),int(chosen_config.split_k_style),chosen_config.split_k_factor,chosen_config.stages};

} 
}

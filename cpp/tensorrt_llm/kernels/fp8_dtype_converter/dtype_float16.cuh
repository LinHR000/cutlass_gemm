/*
 * Adapted from https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * and https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention_utils.h
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "attention_generic.cuh"
#ifdef USE_ROCM
  #include <hip/hip_fp16.h>
#endif

#include <stdint.h>


// FP16 vector types for Q, K, V.
template<>
struct Vec<uint16_t, 1> {
  using Type = uint16_t;
};
template<>
struct Vec<uint16_t, 2> {
  using Type = uint32_t;
};
template<>
struct Vec<uint16_t, 4> {
  using Type = uint2;
};
template<>
struct Vec<uint16_t, 8> {
  using Type = uint4;
};

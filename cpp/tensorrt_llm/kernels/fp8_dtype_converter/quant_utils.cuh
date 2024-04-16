#pragma once

#include <assert.h>
#include <stdint.h>
#include <float.h>
#include <type_traits>
#include "dtype_float16.cuh"
#include <cuda_fp8.h>
#include <string.h>



namespace fp8_e5m2_unscaled {
constexpr int FP8_E4M3 = 0;
constexpr int FP8_E5M2 = 1;

template<typename Tout, typename Tin, int DType>
__inline__ __device__ Tout vec_conversion(const Tin& x)
{
    return x;
}

// fp8 -> half
template<>
__inline__ __device__ uint16_t vec_conversion<uint16_t, uint8_t, FP8_E5M2>(const uint8_t& a)
{
    __half_raw res = __nv_cvt_fp8_to_halfraw(a, __NV_E5M2); 
    return res.x;

    
}

template<>
__inline__ __device__ uint16_t vec_conversion<uint16_t, uint8_t, FP8_E4M3>(const uint8_t& a)
{
    __half_raw res = __nv_cvt_fp8_to_halfraw(a, __NV_E4M3); 
    return res.x;
    
}

// fp8x2 -> half2
template<>
__inline__ __device__ uint32_t vec_conversion<uint32_t, uint16_t,FP8_E5M2>(const uint16_t& a)
{
    union {
        uint16_t u16[2];
        uint32_t u32;
    } tmp;
    __half2_raw res = __nv_cvt_fp8x2_to_halfraw2(a, __NV_E5M2);
    tmp.u16[0] = res.x;
    tmp.u16[1] = res.y;
    return tmp.u32;
}

template<>
__inline__ __device__ uint32_t vec_conversion<uint32_t, uint16_t,FP8_E4M3>(const uint16_t& a)
{
    union {
        uint16_t u16[2];
        uint32_t u32;
    } tmp;
    __half2_raw res = __nv_cvt_fp8x2_to_halfraw2(a, __NV_E4M3);
    tmp.u16[0] = res.x;
    tmp.u16[1] = res.y;
    return tmp.u32;
}

// fp8x4 -> half2x2
template<>
__inline__ __device__ uint2 vec_conversion<uint2, uint32_t,FP8_E5M2>(const uint32_t& a)
{
    union {
        uint2    u32x2;
        uint32_t u32[2];
    } tmp;
    tmp.u32[0] = vec_conversion<uint32_t, uint16_t,FP8_E5M2>((uint16_t)a);
    tmp.u32[1] = vec_conversion<uint32_t, uint16_t,FP8_E5M2>((uint16_t)(a >> 16U));
    return tmp.u32x2;
    
}

template<>
__inline__ __device__ uint2 vec_conversion<uint2, uint32_t,FP8_E4M3>(const uint32_t& a)
{
    union {
        uint2    u32x2;
        uint32_t u32[2];
    } tmp;
    tmp.u32[0] = vec_conversion<uint32_t, uint16_t,FP8_E4M3>((uint16_t)a);
    tmp.u32[1] = vec_conversion<uint32_t, uint16_t,FP8_E4M3>((uint16_t)(a >> 16U));
    return tmp.u32x2;
    
}
// fp8x8 -> half2x4
template<>
__inline__ __device__ uint4 vec_conversion<uint4, uint2, FP8_E5M2>(const uint2& a)
{
    union {
        uint4 u64x2;
        uint2 u64[2];
    } tmp;
    tmp.u64[0] = vec_conversion<uint2, uint32_t,FP8_E5M2>(a.x);
    tmp.u64[1] = vec_conversion<uint2, uint32_t,FP8_E5M2>(a.y);
    return tmp.u64x2;
}

template<>
__inline__ __device__ uint4 vec_conversion<uint4, uint2, FP8_E4M3>(const uint2& a)
{
    union {
        uint4 u64x2;
        uint2 u64[2];
    } tmp;
    tmp.u64[0] = vec_conversion<uint2, uint32_t,FP8_E4M3>(a.x);
    tmp.u64[1] = vec_conversion<uint2, uint32_t,FP8_E4M3>(a.y);
    return tmp.u64x2;
}


// half -> fp8
template<>
__inline__ __device__ uint8_t vec_conversion<uint8_t, uint16_t,FP8_E5M2>(const uint16_t& a)
{
    __half_raw tmp;
    tmp.x = a;
    __nv_fp8_storage_t res = __nv_cvt_halfraw_to_fp8(tmp, __NV_SATFINITE, __NV_E5M2);
    return (uint8_t)res;

}

template<>
__inline__ __device__ uint8_t vec_conversion<uint8_t, uint16_t,FP8_E4M3>(const uint16_t& a)
{
    __half_raw tmp;
    tmp.x = a;
    __nv_fp8_storage_t res = __nv_cvt_halfraw_to_fp8(tmp, __NV_SATFINITE, __NV_E4M3);
    return (uint8_t)res;

}

} // namespace fp8_e5m2_unscaled

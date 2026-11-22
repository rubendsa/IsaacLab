// Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// Device helpers shared by the SRAM-resident megakernels (no CUDA headers needed; NVRTC only).
#pragma once
#include "device_lib.cuh"

#ifndef FULL
#define FULL 0xffffffffu
#endif
#define LOG_SQRT_2PI 0.91893853320467274178f   // log(sqrt(2 pi))
#define PI_F 3.14159265358979323846f

// --------------------------------------------------------------------------------------------
// counter-based RNG (Philox4x32-10) -> uniforms / normals; streams are keyed by
// (seed) and indexed by (global env or transition id, iteration, step, purpose)
// --------------------------------------------------------------------------------------------
__device__ __forceinline__ uint4 philox(uint4 c, u32 k0, u32 k1) {
#pragma unroll
    for (int r = 0; r < 10; ++r) {
        const u32 hi0 = __umulhi(0xD2511F53u, c.x), lo0 = 0xD2511F53u * c.x;
        const u32 hi1 = __umulhi(0xCD9E8D57u, c.z), lo1 = 0xCD9E8D57u * c.z;
        c = make_uint4(hi1 ^ c.y ^ k0, lo1, hi0 ^ c.w ^ k1, lo0);
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return c;
}
__device__ __forceinline__ float u01(u32 x) { return (float)(x >> 8) * (1.0f / 16777216.0f); }        // [0,1)
__device__ __forceinline__ float u01_open(u32 x) { return ((float)(x >> 8) + 1.0f) * (1.0f / 16777216.0f); } // (0,1]
__device__ __forceinline__ float normal01(u32 a, u32 b) {
    const float r = sqrtf(-2.0f * logf(u01_open(a)));
    return r * cospif(2.0f * u01(b));
}

__device__ __forceinline__ float elu(float z) { return z > 0.0f ? z : expm1f(z); }
// ELU derivative expressed through the activation value: a>0 <=> z>0; for z<=0, a=exp(z)-1
__device__ __forceinline__ float delu_from_act(float a) { return a > 0.0f ? 1.0f : a + 1.0f; }

__device__ __forceinline__ float warp_sum(float v) {
    v += __shfl_xor_sync(FULL, v, 16);
    v += __shfl_xor_sync(FULL, v, 8);
    v += __shfl_xor_sync(FULL, v, 4);
    v += __shfl_xor_sync(FULL, v, 2);
    v += __shfl_xor_sync(FULL, v, 1);
    return v;
}
__device__ __forceinline__ double warp_sum_d(double v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}
// Reduce 8 per-lane values over the 32 lanes of a warp with a butterfly reduce-scatter
// (9 shuffles instead of 8 x 5).  Lane L returns the full sum of value index (L >> 2) & 7.
__device__ __forceinline__ float reduce8(const float v[8], int lane) {
    const bool b4 = lane & 16, b3 = lane & 8, b2 = lane & 4;
    float k4[4], k2[2];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float send = b4 ? v[j] : v[j + 4];
        const float keep = b4 ? v[j + 4] : v[j];
        k4[j] = keep + __shfl_xor_sync(FULL, send, 16);
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const float send = b3 ? k4[j] : k4[j + 2];
        const float keep = b3 ? k4[j + 2] : k4[j];
        k2[j] = keep + __shfl_xor_sync(FULL, send, 8);
    }
    float k1 = (b2 ? k2[1] : k2[0]) + __shfl_xor_sync(FULL, b2 ? k2[0] : k2[1], 4);
    k1 += __shfl_xor_sync(FULL, k1, 2);
    k1 += __shfl_xor_sync(FULL, k1, 1);
    return k1;
}

// Stateless keyed pseudo-random permutation of [0, n): a 4-round Feistel network on
// [0, 2^(2*half)) with cycle walking (Black & Rogaway).  Used for mini-batch shuffling: O(1) per
// index, no shared memory, a bijection for every n.
__device__ __forceinline__ u32 lowbias32(u32 x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x;
}
__device__ __forceinline__ u32 feistel_perm(u32 i, u32 n, u32 half, uint4 k) {
    const u32 mask = (1u << half) - 1u;
    u32 x = i;
    do {
        u32 L = x >> half, R = x & mask, t;
        t = L ^ (lowbias32(R ^ k.x) & mask); L = R; R = t;
        t = L ^ (lowbias32(R ^ k.y) & mask); L = R; R = t;
        t = L ^ (lowbias32(R ^ k.z) & mask); L = R; R = t;
        t = L ^ (lowbias32(R ^ k.w) & mask); L = R; R = t;
        x = (L << half) | R;
    } while (x >= n);
    return x;
}

// Isaac Lab wrap_to_pi (torch float remainder semantics)
__device__ __forceinline__ float wrap_to_pi(float a) {
    const float twopi = 2.0f * PI_F;
    float w = fmodf(a + PI_F, twopi);
    if (w < 0.0f) w += twopi;
    return (w == 0.0f && a > 0.0f) ? PI_F : w - PI_F;
}

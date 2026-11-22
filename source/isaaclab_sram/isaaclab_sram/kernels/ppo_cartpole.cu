// Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// SRAM-resident PPO megakernel for the Isaac-Cartpole-Direct MDP.
//
// ONE persistent, cooperative kernel runs the complete RSL-RL PPO training loop:
//   rollout (policy + value MLPs, Gaussian sampling, physics, reward, termination, resets)
//   -> GAE -> global advantage normalisation -> num_epochs x num_mini_batches updates
//   (forward, analytic backward, gradient all-reduce, KL-adaptive learning rate,
//   per-model grad-norm clipping, Adam) -> next iteration,
// without returning to the host and without kernel boundaries.
//
// Where the data lives (RTX PRO 6000 Blackwell: 188 SMs, 256 KB registers + 128 KB L1/SMEM per SM,
// 128 MB L2):
//   * every CTA (one per SM) owns a contiguous slice of environments; their physics state, the
//     rollout buffer of those environments and the mini-batch permutation stay in shared memory
//     (SMEM) when the launch planner finds enough capacity; global rollout storage and an on-demand
//     permutation are fallback paths. Activations stay in SMEM and registers,
//   * every CTA holds a full replica of the actor/critic weights in SMEM and registers; Adam moments
//     are held in registers by the thread that owns the parameter,
//   * gradient all-reduce uses global scratch (~10 KB per CTA, reduce-scatter + all-gather),
//     together with global barrier words and two scalars per CTA for advantage normalisation,
//   * global scratch is small enough to fit in L2 on the measured GPU, but cache residency is not
//     guaranteed. Launch boundaries, statistics, debug dumps, global rollout fallback, and register
//     spills (if reported by the compiler) also use global memory. This is not a zero-DRAM guarantee.
//
// Exactness (verified by scripts/sram/verify_megakernel.py):
//   * physics: closed-form 2-DoF cart-pole integrated exactly like MuJoCo-Warp's `implicitfast`
//     integrator as configured by Isaac Lab 3.0 (actuator damping kv folded into M, semi-implicit
//     Euler, unclamped feed-forward effort, clamped actuator force), float32,
//   * MDP: Isaac-Cartpole-Direct obs / reward / termination / time-out / reset distribution,
//   * PPO: RSL-RL 5.5.1 PPO math (clipped surrogate, clipped value loss, entropy bonus, time-out
//     bootstrapping, GAE, global advantage normalisation, KL-adaptive LR, separate actor / critic
//     grad clipping, Adam with torch semantics).  Mini-batches are stratified per SM (each SM
//     contributes exactly 1/num_mini_batches of its own transitions to every mini-batch) instead
//     of one global permutation: same expectation, no data movement.

#include "device_lib.cuh"

// --------------------------------------------------------------------------------------------
// compile-time configuration (all provided as -D defines by isaaclab_sram/megakernel.py)
// --------------------------------------------------------------------------------------------
#ifndef N_ENVS
#error "N_ENVS must be defined"
#endif
#ifndef N_CTAS
#error "N_CTAS must be defined"
#endif
#ifndef E_MAX
#error "E_MAX must be defined"
#endif
#ifndef STORE_SMEM
#define STORE_SMEM 1
#endif
#ifndef DBG
#define DBG 0
#endif
#ifndef PERM_SMEM
#define PERM_SMEM 1   // materialise the per-iteration mini-batch permutation in SMEM (else: on the fly)
#endif
#ifndef PROF
#define PROF 0
#endif
#ifndef PROF_CTA
#define PROF_CTA 1
#endif
#define PROF_STRIDE 80
#if PROF
#ifndef PROF_MAX_IT
#define PROF_MAX_IT 4096
#endif
#define PROF_MARK(k) do { if (cta == PROF_CTA && tid == 0 && git < PROF_MAX_IT && (k) < PROF_STRIDE) \
    reinterpret_cast<long long*>(g_dbg)[(size_t)git * PROF_STRIDE + (k)] = clock64(); } while (0)
#define PROF_MARK_C(k) do { if (epoch == 0 && mb == 0 && c0 == 0) PROF_MARK(k); } while (0)
#else
#define PROF_MARK(k) do { } while (0)
#define PROF_MARK_C(k) do { } while (0)
#endif

#define FULL 0xffffffffu
#define THREADS 256
#define HID 32
#define OBS 4
#define NNET 2
#define NGRP 4      // THREADS / (NNET * HID): sample groups (warps) per network
#define CHUNK 32    // samples processed per pass
#define SPT 8       // CHUNK / NGRP samples per thread per pass

// flat parameter layout (identical to torch nn.Linear [out, in] row-major), per network
#define OFF_W1 0
#define OFF_B1 (OFF_W1 + HID * OBS)    // 128
#define OFF_W2 (OFF_B1 + HID)          // 160
#define OFF_B2 (OFF_W2 + HID * HID)    // 1184
#define OFF_W3 (OFF_B2 + HID)          // 1216
#define OFF_B3 (OFF_W3 + HID)          // 1248
#define NET_P (OFF_B3 + 1)             // 1249
#define OFF_STD (2 * NET_P)            // 2498  (actor std parameter, "scalar" parameterisation)
#define N_PARAMS (2 * NET_P + 1)       // 2499
#define N_PARAMS_PAD 2500
// conflict-free shared-memory copy of the hidden layer W2 (row stride 33 floats)
#define W2P 33
#define W2N (HID * W2P)                // 1056 floats per network

// all-reduced vector: gradient followed by loss statistics
#define VEC_KL (N_PARAMS_PAD + 0)
#define VEC_SURR (N_PARAMS_PAD + 1)
#define VEC_VLOSS (N_PARAMS_PAD + 2)
#define VEC_LEN (N_PARAMS_PAD + 4)
#define VEC_STRIDE 2560
#define RS_K ((VEC_LEN + N_CTAS - 1) / N_CTAS)   // columns reduced by each CTA

// rollout storage fields (SoA, each [T_STEPS * E_MAX])
#define F_OBS0 0
#define F_ACT 4
#define F_LOGP 5
#define F_VAL 6
#define F_MU 7
#define F_REW 8
#define F_DONE 9
#define F_ADV 10
#define F_RET 11
#define NFIELD 12
#define FSTRIDE (T_STEPS * E_MAX)

// per-sample chunk scalars
#define SA_ACT 0
#define SA_OLDLOGP 1
#define SA_OLDMU 2
#define SA_ADV 3
#define SA_RET 4
#define SA_OLDV 5
#define SA_DSIG 6
#define SA_KL 7
#define SA_SURR 8
#define SA_VLOSS 9
#define NSA 10

// per-iteration statistics written to g_stats[iter * NSTAT + k]
#define ST_EP_COUNT 0
#define ST_EP_RET 1
#define ST_EP_LEN 2
#define ST_TERM 3
#define ST_TIMEOUT 4
#define ST_STEP_REW 5
#define ST_VLOSS 6
#define ST_SURR 7
#define ST_ENTROPY 8
#define ST_KL 9
#define ST_LR 10
#define ST_STD 11
#define ST_GNORM_A 12
#define ST_GNORM_C 13
#define ST_LIMIT 14
#define ST_ADV_MEAN 15
#define ST_ADV_STD 16
#define NSTAT 20


#include "sram_device.cuh"
#include "cartpole_physics.cuh"

// --------------------------------------------------------------------------------------------
// the megakernel
// --------------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(THREADS, 1) sram_ppo_cartpole(
    float* __restrict__ g_params,   // [N_PARAMS_PAD] canonical parameters (in/out)
    float* __restrict__ g_adam_m,   // [N_PARAMS_PAD]
    float* __restrict__ g_adam_v,   // [N_PARAMS_PAD]
    double* __restrict__ g_scal,    // [4]: lr, adam_step, beta1^step, beta2^step
    float* __restrict__ g_env,      // [N_ENVS * 4]: s, th, sd, thd
    int* __restrict__ g_eplen,      // [N_ENVS]
    float* __restrict__ g_epret,    // [N_ENVS]
    float* __restrict__ g_part,     // [(N_CTAS + 1) * VEC_STRIDE]: per-CTA partial vectors + reduced vector
    double* __restrict__ g_partd,   // [N_CTAS * 2]: per-CTA advantage moments
    u32* __restrict__ g_bar,        // grid barrier word (zero-initialised once)
    float* __restrict__ g_stats,    // [total_iters * NSTAT]
    float* __restrict__ g_store,    // [N_CTAS * NFIELD * FSTRIDE] when !STORE_SMEM
    float* __restrict__ g_dbg,      // debug dump (DBG) / clock64 marks (PROF)
    int iter0, int n_iters, u32 seed_lo, u32 seed_hi)
{
    extern __shared__ __align__(16) float smem[];
    __shared__ double s_redd[2 * (THREADS / 32)];
    __shared__ double s_opt[8];       // lr, b1^t, b2^t, candidate lrs (3), next b1^t, next b2^t
    __shared__ float s_cand[4];       // candidate -(lr/bc1) for {keep, decrease, increase}, sqrt(bc2)
    const int tid = threadIdx.x;
    const int cta = blockIdx.x;
    const int lane = tid & 31;
    const int net = tid >> 7;                 // 0 = actor, 1 = critic
    const int grp = (tid >> 5) & (NGRP - 1);  // sample group of this warp inside its network
    const int nb = net * NET_P;               // parameter base of this thread's network
    const int i_me = (lane >> 2) & 7;         // the sample of this warp that this lane "owns" after reduce8
    const int s_me = grp + NGRP * i_me;

    const int env0 = (int)(((long long)cta * N_ENVS) / N_CTAS);
    const int env1 = (int)(((long long)(cta + 1) * N_ENVS) / N_CTAS);
    const int E = env1 - env0;                 // environments owned by this CTA (<= E_MAX)
    const int n_local = E * T_STEPS;           // transitions owned per iteration
    const int mb_local = n_local / N_MB;       // transitions this CTA contributes to each mini-batch

    // ---------------- shared memory carve-up ----------------
    float* s_params = smem;                                // N_PARAMS_PAD (flat; W2 lives in s_w2)
    float* s_w2 = s_params + N_PARAMS_PAD;                 // NNET * W2N (padded W2)
    float* s_x = s_w2 + NNET * W2N;                        // CHUNK * 4
    float* s_a1 = s_x + CHUNK * OBS;                       // NNET * CHUNK * HID
    float* s_a2 = s_a1 + NNET * CHUNK * HID;
    float* s_dz2 = s_a2 + NNET * CHUNK * HID;
    float* s_dz1 = s_dz2 + NNET * CHUNK * HID;
    float* s_out = s_dz1 + NNET * CHUNK * HID;             // NNET * CHUNK  (mu, value)
    float* s_gout = s_out + NNET * CHUNK;                  // NNET * CHUNK  (dL/dmu, dL/dv)
    float* s_sa = s_gout + NNET * CHUNK;                   // NSA * CHUNK
    float* s_red = s_sa + NSA * CHUNK;                     // 64 scratch
    float* s_env = s_red + 64;                             // E_MAX * 4
    float* s_epret = s_env + E_MAX * 4;                    // E_MAX
    int* s_eplen = (int*)(s_epret + E_MAX);                // E_MAX
    float* s_lastv = (float*)(s_eplen + E_MAX);            // E_MAX
    int* s_perm = (int*)(s_lastv + E_MAX);                 // FSTRIDE if PERM_SMEM (this iteration's permutation)
    float* s_after_perm = (float*)(s_perm + (PERM_SMEM ? FSTRIDE : 0));
#if STORE_SMEM
    float* st = s_after_perm;                              // NFIELD * FSTRIDE
#else
    float* st = g_store + (size_t)cta * NFIELD * FSTRIDE;  // L2/DRAM-resident rollout storage
#endif
    float* g_red = g_part + (size_t)N_CTAS * VEC_STRIDE;

    // ---------------- load persistent state ----------------
    for (int i = tid; i < N_PARAMS_PAD; i += THREADS) s_params[i] = g_params[i];
    for (int i = tid; i < NNET * HID * HID; i += THREADS) {
        const int n = i / (HID * HID), r = i % (HID * HID);
        s_w2[n * W2N + (r / HID) * W2P + (r % HID)] = g_params[n * NET_P + OFF_W2 + r];
    }
    for (int e = tid; e < E; e += THREADS) {
        const float4 v = reinterpret_cast<const float4*>(g_env)[env0 + e];
        s_env[e * 4 + 0] = v.x; s_env[e * 4 + 1] = v.y; s_env[e * 4 + 2] = v.z; s_env[e * 4 + 3] = v.w;
        s_eplen[e] = g_eplen[env0 + e];
        s_epret[e] = g_epret[env0 + e];
    }
    if (tid == 0) {   // lr, beta1^t, beta2^t (persisted exactly across launches)
        s_opt[0] = g_scal[0];
        s_opt[1] = g_scal[2];
        s_opt[2] = g_scal[3];
    }

    // parameter ownership (gradient accumulation, all-reduce, Adam): every one of the 2499
    // parameters has exactly one owner thread per CTA, chosen so that all SMEM / L2 accesses of
    // the owner are bank-conflict free / coalesced:
    //   W2[net][8*grp + kk][lane] (8 per thread), W1[net][w>>2][w&3] (1), extra: b1 / b2 / W3 / b3 / std
    const int w = tid & 127;
    int own_x = -1;
    if (w < 32) own_x = nb + OFF_B1 + w;
    else if (w < 64) own_x = nb + OFF_B2 + (w - 32);
    else if (w < 96) own_x = nb + OFF_W3 + (w - 64);
    else if (w == 96) own_x = nb + OFF_B3;
    else if (w == 97 && net == 0) own_x = OFF_STD;
    const int own_w1 = nb + OFF_W1 + (w >> 2) * OBS + (w & 3);
#define OWN_W2(kk) (nb + OFF_W2 + (8 * grp + (kk)) * HID + lane)        /* flat index */
#define PW2(kk) (net * W2N + (8 * grp + (kk)) * W2P + lane)            /* padded SMEM index */

    float m_w2[8], v_w2[8], m_w1, v_w1, m_x = 0.f, v_x = 0.f;
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) { m_w2[kk] = g_adam_m[OWN_W2(kk)]; v_w2[kk] = g_adam_v[OWN_W2(kk)]; }
    m_w1 = g_adam_m[own_w1]; v_w1 = g_adam_v[own_w1];
    if (own_x >= 0) { m_x = g_adam_m[own_x]; v_x = g_adam_v[own_x]; }
    __syncthreads();

    // weights of this thread's (net, j = lane) unit, register resident
    float w1r[OBS], w2row[HID], w2col[HID], b1r, b2r, w3r, b3r;
    auto load_weights = [&]() {
#pragma unroll
        for (int i = 0; i < OBS; ++i) w1r[i] = s_params[nb + OFF_W1 + lane * OBS + i];
#pragma unroll
        for (int k = 0; k < HID; ++k) {
            w2row[k] = s_w2[net * W2N + lane * W2P + k];   // conflict-free (stride 33)
            w2col[k] = s_w2[net * W2N + k * W2P + lane];
        }
        b1r = s_params[nb + OFF_B1 + lane];
        b2r = s_params[nb + OFF_B2 + lane];
        w3r = s_params[nb + OFF_W3 + lane];
        b3r = s_params[nb + OFF_B3];
    };
    // copy padded W2 back into the flat parameter vector (debug dumps / write-back)
    auto w2_to_flat = [&]() {
        for (int i = tid; i < NNET * HID * HID; i += THREADS) {
            const int n = i / (HID * HID), r = i % (HID * HID);
            s_params[n * NET_P + OFF_W2 + r] = s_w2[n * W2N + (r / HID) * W2P + (r % HID)];
        }
    };

    // Forward pass of both networks for Sc samples whose inputs are in s_x.  Thread (net, lane,
    // grp) computes unit `lane` for samples grp + NGRP*i; leaves a1r/a2r in registers, the
    // activations in s_a1 / s_a2, and returns the network output of sample s_me (lanes 4i..4i+3
    // hold sample i of the warp).
    float a1r[SPT], a2r[SPT];
    auto forward = [&](int Sc) -> float {
#pragma unroll
        for (int i = 0; i < SPT; ++i) {
            const int s = grp + NGRP * i;
            if (s < Sc) {
                const float4 xv = *reinterpret_cast<const float4*>(&s_x[s * OBS]);
                float z = b1r;
                z = fmaf(w1r[0], xv.x, z); z = fmaf(w1r[1], xv.y, z);
                z = fmaf(w1r[2], xv.z, z); z = fmaf(w1r[3], xv.w, z);
                const float a = elu(z);
                a1r[i] = a;
                s_a1[(net * CHUNK + s) * HID + lane] = a;
            }
        }
        __syncthreads();
        float part[SPT];
#pragma unroll
        for (int i = 0; i < SPT; ++i) {
            const int s = grp + NGRP * i;
            part[i] = 0.f;
            if (s < Sc) {
                const float4* av = reinterpret_cast<const float4*>(&s_a1[(net * CHUNK + s) * HID]);
                float z0 = b2r, z1 = 0.f, z2 = 0.f, z3 = 0.f;   // 4 independent FMA chains (ILP)
#pragma unroll
                for (int k4 = 0; k4 < HID / 4; ++k4) {
                    const float4 v = av[k4];
                    z0 = fmaf(w2row[4 * k4 + 0], v.x, z0); z1 = fmaf(w2row[4 * k4 + 1], v.y, z1);
                    z2 = fmaf(w2row[4 * k4 + 2], v.z, z2); z3 = fmaf(w2row[4 * k4 + 3], v.w, z3);
                }
                const float a = elu((z0 + z1) + (z2 + z3));
                a2r[i] = a;
                s_a2[(net * CHUNK + s) * HID + lane] = a;
                part[i] = w3r * a;
            }
        }
        return reduce8(part, lane) + b3r;
    };

    // =====================================================================================
    for (int it = 0; it < n_iters; ++it) {
        const int git = iter0 + it;
        load_weights();
        const float std_param = s_params[OFF_STD];
        const float sigma = fminf(fmaxf(std_param, 1e-6f), 1e6f);
        const float log_sigma = logf(sigma);
        PROF_MARK(0);

        // per-CTA rollout statistics
        float st_ep_count = 0.f, st_ep_ret = 0.f, st_ep_len = 0.f, st_term = 0.f, st_tout = 0.f,
              st_rew = 0.f, st_limit = 0.f;

        // ---------------------------- rollout ------------------------------------------------
        for (int t = 0; t < T_STEPS; ++t) {
            for (int c0 = 0; c0 < E; c0 += CHUNK) {
                const int Sc = min(CHUNK, E - c0);
                if (tid < Sc) {
                    const int e = c0 + tid;
                    *reinterpret_cast<float4*>(&s_x[tid * OBS]) = *reinterpret_cast<const float4*>(&s_env[e * 4]);
                }
                __syncthreads();
                const float o = forward(Sc);
                if ((lane & 3) == 0 && s_me < Sc) s_out[net * CHUNK + s_me] = o;
                __syncthreads();
                if (tid < Sc) {
                    const int e = c0 + tid;
                    const int ge = env0 + e;
                    const int tr = t * E + e;            // local transition index
                    float s = s_env[e * 4 + 0], th = s_env[e * 4 + 1], sd = s_env[e * 4 + 2], thd = s_env[e * 4 + 3];
                    const float mu = s_out[tid];
                    const float val = s_out[CHUNK + tid];
                    const uint4 r = philox(make_uint4((u32)ge, (u32)git, (u32)t, 0x41435431u), seed_lo, seed_hi);
                    const float eps = normal01(r.x, r.y);
                    const float act = mu + sigma * eps;
                    const float d = act - mu;
                    const float logp = -(d * d) / (2.0f * (sigma * sigma)) - log_sigma - LOG_SQRT_2PI;   // torch Normal.log_prob
                    st[(F_OBS0 + 0) * FSTRIDE + tr] = s;
                    st[(F_OBS0 + 1) * FSTRIDE + tr] = th;
                    st[(F_OBS0 + 2) * FSTRIDE + tr] = sd;
                    st[(F_OBS0 + 3) * FSTRIDE + tr] = thd;
                    st[F_ACT * FSTRIDE + tr] = act;
                    st[F_LOGP * FSTRIDE + tr] = logp;
                    st[F_VAL * FSTRIDE + tr] = val;
                    st[F_MU * FSTRIDE + tr] = mu;
                    // physics (decimation substeps with a held feed-forward effort)
                    const float f_app = P_ACTION_SCALE * act;
#pragma unroll
                    for (int k = 0; k < P_DECIM; ++k) cartpole_substep(s, th, sd, thd, f_app);
                    if (fabsf(s) >= P_LIMIT) st_limit += 1.f;   // cart joint limit never reached in practice
                    int eplen = s_eplen[e] + 1;
                    const bool term = fabsf(s) > P_MAX_CART;
                    const bool tout = eplen >= P_MAX_EP_LEN;
                    const float tf = term ? 1.0f : 0.0f;
                    const float th_wrapped = wrap_to_pi(th);
                    const float rew = (P_REW_ALIVE * (1.0f - tf) + P_REW_TERM * tf + P_REW_POLE_POS * (th_wrapped * th_wrapped) +
                                       P_REW_CART_VEL * fabsf(sd) + P_REW_POLE_VEL * fabsf(thd)) * P_STEP_DT;
                    float epret = s_epret[e] + rew;
                    st_rew += rew;
                    st[F_REW * FSTRIDE + tr] = tout ? rew + P_GAMMA * val : rew;   // RSL-RL time-out bootstrap
                    st[F_DONE * FSTRIDE + tr] = (term || tout) ? 1.0f : 0.0f;
#if DBG
                    if (g_dbg && it == 0) {   // post-physics, pre-reset state for the reward / dynamics check
                        float* dd = g_dbg + (size_t)cta * 4 * FSTRIDE;
                        dd[0 * FSTRIDE + tr] = s; dd[1 * FSTRIDE + tr] = th; dd[2 * FSTRIDE + tr] = sd; dd[3 * FSTRIDE + tr] = thd;
                    }
#endif
                    if (term || tout) {
                        st_ep_count += 1.f; st_ep_ret += epret; st_ep_len += (float)eplen;
                        st_term += tf; st_tout += tout ? 1.f : 0.f;
                        const uint4 q = philox(make_uint4((u32)ge, (u32)git, (u32)t, 0x52535431u), seed_lo, seed_hi);
                        s = P_RST_S_LO + (P_RST_S_HI - P_RST_S_LO) * u01(q.x);
                        sd = P_RST_SD_LO + (P_RST_SD_HI - P_RST_SD_LO) * u01(q.y);
                        th = P_RST_TH_LO + (P_RST_TH_HI - P_RST_TH_LO) * u01(q.z);
                        thd = P_RST_THD_LO + (P_RST_THD_HI - P_RST_THD_LO) * u01(q.w);
                        s = fminf(fmaxf(s, -P_LIMIT), P_LIMIT);                     // soft joint limits
                        sd = fminf(fmaxf(sd, -P_VLIM_CART), P_VLIM_CART);           // soft velocity limits
                        thd = fminf(fmaxf(thd, -P_VLIM_POLE), P_VLIM_POLE);
                        eplen = 0; epret = 0.f;
                    }
                    s_env[e * 4 + 0] = s; s_env[e * 4 + 1] = th; s_env[e * 4 + 2] = sd; s_env[e * 4 + 3] = thd;
                    s_eplen[e] = eplen; s_epret[e] = epret;
                }
                __syncthreads();
            }
        }
        PROF_MARK(1);

        // ---------------------------- bootstrap values ----------------------------------------
        for (int c0 = 0; c0 < E; c0 += CHUNK) {
            const int Sc = min(CHUNK, E - c0);
            if (tid < Sc) *reinterpret_cast<float4*>(&s_x[tid * OBS]) = *reinterpret_cast<const float4*>(&s_env[(c0 + tid) * 4]);
            __syncthreads();
            const float o = forward(Sc);
            if (net == 1 && (lane & 3) == 0 && s_me < Sc) s_lastv[c0 + s_me] = o;
            __syncthreads();
        }
        PROF_MARK(2);

        // ---------------------------- GAE (thread per env) ------------------------------------
        double adv_sum = 0.0, adv_sq = 0.0;
        for (int e = tid; e < E; e += THREADS) {
            float adv = 0.0f;
            for (int t = T_STEPS - 1; t >= 0; --t) {
                const int tr = t * E + e;
                const float nv = (t == T_STEPS - 1) ? s_lastv[e] : st[F_VAL * FSTRIDE + tr + E];
                const float nnt = 1.0f - st[F_DONE * FSTRIDE + tr];
                const float v = st[F_VAL * FSTRIDE + tr];
                const float delta = st[F_REW * FSTRIDE + tr] + nnt * P_GAMMA * nv - v;
                adv = delta + nnt * P_GAMMA * P_LAMBDA * adv;
                const float ret = adv + v;
                const float a = ret - v;                 // RSL-RL: advantages = returns - values
                st[F_RET * FSTRIDE + tr] = ret;
                st[F_ADV * FSTRIDE + tr] = a;
                adv_sum += (double)a;
                adv_sq += (double)a * (double)a;
            }
        }
        // block-reduce advantage moments and publish them (one deterministic all-reduce)
        adv_sum = warp_sum_d(adv_sum);
        adv_sq = warp_sum_d(adv_sq);
        if (lane == 0) { s_redd[2 * (tid >> 5)] = adv_sum; s_redd[2 * (tid >> 5) + 1] = adv_sq; }
        __syncthreads();
        {
            if (tid == 0) {
                double a = 0.0, b = 0.0;
                for (int i = 0; i < THREADS / 32; ++i) { a += s_redd[2 * i]; b += s_redd[2 * i + 1]; }
                g_partd[2 * cta + 0] = a;
                g_partd[2 * cta + 1] = b;
            }
            // episode statistics are reduced with plain atomics (logging only, no barrier needed)
            {
                float v0 = warp_sum(st_ep_count), v1 = warp_sum(st_ep_ret), v2 = warp_sum(st_ep_len),
                      v3 = warp_sum(st_term), v4 = warp_sum(st_tout), v5 = warp_sum(st_rew), v6 = warp_sum(st_limit);
                if (lane == 0 && (v0 != 0.f || v5 != 0.f || v6 != 0.f)) {
                    float* gs = g_stats + (size_t)git * NSTAT;
                    atomicAdd(&gs[ST_EP_COUNT], v0); atomicAdd(&gs[ST_EP_RET], v1); atomicAdd(&gs[ST_EP_LEN], v2);
                    atomicAdd(&gs[ST_TERM], v3); atomicAdd(&gs[ST_TIMEOUT], v4); atomicAdd(&gs[ST_STEP_REW], v5);
                    atomicAdd(&gs[ST_LIMIT], v6);
                }
            }
            grid_barrier(g_bar);
            if (tid < 32) {   // every CTA sums the per-CTA moments in the same fixed order
                double a = 0.0, b = 0.0;
                for (int c = tid; c < N_CTAS; c += 32) { a += __ldcg(&g_partd[2 * c]); b += __ldcg(&g_partd[2 * c + 1]); }
                a = warp_sum_d(a);
                b = warp_sum_d(b);
                if (tid == 0) { s_redd[0] = a; s_redd[1] = b; }
            }
            __syncthreads();
            const double S = s_redd[0];
            const double Q = s_redd[1];
            const double Ntot = (double)N_ENVS * (double)T_STEPS;
            const double mean = S / Ntot;
            const double var = fmax((Q - Ntot * mean * mean) / (Ntot - 1.0), 0.0);
            const float meanf = (float)mean;
            const float denom = (float)sqrt(var) + 1e-8f;
            for (int tr = tid; tr < n_local; tr += THREADS)
                st[F_ADV * FSTRIDE + tr] = (st[F_ADV * FSTRIDE + tr] - meanf) / denom;
            if (cta == 0 && tid == 0) {
                g_stats[(size_t)git * NSTAT + ST_ADV_MEAN] = meanf;
                g_stats[(size_t)git * NSTAT + ST_ADV_STD] = (float)sqrt(var);
            }
            __syncthreads();   // every warp's normalised advantages are visible before the dump / gathers
        }
        PROF_MARK(3);
        // keyed pseudo-random permutation of this CTA's transitions for this iteration
        const uint4 pkey = philox(make_uint4((u32)cta, (u32)git, 0u, 0x5045524Du), seed_lo, seed_hi);
        u32 phalf = 1;
        while ((1u << (2 * phalf)) < (u32)n_local) ++phalf;
#if PERM_SMEM
        // all threads evaluate the permutation once per iteration (cycle-walk latency hidden by parallelism)
        for (int i = tid; i < n_local; i += THREADS) s_perm[i] = (int)feistel_perm((u32)i, (u32)n_local, phalf, pkey);
        __syncthreads();
#endif

#if DBG
        if (g_dbg && it == 0) {   // dump storage (after normalisation) + permutation
            float* dd = g_dbg + (size_t)N_CTAS * 4 * FSTRIDE + (size_t)cta * (NFIELD + 1) * FSTRIDE;
            for (int i = tid; i < NFIELD * FSTRIDE; i += THREADS) dd[i] = st[i];
            for (int i = tid; i < FSTRIDE; i += THREADS)
                dd[NFIELD * FSTRIDE + i] = (i < n_local) ? (float)feistel_perm((u32)i, (u32)n_local, phalf, pkey) : -1.f;
        }
#endif
        PROF_MARK(4);

        // ---------------------------- PPO update ----------------------------------------------
        const float old_sigma = sigma;   // all transitions of this rollout share the same std
        float it_vloss = 0.f, it_surr = 0.f, it_ent = 0.f, it_kl = 0.f, it_gna = 0.f, it_gnc = 0.f;
        for (int epoch = 0; epoch < N_EPOCHS; ++epoch) {
            for (int mb = 0; mb < N_MB; ++mb) {
                const float cstd = s_params[OFF_STD];
                const float csig = fminf(fmaxf(cstd, 1e-6f), 1e6f);
                const float std_pass = (cstd >= 1e-6f && cstd <= 1e6f) ? 1.0f : 0.0f;   // clamp gradient mask
                const float clog = logf(csig);
                const float inv_sig = 1.0f / csig;
                const float inv_var = inv_sig * inv_sig;
                const float vr_s = old_sigma / csig;   // torch: var_ratio = (p.scale / q.scale) ** 2
                const float vr = vr_s * vr_s;
                const float log_vr = logf(vr);
                float g_w2[8];
#pragma unroll
                for (int kk = 0; kk < 8; ++kk) g_w2[kk] = 0.f;
                float g_w1 = 0.f, g_x = 0.f, g_stat = 0.f;

                for (int c0 = 0; c0 < mb_local; c0 += CHUNK) {
                    const int Sc = min(CHUNK, mb_local - c0);
                    PROF_MARK_C(70);
                    // gather this chunk of the mini-batch from SRAM-resident storage
                    if (tid < Sc) {
#if PERM_SMEM
                        const int tr = s_perm[mb * mb_local + c0 + tid];
#else
                        const int tr = (int)feistel_perm((u32)(mb * mb_local + c0 + tid), (u32)n_local, phalf, pkey);
#endif
                        float4 o;
                        o.x = st[(F_OBS0 + 0) * FSTRIDE + tr]; o.y = st[(F_OBS0 + 1) * FSTRIDE + tr];
                        o.z = st[(F_OBS0 + 2) * FSTRIDE + tr]; o.w = st[(F_OBS0 + 3) * FSTRIDE + tr];
                        *reinterpret_cast<float4*>(&s_x[tid * OBS]) = o;
                        s_sa[SA_ACT * CHUNK + tid] = st[F_ACT * FSTRIDE + tr];
                        s_sa[SA_OLDLOGP * CHUNK + tid] = st[F_LOGP * FSTRIDE + tr];
                        s_sa[SA_OLDMU * CHUNK + tid] = st[F_MU * FSTRIDE + tr];
                        s_sa[SA_ADV * CHUNK + tid] = st[F_ADV * FSTRIDE + tr];
                        s_sa[SA_RET * CHUNK + tid] = st[F_RET * FSTRIDE + tr];
                        s_sa[SA_OLDV * CHUNK + tid] = st[F_VAL * FSTRIDE + tr];
                    }
                    __syncthreads();
                    PROF_MARK_C(71);
                    const float out_me = forward(Sc);
                    PROF_MARK_C(72);

                    // per-sample loss gradient: lanes 4i..4i+3 evaluate sample i of the warp in parallel
                    float gout_me = 0.f;
                    if (s_me < Sc) {
                        if (net == 0) {
                            const float mu = out_me;
                            const float d = s_sa[SA_ACT * CHUNK + s_me] - mu;
                            const float logp = -(d * d) / (2.0f * (csig * csig)) - clog - LOG_SQRT_2PI;
                            const float ratio = expf(logp - s_sa[SA_OLDLOGP * CHUNK + s_me]);
                            const float adv = s_sa[SA_ADV * CHUNK + s_me];
                            const float rcl = fminf(fmaxf(ratio, 1.0f - P_CLIP), 1.0f + P_CLIP);
                            const float su = -adv * ratio, sc = -adv * rcl;
                            const float inr = (ratio >= 1.0f - P_CLIP && ratio <= 1.0f + P_CLIP) ? 1.0f : 0.0f;
                            // torch.max(su, sc): gradient to the larger one, split 1/2 on ties
                            float dldr;
                            if (su > sc) dldr = -adv;
                            else if (su < sc) dldr = -adv * inr;
                            else dldr = 0.5f * (-adv) + 0.5f * (-adv * inr);
                            const float dlogp = dldr * ratio * INV_NMB;
                            gout_me = dlogp * d * inv_var;
                            if ((lane & 3) == 0) {
                                const float dsig = dlogp * (d * d * inv_var - 1.0f) * inv_sig - P_ENT_COEF * INV_NMB * inv_sig;
                                s_sa[SA_DSIG * CHUNK + s_me] = dsig * std_pass;
                                const float dm = (s_sa[SA_OLDMU * CHUNK + s_me] - mu) / csig;
                                s_sa[SA_KL * CHUNK + s_me] = 0.5f * (((vr + dm * dm) - 1.0f) - log_vr);
                                s_sa[SA_SURR * CHUNK + s_me] = fmaxf(su, sc);
                            }
                        } else {
                            const float v = out_me;
                            const float ov = s_sa[SA_OLDV * CHUNK + s_me];
                            const float ret = s_sa[SA_RET * CHUNK + s_me];
                            const float dv = v - ov;
                            const float vc = ov + fminf(fmaxf(dv, -P_CLIP), P_CLIP);
                            const float l1 = (v - ret) * (v - ret), l2 = (vc - ret) * (vc - ret);
                            const float inr = (dv >= -P_CLIP && dv <= P_CLIP) ? 1.0f : 0.0f;
                            float dl;
                            if (l1 > l2) dl = 2.0f * (v - ret);
                            else if (l1 < l2) dl = 2.0f * (vc - ret) * inr;
                            else dl = (v - ret) + (vc - ret) * inr;
                            gout_me = P_VF_COEF * dl * INV_NMB;
                            if ((lane & 3) == 0) s_sa[SA_VLOSS * CHUNK + s_me] = fmaxf(l1, l2);
                        }
                        if ((lane & 3) == 0) s_gout[net * CHUNK + s_me] = gout_me;
                    }
                    // dZ2 for all samples of this thread (broadcast each sample's dL/d(output))
#pragma unroll
                    for (int i = 0; i < SPT; ++i) {
                        const float gi = __shfl_sync(FULL, gout_me, 4 * i);
                        const int s = grp + NGRP * i;
                        if (s < Sc) s_dz2[(net * CHUNK + s) * HID + lane] = gi * w3r * delu_from_act(a2r[i]);
                    }
                    __syncthreads();
                    PROF_MARK_C(73);

                    // stage B: dZ1 for this thread's samples, owned dW2 entries for all samples
#pragma unroll
                    for (int i = 0; i < SPT; ++i) {
                        const int s = grp + NGRP * i;
                        if (s < Sc) {
                            const float4* dv = reinterpret_cast<const float4*>(&s_dz2[(net * CHUNK + s) * HID]);
                            float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
#pragma unroll
                            for (int j4 = 0; j4 < HID / 4; ++j4) {
                                const float4 q = dv[j4];
                                d0 = fmaf(w2col[4 * j4 + 0], q.x, d0); d1 = fmaf(w2col[4 * j4 + 1], q.y, d1);
                                d2 = fmaf(w2col[4 * j4 + 2], q.z, d2); d3 = fmaf(w2col[4 * j4 + 3], q.w, d3);
                            }
                            s_dz1[(net * CHUNK + s) * HID + lane] = ((d0 + d1) + (d2 + d3)) * delu_from_act(a1r[i]);
                        }
                    }
#pragma unroll 4
                    for (int s = 0; s < Sc; ++s) {   // dW2[8grp+kk][lane] += dZ2[s][8grp+kk] * A1[s][lane]
                        const float a = s_a1[(net * CHUNK + s) * HID + lane];
                        const float4 d0 = *reinterpret_cast<const float4*>(&s_dz2[(net * CHUNK + s) * HID + 8 * grp]);
                        const float4 d1 = *reinterpret_cast<const float4*>(&s_dz2[(net * CHUNK + s) * HID + 8 * grp + 4]);
                        g_w2[0] = fmaf(d0.x, a, g_w2[0]); g_w2[1] = fmaf(d0.y, a, g_w2[1]);
                        g_w2[2] = fmaf(d0.z, a, g_w2[2]); g_w2[3] = fmaf(d0.w, a, g_w2[3]);
                        g_w2[4] = fmaf(d1.x, a, g_w2[4]); g_w2[5] = fmaf(d1.y, a, g_w2[5]);
                        g_w2[6] = fmaf(d1.z, a, g_w2[6]); g_w2[7] = fmaf(d1.w, a, g_w2[7]);
                    }
                    __syncthreads();
                    PROF_MARK_C(74);

                    // stage C: remaining parameter gradients, summed by their owner threads
                    {
                        const int j = w >> 2, ii = w & 3;
                        float acc0 = 0.f, acc1 = 0.f;
#pragma unroll 4
                        for (int s = 0; s + 1 < Sc; s += 2) {
                            acc0 = fmaf(s_dz1[(net * CHUNK + s) * HID + j], s_x[s * OBS + ii], acc0);
                            acc1 = fmaf(s_dz1[(net * CHUNK + s + 1) * HID + j], s_x[(s + 1) * OBS + ii], acc1);
                        }
                        if (Sc & 1) acc0 = fmaf(s_dz1[(net * CHUNK + Sc - 1) * HID + j], s_x[(Sc - 1) * OBS + ii], acc0);
                        g_w1 += acc0 + acc1;
                    }
                    if (own_x >= 0) {
                        float acc = 0.f;
                        if (w < 32) {
                            for (int s = 0; s < Sc; ++s) acc += s_dz1[(net * CHUNK + s) * HID + w];
                        } else if (w < 64) {
                            for (int s = 0; s < Sc; ++s) acc += s_dz2[(net * CHUNK + s) * HID + (w - 32)];
                        } else if (w < 96) {
                            for (int s = 0; s < Sc; ++s) acc = fmaf(s_gout[net * CHUNK + s], s_a2[(net * CHUNK + s) * HID + (w - 64)], acc);
                        } else if (w == 96) {
                            for (int s = 0; s < Sc; ++s) acc += s_gout[net * CHUNK + s];
                        } else {
                            for (int s = 0; s < Sc; ++s) acc += s_sa[SA_DSIG * CHUNK + s];
                        }
                        g_x += acc;
                    }
                    if (w == 98 || w == 99) {
                        const int f = (net == 0) ? (w == 98 ? SA_KL : SA_SURR) : (w == 98 ? SA_VLOSS : -1);
                        if (f >= 0) {
                            float acc = 0.f;
                            for (int s = 0; s < Sc; ++s) acc += s_sa[f * CHUNK + s];
                            g_stat += acc;
                        }
                    }
                    __syncthreads();
                    PROF_MARK_C(75);
                }

                // ---- deterministic gradient all-reduce across SMs (L2-resident) ----
                // 1) every CTA stores its partial vector, 2) CTA c reduces columns [c*RS_K, (c+1)*RS_K)
                // in a fixed order, 3) every CTA reads the reduced vector.  Plain stores + 2 barriers,
                // no atomics -> bitwise reproducible training.
                const int q_prof = epoch * N_MB + mb; (void)q_prof;
                PROF_MARK(5 + 3 * q_prof);
                {
                    float* mypart = g_part + (size_t)cta * VEC_STRIDE;
#pragma unroll
                    for (int kk = 0; kk < 8; ++kk) mypart[OWN_W2(kk)] = g_w2[kk];
                    mypart[own_w1] = g_w1;
                    if (own_x >= 0) mypart[own_x] = g_x;
                    if (net == 0 && w == 98) mypart[VEC_KL] = g_stat;
                    if (net == 0 && w == 99) mypart[VEC_SURR] = g_stat;
                    if (net == 1 && w == 98) mypart[VEC_VLOSS] = g_stat;
                }
                if (tid == 32) {
                    // optimiser scalars for all three possible KL outcomes, computed in FP64 (torch
                    // uses Python floats) off the critical path while the other CTAs catch up
                    const double lr = s_opt[0];
                    const double b1n = s_opt[1] * ADAM_B1_D, b2n = s_opt[2] * ADAM_B2_D;   // beta ** step (fp64)
                    const double bc1 = 1.0 - b1n;
                    const double lr_dec = fmax(1e-5, lr / 1.5), lr_inc = fmin(1e-2, lr * 1.5);
                    s_opt[3] = lr; s_opt[4] = lr_dec; s_opt[5] = lr_inc;
                    s_opt[6] = b1n; s_opt[7] = b2n;
                    s_cand[0] = (float)(-(lr / bc1));
                    s_cand[1] = (float)(-(lr_dec / bc1));
                    s_cand[2] = (float)(-(lr_inc / bc1));
                    s_cand[3] = (float)sqrt(1.0 - b2n);
                }
                grid_barrier(g_bar);
                {
                    const int col0 = cta * RS_K;
                    float* s_rs = s_dz1;   // scratch (unused between chunks)
                    for (int jb = 0; jb < RS_K; jb += 16) {
                        const int j = jb + (tid & 15), r = tid >> 4;
                        const int col = col0 + j;
                        float a = 0.f;
                        if (j < RS_K && col < VEC_LEN)
                            for (int c = r; c < N_CTAS; c += 16) a += __ldcg(&g_part[(size_t)c * VEC_STRIDE + col]);
                        s_rs[r * 16 + (tid & 15)] = a;
                        __syncthreads();
                        if (tid < 16 && jb + tid < RS_K && col0 + jb + tid < VEC_LEN) {
                            float b = 0.f;
#pragma unroll
                            for (int rr = 0; rr < 16; ++rr) b += s_rs[rr * 16 + tid];
                            g_red[col0 + jb + tid] = b;
                        }
                        __syncthreads();
                    }
                }
                grid_barrier(g_bar);
                PROF_MARK(6 + 3 * q_prof);

                float gw2[8];
#pragma unroll
                for (int kk = 0; kk < 8; ++kk) gw2[kk] = __ldcg(&g_red[OWN_W2(kk)]);
                const float gw1 = __ldcg(&g_red[own_w1]);
                const float gx = (own_x >= 0) ? __ldcg(&g_red[own_x]) : 0.f;
#if DBG
                if (g_dbg && it == 0 && epoch == 0 && mb == 0 && cta == 0) {
                    w2_to_flat();
                    __syncthreads();
                    float* dd = g_dbg + (size_t)N_CTAS * 4 * FSTRIDE + (size_t)N_CTAS * (NFIELD + 1) * FSTRIDE;
                    for (int i = tid; i < VEC_STRIDE; i += THREADS) dd[i] = (i < VEC_LEN) ? __ldcg(&g_red[i]) : 0.f;
                    for (int i = tid; i < N_PARAMS_PAD; i += THREADS) dd[VEC_STRIDE + N_PARAMS_PAD + i] = s_params[i];
                }
#endif
                // per-model gradient norms (actor incl. std / critic) for clip_grad_norm_
                float sq = gw1 * gw1 + gx * gx;
#pragma unroll
                for (int kk = 0; kk < 8; ++kk) sq = fmaf(gw2[kk], gw2[kk], sq);
                sq = warp_sum(sq);
                if (lane == 0) s_red[tid >> 5] = sq;
                if (tid == 0) {
                    // RSL-RL KL-adaptive learning rate: pick the pre-computed candidate
                    const float kl_mean = __ldcg(&g_red[VEC_KL]) * INV_NMB;
                    int c = 0;
                    if (P_ADAPTIVE) {
                        if (kl_mean > P_DESIRED_KL * 2.0f) c = 1;
                        else if (kl_mean < P_DESIRED_KL / 2.0f && kl_mean > 0.0f) c = 2;
                    }
                    s_red[8] = s_cand[c];
                    s_red[9] = s_cand[3];
                    s_opt[0] = s_opt[3 + c];
                    s_red[10] = kl_mean;
                    s_opt[1] = s_opt[6];
                    s_opt[2] = s_opt[7];
                    if (cta == 0) {
                        it_vloss += __ldcg(&g_red[VEC_VLOSS]) * INV_NMB;
                        it_surr += __ldcg(&g_red[VEC_SURR]) * INV_NMB;
                        it_ent += 0.5f + LOG_SQRT_2PI + clog;
                        it_kl = kl_mean;
                    }
                }
                __syncthreads();
                const float na = sqrtf(s_red[0] + s_red[1] + s_red[2] + s_red[3]);
                const float nc = sqrtf(s_red[4] + s_red[5] + s_red[6] + s_red[7]);
                const float coef = fminf(P_MAX_GRAD_NORM / ((net == 0 ? na : nc) + 1e-6f), 1.0f);
                const float neg_step = s_red[8];
                const float bc2s = s_red[9];
                // Adam (torch.optim.Adam foreach semantics: lerp for m, addcmul for v, addcdiv update)
#pragma unroll
                for (int kk = 0; kk < 8; ++kk) {
                    const float g = gw2[kk] * coef;
                    m_w2[kk] = m_w2[kk] + ADAM_1MB1 * (g - m_w2[kk]);
                    v_w2[kk] = v_w2[kk] * ADAM_B2 + ADAM_1MB2 * g * g;
                    s_w2[PW2(kk)] = s_w2[PW2(kk)] + neg_step * (m_w2[kk] / (sqrtf(v_w2[kk]) / bc2s + ADAM_EPS));
                }
                {
                    const float g = gw1 * coef;
                    m_w1 = m_w1 + ADAM_1MB1 * (g - m_w1);
                    v_w1 = v_w1 * ADAM_B2 + ADAM_1MB2 * g * g;
                    s_params[own_w1] = s_params[own_w1] + neg_step * (m_w1 / (sqrtf(v_w1) / bc2s + ADAM_EPS));
                }
                if (own_x >= 0) {
                    const float g = gx * coef;
                    m_x = m_x + ADAM_1MB1 * (g - m_x);
                    v_x = v_x * ADAM_B2 + ADAM_1MB2 * g * g;
                    s_params[own_x] = s_params[own_x] + neg_step * (m_x / (sqrtf(v_x) / bc2s + ADAM_EPS));
                }
                if (tid == 0 && cta == 0) { it_gna = na; it_gnc = nc; }
                __syncthreads();
#if DBG
                if (g_dbg && it == 0 && epoch == 0 && mb == 0 && cta == 0) {
                    w2_to_flat();
                    __syncthreads();
                    float* dd = g_dbg + (size_t)N_CTAS * 4 * FSTRIDE + (size_t)N_CTAS * (NFIELD + 1) * FSTRIDE;
                    for (int i = tid; i < N_PARAMS_PAD; i += THREADS) dd[VEC_STRIDE + i] = s_params[i];
                }
                if (g_dbg && it == 0 && cta == 0) {   // state after every update: params, Adam m / v, lr, KL
                    float* ds = g_dbg + (size_t)N_CTAS * 4 * FSTRIDE + (size_t)N_CTAS * (NFIELD + 1) * FSTRIDE +
                                VEC_STRIDE + 2 * N_PARAMS_PAD + (size_t)q_prof * (3 * N_PARAMS_PAD + 4);
                    w2_to_flat();
                    __syncthreads();
                    for (int i = tid; i < N_PARAMS_PAD; i += THREADS) ds[i] = s_params[i];
#pragma unroll
                    for (int kk = 0; kk < 8; ++kk) {
                        ds[N_PARAMS_PAD + OWN_W2(kk)] = m_w2[kk];
                        ds[2 * N_PARAMS_PAD + OWN_W2(kk)] = v_w2[kk];
                    }
                    ds[N_PARAMS_PAD + own_w1] = m_w1;
                    ds[2 * N_PARAMS_PAD + own_w1] = v_w1;
                    if (own_x >= 0) { ds[N_PARAMS_PAD + own_x] = m_x; ds[2 * N_PARAMS_PAD + own_x] = v_x; }
                    if (tid == 0) {   // learning rate as a full double (8-byte aligned slot), then the KL estimate
                        *reinterpret_cast<double*>(ds + 3 * N_PARAMS_PAD) = s_opt[0];
                        ds[3 * N_PARAMS_PAD + 2] = s_red[10];
                    }
                }
#endif
                load_weights();
                PROF_MARK(7 + 3 * q_prof);
            }
        }
        if (cta == 0 && tid == 0) {
            float* gs = g_stats + (size_t)git * NSTAT;
            const float nup = (float)(N_EPOCHS * N_MB);
            gs[ST_VLOSS] = it_vloss / nup;
            gs[ST_SURR] = it_surr / nup;
            gs[ST_ENTROPY] = it_ent / nup;
            gs[ST_KL] = it_kl;
            gs[ST_LR] = (float)s_opt[0];
            gs[ST_STD] = s_params[OFF_STD];
            gs[ST_GNORM_A] = it_gna;
            gs[ST_GNORM_C] = it_gnc;
        }
    }

    // ---------------- write back persistent state ----------------
    __syncthreads();
    if (cta == 0) {
        w2_to_flat();
        __syncthreads();
        for (int i = tid; i < N_PARAMS_PAD; i += THREADS) g_params[i] = s_params[i];
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) { g_adam_m[OWN_W2(kk)] = m_w2[kk]; g_adam_v[OWN_W2(kk)] = v_w2[kk]; }
        g_adam_m[own_w1] = m_w1; g_adam_v[own_w1] = v_w1;
        if (own_x >= 0) { g_adam_m[own_x] = m_x; g_adam_v[own_x] = v_x; }
        if (tid == 0) {
            g_scal[0] = s_opt[0];
            g_scal[1] = g_scal[1] + (double)(n_iters * N_EPOCHS * N_MB);
            g_scal[2] = s_opt[1];
            g_scal[3] = s_opt[2];
        }
    }
    for (int e = tid; e < E; e += THREADS) {
        float4 v;
        v.x = s_env[e * 4 + 0]; v.y = s_env[e * 4 + 1]; v.z = s_env[e * 4 + 2]; v.w = s_env[e * 4 + 3];
        reinterpret_cast<float4*>(g_env)[env0 + e] = v;
        g_eplen[env0 + e] = s_eplen[e];
        g_epret[env0 + e] = s_epret[e];
    }
}

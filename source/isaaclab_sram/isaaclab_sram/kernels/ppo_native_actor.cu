// Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

// Experimental native RGB64 actor: explicit resident W/G/Adam shards.
// Inputs, activations and cross-block reductions are global. Separate actor phase.
#include "device_lib.cuh"
#define WARPS (THREADS / 32)
#define FULL_MASK 0xffffffffu
#define CNN_WIDTH 5264
#define ACT_WIDTH 1618

__device__ __forceinline__ int input_dim(int l) {
    return l == 0 ? 699 : l == 1 ? 512 : l == 2 ? 256 : l == 3 ? 128 : l == 4 ? 192 : l == 5 ? 256 : l == 6 ? 288 : 0;
}
__device__ __forceinline__ int output_dim(int l) {
    return l == 0   ? 512
           : l == 1 ? 256
           : l == 2 ? 128
           : l == 3 ? 23
           : l == 4 ? 16
           : l == 5 ? 32
           : l == 6 ? 32
           : l == 7 ? 32
                    : 23;
}
// Each CTA retains one replica of one filter in each convolution.
// MLP, temperature and std ownership remains cyclic and canonical.
__device__ __forceinline__ int row_count(int l, int c) {
    return l >= 4 && l <= 6 ? 1 : (output_dim(l) + N_CTAS - 1 - c) / N_CTAS;
}
__device__ __forceinline__ int shard_base(int l, int c) {
    int b = 0;
    for (int k = 0; k < l; k++) b += row_count(k, c) * (input_dim(k) + 1);
    return b;
}
__device__ __forceinline__ int flat_base(int l) {
    return l == 0   ? 0
           : l == 1 ? 358400
           : l == 2 ? 489728
           : l == 3 ? 522624
           : l == 4 ? 525591
           : l == 5 ? 528679
           : l == 6 ? 536903
           : l == 7 ? 546151
                    : 546183;
}
__device__ __forceinline__ int activation_base(int l) {
    return TILE * (l < 0 ? 0 : l == 0 ? 699 : l == 1 ? 1211 : l == 2 ? 1467 : 1595);
}
__device__ __forceinline__ int cnn_activation_base(int l) { return TILE * (l == 0 ? 0 : l == 1 ? 3600 : 4752); }
__device__ __forceinline__ int cnn_input_channels(int l) { return l == 0 ? 3 : l == 1 ? 16 : 32; }
__device__ __forceinline__ int cnn_input_height(int l) { return l == 0 ? 64 : l == 1 ? 15 : 6; }
__device__ __forceinline__ int cnn_output_height(int l) { return l == 0 ? 15 : l == 1 ? 6 : 4; }
__device__ __forceinline__ int cnn_kernel_size(int l) { return l == 0 ? 8 : l == 1 ? 4 : 3; }
__device__ __forceinline__ int cnn_stride(int l) { return l == 0 ? 4 : l == 1 ? 2 : 1; }
__device__ __forceinline__ float warp_sum(float v) {
    for (int d = 16; d; d >>= 1) v += __shfl_down_sync(FULL_MASK, v, d);
    return v;
}
__device__ __forceinline__ double warp_sum_double(double v) {
    for (int d = 16; d; d >>= 1) v += __shfl_down_sync(FULL_MASK, v, d);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int d = 16; d; d >>= 1) v = fmaxf(v, __shfl_down_sync(FULL_MASK, v, d));
    return v;
}

// RGB normalization matches native addcmul(-.5, uint8, 1/255).
template <int L>
__device__ __forceinline__ float cnn_input(const unsigned char* images, const i64* perm, int batch_offset, int sample,
                                           int channel, int y, int x, const float* cnn) {
    if constexpr (L == 0)
        return fmaf(float(images[((perm[batch_offset + sample] * 3 + channel) * 64 + y) * 64 + x]), 1.f / 255.f, -.5f);
    else
        return cnn[cnn_activation_base(L - 1) +
                   (sample * cnn_input_channels(L) + channel) * cnn_input_height(L) * cnn_input_height(L) +
                   y * cnn_input_height(L) + x];
}

// Replica groups own output filters; their warps divide sample/pixel work.
template <int L>
__device__ __forceinline__ void cnn_forward(const float* w, const unsigned char* images, const i64* perm, int offset,
                                            int count, float* cnn, u32* bar) {
    const int c = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5, channels = output_dim(L + 4),
              h = cnn_output_height(L), k = cnn_kernel_size(L), inputs = input_dim(L + 4);
    const int oc = c % channels, replica = c / channels;
    const int replicas = (N_CTAS + channels - 1 - oc) / channels;
    for (int pair = replica * WARPS + warp; pair < count * h * h; pair += replicas * WARPS) {
        int sample = pair / (h * h), pixel = pair % (h * h);
        const float* rw = w + shard_base(L + 4, c);
        float v = 0.f;
        for (int j = lane; j < inputs; j += 32) {
            int ic = j / (k * k), ky = (j / k) % k, kx = j % k;
            v = fmaf(rw[j],
                     cnn_input<L>(images, perm, offset, sample, ic, (pixel / h) * cnn_stride(L) + ky,
                                  (pixel % h) * cnn_stride(L) + kx, cnn),
                     v);
        }
        v = warp_sum(v);
        if (lane == 0) {
            v += rw[inputs];
            cnn[cnn_activation_base(L) + (sample * channels + oc) * h * h + pixel] = v > 0.f ? v : expm1f(v);
        }
    }
    grid_barrier(bar);
}

template <int L>
__device__ __forceinline__ void cnn_gradient(float* g, const unsigned char* images, const i64* perm, int offset,
                                             int count, const float* cnn, const float* cd) {
    const int c = blockIdx.x, inputs = input_dim(L + 4), h = cnn_output_height(L), k = cnn_kernel_size(L),
              channels = output_dim(L + 4), b = shard_base(L + 4, c);
    const int oc = c % channels, replica = c / channels;
    const int replicas = (N_CTAS + channels - 1 - oc) / channels;
    for (int j = threadIdx.x; j < inputs + 1; j += THREADS) {
        float v = g[b + j];
        for (int point = replica; point < count * h * h; point += replicas) {
            int sample = point / (h * h), y = (point / h) % h, x = point % h;
            float xv = j == inputs ? 1.f
                                   : cnn_input<L>(images, perm, offset, sample, j / (k * k),
                                                  y * cnn_stride(L) + (j / k) % k, x * cnn_stride(L) + j % k, cnn);
            v = fmaf(xv, cd[cnn_activation_base(L) + (sample * channels + oc) * h * h + y * h + x], v);
        }
        g[b + j] = v;
    }
}

// Owners publish input-adjoint partials, followed by a fixed-order reduction.
// No raw image gradient is needed because uint8 observations are constant.
template <int L>
__device__ __forceinline__ void cnn_backward(const float* w, const float* cnn, float* cd, float* partial, int count,
                                             u32* bar) {
    const int c = blockIdx.x, inputs = input_dim(L + 4), h = cnn_output_height(L), hin = cnn_input_height(L),
              channels = output_dim(L + 4), cin = cnn_input_channels(L), k = cnn_kernel_size(L), s = cnn_stride(L);
    const int width = cin * hin * hin, b = shard_base(L + 4, c);
    const int oc = c % channels, replica = c / channels;
    const int replicas = (N_CTAS + channels - 1 - oc) / channels;
    for (int p = replica * THREADS + threadIdx.x; p < count * width; p += replicas * THREADS) {
        int sample = p / width, rem = p % width, ic = rem / (hin * hin), y = (rem / hin) % hin, x = rem % hin;
        float v = 0.f;
        for (int r = 0; r < row_count(L + 4, c); r++)
            for (int ky = 0; ky < k; ky++)
                for (int kx = 0; kx < k; kx++) {
                    int oy = y - ky, ox = x - kx;
                    if (oy >= 0 && ox >= 0 && oy % s == 0 && ox % s == 0 && oy / s < h && ox / s < h)
                        v = fmaf(w[b + r * (inputs + 1) + (ic * k + ky) * k + kx],
                                 cd[cnn_activation_base(L) + (sample * channels + oc) * h * h + (oy / s) * h + ox / s],
                                 v);
                }
        partial[(oc * TILE + sample) * width + rem] = v;
    }
    grid_barrier(bar);
    for (int p = c * THREADS + threadIdx.x; p < count * width; p += N_CTAS * THREADS) {
        int sample = p / width, j = p % width;
        float v = 0.f;
        for (int owner = 0; owner < min(N_CTAS, channels); owner++) v += partial[(owner * TILE + sample) * width + j];
        float activation = cnn[cnn_activation_base(L - 1) + p];
        cd[cnn_activation_base(L - 1) + p] = v * (activation > 0.f ? 1.f : activation + 1.f);
    }
    grid_barrier(bar);
}

__device__ __forceinline__ void spatial_forward(const float* w, const float* states, const i64* perm, int offset,
                                                int count, const float* cnn, const float* xy, float* probs, float* acts,
                                                u32* bar) {
    const int c = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    for (int p = c * THREADS + t; p < count * 635; p += N_CTAS * THREADS)
        acts[(p / 635) * 699 + p % 635] = states[perm[offset + p / 635] * 635 + p % 635];
    if (c < 32)
        for (int sample = warp; sample < count; sample += WARPS) {
            const float temp = fmaxf(expf(w[shard_base(7, c)]), 1e-3f);
            float z = lane < 16 ? cnn[cnn_activation_base(2) + (sample * 32 + c) * 16 + lane] / temp
                                : -__int_as_float(0x7f800000);
            float mx = warp_max(z);
            mx = __shfl_sync(FULL_MASK, mx, 0);
            float p = lane < 16 ? expf(z - mx) : 0.f;
            float norm = warp_sum(p);
            norm = __shfl_sync(FULL_MASK, norm, 0);
            p /= norm;
            if (lane < 16) probs[(sample * 32 + c) * 16 + lane] = p;
            float x = warp_sum(lane < 16 ? p * xy[lane] : 0.f), y = warp_sum(lane < 16 ? p * xy[16 + lane] : 0.f);
            if (lane == 0) {
                acts[sample * 699 + 635 + c] = x;
                acts[sample * 699 + 667 + c] = y;
            }
        }
    grid_barrier(bar);
}

// Coordinate order matches native: all32 x coordinates, then all32 y.
// The learned log-temperature contributes to the same actor-wide norm.
__device__ __forceinline__ void spatial_backward(const float* w, float* g, const float* cnn, const float* probs,
                                                 const float* acts, const float* deltas, const float* xy, float* cd,
                                                 int count, u32* bar) {
    int c = blockIdx.x;
    if (c < 32 && threadIdx.x == 0) {
        const float raw = expf(w[shard_base(7, c)]), temp = fmaxf(raw, 1e-3f);
        float tg = g[shard_base(7, c)];
        for (int sample = 0; sample < count; sample++) {
            float x = acts[sample * 699 + 635 + c], y = acts[sample * 699 + 667 + c],
                  dx = deltas[sample * 699 + 635 + c], dy = deltas[sample * 699 + 667 + c];
            for (int pixel = 0; pixel < 16; pixel++) {
                int index = (sample * 32 + c) * 16 + pixel;
                float feature = cnn[cnn_activation_base(2) + index];
                float d = probs[index] * (dx * (xy[pixel] - x) + dy * (xy[16 + pixel] - y)) / temp;
                if (raw >= 1e-3f) tg -= d * feature;
                cd[cnn_activation_base(2) + index] = d * (feature > 0.f ? 1.f : feature + 1.f);
            }
        }
        g[shard_base(7, c)] = tg;
    }
    grid_barrier(bar);
}

template <int L>
__device__ __forceinline__ void mlp_forward(const float* w, float* acts, int count, u32* bar) {
    int c = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5, b = shard_base(L, c);
    for (int p = warp; p < row_count(L, c) * count; p += WARPS) {
        int row = p / count, sample = p % count;
        float v = 0.f;
        const float* rw = w + b + row * (input_dim(L) + 1);
        for (int j = lane; j < input_dim(L); j += 32)
            v = fmaf(rw[j], acts[activation_base(L - 1) + sample * input_dim(L) + j], v);
        v = warp_sum(v);
        if (lane == 0) {
            v += rw[input_dim(L)];
            acts[activation_base(L) + sample * output_dim(L) + c + row * N_CTAS] = L == 3 || v > 0.f ? v : expm1f(v);
        }
    }
    grid_barrier(bar);
}

template <int L>
__device__ __forceinline__ void mlp_gradient(float* g, const float* acts, const float* d, int count) {
    int c = blockIdx.x, b = shard_base(L, c);
    for (int p = threadIdx.x; p < row_count(L, c) * (input_dim(L) + 1); p += THREADS) {
        int row = p / (input_dim(L) + 1), j = p % (input_dim(L) + 1);
        float v = g[b + p];
        for (int sample = 0; sample < count; sample++)
            v = fmaf(d[activation_base(L) + sample * output_dim(L) + c + row * N_CTAS],
                     j == input_dim(L) ? 1.f : acts[activation_base(L - 1) + sample * input_dim(L) + j], v);
        g[b + p] = v;
    }
}

// Layer0 needs only visual-coordinate adjoints; state observations are fixed.
template <int L>
__device__ __forceinline__ void mlp_backward(const float* w, const float* acts, float* d, float* partial, int count,
                                             u32* bar) {
    int c = blockIdx.x, b = shard_base(L, c);
    const int width = L == 0 ? 64 : input_dim(L), offset = L == 0 ? 635 : 0;
    if (row_count(L, c))
        for (int p = threadIdx.x; p < count * width; p += THREADS) {
            int sample = p / width, j = p % width + offset;
            float v = 0.f;
            for (int row = 0; row < row_count(L, c); row++)
                v = fmaf(w[b + row * (input_dim(L) + 1) + j],
                         d[activation_base(L) + sample * output_dim(L) + c + row * N_CTAS], v);
            partial[(c * TILE + sample) * width + p % width] = v;
        }
    grid_barrier(bar);
    for (int p = c * THREADS + threadIdx.x; p < count * width; p += N_CTAS * THREADS) {
        int sample = p / width, j = p % width;
        float v = 0.f;
        for (int owner = 0; owner < min(N_CTAS, output_dim(L)); owner++)
            v += partial[(owner * TILE + sample) * width + j];
        int index = activation_base(L - 1) + sample * input_dim(L) + j + offset;
        float a = acts[index];
        d[index] = L == 0 ? v : v * (a > 0.f ? 1.f : a + 1.f);
    }
    grid_barrier(bar);
}

__device__ __forceinline__ void ppo_gradient(const float* w, float* g, const float* acts, float* d,
                                             const float* actions, const float* oldlp, const float* advantages,
                                             const i64* perm, int offset, int count, int mini, float clip,
                                             float entropy_coef, float std_min, float std_max, const float* stds,
                                             float* std_grads, float* sample_metrics, u32* bar) {
    if (blockIdx.x == 0) {
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        for (int sample = warp; sample < count; sample += WARPS) {
            i64 index = perm[offset + sample];
            float mu = lane < 23 ? acts[activation_base(3) + sample * 23 + lane] : 0.f;
            float sd = lane < 23 ? stds[lane] : 1.f;
            float delta = lane < 23 ? actions[index * 23 + lane] - mu : 0.f;
            float lp = lane < 23 ? -(delta * delta) / (2.f * sd * sd) - logf(sd) - .9189385332046727f : 0.f;
            float ent = lane < 23 ? logf(sd) + 1.4189385332046727f : 0.f;
            lp = warp_sum(lp);
            lp = __shfl_sync(FULL_MASK, lp, 0);
            ent = warp_sum(ent);
            float ratio = expf(lp - oldlp[index]), adv = advantages[index], sur = -adv * ratio;
            float clipped_ratio = fminf(fmaxf(ratio, 1.f - clip), 1.f + clip), sc = -adv * clipped_ratio;
            float active = (ratio >= 1.f - clip && ratio <= 1.f + clip) ? 1.f : 0.f;
            float dl = (sur > sc ? -adv : sur < sc ? -adv * active : -.5f * adv * (1.f + active)) * ratio / mini;
            if (lane < 23) {
                d[activation_base(3) + sample * 23 + lane] = dl * delta / (sd * sd);
                std_grads[sample * 23 + lane] =
                    dl * ((delta * delta) / (sd * sd * sd) - 1.f / sd) - entropy_coef / (mini * sd);
            }
            if (lane == 0) {
                sample_metrics[sample * 2] = fmaxf(sur, sc) / mini;
                sample_metrics[sample * 2 + 1] = ent / mini;
            }
        }
    }
    grid_barrier(bar);
    const int c = blockIdx.x;
    if (c < 23 && threadIdx.x == 0) {
        float raw = w[shard_base(8, c)], grad = g[shard_base(8, c)];
        if (raw >= std_min && raw <= std_max)
            for (int sample = 0; sample < count; sample++) grad += std_grads[sample * 23 + c];
        g[shard_base(8, c)] = grad;
    }
}

// Accumulate a complete native minibatch before reducing replica gradients.
// Reuse the adjoint mailbox only after the last microtile's reader barrier.
// All replicas receive the same fp32 sum and therefore apply identical Adam.
__device__ __forceinline__ void reduce_cnn_gradients(float* gradients, float* partial, u32* barrier) {
    const int c = blockIdx.x, t = threadIdx.x;
    for (int layer = 4; layer <= 6; layer++) {
        const int offset = layer == 4 ? 0 : layer == 5 ? 193 : 450;
        for (int j = t; j <= input_dim(layer); j += THREADS)
            partial[c * 739 + offset + j] = gradients[shard_base(layer, c) + j];
    }
    grid_barrier(barrier);
    for (int layer = 4; layer <= 6; layer++) {
        const int channels = output_dim(layer), channel = c % channels;
        const int offset = layer == 4 ? 0 : layer == 5 ? 193 : 450;
        for (int j = t; j <= input_dim(layer); j += THREADS) {
            float sum = 0.f;
            for (int owner = channel; owner < N_CTAS; owner += channels) sum += partial[owner * 739 + offset + j];
            gradients[shard_base(layer, c) + j] = sum;
        }
    }
    grid_barrier(barrier);
}

extern "C" __global__ __launch_bounds__(THREADS, 1) void persistent_native_actor(
    float* params, float* adam_m, float* adam_v, float* last_grad, const float* state, const unsigned char* images,
    const float* actions, const float* oldlp, const float* advantages, const i64* perm, const float* xy,
    const float* adam_corrections, int mini, int epochs, int minibatches, float clip, float entropy_coef,
    float max_norm, float std_min, float std_max, float beta1, float beta1c, float beta2, float beta2c, float epsilon,
    int fused_adam, float* cnn, float* cd, float* probs, float* acts, float* d, float* partial, double* norms,
    float* stds, float* std_grads, float* sample_metrics, float* statistics, float* mean_metrics, u32* bar) {
    extern __shared__ float storage[];
    float* w = storage;
    float* g = w + SHARD_SIZE;
    float* m = g + SHARD_SIZE;
    float* v = m + SHARD_SIZE;
    __shared__ double warp_norm[WARPS];
    __shared__ float clipping, corrections[32];
    int c = blockIdx.x, t = threadIdx.x, owned = shard_base(9, c);
    // Import native checkpoint state once, before all PPO minibatches.
    for (int l = 0; l < 9; l++)
        for (int p = t; p < row_count(l, c) * (input_dim(l) + 1); p += THREADS) {
            int row = l >= 4 && l <= 6 ? c % output_dim(l) : c + (p / (input_dim(l) + 1)) * N_CTAS,
                j = p % (input_dim(l) + 1),
                index =
                    flat_base(l) + (j == input_dim(l) ? output_dim(l) * input_dim(l) + row : row * input_dim(l) + j),
                b = shard_base(l, c) + p;
            w[b] = params[index];
            m[b] = adam_m[index];
            v[b] = adam_v[index];
        }
    __syncthreads();
    float total_sur = 0.f, total_ent = 0.f;
    // Keep W/G/m/v in explicit shared shards for the complete update.
    for (int epoch = 0; epoch < epochs; epoch++)
        for (int mb = 0; mb < minibatches; mb++) {
            int update = epoch * minibatches + mb;
            for (int p = t; p < owned; p += THREADS) g[p] = 0.f;
            if (c < 23 && t == 0) stds[c] = fminf(fmaxf(w[shard_base(8, c)], std_min), std_max);
            grid_barrier(bar);
            float mini_sur = 0.f, mini_ent = 0.f;
            for (int start = 0; start < mini; start += TILE) {
                int count = min(TILE, mini - start), offset = mb * mini + start;
                cnn_forward<0>(w, images, perm, offset, count, cnn, bar);
                cnn_forward<1>(w, images, perm, offset, count, cnn, bar);
                cnn_forward<2>(w, images, perm, offset, count, cnn, bar);
                spatial_forward(w, state, perm, offset, count, cnn, xy, probs, acts, bar);
                mlp_forward<0>(w, acts, count, bar);
                mlp_forward<1>(w, acts, count, bar);
                mlp_forward<2>(w, acts, count, bar);
                mlp_forward<3>(w, acts, count, bar);
                ppo_gradient(w, g, acts, d, actions, oldlp, advantages, perm, offset, count, mini, clip, entropy_coef,
                             std_min, std_max, stds, std_grads, sample_metrics, bar);
                if (c == 0 && t == 0)
                    for (int sample = 0; sample < count; sample++) {
                        mini_sur += sample_metrics[2 * sample];
                        mini_ent += sample_metrics[2 * sample + 1];
                    }
                mlp_gradient<3>(g, acts, d, count);
                mlp_backward<3>(w, acts, d, partial, count, bar);
                mlp_gradient<2>(g, acts, d, count);
                mlp_backward<2>(w, acts, d, partial, count, bar);
                mlp_gradient<1>(g, acts, d, count);
                mlp_backward<1>(w, acts, d, partial, count, bar);
                mlp_gradient<0>(g, acts, d, count);
                mlp_backward<0>(w, acts, d, partial, count, bar);
                spatial_backward(w, g, cnn, probs, acts, d, xy, cd, count, bar);
                cnn_gradient<2>(g, images, perm, offset, count, cnn, cd);
                cnn_backward<2>(w, cnn, cd, partial, count, bar);
                cnn_gradient<1>(g, images, perm, offset, count, cnn, cd);
                cnn_backward<1>(w, cnn, cd, partial, count, bar);
                cnn_gradient<0>(g, images, perm, offset, count, cnn, cd);
                grid_barrier(bar);
            }
            reduce_cnn_gradients(g, partial, bar);
            double n = 0.;
            for (int layer = 0; layer < 9; layer++) {
                // Each logical CNN gradient contributes exactly once to norm.
                if (layer >= 4 && layer <= 6 && c >= output_dim(layer)) continue;
                for (int p = t; p < row_count(layer, c) * (input_dim(layer) + 1); p += THREADS) {
                    double gv = g[shard_base(layer, c) + p];
                    n += gv * gv;
                }
            }
            n = warp_sum_double(n);
            if ((t & 31) == 0) warp_norm[t >> 5] = n;
            __syncthreads();
            if (t == 0) {
                double q = 0.;
                for (int j = 0; j < WARPS; j++) q += warp_norm[j];
                norms[c] = q;
            }
            grid_barrier(bar);
            if (t == 0) {
                double q = 0.;
                for (int owner = 0; owner < N_CTAS; owner++) q += norms[owner];
                float norm = (float)sqrt(q);
                clipping = fminf(max_norm / (norm + 1e-6f), 1.f);
                if (c == 0) {
                    statistics[update * 3] = mini_sur;
                    statistics[update * 3 + 1] = mini_ent;
                    statistics[update * 3 + 2] = norm;
                    total_sur += mini_sur;
                    total_ent += mini_ent;
                }
            }
            if (t < 16) {
                float a = adam_corrections[(update * 16 + t) * 2], b = adam_corrections[(update * 16 + t) * 2 + 1];
                corrections[t * 2] = fused_adam ? a / (1.f - powf(beta1, b)) : a;
                corrections[t * 2 + 1] = fused_adam ? sqrtf(1.f - powf(beta2, b)) : b;
            }
            __syncthreads();
            for (int l = 0; l < 9; l++)
                for (int p = t; p < row_count(l, c) * (input_dim(l) + 1); p += THREADS) {
                    int b = shard_base(l, c) + p,
                        tensor = l < 7 ? 2 * l + ((p % (input_dim(l) + 1)) == input_dim(l)) : l + 7;
                    float grad = g[b] * clipping;
                    g[b] = grad;
                    if (fused_adam) {
                        m[b] = fmaf(beta1, m[b], fmaf(-beta1, grad, grad));
                        float square = __fmul_rn(grad, grad);
                        v[b] = fmaf(beta2, v[b], fmaf(-beta2, square, square));
                    } else {
                        m[b] = fmaf(beta1c, grad - m[b], m[b]);
                        v[b] = fmaf(__fmul_rn(beta2c, grad), grad, __fmul_rn(v[b], beta2));
                    }
                    float denom = sqrtf(v[b]) / corrections[tensor * 2 + 1] + epsilon, neg = corrections[tensor * 2];
                    w[b] = fused_adam ? __fsub_rn(w[b], __fmul_rn(-neg, m[b]) / denom) : fmaf(neg, m[b] / denom, w[b]);
                }
            grid_barrier(bar);
        }
    // Only canonical owners export; replicas preserve identical state.
    for (int l = 0; l < 9; l++) {
        if (l >= 4 && l <= 6 && c >= output_dim(l)) continue;
        for (int p = t; p < row_count(l, c) * (input_dim(l) + 1); p += THREADS) {
            int row = l >= 4 && l <= 6 ? c % output_dim(l) : c + (p / (input_dim(l) + 1)) * N_CTAS,
                j = p % (input_dim(l) + 1),
                index =
                    flat_base(l) + (j == input_dim(l) ? output_dim(l) * input_dim(l) + row : row * input_dim(l) + j),
                b = shard_base(l, c) + p;
            params[index] = w[b];
            adam_m[index] = m[b];
            adam_v[index] = v[b];
            last_grad[index] = g[b];
        }
    }
    if (c == 0 && t == 0) {
        mean_metrics[0] = total_sur / (epochs * minibatches);
        mean_metrics[1] = total_ent / (epochs * minibatches);
    }
}

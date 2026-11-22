// Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

// Unified native visual learner. Wc/Wa/G share one arena;
// both networks' m/v are imported before either phase and exported after both.
// Rollouts, activations and communication remain global; not all-SRAM.
#include "device_lib.cuh"
#ifndef MAX_REGS
#define MAX_REGS 224
#endif
namespace critic {
// Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

// Native 1595 -> 512 -> 256 -> 128 -> 1 ELU critic.
// Each block owns complete output-neuron rows, cyclically distributed over the grid.
// Weights and accumulated gradients remain in shared memory; both Adam moments
// remain in statically indexed per-thread register arrays for every minibatch. Activations, input-adjoint partials, synchronization,
// scalar Adam corrections and statistics use global memory. This is not zero-DRAM
// execution, nor a resident full visual learner. Boundary copies use global memory.


#define CRITIC_FULL_MASK 0xffffffffu
#define CRITIC_WARPS (THREADS / 32)
#define CRITIC_ACT_WIDTH 897
#define CRITIC_PART_WIDTH 512
#define CRITIC_MOMENT_SLOTS ((CRITIC_SHARD_SIZE + THREADS - 1) / THREADS)

__device__ __forceinline__ int in_dim(int layer) {
    return layer == 0 ? 1595 : (layer == 1 ? 512 : (layer == 2 ? 256 : 128));
}
__device__ __forceinline__ int out_dim(int layer) {
    return layer == 0 ? 512 : (layer == 1 ? 256 : (layer == 2 ? 128 : 1));
}
__device__ __forceinline__ int row_count(int layer, int cta) {
    return (out_dim(layer) + N_CTAS - 1 - cta) / N_CTAS;
}
__device__ __forceinline__ int shard_base(int layer, int cta) {
    int base = 0;
    for (int l = 0; l < layer; ++l) base += row_count(l, cta) * (in_dim(l) + 1);
    return base;
}
__device__ __forceinline__ int flat_base(int layer) {
    return layer == 0 ? 0 : (layer == 1 ? 817152 : (layer == 2 ? 948480 : 981376));
}
// Map contiguous shared-shard ownership into the unchanged native checkpoint.
// This helper never indexes the register arrays. All moment accesses occur in
// fully unrolled q loops so q is a compile-time constant for each access.
__device__ __forceinline__ int shard_layer(int p, int cta) {
    return p < shard_base(1, cta) ? 0 :
           (p < shard_base(2, cta) ? 1 : (p < shard_base(3, cta) ? 2 : 3));
}
__device__ __forceinline__ int shard_tensor(int p, int cta) {
    const int layer = shard_layer(p, cta);
    const int ni = in_dim(layer), k = p - shard_base(layer, cta);
    return 2 * layer + (k % (ni + 1) == ni);
}
__device__ __forceinline__ int shard_native_index(int p, int cta) {
    const int layer = shard_layer(p, cta);
    const int ni = in_dim(layer), no = out_dim(layer);
    const int k = p - shard_base(layer, cta);
    const int row = cta + (k / (ni + 1)) * N_CTAS, j = k % (ni + 1);
    return flat_base(layer) + (j == ni ? no * ni + row : row * ni + j);
}
__device__ __forceinline__ int act_base(int layer) {
    return CRITIC_TILE * (layer == 0 ? 0 : (layer == 1 ? 512 : (layer == 2 ? 768 : 896)));
}
__device__ __forceinline__ float warp_sum(float value) {
    for (int d = 16; d; d >>= 1) value += __shfl_down_sync(CRITIC_FULL_MASK, value, d);
    return value;
}
__device__ __forceinline__ double warp_sum_double(double value) {
    for (int d = 16; d; d >>= 1) value += __shfl_down_sync(CRITIC_FULL_MASK, value, d);
    return value;
}

template<int L>
__device__ __forceinline__ void forward_layer(
    const float* weights, const float* inputs, const i64* indices, int batch_offset,
    int count, float* acts, u32* barrier
) {
    const int cta = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int ni = in_dim(L), no = out_dim(L), nr = row_count(L, cta);
    const int base = shard_base(L, cta);
    for (int pair = warp; pair < nr * count; pair += CRITIC_WARPS) {
        const int local_row = pair / count, sample = pair % count;
        const int output_row = cta + local_row * N_CTAS;
        const float* row_weights = weights + base + local_row * (ni + 1);
        const float* x = L == 0 ? inputs + indices[batch_offset + sample] * 1595
                                : acts + act_base(L - 1) + sample * ni;
        float sum = 0.f;
        for (int j = lane; j < ni; j += 32) sum = fmaf(row_weights[j], x[j], sum);
        sum = warp_sum(sum);
        if (lane == 0) {
            const float z = sum + row_weights[ni];
            acts[act_base(L) + sample * no + output_row] = (L == 3 || z > 0.f) ? z : expm1f(z);
        }
    }
    grid_barrier(barrier);
}

template<int L>
__device__ __forceinline__ void accumulate_gradient(
    float* gradients, const float* inputs, const i64* indices, int batch_offset,
    int count, const float* acts, const float* deltas
) {
    const int cta = blockIdx.x, ni = in_dim(L), no = out_dim(L), nr = row_count(L, cta);
    const int base = shard_base(L, cta);
    for (int k = threadIdx.x; k < nr * (ni + 1); k += THREADS) {
        const int local_row = k / (ni + 1), j = k % (ni + 1);
        const int output_row = cta + local_row * N_CTAS;
        float sum = gradients[base + k];
        for (int sample = 0; sample < count; ++sample) {
            const float x = j == ni ? 1.f :
                (L == 0 ? inputs[indices[batch_offset + sample] * 1595 + j]
                        : acts[act_base(L - 1) + sample * ni + j]);
            sum = fmaf(deltas[act_base(L) + sample * no + output_row], x, sum);
        }
        gradients[base + k] = sum;
    }
}

template<int L>
__device__ __forceinline__ void backward_layer(
    const float* weights, const float* acts, float* deltas, float* partials,
    int count, u32* barrier
) {
    const int cta = blockIdx.x, ni = in_dim(L), no = out_dim(L), nr = row_count(L, cta);
    const int base = shard_base(L, cta);
    // Each owning block writes an input-adjoint partial, never a weight replica.
    if (nr) {
        for (int p = threadIdx.x; p < count * ni; p += THREADS) {
            const int sample = p / ni, j = p % ni;
            float sum = 0.f;
            for (int r = 0; r < nr; ++r) {
                sum = fmaf(weights[base + r * (ni + 1) + j],
                           deltas[act_base(L) + sample * no + cta + r * N_CTAS], sum);
            }
            partials[(cta * CRITIC_TILE + sample) * CRITIC_PART_WIDTH + j] = sum;
        }
    }
    grid_barrier(barrier);
    for (int p = cta * THREADS + threadIdx.x; p < count * ni; p += N_CTAS * THREADS) {
        const int sample = p / ni, j = p % ni;
        float sum = 0.f;
        for (int owner = 0; owner < min(N_CTAS, no); ++owner)
            sum += partials[(owner * CRITIC_TILE + sample) * CRITIC_PART_WIDTH + j];
        const float activation = acts[act_base(L - 1) + p];
        deltas[act_base(L - 1) + p] = sum * (activation > 0.f ? 1.f : activation + 1.f);
    }
    grid_barrier(barrier);
}


__device__ __forceinline__ void phase(
float* params, float* adam_m, float* adam_v, float* last_gradients,
    const float* inputs, const float* old_values, const float* returns, const i64* indices,
    const float* adam_corrections, int minibatch_size, int epochs, int minibatches,
    float clip, float value_coef, float max_grad_norm, float beta1, float beta1_complement,
    float beta2, float beta2_complement, float epsilon, int fused_adam,
    float* acts, float* deltas, float* partials, double* norm_partials,
    float* sample_losses, float* statistics, float* mean_loss, u32* barrier,
float* weights, float* gradients, float (&moments)[CRITIC_MOMENT_SLOTS], float (&variances)[CRITIC_MOMENT_SLOTS]) {
    __shared__ double warp_norms[CRITIC_WARPS];
    __shared__ float clipping;
    __shared__ float corrections[16];
    const int tid = threadIdx.x, cta = blockIdx.x;
    const int owned = shard_base(4, cta);
    float accumulated_loss = 0.f;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int mb = 0; mb < minibatches; ++mb) {
            const int update = epoch * minibatches + mb;
            for (int k = tid; k < owned; k += THREADS) gradients[k] = 0.f;
            __syncthreads();
            float minibatch_loss = 0.f;
            for (int start = 0; start < minibatch_size; start += CRITIC_TILE) {
                const int count = min(CRITIC_TILE, minibatch_size - start);
                const int batch_offset = mb * minibatch_size + start;
                forward_layer<0>(weights, inputs, indices, batch_offset, count, acts, barrier);
                forward_layer<1>(weights, inputs, indices, batch_offset, count, acts, barrier);
                forward_layer<2>(weights, inputs, indices, batch_offset, count, acts, barrier);
                forward_layer<3>(weights, inputs, indices, batch_offset, count, acts, barrier);
                if (cta == 0) {
                    for (int sample = tid; sample < count; sample += THREADS) {
                        const i64 index = indices[batch_offset + sample];
                        const float v = acts[act_base(3) + sample];
                        const float old = old_values[index], target = returns[index], diff = v - old;
                        const float vc = old + fminf(fmaxf(diff, -clip), clip);
                        const float d = v - target, dc = vc - target;
                        const float l = d * d, lc = dc * dc;
                        const float pass = (diff >= -clip && diff <= clip) ? 1.f : 0.f;
                        // torch.maximum splits ties; torch.clamp includes both exact boundaries.
                        const float derivative = l > lc ? 2.f * d :
                            (l < lc ? 2.f * dc * pass : d + dc * pass);
                        deltas[act_base(3) + sample] = derivative * (value_coef / minibatch_size);
                        sample_losses[sample] = fmaxf(l, lc) / minibatch_size;
                    }
                }
                grid_barrier(barrier);
                if (cta == 0 && tid == 0)
                    for (int sample = 0; sample < count; ++sample) minibatch_loss += sample_losses[sample];
                accumulate_gradient<3>(gradients, inputs, indices, batch_offset, count, acts, deltas);
                backward_layer<3>(weights, acts, deltas, partials, count, barrier);
                accumulate_gradient<2>(gradients, inputs, indices, batch_offset, count, acts, deltas);
                backward_layer<2>(weights, acts, deltas, partials, count, barrier);
                accumulate_gradient<1>(gradients, inputs, indices, batch_offset, count, acts, deltas);
                backward_layer<1>(weights, acts, deltas, partials, count, barrier);
                accumulate_gradient<0>(gradients, inputs, indices, batch_offset, count, acts, deltas);
                // All readers finish before the next tile overwrites global activations.
                grid_barrier(barrier);
            }
            double norm_sum = 0.;
            for (int k = tid; k < owned; k += THREADS) {
                const double g = gradients[k];
                norm_sum += g * g;
            }
            norm_sum = warp_sum_double(norm_sum);
            if ((tid & 31) == 0) warp_norms[tid >> 5] = norm_sum;
            __syncthreads();
            if (tid == 0) {
                double sum = 0.;
                for (int w = 0; w < CRITIC_WARPS; ++w) sum += warp_norms[w];
                norm_partials[cta] = sum;
            }
            grid_barrier(barrier);
            if (tid == 0) {
                double sum = 0.;
                for (int owner = 0; owner < N_CTAS; ++owner) sum += norm_partials[owner];
                const float norm = (float)sqrt(sum);
                clipping = fminf(max_grad_norm / (norm + 1e-6f), 1.f);
                if (cta == 0) {
                    statistics[2 * update] = minibatch_loss;
                    statistics[2 * update + 1] = norm;
                    accumulated_loss += minibatch_loss;
                }
            }
            if (tid < 8) {
                const float first = adam_corrections[(update * 8 + tid) * 2];
                const float second = adam_corrections[(update * 8 + tid) * 2 + 1];
                corrections[tid * 2] = fused_adam ? first / (1.f - powf(beta1, second)) : first;
                corrections[tid * 2 + 1] = fused_adam ? sqrtf(1.f - powf(beta2, second)) : second;
            }
            __syncthreads();
            // Each register slot retains its own native parameter's moments
            // across all optimizer steps. The tensor identity still chooses
            // the original per-parameter checkpoint step/bias corrections.
            #pragma unroll
            for (int q = 0; q < CRITIC_MOMENT_SLOTS; ++q) {
                const int p = tid + q * THREADS;
                if (p < owned) {
                    const int tensor = shard_tensor(p, cta);
                    const float neg_step = corrections[tensor * 2];
                    const float bc2_sqrt = corrections[tensor * 2 + 1];
                    const float g = gradients[p] * clipping;
                    gradients[p] = g;
                    if (fused_adam) {
                        moments[q] = fmaf(beta1, moments[q], fmaf(-beta1, g, g));
                        const float square = __fmul_rn(g, g);
                        variances[q] = fmaf(beta2, variances[q], fmaf(-beta2, square, square));
                    } else {
                        moments[q] = fmaf(beta1_complement, g - moments[q], moments[q]);
                        const float decayed = __fmul_rn(variances[q], beta2);
                        variances[q] = fmaf(__fmul_rn(beta2_complement, g), g, decayed);
                    }
                    const float denom = sqrtf(variances[q]) / bc2_sqrt + epsilon;
                    weights[p] = fused_adam
                        ? __fsub_rn(weights[p], __fmul_rn(-neg_step, moments[q]) / denom)
                        : fmaf(neg_step, moments[q] / denom, weights[p]);
                }
            }
            grid_barrier(barrier);
        }
    }
    // Export clipped-gradient diagnostics before the actor reuses this bank.
    // Weights and moments remain resident until both phases have completed.
    #pragma unroll
    for (int q = 0; q < CRITIC_MOMENT_SLOTS; ++q) {
        const int p = tid + q * THREADS;
        if (p < owned) {
            const int index = shard_native_index(p, cta);
            last_gradients[index] = gradients[p];
        }
    }
    if (cta == 0 && tid == 0) mean_loss[0] = accumulated_loss / (epochs * minibatches);

}
}
namespace actor {
// Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

// Native RGB64 actor: W/G shared, m/v in statically indexed registers.
// Inputs, activations and cross-block reductions are global. This is the second phase of the same launch.

#define ACTOR_WARPS (THREADS / 32)
#define ACTOR_FULL_MASK 0xffffffffu
#define ACTOR_CNN_WIDTH 5264
#define ACTOR_ACT_WIDTH 1618
#define ACTOR_MOMENT_SLOTS ((ACTOR_SHARD_SIZE + THREADS - 1) / THREADS)

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
// The shard itself remains contiguous shared W/G. Register moments use
// p=thread+q*THREADS and every q access is statically unrolled.
__device__ __forceinline__ int shard_layer(int p, int c) {
    #pragma unroll
    for (int layer = 0; layer < 8; ++layer)
        if (p < shard_base(layer + 1, c)) return layer;
    return 8;
}
__device__ __forceinline__ int shard_tensor(int p, int c) {
    const int layer = shard_layer(p, c), ni = input_dim(layer);
    const int k = p - shard_base(layer, c);
    return layer < 7 ? 2 * layer + (k % (ni + 1) == ni) : layer + 7;
}
__device__ __forceinline__ int shard_native_index(int p, int c) {
    const int layer = shard_layer(p, c), ni = input_dim(layer), no = output_dim(layer);
    const int k = p - shard_base(layer, c), j = k % (ni + 1);
    const int row = layer >= 4 && layer <= 6 ? c % no : c + (k / (ni + 1)) * N_CTAS;
    return flat_base(layer) + (j == ni ? no * ni + row : row * ni + j);
}
__device__ __forceinline__ bool shard_canonical(int p, int c) {
    const int layer = shard_layer(p, c);
    return !(layer >= 4 && layer <= 6) || c < output_dim(layer);
}

__device__ __forceinline__ int activation_base(int l) {
    return ACTOR_TILE * (l < 0 ? 0 : l == 0 ? 699 : l == 1 ? 1211 : l == 2 ? 1467 : 1595);
}
__device__ __forceinline__ int cnn_activation_base(int l) { return ACTOR_TILE * (l == 0 ? 0 : l == 1 ? 3600 : 4752); }
__device__ __forceinline__ int cnn_input_channels(int l) { return l == 0 ? 3 : l == 1 ? 16 : 32; }
__device__ __forceinline__ int cnn_input_height(int l) { return l == 0 ? 64 : l == 1 ? 15 : 6; }
__device__ __forceinline__ int cnn_output_height(int l) { return l == 0 ? 15 : l == 1 ? 6 : 4; }
__device__ __forceinline__ int cnn_kernel_size(int l) { return l == 0 ? 8 : l == 1 ? 4 : 3; }
__device__ __forceinline__ int cnn_stride(int l) { return l == 0 ? 4 : l == 1 ? 2 : 1; }
__device__ __forceinline__ float warp_sum(float v) {
    for (int d = 16; d; d >>= 1) v += __shfl_down_sync(ACTOR_FULL_MASK, v, d);
    return v;
}
__device__ __forceinline__ double warp_sum_double(double v) {
    for (int d = 16; d; d >>= 1) v += __shfl_down_sync(ACTOR_FULL_MASK, v, d);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int d = 16; d; d >>= 1) v = fmaxf(v, __shfl_down_sync(ACTOR_FULL_MASK, v, d));
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
    for (int pair = replica * ACTOR_WARPS + warp; pair < count * h * h; pair += replicas * ACTOR_WARPS) {
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
        partial[(oc * ACTOR_TILE + sample) * width + rem] = v;
    }
    grid_barrier(bar);
    for (int p = c * THREADS + threadIdx.x; p < count * width; p += N_CTAS * THREADS) {
        int sample = p / width, j = p % width;
        float v = 0.f;
        for (int owner = 0; owner < min(N_CTAS, channels); owner++) v += partial[(owner * ACTOR_TILE + sample) * width + j];
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
        for (int sample = warp; sample < count; sample += ACTOR_WARPS) {
            const float temp = fmaxf(expf(w[shard_base(7, c)]), 1e-3f);
            float z = lane < 16 ? cnn[cnn_activation_base(2) + (sample * 32 + c) * 16 + lane] / temp
                                : -__int_as_float(0x7f800000);
            float mx = warp_max(z);
            mx = __shfl_sync(ACTOR_FULL_MASK, mx, 0);
            float p = lane < 16 ? expf(z - mx) : 0.f;
            float norm = warp_sum(p);
            norm = __shfl_sync(ACTOR_FULL_MASK, norm, 0);
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
    for (int p = warp; p < row_count(L, c) * count; p += ACTOR_WARPS) {
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
            partial[(c * ACTOR_TILE + sample) * width + p % width] = v;
        }
    grid_barrier(bar);
    for (int p = c * THREADS + threadIdx.x; p < count * width; p += N_CTAS * THREADS) {
        int sample = p / width, j = p % width;
        float v = 0.f;
        for (int owner = 0; owner < min(N_CTAS, output_dim(L)); owner++)
            v += partial[(owner * ACTOR_TILE + sample) * width + j];
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
        for (int sample = warp; sample < count; sample += ACTOR_WARPS) {
            i64 index = perm[offset + sample];
            float mu = lane < 23 ? acts[activation_base(3) + sample * 23 + lane] : 0.f;
            float sd = lane < 23 ? stds[lane] : 1.f;
            float delta = lane < 23 ? actions[index * 23 + lane] - mu : 0.f;
            float lp = lane < 23 ? -(delta * delta) / (2.f * sd * sd) - logf(sd) - .9189385332046727f : 0.f;
            float ent = lane < 23 ? logf(sd) + 1.4189385332046727f : 0.f;
            lp = warp_sum(lp);
            lp = __shfl_sync(ACTOR_FULL_MASK, lp, 0);
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


__device__ __forceinline__ void phase(
float* params, float* adam_m, float* adam_v, float* last_grad, const float* state, const unsigned char* images,
    const float* actions, const float* oldlp, const float* advantages, const i64* perm, const float* xy,
    const float* adam_corrections, int mini, int epochs, int minibatches, float clip, float entropy_coef,
    float max_norm, float std_min, float std_max, float beta1, float beta1c, float beta2, float beta2c, float epsilon,
    int fused_adam, float* cnn, float* cd, float* probs, float* acts, float* d, float* partial, double* norms,
    float* stds, float* std_grads, float* sample_metrics, float* statistics, float* mean_metrics, u32* bar,
float* w, float* g, float (&m)[ACTOR_MOMENT_SLOTS], float (&v)[ACTOR_MOMENT_SLOTS]) {
    __shared__ double warp_norm[ACTOR_WARPS];
    __shared__ float clipping, corrections[32];
    int c = blockIdx.x, t = threadIdx.x, owned = shard_base(9, c);
    float total_sur = 0.f, total_ent = 0.f;
    // Keep W/G shared and m/v in statically indexed registers for the complete update.
    for (int epoch = 0; epoch < epochs; epoch++)
        for (int mb = 0; mb < minibatches; mb++) {
            int update = epoch * minibatches + mb;
            for (int p = t; p < owned; p += THREADS) g[p] = 0.f;
            if (c < 23 && t == 0) stds[c] = fminf(fmaxf(w[shard_base(8, c)], std_min), std_max);
            grid_barrier(bar);
            float mini_sur = 0.f, mini_ent = 0.f;
            for (int start = 0; start < mini; start += ACTOR_TILE) {
                int count = min(ACTOR_TILE, mini - start), offset = mb * mini + start;
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
                for (int j = 0; j < ACTOR_WARPS; j++) q += warp_norm[j];
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
            // Replicas received the same reduced gradient before clipping.
            // Each native tensor retains its own checkpoint step correction.
            #pragma unroll
            for (int q = 0; q < ACTOR_MOMENT_SLOTS; ++q) {
                const int p = t + q * THREADS;
                if (p < owned) {
                    const int tensor = shard_tensor(p, c);
                    const float grad = g[p] * clipping;
                    g[p] = grad;
                    if (fused_adam) {
                        m[q] = fmaf(beta1, m[q], fmaf(-beta1, grad, grad));
                        const float square = __fmul_rn(grad, grad);
                        v[q] = fmaf(beta2, v[q], fmaf(-beta2, square, square));
                    } else {
                        m[q] = fmaf(beta1c, grad - m[q], m[q]);
                        v[q] = fmaf(__fmul_rn(beta2c, grad), grad, __fmul_rn(v[q], beta2));
                    }
                    const float denom = sqrtf(v[q]) / corrections[tensor * 2 + 1] + epsilon;
                    const float neg = corrections[tensor * 2];
                    w[p] = fused_adam ? __fsub_rn(w[p], __fmul_rn(-neg, m[q]) / denom)
                                      : fmaf(neg, m[q] / denom, w[p]);
                }
            }
            grid_barrier(bar);
        }
    // Export clipped-gradient diagnostics from canonical CNN owners only.
    // The outer kernel exports weights and moments after both phases complete.
    #pragma unroll
    for (int q = 0; q < ACTOR_MOMENT_SLOTS; ++q) {
        const int p = t + q * THREADS;
        if (p < owned && shard_canonical(p, c)) {
            const int index = shard_native_index(p, c);
            last_grad[index] = g[p];
        }
    }
    if (c == 0 && t == 0) {
        mean_metrics[0] = total_sur / (epochs * minibatches);
        mean_metrics[1] = total_ent / (epochs * minibatches);
    }

}
}

extern "C" __global__ __maxnreg__(MAX_REGS) void unified_native_learner(
float* critic_params,
float* critic_adam_m,
float* critic_adam_v,
float* critic_last_gradients,
const float* critic_inputs,
const float* critic_old_values,
const float* critic_returns,
const i64* critic_indices,
const float* critic_adam_corrections,
int critic_minibatch_size,
int critic_epochs,
int critic_minibatches,
float critic_clip,
float critic_value_coef,
float critic_max_grad_norm,
float critic_beta1,
float critic_beta1_complement,
float critic_beta2,
float critic_beta2_complement,
float critic_epsilon,
int critic_fused_adam,
float* critic_acts,
float* critic_deltas,
float* critic_partials,
double* critic_norm_partials,
float* critic_sample_losses,
float* critic_statistics,
float* critic_mean_loss,
u32* critic_barrier,
float* actor_params,
float* actor_adam_m,
float* actor_adam_v,
float* actor_last_grad,
const float* actor_state,
const unsigned char* actor_images,
const float* actor_actions,
const float* actor_oldlp,
const float* actor_advantages,
const i64* actor_perm,
const float* actor_xy,
const float* actor_adam_corrections,
int actor_mini,
int actor_epochs,
int actor_minibatches,
float actor_clip,
float actor_entropy_coef,
float actor_max_norm,
float actor_std_min,
float actor_std_max,
float actor_beta1,
float actor_beta1c,
float actor_beta2,
float actor_beta2c,
float actor_epsilon,
int actor_fused_adam,
float* actor_cnn,
float* actor_cd,
float* actor_probs,
float* actor_acts,
float* actor_d,
float* actor_partial,
double* actor_norms,
float* actor_stds,
float* actor_std_grads,
float* actor_sample_metrics,
float* actor_statistics,
float* actor_mean_metrics,
u32* actor_bar) {
    extern __shared__ float storage[];
    float* critic_w = storage;
    float* actor_w = critic_w + CRITIC_SHARD_SIZE;
    float* gradient = actor_w + ACTOR_SHARD_SIZE;
    float critic_m[CRITIC_MOMENT_SLOTS], critic_v[CRITIC_MOMENT_SLOTS];
    float actor_m[ACTOR_MOMENT_SLOTS], actor_v[ACTOR_MOMENT_SLOTS];
    const int t = threadIdx.x, c = blockIdx.x;
    const int critic_owned = critic::shard_base(4, c);
    #pragma unroll
    for (int q = 0; q < CRITIC_MOMENT_SLOTS; ++q) {
        const int p = t + q * THREADS;
        critic_m[q] = 0.f; critic_v[q] = 0.f;
        if (p < critic_owned) {
            const int index = critic::shard_native_index(p, c);
            critic_w[p] = critic_params[index];
            critic_m[q] = critic_adam_m[index];
            critic_v[q] = critic_adam_v[index];
        }
        // Force both values to be materialized before either optimizer phase.
        asm volatile("" : "+f"(critic_m[q]), "+f"(critic_v[q]) :: "memory");
    }
    const int actor_owned = actor::shard_base(9, c);
    #pragma unroll
    for (int q = 0; q < ACTOR_MOMENT_SLOTS; ++q) {
        const int p = t + q * THREADS;
        actor_m[q] = 0.f; actor_v[q] = 0.f;
        if (p < actor_owned) {
            const int index = actor::shard_native_index(p, c);
            actor_w[p] = actor_params[index];
            actor_m[q] = actor_adam_m[index];
            actor_v[q] = actor_adam_v[index];
        }
        // Force both values to be materialized before either optimizer phase.
        asm volatile("" : "+f"(actor_m[q]), "+f"(actor_v[q]) :: "memory");
    }
    grid_barrier(critic_barrier);
    critic::phase(critic_params, critic_adam_m, critic_adam_v, critic_last_gradients, critic_inputs, critic_old_values, critic_returns, critic_indices, critic_adam_corrections, critic_minibatch_size, critic_epochs, critic_minibatches, critic_clip, critic_value_coef, critic_max_grad_norm, critic_beta1, critic_beta1_complement, critic_beta2, critic_beta2_complement, critic_epsilon, critic_fused_adam, critic_acts, critic_deltas, critic_partials, critic_norm_partials, critic_sample_losses, critic_statistics, critic_mean_loss, critic_barrier, critic_w, gradient, critic_m, critic_v);
    grid_barrier(critic_barrier);
    actor::phase(actor_params, actor_adam_m, actor_adam_v, actor_last_grad, actor_state, actor_images, actor_actions, actor_oldlp, actor_advantages, actor_perm, actor_xy, actor_adam_corrections, actor_mini, actor_epochs, actor_minibatches, actor_clip, actor_entropy_coef, actor_max_norm, actor_std_min, actor_std_max, actor_beta1, actor_beta1c, actor_beta2, actor_beta2c, actor_epsilon, actor_fused_adam, actor_cnn, actor_cd, actor_probs, actor_acts, actor_d, actor_partial, actor_norms, actor_stds, actor_std_grads, actor_sample_metrics, actor_statistics, actor_mean_metrics, actor_bar, actor_w, gradient, actor_m, actor_v);
    grid_barrier(critic_barrier);
    // Both network checkpoint states export only after both phases finish.
    #pragma unroll
    for (int q = 0; q < CRITIC_MOMENT_SLOTS; ++q) {
        const int p = t + q * THREADS;
        // Keep critic moments live through the actor phase, with no late reload.
        asm volatile("" : "+f"(critic_m[q]), "+f"(critic_v[q]) :: "memory");
        if (p < critic_owned) {
            const int index = critic::shard_native_index(p, c);
            critic_params[index] = critic_w[p];
            critic_adam_m[index] = critic_m[q];
            critic_adam_v[index] = critic_v[q];
        }
    }
    #pragma unroll
    for (int q = 0; q < ACTOR_MOMENT_SLOTS; ++q) {
        const int p = t + q * THREADS;
        // Keep actor moments live from kernel entry through both phases, with no late reload.
        asm volatile("" : "+f"(actor_m[q]), "+f"(actor_v[q]) :: "memory");
        if (p < actor_owned && actor::shard_canonical(p, c)) {
            const int index = actor::shard_native_index(p, c);
            actor_params[index] = actor_w[p];
            actor_adam_m[index] = actor_m[q];
            actor_adam_v[index] = actor_v[q];
        }
    }
}

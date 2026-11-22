// Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

// Native 1595 -> 512 -> 256 -> 128 -> 1 ELU critic.
// Each block owns complete output-neuron rows, cyclically distributed over the grid.
// Its weights, accumulated gradients and both Adam moments remain in explicit shared
// memory for every minibatch. Activations, input-adjoint partials, synchronization,
// scalar Adam corrections and statistics use global memory. This is not zero-DRAM
// execution, nor a resident full visual learner. Boundary copies use global memory.
#include "device_lib.cuh"

#define FULL_MASK 0xffffffffu
#define WARPS (THREADS / 32)
#define ACT_WIDTH 897
#define PART_WIDTH 512

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
__device__ __forceinline__ int act_base(int layer) {
    return TILE * (layer == 0 ? 0 : (layer == 1 ? 512 : (layer == 2 ? 768 : 896)));
}
__device__ __forceinline__ float warp_sum(float value) {
    for (int d = 16; d; d >>= 1) value += __shfl_down_sync(FULL_MASK, value, d);
    return value;
}
__device__ __forceinline__ double warp_sum_double(double value) {
    for (int d = 16; d; d >>= 1) value += __shfl_down_sync(FULL_MASK, value, d);
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
    for (int pair = warp; pair < nr * count; pair += WARPS) {
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
            partials[(cta * TILE + sample) * PART_WIDTH + j] = sum;
        }
    }
    grid_barrier(barrier);
    for (int p = cta * THREADS + threadIdx.x; p < count * ni; p += N_CTAS * THREADS) {
        const int sample = p / ni, j = p % ni;
        float sum = 0.f;
        for (int owner = 0; owner < min(N_CTAS, no); ++owner)
            sum += partials[(owner * TILE + sample) * PART_WIDTH + j];
        const float activation = acts[act_base(L - 1) + p];
        deltas[act_base(L - 1) + p] = sum * (activation > 0.f ? 1.f : activation + 1.f);
    }
    grid_barrier(barrier);
}

extern "C" __global__ __launch_bounds__(THREADS, 1) void persistent_native_critic(
    float* params, float* adam_m, float* adam_v, float* last_gradients,
    const float* inputs, const float* old_values, const float* returns, const i64* indices,
    const float* adam_corrections, int minibatch_size, int epochs, int minibatches,
    float clip, float value_coef, float max_grad_norm, float beta1, float beta1_complement,
    float beta2, float beta2_complement, float epsilon, int fused_adam,
    float* acts, float* deltas, float* partials, double* norm_partials,
    float* sample_losses, float* statistics, float* mean_loss, u32* barrier
) {
    extern __shared__ float storage[];
    float* weights = storage;
    float* gradients = weights + SHARD_SIZE;
    float* moments = gradients + SHARD_SIZE;
    float* variances = moments + SHARD_SIZE;
    __shared__ double warp_norms[WARPS];
    __shared__ float clipping;
    __shared__ float corrections[16];
    const int tid = threadIdx.x, cta = blockIdx.x;
    const int owned = shard_base(4, cta);
    // Import checkpoint state once at the launch boundary.
    for (int layer = 0; layer < 4; ++layer) {
        const int ni = in_dim(layer), no = out_dim(layer), nr = row_count(layer, cta);
        const int base = shard_base(layer, cta);
        for (int k = tid; k < nr * (ni + 1); k += THREADS) {
            const int r = cta + (k / (ni + 1)) * N_CTAS, j = k % (ni + 1);
            const int index = flat_base(layer) + (j == ni ? no * ni + r : r * ni + j);
            weights[base + k] = params[index];
            moments[base + k] = adam_m[index];
            variances[base + k] = adam_v[index];
        }
    }
    __syncthreads();
    float accumulated_loss = 0.f;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int mb = 0; mb < minibatches; ++mb) {
            const int update = epoch * minibatches + mb;
            for (int k = tid; k < owned; k += THREADS) gradients[k] = 0.f;
            __syncthreads();
            float minibatch_loss = 0.f;
            for (int start = 0; start < minibatch_size; start += TILE) {
                const int count = min(TILE, minibatch_size - start);
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
                for (int w = 0; w < WARPS; ++w) sum += warp_norms[w];
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
            for (int layer = 0; layer < 4; ++layer) {
                const int ni = in_dim(layer), nr = row_count(layer, cta);
                const int base = shard_base(layer, cta);
                for (int k = tid; k < nr * (ni + 1); k += THREADS) {
                    const int tensor = 2 * layer + ((k % (ni + 1)) == ni);
                    const float neg_step = corrections[tensor * 2];
                    const float bc2_sqrt = corrections[tensor * 2 + 1];
                    const int p = base + k;
                    const float g = gradients[p] * clipping;
                    gradients[p] = g;
                    if (fused_adam) {
                        moments[p] = fmaf(beta1, moments[p], fmaf(-beta1, g, g));
                        const float square = __fmul_rn(g, g);
                        variances[p] = fmaf(beta2, variances[p], fmaf(-beta2, square, square));
                    } else {
                        moments[p] = fmaf(beta1_complement, g - moments[p], moments[p]);
                        const float decayed = __fmul_rn(variances[p], beta2);
                        variances[p] = fmaf(__fmul_rn(beta2_complement, g), g, decayed);
                    }
                    const float denom = sqrtf(variances[p]) / bc2_sqrt + epsilon;
                    weights[p] = fused_adam
                        ? __fsub_rn(weights[p], __fmul_rn(-neg_step, moments[p]) / denom)
                        : fmaf(neg_step, moments[p] / denom, weights[p]);
                }
            }
            grid_barrier(barrier);
        }
    }
    // Export native checkpoint state once after all optimizer steps.
    for (int layer = 0; layer < 4; ++layer) {
        const int ni = in_dim(layer), no = out_dim(layer), nr = row_count(layer, cta);
        const int base = shard_base(layer, cta);
        for (int k = tid; k < nr * (ni + 1); k += THREADS) {
            const int r = cta + (k / (ni + 1)) * N_CTAS, j = k % (ni + 1);
            const int index = flat_base(layer) + (j == ni ? no * ni + r : r * ni + j);
            params[index] = weights[base + k];
            adam_m[index] = moments[base + k];
            adam_v[index] = variances[base + k];
            last_gradients[index] = gradients[base + k];
        }
    }
    if (cta == 0 && tid == 0) mean_loss[0] = accumulated_loss / (epochs * minibatches);
}

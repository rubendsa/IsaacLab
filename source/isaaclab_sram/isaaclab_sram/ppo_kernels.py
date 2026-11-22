# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fused CUDA PPO operations for trainable perception policies.

These Triton kernels retain intermediate values in registers and shared memory within each
operation. Model tensors and rollout storage remain in global memory; this module does not
claim that a camera policy or the simulator fits entirely in SRAM. All operations use float32.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gae_kernel(
    rewards,
    dones,
    values,
    last_values,
    returns,
    advantages,
    N: tl.constexpr,
    T: tl.constexpr,
    GAMMA: tl.constexpr,
    LAM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    env = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = env < N
    next_value = tl.load(last_values + env, valid, other=0.0)
    advantage = tl.full((BLOCK,), 0, tl.float32)
    for reverse_t in range(T):
        t = T - 1 - reverse_t
        offset = t * N + env
        value = tl.load(values + offset, valid, other=0.0)
        reward = tl.load(rewards + offset, valid, other=0.0)
        not_done = 1.0 - tl.load(dones + offset, valid, other=0).to(tl.float32)
        delta = reward + not_done * GAMMA * next_value - value
        advantage = delta + not_done * GAMMA * LAM * advantage
        ret = advantage + value
        tl.store(returns + offset, ret, valid)
        # RSL-RL computes advantages from the stored returns, including float32 rounding.
        tl.store(advantages + offset, ret - value, valid)
        next_value = value


@triton.jit
def _advantage_moments_kernel(advantages, means, m2s, counts, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    offset = block * BLOCK + tl.arange(0, BLOCK)
    valid = offset < SIZE
    values = tl.load(advantages + offset, valid, other=0.0)
    count = tl.minimum(SIZE - block * BLOCK, BLOCK)
    mean = tl.div_rn(tl.sum(values, 0), count.to(tl.float32))
    m2 = tl.sum(tl.where(valid, (values - mean) * (values - mean), 0.0), 0)
    tl.store(means + block, mean)
    tl.store(m2s + block, m2)
    tl.store(counts + block, count)


@triton.jit
def _normalize_advantages_kernel(
    advantages,
    means,
    m2s,
    counts,
    SIZE: tl.constexpr,
    PARTS: tl.constexpr,
    PART_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    part = tl.arange(0, PART_BLOCK)
    local_mean = tl.load(means + part, part < PARTS, other=0.0)
    local_m2 = tl.load(m2s + part, part < PARTS, other=0.0)
    count = tl.load(counts + part, part < PARTS, other=0.0)
    mean = tl.div_rn(tl.sum(local_mean * count, 0), float(SIZE))
    # Combine centered moments instead of E[x^2]-E[x]^2 to avoid catastrophic cancellation.
    m2 = tl.sum(local_m2 + count * (local_mean - mean) * (local_mean - mean), 0)
    std = tl.sqrt(m2 / (SIZE - 1))
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(advantages + offset, offset < SIZE, other=0.0)
    tl.store(advantages + offset, (value - mean) / (std + 1.0e-8), offset < SIZE)


def fused_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float,
    lam: float,
    *,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute PPO returns and globally normalized advantages without an autograd graph.

    Inputs have shape ``[time, environments]`` or ``[time, environments, 1]``; the
    bootstrap value has one element per environment. Time-limit bootstrapping must already
    be included in ``rewards``, as in RSL-RL's ``process_env_step``. Normalization uses
    unbiased standard deviation (correction 1); ``normalize=False`` returns raw advantages.
    """
    if rewards.ndim not in (2, 3) or (rewards.ndim == 3 and rewards.shape[-1] != 1):
        raise ValueError("GAE rewards must have shape [T,N] or [T,N,1]")
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("GAE rewards, dones, and values must have identical shapes")
    steps, envs = rewards.shape[:2]
    if steps < 1 or envs < 1 or last_values.numel() != envs:
        raise ValueError("GAE needs a nonempty rollout and one bootstrap value per environment")
    if normalize and rewards.numel() < 2:
        raise ValueError("unbiased advantage normalization requires at least two transitions")
    _check_float_inputs(rewards, values, last_values)
    if dones.device != rewards.device:
        raise ValueError("GAE dones must be on the same CUDA device as rewards")
    returns = torch.empty(rewards.shape, device=rewards.device, dtype=torch.float32)
    advantages = torch.empty_like(returns)
    with torch.cuda.device(rewards.device):
        _gae_kernel[(triton.cdiv(envs, 128),)](
            rewards.contiguous(),
            dones.contiguous(),
            values.contiguous(),
            last_values.contiguous(),
            returns,
            advantages,
            envs,
            steps,
            gamma,
            lam,
            128,
            enable_fp_fusion=False,
        )
        if normalize:
            size = rewards.numel()
            parts = triton.cdiv(size, 1024)
            moments = torch.empty((3, parts), device=rewards.device, dtype=torch.float32)
            _advantage_moments_kernel[(parts,)](advantages, moments[0], moments[1], moments[2], size, 1024)
            _normalize_advantages_kernel[(parts,)](
                advantages,
                moments[0],
                moments[1],
                moments[2],
                size,
                parts,
                triton.next_power_of_2(parts),
                1024,
            )
    return returns, advantages


@triton.jit
def _ppo_forward_kernel(
    mean,
    std,
    values,
    actions,
    old_logp,
    old_values,
    returns,
    advantages,
    partials,
    N: tl.constexpr,
    A: tl.constexpr,
    PARTS: tl.constexpr,
    CLIP: tl.constexpr,
    VALUE_COEF: tl.constexpr,
    ENTROPY_COEF: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    col = tl.arange(0, COLS)
    offset = row[:, None] * A + col[None, :]
    valid = (row[:, None] < N) & (col[None, :] < A)
    mu = tl.load(mean + offset, valid, other=0.0)
    sigma = tl.load(std + offset, valid, other=1.0)
    action = tl.load(actions + offset, valid, other=0.0)
    delta = action - mu
    log_sigma = tl.log(sigma)
    logp_terms = -(delta * delta) / (2.0 * sigma * sigma) - log_sigma - 0.9189385332046727
    logp = tl.sum(tl.where(col[None, :] < A, logp_terms, 0.0), 1)
    entropy = tl.sum(tl.where(col[None, :] < A, log_sigma + 1.4189385332046727, 0.0), 1)
    ratio = tl.exp(logp - tl.load(old_logp + row, row < N, other=0.0))
    adv = tl.load(advantages + row, row < N, other=0.0)
    clipped_ratio = tl.minimum(tl.maximum(ratio, 1.0 - CLIP), 1.0 + CLIP)
    surrogate = tl.maximum(-adv * ratio, -adv * clipped_ratio)
    value = tl.load(values + row, row < N, other=0.0)
    old_value = tl.load(old_values + row, row < N, other=0.0)
    ret = tl.load(returns + row, row < N, other=0.0)
    clipped_value = old_value + tl.minimum(tl.maximum(value - old_value, -CLIP), CLIP)
    error, clipped_error = value - ret, clipped_value - ret
    value_loss = tl.maximum(error * error, clipped_error * clipped_error)
    total = surrogate + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
    part = tl.program_id(0)
    tl.store(partials + part, tl.sum(tl.where(row < N, total, 0.0), 0) / N)
    tl.store(partials + PARTS + part, tl.sum(tl.where(row < N, value_loss, 0.0), 0) / N)
    tl.store(partials + 2 * PARTS + part, tl.sum(tl.where(row < N, surrogate, 0.0), 0) / N)
    tl.store(partials + 3 * PARTS + part, tl.sum(tl.where(row < N, entropy, 0.0), 0) / N)


@triton.jit
def _ppo_reduce_kernel(partials, output, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    field = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    part = tl.load(partials + field * PARTS + col, col < PARTS, other=0.0)
    tl.store(output + field, tl.sum(part, 0))


@triton.jit
def _ppo_backward_kernel(
    mean,
    std,
    values,
    actions,
    old_logp,
    old_values,
    returns,
    advantages,
    grad_output,
    grad_mean,
    grad_std,
    grad_values,
    N: tl.constexpr,
    A: tl.constexpr,
    CLIP: tl.constexpr,
    VALUE_COEF: tl.constexpr,
    ENTROPY_COEF: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    col = tl.arange(0, COLS)
    offset = row[:, None] * A + col[None, :]
    valid = (row[:, None] < N) & (col[None, :] < A)
    mu = tl.load(mean + offset, valid, other=0.0)
    sigma = tl.load(std + offset, valid, other=1.0)
    action = tl.load(actions + offset, valid, other=0.0)
    delta = action - mu
    variance = sigma * sigma
    logp_terms = -(delta * delta) / (2.0 * variance) - tl.log(sigma) - 0.9189385332046727
    logp = tl.sum(tl.where(col[None, :] < A, logp_terms, 0.0), 1)
    ratio = tl.exp(logp - tl.load(old_logp + row, row < N, other=0.0))
    adv = tl.load(advantages + row, row < N, other=0.0)
    clipped_ratio = tl.minimum(tl.maximum(ratio, 1.0 - CLIP), 1.0 + CLIP)
    surrogate, clipped_surrogate = -adv * ratio, -adv * clipped_ratio
    inside = ((ratio >= 1.0 - CLIP) & (ratio <= 1.0 + CLIP)).to(tl.float32)
    # torch.maximum splits the gradient equally on ties; torch.clamp includes its boundaries.
    derivative = tl.where(
        surrogate > clipped_surrogate,
        -adv,
        tl.where(surrogate < clipped_surrogate, -adv * inside, -0.5 * adv * (1.0 + inside)),
    )
    scale = tl.load(grad_output) / N
    dlogp = derivative * ratio * scale
    dmean = dlogp[:, None] * delta / variance
    dstd = dlogp[:, None] * ((delta * delta) / variance - 1.0) / sigma - ENTROPY_COEF * scale / sigma
    tl.store(grad_mean + offset, dmean, valid)
    tl.store(grad_std + offset, dstd, valid)
    value = tl.load(values + row, row < N, other=0.0)
    old_value = tl.load(old_values + row, row < N, other=0.0)
    ret = tl.load(returns + row, row < N, other=0.0)
    dv = value - old_value
    clipped_value = old_value + tl.minimum(tl.maximum(dv, -CLIP), CLIP)
    error, clipped_error = value - ret, clipped_value - ret
    loss, clipped_loss = error * error, clipped_error * clipped_error
    inside_value = ((dv >= -CLIP) & (dv <= CLIP)).to(tl.float32)
    dvalue = tl.where(
        loss > clipped_loss,
        2.0 * error,
        tl.where(loss < clipped_loss, 2.0 * clipped_error * inside_value, error + clipped_error * inside_value),
    )
    tl.store(grad_values + row, VALUE_COEF * scale * dvalue, row < N)


def _check_float_inputs(*tensors: torch.Tensor) -> None:
    device = tensors[0].device
    if any(t.device != device or not t.is_cuda or t.dtype != torch.float32 for t in tensors):
        raise ValueError("fused PPO operations require float32 tensors on one CUDA device")


class _FusedPPOLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, mean, std, values, actions, old_logp, old_values, returns, advantages, clip, value_coef, entropy_coef
    ):
        samples, action_dim = mean.shape
        parts = triton.cdiv(samples, 32)
        partials = torch.empty((4, parts), device=mean.device, dtype=torch.float32)
        output = torch.empty(4, device=mean.device, dtype=torch.float32)
        with torch.cuda.device(mean.device):
            _ppo_forward_kernel[(parts,)](
                mean,
                std,
                values,
                actions,
                old_logp,
                old_values,
                returns,
                advantages,
                partials,
                samples,
                action_dim,
                parts,
                clip,
                value_coef,
                entropy_coef,
                32,
                triton.next_power_of_2(action_dim),
                enable_fp_fusion=False,
            )
            _ppo_reduce_kernel[(4,)](partials, output, parts, triton.next_power_of_2(parts))
        ctx.save_for_backward(mean, std, values, actions, old_logp, old_values, returns, advantages)
        ctx.coefficients = (clip, value_coef, entropy_coef)
        loss, value_loss, surrogate_loss, entropy = output.unbind()
        ctx.mark_non_differentiable(value_loss, surrogate_loss, entropy)
        return loss, value_loss, surrogate_loss, entropy

    @staticmethod
    def backward(ctx, grad_output, _grad_value, _grad_surrogate, _grad_entropy):
        mean, std, values, actions, old_logp, old_values, returns, advantages = ctx.saved_tensors
        grad_mean, grad_std, grad_values = torch.empty_like(mean), torch.empty_like(std), torch.empty_like(values)
        samples, action_dim = mean.shape
        clip, value_coef, entropy_coef = ctx.coefficients
        with torch.cuda.device(mean.device):
            _ppo_backward_kernel[(triton.cdiv(samples, 32),)](
                mean,
                std,
                values,
                actions,
                old_logp,
                old_values,
                returns,
                advantages,
                grad_output.contiguous(),
                grad_mean,
                grad_std,
                grad_values,
                samples,
                action_dim,
                clip,
                value_coef,
                entropy_coef,
                32,
                triton.next_power_of_2(action_dim),
                enable_fp_fusion=False,
            )
        return grad_mean, grad_std, grad_values, None, None, None, None, None, None, None, None


def fused_ppo_loss(
    mean: torch.Tensor,
    std: torch.Tensor,
    values: torch.Tensor,
    actions: torch.Tensor,
    old_logp: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    clip: float,
    value_coef: float,
    entropy_coef: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return differentiable PPO loss and detached value, surrogate, and entropy means.

    ``mean`` and ``actions`` are ``[batch, action_dim]``. ``std`` is a positive scale
    broadcastable to that shape (apply distribution clamping before calling). Value and
    rollout vectors have one element per batch row. Gradients flow through ``mean``,
    ``std`` and ``values``, including reductions for a broadcast standard deviation.
    Higher-order gradients are not supported.
    """
    if mean.ndim != 2 or actions.shape != mean.shape or min(mean.shape) < 1:
        raise ValueError("PPO means and actions must have nonempty shape [batch, action_dim]")
    samples = mean.shape[0]
    if any(t.numel() != samples for t in (values, old_logp, old_values, returns, advantages)):
        raise ValueError("PPO value and rollout vectors need one element per sample")
    _check_float_inputs(mean, std, values, actions, old_logp, old_values, returns, advantages)
    if clip < 0:
        raise ValueError("PPO clipping range must be nonnegative")
    # Broadcasting outside the custom Function lets autograd reduce gradients back to the
    # original std parameter, including stride-zero expanded Gaussian distribution scales.
    std = torch.broadcast_to(std, mean.shape).contiguous()
    inputs = [t.contiguous() for t in (mean, std, values, actions, old_logp, old_values, returns, advantages)]
    return _FusedPPOLoss.apply(*inputs, float(clip), float(value_coef), float(entropy_coef))

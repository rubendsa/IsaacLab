# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent PyTorch parity of fused perception PPO values and gradients."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("triton")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab_sram.ppo_kernels import fused_gae, fused_ppo_loss  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def _torch_loss(mean, std, values, actions, old_logp, old_values, returns, advantages, clip, value_coef, entropy_coef):
    distribution = torch.distributions.Normal(mean, std)
    ratio = torch.exp(distribution.log_prob(actions).sum(-1) - old_logp.reshape(-1))
    adv = advantages.reshape(-1)
    surrogate = torch.maximum(-adv * ratio, -adv * ratio.clamp(1.0 - clip, 1.0 + clip)).mean()
    clipped_values = old_values + (values - old_values).clamp(-clip, clip)
    value_loss = torch.maximum((values - returns).square(), (clipped_values - returns).square()).mean()
    entropy = distribution.entropy().sum(-1).mean()
    return surrogate + value_coef * value_loss - entropy_coef * entropy, value_loss, surrogate, entropy


@pytest.mark.parametrize("std_layout", ["parameter", "expanded", "per_sample"])
def test_visual_ppo_matches_torch_values_and_all_trainable_gradients(std_layout):
    """23-action Kuka batches exercise both clipping branches, padding, and std reduction."""
    torch.manual_seed(25)
    batch, actions_n = 71, 23
    mean = torch.randn(batch, actions_n, device="cuda") * 0.1
    std = torch.rand(actions_n if std_layout != "per_sample" else (batch, actions_n), device="cuda") + 0.5
    values = torch.randn(batch, 1, device="cuda")
    actions = mean + std * torch.randn_like(mean)
    # Deliberately populate active and clipped branches for both signs of advantage.
    old_logp = torch.distributions.Normal(mean, std).log_prob(actions).sum(-1)
    old_logp -= torch.linspace(0.4, 1.6, batch, device="cuda").log()
    old_values = values + torch.linspace(-0.7, 0.7, batch, device="cuda")[:, None]
    returns = torch.randn_like(values)
    advantages = torch.randn(batch, 1, device="cuda")
    inputs = []
    for implementation in (_torch_loss, fused_ppo_loss):
        mu, sigma, value = [tensor.detach().clone().requires_grad_() for tensor in (mean, std, values)]
        scale = sigma.expand_as(mu) if std_layout == "expanded" else sigma
        result = implementation(mu, scale, value, actions, old_logp, old_values, returns, advantages, 0.2, 1.3, 0.017)
        (result[0] * 2.5).backward()
        inputs.append((result, (mu.grad, sigma.grad, value.grad)))
    reference, actual = inputs
    for expected, observed in zip(reference[0], actual[0]):
        torch.testing.assert_close(observed, expected, atol=2e-6, rtol=2e-5)
    for expected, observed in zip(reference[1], actual[1]):
        torch.testing.assert_close(observed, expected, atol=2e-6, rtol=3e-5)
    assert not actual[0][1].requires_grad


def test_value_clip_ties_and_boundaries_match_torch_subgradients():
    """Exactly representable value boundaries protect clamp inclusivity and maximum ties."""
    mean = torch.zeros(8, 1, device="cuda")
    std = torch.ones((), device="cuda")
    values = torch.tensor([-0.5, -0.25, 0.0, 0.25, 0.5, 0.5, -0.5, 0.0], device="cuda")[:, None]
    old_values = torch.zeros_like(values)
    returns = torch.tensor([0.0, 0.5, 0.0, -0.5, 0.0, 0.375, -0.375, 0.5], device="cuda")[:, None]
    actions = torch.zeros_like(mean)
    old_logp = torch.distributions.Normal(mean, std).log_prob(actions).sum(-1)
    advantages = torch.zeros(8, device="cuda")
    expected_grad = None
    for implementation in (_torch_loss, fused_ppo_loss):
        value = values.clone().requires_grad_()
        loss = implementation(mean, std, value, actions, old_logp, old_values, returns, advantages, 0.25, 1.0, 0.0)[0]
        loss.backward()
        if expected_grad is None:
            expected_grad = value.grad
        else:
            torch.testing.assert_close(value.grad, expected_grad, rtol=0, atol=0)


def test_fused_gae_matches_rollout_reference_with_resets_and_unbiased_normalization():
    """Noncontiguous inputs, timeouts supplied as rewards, and terminal bootstrap masking."""
    torch.manual_seed(4)
    steps, envs = 37, 193
    rewards = torch.randn(envs, steps, device="cuda").T[..., None]
    values = torch.randn_like(rewards)
    dones = (torch.rand_like(rewards) < 0.08).to(torch.uint8)
    dones[0, :17] = 1
    dones[-1, 17:34] = 1
    last = torch.randn(envs, 1, device="cuda")
    gamma, lam = 0.99, 0.95
    # This follows RSL-RL compute_returns at its external tensor boundary.
    returns = torch.empty_like(values)
    advantage = torch.zeros_like(last)
    for t in reversed(range(steps)):
        next_value = last if t == steps - 1 else values[t + 1]
        not_terminal = 1.0 - dones[t].float()
        delta = rewards[t] + not_terminal * gamma * next_value - values[t]
        advantage = delta + not_terminal * gamma * lam * advantage
        returns[t] = advantage + values[t]
    raw_advantages = returns - values
    expected = (raw_advantages - raw_advantages.mean()) / (raw_advantages.std() + 1e-8)
    actual_returns, actual_advantages = fused_gae(rewards, dones, values, last, gamma, lam)
    torch.testing.assert_close(actual_returns, returns, rtol=1e-6, atol=2e-6)
    torch.testing.assert_close(actual_advantages, expected, rtol=2e-6, atol=2e-6)
    _, raw = fused_gae(rewards, dones, values, last, gamma, lam, normalize=False)
    torch.testing.assert_close(raw, raw_advantages, rtol=1e-6, atol=2e-6)
    # At a terminal final step the bootstrap value must have no effect.
    changed_last = last + 100.0
    changed_returns, _ = fused_gae(rewards, dones, values, changed_last, gamma, lam, normalize=False)
    torch.testing.assert_close(changed_returns[-1, 17:34], actual_returns[-1, 17:34], rtol=0, atol=0)


def test_constant_advantages_normalize_without_nan():
    rewards = torch.ones(3, 5, device="cuda")
    dones = torch.ones_like(rewards, dtype=torch.uint8)
    values = torch.zeros_like(rewards)
    returns, advantages = fused_gae(rewards, dones, values, values[-1], 0.99, 0.95)
    torch.testing.assert_close(returns, rewards, rtol=0, atol=0)
    torch.testing.assert_close(advantages, values, rtol=0, atol=0)

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native RGB64 PPO with separate resident actor/critic optimizer phases.

Both networks keep weights, gradients and Adam moments in block-owned shared
memory throughout their own 40-step update. The phases are separate launches;
rollouts, activations, reductions, packing and checkpoint copyback still use
global memory. This is not a unified all-SRAM learner or a zero-DRAM claim.

The native observation history adapter is supported through its group reader.
Whole-group materialization and native normalization remain outside the kernels.
"""

from __future__ import annotations

import torch
from rsl_rl.models.mlp_model import MLPModel

from .perception import SramPerceptionPPO
from .persistent_actor import PersistentActor
from .persistent_critic import PersistentCritic


def _consume_native_action_rng(mean: torch.Tensor, std: torch.Tensor, updates: int) -> None:
    """Consume the native Normal.sample RNG calls with equivalent dummy operands.

    PyTorch's tensor/tensor normal implementation first fills a newly allocated
    output with normal_(0,1), then multiplies by std and adds mean. Thus the RNG
    consumption depends on the output shape/device/dtype, not parameter values.
    The caller also preserves mean's contiguous layout and std's broadcast view.
    Complete resumed-update tests verify exact CPU and CUDA RNG state equality.
    """
    for _ in range(updates):
        torch.normal(mean, std)


class SramPersistentFullPerceptionPPO(SramPerceptionPPO):
    """Native camera model and checkpoint format with two persistent updates."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not 1 <= self.storage.num_envs <= 64 or self.storage.num_transitions_per_env != 32:
            raise ValueError("Persistent native visual PPO requires 1–64 environments and 32 rollout steps")
        if self.num_learning_epochs != 5 or self.num_mini_batches != 8:
            raise ValueError("Persistent native visual PPO requires five epochs and eight minibatches")
        if {id(parameter) for parameter in self.actor.parameters()} & {
            id(parameter) for parameter in self.critic.parameters()
        }:
            raise ValueError("Separate optimizer phases require disjoint actor and critic parameters")
        self._persistent_actor = None
        self._persistent_critic = None
        self._discarded_mean = torch.zeros(
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
            23,
            dtype=torch.float32,
            device=self.device,
        )
        # Native GaussianDistribution broadcasts its trainable [23] standard
        # deviations to [minibatch,23] through a stride-zero batch dimension.
        self._discarded_std = torch.ones(23, dtype=torch.float32, device=self.device).expand_as(self._discarded_mean)

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        loaded = super().load(loaded_dict, load_cfg, strict)
        self._persistent_actor = None
        self._persistent_critic = None
        return loaded

    @torch.no_grad()
    def update(self) -> dict[str, float]:
        """Use identical frozen targets and permutation in both optimizer phases."""
        storage = self.storage
        observations = storage.observations.flatten(0, 1)
        batch_size = storage.num_envs * storage.num_transitions_per_env
        indices = torch.randperm(batch_size, requires_grad=False, device=self.device)
        self.optimizer.zero_grad(set_to_none=True)
        if self._persistent_critic is None:
            self._persistent_critic = PersistentCritic(
                self.critic,
                self.optimizer,
                num_mini_batches=self.num_mini_batches,
                num_learning_epochs=self.num_learning_epochs,
                clip_param=self.clip_param,
                value_loss_coef=self.value_loss_coef,
                max_grad_norm=self.max_grad_norm,
            )
        if self._persistent_actor is None:
            self._persistent_actor = PersistentActor(
                self.actor,
                self.optimizer,
                num_mini_batches=self.num_mini_batches,
                num_learning_epochs=self.num_learning_epochs,
                clip_param=self.clip_param,
                entropy_coef=self.entropy_coef,
                max_grad_norm=self.max_grad_norm,
            )
        # Neither normalization is updated during either phase. Calling the
        # stock parent latent method processes only actor's 1D state groups;
        # raw RGB normalization/CNN/temperature/MLP run inside PersistentActor.
        critic_input = self.critic.get_latent(observations).contiguous()
        actor_input = MLPModel.get_latent(self.actor, observations).contiguous()
        value_loss = self._persistent_critic.update(
            critic_input, storage.values.flatten(0, 1), storage.returns.flatten(0, 1), indices
        )
        actor_losses = self._persistent_actor.update(
            actor_input,
            observations[self.actor.obs_groups_2d[0]],
            storage.actions.flatten(0, 1),
            storage.actions_log_prob.flatten(0, 1),
            storage.advantages.flatten(0, 1),
            indices,
        )
        _consume_native_action_rng(
            self._discarded_mean, self._discarded_std, self.num_learning_epochs * self.num_mini_batches
        )
        self.actor.update_normalization(observations)
        self.critic.update_normalization(observations)
        losses = torch.cat((value_loss.reshape(1), actor_losses)).tolist()
        storage.clear()
        return dict(zip(("value", "surrogate", "entropy"), losses, strict=True))

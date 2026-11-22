# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native camera PPO with a persistent critic optimizer phase.

The actor and critic have disjoint parameters and separately clipped gradients.
Their updates can therefore execute in separate phases using the same fixed
rollout targets and minibatch order. The critic kernel retains its parameters,
gradients and Adam moments during all 40 optimizer steps. The native trainable
camera actor still uses library convolutions and fused PPO operations.

This is an intermediate implementation toward a complete resident visual
learner. Rollout inputs and cross-block activation/reduction buffers remain
global allocations; the actor is not persistent.
"""

from __future__ import annotations

import torch

from .perception import SramPerceptionPPO


class SramPersistentCriticPPO(SramPerceptionPPO):
    """Run the native small-rollout critic update in one cooperative launch.

    Supports up to 64 environments, 32 rollout steps, five epochs and eight
    minibatches. Model shape and device residency are checked by the kernel
    wrapper. Native RSL-RL checkpoint objects remain authoritative at each
    launch boundary, including resumed nonzero Adam state.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not 1 <= self.storage.num_envs <= 64 or self.storage.num_transitions_per_env != 32:
            raise ValueError("Persistent native critic requires 1–64 environments and 32 rollout steps")
        if self.num_learning_epochs != 5 or self.num_mini_batches != 8:
            raise ValueError("Persistent native critic requires five epochs and eight minibatches")
        if {id(p) for p in self.actor.parameters()} & {id(p) for p in self.critic.parameters()}:
            raise ValueError("Separate persistent critic updates require disjoint actor and critic parameters")
        self._persistent_critic = None

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Invalidate cached kernel bindings when native optimizer state is replaced."""
        loaded = super().load(loaded_dict, load_cfg, strict)
        self._persistent_critic = None
        return loaded

    def update(self) -> dict[str, float]:
        """Use one native permutation for both complete optimizer phases."""
        from .persistent_critic import PersistentCritic
        from .ppo_kernels import fused_ppo_loss

        storage = self.storage
        observations = storage.observations.flatten(0, 1)
        batch_size = storage.num_envs * storage.num_transitions_per_env
        minibatch_size = batch_size // self.num_mini_batches
        # Native RolloutStorage draws once and reuses this permutation in each
        # epoch. The critic consumes no random numbers, so separating it leaves
        # the actor's discarded samples and subsequent rollout RNG intact.
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
        with torch.no_grad():
            value_loss = self._persistent_critic.update(
                self.critic.get_latent(observations).contiguous(),
                storage.values.flatten(0, 1),
                storage.returns.flatten(0, 1),
                indices,
            )

        actions = storage.actions.flatten(0, 1)
        values = storage.values.flatten(0, 1)
        returns = storage.returns.flatten(0, 1)
        old_log_prob = storage.actions_log_prob.flatten(0, 1)
        advantages = storage.advantages.flatten(0, 1)
        actor_loss_sums = torch.zeros(2, device=self.device)
        for _ in range(self.num_learning_epochs):
            for minibatch in range(self.num_mini_batches):
                selected = indices[minibatch * minibatch_size : (minibatch + 1) * minibatch_size]
                self.actor(observations[selected], stochastic_output=True)
                mean, std = self.actor.output_distribution_params
                # Value inputs are detached rollout values with a zero loss
                # coefficient; only policy/entropy gradients enter this phase.
                loss, _, surrogate, entropy = fused_ppo_loss(
                    mean,
                    std,
                    values[selected],
                    actions[selected],
                    old_log_prob[selected],
                    values[selected],
                    returns[selected],
                    advantages[selected],
                    self.clip_param,
                    0.0,
                    self.entropy_coef,
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                # Critic gradients remain None, so native Adam only advances
                # actor parameters and their individual moment/step tensors.
                self.optimizer.step()
                actor_loss_sums += torch.stack((surrogate.detach(), entropy.detach()))

        self.actor.update_normalization(observations)
        self.critic.update_normalization(observations)
        surrogate_loss, entropy = (actor_loss_sums / (self.num_learning_epochs * self.num_mini_batches)).tolist()
        storage.clear()
        return {"value": float(value_loss), "surrogate": surrogate_loss, "entropy": entropy}

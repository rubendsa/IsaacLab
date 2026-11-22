# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native RGB64 PPO with both networks retained in one optimizer launch.

The kernel retains both networks' weights and Adam moments across the critic
and actor phases, reusing their gradient arena. Rollouts, activations and
communication buffers remain global. GAE, normalization and RNG alignment
remain outside the combined kernel; this is not an all-SRAM learner.
"""

from __future__ import annotations

import torch
from rsl_rl.models.mlp_model import MLPModel

from .persistent_full_perception import SramPersistentFullPerceptionPPO, _consume_native_action_rng
from .persistent_unified import PersistentUnifiedLearner


class SramUnifiedPerceptionPPO(SramPersistentFullPerceptionPPO):
    """Preserve native PPO and checkpoints with one resident-model update."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.storage.num_envs > 16:
            raise ValueError("The unified native visual learner supports at most 16 environments")
        self._unified = None

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        loaded = super().load(loaded_dict, load_cfg, strict)
        self._unified = None
        return loaded

    @torch.no_grad()
    def update(self) -> dict[str, float]:
        """Retain both models across forty critic and forty actor steps."""
        storage = self.storage
        observations = storage.observations.flatten(0, 1)
        batch_size = storage.num_envs * storage.num_transitions_per_env
        indices = torch.randperm(batch_size, requires_grad=False, device=self.device)
        self.optimizer.zero_grad(set_to_none=True)
        if self._unified is None:
            self._unified = PersistentUnifiedLearner(
                self.actor,
                self.critic,
                self.optimizer,
                clip_param=self.clip_param,
                entropy_coef=self.entropy_coef,
                value_loss_coef=self.value_loss_coef,
                max_grad_norm=self.max_grad_norm,
                num_learning_epochs=self.num_learning_epochs,
                num_mini_batches=self.num_mini_batches,
            )
        # Freeze both normalizers throughout the combined update, as native PPO
        # does across its minibatches. The CNN consumes the original uint8 RGB.
        critic_input = self.critic.get_latent(observations).contiguous()
        actor_input = MLPModel.get_latent(self.actor, observations).contiguous()
        value_loss, actor_losses = self._unified.update(
            critic_input,
            actor_input,
            observations[self.actor.obs_groups_2d[0]],
            storage.actions.flatten(0, 1),
            storage.actions_log_prob.flatten(0, 1),
            storage.advantages.flatten(0, 1),
            storage.values.flatten(0, 1),
            storage.returns.flatten(0, 1),
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

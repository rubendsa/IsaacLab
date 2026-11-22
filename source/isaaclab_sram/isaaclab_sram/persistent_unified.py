# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One native critic/actor launch with both networks' weights and moments resident.

The gradient bank is reused after exporting critic gradient diagnostics. Raw
observations, activations, reductions, normalization, GAE and RNG alignment are
outside the resident model state. This is not an entirely on-chip visual learner.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from . import cuda_rt as rt
from .persistent_actor import PersistentActor
from .persistent_critic import PersistentCritic

_KERNEL = Path(__file__).with_name("kernels") / "ppo_native_unified.cu"
_THREADS = 256
_CTAS = 188
_SHARED_BYTES = 66612
_REGISTER_CAP = 224


class PersistentUnifiedLearner:
    """Update the unchanged native networks in one cooperative CUDA launch.

    Supports 1–16 environments with 32 rollout steps on the validated 188-SM
    layout. Each instance binds its scratch buffers to its first CUDA stream;
    cross-stream use is rejected. Checkpoint state is refreshed on every update.
    """

    def __init__(
        self,
        actor,
        critic,
        optimizer,
        *,
        clip_param=0.2,
        entropy_coef=0.005,
        value_loss_coef=1.0,
        max_grad_norm=1.0,
        num_learning_epochs=5,
        num_mini_batches=8,
    ):
        if num_learning_epochs != 5 or num_mini_batches != 8:
            raise ValueError("Unified native PPO requires five epochs and eight minibatches")
        for a in actor.parameters():
            for c in critic.parameters():
                if a is c or (
                    a.device == c.device
                    and max(a.data_ptr(), c.data_ptr())
                    < min(a.data_ptr() + a.numel() * a.element_size(), c.data_ptr() + c.numel() * c.element_size())
                ):
                    raise ValueError("Unified phases require disjoint actor and critic parameter storage")
        # Reuse validated layouts, buffers and preparation without compiling or
        # launching either standalone kernel.
        self.critic = PersistentCritic(
            critic,
            optimizer,
            clip_param=clip_param,
            value_loss_coef=value_loss_coef,
            max_grad_norm=max_grad_norm,
            num_learning_epochs=5,
            num_mini_batches=8,
            _prepare_only=True,
        )
        self.actor = PersistentActor(
            actor,
            optimizer,
            clip_param=clip_param,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            num_learning_epochs=5,
            num_mini_batches=8,
            _prepare_only=True,
        )
        self.device = self.actor.device
        self.optimizer = optimizer
        self.launch_count = 0
        self.shared_bytes = _SHARED_BYTES
        if self.device != self.critic.device or self.actor.num_ctas != _CTAS or self.critic.num_ctas != _CTAS:
            raise ValueError("Unified native layout requires the validated 188-SM CUDA device")
        if self.actor.shard_size != 4253 or self.critic.shard_size != 6200:
            raise ValueError("Native unified parameter shard layout changed")
        with torch.cuda.device(self.device):
            rt.ensure_context(self.device.index)
            if not rt.device_attr("COOPERATIVE_LAUNCH", self.device.index):
                raise ValueError("Unified native learner requires cooperative launch")
            source = _KERNEL.read_text()
            self.module = rt.compile_module(
                source,
                name=_KERNEL.name,
                defines={
                    "THREADS": _THREADS,
                    "N_CTAS": _CTAS,
                    "CRITIC_TILE": 64,
                    "ACTOR_TILE": 8,
                    "CRITIC_SHARD_SIZE": 6200,
                    "ACTOR_SHARD_SIZE": 4253,
                    "MAX_REGS": _REGISTER_CAP,
                },
                options=("-Xptxas=-v",),
            )
            self.kernel = self.module.kernel("unified_native_learner")
            if self.kernel.attr("LOCAL_SIZE_BYTES") != 0:
                raise ValueError("Unified resident model state requires a kernel with zero local memory")
            limit = rt.device_attr("MAX_SHARED_MEMORY_PER_BLOCK_OPTIN", self.device.index)
            if self.shared_bytes + self.kernel.attr("SHARED_SIZE_BYTES") > limit:
                raise ValueError("Compiled unified shared state exceeds the device block limit")
            self.kernel.set_max_dynamic_smem(self.shared_bytes)
            occupancy = self.kernel.occupancy(_THREADS, self.shared_bytes)
            if occupancy < 1 or occupancy * rt.num_sms(self.device.index) < _CTAS:
                raise ValueError("Unified cooperative grid cannot be fully resident")
            self.resource_info = {
                **self.kernel.info(),
                "dynamic_shared_bytes": self.shared_bytes,
                "resident_blocks_per_sm": occupancy,
                "grid_blocks": _CTAS,
                "threads_per_block": _THREADS,
                "register_cap": _REGISTER_CAP,
                "moment_register_values_per_thread": 84,
                "canonical_parameter_count": 1527711,
                "replicated_cnn_parameter_count": 118372,
                "allocated_weight_and_gradient_shared_bytes": self.shared_bytes * _CTAS,
                "logical_resident_weight_and_moment_bytes": (1527711 + 118372) * 12,
                "gradient_bank_scope": "Phase-overlaid shared bank; critic final gradients exported before reuse",
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "cubin_sha256": hashlib.sha256(self.module.cubin).hexdigest(),
                "compiler_log": self.module.log,
                "scope": "Both networks W/m/v entry-to-exit; phase-overlaid G; global rollout/activation/mailboxes",
                "all_sram": False,
                "zero_dram": False,
            }

    @torch.no_grad()
    def update(self, critic_input, actor_input, rgb, actions, old_logp, advantages, values, returns, indices):
        """Return unweighted critic loss and actor [surrogate, entropy] on CUDA."""
        batch = indices.numel()
        if batch % 32 or not 1 <= batch // 32 <= 16:
            raise ValueError("Unified learner supports 1–16 environments with 32-step rollouts")
        c = a = None
        with torch.cuda.device(self.device):
            try:
                c = self.critic._prepare_update(critic_input, values, returns, indices)
                a = self.actor._prepare_update(actor_input, rgb, actions, old_logp, advantages, indices)
                if (
                    c.batch != batch
                    or a.batch != batch
                    or c.indices is not a.indices
                    or c.minibatch_size != a.minibatch_size
                    or c.epochs != a.epochs
                    or c.minibatches != a.minibatches
                ):
                    raise ValueError("Both phases must use identical batch and PPO permutation scheduling")
                # ABI order is defined by the production prepared objects and
                # the combined kernel signature, never an external JSON file.
                if len(c.arguments) != 29 or len(a.arguments) != 39:
                    raise RuntimeError("Unified kernel argument contract changed")
                c.validate()
                a.validate()
                self.kernel.launch(
                    _CTAS, _THREADS, args=(*c.arguments, *a.arguments), smem=self.shared_bytes, cooperative=True
                )
                c.mark_launched()
                a.mark_launched()
                self.launch_count += 1
                # Validate both before mutating either native checkpoint.
                c.validate()
                a.validate()
                c.commit()
                a.commit()
                return c.result, a.result
            finally:
                if c is not None:
                    c.cancel()
                if a is not None:
                    a.cancel()

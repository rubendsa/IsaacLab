# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native model and prepared rollout state retained through one optimizer launch.

Both networks' weights and moments, the raw RGB rollout, and the normalized
numeric PPO inputs remain on-chip through the critic and actor phases. Selected
inputs are republished through global tiles; activations and reductions remain
global. Normalization, GAE, RNG alignment, packing and copyback remain outside.
This is not an all-SRAM learner or a zero-DRAM implementation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from . import cuda_rt as rt
from .persistent_actor import PersistentActor
from .persistent_critic import PersistentCritic

_KERNEL = Path(__file__).with_name("kernels") / "ppo_native_resident.cu"
_THREADS = 256
_CTAS = 188
_SHARED_BYTES = 91200
_MODEL_SHARED_BYTES = 66612
_NUMERIC_SHARED_FLOATS = 6147
_REGISTER_CAP = 255


class PersistentResidentLearner:
    """Retain native models, raw RGB and prepared inputs in one cooperative launch.

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
            raise ValueError("Resident native PPO requires five epochs and eight minibatches")
        for a in actor.parameters():
            for c in critic.parameters():
                if a is c or (
                    a.device == c.device
                    and max(a.data_ptr(), c.data_ptr())
                    < min(a.data_ptr() + a.numel() * a.element_size(), c.data_ptr() + c.numel() * c.element_size())
                ):
                    raise ValueError("Resident phases require disjoint actor and critic parameter storage")
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
            raise ValueError("Resident native layout requires the validated 188-SM CUDA device")
        if self.actor.shard_size != 4253 or self.critic.shard_size != 6200:
            raise ValueError("Native resident parameter shard layout changed")
        with torch.cuda.device(self.device):
            rt.ensure_context(self.device.index)
            if not rt.device_attr("COOPERATIVE_LAUNCH", self.device.index):
                raise ValueError("Resident native learner requires cooperative launch")
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
                raise ValueError("Resident model/input state requires a kernel with zero local memory")
            limit = rt.device_attr("MAX_SHARED_MEMORY_PER_BLOCK_OPTIN", self.device.index)
            if self.shared_bytes + self.kernel.attr("SHARED_SIZE_BYTES") > limit:
                raise ValueError("Compiled resident shared state exceeds the device block limit")
            self.kernel.set_max_dynamic_smem(self.shared_bytes)
            occupancy = self.kernel.occupancy(_THREADS, self.shared_bytes)
            if occupancy < 1 or occupancy * rt.num_sms(self.device.index) < _CTAS:
                raise ValueError("Resident cooperative grid cannot be fully resident")
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
                "allocated_weight_and_gradient_shared_bytes": _MODEL_SHARED_BYTES * _CTAS,
                "allocated_total_dynamic_shared_bytes": self.shared_bytes * _CTAS,
                "numeric_shared_floats_per_cta": _NUMERIC_SHARED_FLOATS,
                "numeric_shared_bytes_per_cta": _NUMERIC_SHARED_FLOATS * 4,
                "numeric_logical_bytes_at_n16": 4622336,
                "numeric_allocated_shared_bytes": _NUMERIC_SHARED_FLOATS * 4 * _CTAS,
                "rgb_register_words_per_thread": 33,
                "rgb_logical_bytes_at_n16": 6291456,
                "rgb_allocated_register_bytes": 33 * 4 * _THREADS * _CTAS,
                "global_rgb_tile_bytes": 98304,
                "global_numeric_tile_bytes": 408832,
                "logical_resident_weight_and_moment_bytes": (1527711 + 118372) * 12,
                "gradient_bank_scope": "Phase-overlaid shared bank; critic final gradients exported before reuse",
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "cubin_sha256": hashlib.sha256(self.module.cubin).hexdigest(),
                "compiler_log": self.module.log,
                "scope": (
                    "Both networks W/m/v, raw RGB and prepared numeric rollout retained across both phases; "
                    "phase-overlaid G; global selected-input tiles, activations and reduction mailboxes; "
                    "normalization/GAE/RNG/packing/copyback outside"
                ),
                "all_sram": False,
                "zero_dram": False,
            }

    @torch.no_grad()
    def update(self, critic_input, actor_input, rgb, actions, old_logp, advantages, values, returns, indices):
        """Return unweighted critic loss and actor [surrogate, entropy] on CUDA."""
        batch = indices.numel()
        if batch % 32 or not 1 <= batch // 32 <= 16:
            raise ValueError("Resident learner supports 1–16 environments with 32-step rollouts")
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
                    raise RuntimeError("Resident kernel argument contract changed")
                if rgb.data_ptr() % 4:
                    raise ValueError("Resident RGB import requires aligned 32-bit words")
                inverse = torch.empty_like(indices)
                inverse.scatter_(0, indices, torch.arange(batch, device=self.device, dtype=torch.int64))
                rgb_tile = torch.empty((8, 3, 64, 64), device=self.device, dtype=torch.uint8)
                numeric_tile = torch.empty(102208, device=self.device, dtype=torch.float32)
                # Prepared advantages/returns already contain the outer GAE result.
                # Any future in-kernel GAE must finish before the kernel imports
                # numeric records; that composition requires its own parity gate.
                c.validate()
                a.validate()
                self.kernel.launch(
                    _CTAS,
                    _THREADS,
                    args=(*c.arguments, *a.arguments, inverse, rgb_tile, numeric_tile),
                    smem=self.shared_bytes,
                    cooperative=True,
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

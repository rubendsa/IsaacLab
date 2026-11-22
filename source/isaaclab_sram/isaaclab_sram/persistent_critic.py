# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Persistent native Kuka critic PPO update with explicit shared-memory state.

The 1595 -> 512 -> 256 -> 128 -> 1 ELU network is unchanged. One cooperative
launch performs all PPO minibatches; weights, gradients and Adam moments stay
in block-owned shared-memory shards during that launch. Normalized inputs,
activation tiles, input-adjoint partials, synchronization and statistics remain
in global memory. Packing and native checkpoint copyback occur at boundaries.
This is a resident critic working set, not a resident simulator or full learner.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn

from . import cuda_rt as rt
from .persistent_state import PackedAdamState, PreparedUpdate, UpdateGuard, prepare_update

_DIMS = (1595, 512, 256, 128, 1)
_THREADS = 256
_TILE = 64
_NUM_PARAMS = 981505
_KERNEL = Path(__file__).with_name("kernels") / "ppo_native_critic.cu"


class PersistentCritic:
    """Update a native critic and its entries in a possibly shared Adam optimizer.

    The update accepts already-normalized observations; normalization remains
    frozen throughout the update and is the caller's responsibility afterward.
    The single supplied permutation is reused for every epoch, as in RSL-RL.
    Actor optimizer entries are untouched. Critic gradients are cleared after
    copyback so an actor-only optimizer step cannot update them a second time.
    """

    def __init__(
        self,
        critic: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        num_mini_batches: int = 8,
        num_learning_epochs: int = 5,
        clip_param: float = 0.2,
        value_loss_coef: float = 1.0,
        max_grad_norm: float = 1.0,
        _prepare_only: bool = False,
    ):
        self.critic = critic
        self.optimizer = optimizer
        self.num_mini_batches = int(num_mini_batches)
        self.num_learning_epochs = int(num_learning_epochs)
        self.clip_param = float(clip_param)
        self.value_loss_coef = float(value_loss_coef)
        self.max_grad_norm = float(max_grad_norm)
        if min(self.num_mini_batches, self.num_learning_epochs) <= 0:
            raise ValueError("PPO epoch and minibatch counts must be positive")
        if not all(math.isfinite(x) for x in (self.clip_param, self.value_loss_coef, self.max_grad_norm)):
            raise ValueError("PPO coefficients must be finite")
        if min(self.clip_param, self.value_loss_coef) < 0 or self.max_grad_norm <= 0:
            raise ValueError("Clipping and value coefficients must be nonnegative; max_grad_norm must be positive")
        mlp = critic.mlp if hasattr(critic, "mlp") else critic
        if not isinstance(mlp, nn.Sequential):
            raise ValueError("PersistentCritic requires a sequential native MLP")
        # Native RSL-RL reuses one stateless ELU instance at all hidden layers.
        # children() deduplicates that module; sequence iteration preserves it.
        layers = list(mlp)
        if len(layers) != 7:
            raise ValueError("PersistentCritic requires the native four-linear-layer ELU critic")
        self.parameters = []
        for i in range(4):
            layer = layers[2 * i]
            if (
                not isinstance(layer, nn.Linear)
                or (layer.in_features, layer.out_features) != _DIMS[i : i + 2]
                or layer.bias is None
            ):
                raise ValueError("PersistentCritic requires 1595 -> 512 -> 256 -> 128 -> 1 with biases")
            self.parameters.extend((layer.weight, layer.bias))
            if i < 3 and (not isinstance(layers[2 * i + 1], nn.ELU) or layers[2 * i + 1].alpha != 1):
                raise ValueError("PersistentCritic requires ELU(alpha=1) hidden activations")
        if {id(p) for p in critic.parameters()} != {id(p) for p in self.parameters}:
            raise ValueError("The critic contains additional unsupported trainable parameters")
        self._parameter_shapes = tuple(p.shape for p in self.parameters)
        self.device = self.parameters[0].device
        if self.device.type != "cuda" or any(
            p.device != self.device or p.dtype != torch.float32 or not p.is_contiguous() or not p.requires_grad
            for p in self.parameters
        ):
            raise ValueError("Critic parameters must be contiguous trainable float32 tensors on one CUDA device")
        if type(optimizer) is not torch.optim.Adam:
            raise ValueError("PersistentCritic supports torch.optim.Adam only")
        self._optimizer_config()
        self._update_guard = UpdateGuard(self.device)
        self.kernel = None
        self.resource_info = {}
        with torch.cuda.device(self.device):
            rt.ensure_context(self.device.index)
            if not rt.device_attr("COOPERATIVE_LAUNCH", self.device.index):
                raise ValueError("The CUDA device does not support cooperative launch")
            self.num_ctas = rt.num_sms(self.device.index)
            self.shard_size = sum(
                math.ceil(out_features / self.num_ctas) * (in_features + 1)
                for in_features, out_features in zip(_DIMS[:-1], _DIMS[1:], strict=True)
            )
            self.shared_bytes = 4 * 4 * self.shard_size
            if not _prepare_only:
                limit = rt.device_attr("MAX_SHARED_MEMORY_PER_BLOCK_OPTIN", self.device.index)
                if self.shared_bytes + 256 > limit:
                    raise ValueError(
                        f"Native critic state needs {self.shared_bytes} shared bytes per block over "
                        f"{self.num_ctas} SMs; device block limit is {limit}. Architecture is not reduced."
                    )
                source = _KERNEL.read_text()
                self.module = rt.compile_module(
                    source,
                    name=_KERNEL.name,
                    defines={
                        "THREADS": _THREADS,
                        "TILE": _TILE,
                        "N_CTAS": self.num_ctas,
                        "SHARD_SIZE": self.shard_size,
                    },
                    options=("-Xptxas=-v",),
                )
                self.kernel = self.module.kernel("persistent_native_critic")
                if self.shared_bytes + self.kernel.attr("SHARED_SIZE_BYTES") > limit:
                    raise ValueError("Compiled static plus dynamic shared memory exceeds the device block limit")
                self.kernel.set_max_dynamic_smem(self.shared_bytes)
                occupancy = self.kernel.occupancy(_THREADS, self.shared_bytes)
                if occupancy < 1 or self.num_ctas > occupancy * rt.num_sms(self.device.index):
                    raise ValueError("The cooperative grid cannot be fully resident")
                self.resource_info = {
                    **self.kernel.info(),
                    "grid_blocks": self.num_ctas,
                    "threads_per_block": _THREADS,
                    "resident_blocks_per_sm": occupancy,
                    "dynamic_shared_bytes_per_block": self.shared_bytes,
                    "parameter_count": _NUM_PARAMS,
                    "working_state_bytes": _NUM_PARAMS * 16,
                    "tile_samples": _TILE,
                    "compiler_log": self.module.log,
                    "cubin_sha256": hashlib.sha256(self.module.cubin).hexdigest(),
                    "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "scope": "Critic weights/gradients/Adam moments in shared; inputs and reduction mailboxes global",
                    "zero_dram_claim": False,
                }
            self._acts = torch.empty(_TILE * 897, dtype=torch.float32, device=self.device)
            self._deltas = torch.empty_like(self._acts)
            self._partials = torch.empty(self.num_ctas * _TILE * 512, dtype=torch.float32, device=self.device)
            self._norm_partials = torch.empty(self.num_ctas, dtype=torch.float64, device=self.device)
            self._sample_losses = torch.empty(_TILE, dtype=torch.float32, device=self.device)
            self._barrier = torch.zeros(1, dtype=torch.int32, device=self.device)
            self.last_gradients = torch.empty(_NUM_PARAMS, dtype=torch.float32, device=self.device)
            self.last_statistics = torch.empty(
                self.num_mini_batches * self.num_learning_epochs, 2, dtype=torch.float32, device=self.device
            )

    def _optimizer_config(self) -> tuple[float, float, float, float, bool, list[dict]]:
        groups = []
        configurations = []
        for parameter in self.parameters:
            matches = [group for group in self.optimizer.param_groups if any(p is parameter for p in group["params"])]
            if len(matches) != 1:
                raise ValueError("Each critic parameter must belong to exactly one optimizer group")
            group = matches[0]
            if any(
                group.get(key, False) for key in ("amsgrad", "maximize", "differentiable", "decoupled_weight_decay")
            ):
                raise ValueError("AMSGrad, maximize, differentiable Adam and decoupled decay are unsupported")
            if group["weight_decay"] != 0:
                raise ValueError("Weight decay is unsupported")
            if group.get("capturable") and not group.get("fused"):
                raise ValueError("Capturable Adam requires fused=True")
            values = (group["lr"], *group["betas"], group["eps"])
            if any(isinstance(value, torch.Tensor) or not math.isfinite(value) for value in values):
                raise ValueError("Adam learning rate, betas and epsilon must be finite Python scalars")
            lr, beta1, beta2, eps = values
            if lr < 0 or not (0 <= beta1 < 1 and 0 <= beta2 < 1) or eps <= 0:
                raise ValueError("Unsupported Adam scalar configuration")
            configurations.append((*values, bool(group.get("fused"))))
            groups.append(group)
        if any(config != configurations[0] for config in configurations):
            raise ValueError("All critic parameter groups must use identical Adam hyperparameters")
        return *configurations[0], groups

    @torch.no_grad()
    def update(self, normalized_inputs, old_values, returns, indices):
        """Perform all native critic steps and return unweighted mean value loss."""
        if self.kernel is None:
            raise RuntimeError("This critic was constructed for preparation only")
        with torch.cuda.device(self.device):
            prepared = self._prepare_update(normalized_inputs, old_values, returns, indices)
            return prepared.launch(self.kernel, self.num_ctas, _THREADS, self.shared_bytes)

    @torch.no_grad()
    @prepare_update
    def _prepare_update(
        self, normalized_inputs: torch.Tensor, old_values: torch.Tensor, returns: torch.Tensor, indices: torch.Tensor
    ) -> PreparedUpdate:
        """Prepare all native critic minibatches, returning unweighted mean value loss.

        Inputs are contiguous CUDA float32 [B,1595], [B,1] and [B,1].
        Indices are a CUDA int64 permutation of 0..B-1. The batch must divide
        evenly into minibatches. All optimizer state is refreshed on entry, so
        native checkpoint loading and consecutive updates remain supported.
        """
        if normalized_inputs.ndim != 2:
            raise ValueError("Normalized inputs must be a two-dimensional tensor")
        batch = normalized_inputs.shape[0]
        if normalized_inputs.shape != (batch, 1595) or batch == 0 or batch % self.num_mini_batches:
            raise ValueError("Normalized inputs must be [B,1595] with B positive and divisible by minibatches")
        for name, tensor, shape, dtype in (
            ("normalized_inputs", normalized_inputs, (batch, 1595), torch.float32),
            ("old_values", old_values, (batch, 1), torch.float32),
            ("returns", returns, (batch, 1), torch.float32),
            ("indices", indices, (batch,), torch.int64),
        ):
            if (
                tensor.shape != shape
                or tensor.device != self.device
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
            ):
                raise ValueError(f"{name} must be contiguous {dtype} with shape {shape} on {self.device}")
        with torch.cuda.device(self.device):
            rt.ensure_context(self.device.index)
            if {id(p) for p in self.critic.parameters()} != {id(p) for p in self.parameters} or any(
                p.shape != shape or not p.requires_grad
                for p, shape in zip(self.parameters, self._parameter_shapes, strict=True)
            ):
                raise ValueError("Native critic parameter identity or shape changed")
            if not torch.equal(torch.sort(indices).values, torch.arange(batch, device=self.device)):
                raise ValueError("indices must be a permutation of the complete batch")
            lr, beta1, beta2, epsilon, fused, groups = self._optimizer_config()
            updates = self.num_learning_epochs * self.num_mini_batches
            state = PackedAdamState.prepare(
                self.parameters,
                self.optimizer,
                groups,
                updates,
                self.device,
                (lr, beta1, beta2, epsilon, fused),
            )
            flat, moments, variances, corrections = state.flat, state.moments, state.variances, state.corrections
            if not bool(
                torch.stack([torch.isfinite(value).all() for value in (normalized_inputs, old_values, returns)]).all()
            ):
                raise ValueError("Persistent inputs must be finite")
            result = torch.empty((), dtype=torch.float32, device=self.device)
            return PreparedUpdate(
                state=state,
                arguments=(
                    flat,
                    moments,
                    variances,
                    self.last_gradients,
                    normalized_inputs,
                    old_values,
                    returns,
                    indices,
                    corrections,
                    batch // self.num_mini_batches,
                    self.num_learning_epochs,
                    self.num_mini_batches,
                    self.clip_param,
                    self.value_loss_coef,
                    self.max_grad_norm,
                    beta1,
                    1.0 - beta1,
                    beta2,
                    1.0 - beta2,
                    epsilon,
                    int(fused),
                    self._acts,
                    self._deltas,
                    self._partials,
                    self._norm_partials,
                    self._sample_losses,
                    self.last_statistics,
                    result,
                    self._barrier,
                ),
                result=result,
                batch=batch,
                indices=indices,
                minibatch_size=batch // self.num_mini_batches,
                epochs=self.num_learning_epochs,
                minibatches=self.num_mini_batches,
            )

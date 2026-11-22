# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Persistent native RGB64 actor PPO phase with explicit shared-memory state.

The state normalizer is frozen and evaluated by the caller; raw uint8 RGB
normalization, CNN, learned spatial temperature, MLP, Gaussian loss, actor-wide
clipping and all Adam steps execute inside one cooperative launch. Global
rollout/activation/reduction buffers remain. This is not unified critic+actor
residency or a zero-DRAM claim.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn

from . import cuda_rt as rt
from .persistent_state import PackedAdamState, PreparedUpdate, UpdateGuard, prepare_update

DIMS = (699, 512, 256, 128, 23)
NUM_PARAMS = 546206
THREADS = 256
TILE = 8
_KERNEL = Path(__file__).with_name("kernels") / "ppo_native_actor.cu"


class PersistentActor:
    """Run the complete RGB64 actor update while its optimizer state stays shared.

    Inputs and normalizers remain caller-owned. The caller must preserve the
    discarded Gaussian sampling RNG consumed by native PPO.
    """

    def _optimizer_config(self) -> tuple[float, float, float, float, bool, list[dict]]:
        groups = []
        configurations = []
        for parameter in self.parameters:
            matches = [group for group in self.optimizer.param_groups if any(p is parameter for p in group["params"])]
            if len(matches) != 1:
                raise ValueError("Each actor parameter must belong to exactly one optimizer group")
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
            raise ValueError("All actor parameter groups must use identical Adam hyperparameters")
        return *configurations[0], groups

    def __init__(
        self,
        actor: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 8,
        clip_param: float = 0.2,
        entropy_coef: float = 0.005,
        max_grad_norm: float = 1.0,
        _prepare_only: bool = False,
    ):
        self.actor = actor
        self.optimizer = optimizer
        if type(num_learning_epochs) is not int or type(num_mini_batches) is not int:
            raise ValueError("Epoch and minibatch counts must be integers")
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.clip_param = float(clip_param)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        if not 1 <= self.num_learning_epochs <= 5 or not 1 <= self.num_mini_batches <= 8:
            raise ValueError("Supports one to five epochs and one to eight minibatches")
        if not all(math.isfinite(v) for v in [self.clip_param, self.entropy_coef, self.max_grad_norm]):
            raise ValueError("Finite PPO coefficients required")
        if self.clip_param < 0 or self.entropy_coef < 0 or self.max_grad_norm <= 0:
            raise ValueError("Unsupported PPO coefficients")
        if not isinstance(actor.mlp, nn.Sequential):
            raise ValueError("A sequential native actor MLP is required")
        # Sequence iteration preserves the native repeated ELU module.
        mlp = list(actor.mlp)
        if len(mlp) != 7:
            raise ValueError("Native four-layer actor MLP required")
        self.parameters = []
        for i in range(4):
            layer = mlp[2 * i]
            if (
                not isinstance(layer, nn.Linear)
                or (layer.in_features, layer.out_features) != DIMS[i : i + 2]
                or layer.bias is None
                or layer.weight.shape != (DIMS[i + 1], DIMS[i])
                or layer.bias.shape != (DIMS[i + 1],)
            ):
                raise ValueError("Native 699 -> 512 -> 256 -> 128 -> 23 actor required")
            self.parameters.extend([layer.weight, layer.bias])
            if i < 3 and (not isinstance(mlp[2 * i + 1], nn.ELU) or mlp[2 * i + 1].alpha != 1):
                raise ValueError("Native ELU required")
        if len(actor.obs_groups_2d) != 1 or len(actor.cnns) != 1 or len(actor.softmaxes) != 1:
            raise ValueError("Exactly one RGB64 camera encoder required")
        group = actor.obs_groups_2d[0]
        self.image_normalizer = actor.image_normalizers[group]
        cnn = list(actor.cnns[group])
        if len(cnn) != 6 or not actor.image_normalizers[group].is_uint8:
            raise ValueError("Native three-layer uint8 RGB CNN required")
        for i, (cin, cout, k, s) in enumerate([(3, 16, 8, 4), (16, 32, 4, 2), (32, 32, 3, 1)]):
            layer = cnn[2 * i]
            if not isinstance(layer, nn.Conv2d) or (
                layer.in_channels,
                layer.out_channels,
                layer.kernel_size,
                layer.stride,
            ) != (cin, cout, (k, k), (s, s)):
                raise ValueError("Native convolution shape required")
            if layer.padding != (0, 0) or layer.dilation != (1, 1) or layer.groups != 1 or layer.bias is None:
                raise ValueError("Unsupported convolution configuration")
            if layer.weight.shape != (cout, cin, k, k) or layer.bias.shape != (cout,):
                raise ValueError("Native convolution parameter shapes are required")
            if not isinstance(cnn[2 * i + 1], nn.ELU) or cnn[2 * i + 1].alpha != 1:
                raise ValueError("Native CNN ELU required")
            self.parameters.extend([layer.weight, layer.bias])
        spatial = actor.softmaxes[group]
        self.spatial = spatial
        if (spatial.num_channels, spatial.height, spatial.width) != (32, 4, 4):
            raise ValueError("Native 32-channel 4x4 spatial softmax required")
        distribution = actor.distribution
        if distribution.std_type != "scalar" or not distribution.std_param.requires_grad:
            raise ValueError("Learned scalar standard deviations required")
        self.parameters.extend([spatial.log_temperature, distribution.std_param])
        self._parameter_shapes = tuple(p.shape for p in self.parameters)
        if {id(p) for p in actor.parameters()} != {id(p) for p in self.parameters}:
            raise ValueError("Unexpected actor parameters")
        self.std_min, self.std_max = distribution.std_range
        if not (math.isfinite(self.std_min) and math.isfinite(self.std_max) and 0 < self.std_min < self.std_max):
            raise ValueError("Finite positive ordered standard-deviation bounds required")
        self.device = self.parameters[0].device
        if self.device.type != "cuda" or any(
            p.device != self.device or p.dtype != torch.float32 or not p.is_contiguous() or not p.requires_grad
            for p in self.parameters
        ):
            raise ValueError("One CUDA device and contiguous trainable fp32 actor required")
        if type(optimizer) is not torch.optim.Adam:
            raise ValueError("torch.optim.Adam required")
        self._validate_model_tensors()
        self._optimizer_config()
        self._update_guard = UpdateGuard(self.device)
        self.kernel = None
        self.resource_info = {}
        with torch.cuda.device(self.device):
            rt.ensure_context(self.device.index)
            if not rt.device_attr("COOPERATIVE_LAUNCH", self.device.index):
                raise ValueError("Cooperative launch required")
            self.num_ctas = rt.num_sms(self.device.index)
            # Spatial channels and scalar deviations each have one owning CTA.
            if self.num_ctas < 32:
                raise ValueError("Native actor requires at least 32 resident CTAs for channel ownership")
            shapes = [(699, 512), (512, 256), (256, 128), (128, 23), (192, 16), (256, 32), (288, 32), (0, 32), (0, 23)]
            self.shard_size = sum(math.ceil(no / self.num_ctas) * (ni + 1) for ni, no in shapes)
            self.shared_bytes = 16 * self.shard_size
            # Each CTA retains one filter per CNN layer, including its optimizer
            # state. The maximum shard is unchanged: CTA0 already owns all three.
            replicated_cnn_parameters = sum(
                (self.num_ctas - channels) * (inputs + 1) for inputs, channels in shapes[4:7]
            )
            if not _prepare_only:
                limit = rt.device_attr("MAX_SHARED_MEMORY_PER_BLOCK_OPTIN", self.device.index)
                if self.shared_bytes + 256 > limit:
                    raise ValueError("Native actor state cannot fit device shared memory; architecture is not reduced")
                source = _KERNEL.read_text()
                self.module = rt.compile_module(
                    source,
                    name=_KERNEL.name,
                    defines={"THREADS": THREADS, "TILE": TILE, "N_CTAS": self.num_ctas, "SHARD_SIZE": self.shard_size},
                    options=("-Xptxas=-v",),
                )
                self.kernel = self.module.kernel("persistent_native_actor")
                if self.shared_bytes + self.kernel.attr("SHARED_SIZE_BYTES") > limit:
                    raise ValueError("Native actor state exceeds shared memory limit")
                self.kernel.set_max_dynamic_smem(self.shared_bytes)
                occupancy = self.kernel.occupancy(THREADS, self.shared_bytes)
                if occupancy < 1 or self.num_ctas > occupancy * rt.num_sms(self.device.index):
                    raise ValueError("Cooperative actor grid cannot be resident")
                self.resource_info = {
                    **self.kernel.info(),
                    "resident_blocks_per_sm": occupancy,
                    "dynamic_shared_bytes_per_block": self.shared_bytes,
                    "grid_blocks": self.num_ctas,
                    "threads_per_block": THREADS,
                    "parameter_count": NUM_PARAMS,
                    "working_state_bytes": NUM_PARAMS * 16,
                    "canonical_parameter_count": NUM_PARAMS,
                    "canonical_working_state_bytes": NUM_PARAMS * 16,
                    "resident_parameter_count": NUM_PARAMS + replicated_cnn_parameters,
                    "resident_working_state_bytes": (NUM_PARAMS + replicated_cnn_parameters) * 16,
                    "replicated_cnn_parameter_count": replicated_cnn_parameters,
                    "replicated_cnn_working_state_bytes": replicated_cnn_parameters * 16,
                    "allocated_model_shared_bytes": self.shared_bytes * self.num_ctas,
                    "compiler_log": self.module.log,
                    "cubin_sha256": hashlib.sha256(self.module.cubin).hexdigest(),
                    "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "scope": (
                        "Canonical and replicated actor W/G/Adam shared during all steps; "
                        "rollout/mailboxes global; separate critic phase"
                    ),
                    "zero_dram_claim": False,
                    "tile_samples": TILE,
                    "cnn_replica_groups": [16, 32, 32],
                    "replica_gradient_mailbox_bytes": self.num_ctas * 739 * 4,
                }

            def buffer(count, dtype=torch.float32):
                return torch.empty(count, dtype=dtype, device=self.device)

            self.cnn = buffer(TILE * 5264)
            self.cd = buffer(TILE * 5264)
            self.probs = buffer(TILE * 512)
            self.acts = buffer(TILE * 1618)
            self.deltas = buffer(TILE * 1618)
            self.partials = buffer(
                max(TILE * max(self.num_ctas * 512, min(self.num_ctas, 32) * 3600), self.num_ctas * 739)
            )
            self.norms = buffer(self.num_ctas, torch.float64)
            self.stds = buffer(23)
            self.std_grads = buffer(TILE * 23)
            self.sample_metrics = buffer(TILE * 2)
            self.last_statistics = buffer((self.num_learning_epochs * self.num_mini_batches, 3))
            self.last_gradients = buffer(NUM_PARAMS)
            self.barrier = torch.zeros(1, dtype=torch.int32, device=self.device)
            self.resource_info["adjoint_mailbox_bytes"] = self.partials.numel() * self.partials.element_size()

    def _validate_model_tensors(self) -> None:
        """Reject changed model storage and malformed checkpoint auxiliary buffers."""
        if {id(p) for p in self.actor.parameters()} != {id(p) for p in self.parameters}:
            raise ValueError("Actor parameter objects changed; rebuild PersistentActor")
        if any(
            p.device != self.device or p.dtype != torch.float32 or not p.is_contiguous() or not p.requires_grad
            for p in self.parameters
        ):
            raise ValueError("Actor parameter dtype/device/contiguity changed")
        if tuple(p.shape for p in self.parameters) != self._parameter_shapes:
            raise ValueError("Actor parameter shapes changed")
        if tuple(self.actor.distribution.std_range) != (self.std_min, self.std_max):
            raise ValueError("Standard-deviation bounds changed; rebuild PersistentActor")
        if not self.image_normalizer.is_uint8:
            raise ValueError("The actor requires the native uint8 image normalizer")
        if sum(p.numel() for p in self.parameters) != NUM_PARAMS:
            raise ValueError("Native actor must contain exactly 546206 parameters")
        if self.spatial.log_temperature.shape != (32,) or self.actor.distribution.std_param.shape != (23,):
            raise ValueError("Temperature/std parameter shapes must be [32] and [23]")
        for name in ("pos_x", "pos_y"):
            value = getattr(self.spatial, name)
            if value.shape != (1, 16) or value.device != self.device or value.dtype != torch.float32:
                raise ValueError("Spatial coordinates must be float32 [1,16] on the actor device")
        for name, expected in (
            ("shift", -0.5),
            ("scale", torch.tensor(1.0 / 255.0, dtype=torch.float32, device="cpu").item()),
        ):
            value = getattr(self.image_normalizer, name)
            if (
                value.shape != ()
                or value.dtype != torch.float32
                or value.device != self.device
                or value.item() != expected
            ):
                raise ValueError("Native uint8 image normalization requires shift=-0.5 and scale=1/255")

    @torch.no_grad()
    def update(self, normalized_state, raw_rgb, actions, old_logp, advantages, indices):
        """Perform the complete native actor update on this wrapper's CUDA stream."""
        if self.kernel is None:
            raise RuntimeError("This actor was constructed for preparation only")
        with torch.cuda.device(self.device):
            prepared = self._prepare_update(normalized_state, raw_rgb, actions, old_logp, advantages, indices)
            return prepared.launch(self.kernel, self.num_ctas, THREADS, self.shared_bytes)

    @torch.no_grad()
    @prepare_update
    def _prepare_update(
        self,
        normalized_state: torch.Tensor,
        raw_rgb: torch.Tensor,
        actions: torch.Tensor,
        old_logp: torch.Tensor,
        advantages: torch.Tensor,
        indices: torch.Tensor,
    ) -> PreparedUpdate:
        """Prepare CUDA [mean surrogate loss, mean entropy] after all actor steps.

        State inputs are frozen normalized [B,635], RGB is uint8 [B,3,64,64],
        actions are float32 [B,23], log probabilities and advantages are float32
        [B] or [B,1], and indices are one int64 permutation reused each epoch.
        Supports B<=2048. Native parameters, moments and individual Adam steps
        are refreshed at every entry; critic optimizer entries remain untouched.
        The caller preserves native discarded-action RNG and updates normalizers.
        """
        if normalized_state.ndim != 2:
            raise ValueError("normalized_state must be [B,635]")
        batch = normalized_state.shape[0]
        if not 1 <= batch <= 2048 or batch % self.num_mini_batches:
            raise ValueError("Batch must be 1–2048 and divisible by minibatch count")
        if old_logp.shape not in ((batch,), (batch, 1)) or advantages.shape not in ((batch,), (batch, 1)):
            raise ValueError("Log probabilities and advantages must be [B] or [B,1]")
        old_logp = old_logp.view(-1)
        advantages = advantages.view(-1)
        for name, tensor, shape, dtype in [
            ("state", normalized_state, (batch, 635), torch.float32),
            ("RGB", raw_rgb, (batch, 3, 64, 64), torch.uint8),
            ("actions", actions, (batch, 23), torch.float32),
            ("old_logp", old_logp, (batch,), torch.float32),
            ("advantages", advantages, (batch,), torch.float32),
            ("indices", indices, (batch,), torch.int64),
        ]:
            if (
                tensor.shape != shape
                or tensor.device != self.device
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
            ):
                raise ValueError(f"Invalid {name} shape/dtype/device/contiguity")
        with torch.cuda.device(self.device):
            rt.ensure_context(self.device.index)
            self._validate_model_tensors()
            coordinates = torch.cat([self.spatial.pos_x.flatten(), self.spatial.pos_y.flatten()]).contiguous()
            if not torch.equal(indices.sort().values, torch.arange(batch, device=self.device)):
                raise ValueError("indices must be full permutation")
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
                torch.stack(
                    [
                        torch.isfinite(value).all()
                        for value in (coordinates, normalized_state, actions, old_logp, advantages)
                    ]
                ).all()
            ):
                raise ValueError("Persistent inputs must be finite")
            result = torch.empty(2, dtype=torch.float32, device=self.device)
            return PreparedUpdate(
                state=state,
                arguments=(
                    flat,
                    moments,
                    variances,
                    self.last_gradients,
                    normalized_state,
                    raw_rgb,
                    actions,
                    old_logp,
                    advantages,
                    indices,
                    coordinates,
                    corrections,
                    batch // self.num_mini_batches,
                    self.num_learning_epochs,
                    self.num_mini_batches,
                    self.clip_param,
                    self.entropy_coef,
                    self.max_grad_norm,
                    self.std_min,
                    self.std_max,
                    beta1,
                    1.0 - beta1,
                    beta2,
                    1.0 - beta2,
                    epsilon,
                    int(fused),
                    self.cnn,
                    self.cd,
                    self.probs,
                    self.acts,
                    self.deltas,
                    self.partials,
                    self.norms,
                    self.stds,
                    self.std_grads,
                    self.sample_metrics,
                    self.last_statistics,
                    result,
                    self.barrier,
                ),
                result=result,
                batch=batch,
                indices=indices,
                minibatch_size=batch // self.num_mini_batches,
                epochs=self.num_learning_epochs,
                minibatches=self.num_mini_batches,
            )

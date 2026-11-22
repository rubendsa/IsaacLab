# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Internal preparation and checkpoint boundary shared by persistent updates."""

from __future__ import annotations

import functools
import math
import threading
from dataclasses import dataclass, field

import torch


class UpdateGuard:
    """Serialize a wrapper's scratch buffers and bind them to one CUDA stream."""

    def __init__(self, device):
        self.device = device
        self.stream = None
        self.lock = threading.Lock()

    def begin(self):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("A persistent update is already prepared or running")
        try:
            current = torch.cuda.current_stream(self.device).cuda_stream
            if self.stream is not None and current != self.stream:
                raise RuntimeError("Persistent scratch buffers are bound to their first CUDA stream")
            self.stream = current
        except BaseException:
            self.lock.release()
            raise

    def check_stream(self):
        if torch.cuda.current_stream(self.device).cuda_stream != self.stream:
            raise RuntimeError("Prepare, launch and commit must use the same CUDA stream")


def prepare_update(function):
    """Reserve scratch until a returned preparation commits or is cancelled."""

    @functools.wraps(function)
    def wrapped(self, *args, **kwargs):
        with torch.cuda.device(self.device):
            self._update_guard.begin()
            try:
                prepared = function(self, *args, **kwargs)
                prepared.guard = self._update_guard
                return prepared
            except BaseException:
                self._update_guard.lock.release()
                raise

    return wrapped


def _tensor_token(tensor):
    return (id(tensor), tensor.data_ptr(), tensor._version, tensor.shape, tensor.dtype, tensor.device)


@dataclass
class PackedAdamState:
    """Native tensor identities and packed buffers for one checkpoint update."""

    parameters: tuple
    optimizer: torch.optim.Optimizer
    states: tuple
    original_states: tuple
    steps: tuple[int, ...]
    updates: int
    flat: torch.Tensor
    moments: torch.Tensor
    variances: torch.Tensor
    corrections: torch.Tensor
    tokens: tuple = field(repr=False)

    @classmethod
    @torch.no_grad()
    def prepare(cls, parameters, optimizer, groups, updates, device, config):
        lr, beta1, beta2, _, fused = config
        states, original_states, steps = [], [], []
        for parameter, group in zip(parameters, groups, strict=True):
            if parameter.device != device or parameter.dtype != torch.float32 or not parameter.is_contiguous():
                raise ValueError("Native parameters must remain contiguous fp32 on the original device")
            original = optimizer.state.get(parameter)
            state = original
            if not state:
                # New optimizer entries remain private until successful commit.
                # Invalid input or launch failure must not change a checkpoint.
                state = {
                    "step": torch.zeros(
                        (), dtype=torch.float32, device=device if fused or group.get("capturable") else "cpu"
                    ),
                    "exp_avg": torch.zeros_like(parameter),
                    "exp_avg_sq": torch.zeros_like(parameter),
                }
            for name in ("exp_avg", "exp_avg_sq"):
                value = state[name]
                if value.shape != parameter.shape or value.dtype != torch.float32 or value.device != device:
                    raise ValueError(f"Invalid Adam {name} checkpoint tensor")
            step_tensor = state["step"]
            if not isinstance(step_tensor, torch.Tensor) or step_tensor.numel() != 1:
                raise ValueError("Adam step must be a scalar tensor")
            step = float(step_tensor.item())
            if not math.isfinite(step) or step < 0 or step != int(step) or step + updates > 2**24:
                raise ValueError("Adam steps must be exact nonnegative float32 integers")
            states.append(state)
            original_states.append(original)
            steps.append(int(step))
        flat = torch.cat([p.flatten() for p in parameters])
        moments = torch.cat([s["exp_avg"].flatten() for s in states])
        variances = torch.cat([s["exp_avg_sq"].flatten() for s in states])
        if not bool(
            torch.isfinite(flat).all()
            & torch.isfinite(moments).all()
            & torch.isfinite(variances).all()
            & (variances >= 0).all()
        ):
            raise ValueError("Checkpoint must be finite with nonnegative Adam second moments")
        corrections = torch.tensor(
            [
                (-lr, step + u + 1)
                if fused
                else (-lr / (1.0 - beta1 ** (step + u + 1)), math.sqrt(1.0 - beta2 ** (step + u + 1)))
                for u in range(updates)
                for step in steps
            ],
            dtype=torch.float32,
            device=device,
        )
        tokens = tuple(
            tuple(_tensor_token(x) for x in (p, s["exp_avg"], s["exp_avg_sq"], s["step"]))
            for p, s in zip(parameters, states, strict=True)
        )
        return cls(
            tuple(parameters),
            optimizer,
            tuple(states),
            tuple(original_states),
            tuple(steps),
            updates,
            flat,
            moments,
            variances,
            corrections,
            tokens,
        )

    def validate(self):
        for parameter, state, original, tokens in zip(
            self.parameters, self.states, self.original_states, self.tokens, strict=True
        ):
            if self.optimizer.state.get(parameter) is not original or (original is not state and original):
                raise RuntimeError("Optimizer state was replaced after preparation")
            current = tuple(_tensor_token(x) for x in (parameter, state["exp_avg"], state["exp_avg_sq"], state["step"]))
            if current != tokens:
                raise RuntimeError("Native checkpoint changed after preparation")

    @torch.no_grad()
    def copyback(self):
        offset = 0
        for parameter, state, step in zip(self.parameters, self.states, self.steps, strict=True):
            end = offset + parameter.numel()
            self.optimizer.state[parameter] = state
            parameter.copy_(self.flat[offset:end].view_as(parameter))
            state["exp_avg"].copy_(self.moments[offset:end].view_as(parameter))
            state["exp_avg_sq"].copy_(self.variances[offset:end].view_as(parameter))
            state["step"].fill_(step + self.updates)
            parameter.grad = None
            offset = end


@dataclass
class PreparedUpdate:
    """One-use prepared launch, retaining all buffers through native copyback."""

    state: PackedAdamState
    arguments: tuple
    result: torch.Tensor
    batch: int
    indices: torch.Tensor
    minibatch_size: int
    epochs: int
    minibatches: int
    guard: UpdateGuard | None = field(default=None, repr=False)
    launched: bool = False
    consumed: bool = False

    def validate(self):
        if self.consumed:
            raise RuntimeError("Prepared update has already been consumed")
        if self.guard is None:
            raise RuntimeError("Prepared update has no scratch reservation")
        self.guard.check_stream()
        self.state.validate()

    def mark_launched(self):
        self.validate()
        if self.launched:
            raise RuntimeError("Prepared update was already launched")
        self.launched = True

    def commit(self):
        self.validate()
        if not self.launched:
            raise RuntimeError("Cannot commit an update before a successful launch")
        try:
            self.state.copyback()
        finally:
            self.cancel()

    def cancel(self):
        if not self.consumed:
            self.consumed = True
            if self.guard is not None:
                self.guard.lock.release()

    def launch(self, kernel, grid, block, shared_bytes):
        try:
            self.validate()
            kernel.launch(grid, block, args=self.arguments, smem=shared_bytes, cooperative=True)
            self.mark_launched()
            self.commit()
        except BaseException:
            self.cancel()
            raise
        return self.result

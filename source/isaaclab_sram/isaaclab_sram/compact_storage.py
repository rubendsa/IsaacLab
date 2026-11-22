# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compact history storage at the native feed-forward RSL-RL rollout boundary.

Only repeated state histories are packed. Raw camera images and PPO targets
retain their original representations. Gathering returns ordinary TensorDict
minibatches; whole-group reconstruction preserves the native normalization
reduction order. All allocations in this adapter are GPU global memory.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from .compact_history import CompactHistory, HistoryLayout


class CompactObservations:
    """Implement the observation operations used by native feed-forward PPO.

    This is a narrow storage adapter, not a general TensorDict implementation.
    Native ``add_transition`` writes through ``[step].copy_(snapshot)``. Native
    minibatch generation and normalization read through ``flatten(0, 1)``.
    Recurrent and distillation storage consumers are unsupported.
    """

    def __init__(self, observations: TensorDict, layouts: dict[str, HistoryLayout]):
        if len(observations.batch_size) != 2 or not layouts or not set(layouts) <= set(observations.keys()):
            raise ValueError("Compact observations require [time, environment] storage and known history groups")
        self.num_steps, self.num_envs = observations.batch_size
        self.batch_size = observations.batch_size
        self.device = next(iter(observations.values())).device
        self.groups = tuple(observations.keys())
        self.dense_bytes = sum(value.numel() * value.element_size() for value in observations.values())
        self.histories = {}
        self.raw = {}
        for name, value in observations.items():
            if name in layouts:
                layout = layouts[name]
                if value.shape != (self.num_steps, self.num_envs, layout.observation_size):
                    raise ValueError(f"Resolved history layout does not match rollout observation group {name}")
                self.histories[name] = CompactHistory(
                    layout, self.num_steps, self.num_envs, dtype=value.dtype, device=value.device
                )
            else:
                # Retain the original image allocation; do not duplicate it.
                self.raw[name] = value
        self.step = 0

    def __getitem__(self, step: int):
        if not isinstance(step, int) or not 0 <= step < self.num_steps:
            raise IndexError("Compact observation writes require a valid integer rollout step")
        return _ObservationWrite(self, step)

    @torch.no_grad()
    def _append(self, step: int, observations: TensorDict) -> None:
        if step == 0:
            for history in self.histories.values():
                history.clear()
            self.step = 0
        if step != self.step:
            raise ValueError("Compact observations must be recorded in rollout order")
        for name, history in self.histories.items():
            history.append(observations[name])
        for name, values in self.raw.items():
            values[step].copy_(observations[name])
        self.step += 1

    def flatten(self, start_dim: int, end_dim: int):
        """Read a completed rollout in native flattened time/environment order."""
        if (start_dim, end_dim) != (0, 1) or self.step != self.num_steps:
            raise ValueError("Compact observations can only flatten a completed rollout over dimensions 0 and 1")
        for history in self.histories.values():
            history.validate()
        return _FlatObservations(self)

    def storage_tensors(self) -> Iterator[torch.Tensor]:
        """Iterate physical allocations without reconstructing dense histories."""
        yield from self.raw.values()
        for history in self.histories.values():
            yield from history.storage_tensors()

    @property
    def storage_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.storage_tensors())


class _ObservationWrite:
    def __init__(self, storage: CompactObservations, step: int):
        self.storage, self.step = storage, step

    def copy_(self, observations: TensorDict) -> None:
        self.storage._append(self.step, observations)


class _FlatObservations:
    def __init__(self, storage: CompactObservations):
        self.storage = storage
        self.batch_size = torch.Size([storage.num_steps * storage.num_envs])

    def __getitem__(self, key: str | torch.Tensor):
        if isinstance(key, str):
            if key in self.storage.histories:
                return self.storage.histories[key].reconstruct().flatten(0, 1)
            return self.storage.raw[key].flatten(0, 1)
        if not isinstance(key, torch.Tensor) or key.ndim != 1 or key.dtype != torch.int64:
            raise TypeError("Compact observation minibatches require one-dimensional int64 tensor indices")
        observations = {
            name: (
                self.storage.histories[name].gather(key)
                if name in self.storage.histories
                else self.storage.raw[name].flatten(0, 1)[key]
            )
            for name in self.storage.groups
        }
        return TensorDict(observations, batch_size=[key.numel()], device=self.storage.device)


def compact_rollout_observations(storage: RolloutStorage, layouts: dict[str, HistoryLayout]) -> CompactObservations:
    """Replace empty native observation storage while retaining all PPO fields.

    Call after constructing the runner and before collecting its first rollout.
    Resumed checkpoints are supported because native RSL-RL does not checkpoint
    partially collected rollout storage.
    """
    if storage.training_type != "rl" or storage.step != 0 or not isinstance(storage.observations, TensorDict):
        raise ValueError("Install compact observations on empty native feed-forward RL rollout storage")
    observations = CompactObservations(storage.observations, layouts)
    storage.observations = observations
    return observations

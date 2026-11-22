# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lossless storage of flattened ObservationManager state histories.

Store each processed frame once, retaining the initial history prefix and the
reset boundaries needed to reproduce native first-frame backfill. No noise,
delay, clipping, scaling or observation function is evaluated by this codec.
Images belong in their original rollout storage; these buffers remain global
allocations and do not by themselves establish SRAM residency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from isaaclab.managers import ObservationManager


@dataclass(frozen=True)
class HistoryTerm:
    """One flattened term, with oldest-to-newest native history ordering."""

    name: str
    frame_size: int
    history_length: int

    def __post_init__(self):
        if not self.name or self.frame_size < 1 or self.history_length < 1:
            raise ValueError("History terms require a name and positive frame size/history length")


@dataclass(frozen=True)
class HistoryLayout:
    """Resolved flat group layout, including the manager's concatenation order."""

    terms: tuple[HistoryTerm, ...]
    order: Literal["term", "time"] = "term"

    def __post_init__(self):
        if not self.terms or len({term.name for term in self.terms}) != len(self.terms):
            raise ValueError("A history layout requires nonempty, uniquely named terms")
        if self.order not in ("term", "time"):
            raise ValueError("History order must be 'term' or 'time'")
        if self.order == "time" and len({term.history_length for term in self.terms}) != 1:
            raise ValueError("Time-major groups require a common history length")

    @property
    def observation_size(self) -> int:
        """Number of scalars in one native flattened group observation."""
        return sum(term.frame_size * term.history_length for term in self.terms)

    def _split(self, observation: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.order == "time":
            history = observation.reshape(observation.shape[0], self.terms[0].history_length, -1)
            return history.split([term.frame_size for term in self.terms], dim=-1)
        pieces = observation.split([term.frame_size * term.history_length for term in self.terms], dim=-1)
        return tuple(
            piece.reshape(observation.shape[0], term.history_length, term.frame_size)
            for piece, term in zip(pieces, self.terms, strict=True)
        )

    def _join(self, histories: list[torch.Tensor]) -> torch.Tensor:
        if self.order == "time":
            return torch.cat(histories, dim=-1).flatten(1)
        return torch.cat([history.flatten(1) for history in histories], dim=-1)


def layouts_from_observation_manager(
    manager: ObservationManager, groups: tuple[str, ...] = ("policy", "proprio", "perception")
) -> dict[str, HistoryLayout]:
    """Discover layouts from the live manager's resolved terms and dimensions.

    The manager's effective term configs and time-major flag are authoritative:
    group history settings override term settings during manager construction.
    Only concatenated flat groups are supported. Higher-dimensional source terms
    are supported when native history flattening produces a flat group.
    """
    layouts = {}
    for group in groups:
        if not manager.group_obs_concatenate[group] or len(manager.group_obs_dim[group]) != 1:
            raise ValueError(f"Compact history requires a concatenated flat observation group: {group}")
        if manager._group_obs_concatenate_dim[group] not in (-1, 1):
            raise ValueError(f"Unsupported concatenation dimension for flat group: {group}")
        terms = []
        for name, shape, cfg in zip(
            manager.active_terms[group],
            manager.group_obs_term_dim[group],
            manager._group_obs_term_cfgs[group],
            strict=True,
        ):
            length = max(1, cfg.history_length)
            if len(shape) != 1 or shape[0] % length or (cfg.history_length > 0 and not cfg.flatten_history_dim):
                raise ValueError(f"Unsupported unflattened history for observation term: {group}.{name}")
            terms.append(HistoryTerm(name, shape[0] // length, length))
        order = "time" if manager._group_obs_time_major[group] else "term"
        layouts[group] = HistoryLayout(tuple(terms), order)
    return layouts


def _equal_bits_per_environment(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Compare stored representations, preserving signed zeros and NaN payloads."""
    size = left.shape[0]
    return (
        left.contiguous().view(torch.uint8).reshape(size, -1) == right.contiguous().view(torch.uint8).reshape(size, -1)
    ).all(dim=1)


class CompactHistory:
    """Streaming lossless rollout buffer for one flattened observation group.

    ``append`` accepts the actual observation snapshot supplied to the policy.
    Its optional reset mask describes resets **before that snapshot**, not the
    done flag returned after taking the associated action. Without a mask,
    resets are inferred only when the snapshot is exact native reset backfill.
    Any other history evolution invalidates the rollout. Validation accumulates
    on the tensor device; ``validate``/first ``gather`` synchronize once per
    appended rollout, rather than once per transition.
    """

    def __init__(
        self,
        layout: HistoryLayout,
        num_steps: int,
        num_envs: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ):
        if num_steps < 1 or num_envs < 1:
            raise ValueError("Compact history requires positive rollout and environment counts")
        self.layout = layout
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dtype = dtype
        self.frames = tuple(
            torch.empty(num_steps + term.history_length - 1, num_envs, term.frame_size, dtype=dtype, device=device)
            for term in layout.terms
        )
        self.reset_starts = torch.empty(num_steps, num_envs, dtype=torch.int32, device=device)
        self._valid = torch.ones(num_envs, dtype=torch.bool, device=device)
        self._env_indices = torch.arange(num_envs, device=device)
        self._lags = tuple(torch.arange(1 - term.history_length, 1, device=device) for term in layout.terms)
        self.step = 0
        self._validated_step = -1

    @property
    def frame_bytes(self) -> int:
        """Allocated processed frames, excluding reset and validation metadata."""
        return sum(frame.numel() * frame.element_size() for frame in self.frames)

    @property
    def storage_bytes(self) -> int:
        """All owned tensor allocations, including indexing and validation metadata."""
        return sum(tensor.numel() * tensor.element_size() for tensor in self.storage_tensors())

    def storage_tensors(self) -> tuple[torch.Tensor, ...]:
        """Return owned buffers for allocation accounting; callers must not mutate them."""
        return (*self.frames, self.reset_starts, self._valid, self._env_indices, *self._lags)

    @torch.no_grad()
    def append(self, observation: torch.Tensor, *, reset: torch.Tensor | None = None) -> None:
        """Record one native snapshot without retaining or mutating the input."""
        if self.step >= self.num_steps:
            raise RuntimeError("Compact history rollout is full; clear it before appending")
        if observation.shape != (self.num_envs, self.layout.observation_size):
            raise ValueError("Observation shape does not match the resolved history layout")
        if observation.dtype != self.dtype or observation.device != self.frames[0].device:
            raise ValueError("Observation dtype/device must match compact history storage")
        if reset is not None and (
            reset.shape != (self.num_envs,) or reset.dtype != torch.bool or reset.device != observation.device
        ):
            raise ValueError("Reset must be a boolean environment vector on the observation device")
        histories = self.layout._split(observation)
        if self.step == 0:
            # Prefix includes exact previous-rollout state, including any reset
            # before collection began. No cross-rollout state is assumed.
            for frames, history, term in zip(self.frames, histories, self.layout.terms, strict=True):
                frames[: term.history_length].copy_(history.transpose(0, 1))
            self.reset_starts[0].fill_(1 - max(term.history_length for term in self.layout.terms))
            if reset is not None:
                repeated = torch.ones_like(self._valid)
                for history in histories:
                    repeated &= _equal_bits_per_environment(history, history[:, -1:].expand_as(history))
                self._valid &= ~reset | repeated
        else:
            starts = self.reset_starts[self.step - 1]
            shifted = torch.ones_like(self._valid)
            repeated = torch.ones_like(self._valid)
            times = torch.full_like(self._env_indices, self.step)
            for i, (frames, history, term) in enumerate(zip(self.frames, histories, self.layout.terms, strict=True)):
                frames[self.step + term.history_length - 1].copy_(history[:, -1])
                candidate = self._term_history(i, times, self._env_indices, starts)
                shifted &= _equal_bits_per_environment(history, candidate)
                repeated &= _equal_bits_per_environment(history, history[:, -1:].expand_as(history))
            if reset is None:
                reset = ~shifted & repeated
                self._valid &= shifted | repeated
            else:
                self._valid &= torch.where(reset, repeated, shifted)
            self.reset_starts[self.step].copy_(torch.where(reset, self.step, starts))
        self.step += 1

    def validate(self) -> None:
        """Reject snapshots that cannot be represented by native shift/reset history."""
        if self.step == 0:
            raise RuntimeError("Compact history is empty")
        if self._validated_step != self.step:
            if not bool(self._valid.all()):
                raise ValueError(
                    "History is neither a one-sample shift nor native reset backfill; cannot pack losslessly"
                )
            self._validated_step = self.step

    def _term_history(self, term_index, times, environments, starts):
        term = self.layout.terms[term_index]
        indices = torch.maximum(times[:, None] + self._lags[term_index], starts[:, None])
        indices = indices + term.history_length - 1
        return self.frames[term_index][indices, environments[:, None]]

    def gather(self, indices: torch.Tensor) -> torch.Tensor:
        """Reconstruct arbitrary flattened time/environment indices, preserving order.

        ``indices`` is a one-dimensional int64 tensor on the buffer device, using
        ``time * num_envs + environment`` as in native RSL-RL flattened storage.
        Duplicates and empty minibatches are supported. Invalid indices raise the
        underlying tensor indexing error instead of silently wrapping.
        """
        self.validate()
        if indices.ndim != 1 or indices.dtype != torch.int64 or indices.device != self.frames[0].device:
            raise ValueError("Gather indices must be a one-dimensional int64 tensor on the storage device")
        starts = self.reset_starts[: self.step].reshape(-1).index_select(0, indices)
        times, environments = indices // self.num_envs, indices % self.num_envs
        histories = [self._term_history(i, times, environments, starts) for i in range(len(self.frames))]
        return self.layout._join(histories)

    def reconstruct(self) -> torch.Tensor:
        """Return all appended native snapshots in ``[time, environment, feature]`` order."""
        indices = torch.arange(self.step * self.num_envs, device=self.frames[0].device)
        return self.gather(indices).reshape(self.step, self.num_envs, self.layout.observation_size)

    def clear(self) -> None:
        """Begin a new rollout; its first snapshot supplies a fresh exact prefix."""
        self.step = 0
        self._validated_step = -1
        self._valid.fill_(True)


def pack_history(
    observations: torch.Tensor, layout: HistoryLayout, *, resets: torch.Tensor | None = None
) -> CompactHistory:
    """Pack existing ``[time, environment, feature]`` native snapshots losslessly."""
    if observations.ndim != 3:
        raise ValueError("History packing requires [time, environment, feature] observations")
    if resets is not None and resets.shape != observations.shape[:2]:
        raise ValueError("Reset masks must have [time, environment] shape")
    packed = CompactHistory(
        layout, observations.shape[0], observations.shape[1], dtype=observations.dtype, device=observations.device
    )
    for step, observation in enumerate(observations):
        packed.append(observation, reset=None if resets is None else resets[step])
    packed.validate()
    return packed

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bitwise determinism of the SRAM-resident PPO megakernel.

The trainer carries all training state between launches in a few device tensors. With a fixed seed, the state after
``N`` PPO iterations must not depend on how the iterations are split across kernel launches or on where the rollout
buffer lives, and the kernel's random streams must follow the seed.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

try:
    import isaaclab_sram  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name != "isaaclab_sram":
        raise
    # The package is not part of the uv-locked workspace, so import it from this checkout when it is not installed.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("cuda.bindings", reason="the megakernel runtime needs the CUDA driver and NVRTC bindings")
pytest.importorskip("rsl_rl", reason="the trainer initializes its networks with RSL-RL")

from isaaclab_sram import megakernel

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required"),
]

NUM_ENVS = 4096
NUM_ITERATIONS = 8
SEED = 5

STATE_TENSORS = ("params", "adam_m", "adam_v", "scal", "env_state", "ep_len", "ep_ret")
"""Trainer tensors that carry the training state from one launch to the next."""

ATOMIC_STATS = ("episode_return_sum", "step_reward_sum")
"""Logging-only statistics the kernel sums with float ``atomicAdd``, so their rounding depends on the arrival order."""

SPLITS = {
    "one_iteration_per_launch": [(NUM_ITERATIONS, 1)],
    "two_train_calls": [(3, None), (5, 2)],
    "three_train_calls": [(1, None), (2, 1), (5, 4)],
}
"""``(num_iterations, iters_per_launch)`` per :meth:`SramPPOCartpole.train` call, ``NUM_ITERATIONS`` in total."""


@dataclass
class Reference:
    """Trainer that ran ``NUM_ITERATIONS`` iterations in one :meth:`SramPPOCartpole.train` call."""

    trainer: megakernel.SramPPOCartpole
    initial_state: dict[str, torch.Tensor]
    launches: int


def _snapshot(trainer: megakernel.SramPPOCartpole) -> dict[str, torch.Tensor]:
    return {name: getattr(trainer, name).clone() for name in STATE_TENSORS}


def _train(seed: int, schedule: list[tuple[int, int | None]], **kwargs) -> tuple[megakernel.SramPPOCartpole, int]:
    """Train a new trainer following ``schedule``; return it with the number of kernel launches."""
    trainer = megakernel.SramPPOCartpole(num_envs=NUM_ENVS, seed=seed, **kwargs)
    launches = sum(trainer.train(n, iters_per_launch=per)["launches"] for n, per in schedule)
    return trainer, launches


def _assert_same_state(actual: megakernel.SramPPOCartpole, expected: megakernel.SramPPOCartpole):
    assert actual.iteration == expected.iteration
    differing = [name for name in STATE_TENSORS if not torch.equal(getattr(actual, name), getattr(expected, name))]
    assert not differing, f"training state differs bitwise: {differing}"


def _assert_same_stats(actual: megakernel.SramPPOCartpole, expected: megakernel.SramPPOCartpole):
    exact = [k for k, name in enumerate(megakernel.STAT_NAMES) if name not in ATOMIC_STATS]
    atomic = [megakernel.STAT_NAMES.index(name) for name in ATOMIC_STATS]
    differing = [megakernel.STAT_NAMES[k] for k in exact if not torch.equal(actual.stats[:, k], expected.stats[:, k])]
    assert not differing, f"per-iteration statistics differ bitwise: {differing}"
    torch.testing.assert_close(actual.stats[:, atomic], expected.stats[:, atomic], rtol=1e-5, atol=1e-2)


@pytest.fixture(scope="module")
def reference() -> Reference:
    trainer = megakernel.SramPPOCartpole(num_envs=NUM_ENVS, seed=SEED)
    initial_state = _snapshot(trainer)
    launches = trainer.train(NUM_ITERATIONS)["launches"]
    return Reference(trainer, initial_state, launches)


def test_reference_trains_in_one_launch_and_updates_the_state(reference):
    """Guards the comparisons below against a kernel that leaves its state untouched."""
    assert reference.launches == 1
    assert reference.trainer.iteration == NUM_ITERATIONS
    unchanged = [
        name for name, value in reference.initial_state.items() if torch.equal(getattr(reference.trainer, name), value)
    ]
    assert not unchanged, f"{NUM_ITERATIONS} iterations did not change {unchanged}"


@pytest.mark.parametrize("schedule", SPLITS.values(), ids=SPLITS.keys())
def test_split_launches_match_a_single_launch_bitwise(reference, schedule):
    """One launch of ``N`` iterations, ``N`` launches of one iteration and uneven splits give identical states."""
    trainer, launches = _train(SEED, schedule)

    assert launches == sum(math.ceil(n / (per or n)) for n, per in schedule)
    assert launches > 1
    _assert_same_state(trainer, reference.trainer)
    _assert_same_stats(trainer, reference.trainer)


def test_rollout_storage_location_does_not_change_the_result(reference):
    """Keeping the rollout buffer in global memory instead of shared memory gives an identical state."""
    if not reference.trainer.plan.store_in_smem:
        pytest.skip(f"the rollout buffer of {NUM_ENVS} environments does not fit in shared memory on this GPU")
    trainer, _ = _train(SEED, [(NUM_ITERATIONS, None)], storage="global")

    assert not trainer.plan.store_in_smem
    _assert_same_state(trainer, reference.trainer)


def test_different_seeds_give_different_results(reference):
    trainer, _ = _train(SEED + 1, [(NUM_ITERATIONS, None)])

    # ``scal`` holds the Adam step counters and the learning rate, which can legitimately match (it is clamped)
    seeded = [name for name in STATE_TENSORS if name != "scal"]
    identical = [name for name in seeded if torch.equal(getattr(trainer, name), getattr(reference.trainer, name))]
    assert not identical, f"seeds {SEED} and {SEED + 1} produced identical {identical}"


@pytest.mark.parametrize(("launch_seed", "same_result"), [(SEED, True), (SEED + 1, False)])
def test_kernel_random_streams_follow_the_seed(reference, launch_seed, same_result):
    """From an identical initial state, only the seed passed to the kernel decides the outcome.

    The trainer is initialized with another seed and then given the reference's initial state, so the copied state is
    complete exactly when the reference seed reproduces the reference bitwise.
    """
    trainer = megakernel.SramPPOCartpole(num_envs=NUM_ENVS, seed=SEED + 2)
    for name, value in reference.initial_state.items():
        getattr(trainer, name).copy_(value)
    trainer.seed = launch_seed
    trainer.train(NUM_ITERATIONS)

    if same_result:
        _assert_same_state(trainer, reference.trainer)
    else:
        assert not torch.equal(trainer.params, reference.trainer.params)
        assert not torch.equal(trainer.env_state, reference.trainer.env_state)

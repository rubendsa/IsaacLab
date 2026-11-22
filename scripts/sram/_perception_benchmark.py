# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native rollout and provenance helpers for the complete PPO benchmark."""

from __future__ import annotations

import copy
import hashlib
import inspect
from importlib.metadata import version
from pathlib import Path

import isaaclab_sram.persistent_unified as unified_learner
import rsl_rl
import torch
from isaaclab_sram.compact_history import HistoryLayout, HistoryTerm
from isaaclab_sram.compact_storage import CompactObservations
from rsl_rl.utils import resolve_class
from tensordict import TensorDict

from isaaclab.utils import to_dict
from isaaclab.utils.buffers import CircularBuffer

from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.core.lift.config.kuka_allegro.agents.rsl_rl_ppo_cfg import KukaAllegroSingleCameraPPORunnerCfg


def native_configuration():
    """Resolve the branch's actual camera PPO configuration without fixture overrides."""
    return to_dict(handle_deprecated_rsl_rl_cfg(KukaAllegroSingleCameraPPORunnerCfg(), version("rsl-rl-lib")))


def history_stream(num_envs, device, shift):
    widths = {"policy": 34, "proprio": 93, "perception": 192}
    buffers = {name: CircularBuffer(5, num_envs, device) for name in widths}

    def advance(reset=None):
        groups = {}
        for name, width in widths.items():
            if reset is not None:
                buffers[name].reset(reset)
            # Values entering the histories are already processed observations.
            buffers[name].append(torch.randn(num_envs, width, device=device) * (0.5 + shift) + shift)
            groups[name] = buffers[name].buffer.flatten(1).clone()
        groups["base_image"] = torch.randint(0, 256, (num_envs, 3, 64, 64), dtype=torch.uint8, device=device)
        return TensorDict(groups, batch_size=[num_envs], device=device)

    return advance


@torch.inference_mode()
def collect(algorithm, num_envs, device, shift):
    advance = history_stream(num_envs, device, shift)
    observations = advance()
    environment_ids = torch.arange(num_envs, device=device)
    for step in range(32):
        algorithm.act(observations)
        dones = (environment_ids + step) % 11 == 0
        observations = advance(environment_ids[dones])
        algorithm.process_env_step(
            observations,
            torch.randn(num_envs, device=device) * 0.2,
            dones.long(),
            {"time_outs": dones & (environment_ids % 2 == 0)},
        )
    algorithm.compute_returns(observations)
    return observations


def clone_rollout(reference, candidate, compact):
    candidate.storage = copy.deepcopy(reference.storage)
    if compact:
        dense = candidate.storage.observations
        layouts = {
            name: HistoryLayout((HistoryTerm("processed_frame", width, 5),))
            for name, width in (("policy", 34), ("proprio", 93), ("perception", 192))
        }
        packed = CompactObservations(dense, layouts)
        for step in range(candidate.storage.num_transitions_per_env):
            packed[step].copy_(dense[step])
        for name in layouts:
            torch.testing.assert_close(packed.flatten(0, 1)[name], dense[name].flatten(0, 1), rtol=0, atol=0)
        candidate.storage.observations = packed


def source_hashes():
    """Hash loaded native/custom sources, including the native visual model.

    Library versions separately identify binary dependencies. The map records
    loaded file paths so an unintended editable checkout is visible in evidence.
    """
    package = Path(unified_learner.__file__).resolve().parent
    native_package = Path(rsl_rl.__file__).resolve().parent
    actor_class, _ = resolve_class(copy.deepcopy(native_configuration()["actor"]))
    objects = (
        KukaAllegroSingleCameraPPORunnerCfg,
        actor_class,
        CircularBuffer,
        to_dict,
        handle_deprecated_rsl_rl_cfg,
        torch.optim.Adam,
        torch.distributions.Normal,
    )
    sources = {
        Path(__file__).resolve(),
        *[Path(inspect.getfile(obj)).resolve() for obj in objects],
        *package.rglob("*.py"),
        *package.rglob("*.cu"),
        *package.rglob("*.cuh"),
        *native_package.rglob("*.py"),
    }
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(sources)}


def versions():
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "rsl_rl": version("rsl-rl-lib"),
        "tensordict": version("tensordict"),
    }

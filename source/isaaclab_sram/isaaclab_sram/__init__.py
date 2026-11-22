# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SRAM-oriented reinforcement learning experiments for Isaac Lab.

The :class:`~isaaclab_sram.megakernel.SramPPOCartpole` trainer runs the complete RSL-RL PPO loop for
``Isaac-Cartpole-Direct`` (physics, policy, rollout buffer, GAE, mini-batch updates and Adam) inside one
persistent CUDA kernel. Small working sets fit in registers and shared memory; cross-block reduction
and checkpoint buffers use global memory. Perception helpers fuse selected operations in the native
Kuka-Allegro camera learner while retaining global image, rollout, parameter and optimizer storage.
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("isaaclab_sram")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from .cartpole import CartpolePhysics, CartpoleTask, PPOConfig, isaaclab_cartpole_configs

__all__ = ["CartpolePhysics", "CartpoleTask", "PPOConfig", "isaaclab_cartpole_configs"]

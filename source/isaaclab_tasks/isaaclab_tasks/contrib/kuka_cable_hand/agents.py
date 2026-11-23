# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO settings: the state-based Kuka-Allegro lift recipe (networks and PPO parameters)."""

from __future__ import annotations

import copy

PPO_RUNNER_CFG: dict = {
    "seed": 42,
    "num_steps_per_env": 24,
    "max_iterations": 1500,
    "save_interval": 50,
    "experiment_name": "kuka_cable_hand_lift",
    "logger": "tensorboard",
    "check_for_nan": True,
    "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
    "actor": {
        "class_name": "MLPModel",
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    },
    "critic": {
        "class_name": "MLPModel",
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": None,
    },
    "algorithm": {
        "class_name": "PPO",
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 1.0e-3,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "entropy_coef": 0.005,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
        "optimizer": "adam",
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "normalize_advantage_per_mini_batch": False,
        "rnd_cfg": None,
        "symmetry_cfg": None,
    },
}
"""Runner configuration in the dictionary form of ``rsl_rl.runners.OnPolicyRunner`` (rsl-rl-lib 5.x)."""


def ppo_runner_cfg(**overrides) -> dict:
    """A copy of :data:`PPO_RUNNER_CFG` with top-level keys replaced."""
    cfg = copy.deepcopy(PPO_RUNNER_CFG)
    cfg.update(overrides)
    return cfg

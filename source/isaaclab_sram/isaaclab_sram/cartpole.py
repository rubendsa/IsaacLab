# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac-Cartpole-Direct task and RSL-RL PPO specification for the SRAM-resident megakernel.

Everything the kernel needs is read from Isaac Lab's own configuration objects
(:class:`isaaclab_tasks.core.cartpole.cartpole_direct_env_cfg.CartpoleEnvCfg` and
:class:`isaaclab_tasks.core.cartpole.agents.rsl_rl_ppo_cfg.CartpoleDirectPPORunnerCfg`).  The
rigid-body constants are the values the Newton / MuJoCo-Warp model builds from the cartpole USD
asset; :func:`physics_from_env` re-extracts them from a live Isaac Lab environment so the constants
below can be re-verified on any checkout (``scripts/sram/verify_megakernel.py`` checks every transition against a
float64 reference of the same equations).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class CartpolePhysics:
    """Rigid-body constants of the Isaac Lab cartpole as integrated by Newton (MuJoCo-Warp)."""

    cart_mass: float = 1.0
    """Cart mass [kg] (``mjw_model.body_mass``)."""
    pole_mass: float = 1.0
    """Pole mass [kg]."""
    pole_com: float = 0.4699999988079071
    """Distance hinge -> pole centre of mass along the pole [m] (``body_ipos``)."""
    pole_inertia: float = 0.08363333344459534
    """Pole inertia about its COM around the hinge axis [kg m^2] (``body_inertia``)."""
    gravity: float = 9.8100004196167
    """|g| [m/s^2]."""
    cart_damping: float = 10.0
    """Isaac ``ImplicitActuatorCfg(damping=10)`` on ``slider_to_cart`` -> MuJoCo velocity servo gain kv."""
    actuator_force_limit: float = 400.0
    """``jnt_actfrcrange`` of the cart joint; clamps the servo force only (feed-forward is unclamped)."""
    cart_limit: float = 4.0
    """Cart joint position limit [m] (only reachable without resets)."""
    cart_vel_limit: float = 100.0
    """Soft joint velocity limit of the cart [m/s] (reset clamp)."""
    pole_vel_limit: float = 8.0
    """Soft joint velocity limit of the pole [rad/s] (reset clamp)."""


@dataclass
class CartpoleTask:
    """MDP constants of Isaac-Cartpole-Direct."""

    physics_dt: float = 1.0 / 120.0
    decimation: int = 2
    episode_length_s: float = 5.0
    action_scale: float = 100.0
    max_cart_pos: float = 3.0
    initial_cart_position_range: tuple[float, float] = (-1.0, 1.0)
    initial_cart_velocity_range: tuple[float, float] = (-0.5, 0.5)
    initial_pole_angle_range: tuple[float, float] = (-0.25 * math.pi, 0.25 * math.pi)
    initial_pole_velocity_range: tuple[float, float] = (-0.25 * math.pi, 0.25 * math.pi)
    rew_scale_alive: float = 1.0
    rew_scale_terminated: float = -2.0
    rew_scale_pole_pos: float = -1.0
    rew_scale_cart_vel: float = -0.01
    rew_scale_pole_vel: float = -0.005
    physics: CartpolePhysics = field(default_factory=CartpolePhysics)

    @property
    def step_dt(self) -> float:
        return self.physics_dt * self.decimation

    @property
    def max_episode_length(self) -> int:
        return math.ceil(self.episode_length_s / self.step_dt)

    @classmethod
    def from_env_cfg(cls, env_cfg, physics: CartpolePhysics | None = None) -> CartpoleTask:
        """Build from an Isaac Lab ``CartpoleEnvCfg`` (direct workflow)."""
        return cls(
            physics_dt=float(env_cfg.sim.dt),
            decimation=int(env_cfg.decimation),
            episode_length_s=float(env_cfg.episode_length_s),
            action_scale=float(env_cfg.action_scale),
            max_cart_pos=float(env_cfg.max_cart_pos),
            initial_cart_position_range=tuple(env_cfg.initial_cart_position_range),
            initial_cart_velocity_range=tuple(env_cfg.initial_cart_velocity_range),
            initial_pole_angle_range=tuple(env_cfg.initial_pole_angle_range),
            initial_pole_velocity_range=tuple(env_cfg.initial_pole_velocity_range),
            rew_scale_alive=float(env_cfg.rew_scale_alive),
            rew_scale_terminated=float(env_cfg.rew_scale_terminated),
            rew_scale_pole_pos=float(env_cfg.rew_scale_pole_pos),
            rew_scale_cart_vel=float(env_cfg.rew_scale_cart_vel),
            rew_scale_pole_vel=float(env_cfg.rew_scale_pole_vel),
            physics=physics or CartpolePhysics(),
        )


@dataclass
class PPOConfig:
    """The subset of RSL-RL's PPO / runner configuration the megakernel implements."""

    num_steps_per_env: int = 16
    max_iterations: int = 150
    hidden_dims: tuple[int, int] = (32, 32)
    init_noise_std: float = 1.0
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.005
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    init_at_random_ep_len: bool = True

    @classmethod
    def from_rsl_rl_cfg(cls, agent_cfg) -> PPOConfig:
        """Build from an Isaac Lab ``RslRlOnPolicyRunnerCfg`` and validate that it is supported."""
        alg = agent_cfg.algorithm
        actor, critic = agent_cfg.actor, agent_cfg.critic
        if list(actor.hidden_dims) != [32, 32] or list(critic.hidden_dims) != [32, 32]:
            raise ValueError("the prototype megakernel is specialised for [32, 32] actor/critic MLPs")
        if actor.activation != "elu" or critic.activation != "elu":
            raise ValueError("the prototype megakernel implements ELU activations only")
        if getattr(actor, "obs_normalization", False) or getattr(critic, "obs_normalization", False):
            raise ValueError("observation normalisation is not implemented in the prototype megakernel")
        if not alg.use_clipped_value_loss:
            raise ValueError("the prototype megakernel implements the clipped value loss only")
        if getattr(alg, "normalize_advantage_per_mini_batch", False):
            raise ValueError("per-mini-batch advantage normalisation is not implemented")
        if getattr(agent_cfg, "clip_actions", None) is not None:
            raise ValueError(
                "action clipping is not implemented in the prototype megakernel (clip_actions must be None)"
            )
        if str(getattr(alg, "optimizer", "adam")).lower() != "adam":
            raise ValueError("the prototype megakernel implements the Adam optimizer only")
        if getattr(alg, "rnd_cfg", None) is not None or getattr(alg, "symmetry_cfg", None) is not None:
            raise ValueError("RND and symmetry augmentation are not implemented in the prototype megakernel")
        if getattr(alg, "use_mixed_precision", False):
            raise ValueError("the prototype megakernel computes in fp32 (use_mixed_precision must be False)")
        dist = getattr(actor, "distribution_cfg", None)
        init_std = float(getattr(dist, "init_std", 1.0)) if dist is not None else 1.0
        std_type = getattr(dist, "std_type", "scalar") if dist is not None else "scalar"
        if std_type != "scalar":
            raise ValueError("only the RSL-RL 'scalar' std parameterisation is implemented")
        return cls(
            num_steps_per_env=int(agent_cfg.num_steps_per_env),
            max_iterations=int(agent_cfg.max_iterations),
            init_noise_std=init_std,
            value_loss_coef=float(alg.value_loss_coef),
            use_clipped_value_loss=bool(alg.use_clipped_value_loss),
            clip_param=float(alg.clip_param),
            entropy_coef=float(alg.entropy_coef),
            num_learning_epochs=int(alg.num_learning_epochs),
            num_mini_batches=int(alg.num_mini_batches),
            learning_rate=float(alg.learning_rate),
            schedule=str(alg.schedule),
            gamma=float(alg.gamma),
            lam=float(alg.lam),
            desired_kl=float(alg.desired_kl) if alg.desired_kl is not None else 0.0,
            max_grad_norm=float(alg.max_grad_norm),
            init_at_random_ep_len=bool(getattr(agent_cfg, "init_at_random_ep_len", True)),
        )


def isaaclab_cartpole_configs(task: str = "Isaac-Cartpole-Direct"):
    """Resolve Isaac Lab's own env + RSL-RL configs for the task (no simulator is started)."""
    import isaaclab_tasks  # noqa: F401  (registers the tasks)
    from isaaclab_tasks.utils import resolve_task_config

    env_cfg, agent_cfg = resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=["physics=newton_mjwarp"])
    return env_cfg, agent_cfg


def physics_from_env(env) -> CartpolePhysics:
    """Re-extract the rigid-body constants from a live Isaac Lab Newton cartpole environment."""
    art = env.unwrapped.scene["cartpole"]
    data = art.data
    names = list(art.body_names)
    mass = data.body_mass.torch[0].cpu().numpy() if hasattr(data.body_mass, "torch") else np.asarray(data.body_mass)[0]
    inertia = (
        data.body_inertia.torch[0].cpu().numpy()
        if hasattr(data.body_inertia, "torch")
        else np.asarray(data.body_inertia)[0]
    )
    com = (
        data.body_com_pose_b.torch[0].cpu().numpy()
        if hasattr(data.body_com_pose_b, "torch")
        else np.asarray(data.body_com_pose_b)[0]
    )
    i_cart, i_pole = names.index("cart"), names.index("pole")
    damping = data.joint_damping.torch[0].cpu().numpy()
    vel_lim = data.soft_joint_vel_limits.torch[0].cpu().numpy()
    pos_lim = data.soft_joint_pos_limits.torch[0].cpu().numpy()
    return CartpolePhysics(
        cart_mass=float(mass[i_cart]),
        pole_mass=float(mass[i_pole]),
        pole_com=float(np.linalg.norm(com[i_pole][:3])),
        pole_inertia=float(np.asarray(inertia[i_pole]).reshape(-1)[0]),
        cart_damping=float(damping[0]),
        cart_limit=float(pos_lim[0][1]),
        cart_vel_limit=float(vel_lim[0]),
        pole_vel_limit=float(vel_lim[1]),
    )


def as_dict(obj) -> dict:
    return asdict(obj)

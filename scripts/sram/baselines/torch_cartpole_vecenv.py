# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac-Cartpole-Direct as a batched PyTorch GPU ``rsl_rl.env.VecEnv`` (closed-form physics, no simulator).

This is the ablation baseline for the SRAM-resident PPO megakernel (``isaaclab_sram``): the *same* MDP as
``Isaac-Cartpole-Direct`` with ``physics=newton_mjwarp``, but with the Newton / MuJoCo-Warp engine and the Isaac
Lab environment machinery (scene, articulation data buffers, actuator models, ``write_data_to_sim``, CUDA-graph
launches, Gymnasium wrappers) replaced by a few dozen elementwise PyTorch ops.  It implements the interface the
stock RSL-RL 5.5.1 ``OnPolicyRunner`` consumes (``get_observations``, ``step``, ``num_envs``, ``num_actions``,
``max_episode_length``, ``episode_length_buf``, ``device``, ``cfg``) with the same return conventions as
:class:`isaaclab_rl.rsl_rl.RslRlVecEnvWrapper`.

Per control step (mirrors ``DirectRLEnv.step`` + ``CartpoleEnv`` + ``RslRlVecEnvWrapper.step``):

1. ``f = action_scale * a`` -- no clipping (``clip_actions=None``), no action noise.
2. ``decimation`` substeps of the closed-form 2-DoF cart-pole integrated like MuJoCo-Warp ``implicitfast``
   (identical equations, constants and float32 operation order as the megakernel's ``cartpole_substep`` in
   ``isaaclab_sram/kernels/cartpole_physics.cuh``; constants from :class:`isaaclab_sram.cartpole.CartpolePhysics`)::

       M     = [[mc + mp, -mp l cos th], [-mp l cos th, Ip + mp l^2]]
       qfrc  = [-mp l sin(th) thd^2 + clamp(-kv sd, -F, F) + f,  mp g l sin th]
       (M + dt diag(kv, 0)) qacc = qfrc ;  qd += dt qacc ;  q += dt qd      (semi-implicit Euler)

   The feed-forward effort ``f`` is never clamped; only the actuator (damping) force is clamped to +-F = 400 N,
   while its implicit derivative ``dt kv`` stays in the mass matrix (this is what Newton 1.6.1 / MJWarp do).
3. ``episode_length += 1``.
4. ``terminated = |x| > max_cart_pos`` (no pole-angle termination), ``time_out = episode_length >= 300``.
5. Reward from the post-physics, pre-reset state:
   ``(alive (1 - T) + term T + pole_pos wrap_to_pi(th)^2 + cart_vel |xd| + pole_vel |thd|) * step_dt``.
6. Same-step reset of finished envs: ``U(range)`` offsets drawn in Isaac Lab's order (x, xd, th, thd) with
   ``torch.rand * (hi - lo) + lo``, clamped to the soft joint limits.
7. Observation ``[x, th, xd, thd]`` of the post-reset state (``th`` is *not* wrapped).

RSL-RL wrapper conventions: ``dones = (terminated | time_out).long()``, ``extras["time_outs"] = time_out`` (the
task is not finite-horizon), ``extras["log"]["Metrics/success_rate"]`` = fraction of the envs reset in the latest
step-with-resets that timed out (the persistent ``extras`` dict object is returned every step, like Isaac Lab).

Isaac Lab quirks reproduced on purpose:

* **first-reset zero velocities** -- the very first ``env.reset()`` (issued by the RSL-RL wrapper constructor)
  runs before Newton has populated ``soft_joint_vel_limits``, so the sampled joint velocities are clamped to 0;
  every later reset samples velocities normally (``first_reset_zero_velocity=True``, the default).
* ``init_at_random_ep_len`` is handled by the stock runner through the ``episode_length_buf`` setter.

Differences from the Isaac Lab env (all below the float32 noise floor of MJWarp itself, see ``--check``):

* Resets are mask-based (``torch.where`` over all envs, 4 uniforms drawn per env per step) instead of
  ``reset_buf.nonzero()`` + indexed writes; same distribution, no host sync, different RNG consumption.
* MJWarp integrates in world coordinates (env origins up to +-126 m) and therefore carries ~1e-6 rad/s of
  extra float32 rounding per step; the closed form here works in joint coordinates.
* The +-4 m cart joint limit (a soft MuJoCo constraint) is not modelled: it is unreachable because episodes
  terminate at |x| > 3 m (largest |x| observed at termination in the stock env: 3.06 m).

Two execution modes share one implementation:

* ``mode="eager"``    -- plain PyTorch, one CUDA kernel per elementwise op (~120 launches per ``step``),
* ``mode="compiled"`` -- the whole physics / reward / termination / reset / observation update is fused with
  ``torch.compile(fullgraph=True)`` (``compile_mode="max-autotune-no-cudagraphs"`` by default;
  ``"reduce-overhead"`` additionally captures it in a CUDA graph).

Run ``python torch_cartpole_vecenv.py --check`` for the numerical cross-checks (float64 reference, eager vs
compiled, the recorded stock Isaac Lab / MJWarp rollouts, and optionally the megakernel's own physics) and
``--bench`` for an (indicative) ``env.step`` micro-benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

_REPO = Path(__file__).resolve().parents[3]
_SRAM_PKG = _REPO / "source" / "isaaclab_sram"
if str(_SRAM_PKG) not in sys.path:
    sys.path.insert(0, str(_SRAM_PKG))

from isaaclab_sram.cartpole import CartpoleTask  # noqa: E402

COMPILE_MODES = ("default", "reduce-overhead", "max-autotune-no-cudagraphs", "max-autotune")


# ------------------------------------------------------------------------------------------------
# constants (float32, folded in the same order as the megakernel's compile-time float32 arithmetic)
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CartpoleConstants:
    """Scalar constants of the MDP; all values are exactly representable in float32."""

    # physics (one MuJoCo-Warp implicitfast substep)
    dt: float
    decimation: int
    mpl: float  # mp * lc
    neg_mpl: float  # -(mp * lc)
    neg_mpgl: float  # -(mp * g * lc)
    m11: float  # Ip + mp * lc * lc
    mh00: float  # (mc + mp) + dt * kv
    mh00_m11: float  # mh00 * m11
    neg_kv: float  # -kv
    act_frc: float  # actuator force limit (clamps the damping force only)
    action_scale: float
    # MDP
    max_cart_pos: float
    max_episode_length: int
    step_dt: float
    rew_alive: float
    rew_term: float
    rew_pole_pos: float
    rew_cart_vel: float
    rew_pole_vel: float
    # reset ranges (lo, hi - lo) in sample_uniform form: rand * (hi - lo) + lo
    rst_lo: tuple[float, float, float, float]  # (x, th, xd, thd)
    rst_span: tuple[float, float, float, float]
    cart_pos_limit: float  # soft joint position limit of the cart
    vel_limit: tuple[float, float]  # soft joint velocity limits (cart, pole)

    @classmethod
    def from_task(cls, task: CartpoleTask) -> CartpoleConstants:
        f = np.float32
        p = task.physics
        mc, mp, lc, ip, g, kv = (
            f(v) for v in (p.cart_mass, p.pole_mass, p.pole_com, p.pole_inertia, p.gravity, p.cart_damping)
        )
        dt = f(task.physics_dt)
        mpl = f(mp * lc)
        m11 = f(ip + f(mpl * lc))
        mh00 = f(f(mc + mp) + f(dt * kv))
        r = (
            task.initial_cart_position_range,
            task.initial_pole_angle_range,
            task.initial_cart_velocity_range,
            task.initial_pole_velocity_range,
        )
        return cls(
            dt=float(dt),
            decimation=int(task.decimation),
            mpl=float(mpl),
            neg_mpl=float(-mpl),
            neg_mpgl=float(-f(f(mp * g) * lc)),
            m11=float(m11),
            mh00=float(mh00),
            mh00_m11=float(f(mh00 * m11)),
            neg_kv=float(-kv),
            act_frc=float(f(p.actuator_force_limit)),
            action_scale=float(f(task.action_scale)),
            max_cart_pos=float(task.max_cart_pos),
            max_episode_length=int(task.max_episode_length),
            step_dt=float(task.step_dt),
            rew_alive=float(task.rew_scale_alive),
            rew_term=float(task.rew_scale_terminated),
            rew_pole_pos=float(task.rew_scale_pole_pos),
            rew_cart_vel=float(task.rew_scale_cart_vel),
            rew_pole_vel=float(task.rew_scale_pole_vel),
            rst_lo=tuple(float(lo) for lo, _ in r),
            rst_span=tuple(float(hi - lo) for lo, hi in r),
            cart_pos_limit=float(p.cart_limit),
            vel_limit=(float(p.cart_vel_limit), float(p.pole_vel_limit)),
        )


# ------------------------------------------------------------------------------------------------
# the MDP as pure tensor functions (shared by the eager and the torch.compile'd paths)
# ------------------------------------------------------------------------------------------------
def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    """``isaaclab.utils.math.wrap_to_pi`` (odd positive multiples of pi map to +pi)."""
    wrapped = (angles + torch.pi) % (2 * torch.pi)
    return torch.where((wrapped == 0) & (angles > 0), torch.pi, wrapped - torch.pi)


def cartpole_substep(s, th, sd, thd, f_app, c: CartpoleConstants):
    """One MuJoCo-Warp ``implicitfast`` substep (same float32 op order as the megakernel's ``cartpole_substep``)."""
    sth = torch.sin(th)
    cth = torch.cos(th)
    m01 = c.neg_mpl * cth
    bias0 = c.mpl * sth * thd * thd
    bias1 = c.neg_mpgl * sth
    f_act = torch.clamp(c.neg_kv * sd, -c.act_frc, c.act_frc)
    qf0 = -bias0 + f_act + f_app
    qf1 = -bias1
    det = c.mh00_m11 - m01 * m01
    qacc0 = (c.m11 * qf0 - m01 * qf1) / det
    qacc1 = (c.mh00 * qf1 - m01 * qf0) / det
    sd = sd + qacc0 * c.dt
    thd = thd + qacc1 * c.dt
    s = s + c.dt * sd
    th = th + c.dt * thd
    return s, th, sd, thd


def cartpole_physics(state: torch.Tensor, actions: torch.Tensor, c: CartpoleConstants):
    """Integrate one control step (``decimation`` substeps, held effort). Returns (s, th, sd, thd)."""
    s, th, sd, thd = state.unbind(1)
    f_app = c.action_scale * actions.reshape(-1)
    for _ in range(c.decimation):
        s, th, sd, thd = cartpole_substep(s, th, sd, thd, f_app, c)
    return s, th, sd, thd


def cartpole_reward(th, sd, thd, terminated, c: CartpoleConstants) -> torch.Tensor:
    """``isaaclab_tasks.core.cartpole.cartpole_direct_env.compute_rewards`` (same op order)."""
    pole_pos = wrap_to_pi(th)
    tf = terminated.float()
    rew_alive = c.rew_alive * (1.0 - tf)
    rew_termination = c.rew_term * tf
    rew_pole_pos = c.rew_pole_pos * torch.square(pole_pos)
    rew_cart_vel = c.rew_cart_vel * torch.abs(sd)
    rew_pole_vel = c.rew_pole_vel * torch.abs(thd)
    return (rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel) * c.step_dt


def cartpole_mdp_step(state, ep_len, success, actions, u, c: CartpoleConstants):
    """One full env step for all envs.

    Args:
        state: [N, 4] (x, th, xd, thd) before the step.
        ep_len: [N] int64 episode lengths before the step.
        success: [] float32 last logged success rate (kept when no env resets).
        actions: [N, 1] raw policy actions.
        u: [N, 4] uniforms in [0, 1) for the reset draws (x, xd, th, thd order like ``_reset_idx``).

    Returns:
        obs [N, 4] (post-reset), reward [N], terminated [N] bool, time_out [N] bool, dones [N] int64,
        new ep_len [N], new success [].
    """
    s, th, sd, thd = cartpole_physics(state, actions, c)
    ep_len = ep_len + 1
    terminated = torch.abs(s) > c.max_cart_pos
    time_out = ep_len >= c.max_episode_length
    reward = cartpole_reward(th, sd, thd, terminated, c)
    reset = terminated | time_out
    # Metrics/success_rate: reset_time_outs[env_ids].float().mean(), only updated on steps with resets
    n_reset = reset.sum()
    success = torch.where(n_reset > 0, time_out.sum() / n_reset.clamp(min=1), success)
    # same-step reset (sample_uniform order: cart pos, cart vel, pole pos, pole vel)
    x_new = torch.clamp(u[:, 0] * c.rst_span[0] + c.rst_lo[0], -c.cart_pos_limit, c.cart_pos_limit)
    xd_new = torch.clamp(u[:, 1] * c.rst_span[2] + c.rst_lo[2], -c.vel_limit[0], c.vel_limit[0])
    th_new = u[:, 2] * c.rst_span[1] + c.rst_lo[1]
    thd_new = torch.clamp(u[:, 3] * c.rst_span[3] + c.rst_lo[3], -c.vel_limit[1], c.vel_limit[1])
    obs = torch.stack(
        (
            torch.where(reset, x_new, s),
            torch.where(reset, th_new, th),
            torch.where(reset, xd_new, sd),
            torch.where(reset, thd_new, thd),
        ),
        dim=1,
    )
    ep_len = torch.where(reset, torch.zeros_like(ep_len), ep_len)
    return obs, reward, terminated, time_out, reset.long(), ep_len, success


def make_env_step(c: CartpoleConstants):
    """Return ``step(state, ep_len, success, actions) -> (obs, reward, terminated, time_out, dones)``.

    The persistent buffers ``state``, ``ep_len`` and ``success`` are updated in place, so that the very same
    function can be run eagerly or compiled (and CUDA-graph captured with static input addresses).
    """

    def env_step(state, ep_len, success, actions):
        u = torch.rand(state.shape[0], 4, device=state.device, dtype=state.dtype)
        obs, reward, terminated, time_out, dones, new_len, new_success = cartpole_mdp_step(
            state, ep_len, success, actions, u, c
        )
        state.copy_(obs)
        ep_len.copy_(new_len)
        success.copy_(new_success)
        return obs, reward, terminated, time_out, dones

    return env_step


# ------------------------------------------------------------------------------------------------
# the VecEnv
# ------------------------------------------------------------------------------------------------
class CartpoleTorchVecEnv(VecEnv):
    """Isaac-Cartpole-Direct (newton_mjwarp MDP) as a batched PyTorch VecEnv for the stock RSL-RL runner."""

    def __init__(
        self,
        task: CartpoleTask | None = None,
        num_envs: int = 4096,
        device: str = "cuda:0",
        mode: str = "eager",
        compile_mode: str = "max-autotune-no-cudagraphs",
        clip_actions: float | None = None,
        seed: int | None = None,
        first_reset_zero_velocity: bool = True,
        cfg: object | None = None,
        warmup: bool = True,
    ):
        """Create and reset the environments (the RSL-RL wrapper resets in its constructor, too).

        Args:
            task: MDP constants (default: the Isaac-Cartpole-Direct values); see :meth:`from_isaaclab_cfg`.
            num_envs: Number of parallel environments.
            device: Torch device string.
            mode: ``"eager"`` or ``"compiled"`` (``torch.compile`` of the whole step).
            compile_mode: ``torch.compile`` mode used when ``mode="compiled"``.
            clip_actions: Optional symmetric action clip applied before the step (RSL-RL wrapper semantics).
            seed: If given, seeds torch/numpy/random(/warp) like ``DirectRLEnv`` before the first reset.
            first_reset_zero_velocity: Reproduce the Isaac Lab 3.0 / Newton first-reset quirk (zero velocities).
            cfg: Object exposed as ``env.cfg`` (e.g. the Isaac Lab ``CartpoleEnvCfg``); defaults to ``task``.
            warmup: Compile (and capture) the step function in the constructor so the first training
                iteration is not charged with compilation; RNG state and env state are restored afterwards.
        """
        if mode not in ("eager", "compiled"):
            raise ValueError(f"unknown mode {mode!r}")
        if mode == "compiled" and compile_mode not in COMPILE_MODES:
            raise ValueError(f"unknown torch.compile mode {compile_mode!r}")
        self.task = task or CartpoleTask()
        self.consts = CartpoleConstants.from_task(self.task)
        self.num_envs = int(num_envs)
        self.num_actions = 1
        self.num_obs = 4
        self.max_episode_length = self.task.max_episode_length
        self.device = device
        self.cfg = cfg if cfg is not None else self.task
        self.clip_actions = clip_actions
        self.mode = mode
        self.compile_mode = compile_mode if mode == "compiled" else None
        self._emit_time_outs = not bool(getattr(self.cfg, "is_finite_horizon", False))
        dev = torch.device(device)
        self._dev = dev
        # persistent buffers (only ever updated in place, so they keep static addresses for CUDA graphs)
        self._state = torch.zeros(self.num_envs, 4, device=dev)
        self._ep_len = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        self._success = torch.zeros((), device=dev)
        self._last_time_outs = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
        self._vel_limits_ready = not first_reset_zero_velocity
        self.extras: dict = {}
        self.obs_buf: dict[str, torch.Tensor] = {}
        if seed is not None:
            self.seed(seed)

        step_fn = make_env_step(self.consts)
        self._clone_outputs = False
        if mode == "compiled":
            if compile_mode in ("reduce-overhead", "max-autotune"):
                # CUDA-graph modes: the mutated buffers must have static addresses, and the graph's output
                # buffers are overwritten by the next replay while RSL-RL still holds the previous observation
                # (Transition.observations), so outputs are cloned out of the graph pool.
                for t in (self._state, self._ep_len, self._success):
                    torch._dynamo.mark_static_address(t)
                self._clone_outputs = True
            step_fn = torch.compile(step_fn, mode=compile_mode, fullgraph=True, dynamic=False)
        self._step_fn = step_fn
        self.compile_time_s = 0.0
        if mode == "compiled" and warmup:
            self._warmup()
        self.reset()

    @classmethod
    def from_isaaclab_cfg(cls, env_cfg, **kwargs) -> CartpoleTorchVecEnv:
        """Build from an Isaac Lab ``CartpoleEnvCfg`` (MDP constants, num_envs, seed are read from it)."""
        kwargs.setdefault("num_envs", env_cfg.scene.num_envs)
        kwargs.setdefault("seed", env_cfg.seed)
        kwargs.setdefault("cfg", env_cfg)
        return cls(CartpoleTask.from_env_cfg(env_cfg), **kwargs)

    # -- VecEnv properties ------------------------------------------------------------------------
    @property
    def episode_length_buf(self) -> torch.Tensor:
        """Episode lengths (int64); RSL-RL randomises them via the setter when ``init_at_random_ep_len``."""
        return self._ep_len

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self._ep_len.copy_(value)

    @property
    def unwrapped(self) -> CartpoleTorchVecEnv:
        return self

    # -- operations -----------------------------------------------------------------------------
    def seed(self, seed: int = -1) -> int:
        """Seed like ``DirectRLEnv.seed`` (``isaaclab.utils.seed.configure_seed`` when available)."""
        try:
            from isaaclab.utils.seed import configure_seed
        except ImportError:  # pragma: no cover - Isaac Lab not importable
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            return seed
        return configure_seed(seed)

    def reset(self) -> tuple[TensorDict, dict]:
        """Reset all envs (``DirectRLEnv.reset`` -> ``CartpoleEnv._reset_idx(all)``)."""
        n, c = self.num_envs, self.consts
        self.extras.setdefault("log", {})["Metrics/success_rate"] = self._last_time_outs.float().mean()
        self._success.copy_(self.extras["log"]["Metrics/success_rate"])

        def sample(k):  # isaaclab.utils.math.sample_uniform, drawn one quantity at a time like _reset_idx
            return torch.rand(n, 1, device=self._dev) * c.rst_span[k] + c.rst_lo[k]

        x, xd, th, thd = sample(0), sample(2), sample(1), sample(3)
        vlim = c.vel_limit if self._vel_limits_ready else (0.0, 0.0)  # first-reset quirk: limits still 0
        x = x.clamp(-c.cart_pos_limit, c.cart_pos_limit)
        xd = xd.clamp(-vlim[0], vlim[0])
        thd = thd.clamp(-vlim[1], vlim[1])
        self._state.copy_(torch.cat((x, th, xd, thd), dim=1))
        self._ep_len.zero_()
        self.obs_buf = {"policy": self._state.clone()}
        return TensorDict(self.obs_buf, batch_size=[self.num_envs]), self.extras

    def get_observations(self) -> TensorDict:
        return TensorDict(self.obs_buf, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        out = self._step_fn(self._state, self._ep_len, self._success, actions.to(self._dev))
        if self._clone_outputs:
            out = tuple(t.clone() for t in out)
        obs, reward, terminated, time_out, dones = out
        self._vel_limits_ready = True  # Newton has populated soft_joint_vel_limits after the first step
        self._last_time_outs = time_out
        self.obs_buf = {"policy": obs}
        if self._emit_time_outs:
            self.extras["time_outs"] = time_out
        self.extras["log"]["Metrics/success_rate"] = self._success
        return TensorDict(self.obs_buf, batch_size=[self.num_envs]), reward, dones, self.extras

    def close(self):
        pass

    def _warmup(self, calls: int = 3):
        """Compile / capture the step on the real buffers, then restore buffers and the CPU / CUDA RNG states
        (so the first reset and the runner's weight initialisation see the same streams as in eager mode)."""
        rng, cpu_rng = torch.cuda.get_rng_state(self._dev), torch.get_rng_state()
        saved = [t.clone() for t in (self._state, self._ep_len, self._success)]
        torch.cuda.synchronize(self._dev)
        t0 = time.perf_counter()
        # the runner steps the env under inference mode with inference-tensor actions; dynamo guards on both
        # (grad mode and the tensor dispatch-key set), so warm up with exactly that or the first step recompiles
        with torch.inference_mode():
            zeros = torch.zeros(self.num_envs, 1, device=self._dev)
            for _ in range(calls):
                out = self._step_fn(self._state, self._ep_len, self._success, zeros)
                if self._clone_outputs:
                    out = tuple(t.clone() for t in out)
        torch.cuda.synchronize(self._dev)
        self.compile_time_s = time.perf_counter() - t0
        del out
        for buf, val in zip((self._state, self._ep_len, self._success), saved):
            buf.copy_(val)
        torch.cuda.set_rng_state(rng, self._dev)
        torch.set_rng_state(cpu_rng)


# ------------------------------------------------------------------------------------------------
# helpers shared with the training script
# ------------------------------------------------------------------------------------------------
@torch.inference_mode()
def evaluate_policy(env: CartpoleTorchVecEnv, policy, num_steps: int) -> dict:
    """Deterministic roll-out with the episode bookkeeping of ``isaaclab.benchmark.stepping.run_play_loop``.

    Resets the env, then accumulates per-env return / length and records them whenever an env is done.
    """
    obs, _ = env.reset()
    n = env.num_envs
    run_ret = torch.zeros(n, device=env._dev)
    run_len = torch.zeros(n, device=env._dev)
    rets, lens, timeouts = [], [], []
    for _ in range(num_steps):
        obs, rew, dones, extras = env.step(policy(obs))
        run_ret += rew
        run_len += 1.0
        d = dones.bool()
        rets.append(run_ret[d].clone())
        lens.append(run_len[d].clone())
        timeouts.append(extras["time_outs"][d].clone())
        run_ret[d] = 0.0
        run_len[d] = 0.0
    r, ln, to = torch.cat(rets).double(), torch.cat(lens).double(), torch.cat(timeouts).double()
    if r.numel() == 0:
        return {"episodes": 0}
    return {
        "episodes": int(r.numel()),
        "return_mean": float(r.mean()),
        "return_std": float(r.std()) if r.numel() > 1 else 0.0,
        "return_max": float(r.max()),
        "episode_length_mean": float(ln.mean()),
        "timeout_fraction": float(to.mean()),
    }


# ------------------------------------------------------------------------------------------------
# numerical cross-checks
# ------------------------------------------------------------------------------------------------
def _substep64(s, th, sd, thd, f, task: CartpoleTask):
    """Independent float64 numpy reference of one implicitfast substep (as in scripts/sram/verify_megakernel.py)."""
    p = task.physics
    dt = float(np.float32(task.physics_dt))
    c, sn = np.cos(th), np.sin(th)
    m00 = p.cart_mass + p.pole_mass
    m01 = -p.pole_mass * p.pole_com * c
    m11 = p.pole_inertia + p.pole_mass * p.pole_com**2
    b0 = p.pole_mass * p.pole_com * sn * thd**2
    b1 = -p.pole_mass * p.gravity * p.pole_com * sn
    fa = np.clip(-p.cart_damping * sd, -p.actuator_force_limit, p.actuator_force_limit)
    q0, q1 = -b0 + fa + f, -b1
    mh = m00 + dt * p.cart_damping
    det = mh * m11 - m01**2
    a0 = (m11 * q0 - m01 * q1) / det
    a1 = (mh * q1 - m01 * q0) / det
    sd = sd + dt * a0
    thd = thd + dt * a1
    return s + dt * sd, th + dt * thd, sd, thd


def _physics64(state: np.ndarray, actions: np.ndarray, task: CartpoleTask) -> np.ndarray:
    s, th, sd, thd = (state[:, i].astype(np.float64) for i in range(4))
    f = (np.float32(task.action_scale) * actions.reshape(-1).astype(np.float32)).astype(np.float64)
    for _ in range(task.decimation):
        s, th, sd, thd = _substep64(s, th, sd, thd, f, task)
    return np.stack([s, th, sd, thd], 1)


def _wrap64(a):
    w = np.mod(a + np.pi, 2 * np.pi)
    return np.where((w == 0) & (a > 0), np.pi, w - np.pi)


def _maxerr(a, b) -> list[float]:
    return [float(v) for v in np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).reshape(-1, 4).max(0)]


def _fmt(v) -> str:
    return "[" + ", ".join(f"{x:.2e}" for x in v) + "]"


def run_checks(args) -> dict:
    """Numerical cross-checks of the torch env; returns a JSON-able report (and prints it)."""
    dev = torch.device(args.device)
    task = CartpoleTask()
    c = CartpoleConstants.from_task(task)
    report: dict = {}
    rng = np.random.default_rng(0)

    # 1) physics vs an independent float64 closed form, incl. wound-up poles and the actuator clamp region
    n = 1 << 16
    st = np.stack(
        [rng.uniform(-3, 3, n), rng.uniform(-12, 12, n), rng.uniform(-60, 60, n), rng.uniform(-25, 25, n)], 1
    ).astype(np.float32)
    act = rng.uniform(-6, 6, (n, 1)).astype(np.float32)
    ref = _physics64(st, act, task)
    st_t, act_t = torch.tensor(st, device=dev), torch.tensor(act, device=dev)
    eager = torch.stack(cartpole_physics(st_t, act_t, c), 1).cpu().numpy()
    scale = np.maximum(np.abs(ref), 1.0)
    report["physics_eager_vs_float64"] = {
        "max_abs": _maxerr(eager, ref),
        "max_rel": [float(v) for v in (np.abs(eager - ref) / scale).max(0)],
    }
    print(
        f"[physics] eager float32 vs float64 reference ({n} random states, |th|<12, |xd|<60, |a|<6): "
        f"max abs {_fmt(report['physics_eager_vs_float64']['max_abs'])}"
    )

    # 2) compiled vs eager (same inputs)
    if not args.skip_compiled:
        phys_c = torch.compile(
            lambda s_, a_: torch.stack(cartpole_physics(s_, a_, c), 1),
            fullgraph=True,
            dynamic=False,
            mode="max-autotune-no-cudagraphs",
        )
        comp = phys_c(st_t, act_t).cpu().numpy()
        report["physics_compiled_vs_eager"] = _maxerr(comp, eager)
        report["physics_compiled_vs_float64"] = _maxerr(comp, ref)
        print(
            f"[physics] torch.compile vs eager: max abs {_fmt(report['physics_compiled_vs_eager'])}; "
            f"vs float64 {_fmt(report['physics_compiled_vs_float64'])}"
        )
        # full MDP step (reward / termination / reset / obs) with identical reset uniforms
        ep = torch.randint(0, task.max_episode_length, (n,), device=dev)
        u = torch.rand(n, 4, device=dev)
        succ = torch.zeros((), device=dev)
        st_mdp = st_t.clone()
        st_mdp[:, 0] = torch.empty(n, device=dev).uniform_(-3.2, 3.2)  # exercise terminations
        e_out = cartpole_mdp_step(st_mdp, ep, succ, act_t, u, c)
        c_out = torch.compile(lambda *a: cartpole_mdp_step(*a, c), fullgraph=True, dynamic=False)(
            st_mdp, ep, succ, act_t, u
        )
        names = ["obs", "reward", "terminated", "time_out", "dones", "ep_len", "success"]
        diffs = {k: float((a.double() - b.double()).abs().max()) for k, a, b in zip(names, e_out, c_out)}
        report["mdp_step_compiled_vs_eager"] = diffs
        print(f"[mdp] torch.compile vs eager full step: {json.dumps({k: f'{v:.2e}' for k, v in diffs.items()})}")
        report["env_level"] = _check_env_modes(args, task)

    # 3) teacher-forced one-step check against recorded stock Isaac Lab (Newton / MJWarp) roll-outs
    for path in args.dataset:
        if not os.path.isfile(path):
            print(f"[isaaclab] dataset {path} not found, skipped")
            continue
        d = np.load(path)
        T, N = d["q_pre"].shape[:2]
        pre = np.concatenate([d["q_pre"], d["qd_pre"]], -1).reshape(T * N, 4)
        post = np.concatenate([d["q_post"], d["qd_post"]], -1).reshape(T * N, 4)
        a = d["action"].reshape(T * N, 1)
        with torch.no_grad():
            s1, th1, sd1, thd1 = cartpole_physics(torch.tensor(pre, device=dev), torch.tensor(a, device=dev), c)
            ours = torch.stack((s1, th1, sd1, thd1), 1).cpu().numpy()
            eplen = torch.tensor(d["episode_length"].reshape(-1), device=dev)
            term = torch.abs(s1) > c.max_cart_pos
            tout = eplen >= c.max_episode_length
            rew = cartpole_reward(th1, sd1, thd1, term, c).cpu().numpy()
        # the cart joint limit (|x| >= 4) is only reachable with resets disabled; exclude those samples
        ok = (np.abs(pre[:, 0]) < 3.9) & (np.abs(post[:, 0]) < 3.9)
        err = np.abs(ours - post)[ok]
        ref64 = _physics64(pre[ok], a[ok], task)
        name = Path(path).stem
        rep = {
            "transitions": int(ok.sum()),
            "excluded_near_joint_limit": int((~ok).sum()),
            "one_step_max_abs_err_vs_mjwarp": [float(v) for v in err.max(0)],
            "one_step_p999_abs_err_vs_mjwarp": [float(v) for v in np.quantile(err, 0.999, axis=0)],
            "float64_ref_max_abs_err_vs_mjwarp": [float(v) for v in np.abs(ref64 - post[ok]).max(0)],
            "max_abs_err_vs_float64_ref": [float(v) for v in np.abs(ours[ok] - ref64).max(0)],
        }
        if "reward" in d.files and float(d["max_cart_pos"]) < 1e6:
            rec_term = d["terminated"].reshape(-1)
            rec_tout = d["truncated"].reshape(-1)
            rep["reward_max_abs_err"] = float(np.abs(rew - d["reward"].reshape(-1)).max())
            rep["terminated_mismatches"] = int((term.cpu().numpy() != rec_term).sum())
            rep["time_out_mismatches"] = int((tout.cpu().numpy() != rec_tout).sum())
            rep["terminations"] = int(rec_term.sum())
            rep["time_outs"] = int(rec_tout.sum())
            # post-reset observations of reset envs must lie in the configured reset ranges
            obs = d["obs"].reshape(-1, 4)
            rs = obs[rec_term | rec_tout]
            lo = np.array([c.rst_lo[0], c.rst_lo[1], c.rst_lo[2], c.rst_lo[3]])
            hi = lo + np.array(c.rst_span)
            rep["reset_states_outside_ranges"] = int(((rs < lo - 1e-6) | (rs > hi + 1e-6)).sum())
            # non-reset envs observe the post-physics state
            keep = ~(rec_term | rec_tout)
            rep["obs_equals_post_state_max_err"] = float(np.abs(obs[keep] - post[keep]).max())
        if "q_init" in d.files and "qd_init" in d.files:
            # first reset with the env seed (42 in collect.py): torch's CUDA Philox stream reproduces the draws
            init = np.concatenate([d["q_init"], d["qd_init"]], -1)
            env0 = CartpoleTorchVecEnv(task, num_envs=init.shape[0], device=args.device, seed=args.dataset_seed)
            mine = env0._state.cpu().numpy()
            rep[f"first_reset_max_abs_diff_seed_{args.dataset_seed}"] = _maxerr(mine, init)
            rep["first_reset_velocities_all_zero"] = bool(np.all(init[:, 2:] == 0) and np.all(mine[:, 2:] == 0))
        report[f"isaaclab_{name}"] = rep
        print(
            f"[isaaclab] {name}: {rep['transitions']} transitions (excluded {rep['excluded_near_joint_limit']} "
            f"near the +-4 m joint limit)\n"
            f"           one-step max |torch - MJWarp| {_fmt(rep['one_step_max_abs_err_vs_mjwarp'])} "
            f"(p99.9 {_fmt(rep['one_step_p999_abs_err_vs_mjwarp'])})\n"
            f"           float64 closed form vs MJWarp       {_fmt(rep['float64_ref_max_abs_err_vs_mjwarp'])}; "
            f"torch vs float64 {_fmt(rep['max_abs_err_vs_float64_ref'])}"
        )
        extra = {
            k: v
            for k, v in rep.items()
            if k.startswith(("reward", "terminat", "time_out", "reset", "obs_", "first_reset"))
        }
        if extra:
            print(f"           {json.dumps(extra)}")

    # 4) against the megakernel's own float32 physics (its debug build dumps obs/action/post-physics states)
    if args.megakernel:
        report["megakernel"] = _check_against_megakernel(args, task, c)
    return report


def _check_env_modes(args, task: CartpoleTask) -> dict:
    """Env-level eager vs compiled (both compile flavours): same seed -> same first reset; then a fixed feedback
    policy for the full 300-step episode (no env terminates), compared step by step; then the synchronous
    time-out reset at step 300 (fresh draws differ between modes since inductor has its own RNG stream)."""
    n, steps = 1024, task.max_episode_length
    # LQR state feedback a = K s for the linearised cart-pole (Q = diag(1, 10, .1, .1), R = 1): stable closed
    # loop, |x| < 2.6 m from the reset distribution, |a| up to ~5 (500 N > the 400 N actuator limit)
    gains = torch.tensor([[1.0, -5.525, 1.194, -1.185]], device=args.device)
    envs = {"eager": CartpoleTorchVecEnv(task, n, args.device, mode="eager", seed=7)}
    for cm in ("max-autotune-no-cudagraphs", "reduce-overhead"):
        envs[cm] = CartpoleTorchVecEnv(task, n, args.device, mode="compiled", compile_mode=cm, seed=7)
    obs = {k: e.get_observations()["policy"] for k, e in envs.items()}
    rep = {k: {"first_reset_max_diff": float((obs[k] - obs["eager"]).abs().max())} for k in envs if k != "eager"}
    traj = {k: [] for k in envs}
    with torch.inference_mode():
        for t in range(steps):
            for k, e in envs.items():
                o, r, d, ex = e.step((obs[k] * gains).sum(1, keepdim=True))
                obs[k] = o["policy"]
                traj[k].append(
                    (
                        o["policy"].clone(),
                        r.clone(),
                        d.clone(),
                        ex["time_outs"].clone(),
                        e.episode_length_buf.clone(),
                        float(ex["log"]["Metrics/success_rate"]),
                    )
                )
    for k in rep:
        pre = range(steps - 1)  # steps before the synchronous time-out
        dobs = torch.stack([(traj[k][t][0] - traj["eager"][t][0]).abs().max(0).values for t in pre])
        rep[k].update(
            {
                "obs_max_diff_first_10_steps": [float(v) for v in dobs[:10].max(0).values],
                "obs_max_diff_299_steps": [float(v) for v in dobs.max(0).values],
                "reward_max_diff": float(max((traj[k][t][1] - traj["eager"][t][1]).abs().max() for t in pre)),
                "dones_mismatch_before_timeout": int(sum((traj[k][t][2] != traj["eager"][t][2]).sum() for t in pre)),
                "ep_len_mismatch": int(sum((traj[k][t][4] != traj["eager"][t][4]).sum() for t in range(steps))),
            }
        )
        last = traj[k][-1]
        ranges = (
            task.initial_cart_position_range,
            task.initial_pole_angle_range,
            task.initial_cart_velocity_range,
            task.initial_pole_velocity_range,
        )
        lo = torch.tensor([r[0] for r in ranges], device=args.device)
        hi = torch.tensor([r[1] for r in ranges], device=args.device)
        rep[k]["timeout_step"] = {
            "all_done": bool(last[2].all()),
            "all_time_outs": bool(last[3].all()),
            "ep_len_all_zero": bool((last[4] == 0).all()),
            "success_rate": last[5],
            "reset_states_in_range": bool(((last[0] >= lo) & (last[0] <= hi)).all()),
            "reset_velocities_nonzero": bool((last[0][:, 2:].abs() > 0).float().mean() > 0.99),
        }
    eager_last = traj["eager"][-1]
    rep["eager_timeout_step"] = {"all_time_outs": bool(eager_last[3].all()), "success_rate": eager_last[5]}
    for k, v in rep.items():
        print(f"[env] {k}: {json.dumps(v)}")
    return rep


def _check_against_megakernel(args, task: CartpoleTask, c: CartpoleConstants) -> dict:
    """Re-integrate every transition of one megakernel iteration with this env's physics / MDP functions."""
    from isaaclab_sram import megakernel as mk

    trainer = mk.SramPPOCartpole(num_envs=args.mk_num_envs, seed=args.mk_seed, debug=True)
    plan, ppo = trainer.plan, trainer.ppo
    if args.mk_warmup_iters:
        trainer.train(args.mk_warmup_iters)
    trainer.dbg.zero_()
    trainer.train(1)
    torch.cuda.synchronize()
    # debug-dump layout of kernels/ppo_cartpole.cu (DBG build), as parsed by scripts/sram/verify_megakernel.py
    T, G, EM = ppo.num_steps_per_env, plan.num_ctas, plan.envs_per_cta_max
    fs = T * EM
    dbg = trainer.dbg.detach().cpu().numpy()
    post = dbg[: G * 4 * fs].reshape(G, 4, fs)
    store = dbg[G * 4 * fs : G * 4 * fs + G * (mk.NFIELD + 1) * fs].reshape(G, mk.NFIELD + 1, fs)
    obs_l, act_l, post_l, rew_l, done_l, val_l = [], [], [], [], [], []
    for g in range(G):
        e0, e1 = (g * plan.num_envs) // G, ((g + 1) * plan.num_envs) // G
        e = e1 - e0
        st = store[g, :, : e * T]
        obs_l.append(st[0:4].reshape(4, T * e).T)
        act_l.append(st[4].reshape(-1, 1))
        rew_l.append(st[8].reshape(-1))
        done_l.append(st[9].reshape(-1))
        val_l.append(st[6].reshape(-1))
        post_l.append(post[g, :, : e * T].reshape(4, T * e).T)
    obs, act, pst = np.concatenate(obs_l), np.concatenate(act_l), np.concatenate(post_l)
    rew_k, done_k, val_k = np.concatenate(rew_l), np.concatenate(done_l) > 0.5, np.concatenate(val_l)
    dev = torch.device(args.device)
    with torch.no_grad():
        s1, th1, sd1, thd1 = cartpole_physics(torch.tensor(obs, device=dev), torch.tensor(act, device=dev), c)
        ours = torch.stack((s1, th1, sd1, thd1), 1).cpu().numpy()
        term = torch.abs(s1) > c.max_cart_pos
        rew = cartpole_reward(th1, sd1, thd1, term, c).double().cpu().numpy()
        term = term.cpu().numpy()
    ref64 = _physics64(obs, act, task)
    # reward incl. RSL-RL time-out bootstrap (r + gamma V on time-outs); time-outs = done & ~terminated
    tout = done_k & ~term
    boot = rew + np.where(tout, ppo.gamma * val_k, 0.0)
    rep = {
        "transitions": int(obs.shape[0]),
        "physics_max_abs_err_torch_vs_megakernel": _maxerr(ours, pst),
        "physics_max_abs_err_float64_vs_megakernel": _maxerr(ref64, pst),
        "physics_max_abs_err_torch_vs_float64": _maxerr(ours, ref64),
        "bitwise_equal_fraction": float(np.mean(np.all(ours == pst, axis=1))),
        "reward_max_abs_err": float(np.abs(boot - rew_k).max()),
        "terminated_flag_mismatches": int(((term | tout) != done_k).sum()),
    }
    print(
        f"[megakernel] {rep['transitions']} transitions of one megakernel iteration (seed {args.mk_seed}, "
        f"after {args.mk_warmup_iters} warm-up iterations)\n"
        f"           physics max |torch - megakernel| {_fmt(rep['physics_max_abs_err_torch_vs_megakernel'])}; "
        f"bitwise-equal fraction {rep['bitwise_equal_fraction']:.4f}\n"
        f"           float64 vs megakernel {_fmt(rep['physics_max_abs_err_float64_vs_megakernel'])}; "
        f"torch vs float64 {_fmt(rep['physics_max_abs_err_torch_vs_float64'])}\n"
        f"           bootstrapped reward max err {rep['reward_max_abs_err']:.2e}; "
        f"done-flag mismatches {rep['terminated_flag_mismatches']}"
    )
    return rep


# ------------------------------------------------------------------------------------------------
# env.step micro-benchmark (indicative only on a shared GPU)
# ------------------------------------------------------------------------------------------------
def run_bench(args) -> dict:
    """Time ``env.step`` alone (random actions, inference mode) and count CUDA kernels per step."""
    out = {}
    modes = [("eager", None)] + [("compiled", m) for m in args.bench_compile_modes]
    for mode, cmode in modes:
        label = mode if cmode is None else f"compiled[{cmode}]"
        env = CartpoleTorchVecEnv(
            num_envs=args.num_envs,
            device=args.device,
            mode=mode,
            compile_mode=cmode or "max-autotune-no-cudagraphs",
            seed=0,
        )
        acts = torch.randn(args.bench_steps, env.num_envs, 1, device=env._dev)
        with torch.inference_mode():
            for i in range(20):
                env.step(acts[i])
            torch.cuda.synchronize()
            # host-return time per step (what the runner sees) and GPU time per step
            t0 = time.perf_counter()
            ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ev0.record()
            for i in range(args.bench_steps):
                env.step(acts[i])
            ev1.record()
            host_s = time.perf_counter() - t0
            torch.cuda.synchronize()
            gpu_ms = ev0.elapsed_time(ev1)
            # kernels launched by one step
            from torch.profiler import ProfilerActivity, profile

            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                env.step(acts[0])
                torch.cuda.synchronize()
            kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        r = {
            "host_ms_per_step": 1e3 * host_s / args.bench_steps,
            "gpu_ms_per_step": gpu_ms / args.bench_steps,
            "cuda_kernels_per_step": len(kernels),
            "env_steps_per_s": args.num_envs * args.bench_steps / (gpu_ms / 1e3),
            "compile_time_s": env.compile_time_s,
        }
        out[label] = r
        print(
            f"[bench] {label:38s} {r['host_ms_per_step']:.3f} ms/step host-return, {r['gpu_ms_per_step']:.3f} "
            f"ms/step GPU (events, back-to-back), {r['cuda_kernels_per_step']} CUDA kernels/step, "
            f"compile {r['compile_time_s']:.1f} s"
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="run the numerical cross-checks")
    ap.add_argument("--bench", action="store_true", help="env.step micro-benchmark (indicative only)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num_envs", type=int, default=4096)
    ap.add_argument("--skip_compiled", action="store_true", help="skip the torch.compile cross-checks")
    ap.add_argument(
        "--dataset",
        nargs="*",
        default=[],
        help="recorded stock Isaac Lab (Newton/MJWarp) roll-outs (record_newton_rollouts.py)",
    )
    ap.add_argument("--dataset_seed", type=int, default=42, help="env seed the datasets were recorded with")
    ap.add_argument("--megakernel", action="store_true", help="also compare with the megakernel's physics (NVRTC)")
    ap.add_argument("--mk_num_envs", type=int, default=4096)
    ap.add_argument("--mk_seed", type=int, default=3)
    ap.add_argument(
        "--mk_warmup_iters",
        type=int,
        default=20,
        help="train the megakernel this many iterations first (trained policies push states harder)",
    )
    ap.add_argument("--bench_steps", type=int, default=300)
    ap.add_argument("--bench_compile_modes", nargs="*", default=["max-autotune-no-cudagraphs", "reduce-overhead"])
    ap.add_argument("--out", default=None, help="write the JSON report here")
    args = ap.parse_args()
    if not (args.check or args.bench):
        ap.error("nothing to do: pass --check and/or --bench")
    report = {}
    if args.check:
        report["check"] = run_checks(args)
    if args.bench:
        report["bench"] = run_bench(args)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

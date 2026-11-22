# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record reference trajectories of the stock Isaac-Cartpole-Direct env (Newton / MJWarp).

Used by ``torch_cartpole_vecenv.py --check --dataset <file>.npz`` to compare the closed-form physics with the
real simulator transition by transition.

Usage (one config per process; the sim is a process-wide singleton), with
REC=scripts/sram/baselines/record_newton_rollouts.py:
  OMP_NUM_THREADS=1 ./isaaclab.sh -p $REC --actions zero   --resets off
  OMP_NUM_THREADS=1 ./isaaclab.sh -p $REC --actions random --resets off
  OMP_NUM_THREADS=1 ./isaaclab.sh -p $REC --actions random --resets on
  OMP_NUM_THREADS=1 ./isaaclab.sh -p $REC --actions zero   --resets on
  # diagnostics: all envs at the world origin (isolates MJWarp's world-coordinate float32 rounding)
  OMP_NUM_THREADS=1 ./isaaclab.sh -p $REC --actions random --resets off --spacing 0
  # effort-limit test: |a| up to 10 -> 1000 N applied force, cart speeds > 40 m/s (actuator clamp region)
  OMP_NUM_THREADS=1 ./isaaclab.sh -p $REC --actions random --resets off --action_amp 10 --steps 120
All runs use env seed 42 (identical initial states across files) and action seed 1234.

Actions go through the env's normal path: env.step(a) -> _pre_physics_step (a*100 N) -> _apply_action ->
set_joint_effort_target -> Newton control.joint_f -> MJWarp qfrc_applied.

--resets off : cfg.max_cart_pos=1e9 and cfg.episode_length_s=1e6, so neither termination nor time-out fires
               (pure physics; the cart may reach its +-4 m joint limit).
--resets on  : the stock MDP; terminal (pre-reset) states are snapshotted inside _get_dones and the post-reset
               states are recorded so a reference can be teacher-forced through resets.

Recorded per control step t (arrays are [T, N, ...]):
  q_pre, qd_pre     joint pos/vel the physics step starts from (after the previous step's resets)
  action            raw action passed to env.step (before the x100 scale)
  q_post, qd_post   joint pos/vel after the 2 physics substeps, before any reset (read in _get_dones)
  reward, terminated, truncated, episode_length (after increment, before reset)
  obs               observation returned by env.step (post-reset state)
  nefc, niter       MJWarp constraint rows / Newton-solver iterations of the last substep in the step
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import warp as wp  # noqa: E402

wp.config.enable_backward = False

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.app import launch_simulation  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

TASK = "Isaac-Cartpole-Direct"


def make_cfg(num_envs: int, save_to_mjcf: str | None = None, use_cuda_graph: bool | None = None):
    cfg, agent_cfg = resolve_task_config(TASK, "rsl_rl_cfg_entry_point", overrides=["physics=newton_mjwarp"])
    cfg.scene.num_envs = num_envs
    cfg.sim.device = "cuda:0"
    cfg.sim.visualizer_cfgs = []
    if save_to_mjcf is not None:
        cfg.sim.physics.solver_cfg.save_to_mjcf = save_to_mjcf
    if use_cuda_graph is not None:
        cfg.sim.physics.use_cuda_graph = use_cuda_graph
    return cfg, agent_cfg


def launch(cfg):
    return launch_simulation(cfg, {"visualizer": ["none"], "device": "cuda:0", "livestream": 0})


def make_env(cfg):
    return gym.make(TASK, cfg=cfg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=4096)
    p.add_argument("--steps", type=int, default=320)
    p.add_argument("--actions", choices=["zero", "random"], default="zero")
    p.add_argument("--resets", choices=["off", "on"], default="off")
    p.add_argument("--seed", type=int, default=42, help="env seed (reset sampling)")
    p.add_argument("--action_seed", type=int, default=1234)
    p.add_argument("--spacing", type=float, default=None, help="override scene.env_spacing (stock: 4.0)")
    p.add_argument(
        "--action_amp", type=float, default=1.0, help="random actions ~ U(-amp, amp) (stock policy is unclipped)"
    )
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    tag = "" if args.spacing is None else f"_spacing{args.spacing:g}"
    if args.action_amp != 1.0:
        tag += f"_amp{args.action_amp:g}"
    out = args.out or f"ref_{args.actions}_resets{args.resets}_n{args.num_envs}{tag}.npz"
    cfg, _ = make_cfg(args.num_envs)
    if args.spacing is not None:
        cfg.scene.env_spacing = args.spacing
    cfg.seed = args.seed
    if args.resets == "off":
        cfg.max_cart_pos = 1.0e9
        cfg.episode_length_s = 1.0e6

    with launch(cfg):
        env = make_env(cfg)
        uenv = env.unwrapped
        art = uenv.cartpole
        from isaaclab_newton.physics import NewtonManager

        solver = NewtonManager._solver
        dev = uenv.device
        N, T = args.num_envs, args.steps
        cart = int(uenv._cart_dof_idx[0])
        pole = int(uenv._pole_dof_idx[0])
        assert (cart, pole) == (0, 1), (cart, pole)

        # snapshot of the post-physics / pre-reset state, taken inside _get_dones
        snap = {}
        orig_get_dones = uenv._get_dones

        def get_dones_hook():
            res = orig_get_dones()
            snap["q"] = uenv.joint_pos.clone()
            snap["qd"] = uenv.joint_vel.clone()
            snap["eplen"] = uenv.episode_length_buf.clone()
            return res

        uenv._get_dones = get_dones_hook

        obs, _ = env.reset()
        # actions
        if args.actions == "zero":
            actions = torch.zeros((T, N, 1), device=dev)
        else:
            g = torch.Generator(device=dev)
            g.manual_seed(args.action_seed)
            actions = (torch.rand((T, N, 1), generator=g, device=dev) * 2.0 - 1.0) * args.action_amp

        rec = {
            k: []
            for k in [
                "q_pre",
                "qd_pre",
                "q_post",
                "qd_post",
                "reward",
                "terminated",
                "truncated",
                "episode_length",
                "obs",
                "nefc",
                "niter",
                "mj_qpos",
                "mj_qvel",
            ]
        }
        q0 = art.data.joint_pos.torch.clone()
        qd0 = art.data.joint_vel.torch.clone()
        obs0 = obs["policy"].clone()
        t0 = time.time()
        for t in range(T):
            rec["q_pre"].append(art.data.joint_pos.torch.clone())
            rec["qd_pre"].append(art.data.joint_vel.torch.clone())
            obs, rew, term, trunc, extras = env.step(actions[t])
            rec["q_post"].append(snap["q"])
            rec["qd_post"].append(snap["qd"])
            rec["episode_length"].append(snap["eplen"])
            rec["reward"].append(rew.clone())
            rec["terminated"].append(term.clone())
            rec["truncated"].append(trunc.clone())
            rec["obs"].append(obs["policy"].clone())
            rec["nefc"].append(torch.as_tensor(solver.mjw_data.nefc.numpy()).clone())
            rec["niter"].append(torch.as_tensor(solver.mjw_data.solver_niter.numpy()).clone())
            rec["mj_qpos"].append(torch.as_tensor(solver.mjw_data.qpos.numpy()).clone())
            rec["mj_qvel"].append(torch.as_tensor(solver.mjw_data.qvel.numpy()).clone())
        torch.cuda.synchronize()
        print(f"rollout {T} steps x {N} envs took {time.time() - t0:.2f}s (contended host, indicative only)")

        data = {k: torch.stack(v).cpu().numpy() for k, v in rec.items()}
        data["action"] = actions.cpu().numpy()
        data["q_init"] = q0.cpu().numpy()
        data["qd_init"] = qd0.cpu().numpy()
        data["obs_init"] = obs0.cpu().numpy()
        data["env_origins"] = uenv.scene.env_origins.cpu().numpy()
        data["physics_dt"] = np.float64(uenv.physics_dt)
        data["step_dt"] = np.float64(uenv.step_dt)
        data["decimation"] = np.int64(uenv.cfg.decimation)
        data["action_scale"] = np.float64(uenv.cfg.action_scale)
        data["max_cart_pos"] = np.float64(uenv.cfg.max_cart_pos)
        data["max_episode_length"] = np.int64(uenv.max_episode_length)
        data["soft_joint_vel_limits"] = art.data.soft_joint_vel_limits.torch.cpu().numpy()
        data["mjw_timestep"] = solver.mjw_model.opt.timestep.numpy()
        np.savez_compressed(out, **data)
        print("saved", out, {k: v.shape for k, v in data.items() if hasattr(v, "shape")})
        # quick sanity
        print(
            "MJWarp qpos == Newton joint_pos (post-step, last):",
            np.abs(data["mj_qpos"][-1] - data["obs"][-1][:, :2]).max() if args.resets == "off" else "n/a",
        )
        print(
            "nefc>0 steps:",
            int((data["nefc"] > 0).sum()),
            " max niter:",
            int(data["niter"].max()),
            " niter hist:",
            np.bincount(data["niter"].reshape(-1)),
        )
        print(
            "initial |qd| max:",
            np.abs(data["qd_init"]).max(axis=0),
            " soft vel limits:",
            data["soft_joint_vel_limits"][0],
        )
        env.close()


if __name__ == "__main__":
    main()

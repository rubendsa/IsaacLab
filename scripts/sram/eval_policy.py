# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic evaluation of a Cartpole policy in the stock Isaac Lab environment.

Every policy goes through the same protocol, whether stock RSL-RL or the SRAM megakernel trained it:

1. Resolve ``Isaac-Cartpole-Direct`` from Isaac Lab's registry with ``physics=newton_mjwarp``, then set
   ``num_envs`` and ``seed``. Agent and env seeds are the same value, as in the stock entry points.
2. Create the environment the way the stock play entry points do: kitless ``launch_simulation``,
   ``gym.make`` (``--frontend torch``), then ``RslRlVecEnvWrapper(clip_actions=agent_cfg.clip_actions)``.
   The wrapper constructor resets every env.
3. Load the policy. A checkpoint goes through ``OnPolicyRunner(env, to_dict(agent_cfg)).load(path)``,
   which is strict and also loads the optimizer, exactly like ``play.py``, then
   ``get_inference_policy()``. You can instead pass an in-memory RSL-RL actor (``MLPModel``) or any
   callable ``TensorDict -> actions``.
4. Call ``env.reset()``. Every env starts a new episode with ``episode_length_buf == 0``, so the starts
   are aligned and no random initial episode length is applied. Then run
   ``num_episodes * max_episode_length`` steps with the deterministic mean action under
   ``torch.inference_mode()``. No episode is longer than ``max_episode_length`` steps, so every env
   completes at least ``num_episodes`` episodes.
5. Record reward, done, time-out and terminated on the GPU without per-step host syncs, then compute
   all statistics offline.

Steps 2-4 are ``scripts/benchmarks/play.py`` (``isaaclab.benchmark.stepping.run_play_loop``): same env
construction, seeding order, reset and float32 return accumulation. The ``benchmark_play_equivalent``
block therefore reproduces that benchmark's ``reward`` / ``ep_length`` / ``success_rate``. Run it with
``--warmup_steps 0 --num_steps <num_steps>``, because its warm-up steps also count toward its episode
statistics.

Reported metrics:

* ``first_episodes`` (headline): the first ``num_episodes`` episodes of every env, so exactly
  ``num_envs * num_episodes`` episodes, each env weighted equally.
* ``by_episode_index``: the same split by episode index. Episode 1 starts from the ``env.reset()`` of
  step 4. Because this is the *second* reset, it already samples joint velocities: the Isaac Lab 3.0 /
  Newton zero-velocity quirk affects only the wrapper's first reset. ``initial_state`` records the
  observed start ranges.
* ``all_completed_episodes``: every episode completed in the window. Envs that fail often add more
  episodes, so this is biased toward failures. This is what the play benchmark aggregates.
* ``per_env``: completed-episode counts and per-env return statistics. ``--per_env_npz`` writes the full
  per-env arrays.

Compare policies on episode return (max ~5.0). A do-nothing policy keeps the cart inside the limits, so
it reaches length 300 and time-out fraction 1.0, with a return of about -20.

Usage::

    python scripts/sram/eval_policy.py --checkpoint <run_dir | model_XX.pt> [--num_envs 4096] [--num_episodes 2]
        [--seed 42] [--output eval.json] [--per_env_npz eval_per_env.npz]

Programmatic::

    import eval_policy  # scripts/sram on sys.path

    result = eval_policy.evaluate(checkpoint="logs/.../model_149.pt")
    result = eval_policy.evaluate(policy=actor)  # in-memory RSL-RL MLPModel
"""

from __future__ import annotations

# Warp records ``enable_backward`` when a module is created, so set it before anything defines Warp kernels
# (same as scripts/reinforcement_learning/train.py).
import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import contextlib  # noqa: E402
import importlib.metadata  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable, Sequence  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

DEFAULT_TASK = "Isaac-Cartpole-Direct"
DEFAULT_AGENT = "rsl_rl_cfg_entry_point"
DEFAULT_PHYSICS = "physics=newton_mjwarp"
PROTOCOL_VERSION = 1


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------
def resolve_checkpoint(path: str) -> str:
    """A ``model_*.pt`` path, or a run directory, resolved to its naturally last ``model_*.pt``."""
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        return path
    models = [f for f in os.listdir(path) if re.fullmatch(r"model_.*\.pt", f)]
    if not models:
        raise FileNotFoundError(f"no model_*.pt checkpoint in {path}")
    key = lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]  # noqa: E731
    return os.path.join(path, sorted(models, key=key)[-1])


def _summary(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {"count": 0}
    q = np.percentile(v, [1, 5, 25, 50, 75, 95, 99])
    return {
        "count": int(v.size),
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "min": float(v.min()),
        "p01": float(q[0]),
        "p05": float(q[1]),
        "p25": float(q[2]),
        "p50": float(q[3]),
        "p75": float(q[4]),
        "p95": float(q[5]),
        "p99": float(q[6]),
        "max": float(v.max()),
    }


def _episode_block(ret: np.ndarray, length: np.ndarray, timeout: np.ndarray, terminated: np.ndarray) -> dict:
    n = int(ret.size)
    if n == 0:
        return {"num_episodes": 0}
    return {
        "num_episodes": n,
        "return_mean": float(np.mean(ret, dtype=np.float64)),
        "return_std": float(np.std(ret.astype(np.float64), ddof=1)) if n > 1 else 0.0,
        "length_mean": float(np.mean(length, dtype=np.float64)),
        "length_std": float(np.std(length.astype(np.float64), ddof=1)) if n > 1 else 0.0,
        "timeout_fraction": float(np.mean(timeout)),
        "terminated_fraction": float(np.mean(terminated)),
        "terminated_and_timeout_fraction": float(np.mean(timeout & terminated)),
        "return": _summary(ret),
        "length": _summary(length),
    }


def _versions() -> dict[str, str | None]:
    out = {}
    for dist in ("isaaclab", "isaaclab_tasks", "isaaclab_rl", "rsl-rl-lib", "newton", "warp-lang", "mujoco-warp"):
        try:
            out[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            out[dist] = None
    out["torch"] = torch.__version__
    return out


# ------------------------------------------------------------------------------------------------
# rollout + statistics
# ------------------------------------------------------------------------------------------------
def _rollout(env, policy: Callable, num_steps: int) -> dict[str, Any]:
    """``run_play_loop`` without per-step host syncs: reset, then *num_steps* deterministic steps."""
    u = env.unwrapped
    n, dev = u.num_envs, u.device
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    ep_len_at_start = u.episode_length_buf.detach().clone()
    obs0 = obs["policy"].detach().clone()
    with torch.inference_mode():
        rew = torch.zeros(num_steps, n, device=dev, dtype=torch.float32)
        done = torch.zeros(num_steps, n, device=dev, dtype=torch.bool)
        tout = torch.zeros(num_steps, n, device=dev, dtype=torch.bool)
        term = torch.zeros(num_steps, n, device=dev, dtype=torch.bool)
        act_abs_max = torch.zeros((), device=dev)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for t in range(num_steps):
            actions = policy(obs)
            obs, reward, dones, extras = env.step(actions)
            rew[t] = torch.as_tensor(reward, device=dev).reshape(n)
            done[t] = torch.as_tensor(dones, device=dev).reshape(n).to(torch.bool)
            if "time_outs" in extras:
                tout[t] = torch.as_tensor(extras["time_outs"], device=dev).reshape(n).to(torch.bool)
            terminated = getattr(u, "reset_terminated", None)
            if isinstance(terminated, torch.Tensor):
                term[t] = terminated.reshape(n).to(torch.bool)
            else:  # frontends without a torch reset_terminated buffer
                term[t] = done[t] & ~tout[t]
            act_abs_max = torch.maximum(act_abs_max, actions.detach().abs().max())
        torch.cuda.synchronize(dev)
        wall = time.perf_counter() - t0
    return {
        "rew": rew.cpu().numpy(),
        "done": done.cpu().numpy(),
        "timeout": tout.cpu().numpy(),
        "terminated": term.cpu().numpy(),
        "ep_len_at_start": ep_len_at_start.cpu().numpy(),
        "obs0": obs0.cpu().numpy(),
        "action_abs_max": float(act_abs_max.item()),
        "rollout_wall_s": wall,
    }


def episode_statistics(trace: dict[str, np.ndarray], num_episodes: int) -> dict[str, Any]:
    """Split the recorded trace into episodes and aggregate.

    The per-env return accumulator is float32 and is reset when the env's done flag is set, exactly like
    ``run_play_loop``. The per-episode returns therefore match the play benchmark's bit for bit, and
    ``statistics.mean`` / ``stdev`` run over the same multiset of floats.
    """
    rew, done, tout, term = trace["rew"], trace["done"], trace["timeout"], trace["terminated"]
    T, N = rew.shape
    run_ret = np.zeros(N, np.float32)
    run_len = np.zeros(N, np.int64)
    k = np.zeros(N, np.int64)
    cols: dict[str, list[np.ndarray]] = {c: [] for c in ("env", "k", "t_end", "ret", "len", "timeout", "terminated")}
    bench_success: list[float] = []
    for t in range(T):
        run_ret += rew[t]
        run_len += 1
        d = np.flatnonzero(done[t])
        if d.size == 0:
            continue
        cols["env"].append(d)
        cols["k"].append(k[d].copy())
        cols["t_end"].append(np.full(d.size, t))
        cols["ret"].append(run_ret[d].copy())
        cols["len"].append(run_len[d].copy())
        cols["timeout"].append(tout[t, d].copy())
        cols["terminated"].append(term[t, d].copy())
        # play benchmark: extras["log"]["Metrics/success_rate"] (float32 mean of reset_time_outs over the envs reset
        # this step) appended once per done env
        value = float(np.float32(np.float32(tout[t, d].sum()) / np.float32(d.size)))
        bench_success.extend([value] * d.size)
        run_ret[d] = 0.0
        run_len[d] = 0
        k[d] += 1
    ep = {c: (np.concatenate(v) if v else np.zeros(0)) for c, v in cols.items()}
    ret, length = ep["ret"].astype(np.float32), ep["len"].astype(np.int64)
    timeout, terminated = ep["timeout"].astype(bool), ep["terminated"].astype(bool)

    # benchmark-equivalent aggregate over all completed episodes (statistics.* on the same Python floats)
    ret_list = [float(x) for x in ret]
    len_list = [float(x) for x in length]
    bench = {
        "reward": {
            "mean": statistics.mean(ret_list) if ret_list else 0.0,
            "std": statistics.stdev(ret_list) if len(ret_list) > 1 else 0.0,
            "peak": max(ret_list) if ret_list else 0.0,
        },
        "ep_length": {
            "mean": statistics.mean(len_list) if len_list else 0.0,
            "std": statistics.stdev(len_list) if len(len_list) > 1 else 0.0,
            "peak": max(len_list) if len_list else 0.0,
        },
        "success_rate": round(sum(bench_success) / len(bench_success), 4) if bench_success else None,
    }

    # first `num_episodes` episodes of every env (each env completes >= num_episodes in the window)
    completed = np.bincount(ep["env"].astype(np.int64), minlength=N) if ep["env"].size else np.zeros(N, np.int64)
    K = int(num_episodes)
    first = ep["k"] < K
    first_ret = np.full((N, K), np.nan, np.float32)
    first_len = np.zeros((N, K), np.int64)
    first_tout = np.zeros((N, K), bool)
    first_term = np.zeros((N, K), bool)
    e_idx, k_idx = ep["env"][first].astype(np.int64), ep["k"][first].astype(np.int64)
    first_ret[e_idx, k_idx] = ret[first]
    first_len[e_idx, k_idx] = length[first]
    first_tout[e_idx, k_idx] = timeout[first]
    first_term[e_idx, k_idx] = terminated[first]
    envs_short = int((completed < K).sum())

    per_env_mean = np.nanmean(first_ret.astype(np.float64), axis=1) if K > 0 else np.zeros(N)
    all_mean = np.zeros(N)
    np.add.at(all_mean, ep["env"].astype(np.int64), ret.astype(np.float64))
    all_mean = np.divide(all_mean, np.maximum(completed, 1))
    worst = np.argsort(per_env_mean)[:5]
    hist_vals, hist_counts = np.unique(completed, return_counts=True)
    stats = {
        "first_episodes": {
            "episodes_per_env": K,
            "envs_with_fewer_completed_episodes": envs_short,
            **_episode_block(
                first_ret[~np.isnan(first_ret)],
                first_len[~np.isnan(first_ret)],
                first_tout[~np.isnan(first_ret)],
                first_term[~np.isnan(first_ret)],
            ),
        },
        "by_episode_index": [
            {"episode_index": j, **_episode_block(first_ret[:, j], first_len[:, j], first_tout[:, j], first_term[:, j])}
            for j in range(K)
        ],
        "all_completed_episodes": _episode_block(ret, length, timeout, terminated),
        "benchmark_play_equivalent": bench,
        "per_env": {
            "completed_episodes": _summary(completed),
            "completed_episodes_histogram": {str(int(v)): int(c) for v, c in zip(hist_vals, hist_counts)},
            "first_episodes_mean_return": _summary(per_env_mean),
            "all_completed_mean_return": _summary(all_mean),
            "fraction_envs_all_first_episodes_timed_out": float(np.mean(first_tout.all(axis=1))) if K else None,
            "fraction_envs_any_first_episode_terminated": float(np.mean(first_term.any(axis=1))) if K else None,
            "worst_envs": [
                {
                    "env": int(e),
                    "first_returns": [float(x) for x in first_ret[e]],
                    "first_lengths": [int(x) for x in first_len[e]],
                    "completed": int(completed[e]),
                }
                for e in worst
            ],
        },
        "unfinished_episodes_at_end": {
            "count": int((run_len > 0).sum()),
            "length_mean": float(run_len[run_len > 0].mean()) if (run_len > 0).any() else 0.0,
        },
    }
    arrays = {
        "completed_episodes": completed,
        "first_returns": first_ret,
        "first_lengths": first_len,
        "first_timeout": first_tout,
        "first_terminated": first_term,
        "all_mean_return": all_mean,
        "episode_env": ep["env"].astype(np.int64),
        "episode_index": ep["k"].astype(np.int64),
        "episode_end_step": ep["t_end"].astype(np.int64),
        "episode_return": ret,
        "episode_length": length,
        "episode_timeout": timeout,
        "episode_terminated": terminated,
    }
    return {"stats": stats, "arrays": arrays}


# ------------------------------------------------------------------------------------------------
# evaluation entry point
# ------------------------------------------------------------------------------------------------
def evaluate(
    policy: Callable | torch.nn.Module | None = None,
    *,
    checkpoint: str | None = None,
    task: str = DEFAULT_TASK,
    num_envs: int = 4096,
    num_episodes: int = 2,
    num_steps: int | None = None,
    seed: int = 42,
    device: str = "cuda:0",
    agent: str = DEFAULT_AGENT,
    hydra_overrides: Sequence[str] = (DEFAULT_PHYSICS,),
    frontend: str = "torch",
    per_env_npz: str | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Evaluate a policy with the fixed protocol described in the module docstring.

    Args:
        policy: In-memory policy: an RSL-RL ``MLPModel`` actor (deterministic output = mean action), or any
            callable mapping the wrapper's observation ``TensorDict`` to actions ``(num_envs, num_actions)``.
        checkpoint: RSL-RL checkpoint (``model_*.pt``) or run directory, loaded exactly like ``play.py``.
        task: Gym task id.
        num_envs: Number of parallel environments.
        num_episodes: Episodes per env in the headline statistics. The rollout length is
            ``num_episodes * max_episode_length`` steps unless *num_steps* is given.
        num_steps: Explicit rollout length; must be >= ``num_episodes * max_episode_length``.
        seed: Environment seed. Also the agent seed, as in the stock entry points.
        device: Simulation and policy device.
        agent: Agent config entry point used to build the runner for checkpoints.
        hydra_overrides: Hydra overrides / preset selectors applied when resolving the task config.
        frontend: Environment frontend, ``"torch"`` (stock ``gym.make``) or ``"warp"``.
        per_env_npz: Optional path for the full per-env / per-episode arrays.
        label: Free-form label stored in the result.

    Returns:
        JSON-serialisable result dict.
    """
    if (policy is None) == (checkpoint is None):
        raise ValueError("pass exactly one of `policy` (in-memory actor) or `checkpoint`")

    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab.benchmark import capture
    from isaaclab.utils import to_dict

    from isaaclab_rl.entrypoints import common
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

    import isaaclab_tasks  # noqa: F401  (registers the tasks)
    from isaaclab_tasks.utils import resolve_task_config

    with contextlib.suppress(ImportError):
        import isaaclab_tasks_experimental  # noqa: F401  (warp frontend tasks)

    overrides = list(hydra_overrides)
    env_cfg, agent_cfg = resolve_task_config(task, agent, overrides=overrides)
    physics_backend = capture.run_config_from_env_cfg(env_cfg).physics_backend

    parser = argparse.ArgumentParser(add_help=False)
    add_launcher_args(parser)
    common.add_frontend_args(parser)
    launcher_args = parser.parse_args(["--visualizer", "none", "--device", device, "--frontend", frontend])

    ckpt_path = resolve_checkpoint(checkpoint) if checkpoint is not None else None
    result: dict[str, Any] = {"label": label, "created_at": datetime.now(timezone.utc).isoformat()}
    t_setup = time.perf_counter()
    prev_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    # torch defaults (what the play benchmark runs with); set explicitly so an earlier TF32 training run in
    # the same process cannot change the policy numerics
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with launch_simulation(env_cfg, launcher_args), contextlib.ExitStack() as cleanup:
            # identical to the play benchmark: num_envs override, agent seed -> env seed, rsl-rl version shims
            env_cfg.scene.num_envs = int(num_envs)
            agent_cfg.seed = int(seed)
            env_cfg.seed = agent_cfg.seed
            agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))

            env = common.create_isaaclab_env(task, env_cfg, launcher_args, convert_marl_to_single_agent=True)
            cleanup.callback(env.close)
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            u = env.unwrapped
            max_ep_len = int(u.max_episode_length)
            steps = int(num_steps) if num_steps is not None else int(num_episodes) * max_ep_len
            if steps < int(num_episodes) * max_ep_len:
                raise ValueError(f"num_steps={steps} < num_episodes * max_episode_length = {num_episodes * max_ep_len}")

            policy_info: dict[str, Any]
            if ckpt_path is not None:
                from rsl_rl.runners import DistillationRunner, OnPolicyRunner

                runner_cls = {"OnPolicyRunner": OnPolicyRunner, "DistillationRunner": DistillationRunner}[
                    agent_cfg.class_name
                ]
                runner = runner_cls(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
                infos = runner.load(ckpt_path)  # load_cfg=None, strict=True: actor + critic + optimizer
                pol = runner.get_inference_policy(device=u.device)
                raw = torch.load(ckpt_path, weights_only=False, map_location="cpu")
                policy_info = {
                    "source": "checkpoint",
                    "checkpoint": ckpt_path,
                    "loaded_with": f"rsl_rl.runners.{agent_cfg.class_name}.load (load_cfg=None, strict=True)",
                    "iter": raw.get("iter"),
                    "infos": infos if isinstance(infos, (dict, type(None))) else str(infos),
                    "learning_rate": raw["optimizer_state_dict"]["param_groups"][0]["lr"],
                    "action_std": [float(x) for x in raw["actor_state_dict"]["distribution.std_param"].reshape(-1)]
                    if "distribution.std_param" in raw["actor_state_dict"]
                    else None,
                }
            else:
                pol = policy
                if isinstance(pol, torch.nn.Module):
                    pol = pol.to(u.device).eval()
                policy_info = {"source": "in_memory", "type": type(policy).__name__}

            trace = _rollout(env, pol, steps)
            setup_s = time.perf_counter() - t_setup - trace["rollout_wall_s"]
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev_tf32

    out = episode_statistics(trace, num_episodes)
    obs0 = trace["obs0"]
    result.update(
        {
            "protocol": {
                "version": PROTOCOL_VERSION,
                "task": task,
                "physics": physics_backend,
                "hydra_overrides": overrides,
                "frontend": frontend,
                "num_envs": int(num_envs),
                "num_episodes": int(num_episodes),
                "num_steps": steps,
                "max_episode_length": max_ep_len,
                "step_dt": float(u.step_dt),
                "seed": int(seed),
                "device": device,
                "actions": "deterministic mean (RSL-RL inference policy), torch.inference_mode, TF32 off",
                "clip_actions": agent_cfg.clip_actions,
                "start": "env.reset() after the wrapper's reset: all episodes start aligned (episode_length_buf == 0)",
                "benchmark_play_equivalent_command": (
                    f"scripts/benchmarks/play.py --rl_library rsl_rl --task {task} --num_envs {num_envs} --seed {seed}"
                    f" --warmup_steps 0 --num_steps {steps} --checkpoint <ckpt> --visualizer none {' '.join(overrides)}"
                ),
            },
            "policy": policy_info,
            "initial_state": {
                "episode_length_buf_max": int(trace["ep_len_at_start"].max()),
                "obs_abs_max": {
                    name: float(np.abs(obs0[:, i]).max())
                    for i, name in enumerate(("cart_pos", "pole_angle", "cart_vel", "pole_vel"))
                }
                if obs0.shape[1] == 4
                else [float(x) for x in np.abs(obs0).max(axis=0)],
            },
            **out["stats"],
            "action_abs_max": trace["action_abs_max"],
            "timing_indicative": {
                "setup_s": setup_s,
                "rollout_wall_s": trace["rollout_wall_s"],
                "env_steps_per_s": steps * int(num_envs) / max(trace["rollout_wall_s"], 1e-9),
                "note": "host wall clock on a shared GPU; not a benchmark",
            },
            "versions": _versions(),
        }
    )
    if result["initial_state"]["episode_length_buf_max"] != 0:
        raise RuntimeError("episodes did not start aligned (episode_length_buf != 0 after reset)")
    if per_env_npz:
        os.makedirs(os.path.dirname(os.path.abspath(per_env_npz)), exist_ok=True)
        np.savez_compressed(per_env_npz, **out["arrays"])
        result["per_env"]["arrays_npz"] = os.path.abspath(per_env_npz)
    return result


def format_summary(result: dict[str, Any]) -> str:
    """Short human-readable summary of an :func:`evaluate` result."""
    fe, al, b = result["first_episodes"], result["all_completed_episodes"], result["benchmark_play_equivalent"]
    p, pe = result["protocol"], result["per_env"]
    done, per_env_ret = pe["completed_episodes"], pe["first_episodes_mean_return"]
    src = result["policy"].get("checkpoint") or result["policy"].get("type")
    lines = [
        f"[eval] {src}",
        f"[eval] {p['task']} / {p['physics']} / {p['frontend']}: {p['num_envs']} envs x {p['num_steps']} steps,"
        f" seed {p['seed']}, deterministic actions, aligned starts",
        f"[eval] first {fe['episodes_per_env']} episodes/env ({fe['num_episodes']} episodes): return"
        f" {fe['return_mean']:.4f} +- {fe['return_std']:.4f} (min {fe['return']['min']:.4f}),"
        f" length {fe['length_mean']:.2f}, time-out {fe['timeout_fraction']:.4f},"
        f" terminated {fe['terminated_fraction']:.4f}",
    ]
    for blk in result["by_episode_index"]:
        lines.append(
            f"[eval]   episode {blk['episode_index'] + 1}: return {blk['return_mean']:.4f} +- {blk['return_std']:.4f},"
            f" length {blk['length_mean']:.2f}, time-out {blk['timeout_fraction']:.4f}"
        )
    lines += [
        f"[eval] all {al['num_episodes']} completed episodes: return {al['return_mean']:.4f} +- {al['return_std']:.4f},"
        f" length {al['length_mean']:.2f}, time-out {al['timeout_fraction']:.4f}"
        f"  (benchmark-play equivalent: reward {b['reward']['mean']:.6f} +- {b['reward']['std']:.6f},"
        f" ep_length {b['ep_length']['mean']:.4f}, success_rate {b['success_rate']})",
        f"[eval] per env: completed episodes {done['min']:.0f}..{done['max']:.0f},"
        f" mean first-episode return p05 {per_env_ret['p05']:.4f} p50 {per_env_ret['p50']:.4f}"
        f" min {per_env_ret['min']:.4f},"
        f" all first episodes timed out in {100 * pe['fraction_envs_all_first_episodes_timed_out']:.2f}% of envs",
    ]
    return "\n".join(lines)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    return str(o)


def write_json(result: dict[str, Any], path: str) -> str:
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(result, fh, indent=1, default=_json_default)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate an RSL-RL checkpoint in the stock Isaac Lab Cartpole env (deterministic, aligned).",
        epilog="Extra positional tokens are Hydra overrides / preset selectors (default: physics=newton_mjwarp).",
    )
    parser.add_argument("--checkpoint", required=True, help="model_*.pt or an RSL-RL run directory (last model_*.pt).")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--agent", default=DEFAULT_AGENT, help="Agent config entry point used to build the runner.")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--num_episodes", type=int, default=2, help="Episodes per env (sets the rollout length).")
    parser.add_argument(
        "--num_steps", type=int, default=None, help="Explicit rollout length (>= episodes x max length)."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frontend", choices=["torch", "warp"], default="torch")
    parser.add_argument("--output", default=None, help="Write the full JSON result here (otherwise printed).")
    parser.add_argument("--per_env_npz", default=None, help="Write per-env / per-episode arrays (.npz).")
    parser.add_argument("--label", default=None)
    args, hydra = parser.parse_known_args(argv)
    unknown = [a for a in hydra if a.startswith("-")]
    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    if not any(a.startswith("physics=") for a in hydra):
        hydra = [DEFAULT_PHYSICS, *hydra]

    result = evaluate(
        checkpoint=args.checkpoint,
        task=args.task,
        num_envs=args.num_envs,
        num_episodes=args.num_episodes,
        num_steps=args.num_steps,
        seed=args.seed,
        device=args.device,
        agent=args.agent,
        hydra_overrides=hydra,
        frontend=args.frontend,
        per_env_npz=args.per_env_npz,
        label=args.label,
    )
    print(format_summary(result))
    if args.output:
        print(f"[eval] wrote {write_json(result, args.output)}")
    else:
        print(json.dumps(result, indent=1, default=_json_default))
    return 0


if __name__ == "__main__":
    sys.exit(main())

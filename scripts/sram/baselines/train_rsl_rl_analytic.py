# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train Isaac-Cartpole-Direct with the STOCK RSL-RL ``OnPolicyRunner`` on the analytic PyTorch VecEnv.

Ablation for the SRAM-resident PPO megakernel (``isaaclab_sram``): the environment is the identical
Isaac-Cartpole-Direct MDP with closed-form MuJoCo-Warp physics (``torch_cartpole_vecenv.py``), i.e. Isaac Lab's
MDP with the Newton / Isaac Lab env machinery removed, while everything on the learning side is stock: the agent
config is Isaac Lab's ``CartpoleDirectPPORunnerCfg`` resolved from the task registry, converted with
``isaaclab_rl.rsl_rl.handle_deprecated_rsl_rl_cfg`` and turned into an ``OnPolicyRunner`` by
``isaaclab_rl.rsl_rl.create_rsl_rl_runner`` -- the same sequence, seeding, log layout and Torch backend flags
(TF32 matmul on) as ``isaaclab_rl/entrypoints/backends/train_rsl_rl.py``.

Environments (``--env``):

* ``analytic`` (default) with ``--mode eager`` (plain PyTorch ops) or ``--mode compiled`` (the whole env step
  fused by ``torch.compile``; ``--compile_mode max-autotune-no-cudagraphs`` | ``reduce-overhead`` | ...),
* ``isaaclab``: the real Newton / MJWarp Isaac Lab env (``--frontend torch|warp``) behind the stock
  ``RslRlVecEnvWrapper`` -- the stock training run, but with this script's instrumentation, so that the per-phase
  breakdown (``--breakdown``) is apples-to-apples.

``--policy_compile_mode`` sets the stock RSL-RL ``torch_compile_mode`` knob (actor / critic ``torch.compile``).

Per-iteration metrics are captured from the runner's own logger call (collection / learning time as measured by
the runner with ``time.time()``, steps/s, Train/mean_reward, ...) and written to ``<log_dir>/ablation_metrics.json``
next to the stock TensorBoard events and checkpoints (``model_<it>.pt`` loads in ``scripts/benchmarks/play.py``).
``--breakdown`` additionally wraps the runner's phases (act, env.step, check_nan, process_env_step, logger,
compute_returns, update) with ``torch.cuda.synchronize()`` timers; this perturbs the timings slightly (two extra
syncs per wrapped call) and is therefore off by default.

All timings taken on a shared GPU are indicative only.

Usage (run from a scratch directory: logs go to ``$PWD/logs/rsl_rl/<experiment>/<timestamp>_<run_name>``)::

    OMP_NUM_THREADS=1 .venv/bin/python scripts/sram/baselines/train_rsl_rl_analytic.py --mode eager
    OMP_NUM_THREADS=1 .venv/bin/python scripts/sram/baselines/train_rsl_rl_analytic.py --mode compiled
    OMP_NUM_THREADS=1 .venv/bin/python scripts/sram/baselines/train_rsl_rl_analytic.py --env isaaclab --breakdown
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
_dont_write_bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # keep the repo free of __pycache__
import torch_cartpole_vecenv as tcv  # noqa: E402  (also puts source/isaaclab_sram on sys.path)

sys.dont_write_bytecode = _dont_write_bytecode

COLLECT_PHASES = ("act", "env_step", "check_nan", "process_env_step", "logger_step")
LEARN_PHASES = ("compute_returns", "update")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stock RSL-RL PPO on the analytic Isaac-Cartpole-Direct VecEnv.")
    ap.add_argument("--task", default="Isaac-Cartpole-Direct")
    ap.add_argument("--agent", default="rsl_rl_cfg_entry_point")
    ap.add_argument("--env", choices=["analytic", "isaaclab"], default="analytic")
    ap.add_argument("--mode", choices=["eager", "compiled"], default="eager", help="analytic env execution mode")
    ap.add_argument(
        "--compile_mode",
        choices=tcv.COMPILE_MODES,
        default="max-autotune-no-cudagraphs",
        help="torch.compile mode of the fused env step (--mode compiled)",
    )
    ap.add_argument(
        "--frontend", choices=["torch", "warp"], default="torch", help="Isaac Lab env frontend (--env isaaclab)"
    )
    ap.add_argument(
        "--policy_compile_mode",
        choices=["none", "default", "max-autotune-no-cudagraphs"],
        default="none",
        help="stock RSL-RL torch_compile_mode for actor / critic (Isaac Lab default: none)",
    )
    ap.add_argument("--num_envs", type=int, default=None)
    ap.add_argument("--max_iterations", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None, help="env device (the runner uses the agent cfg's device)")
    ap.add_argument("--experiment_name", default=None)
    ap.add_argument("--run_name", default=None, help="suffix of the run directory (default: derived from the variant)")
    ap.add_argument(
        "--no_first_reset_quirk",
        action="store_true",
        help="sample velocities on the very first reset too (Isaac Lab 3.0 + Newton zeroes them)",
    )
    ap.add_argument("--breakdown", action="store_true", help="synchronized per-phase timing of the runner loop")
    ap.add_argument(
        "--profile_iteration",
        type=int,
        default=None,
        help="torch.profiler one iteration (collection + learning): CUDA kernels and kernel time per phase",
    )
    ap.add_argument(
        "--eval_steps",
        type=int,
        default=600,
        help="after training, roll out the deterministic policy for this many steps on a fresh analytic "
        "env (play-benchmark episode bookkeeping); 0 disables",
    )
    ap.add_argument("--eval_num_envs", type=int, default=4096)
    ap.add_argument("--steady_state_from", type=int, default=3, help="first iteration counted in steady-state medians")
    return ap.parse_args(argv)


def variant_name(args) -> str:
    if args.env == "isaaclab":
        name = f"isaaclab_{args.frontend}"
    else:
        name = "analytic_eager" if args.mode == "eager" else f"analytic_compiled_{args.compile_mode}"
    if args.policy_compile_mode != "none":
        name += f"_policy-{args.policy_compile_mode}"
    if args.breakdown:
        name += "_breakdown"
    if args.profile_iteration is not None:
        name += "_profile"
    return name


# ------------------------------------------------------------------------------------------------
# instrumentation (observes the stock runner; does not change what it computes)
# ------------------------------------------------------------------------------------------------
class Instrumentation:
    """Captures the runner's per-iteration log values and, optionally, a synchronized phase breakdown or a
    ``torch.profiler`` trace of one iteration."""

    def __init__(self, runner, env, breakdown: bool, profile_iteration: int | None = None):
        self.runner = runner
        self.rows: list[dict] = []
        self.breakdown = breakdown
        self.profile_iteration = profile_iteration
        self.profile: dict | None = None
        self._prof = None
        self.acc = {k: 0.0 for k in COLLECT_PHASES + LEARN_PHASES}
        self.calls = {k: 0 for k in COLLECT_PHASES + LEARN_PHASES}
        self._orig_log = runner.logger.log
        runner.logger.log = self._log
        if breakdown or profile_iteration is not None:
            import rsl_rl.runners.on_policy_runner as opr

            runner.alg.act = self._timed("act", runner.alg.act)
            env.step = self._timed("env_step", env.step)  # instance attribute shadows the method
            opr.check_nan = self._timed("check_nan", opr.check_nan)
            runner.alg.process_env_step = self._timed("process_env_step", runner.alg.process_env_step)
            runner.logger.process_env_step = self._timed("logger_step", runner.logger.process_env_step)
            runner.alg.compute_returns = self._timed("compute_returns", runner.alg.compute_returns)
            runner.alg.update = self._timed("update", runner.alg.update)

    def _timed(self, name, fn):
        if not self.breakdown:  # profiling only: label the phase, no synchronisation

            def labelled(*a, **k):
                with torch.profiler.record_function(f"phase::{name}"):
                    return fn(*a, **k)

            return labelled

        def wrapper(*a, **k):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.profiler.record_function(f"phase::{name}"):
                out = fn(*a, **k)
            torch.cuda.synchronize()
            self.acc[name] += time.perf_counter() - t0
            self.calls[name] += 1
            return out

        return wrapper

    def _start_profile(self):
        from torch.profiler import ProfilerActivity, profile

        torch.cuda.synchronize()
        self._prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
        self._prof.__enter__()
        self._prof_t0 = time.perf_counter()

    def _stop_profile(self, collect_time, learn_time):
        torch.cuda.synchronize()
        wall = time.perf_counter() - self._prof_t0
        self._prof.__exit__(None, None, None)
        evts = self._prof.events()
        cuda = torch.autograd.DeviceType.CUDA
        # GPU-timeline spans of our record_function ranges (kineto gpu_user_annotation events) ...
        spans = sorted(
            (e.time_range.start, e.time_range.end, e.name[7:])
            for e in evts
            if e.device_type == cuda and e.name.startswith("phase::")
        )
        # ... and the real device activities (kernels, memcpy / memset)
        kernels = [e for e in evts if e.device_type == cuda and not e.name.startswith("phase::")]
        per = {name: {"cuda_kernels": 0, "kernel_time_ms": 0.0} for _, _, name in spans}
        per["unattributed"] = {"cuda_kernels": 0, "kernel_time_ms": 0.0}
        starts = [s for s, _, _ in spans]
        # phases run back to back on one stream, so a kernel belongs to the span that contains its start time;
        # this also catches CUDA-graph kernels (Newton / MJWarp) and backward kernels launched by the autograd thread
        for k in kernels:
            i = bisect.bisect_right(starts, k.time_range.start) - 1
            name = spans[i][2] if i >= 0 and k.time_range.start <= spans[i][1] else "unattributed"
            per[name]["cuda_kernels"] += 1
            per[name]["kernel_time_ms"] += k.time_range.elapsed_us() / 1e3
        self.profile = {
            "iteration": self.profile_iteration,
            "runner_collect_s": collect_time,
            "runner_learn_s": learn_time,
            "profiled_wall_s": wall,
            "cuda_kernels": len(kernels),
            "kernel_time_ms": sum(v["kernel_time_ms"] for v in per.values()),
            "phases": per,
            "note": "wall time is inflated by the profiler; kernel counts / kernel times are the useful numbers. "
            "Device activities are attributed to the phase whose GPU-timeline span contains their start.",
        }
        self._prof = None

    def _log(
        self,
        it,
        start_it,
        total_it,
        collect_time,
        learn_time,
        loss_dict,
        learning_rate,
        action_std,
        rnd_weight,
        *a,
        **k,
    ):
        if self._prof is not None and it == self.profile_iteration:
            self._stop_profile(collect_time, learn_time)
        lg = self.runner.logger
        n = self.runner.cfg["num_steps_per_env"] * lg.num_envs
        row = {
            "it": it,
            "collect_s": collect_time,
            "learn_s": learn_time,
            "steps_per_s": int(n / (collect_time + learn_time)),
            "mean_reward": statistics.mean(lg.rewbuffer) if len(lg.rewbuffer) else None,
            "mean_episode_length": statistics.mean(lg.lenbuffer) if len(lg.lenbuffer) else None,
            "learning_rate": float(learning_rate),
            "action_std": float(action_std.mean().item()),
        }
        row.update({f"loss_{key}": float(v) for key, v in loss_dict.items()})
        if lg.ep_extras and "Metrics/success_rate" in lg.ep_extras[-1]:
            row["success_rate"] = float(torch.as_tensor(lg.ep_extras[-1]["Metrics/success_rate"]).float().mean())
        if self.breakdown:
            row.update({f"t_{key}": v for key, v in self.acc.items()})
            row["t_collect_other"] = collect_time - sum(self.acc[key] for key in COLLECT_PHASES)
            row["t_learn_other"] = learn_time - sum(self.acc[key] for key in LEARN_PHASES)
            self.acc = dict.fromkeys(self.acc, 0.0)
        self.rows.append(row)
        out = self._orig_log(
            it, start_it, total_it, collect_time, learn_time, loss_dict, learning_rate, action_std, rnd_weight, *a, **k
        )
        if self.profile_iteration is not None and it == self.profile_iteration - 1:
            self._start_profile()  # covers the next iteration's collection + learning (not its logging)
        return out

    def summary(self, steady_from: int, steps_per_iter: int) -> dict:
        rows = [r for r in self.rows if r["it"] >= steady_from] or self.rows

        def med(key):
            return float(np.median([r[key] for r in rows]))

        out = {
            "iterations": len(self.rows),
            "steady_state_from_iteration": steady_from,
            "median_collect_ms": 1e3 * med("collect_s"),
            "median_learn_ms": 1e3 * med("learn_s"),
            "median_iteration_ms": 1e3 * float(np.median([r["collect_s"] + r["learn_s"] for r in rows])),
            "median_steps_per_s": med("steps_per_s"),
            "first_iteration_collect_s": self.rows[0]["collect_s"],
            "first_iteration_learn_s": self.rows[0]["learn_s"],
            "total_collect_plus_learn_s": float(sum(r["collect_s"] + r["learn_s"] for r in self.rows)),
            "env_steps": steps_per_iter * len(self.rows),
        }
        rew = [(r["it"], r["mean_reward"]) for r in self.rows if r["mean_reward"] is not None]
        if rew:
            out["final_mean_reward"] = rew[-1][1]
            out["max_mean_reward"] = max(v for _, v in rew)
            out["final_mean_episode_length"] = self.rows[-1]["mean_episode_length"]
            for thr in (4.0, 4.5, 4.8, 4.9):
                hit = next((i for i, v in rew if v >= thr), None)
                out[f"first_iteration_mean_reward_ge_{thr}"] = hit
            out["mean_reward_at"] = {str(i): v for i, v in rew if i in (25, 50, 75, 100, 125, 149)}
        if self.breakdown:
            keys = [f"t_{k}" for k in COLLECT_PHASES + LEARN_PHASES] + ["t_collect_other", "t_learn_other"]
            out["median_breakdown_ms_per_iteration"] = {k[2:]: 1e3 * med(k) for k in keys}
            ns = self.runner.cfg["num_steps_per_env"]
            out["median_breakdown_ms_per_env_step"] = {k[2:]: 1e3 * med(k) / ns for k in keys[: len(COLLECT_PHASES)]}
        return out


# ------------------------------------------------------------------------------------------------
# main (mirrors isaaclab_rl/entrypoints/backends/train_rsl_rl.py)
# ------------------------------------------------------------------------------------------------
def main(argv=None) -> dict:
    args = parse_args(argv)
    from isaaclab_rl.entrypoints.common import scoped_torch_backend_flags

    # the stock entrypoint wraps the whole run in these Torch backend flags (TF32 matmul ON)
    with scoped_torch_backend_flags(
        cuda_matmul_allow_tf32=True, cudnn_allow_tf32=True, cudnn_deterministic=False, cudnn_benchmark=True
    ):
        return _run(args)


def _run(args) -> dict:
    from isaaclab_rl.entrypoints.common import dump_train_configs, write_run_manifest
    from isaaclab_rl.rsl_rl import check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

    import isaaclab_tasks  # noqa: F401  (registers the tasks)
    from isaaclab_tasks.utils import resolve_task_config

    installed_version = check_rsl_rl_version()
    env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, overrides=["physics=newton_mjwarp"])
    # cli_args_rsl_rl.update_rsl_rl_cfg
    if args.seed is not None:
        agent_cfg.seed = args.seed
    if args.experiment_name is not None:
        agent_cfg.experiment_name = args.experiment_name
    agent_cfg.run_name = args.run_name or variant_name(args)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    # common.apply_env_overrides
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if args.device is not None:
        env_cfg.sim.device = args.device
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.policy_compile_mode != "none":
        agent_cfg.torch_compile_mode = args.policy_compile_mode
    env_cfg.seed = agent_cfg.seed

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, run_name)
    print(f"[INFO] Logging experiment in directory: {log_dir}")
    write_run_manifest(log_dir, library="rsl_rl", task=args.task, metadata={"agent": args.agent})

    with contextlib.ExitStack() as stack:
        t_env = time.perf_counter()
        if args.env == "analytic":
            env = tcv.CartpoleTorchVecEnv.from_isaaclab_cfg(
                env_cfg,
                device=env_cfg.sim.device,
                mode=args.mode,
                compile_mode=args.compile_mode,
                clip_actions=agent_cfg.clip_actions,
                first_reset_zero_velocity=not args.no_first_reset_quirk,
            )
        else:
            env = _make_isaaclab_env(args, env_cfg, agent_cfg, stack)
        env_creation_s = time.perf_counter() - t_env
        steps_per_iter = env.num_envs * agent_cfg.num_steps_per_env

        runner = create_rsl_rl_runner(env, agent_cfg, log_dir=log_dir)
        runner.add_git_repo_to_log(__file__)
        dump_train_configs(log_dir, env_cfg, agent_cfg)
        inst = Instrumentation(runner, env, args.breakdown, args.profile_iteration)

        t0 = time.time()
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=agent_cfg.init_at_random_ep_len
        )
        train_wall_s = time.time() - t0
        print(f"Training time: {round(train_wall_s, 2)} seconds")

        summary = inst.summary(args.steady_state_from, steps_per_iter)
        evaluation = None
        if args.eval_steps > 0:
            policy = runner.get_inference_policy(device=agent_cfg.device)
            eval_env = tcv.CartpoleTorchVecEnv.from_isaaclab_cfg(
                env_cfg, num_envs=args.eval_num_envs, device=env_cfg.sim.device, mode="eager"
            )
            evaluation = tcv.evaluate_policy(eval_env, policy, args.eval_steps)
            evaluation.update(
                {
                    "num_envs": args.eval_num_envs,
                    "num_steps": args.eval_steps,
                    "env": "analytic (eager)",
                    "actions": "deterministic (policy mean)",
                }
            )

    result = {
        "variant": variant_name(args),
        "log_dir": log_dir,
        "checkpoint": os.path.join(log_dir, f"model_{runner.current_learning_iteration}.pt"),
        "config": {
            "task": args.task,
            "env": args.env,
            "mode": args.mode if args.env == "analytic" else None,
            "compile_mode": args.compile_mode if (args.env == "analytic" and args.mode == "compiled") else None,
            "frontend": args.frontend if args.env == "isaaclab" else None,
            "policy_compile_mode": args.policy_compile_mode,
            "num_envs": env.num_envs,
            "num_steps_per_env": agent_cfg.num_steps_per_env,
            "max_iterations": agent_cfg.max_iterations,
            "seed": agent_cfg.seed,
            "breakdown": args.breakdown,
            "first_reset_zero_velocity": not args.no_first_reset_quirk,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        },
        "env_creation_s": env_creation_s,
        "env_compile_s": getattr(env, "compile_time_s", None),
        "train_wall_s": train_wall_s,
        "summary": summary,
        "evaluation": evaluation,
        "profile": inst.profile,
        "rows": inst.rows,
        "host": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "note": "shared GPU, timings indicative only",
        },
    }
    with open(os.path.join(log_dir, "ablation_metrics.json"), "w") as fh:
        json.dump(result, fh, indent=1)
    _print_summary(result)
    return result


def _make_isaaclab_env(args, env_cfg, agent_cfg, stack: contextlib.ExitStack):
    """The stock Newton / MJWarp env behind the stock RslRlVecEnvWrapper (as train_rsl_rl.py builds it)."""
    import gymnasium as gym

    from isaaclab.app import launch_simulation

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    env_cfg.sim.visualizer_cfgs = []
    stack.enter_context(
        launch_simulation(env_cfg, {"visualizer": ["none"], "device": env_cfg.sim.device, "livestream": 0})
    )
    if args.frontend == "torch":
        env = gym.make(args.task, cfg=env_cfg)
    else:
        from isaaclab_experimental.envs.frontend import WarpFrontend

        env = WarpFrontend.build_env(env_cfg, args.task)
    stack.callback(env.close)
    return RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)


def _print_summary(result: dict):
    s = result["summary"]
    print("=" * 100)
    print(
        f"[ablation] {result['variant']}  ({s['iterations']} iterations, {result['config']['num_envs']} envs; "
        f"shared GPU -> indicative only)"
    )
    print(
        f"  steady state (it >= {s['steady_state_from_iteration']}, medians): collection "
        f"{s['median_collect_ms']:.1f} ms + learning {s['median_learn_ms']:.1f} ms = "
        f"{s['median_iteration_ms']:.1f} ms/iteration, {s['median_steps_per_s']:.0f} steps/s"
    )
    print(
        f"  iteration 0: collection {s['first_iteration_collect_s']:.3f} s, learning "
        f"{s['first_iteration_learn_s']:.3f} s; env creation {result['env_creation_s']:.1f} s (compile "
        f"{result['env_compile_s'] or 0:.1f} s); sum of collect+learn {s['total_collect_plus_learn_s']:.1f} s"
    )
    if "final_mean_reward" in s:
        print(
            f"  Train/mean_reward: final {s['final_mean_reward']:.3f}, max {s['max_mean_reward']:.3f}; first "
            f"iteration >= 4.5: {s['first_iteration_mean_reward_ge_4.5']}, >= 4.8: "
            f"{s['first_iteration_mean_reward_ge_4.8']}, >= 4.9: {s['first_iteration_mean_reward_ge_4.9']}"
        )
        print(f"  mean reward at iterations: {json.dumps({k: round(v, 3) for k, v in s['mean_reward_at'].items()})}")
    if "median_breakdown_ms_per_iteration" in s:
        print(
            "  synchronized breakdown (median ms per iteration): "
            + ", ".join(f"{k} {v:.2f}" for k, v in s["median_breakdown_ms_per_iteration"].items())
        )
    if result.get("profile"):
        p = result["profile"]
        print(
            f"  profiled iteration {p['iteration']}: {p['cuda_kernels']} CUDA kernels, {p['kernel_time_ms']:.2f} ms "
            f"kernel time (runner: collection {1e3 * p['runner_collect_s']:.1f} ms + learning "
            f"{1e3 * p['runner_learn_s']:.1f} ms, profiler-inflated)"
        )
        print(
            "    per phase (kernels / kernel ms): "
            + ", ".join(f"{k} {v['cuda_kernels']} / {v['kernel_time_ms']:.2f}" for k, v in p["phases"].items())
        )
    if result["evaluation"]:
        e = result["evaluation"]
        print(
            f"  deterministic evaluation on the analytic env ({e['num_envs']} envs x {e['num_steps']} steps): "
            f"return {e.get('return_mean', float('nan')):.3f} +- {e.get('return_std', float('nan')):.3f} over "
            f"{e['episodes']} episodes, length {e.get('episode_length_mean', float('nan')):.1f}, "
            f"time-out fraction {e.get('timeout_fraction', float('nan')):.3f}"
        )
    print(f"  metrics: {os.path.join(result['log_dir'], 'ablation_metrics.json')}")
    print(f"  checkpoint: {result['checkpoint']}")
    print("=" * 100)


if __name__ == "__main__":
    main()

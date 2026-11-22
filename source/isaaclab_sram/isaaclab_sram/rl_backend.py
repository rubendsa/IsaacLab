# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``--rl_library sram`` backend for Isaac Lab's unified RL entry point (``scripts/reinforcement_learning/train.py``).

The backend trains ``Isaac-Cartpole-Direct`` with the SRAM-resident PPO megakernel instead of the
simulator plus RSL-RL. The result is a normal RSL-RL run directory (``run.json``, ``params/*.yaml``,
``model_<it>.pt``, TensorBoard events), so the stock ``play.py`` entry points load it with
``--rl_library rsl_rl``.

**Registration without editing isaaclab_rl.** ``isaaclab_rl.entrypoints.dispatch.run_cli`` reads
``_BACKEND_MODULES[action]`` only after ``load_external_tasks()``, at call time. Adding an entry before
the CLI runs is enough. :func:`register` does this, and ``scripts/sram/train_sram.py --via-isaaclab-cli``
wraps it for the command line: it registers the backend and then runs the unmodified stock
``scripts/reinforcement_learning/train.py``::

    python scripts/sram/train_sram.py --via-isaaclab-cli --rl_library sram --task Isaac-Cartpole-Direct \\
        --num_envs 4096 --max_iterations 150 physics=newton_mjwarp

The run goes to ``logs/rsl_rl/<experiment_name>/<timestamp>_sram``, the stock RSL-RL log root, with a
``library: rsl_rl`` manifest. ``play.py --rl_library rsl_rl --checkpoint latest`` therefore picks it up.
``--log_root`` overrides the location.

The command line is the stock RSL-RL one: common train args, ``--experiment_name``, ``--run_name``,
launcher args, and Hydra preset tokens such as ``physics=newton_mjwarp``. Options that need a simulator
or a different learner are rejected: ``--video``, ``--distributed``, ``--checkpoint`` resume and
``--capture_env_sensors``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from typing import Any

BACKEND_NAME = "sram"
SUPPORTED_TASKS = ("Isaac-Cartpole-Direct",)
DEFAULT_AGENT_ENTRY_POINT = "rsl_rl_cfg_entry_point"
SUMMARY_FILENAME = "sram_summary.json"


# ------------------------------------------------------------------------------------------------
# registration
# ------------------------------------------------------------------------------------------------
def register() -> None:
    """Add ``--rl_library sram`` to Isaac Lab's unified ``train`` entry point (process-local, idempotent)."""
    from isaaclab_rl.entrypoints import dispatch

    dispatch._BACKEND_MODULES["train"].setdefault(BACKEND_NAME, __name__)


# ------------------------------------------------------------------------------------------------
# configuration checks
# ------------------------------------------------------------------------------------------------
def check_supported(task_name: str, env_cfg: Any, agent_cfg: Any) -> None:
    """Raise ``ValueError`` unless the resolved configs describe what the megakernel implements."""
    task = task_name.split(":")[-1]
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"the SRAM megakernel implements {SUPPORTED_TASKS} only, not {task!r}")
    physics = env_cfg.sim.physics
    solver = getattr(physics, "solver_cfg", None)
    if type(physics).__name__ != "NewtonCfg" or type(solver).__name__ != "MJWarpSolverCfg":
        raise ValueError(
            f"the megakernel integrates the Newton / MuJoCo-Warp model (physics=newton_mjwarp); the resolved physics"
            f" config is {type(physics).__name__}/{type(solver).__name__}"
        )
    if getattr(solver, "integrator", "implicitfast") != "implicitfast" or int(getattr(physics, "num_substeps", 1)) != 1:
        raise ValueError("the megakernel implements MuJoCo-Warp 'implicitfast' with num_substeps=1 only")
    if int(env_cfg.observation_space) != 4 or int(env_cfg.action_space) != 1:
        raise ValueError("the megakernel is specialised for 4 observations and 1 action")
    if not str(env_cfg.sim.device).startswith("cuda"):
        raise ValueError(f"the megakernel runs on a CUDA device, got {env_cfg.sim.device!r}")
    if agent_cfg.class_name != "OnPolicyRunner" or agent_cfg.algorithm.class_name != "PPO":
        raise ValueError("the megakernel implements RSL-RL OnPolicyRunner + PPO only")
    if agent_cfg.clip_actions is not None:
        raise ValueError("action clipping is not implemented in the megakernel (clip_actions must be None)")
    alg = agent_cfg.algorithm
    if str(getattr(alg, "optimizer", "adam")).lower() != "adam":
        raise ValueError("the megakernel implements the Adam optimizer only")
    for name in ("rnd_cfg", "symmetry_cfg"):
        if getattr(alg, name, None) is not None:
            raise ValueError(f"algorithm.{name} is not implemented in the megakernel")
    if getattr(alg, "use_mixed_precision", False):
        raise ValueError("the megakernel computes in fp32; use_mixed_precision must be False")
    for model in (agent_cfg.actor, agent_cfg.critic):
        if getattr(model, "class_name", "MLPModel") != "MLPModel":
            raise ValueError("actor and critic must be rsl_rl MLPModel")
    # hidden dims, activation, obs normalisation, std type, clipped value loss, per-mini-batch advantage
    # normalisation are validated by PPOConfig.from_rsl_rl_cfg


# ------------------------------------------------------------------------------------------------
# logging helpers
# ------------------------------------------------------------------------------------------------
_TB_TAGS = {  # RSL-RL Logger tags (rsl_rl/utils/logger.py) -> megakernel statistics
    "Loss/value": "value_loss",
    "Loss/surrogate": "surrogate_loss",
    "Loss/entropy": "entropy",
    "Loss/learning_rate": "learning_rate",
    "Policy/mean_std": "action_std",
    "Train/mean_reward": "mean_episode_return",
    "Train/mean_episode_length": "mean_episode_length",
    "Metrics/success_rate": "timeout_fraction",
    "Sram/kl": "kl",
    "Sram/grad_norm_actor": "grad_norm_actor",
    "Sram/grad_norm_critic": "grad_norm_critic",
    "Sram/completed_episodes": "episodes",
    "Sram/adv_mean": "adv_mean",
    "Sram/adv_std": "adv_std",
    "Sram/mean_step_reward": "mean_step_reward",
}

LOG_HEADER = (
    "[SRAM] per-iteration log (RSL-RL quantities): return / length = mean over the episodes that finished in the\n"
    "       iteration (RSL-RL prints a 100-episode moving window), eps = finished episodes, timeout = fraction of\n"
    "       them that timed out, losses = means over the epochs x mini-batches, lr = KL-adaptive LR after the update,\n"
    "       std = action std, steps/s = env steps / GPU time of the launch (shared GPU: indicative only)."
)


def _with_derived(stats: dict) -> dict:
    import numpy as np

    out = dict(stats)
    out["timeout_fraction"] = out["timed_out"] / np.maximum(out["episodes"], 1.0)
    return out


def format_iteration(it: int, total: int, s: dict, j: int, steps_per_s: float) -> str:
    """One RSL-RL-style log line for iteration *it* (row *j* of the stats table *s*)."""
    return (
        f"[SRAM] it {it:4d}/{total}  return {s['mean_episode_return'][j]:8.4f}  len {s['mean_episode_length'][j]:6.1f}"
        f"  eps {int(s['episodes'][j]):5d}  timeout {s['timeout_fraction'][j]:5.3f}"
        f" | value {s['value_loss'][j]:.5f}  surrogate {s['surrogate_loss'][j]:+.5f}  entropy {s['entropy'][j]:+.4f}"
        f"  kl {s['kl'][j]:.4f}  lr {s['learning_rate'][j]:.2e}  std {s['action_std'][j]:.3f}"
        f" | {steps_per_s:.3e} steps/s"
    )


def _tensorboard_writer(log_dir: str):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:  # tensorboard missing: keep training, skip the event file
        print("[SRAM] tensorboard is not installed; no event file is written")
        return None
    return SummaryWriter(log_dir=log_dir, flush_secs=10)


def _jsonable(x: Any) -> Any:
    import numpy as np

    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(float(x)) else str(float(x))
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


# ------------------------------------------------------------------------------------------------
# training run (shared by the --rl_library sram backend and scripts/sram/train_sram.py)
# ------------------------------------------------------------------------------------------------
def train_and_export(
    task_name: str,
    env_cfg: Any,
    agent_cfg: Any,
    log_dir: str,
    *,
    iters_per_launch: int | None = None,
    num_ctas: int | None = None,
    storage: str = "auto",
    tensorboard: bool = True,
    quiet: bool = False,
    agent_entry_point: str = DEFAULT_AGENT_ENTRY_POINT,
) -> dict:
    """Train with the megakernel from Isaac Lab configs, log per iteration, export and verify an RSL-RL run dir.

    Uses ``env_cfg.scene.num_envs``, ``env_cfg.sim.device``, ``agent_cfg.seed``,
    ``agent_cfg.max_iterations`` and ``agent_cfg.init_at_random_ep_len``. The MDP comes from
    :meth:`CartpoleTask.from_env_cfg`, and the PPO hyper-parameters from
    :meth:`PPOConfig.from_rsl_rl_cfg`.

    Returns:
        The summary dict, also written to ``<log_dir>/sram_summary.json``.
    """
    import numpy as np
    import torch

    from . import export
    from .cartpole import CartpoleTask, PPOConfig
    from .megakernel import SramPPOCartpole

    check_supported(task_name, env_cfg, agent_cfg)
    task = CartpoleTask.from_env_cfg(env_cfg)
    ppo = PPOConfig.from_rsl_rl_cfg(agent_cfg)
    ppo.init_at_random_ep_len = bool(agent_cfg.init_at_random_ep_len)
    num_envs, seed, device = int(env_cfg.scene.num_envs), int(agent_cfg.seed), str(env_cfg.sim.device)
    total = int(ppo.max_iterations)
    if total < 1:
        raise ValueError("max_iterations must be >= 1")
    per = int(iters_per_launch or total)

    log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    export.write_run_manifest(
        log_dir, task=task_name, agent_entry_point=agent_entry_point, metadata={"trainer": "isaaclab_sram"}
    )
    print(f"[SRAM] logging run in: {log_dir}")

    t0 = time.perf_counter()
    trainer = SramPPOCartpole(
        task, ppo, num_envs=num_envs, seed=seed, num_ctas=num_ctas, storage=storage, device=device
    )
    build_s = time.perf_counter() - t0
    kernel = trainer.describe()
    print(
        f"[SRAM] {task_name}: {num_envs} envs, seed {seed}, {total} iterations x {ppo.num_steps_per_env} steps;"
        f" kernel: {kernel['num_ctas']} CTAs x {kernel['threads_per_cta']} threads, rollout storage in"
        f" {kernel['rollout_storage']}, {kernel['dynamic_smem_bytes_per_cta']} B SMEM/CTA,"
        f" {kernel['registers_per_thread']} regs/thread (built in {build_s:.2f} s)"
    )
    writer = _tensorboard_writer(log_dir) if tensorboard else None
    if not quiet:
        print(LOG_HEADER)

    gpu_s, wall_s, launches, done = 0.0, 0.0, [], 0
    steps_per_iter = num_envs * ppo.num_steps_per_env
    while done < total:
        k = min(per, total - done)
        r = trainer.train(k)
        gpu_s += r["gpu_time_s"]
        wall_s += r["wall_time_s"]
        launches.append({"first_iteration": r["first_iteration"], "iterations": k, "gpu_time_s": r["gpu_time_s"]})
        s = _with_derived(r["stats"])
        sps = k * steps_per_iter / max(r["gpu_time_s"], 1e-12)
        for j in range(k):
            it = r["first_iteration"] + j
            if not quiet:
                print(format_iteration(it, total, s, j, sps))
            if writer is not None:
                for tag, key in _TB_TAGS.items():
                    writer.add_scalar(tag, float(s[key][j]), it)
                writer.add_scalar("Perf/total_fps", sps, it)
                writer.add_scalar("Perf/iteration_time", r["gpu_time_s"] / k, it)
        done += k
    if writer is not None:
        writer.close()

    steps = total * steps_per_iter
    timing = {
        "build_s": build_s,
        "gpu_time_s": gpu_s,
        "wall_time_s": wall_s,
        "launches": len(launches),
        "iterations_per_launch": per,
        "env_steps": steps,
        "env_steps_per_s_gpu": steps / max(gpu_s, 1e-12),
        "ms_per_iteration_gpu": 1e3 * gpu_s / total,
        "note": "CUDA-event time on a shared GPU: indicative only, not a benchmark",
    }
    exp = export.export_trainer(
        trainer,
        log_dir,
        task=task_name,
        env_cfg=env_cfg,
        agent_cfg=agent_cfg,
        agent_entry_point=agent_entry_point,
        write_manifest=False,
        verify=True,
        infos={"gpu_time_s": gpu_s, "env_steps": steps},
    )
    st = _with_derived(trainer.stats_table(0, trainer.iteration))
    last = slice(max(0, total - 10), total)
    final = {
        "iteration": total - 1,
        "mean_episode_return": float(st["mean_episode_return"][-1]),
        "mean_episode_length": float(st["mean_episode_length"][-1]),
        "timeout_fraction": float(st["timeout_fraction"][-1]),
        "mean_episode_return_last10_iterations": float(
            st["episode_return_sum"][last].sum() / max(st["episodes"][last].sum(), 1.0)
        ),
        "learning_rate": float(trainer.scal[0].item()),
        "action_std": float(st["action_std"][-1]),
        "adam_step": exp["adam_step"],
    }
    first_good = np.flatnonzero(st["mean_episode_return"] >= 4.9)
    summary = {
        "task": task_name,
        "trainer": "isaaclab_sram.megakernel.SramPPOCartpole",
        "log_dir": log_dir,
        "checkpoint": exp["checkpoint"],
        "num_envs": num_envs,
        "seed": seed,
        "max_iterations": total,
        "device": device,
        "first_iteration_with_mean_return_ge_4.9": int(first_good[0]) if first_good.size else None,
        "final": final,
        "timing_indicative": timing,
        "launches": launches,
        "kernel": kernel,
        "task_cfg": asdict(task),
        "ppo_cfg": asdict(ppo),
        "export": exp,
        "versions": {"torch": torch.__version__, "rsl-rl-lib": importlib.metadata.version("rsl-rl-lib")},
        "stats": {k: v for k, v in st.items()},
    }
    summary = _jsonable(summary)
    with open(os.path.join(log_dir, SUMMARY_FILENAME), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(
        f"[SRAM] trained {total} iterations ({steps:,} env steps) in {gpu_s * 1e3:.1f} ms GPU time"
        f" ({timing['ms_per_iteration_gpu']:.3f} ms/iteration, {timing['env_steps_per_s_gpu']:.3e} env steps/s;"
        f" shared GPU, indicative)"
    )
    print(
        f"[SRAM] final: mean episode return {final['mean_episode_return']:.4f}"
        f" (last 10 iterations {final['mean_episode_return_last10_iterations']:.4f}),"
        f" length {final['mean_episode_length']:.1f}, lr {final['learning_rate']:.3e}, std {final['action_std']:.4f}"
    )
    v = exp["verification"]
    print(
        f"[SRAM] exported RSL-RL checkpoint: {exp['checkpoint']} (iter {exp['iter']}, Adam step {exp['adam_step']},"
        f" lr {exp['learning_rate']:.4g}; strict reload {'OK' if v['ok'] else 'FAILED'},"
        f" params/moments bit-exact: {v.get('params_max_abs_diff') == 0.0 and v.get('exp_avg_max_abs_diff') == 0.0})"
    )
    print(f"[SRAM] summary: {os.path.join(log_dir, SUMMARY_FILENAME)}")
    return summary


def play_commands(checkpoint: str, task: str = SUPPORTED_TASKS[0], num_envs: int = 4096) -> list[str]:
    """Stock commands that load the exported checkpoint (printed after training)."""
    return [
        f"python scripts/benchmarks/play.py --rl_library rsl_rl --task {task} --checkpoint {checkpoint}"
        f" --num_envs {num_envs} --warmup_steps 0 --num_steps 600 --visualizer none physics=newton_mjwarp",
        f"python scripts/reinforcement_learning/play.py --rl_library rsl_rl --task {task} --checkpoint {checkpoint}"
        f" --visualizer none physics=newton_mjwarp",
        f"python scripts/sram/eval_policy.py --checkpoint {checkpoint} --num_envs {num_envs}",
    ]


# ------------------------------------------------------------------------------------------------
# unified-CLI backend
# ------------------------------------------------------------------------------------------------
def _parse_args(argv: list[str]) -> argparse.Namespace:
    from isaaclab.app import add_launcher_args

    from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
    from isaaclab_rl.entrypoints.common import add_common_train_args, set_hydra_args

    from isaaclab_tasks.utils import setup_preset_cli

    parser = argparse.ArgumentParser(
        description="Train Isaac-Cartpole-Direct with the SRAM-resident PPO megakernel and export an RSL-RL run."
    )
    add_common_train_args(
        parser,
        agent_default=DEFAULT_AGENT_ENTRY_POINT,
        agent_help="RSL-RL agent config entry point (the megakernel reads its PPO settings).",
    )
    cli_args.add_rsl_rl_args(parser)
    group = parser.add_argument_group("sram", description="SRAM megakernel options.")
    group.add_argument(
        "--iters_per_launch",
        type=int,
        default=None,
        help="PPO iterations per kernel launch (default: all in one launch).",
    )
    group.add_argument("--num_ctas", type=int, default=None, help="Persistent CTAs (default: one per SM).")
    group.add_argument(
        "--storage",
        choices=["auto", "smem", "global"],
        default="auto",
        help="Rollout buffer location (default: shared memory when it fits).",
    )
    group.add_argument(
        "--log_root",
        type=str,
        default=None,
        help="Experiment log root (default: logs/rsl_rl/<experiment_name>, found by play --checkpoint latest).",
    )
    group.add_argument("--no_tensorboard", action="store_true", help="Do not write a TensorBoard event file.")
    group.add_argument("--quiet", action="store_true", help="Do not print the per-iteration log lines.")
    add_launcher_args(parser)
    args, remaining = setup_preset_cli(parser, argv)
    set_hydra_args(remaining)  # Hydra overrides / presets (e.g. physics=newton_mjwarp) for resolve_task_config
    return args


def _reject_unsupported(args: argparse.Namespace) -> None:
    errors = []
    if getattr(args, "video", False):
        errors.append("--video (no simulator runs during SRAM training; record videos with play.py)")
    if getattr(args, "distributed", False):
        errors.append("--distributed (the megakernel is single-GPU)")
    if getattr(args, "checkpoint", None):
        errors.append("--checkpoint (resuming into the megakernel is not implemented)")
    if getattr(args, "capture_env_sensors", 0):
        errors.append("--capture_env_sensors")
    if errors:
        raise SystemExit("[SRAM] unsupported option(s) for --rl_library sram: " + "; ".join(errors))
    if getattr(args, "frontend", "torch") != "torch":
        print("[SRAM] --frontend is ignored: the megakernel simulates the MDP itself")
    if getattr(args, "logger", None) not in (None, "tensorboard"):
        print(f"[SRAM] --logger {args.logger} is not supported; writing TensorBoard events only")


def run(argv: list[str]) -> int:
    """Backend entry point called by ``isaaclab_rl.entrypoints.dispatch.run_cli('train', ...)``."""
    from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
    from isaaclab_rl.entrypoints.common import apply_env_overrides
    from isaaclab_rl.rsl_rl import check_rsl_rl_version, handle_deprecated_rsl_rl_cfg

    import isaaclab_tasks  # noqa: F401  (registers the tasks)
    from isaaclab_tasks.utils import resolve_task_config

    args = _parse_args(argv)
    if not args.task:
        raise SystemExit("[SRAM] --task is required")
    _reject_unsupported(args)
    env_cfg, agent_cfg = resolve_task_config(args.task, args.agent)
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    apply_env_overrides(args, env_cfg)
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    env_cfg.seed = agent_cfg.seed
    if not agent_cfg.run_name:
        agent_cfg.run_name = BACKEND_NAME
    log_root = os.path.abspath(args.log_root or os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    run_name = (args.run_timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + f"_{agent_cfg.run_name}"
    print(f"[INFO] Logging experiment in directory: {log_root}")
    summary = train_and_export(
        args.task,
        env_cfg,
        agent_cfg,
        os.path.join(log_root, run_name),
        iters_per_launch=args.iters_per_launch,
        num_ctas=args.num_ctas,
        storage=args.storage,
        tensorboard=not args.no_tensorboard,
        quiet=args.quiet,
        agent_entry_point=args.agent,
    )
    print("[SRAM] load it with the stock entry points, e.g.:")
    for cmd in play_commands(summary["checkpoint"], args.task.split(":")[-1], summary["num_envs"]):
        print(f"         {cmd}")
    return 0


if __name__ == "__main__":  # python -m isaaclab_sram.rl_backend <train args>  (same as --rl_library sram)
    import warp as wp

    wp.config.enable_backward = False
    sys.exit(run(sys.argv[1:]))

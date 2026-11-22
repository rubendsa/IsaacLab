# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train Isaac-Cartpole-Direct with the SRAM-resident PPO megakernel and export an RSL-RL checkpoint.

Standalone mode. The task and agent configs are resolved from Isaac Lab's registry
(:func:`isaaclab_sram.cartpole.isaaclab_cartpole_configs`, then ``CartpoleTask.from_env_cfg`` and
``PPOConfig.from_rsl_rl_cfg``). No simulator runs during training::

    python scripts/sram/train_sram.py [--num_envs 4096] [--seed 42] [--max_iterations 150]
        [--iters_per_launch N] [--log_dir logs/sram/cartpole_direct/<timestamp>] [--eval]

The script prints one RSL-RL-style line per iteration. It writes ``<log_dir>/sram_summary.json``,
TensorBoard events and an RSL-RL run directory (``run.json``, ``params/{env,agent}.yaml``,
``model_<max_iterations-1>.pt``) that the stock ``scripts/reinforcement_learning/play.py`` and
``scripts/benchmarks/play.py`` load with ``--rl_library rsl_rl --checkpoint <path>``. With ``--eval`` it
then evaluates the exported checkpoint in the stock Newton environment (``scripts/sram/eval_policy.py``
protocol) and writes ``<log_dir>/eval.json``.

Mode for Isaac Lab's unified entry point. This registers the ``--rl_library sram`` backend at runtime
(:mod:`isaaclab_sram.rl_backend`; ``isaaclab_rl`` is not edited) and then runs the unmodified
``scripts/reinforcement_learning/train.py`` with the remaining arguments::

    python scripts/sram/train_sram.py --via-isaaclab-cli --rl_library sram --task Isaac-Cartpole-Direct \\
        --num_envs 4096 --max_iterations 150 physics=newton_mjwarp

The run is written to ``logs/rsl_rl/cartpole_direct/<timestamp>_sram`` (stock log root and manifest).
``play.py --rl_library rsl_rl --task Isaac-Cartpole-Direct --checkpoint latest`` therefore finds it.

``isaaclab_sram`` is not installed in the venv. If it cannot be imported, ``source/isaaclab_sram`` is put
on ``sys.path``.
"""

from __future__ import annotations

# as scripts/reinforcement_learning/train.py: set before anything defines Warp kernels
import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import importlib.metadata  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import runpy  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SRAM_SRC = REPO_ROOT / "source" / "isaaclab_sram"
STOCK_TRAIN = REPO_ROOT / "scripts" / "reinforcement_learning" / "train.py"
VIA_CLI_FLAG = "--via-isaaclab-cli"


def bootstrap_import_path() -> None:
    """Make ``isaaclab_sram`` importable without installing it (and ``eval_policy`` next to this script)."""
    try:
        import isaaclab_sram  # noqa: F401
    except ModuleNotFoundError:
        sys.path.insert(0, str(SRAM_SRC))
        import isaaclab_sram  # noqa: F401
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)


def run_via_isaaclab_cli(argv: list[str]) -> None:
    """Register ``--rl_library sram`` and run the stock unified ``train.py`` with *argv* (exits with its code)."""
    bootstrap_import_path()
    from isaaclab_sram import rl_backend

    rl_backend.register()
    sys.argv = [str(STOCK_TRAIN), *argv]
    runpy.run_path(str(STOCK_TRAIN), run_name="__main__")  # raises SystemExit with the CLI's exit code


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Isaac-Cartpole-Direct with the SRAM-resident PPO megakernel and export an RSL-RL run.",
        epilog=f"Use '{VIA_CLI_FLAG} <train.py args>' to go through Isaac Lab's unified train.py (--rl_library sram).",
    )
    p.add_argument("--task", default="Isaac-Cartpole-Direct")
    p.add_argument("--num_envs", type=int, default=None, help="Default: the env config (4096).")
    p.add_argument("--seed", type=int, default=None, help="Default: the agent config (42).")
    p.add_argument("--max_iterations", type=int, default=None, help="Default: the agent config (150).")
    p.add_argument(
        "--iters_per_launch", type=int, default=None, help="PPO iterations per kernel launch (default: all)."
    )
    p.add_argument("--num_ctas", type=int, default=None, help="Persistent CTAs (default: one per SM).")
    p.add_argument("--storage", choices=["auto", "smem", "global"], default="auto", help="Rollout buffer location.")
    p.add_argument("--device", default=None, help="CUDA device (default: the env config, cuda:0).")
    p.add_argument("--log_dir", default=None, help="Run directory (default: logs/sram/cartpole_direct/<timestamp>).")
    p.add_argument("--summary_json", default=None, help="Also copy the JSON summary here.")
    p.add_argument("--no_tensorboard", action="store_true", help="Do not write a TensorBoard event file.")
    p.add_argument("--quiet", action="store_true", help="Do not print the per-iteration log lines.")
    p.add_argument("--eval", action="store_true", help="Evaluate the exported checkpoint in the stock Newton env.")
    p.add_argument("--eval_num_envs", type=int, default=4096)
    p.add_argument("--eval_num_episodes", type=int, default=2)
    p.add_argument("--eval_seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if VIA_CLI_FLAG in argv:
        run_via_isaaclab_cli([a for a in argv if a != VIA_CLI_FLAG])
        return 0
    args = parse_args(argv)
    bootstrap_import_path()
    from isaaclab_sram import rl_backend
    from isaaclab_sram.cartpole import isaaclab_cartpole_configs

    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

    env_cfg, agent_cfg = isaaclab_cartpole_configs(args.task)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if args.seed is not None:
        agent_cfg.seed = args.seed
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.device is not None:
        env_cfg.sim.device = agent_cfg.device = args.device
    env_cfg.seed = agent_cfg.seed
    log_dir = args.log_dir or os.path.join(
        "logs", "sram", agent_cfg.experiment_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    summary = rl_backend.train_and_export(
        args.task,
        env_cfg,
        agent_cfg,
        log_dir,
        iters_per_launch=args.iters_per_launch,
        num_ctas=args.num_ctas,
        storage=args.storage,
        tensorboard=not args.no_tensorboard,
        quiet=args.quiet,
    )
    summary_path = os.path.join(summary["log_dir"], rl_backend.SUMMARY_FILENAME)

    if args.eval:
        import eval_policy

        result = eval_policy.evaluate(
            checkpoint=summary["checkpoint"],
            task=args.task,
            num_envs=args.eval_num_envs,
            num_episodes=args.eval_num_episodes,
            seed=args.eval_seed,
            device=str(env_cfg.sim.device),
            per_env_npz=os.path.join(summary["log_dir"], "eval_per_env.npz"),
            label="sram",
        )
        eval_path = eval_policy.write_json(result, os.path.join(summary["log_dir"], "eval.json"))
        print(eval_policy.format_summary(result))
        print(f"[eval] wrote {eval_path}")
        fe = result["first_episodes"]
        summary["eval"] = {
            "json": eval_path,
            "protocol": result["protocol"],
            "first_episodes": {
                k: fe[k]
                for k in (
                    "num_episodes",
                    "return_mean",
                    "return_std",
                    "length_mean",
                    "timeout_fraction",
                    "terminated_fraction",
                )
            },
            "benchmark_play_equivalent": result["benchmark_play_equivalent"],
        }
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=1)
    else:
        print("[SRAM] evaluate / play the export with:")
        for cmd in rl_backend.play_commands(summary["checkpoint"], args.task, summary["num_envs"]):
            print(f"         {cmd}")
    if args.summary_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_json)), exist_ok=True)
        shutil.copyfile(summary_path, args.summary_json)
        print(f"[SRAM] summary copied to {os.path.abspath(args.summary_json)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train real Kuka-Allegro lift from RGB/depth with stock or fused perception PPO.

Example::

    OMP_NUM_THREADS=1 uv run python scripts/sram/train_kuka_perception.py \
        --backend sram --camera rgb64 --num_envs 4096 --iterations 100

The full task, renderer, CNN, privileged critic and PPO settings come from Isaac
Lab. The report separates iteration timings, finite-parameter checks, CNN weight
changes and checkpoint replay. A short run establishes integration; it does not
establish convergence. Persistent backends retain selected network working state
during their optimizer phases; all paths still use global rollout buffers and
global CNN activations. This is not a complete SRAM-resident visual learner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source" / "isaaclab_sram"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("torch", "sram", "sram_graph", "sram_critic", "sram_full", "sram_unified"), default="sram"
    )
    parser.add_argument("--camera", choices=("rgb64", "depth64"), default="rgb64")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--warmup_iterations", type=int, default=5, help="Exclude initial iterations from timing medians."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Resume a native RSL-RL checkpoint.")
    parser.add_argument("--smoke_reset_bank", type=int, default=None, help="Explicit smoke-only smaller reset bank.")
    parser.add_argument(
        "--compact_history",
        action="store_true",
        help="Store processed state frames once and reconstruct native histories.",
    )
    parser.add_argument(
        "--ieee_fp32", action="store_true", help="Disable TF32 library operations for controlled comparisons."
    )
    args = parser.parse_args()
    if args.num_envs < 1 or args.iterations < 1 or args.warmup_iterations < 0:
        parser.error("num_envs and iterations must be positive; warmup_iterations must be nonnegative")
    if args.backend in ("sram_critic", "sram_full") and args.num_envs > 64:
        parser.error("Persistent camera backends currently support at most 64 environments")
    if args.backend == "sram_unified" and args.num_envs > 16:
        parser.error("sram_unified currently supports at most 16 environments")
    if args.backend in ("sram_full", "sram_unified") and args.camera != "rgb64":
        parser.error(f"{args.backend} currently supports the native RGB64 actor only")
    return args


def observation_storage_tensors(observations):
    """Count physical rollout allocations for either native or compact storage."""
    return (
        list(observations.storage_tensors())
        if hasattr(observations, "storage_tensors")
        else list(observations.values())
    )


def main():
    args = parse_args()
    import warp as wp

    wp.config.enable_backward = False
    import gymnasium as gym
    import torch
    from rsl_rl.utils import resolve_class

    from isaaclab.app import launch_simulation
    from isaaclab.utils import to_dict

    from isaaclab_rl.entrypoints.common import dump_train_configs, scoped_torch_backend_flags, write_run_manifest
    from isaaclab_rl.rsl_rl import (
        RslRlVecEnvWrapper,
        check_rsl_rl_version,
        create_rsl_rl_runner,
        handle_deprecated_rsl_rl_cfg,
    )

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config

    task = "Isaac-Lift-KukaAllegro-Camera"
    overrides = ["physics=newton_mjwarp", "renderer=newton_renderer", f"presets=single_camera,{args.camera}"]
    env_cfg, agent_cfg = resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=overrides)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = agent_cfg.device = args.device
    env_cfg.sim.visualizer_cfgs = []
    env_cfg.seed = agent_cfg.seed = args.seed
    agent_cfg.max_iterations = args.iterations
    agent_cfg.logger = "tensorboard"
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    if args.backend == "sram":
        agent_cfg.algorithm.class_name = "isaaclab_sram.perception:SramPerceptionPPO"
    elif args.backend == "sram_graph":
        agent_cfg.algorithm.class_name = "isaaclab_sram.perception:SramGraphPerceptionPPO"
    elif args.backend == "sram_critic":
        agent_cfg.algorithm.class_name = "isaaclab_sram.persistent_perception:SramPersistentCriticPPO"
    elif args.backend == "sram_full":
        agent_cfg.algorithm.class_name = "isaaclab_sram.persistent_full_perception:SramPersistentFullPerceptionPPO"
    elif args.backend == "sram_unified":
        agent_cfg.algorithm.class_name = "isaaclab_sram.unified_perception:SramUnifiedPerceptionPPO"
    if args.smoke_reset_bank is not None:
        env_cfg.events.conditional_reset.params["buffer_size_per_group"] = args.smoke_reset_bank
        env_cfg.events.conditional_reset.params["oversample_factor"] = 1.0
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    output = (args.output or Path("logs/sram/kuka_perception") / f"{stamp}_{args.backend}_{args.camera}").resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a new run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    env_cfg.log_dir = str(output)
    write_run_manifest(
        str(output),
        library="rsl_rl",
        task=task,
        metadata={"agent": "rsl_rl_cfg_entry_point", "trainer": f"perception_{args.backend}"},
    )
    dump_train_configs(str(output), env_cfg, agent_cfg)
    report = {
        "backend": args.backend,
        "task": task,
        "camera": args.camera,
        "overrides": overrides,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "iterations": args.iterations,
        "warmup_iterations": args.warmup_iterations,
        "smoke_reset_bank": args.smoke_reset_bank,
        "compact_history": args.compact_history,
        "precision": (
            "Float32 with TF32 disabled"
            if args.ieee_fp32
            else "Float32 with TF32 library operations; custom reductions and persistent critic use IEEE float32"
        ),
        "resume_checkpoint": str(args.checkpoint.resolve()) if args.checkpoint is not None else None,
        "resume_checkpoint_sha256": (
            hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint is not None else None
        ),
        "resume_scope": "Model and optimizer only; simulator, curriculum and reset-bank state initialize anew",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "process_id": os.getpid(),
        "output": str(output),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(args.device),
        "storage_scope": (
            "Actor and critic parameters/gradients/Adam in shared memory during their separate persistent phases; "
            "global rollout, activation and cross-block reduction buffers; separate GAE and normalizer kernels"
            if args.backend == "sram_full"
            else (
                "Persistent critic parameters/gradients/Adam during its update; global rollout, actor, activation and "
                "cross-block reduction buffers"
                if args.backend == "sram_critic"
                else "Global rollout/model/optimizer allocations; temporary working tiles in individual CUDA kernels"
            )
        ),
        "timing_scope": (
            "Native runner collection/learning timers with CUDA synchronization before update returns; "
            "total iteration wall time is synchronized; split phases retain host-timer attribution. "
            "Initial compilation is included in warmup, logging/checkpoint IO excluded from iteration timings"
        ),
        "convergence_claim": False,
        "update_rng": "Discarded native action samples omitted"
        if args.backend == "sram_graph"
        else "Native PPO samples",
        "rows": [],
    }
    if args.backend == "sram_unified":
        report["storage_scope"] = (
            "Both actor and critic weights/Adam retained across one cooperative launch; phase-overlaid gradient "
            "arena, with critic gradient diagnostics exported before reuse; global rollout/activation/transport; "
            "GAE, normalization and RNG alignment outside the kernel"
        )
    report["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
    ).strip()
    repo = Path(__file__).resolve().parents[2]
    report["git_status"] = subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True).splitlines()
    package = repo / "source/isaaclab_sram/isaaclab_sram"
    source_files = [
        Path(__file__),
        *package.glob("*.py"),
        *(path for path in (package / "kernels").iterdir() if path.suffix in (".cu", ".cuh")),
    ]
    report["source_sha256"] = {
        str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(source_files)
    }
    report["gpu_processes_start"] = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"], text=True
    ).splitlines()
    t0 = time.perf_counter()
    with (
        scoped_torch_backend_flags(
            cuda_matmul_allow_tf32=not args.ieee_fp32,
            cudnn_allow_tf32=not args.ieee_fp32,
            cudnn_deterministic=False,
            cudnn_benchmark=True,
        ),
        launch_simulation(env_cfg, {"visualizer": ["none"], "device": args.device, "livestream": 0}),
    ):
        base = gym.make(task, cfg=env_cfg)
        try:
            env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
            runner = create_rsl_rl_runner(env, agent_cfg, log_dir=str(output))
            if args.checkpoint is not None:
                runner.load(str(args.checkpoint), map_location=args.device)
            alg = runner.alg
            if args.compact_history:
                from isaaclab_sram.compact_history import layouts_from_observation_manager
                from isaaclab_sram.compact_storage import compact_rollout_observations

                layouts = layouts_from_observation_manager(base.unwrapped.observation_manager)
                compact = compact_rollout_observations(alg.storage, layouts)
                report["compact_history_layouts"] = {
                    name: {"order": layout.order, "terms": [vars(term) for term in layout.terms]}
                    for name, layout in layouts.items()
                }
                report["rollout_dense_observation_bytes"] = compact.dense_bytes
                report["compact_history_frames_bytes"] = sum(h.frame_bytes for h in compact.histories.values())
                report["compact_history_storage_bytes"] = sum(h.storage_bytes for h in compact.histories.values())
                report["compact_history_scope"] = (
                    "Exact processed frames and reset boundaries in global storage; unchanged raw images. "
                    "Minibatch gathering reconstructs histories; normalization reconstructs complete state groups."
                )
            report["startup_s"] = time.perf_counter() - t0
            report["fused_spatial_softmax_modules"] = getattr(alg, "fused_spatial_softmax_modules", 0)
            obs = env.get_observations()
            report["observations"] = {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in obs.items()}
            report["rollout_observation_bytes"] = sum(
                v.numel() * v.element_size() for v in observation_storage_tensors(alg.storage.observations)
            )
            image = obs["base_image"].float()
            report["initial_image"] = {"min": image.min().item(), "max": image.max().item(), "std": image.std().item()}
            import numpy as np
            from PIL import Image

            preview = image[:16].detach()
            if args.camera == "depth64":
                preview = torch.nan_to_num(preview, nan=2.0, posinf=2.0).clamp(0, 2).repeat(1, 3, 1, 1) * 127.5
                report["camera_preview_scale"] = "Depth in meters: black=0m, white=2m or farther"
            else:
                report["camera_preview_scale"] = "Native uint8 RGB"
            pixels = preview.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            n, height, width, _ = pixels.shape
            grid = np.zeros((math.ceil(n / 4) * height, 4 * width, 3), dtype=np.uint8)
            for i, pixel_image in enumerate(pixels):
                row, col = divmod(i, 4)
                grid[row * height : (row + 1) * height, col * width : (col + 1) * width] = pixel_image
            preview_path = output / "camera_preview.jpg"
            Image.fromarray(grid).save(preview_path, quality=95)
            report["camera_preview"] = str(preview_path)
            before = {k: v.detach().clone() for k, v in alg.actor.named_parameters()}
            logger_log = runner.logger.log
            original_update = alg.update

            def synchronized_update():
                losses = original_update()
                torch.cuda.synchronize(args.device)
                return losses

            alg.update = synchronized_update

            def record_log(*positional, **kw):
                collect, learn = kw["collect_time"], kw["learn_time"]
                row = {
                    "iteration": kw["it"],
                    "collection_s": collect,
                    "learning_s": learn,
                    "iteration_s": collect + learn,
                    "losses": dict(kw["loss_dict"]),
                }
                row["transitions_per_s"] = args.num_envs * agent_cfg.num_steps_per_env / (collect + learn)
                if not all(math.isfinite(v) for v in row["losses"].values()):
                    raise FloatingPointError(f"Nonfinite training loss: {row}")
                logger = runner.logger
                if logger.rewbuffer:
                    row["mean_episode_reward"] = statistics.mean(logger.rewbuffer)
                    row["mean_episode_length"] = statistics.mean(logger.lenbuffer)
                row["task_metrics"] = {}
                for key in dict.fromkeys(key for extra in logger.ep_extras for key in extra):
                    if key.startswith(("Metrics/", "Curriculum/", "Episode_Reward/")):
                        vals = [
                            torch.as_tensor(extra[key], device=args.device).flatten()
                            for extra in logger.ep_extras
                            if key in extra
                        ]
                        row["task_metrics"][key] = torch.cat(vals).float().mean().item()
                report["rows"].append(row)
                logger_log(*positional, **kw)

            runner.logger.log = record_log
            train_start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(args.device)
            runner.learn(args.iterations, init_at_random_ep_len=agent_cfg.init_at_random_ep_len)
            torch.cuda.synchronize(args.device)
            report["train_wall_s"] = time.perf_counter() - train_start
            report["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
            report["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(args.device)
            if getattr(alg, "_persistent_critic", None) is not None:
                report["persistent_critic_resources"] = alg._persistent_critic.resource_info
            if getattr(alg, "_persistent_actor", None) is not None:
                report["persistent_actor_resources"] = alg._persistent_actor.resource_info
            if getattr(alg, "_unified", None) is not None:
                report["unified_learner_resources"] = alg._unified.resource_info
                report["unified_learner_launches"] = alg._unified.launch_count
            storage_tensors = (
                observation_storage_tensors(alg.storage.observations)
                + [
                    getattr(alg.storage, name)
                    for name in ("actions", "rewards", "dones", "values", "returns", "advantages", "actions_log_prob")
                ]
                + list(alg.storage.distribution_params)
            )
            report["rollout_storage_bytes"] = sum(t.numel() * t.element_size() for t in storage_tensors)
            report["actor_parameter_max_abs_change"] = {
                k: (v.detach() - before[k]).abs().max().item() for k, v in alg.actor.named_parameters()
            }
            report["cnn_changed"] = any(
                v > 0 for k, v in report["actor_parameter_max_abs_change"].items() if k.startswith("cnns.")
            )
            report["all_parameters_finite"] = all(
                bool(torch.isfinite(v).all()) for model in (alg.actor, alg.critic) for v in model.parameters()
            )
            checkpoint = output / f"model_{runner.current_learning_iteration}.pt"
            report["checkpoint"] = str(checkpoint)
            with torch.no_grad():
                obs = env.get_observations()
                reference_actions = alg.actor(obs).clone()
                saved = torch.load(checkpoint, map_location=args.device, weights_only=False)
                actor_class, actor_kwargs = resolve_class(to_dict(agent_cfg.actor))
                native_actor = actor_class(obs, runner.cfg["obs_groups"], "actor", env.num_actions, **actor_kwargs).to(
                    args.device
                )
                native_actor.load_state_dict(saved["actor_state_dict"], strict=True)
                replay_actions = native_actor(obs)
                report["checkpoint_tf32_action_max_abs_error"] = (reference_actions - replay_actions).abs().max().item()
                # Native coordinate expectation uses matmul and therefore TF32;
                # the fused reduction computes float32. Compare checkpoint math
                # in full precision, and retain the production-precision delta.
                with scoped_torch_backend_flags(
                    cuda_matmul_allow_tf32=False,
                    cudnn_allow_tf32=False,
                    cudnn_deterministic=False,
                    cudnn_benchmark=True,
                ):
                    precise_actions = alg.actor(obs)
                    precise_replay = native_actor(obs)
                report["checkpoint_action_max_abs_error"] = (precise_actions - precise_replay).abs().max().item()
                report["checkpoint_verification_precision"] = "IEEE float32 (TF32 disabled for both actors)"
                report["checkpoint_native_actor_class"] = type(native_actor).__name__
                report["checkpoint_native_actor_matches"] = bool(
                    torch.allclose(precise_actions, precise_replay, rtol=1e-4, atol=1e-5)
                )
                zero_image_obs = obs.clone()
                zero_image_obs["base_image"].zero_()
                report["image_action_sensitivity_max_abs"] = (
                    (reference_actions - alg.actor(zero_image_obs)).abs().max().item()
                )
            measured = report["rows"][args.warmup_iterations :]
            report["measured_iterations"] = len(measured)
            report["timing_medians"] = (
                {
                    k: statistics.median(row[k] for row in measured)
                    for k in ("collection_s", "learning_s", "iteration_s", "transitions_per_s")
                }
                if measured
                else {}
            )
            report["integration_passed"] = (
                report["cnn_changed"]
                and report["all_parameters_finite"]
                and report["checkpoint_native_actor_matches"]
                and report["image_action_sensitivity_max_abs"] > 0.0
                and report["initial_image"]["std"] > 0.0
            )
            report["finished_utc"] = datetime.now(timezone.utc).isoformat()
            (output / "perception_summary.json").write_text(json.dumps(report, indent=2) + "\n")
            print(
                "PERCEPTION_RESULT",
                json.dumps({k: v for k, v in report.items() if k not in ("rows", "actor_parameter_max_abs_change")}),
            )
            if not report["integration_passed"]:
                raise RuntimeError(f"Perception integration check failed; inspect {output / 'perception_summary.json'}")
        finally:
            base.close()


if __name__ == "__main__":
    main()

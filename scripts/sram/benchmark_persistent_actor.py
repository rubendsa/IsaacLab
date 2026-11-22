# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Benchmark the native RGB64 Kuka actor's complete forty-step PPO update.

Every warmup and repeat restores the same resumed native model and Adam state
outside timing. The synthetic rollout deliberately covers clipped and unclipped
policy ratios. Both paths train the native CNN, spatial temperatures, MLP and
Gaussian standard deviation. This component benchmark excludes simulation,
normalizer updates, critic learning and full-learner RNG alignment draws. It is
not a convergence measurement or a claim of complete SRAM residency.

Examples::

    OMP_NUM_THREADS=1 uv run python scripts/sram/benchmark_persistent_actor.py \
        --backend both --adam fused --num_envs 64 --warmup 3 --repeats 5 --output /tmp/actor.json

    # Configure the profiler to start disabled; collect only measured updates.
    OMP_NUM_THREADS=1 uv run python scripts/sram/benchmark_persistent_actor.py \
        --backend persistent --warmup 2 --repeats 1 --profile --output /tmp/actor-profile.json
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from benchmark_persistent_critic import compare_states, restore, snapshot, state_summary, tensor_summary, timing_summary

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("torch", "persistent", "both"), default="both")
    parser.add_argument("--adam", choices=("default", "fused"), default="default")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--burnin_steps", type=int, default=1, help="Untimed native steps before the shared checkpoint."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true", help="Enable CUDA profiler around measured updates only.")
    args = parser.parse_args()
    if not 1 <= args.num_envs <= 64 or args.repeats < 1 or args.warmup < 0 or args.burnin_steps < 0:
        parser.error("num_envs must be 1–64, repeats positive, and warmup/burnin_steps nonnegative")
    if args.profile and args.backend == "both":
        parser.error("Select one backend when using --profile")
    if args.output.exists():
        parser.error(f"Output already exists; select a new evidence file: {args.output}")
    return args


def make_optimizer(actor, algorithm, adam):
    """Create the actual Adam variant with explicit state, including zero-step runs."""
    import torch

    optimizer = torch.optim.Adam(
        actor.parameters(), lr=algorithm["learning_rate"], **({"fused": True} if adam == "fused" else {})
    )
    for parameter in actor.parameters():
        optimizer.state[parameter] = {
            "step": torch.zeros((), device=parameter.device if adam == "fused" else "cpu"),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.zeros_like(parameter),
        }
    return optimizer


def torch_minibatch(actor, optimizer, rollout, selected, algorithm):
    """Native CNN/MLP/distribution, clipped surrogate, entropy, clipping and Adam."""
    import torch

    # The actor's deterministic forward does not update its Normal object.
    # Update explicitly, omitting the discarded sample in both timed paths.
    mean = actor(rollout["observations"][selected], stochastic_output=False)
    actor.distribution.update(mean)
    logp = actor.get_output_log_prob(rollout["actions"][selected])
    ratio = torch.exp(logp - rollout["old_logp"][selected].squeeze(-1))
    advantages = rollout["advantages"][selected].squeeze(-1)
    surrogate = -advantages * ratio
    clipped = -advantages * ratio.clamp(1.0 - algorithm["clip_param"], 1.0 + algorithm["clip_param"])
    surrogate_loss = torch.max(surrogate, clipped).mean()
    entropy = actor.output_entropy.mean()
    loss = surrogate_loss - algorithm["entropy_coef"] * entropy
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), algorithm["max_grad_norm"])
    optimizer.step()
    return torch.stack((surrogate_loss.detach(), entropy.detach()))


def torch_update(actor, optimizer, rollout, algorithm):
    import torch

    metrics = torch.zeros(2, device=rollout["actions"].device)
    width = rollout["actions"].shape[0] // algorithm["num_mini_batches"]
    for _ in range(algorithm["num_learning_epochs"]):
        for minibatch in range(algorithm["num_mini_batches"]):
            selected = rollout["permutation"][minibatch * width : (minibatch + 1) * width]
            metrics += torch_minibatch(actor, optimizer, rollout, selected, algorithm)
    return metrics / (algorithm["num_learning_epochs"] * algorithm["num_mini_batches"])


def main():
    args = parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from rsl_rl.utils import resolve_class
    from tensordict import TensorDict

    from isaaclab.utils import to_dict

    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

    from isaaclab_tasks.core.lift.config.kuka_allegro.agents.rsl_rl_ppo_cfg import (
        KukaAllegroSingleCameraPPORunnerCfg,
    )

    if torch.device(args.device).type != "cuda":
        raise ValueError("This CUDA-event benchmark requires a CUDA device")
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cfg = to_dict(handle_deprecated_rsl_rl_cfg(KukaAllegroSingleCameraPPORunnerCfg(), version("rsl-rl-lib")))
    algorithm = cfg["algorithm"]
    if (
        cfg["num_steps_per_env"] != 32
        or algorithm["num_mini_batches"] != 8
        or algorithm["num_learning_epochs"] != 5
        or algorithm["schedule"] != "fixed"
    ):
        raise ValueError("Native camera configuration changed; review the fixed forty-step benchmark contract")
    batch = args.num_envs * cfg["num_steps_per_env"]
    width = batch // algorithm["num_mini_batches"]
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    observations = TensorDict(
        {
            "policy": torch.randn(batch, 170, generator=generator) * 0.7,
            "proprio": torch.randn(batch, 465, generator=generator) * 0.5,
            "base_image": torch.randint(0, 256, (batch, 3, 64, 64), dtype=torch.uint8, generator=generator),
        },
        batch_size=[batch],
    ).to(args.device)
    actor_class, actor_cfg = resolve_class(cfg["actor"])
    initial_actor = actor_class(observations[:1], cfg["obs_groups"], "actor", 23, **actor_cfg).to(args.device).train()
    initial_actor.update_normalization(observations)
    generator.manual_seed(args.seed + 2)
    with torch.no_grad():
        mean = initial_actor(observations)
        initial_actor.distribution.update(mean)
        actions = mean + initial_actor.output_std * torch.randn(batch, 23, generator=generator).to(args.device)
        # Frozen off-policy ratios deliberately exercise both clipping branches.
        offsets = torch.tensor([-0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6], device=args.device)
        old_logp = initial_actor.get_output_log_prob(actions) - offsets[torch.arange(batch, device=args.device) % 7]
        advantages = torch.randn(batch, 1, generator=generator).to(args.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        normalized_state = initial_actor.obs_normalizer(
            torch.cat([observations[group] for group in initial_actor.obs_groups], -1)
        ).contiguous()
    rollout = {
        "observations": observations,
        "normalized_state": normalized_state,
        "raw_rgb": observations["base_image"],
        "actions": actions.contiguous(),
        "old_logp": old_logp[:, None].contiguous(),
        "advantages": advantages.contiguous(),
        "permutation": torch.randperm(batch, generator=generator).to(args.device),
    }
    initial_optimizer = make_optimizer(initial_actor, algorithm, args.adam)
    for step in range(args.burnin_steps):
        offset = step % algorithm["num_mini_batches"] * width
        torch_minibatch(
            initial_actor, initial_optimizer, rollout, rollout["permutation"][offset : offset + width], algorithm
        )
    saved = snapshot(initial_actor, initial_optimizer)
    torch.cuda.synchronize(args.device)
    paths = [
        Path(__file__),
        Path(__file__).with_name("benchmark_persistent_critic.py"),
        ROOT / "source/isaaclab_tasks/isaaclab_tasks/core/lift/config/kuka_allegro/agents/rsl_rl_ppo_cfg.py",
        ROOT / "source/isaaclab_sram/isaaclab_sram/persistent_actor.py",
        ROOT / "source/isaaclab_sram/isaaclab_sram/kernels/ppo_native_actor.cu",
        ROOT / "source/isaaclab_sram/isaaclab_sram/cuda_rt.py",
        Path(inspect.getfile(actor_class)),
        Path(inspect.getfile(type(initial_actor.distribution))),
        Path(inspect.getfile(type(initial_actor.softmaxes["base_image"]))),
    ]
    report = {
        "scope": "Native RGB64 actor on a synthetic frozen rollout; no critic, GAE, simulation or convergence",
        "timing_scope": "Forty actor steps including wrapper validation/pack/flush; restoration and compile excluded",
        "cuda_event_note": "Events include host launch gaps; synchronized host wall includes CPU wrapper work",
        "normalization_scope": (
            "Normalizer is updated once before burn-in and frozen; native evaluates it per minibatch, "
            "persistent consumes the pre-normalized state"
        ),
        "rng_scope": (
            "Both component paths omit the native PPO discarded Gaussian samples. "
            "Torch updates the native distribution explicitly after its deterministic forward; "
            "full-learner RNG alignment draws are excluded"
        ),
        "method": "Every warmup/repeat restores identical model/normalizer/Adam tensors outside timing",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines(),
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "gpu": torch.cuda.get_device_name(args.device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "rsl_rl": version("rsl-rl-lib"),
        "gpu_snapshot": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,clocks.sm,power.draw,temperature.gpu",
                "--format=csv,noheader",
            ],
            text=True,
        ).splitlines(),
        "precision": "IEEE float32, TF32 disabled, deterministic algorithms enabled",
        "actor_class": type(initial_actor).__name__,
        "actor_configuration": cfg["actor"],
        "ppo_configuration": algorithm,
        "optimizer_configuration": {
            key: value for key, value in initial_optimizer.param_groups[0].items() if key != "params"
        },
        "batch_size": batch,
        "minibatch_size": width,
        "update_steps": algorithm["num_mini_batches"] * algorithm["num_learning_epochs"],
        "parameter_count": sum(parameter.numel() for parameter in initial_actor.parameters()),
        "metric_names": ["surrogate_loss", "entropy"],
        "inputs": {key: tensor_summary(value) for key, value in rollout.items() if isinstance(value, torch.Tensor)},
        "initial_state": state_summary(initial_actor, initial_optimizer),
        "results": {},
    }
    outputs = {}
    backends = ("torch", "persistent") if args.backend == "both" else (args.backend,)
    for backend in backends:
        # Reconstruct a native actor: cached Normal tensors from prior forwards
        # make deep-copying the original live module inappropriate.
        actor = actor_class(observations[:1], cfg["obs_groups"], "actor", 23, **actor_cfg).to(args.device).train()
        optimizer = make_optimizer(actor, algorithm, args.adam)
        restore(actor, optimizer, saved)
        persistent = None
        if backend == "persistent":
            from isaaclab_sram.persistent_actor import PersistentActor

            persistent = PersistentActor(
                actor,
                optimizer,
                **{
                    key: algorithm[key]
                    for key in (
                        "num_mini_batches",
                        "num_learning_epochs",
                        "clip_param",
                        "entropy_coef",
                        "max_grad_norm",
                    )
                },
            )

        def update():
            if persistent is None:
                return torch_update(actor, optimizer, rollout, algorithm)
            return persistent.update(
                rollout["normalized_state"],
                rollout["raw_rgb"],
                rollout["actions"],
                rollout["old_logp"],
                rollout["advantages"],
                rollout["permutation"],
            )

        for _ in range(args.warmup):
            restore(actor, optimizer, saved)
            update()
        torch.cuda.synchronize(args.device)
        gpu_ms, wall_ms, metrics, signatures = [], [], [], []
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(args.repeats):
            restore(actor, optimizer, saved)
            torch.cuda.synchronize(args.device)
            if args.profile:
                torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push("actor_timed_update")
            tick = time.perf_counter()
            start.record()
            losses = update()
            stop.record()
            stop.synchronize()
            wall_ms.append((time.perf_counter() - tick) * 1000)
            gpu_ms.append(start.elapsed_time(stop))
            torch.cuda.nvtx.range_pop()
            if args.profile:
                torch.cuda.cudart().cudaProfilerStop()
            metrics.append(losses.tolist())
            summary = state_summary(actor, optimizer)
            signatures.append({key: summary[key]["sha256"] for key in ("parameters", "exp_avg", "exp_avg_sq")})
        outputs[backend] = snapshot(actor, optimizer, cpu=True)
        result = {
            "cuda_events": timing_summary(gpu_ms),
            "host_wall": timing_summary(wall_ms),
            "metrics": metrics,
            "output_state": summary,
            "repeat_state_sha256": signatures,
            "repeats_bitwise_equal": all(signature == signatures[0] for signature in signatures),
            "all_finite": all(summary[key]["all_finite"] for key in ("parameters", "exp_avg", "exp_avg_sq"))
            and all(math.isfinite(value) for row in metrics for value in row),
            "kernel_resources": persistent.resource_info if persistent is not None else None,
        }
        report["results"][backend] = result
        print(
            json.dumps(
                {
                    "backend": backend,
                    "adam": args.adam,
                    "cuda_events": {key: value for key, value in result["cuda_events"].items() if key != "samples_ms"},
                    "host_wall": {key: value for key, value in result["host_wall"].items() if key != "samples_ms"},
                    "metrics": metrics[-1],
                    "all_finite": result["all_finite"],
                    "repeats_bitwise_equal": result["repeats_bitwise_equal"],
                }
            ),
            flush=True,
        )
    if args.backend == "both":
        report["parity"] = compare_states(outputs["torch"], outputs["persistent"])
        paired = list(
            zip(report["results"]["torch"]["metrics"], report["results"]["persistent"]["metrics"], strict=True)
        )
        report["metric_max_abs_difference"] = [max(abs(left[i] - right[i]) for left, right in paired) for i in range(2)]
        report["parity"]["metrics_passed"] = all(
            math.isclose(left[i], right[i], rel_tol=3e-4, abs_tol=2e-5) for left, right in paired for i in range(2)
        )
        report["parity"]["passed"] &= report["parity"]["metrics_passed"]
        report["cuda_event_median_speedup"] = (
            report["results"]["torch"]["cuda_events"]["median_ms"]
            / report["results"]["persistent"]["cuda_events"]["median_ms"]
        )
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not all(result["all_finite"] and result["repeats_bitwise_equal"] for result in report["results"].values()):
        raise RuntimeError(f"Nonfinite or nondeterministic repeat; inspect {args.output}")
    if "parity" in report and not report["parity"]["passed"]:
        raise RuntimeError(f"Persistent actor differs from native update; inspect {args.output}")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "parity_passed": report.get("parity", {}).get("passed"),
                "cuda_event_median_speedup": report.get("cuda_event_median_speedup"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

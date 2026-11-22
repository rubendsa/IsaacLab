# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Benchmark the native Kuka camera critic's complete 40-step PPO update.

The resolved native critic consumes a fixed normalized synthetic rollout. Every
warmup and measured repeat restores the same parameters, Adam moments and step
counters outside the timed region. This isolates learner execution; it does not
measure cameras, simulation, policy learning, convergence or complete SRAM
residency. CUDA events include host launch gaps and the wrapper's packing and
flushing work, while synchronized host wall time is reported separately.

Examples::

    OMP_NUM_THREADS=1 uv run python scripts/sram/benchmark_persistent_critic.py \
        --backend both --num_envs 64 --warmup 2 --repeats 10 --output /tmp/critic.json

    # Use the profiler's "profile from start off" option with --profile.
    OMP_NUM_THREADS=1 uv run python scripts/sram/benchmark_persistent_critic.py \
        --backend persistent --warmup 2 --repeats 1 --profile --output /tmp/critic-profile.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "isaaclab_sram"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("torch", "persistent", "both"), default="both")
    parser.add_argument(
        "--adam", choices=("default", "fused"), default="default", help="Use the same Adam variant in both backends."
    )
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--burnin_steps", type=int, default=1, help="Untimed native Adam steps before the shared checkpoint."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true", help="Enable CUDA profiler only around measured updates.")
    args = parser.parse_args()
    if not 1 <= args.num_envs <= 64 or args.repeats < 1 or args.warmup < 0 or args.burnin_steps < 0:
        parser.error("num_envs must be 1–64, repeats positive, and warmup/burnin_steps nonnegative")
    if args.profile and args.backend == "both":
        parser.error("Select one backend when using --profile")
    if args.output.exists():
        parser.error(f"Output already exists; select a new evidence file: {args.output}")
    return args


def tensor_summary(tensor):
    """Summarize all values outside timing, including an exact byte fingerprint."""
    import torch

    values = tensor.detach().cpu().contiguous()
    double = values.double()
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "sha256": hashlib.sha256(values.numpy().tobytes()).hexdigest(),
        "all_finite": bool(torch.isfinite(values).all()),
        "l2": double.norm().item(),
        "mean": double.mean().item(),
        "max_abs": double.abs().max().item(),
    }


def snapshot(critic, optimizer, *, cpu=False):
    """Copy authoritative native tensors; preserve their addresses during restoration."""

    def clone(value):
        value = value.detach().clone()
        return value.cpu() if cpu else value

    return {
        "model": {name: clone(value) for name, value in critic.state_dict().items()},
        "optimizer": {
            name: {key: clone(value) for key, value in optimizer.state[parameter].items()}
            for name, parameter in critic.named_parameters()
        },
    }


def restore(critic, optimizer, saved):
    import torch

    with torch.no_grad():
        for name, value in critic.state_dict().items():
            value.copy_(saved["model"][name])
        for name, parameter in critic.named_parameters():
            parameter.grad = None
            for key, value in optimizer.state[parameter].items():
                value.copy_(saved["optimizer"][name][key])


def state_summary(critic, optimizer):
    import torch

    named = list(critic.named_parameters())
    result = {"parameters": tensor_summary(torch.cat([parameter.detach().flatten() for _, parameter in named]))}
    for key in ("exp_avg", "exp_avg_sq"):
        result[key] = tensor_summary(torch.cat([optimizer.state[parameter][key].flatten() for _, parameter in named]))
    result["steps"] = {name: optimizer.state[parameter]["step"].item() for name, parameter in named}
    return result


def torch_minibatch(critic, optimizer, inputs, old_values, returns, indices, cfg):
    """Native clipped value objective, autograd, critic-wide clipping and Adam."""
    import torch

    values = critic.mlp(inputs[indices])
    old = old_values[indices]
    target = returns[indices]
    clipped = old + (values - old).clamp(-cfg["clip_param"], cfg["clip_param"])
    value_loss = torch.maximum((values - target).square(), (clipped - target).square()).mean()
    optimizer.zero_grad(set_to_none=True)
    (cfg["value_loss_coef"] * value_loss).backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg["max_grad_norm"])
    optimizer.step()
    return value_loss.detach()


def torch_update(critic, optimizer, inputs, old_values, returns, permutation, cfg):
    import torch

    losses = torch.zeros((), device=inputs.device)
    width = inputs.shape[0] // cfg["num_mini_batches"]
    for _ in range(cfg["num_learning_epochs"]):
        for minibatch in range(cfg["num_mini_batches"]):
            indices = permutation[minibatch * width : (minibatch + 1) * width]
            losses += torch_minibatch(critic, optimizer, inputs, old_values, returns, indices, cfg)
    return losses / (cfg["num_learning_epochs"] * cfg["num_mini_batches"])


def timing_summary(samples):
    if len(samples) == 1:
        p10 = p90 = samples[0]
    else:
        quantiles = statistics.quantiles(samples, n=10, method="inclusive")
        p10, p90 = quantiles[0], quantiles[-1]
    return {"median_ms": statistics.median(samples), "p10_ms": p10, "p90_ms": p90, "samples_ms": samples}


def compare_states(reference, actual):
    """Check every native model/Adam tensor, recording tolerances and actual errors."""
    import torch

    checks = {}
    groups = [("model", reference["model"], actual["model"], 3e-4, 2e-5)]
    for key, atol in (("exp_avg", 2e-6), ("exp_avg_sq", 2e-8), ("step", 0.0)):
        groups.append(
            (
                key,
                {name: state[key] for name, state in reference["optimizer"].items()},
                {name: state[key] for name, state in actual["optimizer"].items()},
                0.0 if key == "step" else 5e-4,
                atol,
            )
        )
    for group, expected, observed, rtol, atol in groups:
        if expected.keys() != observed.keys():
            raise RuntimeError(f"Native checkpoint keys differ in {group}")
        for name, value in expected.items():
            other = observed[name]
            checks[f"{group}.{name}"] = {
                "passed": bool(torch.allclose(other, value, rtol=rtol, atol=atol)),
                "max_abs_error": (other.double() - value.double()).abs().max().item(),
                "rtol": rtol,
                "atol": atol,
            }
    return {"passed": all(check["passed"] for check in checks.values()), "checks": checks}


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
    torch.backends.cudnn.benchmark = False
    cfg = to_dict(handle_deprecated_rsl_rl_cfg(KukaAllegroSingleCameraPPORunnerCfg(), version("rsl-rl-lib")))
    algorithm = cfg["algorithm"]
    if (
        cfg["num_steps_per_env"] != 32
        or algorithm["num_mini_batches"] != 8
        or algorithm["num_learning_epochs"] != 5
        or not algorithm["use_clipped_value_loss"]
        or algorithm["schedule"] != "fixed"
    ):
        raise ValueError("Native camera PPO configuration changed; review the benchmark's fixed 40-step contract")
    batch = args.num_envs * cfg["num_steps_per_env"]
    observations = TensorDict(
        {
            name: torch.randn(batch, width, device=args.device) * 0.7 + 0.2
            for name, width in (("policy", 170), ("proprio", 465), ("perception", 960))
        },
        batch_size=[batch],
    )
    critic_class, critic_cfg = resolve_class(cfg["critic"])
    initial_critic = critic_class(observations, cfg["obs_groups"], "critic", 1, **critic_cfg).to(args.device)
    initial_critic.train()
    initial_critic.update_normalization(observations)
    with torch.no_grad():
        inputs = initial_critic.get_latent(observations).contiguous()
        prediction = initial_critic.mlp(inputs)
        old_values = (prediction + torch.randn_like(prediction) * 0.25).contiguous()
        returns = (prediction + torch.randn_like(prediction) * 0.4 + 0.1).contiguous()
    permutation = torch.randperm(batch, device=args.device)
    adam_options = {"fused": True} if args.adam == "fused" else {}
    initial_optimizer = torch.optim.Adam(initial_critic.parameters(), lr=algorithm["learning_rate"], **adam_options)
    # Explicit zero state also supports --burnin_steps 0 without changing the
    # first repeat through lazy optimizer initialization.
    for parameter in initial_critic.parameters():
        initial_optimizer.state[parameter] = {
            "step": torch.zeros((), dtype=torch.float32, device=args.device if args.adam == "fused" else "cpu"),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.zeros_like(parameter),
        }
    width = batch // algorithm["num_mini_batches"]
    for step in range(args.burnin_steps):
        offset = step % algorithm["num_mini_batches"] * width
        torch_minibatch(
            initial_critic,
            initial_optimizer,
            inputs,
            old_values,
            returns,
            permutation[offset : offset + width],
            algorithm,
        )
    torch.cuda.synchronize(args.device)
    paths = [
        Path(__file__),
        ROOT / "source/isaaclab_tasks/isaaclab_tasks/core/lift/config/kuka_allegro/agents/rsl_rl_ppo_cfg.py",
        ROOT / "source/isaaclab_sram/isaaclab_sram/persistent_critic.py",
        ROOT / "source/isaaclab_sram/isaaclab_sram/kernels/ppo_native_critic.cu",
        ROOT / "source/isaaclab_sram/isaaclab_sram/cuda_rt.py",
    ]
    report = {
        "scope": (
            "Standalone native critic on a fixed normalized synthetic rollout; no actor, GAE, simulation or images"
        ),
        "timing_scope": (
            "Complete 40-step update including wrapper pack/flush; restoration and compilation warmup excluded"
        ),
        "cuda_event_note": (
            "Event spans may include host launch gaps; host wall includes CPU wrapper work and final sync"
        ),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines(),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
            if path.exists()
        },
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
        "critic_class": type(initial_critic).__name__,
        "critic_configuration": cfg["critic"],
        "ppo_configuration": algorithm,
        "optimizer_configuration": {
            key: value for key, value in initial_optimizer.param_groups[0].items() if key != "params"
        },
        "batch_size": batch,
        "minibatch_size": width,
        "update_steps": algorithm["num_mini_batches"] * algorithm["num_learning_epochs"],
        "normalization_scope": "Native normalizer updated once before burn-in, then frozen for every measured update",
        "metric_accumulation": "Both backends return a CUDA scalar; host metric conversion occurs after timing",
        "parameter_count": sum(parameter.numel() for parameter in initial_critic.parameters()),
        "inputs": {
            name: tensor_summary(value)
            for name, value in (
                ("normalized_observations", inputs),
                ("old_values", old_values),
                ("returns", returns),
                ("permutation", permutation),
            )
        },
        "initial_state": state_summary(initial_critic, initial_optimizer),
        "results": {},
    }
    outputs = {}
    backends = ("torch", "persistent") if args.backend == "both" else (args.backend,)
    for backend in backends:
        critic = copy.deepcopy(initial_critic)
        optimizer = torch.optim.Adam(critic.parameters(), lr=algorithm["learning_rate"], **adam_options)
        optimizer.load_state_dict(copy.deepcopy(initial_optimizer.state_dict()))
        persistent = None
        if backend == "persistent":
            from isaaclab_sram.persistent_critic import PersistentCritic

            persistent = PersistentCritic(
                critic,
                optimizer,
                **{
                    key: algorithm[key]
                    for key in (
                        "num_mini_batches",
                        "num_learning_epochs",
                        "clip_param",
                        "value_loss_coef",
                        "max_grad_norm",
                    )
                },
            )

        def update():
            if persistent is not None:
                return persistent.update(inputs, old_values, returns, permutation)
            return torch_update(critic, optimizer, inputs, old_values, returns, permutation, algorithm)

        saved = snapshot(critic, optimizer)
        for _ in range(args.warmup):
            restore(critic, optimizer, saved)
            update()
        torch.cuda.synchronize(args.device)
        gpu_ms, wall_ms, losses, signatures = [], [], [], []
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for repeat in range(args.repeats):
            restore(critic, optimizer, saved)
            torch.cuda.synchronize(args.device)
            if args.profile:
                torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push("critic_timed_update")
            tick = time.perf_counter()
            start.record()
            value_loss = update()
            stop.record()
            stop.synchronize()
            wall_ms.append((time.perf_counter() - tick) * 1000)
            gpu_ms.append(start.elapsed_time(stop))
            torch.cuda.nvtx.range_pop()
            if args.profile:
                torch.cuda.cudart().cudaProfilerStop()
            losses.append(float(value_loss))
            summary = state_summary(critic, optimizer)
            signatures.append({key: summary[key]["sha256"] for key in ("parameters", "exp_avg", "exp_avg_sq")})
        outputs[backend] = snapshot(critic, optimizer, cpu=True)
        report["results"][backend] = {
            "cuda_events": timing_summary(gpu_ms),
            "host_wall": timing_summary(wall_ms),
            "value_losses": losses,
            "output_state": summary,
            "repeat_state_sha256": signatures,
            "repeats_bitwise_equal": all(signature == signatures[0] for signature in signatures),
            "all_finite": all(summary[key]["all_finite"] for key in ("parameters", "exp_avg", "exp_avg_sq"))
            and all(math.isfinite(loss) for loss in losses),
            "kernel_resources": persistent.resource_info if persistent is not None else None,
        }
        print(
            json.dumps(
                {
                    "backend": backend,
                    "adam": args.adam,
                    "cuda_events": {key: value for key, value in timing_summary(gpu_ms).items() if key != "samples_ms"},
                    "host_wall": {key: value for key, value in timing_summary(wall_ms).items() if key != "samples_ms"},
                    "value_loss": losses[-1],
                    "all_finite": report["results"][backend]["all_finite"],
                    "repeats_bitwise_equal": report["results"][backend]["repeats_bitwise_equal"],
                }
            ),
            flush=True,
        )
    if args.backend == "both":
        report["parity"] = compare_states(outputs["torch"], outputs["persistent"])
        report["value_loss_max_abs_difference"] = max(
            abs(left - right)
            for left, right in zip(
                report["results"]["torch"]["value_losses"], report["results"]["persistent"]["value_losses"], strict=True
            )
        )
        report["parity"]["value_loss_passed"] = all(
            math.isclose(left, right, rel_tol=3e-4, abs_tol=2e-5)
            for left, right in zip(
                report["results"]["torch"]["value_losses"], report["results"]["persistent"]["value_losses"], strict=True
            )
        )
        report["parity"]["passed"] &= report["parity"]["value_loss_passed"]
        report["cuda_event_median_speedup"] = (
            report["results"]["torch"]["cuda_events"]["median_ms"]
            / report["results"]["persistent"]["cuda_events"]["median_ms"]
        )
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not all(result["all_finite"] for result in report["results"].values()):
        raise RuntimeError(f"Nonfinite output; inspect {args.output}")
    if not all(result["repeats_bitwise_equal"] for result in report["results"].values()):
        raise RuntimeError(f"Repeated identical updates produced different state; inspect {args.output}")
    if "parity" in report and not report["parity"]["passed"]:
        raise RuntimeError(f"Persistent critic differs from native update; inspect {args.output}")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "parity_passed": report["parity"]["passed"] if "parity" in report else None,
                "cuda_event_median_speedup": report.get("cuda_event_median_speedup"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

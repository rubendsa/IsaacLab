# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Screen matched complete native PPO and production unified learner updates.

Run on an otherwise idle GPU; the process monitor screens each complete cycle. This benchmarks PPO.update(), including
normalization, native-equivalent RNG consumption and metric conversion. Rollout
collection, GAE, checkpoint restoration, validation, compilation and simulator
work are excluded. Process polling cannot exclude interference between samples.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import _perception_benchmark as oracle
import torch
from rsl_rl.algorithms import PPO

DESKTOP = {"/usr/lib/x86_64-linux-gnu/dcv/dcvagent", "/usr/libexec/gnome-remote-desktop-daemon"}
REPOSITORY = Path(__file__).resolve().parents[2]


class Monitor:
    """Serialize explicit and background samples so no pending query is missed."""

    def __init__(self):
        self.samples = []
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self):
        with self.lock:
            start = time.monotonic()
            row = {"start": start, "processes": [], "external_pids": [], "returncode": -1}
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                row["returncode"] = result.returncode
                for line in result.stdout.splitlines():
                    if not line.strip():
                        continue
                    pid, name = line.split(",", 1)
                    pid, name = int(pid.strip()), name.strip()
                    row["processes"].append({"pid": pid, "name": name})
                    if pid != os.getpid() and name not in DESKTOP:
                        row["external_pids"].append(pid)
                if result.returncode:
                    row["error"] = result.stderr
            except Exception as exc:
                row["returncode"] = -1
                row["error"] = repr(exc)
            row["end"] = time.monotonic()
            self.samples.append(row)
            return row

    def run(self):
        while not self.stop.is_set():
            sample = self.sample()
            self.stop.wait(max(0.0, 0.1 - (time.monotonic() - sample["start"])))

    def quiet(self):
        now = time.monotonic()
        with self.lock:
            samples = list(self.samples)
        if not samples or now - samples[-1]["end"] > 0.2:
            return False
        since, previous = None, now
        for row in reversed(samples):
            if (
                row["external_pids"]
                or row["returncode"]
                or previous - row["start"] > 0.3
                or row["end"] - row["start"] > 0.3
            ):
                break
            since, previous = row["start"], row["start"]
        return since is not None and now - since >= 0.5

    def interval(self, start, end):
        with self.lock:
            rows = [row for row in self.samples if row["start"] <= end and row["end"] >= start]
        gaps = [b["start"] - a["start"] for a, b in zip(rows, rows[1:])]
        foreign = sorted({pid for row in rows for pid in row["external_pids"]})
        durations = [row["end"] - row["start"] for row in rows]
        boundary_gaps = (
            [max(0.0, rows[0]["start"] - start), max(0.0, end - rows[-1]["end"])] if rows else [end - start] * 2
        )
        valid = len(rows) >= 2 and not foreign and not any(row["returncode"] for row in rows)
        valid = valid and max([*gaps, *durations, *boundary_gaps], default=0.0) <= 0.3
        return {
            "valid": valid,
            "external_pids": foreign,
            "process_samples": rows,
            "max_process_sample_gap_ms": 1000 * max(gaps, default=0.0),
            "max_process_query_duration_ms": 1000 * max(durations, default=0.0),
            "boundary_process_sample_gap_ms": [1000 * gap for gap in boundary_gaps],
        }


def device_state():
    start = time.monotonic()
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw,pstate",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return {
            "start": start,
            "end": time.monotonic(),
            "returncode": result.returncode,
            "columns": ["gpu_index", "sm_clock_MHz", "memory_clock_MHz", "temperature_C", "power_W", "pstate"],
            "csv": result.stdout.strip(),
        }
    except Exception as exc:
        return {"start": start, "end": time.monotonic(), "returncode": -1, "error": repr(exc)}


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def snapshot(algorithm):
    return cpu_copy(algorithm.save())


@torch.no_grad()
def restore(algorithm, checkpoint, rollout, rng, device):
    # Direct native state restoration deliberately preserves warmed wrappers.
    # Their production preparation refreshes parameters/moments on every call.
    algorithm.actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    algorithm.critic.load_state_dict(checkpoint["critic_state_dict"], strict=True)
    algorithm.optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer_state_dict"]))
    algorithm.optimizer.zero_grad(set_to_none=True)
    algorithm.learning_rate = algorithm.optimizer.param_groups[0]["lr"]
    algorithm.storage = copy.deepcopy(rollout)
    torch.random.set_rng_state(rng[0])
    torch.cuda.set_rng_state(rng[1], device)
    assert all(group["fused"] and group["foreach"] is False for group in algorithm.optimizer.param_groups)


def check_state(expected, actual):
    errors = {}
    for role in ("actor_state_dict", "critic_state_dict"):
        assert expected[role].keys() == actual[role].keys()
        for name, value in expected[role].items():
            other = actual[role][name]
            torch.testing.assert_close(other, value, rtol=3e-4, atol=2e-5, msg=f"{role}.{name}")
            errors[f"{role}.{name}"] = (other.double() - value.double()).abs().max().item()
    left, right = expected["optimizer_state_dict"], actual["optimizer_state_dict"]
    assert left["param_groups"] == right["param_groups"], "Optimizer group configuration differs"
    assert left["state"].keys() == right["state"].keys()
    for index, state in left["state"].items():
        assert state.keys() == right["state"][index].keys()
        for key, value in state.items():
            other = right["state"][index][key]
            if key == "step":
                torch.testing.assert_close(other, value, rtol=0, atol=0)
            else:
                torch.testing.assert_close(
                    other, value, rtol=5e-4, atol=2e-6 if key == "exp_avg" else 2e-8, msg=f"Adam[{index}].{key}"
                )
            errors[f"Adam[{index}].{key}"] = (other.double() - value.double()).abs().max().item()
    return {"passed": True, "max_abs_error": max(errors.values()), "per_tensor_max_abs_error": errors}


def hashes():
    return {
        **oracle.source_hashes(),
        str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, help="Accepted cycles; default 6 for two/three backends, 8 for four")
    parser.add_argument("--deadline-seconds", type=float, default=300.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--separate-full", action="store_true", help="Include two-launch production backend")
    parser.add_argument("--resident", action="store_true", help="Include production model/input-resident backend")
    parser.add_argument("--compact", action="store_true", help="Use the same compact history storage for every backend")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    backend_count = 2 + int(args.separate_full) + int(args.resident)
    if args.cycles is None:
        args.cycles = 8 if backend_count == 4 else 6
    if args.cycles % backend_count:
        parser.error("Accepted cycles must be a multiple of backend count to balance measurement order")
    if args.output.exists() or args.cycles < 1 or args.warmup < 1 or not 0 < args.deadline_seconds <= 300:
        parser.error("Use fresh output, positive cycles/warmup, and deadline in (0,300]")
    source_before = hashes()
    if not Path(oracle.unified_learner.__file__).resolve().is_relative_to(REPOSITORY):
        parser.error("isaaclab_sram must be imported from this checkout; use its uv-managed environment")
    report = {
        "scope": (
            "Complete native RGB64 PPO.update at N16/B512: forty actor and critic Adam steps, normalization, "
            "metric conversion, permutation and native-equivalent discarded action RNG included. No simulator claim."
        ),
        "excluded": (
            "Rollout generation, GAE, restore, compilation, warmup, parity validation, "
            "clock/process queries outside update timing."
        ),
        "method": (
            "Identical resumed native checkpoint and rollout each measurement; exact entry RNG restores the "
            "identical permutation; rotate backend order by accepted cycle index; "
            f"{args.cycles} target accepted cycles; retain all rejected cycles."
        ),
        "monitor_policy": (
            "Require 500 ms quiet with latest query within 200 ms and no query duration/sample gap over 300 ms; "
            "sample all CUDA compute processes every 100 ms plus explicit boundaries; require at least two samples "
            "and reject external process, query failure, query duration, sample gap or either boundary gap over "
            "300 ms. Presence sampling cannot prove absence between samples."
        ),
        "timing_scope": (
            "Host wall includes final synchronization. CUDA event interval includes CPU launch/synchronization "
            "gaps, not isolated device execution time."
        ),
        "desktop_process_exclusions": sorted(DESKTOP),
        "pid": os.getpid(),
        "num_envs": 16,
        "batch": 512,
        "adam": "matched fused",
        "storage": "compact for all" if args.compact else "dense for all",
        "target_valid_cycles": args.cycles,
        "warmup_each": args.warmup,
        "deadline_seconds": args.deadline_seconds,
        "source_sha256": source_before,
        "versions": oracle.versions(),
        "source_unchanged": False,
        "git_commit": subprocess.check_output(["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "-C", str(REPOSITORY), "status", "--porcelain"], text=True),
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cycles": [],
        "timings_valid": False,
        "publication_allowed": False,
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)

    monitor = Monitor()
    monitor.thread.start()
    try:
        torch.set_num_threads(1)
        torch.cuda.set_device(args.device)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        report["precision"] = "fp32; TF32 disabled; deterministic algorithms/cuDNN; benchmark/autotuning disabled"
        report["gpu"] = torch.cuda.get_device_name(args.device)
        cfg = oracle.native_configuration()
        cfg.update(multi_gpu=None, torch_compile_mode=None)
        report["native_configuration"] = copy.deepcopy(cfg)
        env = SimpleNamespace(num_envs=16, num_actions=23)
        torch.manual_seed(61)
        initial = oracle.history_stream(16, args.device, 0.0)()
        native = PPO.construct_algorithm(initial, env, copy.deepcopy(cfg), args.device)
        assert type(native) is PPO and native.update.__func__ is PPO.update
        native.train_mode()
        for group in native.optimizer.param_groups:
            group["fused"], group["foreach"] = True, False
        oracle.collect(native, 16, args.device, 0.0)
        native.update()  # Real native warm-start establishes all Adam moments.
        for index, parameter in enumerate([*native.actor.parameters(), *native.critic.parameters()]):
            native.optimizer.state[parameter]["step"].add_(3 * index)
        torch.manual_seed(71)
        last = oracle.collect(native, 16, args.device, 0.5)
        checkpoint = snapshot(native)
        if args.compact:
            oracle.clone_rollout(native, native, compact=True)
        rollout = copy.deepcopy(native.storage)
        torch.manual_seed(81)
        rng = (torch.random.get_rng_state().clone(), torch.cuda.get_rng_state(args.device).clone())
        permutation = torch.randperm(512, requires_grad=False, device=args.device)
        report["permutation_sha256"] = hashlib.sha256(permutation.cpu().numpy().tobytes()).hexdigest()
        report["resumed_step_values"] = [
            int(state["step"]) for state in checkpoint["optimizer_state_dict"]["state"].values()
        ]
        candidates = {"native_ppo_fused_adam": native}
        backends = [("unified", "isaaclab_sram.unified_perception:SramUnifiedPerceptionPPO")]
        if args.separate_full:
            backends.append(
                ("separate_full", "isaaclab_sram.persistent_full_perception:SramPersistentFullPerceptionPPO")
            )
        if args.resident:
            backends.append(("resident", "isaaclab_sram.resident_perception:SramResidentPerceptionPPO"))
        for label, class_name in backends:
            candidate_cfg = copy.deepcopy(cfg)
            candidate_cfg["algorithm"]["class_name"] = class_name
            algorithm = PPO.construct_algorithm(last, env, candidate_cfg, args.device)
            algorithm.train_mode()
            candidates[label] = algorithm
        restore(native, checkpoint, rollout, rng, args.device)
        expected_losses = native.update()
        expected_state = snapshot(native)
        expected_rng = (torch.random.get_rng_state(), torch.cuda.get_rng_state(args.device))

        def validate(algorithm, losses):
            assert torch.equal(torch.random.get_rng_state(), expected_rng[0]), "CPU RNG differs"
            assert torch.equal(torch.cuda.get_rng_state(args.device), expected_rng[1]), "CUDA RNG differs"
            assert losses.keys() == expected_losses.keys()
            for key in losses:
                torch.testing.assert_close(
                    torch.tensor(losses[key]),
                    torch.tensor(expected_losses[key]),
                    rtol=3e-4,
                    atol=2e-5,
                    msg=f"{key} loss",
                )
            return {
                "state_parity": check_state(expected_state, snapshot(algorithm)),
                "losses": losses,
                "cpu_rng_equal": True,
                "cuda_rng_equal": True,
            }

        for _ in range(args.warmup):
            for algorithm in candidates.values():
                restore(algorithm, checkpoint, rollout, rng, args.device)
                validate(algorithm, algorithm.update())
        torch.cuda.synchronize(args.device)
        report["resources"] = {}
        for label, algorithm in candidates.items():
            report["resources"][label] = {
                name: getattr(algorithm, name).resource_info
                for name in ("_unified", "_resident", "_persistent_actor", "_persistent_critic")
                if getattr(algorithm, name, None) is not None
            }
        report["optimizer_flags"] = {
            label: [
                {key: group.get(key) for key in ("fused", "foreach", "capturable")}
                for group in algorithm.optimizer.param_groups
            ]
            for label, algorithm in candidates.items()
        }
        report["algorithm_classes"] = {
            label: f"{type(algorithm).__module__}.{type(algorithm).__qualname__}"
            for label, algorithm in candidates.items()
        }
        report["expected_losses"] = expected_losses
        # The collection deadline starts after initialization/warmup, not compile.
        deadline = time.monotonic() + args.deadline_seconds
        report["measurement_started_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        accepted = 0
        while accepted < args.cycles and time.monotonic() < deadline:
            if not monitor.quiet():
                time.sleep(0.05)
                continue
            start = time.monotonic()
            monitor.sample()
            before = device_state()
            labels = list(candidates)
            rotation = accepted % len(labels)
            order = labels[rotation:] + labels[:rotation]
            rows = []
            for label in order:
                algorithm = candidates[label]
                restore(algorithm, checkpoint, rollout, rng, args.device)
                torch.cuda.synchronize(args.device)
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                host_start = time.monotonic()
                begin.record()
                losses = algorithm.update()
                end.record()
                end.synchronize()
                host_end = time.monotonic()
                row = {
                    "backend": label,
                    "start": host_start,
                    "end": host_end,
                    "host_ms": 1000 * (host_end - host_start),
                    "cuda_ms_including_host_launch_gaps": begin.elapsed_time(end),
                }
                row.update(validate(algorithm, losses))  # All state/RNG checks outside timing.
                rows.append(row)
            after = device_state()
            monitor.sample()
            finish = time.monotonic()
            cycle = {
                "attempt": len(report["cycles"]),
                "start": start,
                "end": finish,
                "order": order,
                "device_before": before,
                "device_after": after,
                "rows": rows,
                **monitor.interval(start, finish),
            }
            cycle["valid"] = cycle["valid"] and not before["returncode"] and not after["returncode"]
            report["cycles"].append(cycle)
            accepted += int(cycle["valid"])
            report["valid_cycle_count"] = accepted
            save()
            print(
                json.dumps(
                    {
                        "attempt": cycle["attempt"],
                        "valid": cycle["valid"],
                        "accepted": accepted,
                        "external_pids": cycle["external_pids"],
                        "host_ms": {row["backend"]: row["host_ms"] for row in rows},
                    }
                ),
                flush=True,
            )
        valid = [cycle for cycle in report["cycles"] if cycle["valid"]]
        report["source_unchanged"] = hashes() == source_before
        assert report["source_unchanged"], "Sources changed during measurement"
        report["timings_valid"] = len(valid) == args.cycles
        report["deadline_reached"] = time.monotonic() >= deadline
        for metric in ("host_ms", "cuda_ms_including_host_launch_gaps"):
            report[f"median_{metric}"] = (
                {
                    label: statistics.median(
                        row[metric] for cycle in valid for row in cycle["rows"] if row["backend"] == label
                    )
                    for label in candidates
                }
                if valid
                else {}
            )
        medians = report["median_host_ms"]
        report["native_over_candidate_host_ratio"] = {
            label: medians["native_ppo_fused_adam"] / value for label, value in medians.items()
        }
        report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    except Exception as exc:
        report["fatal_error"] = repr(exc)
        report["timings_valid"] = False
        raise
    finally:
        monitor.stop.set()
        monitor.thread.join(timeout=10)
        with monitor.lock:
            report["all_process_samples"] = list(monitor.samples)
        save()
    print(
        json.dumps({key: report.get(key) for key in ("timings_valid", "valid_cycle_count", "median_host_ms")}, indent=2)
    )


if __name__ == "__main__":
    main()

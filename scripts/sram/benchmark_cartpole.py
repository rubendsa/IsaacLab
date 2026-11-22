# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark: SRAM-resident megakernel vs stock Isaac Lab training on Isaac-Cartpole-Direct.

Every configuration runs as its own process, strictly sequentially, and is timed end to end
(process start -> trained checkpoint on disk) as well as per training iteration.

Configurations
--------------
``stock_torch``              stock Isaac Lab 3.0: Newton (MuJoCo-Warp) physics, torch env, RSL-RL PPO
``stock_warp``               stock Isaac Lab 3.0 with the experimental Warp env frontend (``--frontend warp``)
``torch_analytic``           ablation: same MDP with closed-form physics as a batched torch VecEnv + stock RSL-RL
``torch_analytic_compiled``  ablation: that env step fused by ``torch.compile`` (CUDA graph) + stock RSL-RL
``sram``                     this prototype: the whole PPO loop in one SRAM-resident CUDA megakernel

Run on a quiet host: stock training is host-latency bound and the megakernel is timed on the GPU, so other GPU
jobs or CPU-heavy processes distort the numbers. ``run``, ``stock_scaling`` and ``locomotion`` watch for other
GPU compute processes and, with ``--quiet_retries N``, repeat a measurement that overlapped one.

Usage::

    python scripts/sram/benchmark_cartpole.py run --out /tmp/sram_bench --seeds 1 2 3 --quiet_retries 3
    python scripts/sram/benchmark_cartpole.py eval --out /tmp/sram_bench
    python scripts/sram/benchmark_cartpole.py scaling --out /tmp/sram_bench
    python scripts/sram/benchmark_cartpole.py stock_scaling --out /tmp/sram_bench
    python scripts/sram/benchmark_cartpole.py report --out /tmp/sram_bench
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY = sys.executable
SRAM_SRC = os.path.join(REPO, "source", "isaaclab_sram")

ITER_RE = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
FIELD_RE = re.compile(r"^\s*([A-Za-z/_ ]+):\s+([-+0-9.eE:]+)s?\s*$")


def _load_json(path: str):
    with open(path) as fh:
        return json.load(fh)


def parse_rsl_rl_log(text: str) -> list[dict]:
    """Parse the per-iteration blocks printed by rsl_rl's OnPolicyRunner."""
    recs, cur = [], None
    for line in text.splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = ITER_RE.search(line)
        if m:
            if cur:
                recs.append(cur)
            cur = {"iteration": int(m.group(1))}
            continue
        if cur is None:
            continue
        f = FIELD_RE.match(line)
        if not f:
            continue
        key, val = f.group(1).strip(), f.group(2)
        mapping = {
            "Steps per second": "steps_per_s",
            "Collection time": "collection_s",
            "Learning time": "learning_s",
            "Mean reward": "mean_episode_return",
            "Mean episode length": "mean_episode_length",
            "Iteration time": "iteration_s",
            "Mean action std": "action_std",
            "Mean value loss": "value_loss",
        }
        if key in mapping and ":" not in val:
            cur[mapping[key]] = float(val)
    if cur:
        recs.append(cur)
    return recs


IGNORED_GPU_APPS = re.compile(r"dcvagent|gnome|Xorg|remote-desktop")


def _gpu_compute_pids() -> dict[int, str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    pids = {}
    for line in out.splitlines():
        pid, _, name = line.partition(",")
        if pid.strip().isdigit() and not IGNORED_GPU_APPS.search(name):
            pids[int(pid)] = name.strip()
    return pids


def _descends_from(pid: int, root: int) -> bool | None:
    """True/False, or None if the process exited before its ancestry could be read."""
    first = True
    for _ in range(64):
        if pid == root:
            return True
        if pid <= 1:
            return False
        try:
            with open(f"/proc/{pid}/stat") as fh:
                pid = int(fh.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            return None if first else False
        first = False
    return False


class ForeignGpuWatch:
    """Records GPU compute processes that are not descendants of this process while a measurement runs.

    Stock Isaac Lab training is host-latency bound and the megakernel is timed on the GPU, so any other GPU job
    during a run invalidates it. ``run`` retries such runs (``--quiet_retries``).
    """

    def __init__(self, period_s: float = 2.0):
        import threading

        self.period_s, self.seen, self.vanished = period_s, {}, {}
        self.polls = self.foreign_polls = 0  # foreign_polls * period_s ~ seconds with another GPU job present
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        me = os.getpid()
        while not self._stop.is_set():
            foreign = False
            for pid, name in _gpu_compute_pids().items():
                ours = _descends_from(pid, me)
                if ours is None:  # exited between nvidia-smi and /proc: one of ours finishing, or a sub-2 s job
                    self.vanished[pid] = name
                elif not ours:
                    self.seen[pid] = name
                    foreign = True
            self.polls += 1
            self.foreign_polls += foreign
            self._stop.wait(self.period_s)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def run_proc(
    cmd: list[str], cwd: str, log_path: str, env: dict | None = None, timeout: int = 3600
) -> tuple[float, str, int]:
    e = dict(os.environ)
    e.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
    if env:
        e.update(env)
    t0 = time.perf_counter()
    p = subprocess.run(
        cmd, cwd=cwd, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout
    )
    wall = time.perf_counter() - t0
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as fh:
        fh.write(p.stdout)
    return wall, p.stdout, p.returncode


def watched(fn, retries: int, tag: str) -> dict:
    """Call ``fn() -> dict`` under a ForeignGpuWatch; repeat (up to *retries* times) while other GPU jobs ran."""
    for attempt in range(retries + 1):
        with ForeignGpuWatch() as watch:
            row = fn()
        row["foreign_gpu_processes"] = sorted(set(watch.seen.values()))
        row["foreign_gpu_seconds"] = watch.foreign_polls * watch.period_s
        row["attempt"] = attempt + 1
        if not watch.seen:
            break
        print(
            f"[{tag}] other GPU processes ran during attempt {attempt + 1}: {row['foreign_gpu_processes']}", flush=True
        )
    return row


SRAM_RUNNER = r"""
import json, sys, time
t0 = time.perf_counter()
sys.path.insert(0, {src!r})
import torch
from isaaclab_sram.cartpole import CartpoleTask, PPOConfig, isaaclab_cartpole_configs
env_cfg, agent_cfg = isaaclab_cartpole_configs()
task = CartpoleTask.from_env_cfg(env_cfg)
ppo = PPOConfig.from_rsl_rl_cfg(agent_cfg)
from isaaclab_sram.megakernel import SramPPOCartpole
t1 = time.perf_counter()
tr = SramPPOCartpole(task, ppo, num_envs={num_envs}, seed={seed})
torch.cuda.synchronize()
t2 = time.perf_counter()
# the GPU drives a display (kernel watchdog on): above 16K envs every iteration gets its own launch, which
# does not change the result (training is bitwise invariant to launch chunking); GPU time includes the gaps
res = tr.train({iterations}, iters_per_launch=None if {num_envs} <= 16384 else 1)
t3 = time.perf_counter()
ckpt = None
try:
    from isaaclab_sram.export import export_rsl_rl_run
    ckpt = export_rsl_rl_run(tr, {out!r})
except Exception as exc:  # exporter optional for timing
    print("export failed:", repr(exc))
t4 = time.perf_counter()
st = res["stats"]
print("SRAM_RESULT " + json.dumps(dict(
    imports_and_cfg_s=t1 - t0, compile_and_init_s=t2 - t1, nvrtc_compile_s=tr.compile_s, train_wall_s=t3 - t2,
    train_gpu_s=res["gpu_time_s"], export_s=t4 - t3, env_steps=res["env_steps"], checkpoint=str(ckpt) if ckpt else None,
    describe=tr.describe(),
    per_iter=[dict(iteration=i, mean_episode_return=float(st["mean_episode_return"][i]),
                   mean_episode_length=float(st["mean_episode_length"][i]), action_std=float(st["action_std"][i]),
                   value_loss=float(st["value_loss"][i]), learning_rate=float(st["learning_rate"][i]))
              for i in range(len(st["mean_episode_return"]))])))
"""


def run_config(config: str, seed: int, args, out_dir: str) -> dict:
    """Run one configuration; retry up to ``--quiet_retries`` times if a foreign GPU process was seen meanwhile."""
    tag = f"{config}_s{seed}"
    retries = getattr(args, "quiet_retries", 0)
    for attempt in range(retries + 1):
        work = os.path.join(out_dir, "work", tag)
        if os.path.exists(work):  # keep an earlier attempt's files, but out of the checkpoint search path
            k = 1
            while os.path.exists(f"{work}.prev{k}"):
                k += 1
            os.rename(work, f"{work}.prev{k}")
        with ForeignGpuWatch() as watch:
            rec = _run_config_once(config, seed, args, out_dir)
        rec["foreign_gpu_processes"] = sorted(set(watch.seen.values()))
        rec["foreign_gpu_seconds"] = watch.foreign_polls * watch.period_s
        rec["attempt"] = attempt + 1
        if not watch.seen:
            break
        print(
            f"[{tag}] other GPU processes ran during attempt {attempt + 1}: {rec['foreign_gpu_processes']}", flush=True
        )
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    print(
        f"[{tag}] rc={rec.get('returncode')} wall={rec.get('process_wall_s', 0):.2f}s "
        f"loop={rec.get('sum_iteration_s', 0):.3f}s it_med={1e3 * (rec.get('median_iteration_s') or 0):.2f}ms",
        flush=True,
    )
    return rec


def _run_config_once(config: str, seed: int, args, out_dir: str) -> dict:
    tag = f"{config}_s{seed}"
    work = os.path.join(out_dir, "work", tag)
    os.makedirs(work, exist_ok=True)
    log = os.path.join(out_dir, "logs", tag + ".log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    rec = {"config": config, "seed": seed, "num_envs": args.num_envs, "iterations": args.iterations}
    if config in ("stock_torch", "stock_warp"):
        cmd = [
            PY,
            os.path.join(REPO, "scripts", "reinforcement_learning", "train.py"),
            "--rl_library",
            "rsl_rl",
            "--task",
            "Isaac-Cartpole-Direct",
            "--num_envs",
            str(args.num_envs),
            "--max_iterations",
            str(args.iterations),
            "--seed",
            str(seed),
            "--visualizer",
            "none",
            "--run_name",
            tag,
        ]
        if config == "stock_warp":
            cmd += ["--frontend", "warp"]
        cmd += ["physics=newton_mjwarp"]
        wall, out, rc = run_proc(cmd, work, log)
        it = parse_rsl_rl_log(out)
        m = re.search(r"Training time:\s*([0-9.]+)\s*seconds", out)
        ck = sorted(
            glob.glob(os.path.join(work, "logs", "rsl_rl", "*", f"*{tag}*", "model_*.pt")),
            key=lambda p: int(re.search(r"model_(\d+)", p).group(1)),
        )
        rec.update(
            process_wall_s=wall,
            returncode=rc,
            learn_wall_s=float(m.group(1)) if m else None,
            per_iter=it,
            checkpoint=ck[-1] if ck else None,
        )
    elif config in ("torch_analytic", "torch_analytic_compiled"):
        script = os.path.join(REPO, "scripts", "sram", "baselines", "train_rsl_rl_analytic.py")
        mode = (
            ["--mode", "eager"]
            if config == "torch_analytic"
            else ["--mode", "compiled", "--compile_mode", "reduce-overhead"]
        )
        cmd = [
            PY,
            script,
            "--num_envs",
            str(args.num_envs),
            "--max_iterations",
            str(args.iterations),
            "--seed",
            str(seed),
            "--eval_steps",
            "0",
            "--run_name",
            tag,
        ] + mode
        wall, out, rc = run_proc(cmd, work, log)
        it = parse_rsl_rl_log(out)
        ck = sorted(
            glob.glob(os.path.join(work, "logs", "**", "model_*.pt"), recursive=True),
            key=lambda p: int(re.search(r"model_(\d+)", p).group(1)),
        )
        rec.update(process_wall_s=wall, returncode=rc, per_iter=it, checkpoint=ck[-1] if ck else None)
    elif config == "sram":
        code = SRAM_RUNNER.format(
            src=SRAM_SRC, num_envs=args.num_envs, seed=seed, iterations=args.iterations, out=os.path.join(work, "logs")
        )
        wall, out, rc = run_proc([PY, "-c", code], work, log)
        m = re.search(r"SRAM_RESULT (.*)$", out, re.M)
        res = json.loads(m.group(1)) if m else {}
        rec.update(process_wall_s=wall, returncode=rc, **res)
    else:
        raise ValueError(config)
    # derived quantities
    if rec.get("per_iter") and config != "sram":
        its = rec["per_iter"]
        # collection + learning (1 ms resolution) rather than RSL-RL's 10-ms-resolution "Iteration time"
        times = [
            r["collection_s"] + r["learning_s"] if "collection_s" in r and "learning_s" in r else r.get("iteration_s")
            for r in its
        ]
        times = [x for x in times if x is not None]
        rec["sum_iteration_s"] = sum(times)
        steady = times[3:] if len(times) > 6 else times
        rec["median_iteration_s"] = statistics.median(steady) if steady else None
        rec["median_collection_s"] = statistics.median(
            [r["collection_s"] for r in its[3:] if "collection_s" in r] or [0]
        )
        rec["median_learning_s"] = statistics.median([r["learning_s"] for r in its[3:] if "learning_s" in r] or [0])
    if config == "sram" and rec.get("train_gpu_s"):
        rec["sum_iteration_s"] = rec["train_gpu_s"]
        rec["median_iteration_s"] = rec["train_gpu_s"] / args.iterations
    return rec


def cmd_run(args):
    os.makedirs(args.out, exist_ok=True)
    for seed in args.seeds:
        for config in args.configs:
            run_config(config, seed, args, args.out)


SCALING_RUNNER = r"""
import json, sys
sys.path.insert(0, {src!r})
import torch
from isaaclab_sram.megakernel import SramPPOCartpole
out = []
for n, storage in {cases!r}:
    try:
        tr = SramPPOCartpole(num_envs=n, seed=1, storage=storage)
        # the GPU drives a display (kernel watchdog on): long iterations get one launch each, which
        # does not change the result (training is bitwise invariant to launch chunking)
        per = None if n <= 16384 else 1
        tr.train(3, iters_per_launch=per)          # warm-up (GPU clocks, L2)
        r = tr.train({iterations}, iters_per_launch=per)
        st = r["stats"]
        out.append(dict(num_envs=n, storage=tr.describe()["rollout_storage"], describe=tr.describe(),
                        gpu_s=r["gpu_time_s"], iterations={iterations}, env_steps_per_s=r["env_steps_per_s"],
                        ms_per_iteration=1e3 * r["gpu_time_s"] / {iterations},
                        final_return=float(st["mean_episode_return"][-5:].mean())))
    except Exception as exc:
        out.append(dict(num_envs=n, storage=storage, error=repr(exc)))
    print("SCALE " + json.dumps(out[-1]), flush=True)
"""


def cmd_scaling(args):
    os.makedirs(args.out, exist_ok=True)
    cases = [(int(n), s) for n in args.env_counts for s in args.storages]
    code = SCALING_RUNNER.format(src=SRAM_SRC, cases=cases, iterations=args.iterations)
    wall, out, rc = run_proc([PY, "-c", code], args.out, os.path.join(args.out, "scaling.log"), timeout=7200)
    rows = [json.loads(m.group(1)) for m in re.finditer(r"^SCALE (.*)$", out, re.M)]
    with open(os.path.join(args.out, "scaling.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    for r in rows:
        print(
            r
            if "error" in r
            else f"{r['num_envs']:>7} {r['storage']:<28} {r['ms_per_iteration']:8.3f} ms/it "
            f"{r['env_steps_per_s'] / 1e6:8.1f} M steps/s  return {r['final_return']:.2f}"
        )


def cmd_eval(args):
    """Evaluate every trained checkpoint with the same protocol in the stock Isaac Lab Newton env."""
    for path in sorted(glob.glob(os.path.join(args.out, "*_s[0-9]*.json"))):
        tag = os.path.basename(path)[: -len(".json")]
        if args.tags and tag not in args.tags:
            continue
        rec = _load_json(path)
        ck = rec.get("checkpoint")
        if not ck or not os.path.exists(ck):
            print(f"[eval] {tag}: no checkpoint, skipped", flush=True)
            continue
        out_json = os.path.join(args.out, "eval", f"{tag}.json")
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        cmd = [
            PY,
            os.path.join(REPO, "scripts", "sram", "eval_policy.py"),
            "--checkpoint",
            ck,
            "--num_envs",
            str(args.num_envs),
            "--num_episodes",
            str(args.num_episodes),
            "--seed",
            str(args.eval_seed),
            "--label",
            tag,
            "--output",
            out_json,
        ]
        wall, out, rc = run_proc(cmd, os.path.join(args.out, "work"), os.path.join(args.out, "logs", f"eval_{tag}.log"))
        if rc != 0 or not os.path.exists(out_json):
            print(f"[eval] {tag}: failed rc={rc}", flush=True)
            continue
        fe = _load_json(out_json)["first_episodes"]
        rec["eval"] = {
            k: fe.get(k)
            for k in (
                "num_episodes",
                "return_mean",
                "return_std",
                "length_mean",
                "timeout_fraction",
                "terminated_fraction",
            )
        }
        rec["eval"]["seed"] = args.eval_seed
        with open(path, "w") as fh:
            json.dump(rec, fh, indent=1)
        print(
            f"[eval] {tag}: return {fe['return_mean']:.4f} +- {fe['return_std']:.4f} "
            f"terminated {fe['terminated_fraction']:.5f}",
            flush=True,
        )


LOCO_TASKS = {  # key -> (task, extra Hydra overrides)
    "anymal_c": ("IsaacContrib-Velocity-Flat-AnymalC", []),
    "g1": ("Isaac-Velocity-Flat-G1", []),
    # DexSuite (Kuka-Allegro lift): state and single-camera RGB64 / depth64 (Newton Warp renderer)
    "dex_state": ("Isaac-Lift-KukaAllegro", []),
    "dex_rgb64": ("Isaac-Lift-KukaAllegro-Camera", ["renderer=newton_renderer", "presets=single_camera,rgb64"]),
    "dex_depth64": ("Isaac-Lift-KukaAllegro-Camera", ["renderer=newton_renderer", "presets=single_camera,depth64"]),
}


def cmd_locomotion(args):
    """Stock Isaac Lab per-iteration timing for locomotion tasks (input of the projection model)."""
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for key in args.tasks:
        task, extra = LOCO_TASKS[key]
        work = os.path.join(args.out, "work", f"loco_{key}")
        os.makedirs(work, exist_ok=True)
        cmd = [
            PY,
            os.path.join(REPO, "scripts", "reinforcement_learning", "train.py"),
            "--rl_library",
            "rsl_rl",
            "--task",
            task,
            "--num_envs",
            str(args.num_envs),
            "--max_iterations",
            str(args.iterations),
            "--seed",
            "1",
            "--visualizer",
            "none",
            "--run_name",
            f"loco_{key}",
            "physics=newton_mjwarp",
        ] + extra

        def once(cmd=cmd, work=work, key=key, task=task):
            wall, out, rc = run_proc(cmd, work, os.path.join(args.out, "logs", f"loco_{key}.log"))
            its = parse_rsl_rl_log(out)[2:]  # drop warm-up iterations (CUDA graph capture, JIT)
            med = lambda k: statistics.median([r[k] for r in its if k in r]) if its else None  # noqa: E731
            return dict(
                task_key=key,
                task=task,
                num_envs=args.num_envs,
                returncode=rc,
                process_wall_s=wall,
                collection_s=med("collection_s"),
                learning_s=med("learning_s"),
                iteration_s=med("iteration_s"),
                steps_per_s=med("steps_per_s"),
                iterations_measured=len(its),
            )

        row = watched(once, args.quiet_retries, f"loco_{key}")
        rows.append(row)
        print(row, flush=True)
    with open(os.path.join(args.out, "locomotion.json"), "w") as fh:
        json.dump(rows, fh, indent=1)


def cmd_stock_scaling(args):
    """Stock Isaac Lab (Newton + RSL-RL) iteration time vs environment count (throughput only)."""
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for n in args.env_counts:
        tag = f"stock_scaling_{n}"
        work = os.path.join(args.out, "work", tag)
        os.makedirs(work, exist_ok=True)
        cmd = [
            PY,
            os.path.join(REPO, "scripts", "reinforcement_learning", "train.py"),
            "--rl_library",
            "rsl_rl",
            "--task",
            "Isaac-Cartpole-Direct",
            "--num_envs",
            str(n),
            "--max_iterations",
            str(args.iterations),
            "--seed",
            "1",
            "--visualizer",
            "none",
            "--run_name",
            tag,
            "physics=newton_mjwarp",
        ]

        def once(cmd=cmd, work=work, tag=tag, n=n):
            wall, out, rc = run_proc(cmd, work, os.path.join(args.out, "logs", f"{tag}.log"))
            its = parse_rsl_rl_log(out)[3:]  # drop warm-up iterations
            # collection + learning (1 ms resolution) rather than RSL-RL's 10-ms-resolution "Iteration time"
            times = [r["collection_s"] + r["learning_s"] for r in its if "collection_s" in r and "learning_s" in r]
            it_s = statistics.median(times) if times else None
            return dict(
                num_envs=n,
                returncode=rc,
                process_wall_s=wall,
                iterations_measured=len(its),
                median_iteration_s=it_s,
                env_steps_per_s=(n * 16 / it_s) if it_s else None,
                median_collection_s=statistics.median([r["collection_s"] for r in its if "collection_s" in r] or [0]),
                median_learning_s=statistics.median([r["learning_s"] for r in its if "learning_s" in r] or [0]),
            )

        row = watched(once, args.quiet_retries, tag)
        rows.append(row)
        print(row, flush=True)
    with open(os.path.join(args.out, "stock_scaling.json"), "w") as fh:
        json.dump(rows, fh, indent=1)


def cmd_report(args):
    recs = [_load_json(p) for p in sorted(glob.glob(os.path.join(args.out, "*_s[0-9]*.json")))]
    by = {}
    for r in recs:
        by.setdefault(r["config"], []).append(r)
    print("| config | seeds | process wall [s] | training loop [s] | median iteration [ms] | final train return |")
    print("|---|---|---|---|---|---|")
    for cfg, rs in by.items():

        def f(k, rs=rs):
            vals = [r[k] for r in rs if r.get(k) is not None]
            return statistics.median(vals) if vals else float("nan")

        fr = (
            statistics.median([r["per_iter"][-1]["mean_episode_return"] for r in rs if r.get("per_iter")])
            if rs
            else float("nan")
        )
        print(
            f"| {cfg} | {len(rs)} | {f('process_wall_s'):.2f} | {f('sum_iteration_s'):.3f} "
            f"| {1e3 * f('median_iteration_s'):.2f} | {fr:.3f} |"
        )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    r.add_argument(
        "--configs",
        nargs="+",
        default=["stock_torch", "stock_warp", "torch_analytic", "torch_analytic_compiled", "sram"],
    )
    r.add_argument("--iterations", type=int, default=150)
    r.add_argument("--num_envs", type=int, default=4096)
    r.add_argument(
        "--quiet_retries",
        type=int,
        default=0,
        help="re-run a configuration up to this many times when another GPU process ran during it",
    )
    s = sub.add_parser("scaling")
    s.add_argument("--out", required=True)
    s.add_argument(
        "--env_counts", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
    )
    s.add_argument("--storages", nargs="+", default=["auto"])
    s.add_argument("--iterations", type=int, default=20)
    ev = sub.add_parser("eval")
    ev.add_argument("--out", required=True)
    ev.add_argument("--num_envs", type=int, default=4096)
    ev.add_argument("--num_episodes", type=int, default=2)
    ev.add_argument("--eval_seed", type=int, default=10002)
    ev.add_argument("--tags", nargs="*", default=None, help="evaluate only these runs, e.g. sram_s1 stock_torch_s2")
    lo = sub.add_parser("locomotion")
    lo.add_argument("--out", required=True)
    lo.add_argument("--tasks", nargs="+", default=["anymal_c", "g1"])
    lo.add_argument("--num_envs", type=int, default=4096)
    lo.add_argument("--iterations", type=int, default=12)
    lo.add_argument("--quiet_retries", type=int, default=0)
    ss = sub.add_parser("stock_scaling")
    ss.add_argument("--out", required=True)
    ss.add_argument(
        "--env_counts", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
    )
    ss.add_argument("--iterations", type=int, default=20)
    ss.add_argument("--quiet_retries", type=int, default=0)
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    {
        "run": cmd_run,
        "scaling": cmd_scaling,
        "eval": cmd_eval,
        "stock_scaling": cmd_stock_scaling,
        "locomotion": cmd_locomotion,
        "report": cmd_report,
    }[args.cmd](args)


if __name__ == "__main__":
    main()

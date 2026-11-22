# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Turn a ``run_benchmarks.sh`` output directory into the tables of docs/sram/SRAM_TRAINING.md.

Every number is recomputed from the raw records:

* stock and ablation runs: RSL-RL's per-iteration collection + learning times (1 ms resolution; its "Iteration
  time" field has only 10 ms resolution), median over iterations 4-150, then median over seeds;
* SRAM runs: CUDA-event GPU time of the training launch divided by the number of iterations;
* runs that overlapped a foreign GPU process in ``contention.log`` are flagged with a dagger.

``--export DIR`` also writes a copy of the measurements without host names, user paths or checkpoint paths
(the data behind the report, ``docs/sram/data/results``).

Usage::

    python scripts/sram/summarize_benchmarks.py --bench <dir> [--out summary.md] [--export docs/sram/data/results]
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_cartpole import parse_rsl_rl_log  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ORDER = ["stock_torch", "stock_warp", "torch_analytic", "torch_analytic_compiled", "sram"]
LABELS = {
    "stock_torch": "Stock Isaac Lab 3.0: Newton MuJoCo-Warp, torch env, RSL-RL",
    "stock_warp": "Stock, experimental Warp env frontend (`--frontend warp`)",
    "torch_analytic": "Ablation: same MDP as a batched torch env, stock RSL-RL",
    "torch_analytic_compiled": "Ablation: that env fused by `torch.compile` + CUDA graph, stock RSL-RL",
    "sram": "**SRAM megakernel (this branch)**",
}
TASK_LABELS = {
    "cartpole": "Cartpole",
    "anymal_c": "ANYmal-C flat",
    "g1": "Unitree G1 flat",
    "dex_state": "DexSuite Kuka-Allegro lift, state",
    "dex_rgb64": "DexSuite Kuka-Allegro lift, RGB 64×64 camera",
    "dex_depth64": "DexSuite Kuka-Allegro lift, depth 64×64 camera",
}
STEPS_PER_ENV = 16  # Isaac-Cartpole-Direct RSL-RL config
THRESHOLD = 4.5


def load(path):
    with open(path) as fh:
        return json.load(fh)


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def foreign_sightings(bench):
    """Times at which a GPU compute process from outside this checkout was running."""
    out = []
    path = os.path.join(bench, "contention.log")
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            ts, _, rest = line.strip().partition(" ")
            procs = [p for p in rest.split("gpu_apps=", 1)[-1].split(";") if p.strip()]
            if any(REPO not in p for p in procs):
                out.append(dt.datetime.fromisoformat(ts))
    return out


def contended(path, rec, sightings, pad_s=5.0):
    """True if a foreign GPU process was seen while this run's process was alive.

    Uses the run's own watcher when it has one, otherwise ``contention.log`` (5 s monitor period).

    The run ends when its training log is written (``eval`` rewrites the record later, so not the record's mtime).
    """
    if "foreign_gpu_processes" in rec:  # the run watched for other GPU processes itself (every 2 s)
        return bool(rec["foreign_gpu_processes"])
    tag = os.path.basename(path)[: -len(".json")]
    log = os.path.join(os.path.dirname(path), "logs", f"{tag}.log")
    end = dt.datetime.fromtimestamp(os.path.getmtime(log if os.path.exists(log) else path), tz=dt.timezone.utc)
    start = end - dt.timedelta(seconds=rec.get("process_wall_s") or 0)
    pad = dt.timedelta(seconds=pad_s)
    return any(start - pad <= s <= end + pad for s in sightings)


def iter_times(rec):
    its = rec.get("per_iter") or []
    if rec["config"] == "sram":
        return [rec["train_gpu_s"] / max(1, len(its))] * len(its)
    return [
        (r["collection_s"] + r["learning_s"]) if "collection_s" in r and "learning_s" in r else r.get("iteration_s")
        for r in its
    ]


def first_crossing(rec, threshold):
    """(iteration index, cumulative training-loop seconds) at which the mean training return first reaches it."""
    cum = 0.0
    for i, (r, t) in enumerate(zip(rec.get("per_iter") or [], iter_times(rec))):
        cum += t or 0.0
        if (r.get("mean_episode_return") or -1e9) >= threshold:
            return i + 1, cum
    return None, None


def fmt_t(s):
    if s is None:
        return "-"
    if s < 0.01:
        return f"{1e3 * s:.2f} ms"
    if s < 1:
        return f"{1e3 * s:.0f} ms"
    return f"{s:.1f} s"


def summarize_runs(bench, sightings):
    runs = {}
    for p in sorted(glob.glob(os.path.join(bench, "*_s[0-9]*.json"))):
        r = load(p)
        r["_contended"] = contended(p, r, sightings)
        runs.setdefault(r["config"], []).append(r)
    rows = {}
    for cfg in ORDER:
        rs = runs.get(cfg, [])
        if not rs:
            continue
        per_seed = [med(iter_times(r)[3:]) for r in rs]
        coll = [med([x.get("collection_s") for x in (r.get("per_iter") or [])[3:]]) for r in rs]
        learn = [med([x.get("learning_s") for x in (r.get("per_iter") or [])[3:]]) for r in rs]
        cross = [first_crossing(r, THRESHOLD) for r in rs]
        ev = [r["eval"]["return_mean"] for r in rs if r.get("eval")]
        rows[cfg] = dict(
            seeds=[r["seed"] for r in rs],
            per_iteration_s=med(per_seed),
            per_iteration_s_by_seed=per_seed,
            collection_s=med(coll) if cfg != "sram" else None,
            learning_s=med(learn) if cfg != "sram" else None,
            loop_s=med([sum(x for x in iter_times(r) if x) for r in rs]),
            process_wall_s=med([r.get("process_wall_s") for r in rs]),
            iterations_to_threshold=med([c[0] for c in cross]),
            iterations_to_threshold_by_seed=[c[0] for c in cross],
            time_to_threshold_s=med([c[1] for c in cross]),
            eval_return_by_seed=ev,
            eval_return_mean=statistics.mean(ev) if ev else None,
            eval_return_spread=(max(ev) - min(ev)) if len(ev) > 1 else None,
            final_train_return=med(
                [
                    statistics.mean([x["mean_episode_return"] for x in r["per_iter"][-10:]])
                    for r in rs
                    if r.get("per_iter")
                ]
            ),
            contended_seeds=[r["seed"] for r in rs if r["_contended"]],
        )
    return runs, rows


def stock_scaling_rows(bench, base_row):
    """Stock throughput vs env count, recomputed from the training logs with 1 ms resolution.

    Every stock measurement at an env count is a candidate: the sweeps in ``<bench>`` and ``<bench>/stock_rep*/``
    (median of iterations 4-20) and the 150-iteration training runs (main runs at 4,096 envs, ``envs_<N>/``; median of
    iterations 4-150 per seed). The fastest candidate without another GPU job during it is used: host load the GPU
    watcher cannot see (other processes' CPU work) only ever slows stock down, so this favours the stock baseline. If
    every candidate overlapped another GPU job, the fastest one is used and flagged as contended.
    """
    by_n = {}
    for d in [bench] + sorted(glob.glob(os.path.join(bench, "stock_rep*"))):
        path = os.path.join(d, "stock_scaling.json")
        if not os.path.exists(path):
            continue
        for r in load(path):
            log = os.path.join(d, "logs", f"stock_scaling_{r['num_envs']}.log")
            if os.path.exists(log):
                with open(log) as fh:
                    its = parse_rsl_rl_log(fh.read())[3:]
                t = med([x["collection_s"] + x["learning_s"] for x in its if "collection_s" in x and "learning_s" in x])
            else:
                t = r.get("median_iteration_s")
            if t:
                by_n.setdefault(r["num_envs"], []).append(
                    dict(t=t, contended=bool(r.get("foreign_gpu_processes")), foreign_s=r.get("foreign_gpu_seconds"))
                )
    # the 150-iteration training runs at the same env counts (main runs, time-to-policy runs) are measurements too
    for d, n_fixed in [(bench, 4096)] + [
        (p, int(p.rsplit("_", 1)[1])) for p in glob.glob(os.path.join(bench, "envs_*"))
    ]:
        for p in glob.glob(os.path.join(d, "stock_torch_s[0-9]*.json")):
            r = load(p)
            t = med(iter_times(r)[3:])
            if t:
                by_n.setdefault(n_fixed, []).append(
                    dict(t=t, contended=bool(r.get("foreign_gpu_processes")), foreign_s=r.get("foreign_gpu_seconds"))
                )
    out = []
    for n, cands in by_n.items():
        clean = [c for c in cands if not c["contended"]]
        best = min(clean or cands, key=lambda c: c["t"])
        out.append(
            dict(
                num_envs=n,
                iteration_s=best["t"],
                iteration_s_by_sweep=[c["t"] for c in cands],
                contended=best["contended"],
                foreign_gpu_seconds=best["foreign_s"],
                env_steps_per_s=n * STEPS_PER_ENV / best["t"],
            )
        )
    return sorted(out, key=lambda r: r["num_envs"])


def time_to_policy_rows(bench, sightings, main_rows):
    out = []
    for d in sorted(glob.glob(os.path.join(bench, "envs_*")), key=lambda p: int(p.rsplit("_", 1)[1])):
        _, rows = summarize_runs(d, sightings)
        out.append(dict(num_envs=int(d.rsplit("_", 1)[1]), **rows))
    if out and all(r["num_envs"] != 4096 for r in out):
        out.append(dict(num_envs=4096, **{k: v for k, v in main_rows.items() if k in ("sram", "stock_torch")}))
    return sorted(out, key=lambda r: r["num_envs"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--export", default=None, help="write the measurements without host-specific fields here")
    a = ap.parse_args()
    sightings = foreign_sightings(a.bench)
    runs, rows = summarize_runs(a.bench, sightings)
    L = []
    n_seeds = max((len(v["seeds"]) for v in rows.values()), default=0)
    L.append(f"### Cartpole: 4,096 envs, 150 PPO iterations, {n_seeds} seeds per configuration\n")
    L.append(
        "| Configuration | Time per PPO iteration | Collection | Learning | 150 iterations (training loop) "
        f"| Process wall time | Iterations to return ≥ {THRESHOLD} | Eval return in the stock env |"
    )
    L.append("|---|---|---|---|---|---|---|---|")
    for cfg, r in rows.items():
        flag = " †" if r["contended_seeds"] else ""
        ev = (
            f"{r['eval_return_mean']:.3f} ({min(r['eval_return_by_seed']):.3f}-{max(r['eval_return_by_seed']):.3f})"
            if r["eval_return_by_seed"]
            else "-"
        )
        it_c = "/".join(str(x) if x is not None else "-" for x in r["iterations_to_threshold_by_seed"])
        L.append(
            f"| {LABELS[cfg]}{flag} | {fmt_t(r['per_iteration_s'])} | {fmt_t(r['collection_s'])} | "
            f"{fmt_t(r['learning_s'])} | {fmt_t(r['loop_s'])} | {fmt_t(r['process_wall_s'])} | {it_c} | {ev} |"
        )
    st, sr = rows.get("stock_torch"), rows.get("sram")
    if st and sr:
        L.append(
            f"\nSRAM vs stock: **{st['per_iteration_s'] / sr['per_iteration_s']:.0f}×** per PPO iteration, "
            f"{st['loop_s'] / sr['loop_s']:.0f}× for the 150-iteration training loop, "
            f"{st['process_wall_s'] / sr['process_wall_s']:.1f}× in process wall time (start-up dominated). "
            f"Time to a mean training return ≥ {THRESHOLD}: {fmt_t(st['time_to_threshold_s'])} stock, "
            f"{fmt_t(sr['time_to_threshold_s'])} SRAM."
        )
    if any(r["contended_seeds"] for r in rows.values()):
        L.append(
            "\n† a foreign GPU process was seen during seed(s) "
            + "; ".join(f"{c}: {r['contended_seeds']}" for c, r in rows.items() if r["contended_seeds"])
        )
    # scaling
    sc_path = os.path.join(a.bench, "scaling.json")
    stock_rows = stock_scaling_rows(a.bench, st)
    sc = [r for r in load(sc_path) if "error" not in r] if os.path.exists(sc_path) else []
    if sc:
        stock_by_n = {r["num_envs"]: r for r in stock_rows}
        L.append("\n### Throughput vs environment count (env steps per second, PPO updates included)\n")
        L.append(
            "| Envs | SRAM rollout buffer | SRAM time per iteration | SRAM steps/s "
            "| Stock time per iteration | Stock steps/s | SRAM / stock |"
        )
        L.append("|---|---|---|---|---|---|---|")
        for r in sc:
            s = stock_by_n.get(r["num_envs"])
            tier = "shared memory" if "shared" in r["storage"] else "global (L2 / DRAM)"
            if s and s.get("contended"):
                tier += " (stock †)"
            L.append(
                f"| {r['num_envs']:,} | {tier} | {r['ms_per_iteration']:.2f} ms | {r['env_steps_per_s'] / 1e6:.1f} M | "
                + (
                    f"{fmt_t(s['iteration_s'])} | {s['env_steps_per_s'] / 1e6:.2f} M | "
                    f"{r['env_steps_per_s'] / s['env_steps_per_s']:.0f}× |"
                    if s
                    else "- | - | - |"
                )
            )
    # time to policy vs env count (runs in envs_<N>/ plus the main 4,096-env runs)
    ttp = time_to_policy_rows(a.bench, sightings, rows)
    if ttp:
        L.append(f"\n### Time to a mean training return ≥ {THRESHOLD} vs environment count (median over seeds)\n")
        L.append(
            "| Envs | SRAM: iterations | SRAM: time | Stock: iterations | Stock: time per iteration "
            "| Stock: time | Stock / SRAM time |"
        )
        L.append("|---|---|---|---|---|---|---|")
        for r in ttp:
            s, m = r.get("stock_torch") or {}, r.get("sram") or {}
            ratio = (
                f"{s['time_to_threshold_s'] / m['time_to_threshold_s']:.0f}×"
                if s.get("time_to_threshold_s") and m.get("time_to_threshold_s")
                else "-"
            )
            flag = " †" if (s.get("contended_seeds") or m.get("contended_seeds")) else ""
            L.append(
                f"| {r['num_envs']:,}{flag} | {m.get('iterations_to_threshold') or '-'} | "
                f"{fmt_t(m.get('time_to_threshold_s'))} | {s.get('iterations_to_threshold') or '-'} | "
                f"{fmt_t(s.get('per_iteration_s'))} | {fmt_t(s.get('time_to_threshold_s'))} | {ratio} |"
            )
    # stock locomotion / DexSuite timings
    lo_path = os.path.join(a.bench, "locomotion.json")
    if os.path.exists(lo_path):
        L.append("\n### Stock Isaac Lab 3.0 per-iteration time, 4,096 envs (median, warm-up excluded)\n")
        L.append("| Task | Collection | Learning | Iteration | Env steps/s |")
        L.append("|---|---|---|---|---|")
        for r in load(lo_path):
            name = TASK_LABELS.get(r.get("task_key"), r.get("task"))
            if r.get("collection_s") is None:
                L.append(f"| {name} | failed (exit code {r.get('returncode')}) | | | |")
                continue
            it = r["collection_s"] + r["learning_s"]
            L.append(
                f"| {name} | {fmt_t(r['collection_s'])} | {fmt_t(r['learning_s'])} | {fmt_t(it)} | "
                f"{(r.get('steps_per_s') or 0) / 1e3:,.0f} k |"
            )
    # PPO update cost
    uc_path = os.path.join(a.bench, "update_cost.json")
    if os.path.exists(uc_path):
        uc = load(uc_path)
        L.append("\n### The PPO update alone: stock RSL-RL vs the same math in one CUDA graph (4,096 envs, TF32)\n")
        L.append(
            "| Task | GFLOP per update | Stock RSL-RL | Kernel launches | CUDA graph | Graph speedup "
            "| Achieved TFLOP/s (graph) |"
        )
        L.append("|---|---|---|---|---|---|---|")
        for key in ("cartpole", "anymal_c", "g1"):
            e = uc.get(key)
            if not e:
                continue
            m = e.get("tf32_on") or e.get("tf32") or {}
            L.append(
                f"| {TASK_LABELS[key]} | {e.get('update_gflop', 0):.1f} | {fmt_t(m.get('stock_s'))} | "
                f"{m.get('stock_host_launches', '-'):,} | {fmt_t(m.get('graphed_s'))} | "
                f"{m['stock_s'] / m['graphed_s']:.1f}× | {m.get('graphed_tflops', 0):.1f} |"
            )
        if sr and uc.get("cartpole"):
            g = (uc["cartpole"].get("tf32_on") or {}).get("graphed_s")
            if g:
                L.append(
                    "\nThe megakernel runs a whole Cartpole iteration (rollout + update) in "
                    f"{fmt_t(sr['per_iteration_s'])}, {g / sr['per_iteration_s']:.0f}× less than the graph-captured "
                    "update alone."
                )
    pj = os.path.join(a.bench, "projection.json")
    if os.path.exists(pj):
        L.append("\n### Projection (model, not a measurement)\n")
        L.append(load(pj)["markdown"])
    text = "\n".join(L) + "\n"
    print(text)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
    if a.export:
        export(a.bench, a.export, runs, rows, stock_rows, sc, ttp)


SCRUB_KEYS = {
    "checkpoint",
    "host",
    "argv",
    "out",
    "footprint_dir",
    "nvidia_smi_start",
    "nvidia_smi_end",
    "loadavg",
    # profiler-truncated CUDA kernel names, not used by the report
    "gemm_kernel_names",
    "top_kernels_by_count",
    "top_kernels_ms",
}


def _scrub(x):
    if isinstance(x, dict):
        return {k: _scrub(v) for k, v in x.items() if k not in SCRUB_KEYS and not k.startswith("_")}
    if isinstance(x, list):
        return [_scrub(v) for v in x]
    if isinstance(x, str) and ("/home/" in x or "/Users/" in x):
        return re.sub(r"(/home|/Users)/[^ ,;:'\"]*", "<path>", x)
    return x


def export(bench, out_dir, runs, rows, stock_rows, scaling, ttp):
    os.makedirs(out_dir, exist_ok=True)

    def dump(name, obj):
        with open(os.path.join(out_dir, name), "w") as fh:
            json.dump(_scrub(obj), fh, indent=1)
            fh.write("\n")

    dump("cartpole_summary.json", rows)
    dump("cartpole_runs.json", {cfg: rs for cfg, rs in runs.items()})
    if scaling:
        dump("scaling_sram.json", scaling)
    if stock_rows:
        dump("scaling_stock.json", stock_rows)
    if ttp:
        dump("time_to_policy.json", ttp)
    # every training run behind the report: attempts and overlap with other GPU jobs (the measurement-hygiene notes)
    attempts = []
    for d in [bench] + sorted(glob.glob(os.path.join(bench, "envs_*"))):
        for p in sorted(glob.glob(os.path.join(d, "*_s[0-9]*.json"))):
            r = load(p)
            attempts.append(
                dict(
                    run=os.path.relpath(p, bench)[: -len(".json")],
                    attempts=r.get("attempt", 1),
                    other_gpu_job_during_kept_attempt=bool(r.get("foreign_gpu_processes")),
                    other_gpu_job_seconds=r.get("foreign_gpu_seconds"),
                )
            )
    originals = (
        sorted(
            f[: -len(".json")] for f in os.listdir(os.path.join(bench, "contended_originals")) if f.endswith(".json")
        )
        if os.path.isdir(os.path.join(bench, "contended_originals"))
        else []
    )
    dump("run_attempts.json", dict(first_pass_runs_repeated=originals, runs=attempts))
    for name in (
        "locomotion.json",
        "update_cost.json",
        "projection.json",
        "verify_4096.json",
        "verify_4096_warm.json",
        "profile_4096.json",
    ):
        p = os.path.join(bench, name)
        if os.path.exists(p):
            dump(name.replace("locomotion.json", "stock_iteration_times.json"), load(p))
    prov = os.path.join(bench, "provenance.txt")
    if os.path.exists(prov):
        with open(prov) as fh:
            lines = fh.read().splitlines()
        # drop the host name (second line) and keep date, GPU, versions and the git state
        keep = [ln for i, ln in enumerate(lines) if i != 1 and "/home/" not in ln]
        with open(os.path.join(out_dir, "provenance.txt"), "w") as fh:
            fh.write("\n".join(keep) + "\n")
    print("exported to", out_dir)


if __name__ == "__main__":
    main()

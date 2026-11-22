# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""How fast would SRAM-resident training be for locomotion? (A model with stated assumptions.)

Combines MEASURED inputs:
  * per-iteration stock training times (collection / learning) for ANYmal-C flat and G1 flat on
    Newton MJWarp + RSL-RL at 4,096 envs (``benchmark_cartpole.py locomotion`` output),
  * the training working set and per-step physics FLOPs of those tasks (docs/sram/data/locomotion_footprint.json,
    measured by instantiating the tasks and walking every device buffer; FLOPs are an analytic count
    of the MuJoCo-Warp pipeline at the observed contact / solver-iteration counts),
  * the PPO-update cost of those networks: stock RSL-RL vs the same update captured in CUDA graphs, and its FLOP
    count (scripts/sram/projection/measure_update_cost.py output, optional; without it the update is counted as
    6 FLOPs per MAC),
  * this GPU's measured on-chip capacities / bandwidths and the achieved efficiency of the Cartpole
    megakernel,
with three explicit efficiency scenarios for the parts that do not exist yet (a persistent,
SRAM-resident physics pipeline for articulated bodies with contacts and a generic SRAM learner).

Usage::

    python scripts/sram/projection/project_locomotion.py --stock <bench>/locomotion.json \
        [--update_cost <scratch>/update_cost.json] [--sram_cartpole <bench>/sram_s1.json] --out <md>
"""

from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FOOTPRINT = os.path.join(HERE, "..", "..", "..", "docs", "sram", "data", "locomotion_footprint.json")

# efficiency scenarios for code that would have to be written (fraction of the measured FP32 FFMA peak,
# 112.8 TFLOP/s on this GPU), and the per-sync overhead of a persistent kernel (measured: grid barrier
# 0.73 us; Cartpole megakernel all-reduce 4.4 us per mini-batch incl. two barriers)
SCENARIOS = {
    "conservative": dict(physics_eff=0.02, learner_eff=0.08, pad=3.0),  # 2.3 / 9 TFLOP/s, 3x padded FLOPs
    "expected": dict(physics_eff=0.05, learner_eff=0.25, pad=2.0),  # 5.6 / 28 TFLOP/s
    "optimistic": dict(physics_eff=0.12, learner_eff=0.45, pad=1.5),  # 13.5 / 51 TFLOP/s (TF32 MMA)
}
FP32_PEAK = 112.8e12
SYNC_PER_MINIBATCH_S = 4.4e-6
SYNC_PER_ENV_STEP_S = 2.0e-6


def _load_json(path: str):
    with open(path) as fh:
        return json.load(fh)


def project(task: dict, stock: dict, upd: dict | None) -> dict:
    n, T = task["num_envs"], task["num_steps_per_env"]
    substeps = task["mjwarp_steps_per_env_step"]
    flops_phys = n * T * substeps * task["physics_flops_per_mjwarp_step"]
    flops_act = n * T * task.get("actuator_flops_per_env_step", 0.0)
    macs = task["actor_macs"] + task["critic_macs"]
    flops_inf = n * T * 2 * macs
    nmb = task["num_learning_epochs"] * task["num_mini_batches"]
    if upd and upd.get("update_gflop"):  # counted by measure_update_cost.py (no input gradient for the first layer)
        flops_upd = upd["update_gflop"] * 1e9
    else:  # fwd (2) + bwd (4) FLOPs per MAC
        flops_upd = task["num_learning_epochs"] * n * T * 6 * macs
    out = dict(
        task=task["name"],
        working_set_mib=task["lean_working_set_bytes"] / 2**20,
        fits=task["fits"],
        flops_gflop=dict(
            physics=flops_phys / 1e9, actuator=flops_act / 1e9, inference=flops_inf / 1e9, update=flops_upd / 1e9
        ),
        stock_s=dict(
            collection=stock["collection_s"],
            learning=stock["learning_s"],
            iteration=stock["collection_s"] + stock["learning_s"],
        ),
    )
    if upd:
        out["measured_update"] = {k: upd[k] for k in ("update_gflop", "tf32_on", "tf32_off") if k in upd}
    sc = {}
    for name, p in SCENARIOS.items():
        phys = p["pad"] * (flops_phys + flops_act) / (p["physics_eff"] * FP32_PEAK) + T * SYNC_PER_ENV_STEP_S
        inf = flops_inf / (p["learner_eff"] * FP32_PEAK)
        upd_s = flops_upd / (p["learner_eff"] * FP32_PEAK) + nmb * SYNC_PER_MINIBATCH_S
        tot = phys + inf + upd_s
        sc[name] = dict(
            physics_s=phys,
            inference_s=inf,
            update_s=upd_s,
            iteration_s=tot,
            speedup_vs_stock=out["stock_s"]["iteration"] / tot,
            learner_only_speedup=out["stock_s"]["iteration"] / (stock["collection_s"] + inf + upd_s),
        )
    out["scenarios"] = sc
    return out


def to_markdown(rows: list[dict]) -> str:
    L = [
        "| task | working set @4096 (lean) | stock iteration | SRAM learner only (expected) "
        "| everything SRAM-resident: conservative / expected / optimistic |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        s = r["scenarios"]
        L.append(
            f"| {r['task']} | {r['working_set_mib']:.1f} MiB ({r['fits']}) | {1e3 * r['stock_s']['iteration']:.0f} ms "
            f"| {1e3 * r['stock_s']['iteration'] / s['expected']['learner_only_speedup']:.0f} ms "
            f"({s['expected']['learner_only_speedup']:.2f}x) | "
            + " / ".join(
                f"{1e3 * s[k]['iteration_s']:.1f} ms ({s[k]['speedup_vs_stock']:.0f}x)"
                for k in ("conservative", "expected", "optimistic")
            )
            + " |"
        )
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", required=True, help="locomotion.json from benchmark_cartpole.py locomotion")
    ap.add_argument("--update_cost", default=None)
    ap.add_argument("--footprint", default=FOOTPRINT)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    fp = _load_json(a.footprint)
    stock = {r["task_key"]: r for r in _load_json(a.stock)}
    upd = _load_json(a.update_cost) if a.update_cost and os.path.exists(a.update_cost) else {}
    rows = []
    for key, task in fp["tasks"].items():
        if key in stock:
            rows.append(project(task, stock[key], upd.get(key)))
    md = to_markdown(rows)
    print(md)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(dict(scenarios=SCENARIOS, rows=rows, markdown=md), fh, indent=1)


if __name__ == "__main__":
    main()

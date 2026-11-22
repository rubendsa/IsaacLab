# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the report figures (light + dark variants) from benchmark_cartpole.py outputs.

Charts follow a fixed mark spec: 2 px lines, >= 8 px markers with a 2 px surface ring, solid hairline
grids, one y-axis, a legend for >= 2 series plus direct labels, text in ink tokens (never the series
color) and a validated categorical order (blue, orange, aqua, yellow; adjacent CVD dE >= 8 in both
modes).  Every chart has a table twin in docs/sram/SRAM_TRAINING.md.

Usage::

    python scripts/sram/make_report_figures.py --bench <dir from benchmark_cartpole.py> --out docs/sram/figures
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

THEMES = {
    "light": dict(
        surface="#fcfcfb",
        ink="#0b0b0b",
        ink2="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
        gray="#898781",
    ),
    "dark": dict(
        surface="#1a1a19",
        ink="#f0efec",
        ink2="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        series=["#3987e5", "#d95926", "#199e70", "#c98500"],
        gray="#898781",
    ),
}
# fixed entity -> slot mapping (color follows the entity, never its rank)
CONFIGS = [
    ("sram", "SRAM megakernel (this branch)"),
    ("stock_torch", "Stock Isaac Lab 3.0 (Newton + RSL-RL)"),
    ("stock_warp", "Stock, Warp env frontend"),
    ("torch_analytic_compiled", "Ablation: compiled analytic env + RSL-RL"),
    ("torch_analytic", "Ablation: analytic torch env + RSL-RL"),
]
CURVES = ["sram", "stock_torch", "stock_warp", "torch_analytic_compiled"]  # <= 4 series on the line chart
LABEL = dict(CONFIGS)
SHORT = {
    "sram": "SRAM megakernel",
    "stock_torch": "Stock Isaac Lab 3.0",
    "stock_warp": "Stock + Warp env",
    "torch_analytic_compiled": "Compiled analytic env + RSL-RL",
    "torch_analytic": "Analytic env + RSL-RL",
}
SLOT = {c: i for i, c in enumerate(CURVES)}
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summarize_benchmarks as sb  # noqa: E402

PX = 72.0 / 100.0  # 1 px at the 100-dpi layout scale, in points
# label for global-memory rollouts: the 128 MiB L2 of the RTX PRO 6000 Blackwell holds a rollout buffer of up to
# ~100 MiB next to the kernel's other data (measured: throughput is flat up to 131K envs = 96 MiB, then drops)
L2_RESIDENT_BYTES = 100 * 2**20


def _load_json(path: str):
    with open(path) as fh:
        return json.load(fh)


def style(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.spines["bottom"].set_linewidth(1 * PX)
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], length=0, labelsize=9)
    ax.grid(True, which="major", color=t["grid"], linewidth=1 * PX, linestyle="-")
    ax.set_axisbelow(True)


def new_fig(t, w=8.0, h=4.6):
    plt.rcParams["font.family"] = ["Ubuntu", "Liberation Sans", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(w, h), dpi=100)
    fig.patch.set_facecolor(t["surface"])
    style(ax, t)
    return fig, ax


def title(fig, ax, t, text, sub):
    ax.set_title(text, loc="left", color=t["ink"], fontsize=12.5, fontweight="semibold", pad=26)
    ax.text(0.0, 1.035, sub, transform=ax.transAxes, color=t["ink2"], fontsize=9.5, va="bottom")


def load_runs(bench):
    runs = {}
    for p in sorted(glob.glob(os.path.join(bench, "*_s[0-9]*.json"))):
        r = _load_json(p)
        runs.setdefault(r["config"], []).append(r)
    return runs


def curve(r):
    """(cumulative training-loop seconds, mean training-episode return) per iteration."""
    its = r.get("per_iter") or []
    if r["config"] == "sram":
        dt = r["train_gpu_s"] / max(1, len(its))
        t = np.arange(1, len(its) + 1) * dt
    else:
        # collection + learning time have 1 ms resolution in RSL-RL's log ("Iteration time" only 10 ms)
        t = np.cumsum(
            [
                (x["collection_s"] + x["learning_s"])
                if "collection_s" in x and "learning_s" in x
                else x.get("iteration_s", np.nan)
                for x in its
            ]
        )
    y = np.array([x.get("mean_episode_return", np.nan) for x in its], dtype=float)
    return t, y


def fig_learning(runs, t, out):
    fig, ax = new_fig(t, h=5.2)
    ax.set_xscale("log")
    for cfg in CURVES:
        rs = [r for r in runs.get(cfg, []) if r.get("per_iter")]
        if not rs:
            continue
        cs = [curve(r) for r in rs]
        n = min(len(c[0]) for c in cs)
        tt = np.median(np.stack([c[0][:n] for c in cs]), 0)
        yy = np.median(np.stack([c[1][:n] for c in cs]), 0)
        col = t["series"][SLOT[cfg]]
        total = tt[-1]
        took = f"{total:.1f} s" if total >= 1 else f"{1e3 * total:.0f} ms"
        # the legend carries identity + total time (end labels would collide where the baselines converge)
        ax.plot(
            tt,
            yy,
            color=col,
            linewidth=2 * PX,
            solid_joinstyle="round",
            solid_capstyle="round",
            label=f"{SHORT[cfg]}: {took}",
            zorder=3,
        )
        ax.plot(
            tt[-1:],
            yy[-1:],
            "o",
            color=col,
            markersize=8 * PX,
            markeredgecolor=t["surface"],
            markeredgewidth=2 * PX,
            zorder=4,
        )
        if cfg == "sram":
            ax.annotate(
                f"trained in {took}",
                (tt[-1], yy[-1]),
                xytext=(8, -16),
                textcoords="offset points",
                color=t["ink"],
                fontsize=9,
            )
    ax.set_xlabel("training-loop wall-clock time [s] (log scale)", color=t["ink2"], fontsize=9.5)
    ax.set_ylabel("mean training-episode return", color=t["ink2"], fontsize=9.5)
    ax.set_ylim(top=5.6)
    # below the plot: inside, it would cover the stock curves, which converge in the lower right
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        labelcolor=t["ink"],
    )
    title(
        fig,
        ax,
        t,
        "Same task, same PPO: SRAM megakernel vs stock Isaac Lab on Cartpole",
        "Isaac-Cartpole-Direct, 4,096 envs, 150 PPO iterations, median over seeds; dots mark the end of training",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=t["surface"])
    plt.close(fig)


def fig_iteration(main_rows, t, out):
    """Median time per PPO iteration per configuration (summarize_benchmarks.summarize_runs rows)."""
    fig, ax = new_fig(t, h=3.4)
    ax.set_xscale("log")
    rows = [
        (cfg, main_rows[cfg]["per_iteration_s"]) for cfg, _ in CONFIGS if main_rows.get(cfg, {}).get("per_iteration_s")
    ]
    rows = rows[::-1]
    for i, (cfg, v) in enumerate(rows):
        col = t["series"][0] if cfg == "sram" else t["gray"]  # emphasis: one hue, the rest gray
        ax.hlines(i, 1e-4, v, color=t["grid"], linewidth=2 * PX, zorder=2)
        ax.plot(
            [v], [i], "o", color=col, markersize=10 * PX, markeredgecolor=t["surface"], markeredgewidth=2 * PX, zorder=3
        )
        ax.annotate(
            f"{1e3 * v:.2f} ms" if v < 0.01 else f"{1e3 * v:.0f} ms",
            (v, i),
            xytext=(8, -3.5),
            textcoords="offset points",
            color=t["ink"],
            fontsize=9,
        )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABEL[c] for c, _ in rows], color=t["ink2"], fontsize=9)
    ax.grid(False, axis="y")
    ax.set_xlim(1e-4, 1.0)
    ax.set_xlabel(
        "time per PPO iteration (65,536 transitions + 20 mini-batch updates), log scale [s]",
        color=t["ink2"],
        fontsize=9,
    )
    title(fig, ax, t, "Time per PPO iteration", "median over steady-state iterations and seeds")
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=t["surface"])
    plt.close(fig)


def fig_scaling(rows, t, out, stock_ref=None, stock_rows=None):
    rows = [r for r in rows if "error" not in r]
    if not rows:
        return
    fig, ax = new_fig(t)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    x = np.array([r["num_envs"] for r in rows])
    y = np.array([r["env_steps_per_s"] for r in rows])
    col = t["series"][0]
    ax.plot(x, y, color=col, linewidth=2 * PX, zorder=3)
    ax.plot(x, y, "o", color=col, markersize=8 * PX, markeredgecolor=t["surface"], markeredgewidth=2 * PX, zorder=4)
    for r in rows:
        tier = (
            "SMEM"
            if "shared" in r["storage"]
            else ("L2" if r["describe"]["rollout_buffer_bytes_total"] <= L2_RESIDENT_BYTES else "DRAM")
        )
        ax.annotate(
            tier,
            (r["num_envs"], r["env_steps_per_s"]),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            color=t["ink2"],
            fontsize=8,
        )
    if stock_rows:  # stock Isaac Lab measured at several environment counts (second series)
        sx = np.array([r["num_envs"] for r in stock_rows])
        sy = np.array([r["env_steps_per_s"] for r in stock_rows])
        o = np.argsort(sx)
        col2 = t["series"][1]
        ax.plot(sx[o], sy[o], color=col2, linewidth=2 * PX, zorder=3, label="Stock Isaac Lab 3.0 (Newton + RSL-RL)")
        ax.plot(
            sx[o],
            sy[o],
            "o",
            color=col2,
            markersize=8 * PX,
            markeredgecolor=t["surface"],
            markeredgewidth=2 * PX,
            zorder=4,
        )
        ax.lines[0].set_label("SRAM megakernel")
        ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=t["ink"])
    elif stock_ref:
        ax.axhline(stock_ref, color=t["gray"], linewidth=1 * PX)
        ax.annotate(
            f"stock Isaac Lab @4,096 envs: {stock_ref / 1e6:.2f} M steps/s",
            (x[0], stock_ref),
            xytext=(0, 5),
            textcoords="offset points",
            color=t["ink2"],
            fontsize=8.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v // 1024}K" if v >= 1024 else str(v) for v in x])
    ax.set_xlabel("parallel environments", color=t["ink2"], fontsize=9.5)
    ax.set_ylabel("env steps / s, PPO updates included", color=t["ink2"], fontsize=9.5)
    title(
        fig,
        ax,
        t,
        "Training throughput vs. environment count",
        "labels: where the megakernel's rollout buffer lives: SMEM (shared memory), L2 or DRAM",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=t["surface"])
    plt.close(fig)


def fig_time_to_policy(rows, t, out, threshold):
    """Time to a mean training return >= threshold vs env count, SRAM and stock (median over seeds)."""
    series = [("sram", "SRAM megakernel"), ("stock_torch", "Stock Isaac Lab 3.0 (Newton + RSL-RL)")]
    fig, ax = new_fig(t)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    xs_all = []
    for cfg, label in series:
        pts = [(r["num_envs"], (r.get(cfg) or {}).get("time_to_threshold_s")) for r in rows]
        pts = [(x, y) for x, y in pts if y]
        if not pts:
            continue
        x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
        xs_all += list(x)
        col = t["series"][SLOT[cfg]]
        ax.plot(x, y, color=col, linewidth=2 * PX, zorder=3, label=label)
        ax.plot(x, y, "o", color=col, markersize=8 * PX, markeredgecolor=t["surface"], markeredgewidth=2 * PX, zorder=4)
        for xi, yi in zip(x, y):  # values are few (4-5 per series): label each point
            ax.annotate(
                f"{yi:.1f} s" if yi >= 1 else f"{1e3 * yi:.0f} ms",
                (xi, yi),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                color=t["ink"],
                fontsize=8.5,
            )
    xs = sorted(set(xs_all))
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v // 1024}K" if v >= 1024 else str(v) for v in xs])
    ax.set_xlabel("parallel environments", color=t["ink2"], fontsize=9.5)
    ax.set_ylabel(f"training-loop time to mean return ≥ {threshold} [s], log scale", color=t["ink2"], fontsize=9.5)
    # left, between the two series (the right side holds the value labels of the larger env counts)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.62), frameon=False, fontsize=8.5, labelcolor=t["ink"])
    title(
        fig,
        ax,
        t,
        "Time to a working Cartpole policy vs. environment count",
        "median over 3 seeds; ~30-50 PPO iterations at every env count, so the time follows the time per iteration",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=t["surface"])
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = load_runs(a.bench)
    scaling = (
        _load_json(os.path.join(a.bench, "scaling.json"))
        if os.path.exists(os.path.join(a.bench, "scaling.json"))
        else []
    )
    sightings = sb.foreign_sightings(a.bench)
    _, main_rows = sb.summarize_runs(a.bench, sightings)
    stock_it = main_rows.get("stock_torch", {}).get("per_iteration_s")
    stock_ref = 4096 * 16 / stock_it if stock_it else None
    # stock throughput vs env count, recomputed from the training logs (1 ms resolution)
    stock_rows = sb.stock_scaling_rows(a.bench, main_rows.get("stock_torch"))
    ttp = sb.time_to_policy_rows(a.bench, sightings, main_rows)
    for mode, t in THEMES.items():
        if runs:
            fig_learning(runs, t, os.path.join(a.out, f"learning_curves_{mode}.png"))
            fig_iteration(main_rows, t, os.path.join(a.out, f"iteration_time_{mode}.png"))
        if scaling:
            fig_scaling(scaling, t, os.path.join(a.out, f"scaling_{mode}.png"), stock_ref, stock_rows)
        if ttp:
            fig_time_to_policy(ttp, t, os.path.join(a.out, f"time_to_policy_{mode}.png"), sb.THRESHOLD)
    print("figures written to", a.out)


if __name__ == "__main__":
    main()

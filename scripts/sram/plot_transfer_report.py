# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render offline transfer-report figures from archived measurements; never imports CUDA.

From the repository root::

    uv run python scripts/sram/plot_transfer_report.py

Writes three JPEG figures, plotted_samples.csv, statistics.csv, and provenance.json
under docs/sram/figures/transfer. For PNG/SVG exports, use --formats png svg --out /tmp/transfer.
Dependencies: matplotlib and numpy. Input files are read without modification.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
DATA = "docs/sram/data/"
BLUE = "#236CB0"
GREEN = "#26865D"
ORANGE = "#D47A1F"
PURPLE = "#7752B5"
RED = "#B84B57"
INK = "#172B43"
MUTED = "#526278"
GPU = "RTX PRO 6000 Blackwell Server Edition"
COLORS = {"torch": BLUE, "sram": GREEN, "sram_graph": ORANGE, "sram_unified": PURPLE}
LABELS = {"torch": "Native PyTorch", "sram": "Fused operations", "sram_graph": "Fused + graph"}


class Evidence:
    """Read-only inputs and exact values behind every plotted statistic."""

    def __init__(self):
        self.inputs = {}
        self.samples = []
        self.statistics = []

    def load(self, relative):
        path = ROOT / relative
        raw = path.read_bytes()
        decoded = gzip.decompress(raw) if path.suffix == ".gz" else raw
        self.inputs[relative] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "decoded_sha256": hashlib.sha256(decoded).hexdigest(),
            "bytes": len(raw),
        }
        return json.loads(decoded)

    def record(self, figure, scope, backend, metric, values, *, indices=None, unit="ms", summary_only=False):
        values = np.asarray(values, dtype=float)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(f"Missing/nonfinite evidence: {scope}/{backend}/{metric}")
        if indices is None:
            indices = range(len(values))
        for index, value in zip(indices, values, strict=True):
            self.samples.append(
                {
                    "figure": figure,
                    "scope": scope,
                    "backend": backend,
                    "metric": metric,
                    "sample_index": index,
                    "value": float(value),
                    "unit": unit,
                    "record_kind": "archived_median" if summary_only else "raw_sample",
                }
            )
        result = {
            "figure": figure,
            "scope": scope,
            "backend": backend,
            "metric": metric,
            "n": "100 repeats; median only" if summary_only else len(values),
            "unit": unit,
            "median": float(np.median(values)),
            "p10": "" if summary_only else float(np.quantile(values, 0.1, method="linear")),
            "p90": "" if summary_only else float(np.quantile(values, 0.9, method="linear")),
            "minimum": "" if summary_only else float(values.min()),
            "maximum": "" if summary_only else float(values.max()),
        }
        self.statistics.append(result)
        return result


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#AAB5C3",
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "svg.fonttype": "none",
            "svg.hashsalt": "isaaclab-transfer-report-v1",
            "savefig.facecolor": "white",
        }
    )


def title(fig, heading, subtitle):
    fig.suptitle(heading, x=0.065, y=0.976, ha="left", fontsize=19, fontweight="bold")
    fig.text(0.065, 0.933, subtitle, color=MUTED, fontsize=10.5, ha="left")


def grid(ax, *, latency=True):
    ax.grid(axis="y", color="#E6EBF1", linewidth=0.7)
    ax.set_axisbelow(True)
    if latency:
        ax.set_ylim(bottom=0)


def footer(fig, lines):
    fig.text(0.065, 0.095, "\n".join(lines), fontsize=9, color=MUTED, linespacing=1.5, va="top")


def rolling(ax, rows, field, color, label):
    xs = np.array([r["iteration"] for r in rows])
    ys = np.array([r[field] for r in rows]) * 1000
    ax.plot(xs, ys, color=color, alpha=0.24, linewidth=0.8)
    medians = np.array([np.median(ys[i - 4 : i + 5]) for i in range(4, len(ys) - 4)])
    ax.plot(xs[4:-4], medians, color=color, linewidth=2.0, label=label)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Latency per iteration (ms)")
    grid(ax)


def distribution(ax, values, y, color, *, label=True):
    """Every sample, descriptive p10–p90 range and median; never a confidence interval."""
    values = np.asarray(values, dtype=float)
    jitter = np.random.default_rng(17).uniform(-0.11, 0.11, len(values))
    ax.scatter(values, y + jitter, s=12, color=color, alpha=0.33, linewidths=0, zorder=2)
    p10, median, p90 = np.quantile(values, [0.1, 0.5, 0.9], method="linear")
    ax.plot([p10, p90], [y, y], color=color, linewidth=4, solid_capstyle="round", zorder=3)
    ax.plot(median, y, marker="|", color=INK, markersize=15, markeredgewidth=2.1, zorder=4)
    if label:
        ax.annotate(
            f"{median:,.2f} ms",
            (median, y),
            xytext=(6, 11),
            textcoords="offset points",
            color=color,
            fontsize=9.5,
            weight="bold",
        )


def distribution_axes(ax, positions, labels, xmax):
    ax.set_yticks(positions, labels)
    ax.set_xlim(0, xmax)
    ax.invert_yaxis()
    ax.set_xlabel("Synchronized host latency (ms; lower is better)")
    ax.grid(axis="x", color="#E6EBF1", linewidth=0.7)
    ax.set_axisbelow(True)


def camera(evidence, path, backend, envs, count):
    report = evidence.load(DATA + path)
    if (report["backend"], report["num_envs"], report["iterations"]) != (backend, envs, count):
        raise ValueError(f"Unexpected camera experiment: {path}")
    if not report["integration_passed"] or report["warmup_iterations"] != 5:
        raise ValueError(f"Incomplete/unexpected camera experiment: {path}")
    if [r["iteration"] for r in report["rows"]] != list(range(count)):
        raise ValueError(f"Nonconsecutive camera rows: {path}")
    return report


def save(fig, out, name, formats):
    for extension in formats:
        path = out / f"{name}.{extension}"
        if extension == "jpg":
            fig.savefig(path, dpi=160, pil_kwargs={"quality": 90, "optimize": True})
        elif extension == "png":
            fig.savefig(path, dpi=180, metadata={"Software": "Isaac Lab archived-evidence plotter"})
        else:
            fig.savefig(path, metadata={"Date": None, "Title": name})
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def native_4096(evidence, out, formats):
    name = "native_4096"
    reports = {
        b: camera(evidence, f"perception/{b}_rgb64_4096_200.json", b, 4096, 200)
        for b in ("torch", "sram", "sram_graph")
    }
    comparison = evidence.load(DATA + "perception/comparison_4096_200.json")
    assert comparison["identical_source_sha256"]
    fig, axs = plt.subplots(2, 2, figsize=(14.5, 10.5))
    fig.subplots_adjust(left=0.075, right=0.975, top=0.84, bottom=0.19, hspace=0.46, wspace=0.31)
    title(
        fig,
        "Native RGB64 task · 4,096 environments",
        f"{GPU}  |  FP32 model; TF32 library operations allowed  |  seed 42; 200 iterations",
    )
    for backend, report in reports.items():
        rows = report["rows"][5:]
        for ax, field in zip(axs[0], ("iteration_s", "learning_s"), strict=True):
            rolling(ax, rows, field, COLORS[backend], LABELS[backend])
            result = evidence.record(
                name,
                "real_camera_N4096",
                backend,
                field,
                [r[field] * 1000 for r in rows],
                indices=[r["iteration"] for r in rows],
            )
            expected = comparison["runs"][backend]["all"][field]["median_ms"]
            np.testing.assert_allclose(result["median"], expected, rtol=1e-12)
        reward = [r for r in report["rows"] if "mean_episode_reward" in r]
        axs[1, 0].plot(
            [r["iteration"] for r in reward],
            [r["mean_episode_reward"] for r in reward],
            color=COLORS[backend],
            linewidth=1.5,
        )
        evidence.record(
            name,
            "real_camera_N4096",
            backend,
            "logged_mean_episode_reward",
            [r["mean_episode_reward"] for r in reward],
            indices=[r["iteration"] for r in reward],
            unit="reward",
        )
    axs[0, 0].set_title("A  Complete iteration: raw + 9-row median")
    axs[0, 1].set_title("B  Learner phase: raw + 9-row median")
    for ax in axs[0]:
        ax.set_xlim(5, 199)
        ax.set_xticks([5, 50, 100, 150, 199])
    axs[1, 0].set_title("C  Logged reward: training trace, not convergence")
    axs[1, 0].set_xlabel("Training iteration")
    axs[1, 0].set_ylabel("Mean episode reward (logger running window)")
    axs[1, 0].set_xlim(0, 199)
    grid(axs[1, 0], latency=False)
    ax = axs[1, 1]
    ax.set_title("D  All measured latencies + median / p10–p90")
    labels, positions = [], []
    for group, field in enumerate(("iteration_s", "learning_s")):
        for i, backend in enumerate(reports):
            y = i + group * 3.7
            distribution(ax, [r[field] * 1000 for r in reports[backend]["rows"][5:]], y, COLORS[backend], label=False)
            median = np.median([r[field] * 1000 for r in reports[backend]["rows"][5:]])
            ax.text(1680, y, f"{median:,.1f}", va="center", ha="right", color=COLORS[backend], fontsize=9.5)
            labels.append(
                ("Total · " if group == 0 else "Learner · ")
                + {"torch": "native", "sram": "fused", "sram_graph": "graph"}[backend]
            )
            positions.append(y)
    distribution_axes(ax, positions, labels, 1730)
    ax.set_ylim(6.5, -0.7)
    ax.set_xlabel("Host latency (ms; medians at right)")
    fig.legend(
        handles=[Line2D([0], [0], color=COLORS[b], lw=2, label=LABELS[b]) for b in reports],
        loc="upper left",
        bbox_to_anchor=(0.063, 0.905),
        ncol=3,
        frameon=False,
        columnspacing=3,
    )
    footer(
        fig,
        [
            (
                "Timing rows 5–199 only: 195 samples/run; startup, compilation warmup, logging and "
                "checkpoint writes excluded."
            ),
            (
                "Total includes simulation/rendering/collection + learning. Learner is host-timer "
                "attribution; phase medians are not additive."
            ),
            (
                "Sequential runs, one GPU/seed, independent trajectories. p10–p90 is within-run "
                "spread, not a confidence interval. Task success remains near zero."
            ),
            (
                "Source: docs/sram/data/perception/*4096_200.json · Raw timing traces are thin; "
                "full centered 9-row medians are bold."
            ),
        ],
    )
    save(fig, out, name, formats)


def bounded_16(evidence, out, formats):
    name = "bounded_16"
    reports = {
        b: camera(evidence, f"unified_performance/camera_{b}16_30.json.gz", b, 16, 30)
        for b in ("torch", "sram_unified")
    }
    raw = evidence.load(DATA + "unified_performance/synthetic_full16_four.json.gz")
    audit = evidence.load(DATA + "unified_performance/synthetic_four_independent_audit.json.gz")
    summary = evidence.load(DATA + "unified_performance/summary.json")
    assert audit["audit_passed"] and raw["source_unchanged"] and raw["timings_valid"]
    assert raw["num_envs"] == 16 and raw["batch"] == 512 and raw["adam"] == "matched fused"
    cycles = [cycle for cycle in raw["cycles"] if cycle["valid"]]
    assert len(cycles) == 8
    fig = plt.figure(figsize=(14.5, 10.5))
    gs = fig.add_gridspec(
        2, 2, left=0.12, right=0.975, top=0.84, bottom=0.22, hspace=0.48, wspace=0.43, width_ratios=[1.05, 1.0]
    )
    ax_camera, ax_reward, ax_synthetic = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
    title(
        fig,
        "Two different experiments · 16 environments",
        f"{GPU}  |  IEEE FP32 (TF32 disabled)  |  native RGB64 model and objective",
    )
    fig.text(0.12, 0.878, "REAL CAMERA RUNS", color=INK, fontsize=12, fontweight="bold")
    fig.text(0.59, 0.878, "SYNTHETIC FROZEN ROLLOUT", color=INK, fontsize=12, fontweight="bold")
    labels, positions = [], []
    for group, field in enumerate(("iteration_s", "learning_s")):
        for i, (backend, report) in enumerate(reports.items()):
            rows = report["rows"][5:]
            y = i + group * 2.8
            vals = [r[field] * 1000 for r in rows]
            distribution(ax_camera, vals, y, COLORS[backend])
            stats = evidence.record(
                name, "real_camera_N16", backend, field, vals, indices=[r["iteration"] for r in rows]
            )
            expected = summary["camera"]["runs"][backend]["timings"][field]["median"] * 1000
            np.testing.assert_allclose(stats["median"], expected, rtol=1e-12)
            labels.append(("Total · " if group == 0 else "Learner · ") + ("native" if i == 0 else "combined"))
            positions.append(y)
    distribution_axes(ax_camera, positions, labels, 705)
    ax_camera.set_xlabel("Host latency (ms; lower is better)")
    ax_camera.set_ylim(4.7, -0.7)
    ax_camera.set_title("A  25 measured iterations per run")
    ax_camera.text(
        0.0,
        -0.30,
        "All dots = iterations; line = p10–p90; tick = median",
        transform=ax_camera.transAxes,
        color=MUTED,
        fontsize=9,
    )
    for backend, report in reports.items():
        rows = [r for r in report["rows"] if "mean_episode_reward" in r]
        ax_reward.plot(
            [r["iteration"] for r in rows],
            [r["mean_episode_reward"] for r in rows],
            color=COLORS[backend],
            marker="o",
            markersize=2.5,
            linewidth=1.5,
            label="Native PyTorch" if backend == "torch" else "Combined model-resident",
        )
        evidence.record(
            name,
            "real_camera_N16",
            backend,
            "logged_mean_episode_reward",
            [r["mean_episode_reward"] for r in rows],
            indices=[r["iteration"] for r in rows],
            unit="reward",
        )
    ax_reward.set_title("B  Short training trace: zero logged success")
    ax_reward.set_xlabel("Training iteration")
    ax_reward.set_ylabel("Logged mean episode reward")
    ax_reward.set_xlim(0, 29)
    ax_reward.set_ylim(top=0)
    ax_reward.legend(frameon=False, fontsize=9, loc="lower left")
    grid(ax_reward, latency=False)
    candidates = [
        ("native_ppo_fused_adam", "Native PPO", BLUE),
        ("separate_full", "Separate actor\n+ critic launches", ORANGE),
        ("unified", "Combined\nmodel-resident", PURPLE),
        ("resident", "Combined model\n+ retained inputs", RED),
    ]
    stats = {}
    for y, (backend, label, color) in enumerate(candidates):
        vals = []
        for cycle in cycles:
            row = next(row for row in cycle["rows"] if row["backend"] == backend)
            assert row["state_parity"]["passed"] and row["cpu_rng_equal"] and row["cuda_rng_equal"]
            vals.append(row["host_ms"])
        distribution(ax_synthetic, vals, y, color)
        stats[backend] = evidence.record(name, "synthetic_PPO_update_N16_B512", backend, "update_host_ms", vals)
        np.testing.assert_allclose(
            stats[backend]["median"], summary["synthetic"]["host_ms"][backend]["median"], rtol=1e-12
        )
    distribution_axes(ax_synthetic, range(4), [v[1] for v in candidates], 325)
    ax_synthetic.set_ylim(4.2, -0.8)
    ax_synthetic.set_title("C  Complete PPO.update, 8 screened cycles")
    ax_synthetic.axhspan(2.55, 3.55, color=RED, alpha=0.055, zorder=0)
    slower = 100 * (stats["resident"]["median"] / stats["unified"]["median"] - 1)
    ax_synthetic.text(
        146, 3.13, f"+{slower:.2f}% latency\nvs model-only combined", color=RED, fontsize=11, weight="bold", va="center"
    )
    ax_synthetic.text(
        0.0,
        -0.11,
        "Same checkpoint, rollout, permutation and RNG; fused Adam.\n"
        "40 minibatches + normalizers + metric conversion + final sync.\n"
        "No collection or GAE. Dots = all eight accepted measurements.",
        transform=ax_synthetic.transAxes,
        fontsize=9.3,
        color=MUTED,
        linespacing=1.5,
        va="top",
    )
    footer(
        fig,
        [
            (
                "Camera: rows 5–29, seed 42, separate trajectories; total includes "
                "simulation/rendering. Learner includes GAE + PPO.update."
            ),
            (
                "Camera Adam differs: native default vs combined fused. "
                "Startup/warmup/logging/checkpoint writes excluded; phase timings retain host attribution."
            ),
            (
                "Synthetic: matched fused Adam, dense storage, state/loss/exact RNG gates passed. "
                "Three warmups/backend; eight balanced cycles; no rejected cycles."
            ),
            (
                "All p10–p90 lines show descriptive sample spread, not confidence intervals. These "
                "N16 scopes do not establish N64/N4096 gains or convergence."
            ),
            "Source: docs/sram/data/unified_performance/ · Camera and synthetic measurements are not interchangeable.",
        ],
    )
    save(fig, out, name, formats)


def operations(evidence, out, formats):
    name = "portable_operations"
    report = evidence.load(DATA + "kuka_kernel_microbenchmark.json")
    memory = evidence.load(DATA + "persistent_critic/compact_rgb64_native64_summary.json")
    assert report["shapes"] == {"ppo": [16384, 23], "spatial": [16384, 32, 4, 4], "gae": [32, 4096, 1]}
    assert memory["num_envs"] == 64 and memory["compact_history"]
    fig = plt.figure(figsize=(14.5, 9.6))
    gs = fig.add_gridspec(2, 3, left=0.09, right=0.97, top=0.81, bottom=0.20, hspace=0.77, wspace=0.43)
    title(
        fig,
        "Reusable operations and history storage",
        f"{GPU}  |  FP32; TF32 disabled  |  captured CUDA-event medians, 100 repetitions",
    )
    cases = [
        ("ppo_forward_backward", "PPO loss + gradients", "16,384 samples × 23 actions"),
        ("spatial_softmax_forward_backward", "Spatial softmax + gradients", "16,384 × 32 channels × 4 × 4"),
        ("gae_and_global_normalization", "GAE + global normalization", "32 steps × 4,096 environments"),
    ]
    eager_rows = []
    for col, (case, label, shape) in enumerate(cases):
        result = report["results"][case]
        selected = result["captured_graph"]
        vals = [selected[b]["gpu_ms_median"] * 1000 for b in ("torch", "fused")]
        ax = fig.add_subplot(gs[0, col])
        ax.barh([1, 0], vals, height=0.43, color=[BLUE, GREEN], alpha=0.9)
        ax.set_yticks([1, 0], ["PyTorch", "Fused"])
        for y, backend, value in zip([1, 0], ("torch", "fused"), vals, strict=True):
            assert selected[backend]["repeats"] == 100
            ax.text(value + max(vals) * 0.025, y, f"{value:,.2f} µs", va="center", fontsize=10, weight="bold")
            evidence.record(name, "isolated_captured_CUDA_event", backend, case, [value], unit="us", summary_only=True)
        ax.set_xlim(0, max(vals) * 1.38)
        ax.set_ylim(-0.6, 1.7)
        ax.grid(axis="x", color="#E6EBF1", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_xlabel("Latency (µs; lower is better)")
        ax.set_title(f"{chr(65 + col)}  {label}", fontsize=11.5, pad=27)
        ax.text(0, 1.035, shape, transform=ax.transAxes, fontsize=9.3, color=MUTED)
        ax.text(
            0.0,
            -0.35,
            f"PyTorch / fused median: {vals[0] / vals[1]:.2f}×",
            transform=ax.transAxes,
            fontsize=10.3,
            color=INK,
        )
        eager = [result[b]["gpu_ms_median"] * 1000 for b in ("torch", "fused")]
        eager_rows.append([label, f"{eager[0]:,.2f}", f"{eager[1]:,.2f}", f"{eager[0] / eager[1]:.2f}×"])
        for backend, value in zip(("torch", "fused"), eager, strict=True):
            evidence.record(name, "isolated_eager_CUDA_event", backend, case, [value], unit="us", summary_only=True)
    ax = fig.add_subplot(gs[1, 0])
    byte_values = [memory["rollout_dense_observation_bytes"], memory["rollout_observation_bytes"]]
    mib = [value / 2**20 for value in byte_values]
    ax.barh([1, 0], mib, height=0.43, color=[BLUE, GREEN], alpha=0.9)
    ax.set_yticks([1, 0], ["Dense", "Compact"])
    for y, backend, value, byte_value in zip([1, 0], ("dense", "compact"), mib, byte_values, strict=True):
        ax.text(value + 0.6, y, f"{value:.2f} MiB", va="center", fontsize=10, weight="bold")
        evidence.record(
            name,
            "exact_global_observation_storage_N64_B2048",
            backend,
            "rollout_observation_bytes",
            [byte_value],
            unit="bytes",
        )
    ax.set_xlim(0, max(mib) * 1.38)
    ax.set_ylim(-0.6, 1.7)
    ax.grid(axis="x", color="#E6EBF1", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_title("D  Global observation storage", fontsize=11.5, pad=27)
    ax.text(0, 1.035, "64 environments × 32 steps; RGB unchanged", transform=ax.transAxes, fontsize=9.3, color=MUTED)
    ax.set_xlabel("Memory (MiB; lower is better)")
    reduction = 100 * (1 - byte_values[1] / byte_values[0])
    ax.text(0, -0.35, f"{reduction:.1f}% fewer bytes; not peak VRAM or SRAM", transform=ax.transAxes, fontsize=10)
    table_ax = fig.add_subplot(gs[1, 1:])
    table_ax.axis("off")
    table_ax.set_title("E  Eager-launch control (CUDA-event medians)", fontsize=11.5, pad=27)
    table = table_ax.table(
        cellText=eager_rows,
        colLabels=["Operation", "PyTorch (µs)", "Fused (µs)", "Ratio"],
        cellLoc="left",
        colLoc="left",
        colWidths=[0.43, 0.22, 0.21, 0.14],
        bbox=[0, 0.12, 1, 0.88],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#DCE3EC")
        cell.set_linewidth(0.6)
        cell.set_text_props(color=INK)
        if row == 0:
            cell.set_facecolor("#EDF2F7")
            cell.set_text_props(weight="bold")
    table_ax.text(
        0,
        -0.15,
        "Eager intervals include host launch gaps. Captured comparisons above use graphs for both paths.",
        transform=table_ax.transAxes,
        fontsize=9.3,
        color=MUTED,
    )
    footer(
        fig,
        [
            (
                "Isolated operations exclude CNN/optimizer/gathering/simulation. Ratios do not add or "
                "predict whole-task speedup."
            ),
            (
                "Compilation/capture excluded by warmup. Raw repetitions were not archived; no invented "
                "distributions or confidence intervals."
            ),
            (
                "Memory is exact global observation accounting, including raw RGB; not a measured speed "
                "gain or peak allocator usage."
            ),
            (
                "Sources: data/kuka_kernel_microbenchmark.json and "
                "data/persistent_critic/compact_rgb64_native64_summary.json (under docs/sram/)."
            ),
        ],
    )
    save(fig, out, name, formats)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/sram/figures/transfer")
    parser.add_argument("--formats", choices=("jpg", "png", "svg"), nargs="+", default=["jpg"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    evidence = Evidence()
    style()
    native_4096(evidence, args.out, args.formats)
    bounded_16(evidence, args.out, args.formats)
    operations(evidence, args.out, args.formats)
    write_csv(args.out / "plotted_samples.csv", evidence.samples)
    write_csv(args.out / "statistics.csv", evidence.statistics)
    for relative, expected in evidence.inputs.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected["sha256"], relative
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    provenance = {
        "script": "scripts/sram/plot_transfer_report.py",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "plotting_base_commit": commit,
        "reproduction": "uv run python scripts/sram/plot_transfer_report.py",
        "requested_formats": args.formats,
        "packages": {"matplotlib": matplotlib.__version__, "numpy": np.__version__},
        "inputs": evidence.inputs,
        "outputs": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.out.iterdir())
            if p.suffix in {".jpg", ".png", ".svg", ".csv"}
        },
        "settings": {
            "latency_units": "ms for task/update; us for operation kernels",
            "percentiles": "NumPy linear p10/p90",
            "camera_warmup_rows_excluded": [0, 1, 2, 3, 4],
            "reward_rows": "All available logged rewards, without interpolating missing rows",
            "rolling_timing_median": "Centered nine post-warmup rows, full windows only",
            "distribution": "All accepted samples plus p10/p90 and median; no confidence intervals",
            "scatter_jitter": "Deterministic seed17; display-only, independent of measurement order",
            "microbenchmark": "Only archived medians plotted; raw repetitions unavailable",
            "precision": {
                "native4096": (
                    "Historical FP32 model/TF32-allowed library operations; c117928 train driver configuration"
                ),
                "real16_and_synthetic16": "FP32 with TF32 disabled, explicit raw report field",
                "microbenchmark": "FP32 with TF32 disabled in archived source-hashed benchmark_ppo_kernels.py",
            },
        },
        "interpretation": [
            "Camera and synthetic scopes are separate; no additive phase medians or stacked phase attribution.",
            "Camera curves are independent single-seed trajectories, not paired convergence evidence.",
            (
                "Synthetic eight-cycle comparison was a prepared working tree with source hashes, "
                "not a clean-commit measurement."
            ),
            (
                "Raw synthetic publication_allowed=false is retained; the archived independent "
                "audit accepts this bounded evidence."
            ),
            (
                "Input-resident synthetic update is 12.34% slower than model-only combined; neither "
                "establishes N64/N4096 speedup."
            ),
            "No new GPU timing or training experiment was run to produce these figures.",
        ],
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.out),
                "figures": 3,
                "sample_records": len(evidence.samples),
                "source_files": len(evidence.inputs),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Plot three completed native 200-iteration Kuka reports without using CUDA.

Example (from the repository root)::

    uv run python scripts/sram/plot_perception_timings.py \
        --stock docs/sram/data/perception/torch_rgb64_4096_200.json \
        --fused docs/sram/data/perception/sram_rgb64_4096_200.json \
        --graph docs/sram/data/perception/sram_graph_rgb64_4096_200.json \
        --out docs/sram/data/perception/kuka_rgb64_200_timings.jpg

Use an ``.svg`` output path for external vector exports. Add
``--preview_png /tmp/kuka_rgb64_200_timings.png`` for a PNG preview.

Rows 0-4 are excluded. The rolling median is centered, uses exactly nine
post-warmup rows, and is omitted at the first/last four measured iterations.
Dependencies: matplotlib. All other processing uses the standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

COLORS = {"torch": "#2474B5", "sram": "#28834A", "sram_graph": "#DE7D18"}
LABELS = {"torch": "Stock PyTorch", "sram": "Fused", "sram_graph": "Fused + CUDA graph"}
WARMUP_ROWS = 5
ROLLING_WINDOW = 9


def read_report(path: Path, backend: str) -> dict:
    report = json.loads(path.read_text())
    if report.get("backend") != backend:
        raise ValueError(f"{path}: expected backend {backend!r}, got {report.get('backend')!r}")
    if report.get("iterations") != 200 or len(report.get("rows", [])) != 200:
        raise ValueError(f"{path}: expected a completed 200-iteration report")
    if report.get("integration_passed") is not True or not report.get("finished_utc"):
        raise ValueError(f"{path}: report does not record successful completion")
    if report.get("warmup_iterations") != WARMUP_ROWS:
        raise ValueError(f"{path}: expected five warmup iterations")
    if [row["iteration"] for row in report["rows"]] != list(range(200)):
        raise ValueError(f"{path}: expected consecutive, zero-based iterations 0-199")
    for row in report["rows"]:
        for field in ("iteration_s", "learning_s"):
            if not math.isfinite(row[field]) or row[field] < 0:
                raise ValueError(f"{path}: invalid {field} at iteration {row['iteration']}")
    return report


def rolling_median(xs: list[int], ys: list[float]) -> tuple[list[int], list[float]]:
    """Full, centered nine-row windows only; no warmup values enter a window."""
    half = ROLLING_WINDOW // 2
    return xs[half:-half], [statistics.median(ys[i - half : i + half + 1]) for i in range(half, len(ys) - half)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Publication figure output (.jpg or .svg)")
    parser.add_argument("--preview_png", type=Path, help="Optional PNG preview of the same figure")
    args = parser.parse_args()
    if args.out.suffix.lower() not in {".jpg", ".svg"}:
        parser.error("--out must end in .jpg or .svg")
    if args.preview_png is not None and args.preview_png.suffix.lower() != ".png":
        parser.error("--preview_png must end in .png")

    paths = {"torch": args.stock, "sram": args.fused, "sram_graph": args.graph}
    # Validate all three completed reports before importing the plotting library.
    reports = {backend: read_report(path, backend) for backend, path in paths.items()}
    stock = reports["torch"]
    for key in ("task", "camera", "num_envs", "seed", "gpu", "overrides"):
        if any(report[key] != stock[key] for report in reports.values()):
            raise ValueError(f"Reports disagree on {key!r}; this figure requires a matched configuration")
    if stock["camera"] != "rgb64" or stock["num_envs"] != 4096:
        raise ValueError("This figure is for the native 4,096-environment RGB64 workload")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#64748B",
            "text.color": "#172033",
            "axes.labelcolor": "#172033",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "svg.fonttype": "none",
            "svg.hashsalt": "kuka-rgb64-timings-200",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.73, bottom=0.32, wspace=0.23)
    fig.suptitle("Kuka-Allegro RGB64 training timings", x=0.075, y=0.97, ha="left", fontsize=17, fontweight="bold")
    fig.text(
        0.075,
        0.91,
        f"{stock['num_envs']:,} environments | 200 iterations per run | seed {stock['seed']} | {stock['gpu']}",
        ha="left",
        fontsize=10,
        color="#475569",
    )

    panels = [
        ("iteration_s", "Total iteration: synchronized wall time"),
        ("learning_s", "Learner phase: host-timer attribution"),
    ]
    medians = {backend: {} for backend in reports}
    for ax, (field, title) in zip(axes, panels, strict=True):
        for backend, report in reports.items():
            rows = report["rows"][WARMUP_ROWS:]
            xs = [row["iteration"] for row in rows]
            ys = [1000.0 * row[field] for row in rows]
            smooth_x, smooth_y = rolling_median(xs, ys)
            ax.plot(xs, ys, color=COLORS[backend], alpha=0.25, linewidth=0.7, zorder=2)
            ax.plot(smooth_x, smooth_y, color=COLORS[backend], linewidth=2.0, zorder=3)
            medians[backend][field] = statistics.median(ys)
        ax.set_title(title, loc="left", pad=10)
        ax.set_xlabel("Training iteration (zero-based)")
        ax.set_ylabel("Time per iteration (ms)")
        ax.set_xlim(5, 199)
        ax.set_ylim(bottom=0)  # A zero baseline avoids visually inflating small total-time differences.
        ax.set_xticks([5, 50, 100, 150, 199])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(axis="y", color="#E2E8F0", linewidth=0.7)
        ax.set_axisbelow(True)
        median_text = (
            "Median: "
            + " | ".join(
                f"{short} {medians[backend][field]:,.1f}"
                for backend, short in [("torch", "stock"), ("sram", "fused"), ("sram_graph", "graph")]
            )
            + " ms"
        )
        ax.text(0, -0.30, median_text, transform=ax.transAxes, ha="left", va="top", fontsize=9, color="#475569")

    handles = [Line2D([0], [0], color=COLORS[b], linewidth=2, label=LABELS[b]) for b in reports]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.066, 0.867),
        ncol=3,
        frameon=False,
        handlelength=2.7,
        columnspacing=2.5,
    )
    caption = (
        "Iterations 0-4 excluded; 195 measured rows per run. Thin lines: raw times. "
        "Solid lines: centered 9-iteration median (full windows only).\n"
        "Times exclude startup, compilation warmup, logging and checkpoint writes. "
        "Learner split can include outstanding asynchronous collection work.\n"
        "Sequential runs on one GPU, one training seed; no confidence bands. "
        "Evolving trajectories can change simulation/rendering work and total time."
    )
    fig.text(0.075, 0.12, caption, ha="left", va="top", fontsize=8.7, color="#475569", linespacing=1.6)
    provenance = {
        "inputs": {b: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for b, p in paths.items()},
        "excluded_iterations": [0, 1, 2, 3, 4],
        "measured_iteration_range": [5, 199],
        "rolling_median": {"window": 9, "centered": True, "full_windows_only": True},
        "units": "milliseconds",
        "medians_ms": medians,
        "caption": caption,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix.lower() == ".svg":
        fig.savefig(
            args.out,
            format="svg",
            metadata={
                "Title": "Kuka-Allegro RGB64 training timings",
                "Description": json.dumps(provenance, sort_keys=True),
                "Date": None,
            },
        )
        args.out.write_text("\n".join(line.rstrip() for line in args.out.read_text().splitlines()) + "\n")
    else:
        fig.savefig(args.out, format="jpg", dpi=180, pil_kwargs={"quality": 92, "optimize": True})
    if args.preview_png is not None:
        args.preview_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.preview_png, format="png", dpi=150)
    plt.close(fig)
    print(json.dumps({"output": str(args.out.resolve()), "medians_ms": medians}, indent=2))


if __name__ == "__main__":
    main()

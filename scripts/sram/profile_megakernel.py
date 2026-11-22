# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Where does one megakernel iteration spend its time? Per-phase breakdown from in-kernel clock64 marks.

The profile build (``SramPPOCartpole(..., profile=True)``) records ``clock64()`` on SM 0, thread 0 at phase
boundaries of every iteration. The marks are converted to microseconds with the clock rate implied by the CUDA-event
time of the same launch. The profile build adds a few register writes per phase; compare its event time with a
normal build's (``--compare``) to see the overhead.

Usage::

    python scripts/sram/profile_megakernel.py [--num_envs 4096] [--iterations 100] [--out profile.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "source", "isaaclab_sram"))

import torch  # noqa: E402
from isaaclab_sram.megakernel import PROF_STRIDE, SramPPOCartpole  # noqa: E402

WARMUP = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_envs", type=int, default=4096)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    n_it = a.iterations

    ref = SramPPOCartpole(num_envs=a.num_envs, seed=a.seed)
    ref.train(WARMUP)
    ref_s = ref.train(n_it)["gpu_time_s"] / n_it

    tr = SramPPOCartpole(num_envs=a.num_envs, seed=a.seed, profile=True)
    tr.train(WARMUP)
    gpu_s = tr.train(n_it)["gpu_time_s"]
    total = WARMUP + n_it
    marks = tr.dbg.view(torch.int64)[: PROF_STRIDE * total].reshape(total, PROF_STRIDE).cpu().numpy()[WARMUP:]
    per_it_cycles = np.diff(marks[:, 0]).astype(np.float64)  # iteration start to next iteration start
    hz = per_it_cycles.sum() / (gpu_s * (len(per_it_cycles) / n_it))

    def us(c):
        return float(np.median(c) / hz * 1e6)

    n_mb = tr.ppo.num_learning_epochs * tr.ppo.num_mini_batches
    compute = sum(marks[:, 5 + 3 * q] - (marks[:, 4] if q == 0 else marks[:, 7 + 3 * (q - 1)]) for q in range(n_mb))
    allreduce = sum(marks[:, 6 + 3 * q] - marks[:, 5 + 3 * q] for q in range(n_mb))
    optim = sum(marks[:, 7 + 3 * q] - marks[:, 6 + 3 * q] for q in range(n_mb))
    phases = {
        "rollout: 16 control steps (policy + value MLPs, sampling, 2 physics substeps, reward, reset)": us(
            marks[:, 1] - marks[:, 0]
        ),
        "bootstrap value of the last observation": us(marks[:, 2] - marks[:, 1]),
        "GAE + global advantage normalization (1 all-reduce)": us(marks[:, 3] - marks[:, 2]),
        "mini-batch permutation": us(marks[:, 4] - marks[:, 3]),
        f"update: forward + backward on this SM's samples ({n_mb} mini-batches)": us(compute),
        f"update: deterministic gradient all-reduce across SMs ({n_mb}x, 2 grid barriers each)": us(allreduce),
        f"update: KL-adaptive LR, gradient clipping, Adam ({n_mb}x)": us(optim),
    }
    res = dict(
        num_envs=a.num_envs,
        iterations=n_it,
        describe=tr.describe(),
        event_time_per_iteration_us=1e6 * gpu_s / n_it,
        event_time_per_iteration_us_normal_build=1e6 * ref_s,
        implied_sm_clock_ghz=hz / 1e9,
        phases_us=phases,
        phases_sum_us=sum(phases.values()),
    )
    for k, v in phases.items():
        print(f"{v:8.1f} us  {k}")
    print(
        f"{res['phases_sum_us']:8.1f} us  sum of phases (profile build {res['event_time_per_iteration_us']:.1f} us, "
        f"normal build {res['event_time_per_iteration_us_normal_build']:.1f} us per iteration by CUDA events)"
    )
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()

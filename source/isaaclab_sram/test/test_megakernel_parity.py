# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Numerical parity of the SRAM-resident PPO megakernel with RSL-RL.

Runs ``scripts/sram/verify_megakernel.py`` for several launch configurations. The script trains one iteration with the
kernel's debug build and re-derives everything that iteration computed with independent references: a float64 physics
model, RSL-RL's own networks and PPO loss, torch autograd and ``torch.optim.Adam``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("cuda.bindings", reason="the megakernel runtime needs the CUDA driver and NVRTC bindings")
pytest.importorskip("rsl_rl", reason="the verification replays the iteration with RSL-RL")

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required"),
]

VERIFY_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "sram" / "verify_megakernel.py"

EXPECTED_CHECKS = {
    "init_params_roundtrip",
    "physics_vs_float64_reference",
    "transition_linkage",
    "reward_termination_timeout_bootstrap",
    "termination_and_timeout_flags",
    "episode_counters_after_rollout",
    "actor_mean_vs_rsl_rl_MLPModel",
    "critic_value_vs_rsl_rl_MLPModel",
    "log_prob",
    "gae_returns",
    "global_advantage_normalisation",
    "first_minibatch_gradient_vs_autograd",
    "kl_mean",
    "surrogate_loss",
    "value_loss",
    "first_adam_step_params_vs_torch",
    "iteration_final_params_vs_rsl_rl_replay",
    "iteration_final_learning_rate",
}
"""Checks the script must report, named without their parenthesized suffix."""

MAX_RELATIVE_L2_ERROR = 1e-5
"""Bound on the relative L2 error of the first mini-batch gradient and of the parameters after the iteration.

Both are about 1e-7 in float32; the script itself only bounds the element-wise errors.
"""

CONFIGS = [
    pytest.param(["--num_envs", "4096"], id="4096_envs"),
    # environments split unevenly over the CTAs (2 or 3 per CTA on 188 SMs)
    pytest.param(["--num_envs", "512"], id="512_envs"),
    # one environment per CTA, and fewer CTAs than SMs on GPUs with more than 96 SMs
    pytest.param(["--num_envs", "96"], id="96_envs"),
    pytest.param(["--num_envs", "4096", "--storage", "global"], id="4096_envs_global_rollout_storage"),
    # checks the 4th iteration: non-zero Adam moments and an adapted learning rate carried across launches
    pytest.param(["--num_envs", "4096", "--warmup_iters", "3"], id="4096_envs_after_3_iterations"),
]


@pytest.fixture(scope="module")
def verify_script() -> Path:
    if not VERIFY_SCRIPT.is_file():
        pytest.skip(f"test requires an Isaac Lab source checkout with {VERIFY_SCRIPT.name}")
    return VERIFY_SCRIPT


def _option(args: list[str], name: str) -> str | None:
    return args[args.index(name) + 1] if name in args else None


@pytest.mark.parametrize("args", CONFIGS)
def test_verify_megakernel_passes_all_checks(verify_script, args, tmp_path):
    report_path = tmp_path / "report.json"
    proc = subprocess.run(
        [sys.executable, str(verify_script), *args, "--out", str(report_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=900,
    )
    output = (
        f"exit code {proc.returncode}\n--- stdout ---\n{proc.stdout[-8000:]}\n--- stderr ---\n{proc.stderr[-8000:]}"
    )
    assert report_path.is_file(), f"{verify_script.name} wrote no report\n{output}"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    checks = {name.split(" (")[0]: result for name, result in report.items() if isinstance(result, dict)}
    failed = {name: result for name, result in checks.items() if not result["ok"]}
    assert not failed, f"failed checks: {failed}\n{output}"
    assert report["failures"] == [], output
    assert proc.returncode == 0, output
    assert checks.keys() >= EXPECTED_CHECKS, f"checks not reported: {sorted(EXPECTED_CHECKS - checks.keys())}"
    assert report["reset_range_violations"] == 0
    assert report["permutations_are_bijections"]
    assert report["first_minibatch_gradient_relative_l2_err"] < MAX_RELATIVE_L2_ERROR
    assert report["iteration_final_params_relative_l2_err"] < MAX_RELATIVE_L2_ERROR

    # the script ran the requested launch configuration: one persistent CTA per SM, at most one per environment
    num_envs = int(_option(args, "--num_envs"))
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    assert report["num_envs"] == num_envs
    assert report["num_ctas"] == min(num_sms, num_envs)
    if _option(args, "--storage") is not None:
        assert report["storage"] == _option(args, "--storage")

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Protect checkpoint transaction failures without requiring a CUDA launch.

Full learner tests own numerical CUDA parity. These CPU tests cover host errors
that successful kernel updates do not exercise: rejected launches, stale native
state, double commit, failed preparation and scratch-stream misuse.
"""

import copy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from isaaclab_sram.persistent_critic import PersistentCritic
from isaaclab_sram.persistent_state import PackedAdamState, PreparedUpdate, UpdateGuard


def _prepared(monkeypatch, *, initialized=True):
    stream = SimpleNamespace(cuda_stream=7)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: stream)
    parameters = (torch.nn.Parameter(torch.tensor([1.0, -2.0])), torch.nn.Parameter(torch.tensor([0.5])))
    optimizer = torch.optim.Adam(parameters, lr=7e-5)
    if initialized:
        for parameter, step in zip(parameters, (7, 11), strict=True):
            optimizer.state[parameter] = {
                "step": torch.tensor(float(step)),
                "exp_avg": torch.zeros_like(parameter),
                "exp_avg_sq": torch.zeros_like(parameter),
            }
    with torch.no_grad():
        state = PackedAdamState.prepare(
            parameters,
            optimizer,
            [optimizer.param_groups[0]] * 2,
            3,
            torch.device("cpu"),
            (7e-5, 0.9, 0.999, 1e-8, False),
        )
    guard = UpdateGuard(torch.device("cpu"))
    guard.begin()
    plan = PreparedUpdate(state, (), torch.zeros(()), 32, torch.arange(32), 4, 1, 8, guard)
    return plan, stream


def test_commit_preserves_native_adam_tensor_steps_and_rejects_reuse(monkeypatch):
    plan, _ = _prepared(monkeypatch)
    # Independent PyTorch steps supply what a successful device update exports.
    # Unequal entry steps detect a common-step or hardcoded forty-step copyback.
    reference_parameters = [torch.nn.Parameter(p.detach().clone()) for p in plan.state.parameters]
    reference = torch.optim.Adam(reference_parameters, lr=7e-5)
    reference.load_state_dict(copy.deepcopy(plan.state.optimizer.state_dict()))
    for _ in range(3):
        for parameter in reference_parameters:
            parameter.grad = torch.full_like(parameter, 0.25)
        reference.step()
    with torch.no_grad():
        plan.state.flat.copy_(torch.cat([p.flatten() for p in reference_parameters]))
        plan.state.moments.copy_(torch.cat([reference.state[p]["exp_avg"].flatten() for p in reference_parameters]))
        plan.state.variances.copy_(
            torch.cat([reference.state[p]["exp_avg_sq"].flatten() for p in reference_parameters])
        )
    plan.mark_launched()
    plan.commit()
    for actual, expected in zip(plan.state.parameters, reference_parameters, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                plan.state.optimizer.state[actual][key], reference.state[expected][key], rtol=0, atol=0
            )
        assert actual.grad is None
    with pytest.raises(RuntimeError, match="consumed"):
        plan.commit()


@pytest.mark.parametrize("initialized", [False, True])
def test_rejected_launch_does_not_mutate_native_checkpoint(monkeypatch, initialized):
    plan, _ = _prepared(monkeypatch, initialized=initialized)
    if not initialized:
        assert len(plan.state.optimizer.state) == 0
    before = [
        (p.detach().clone(), {k: v.clone() for k, v in plan.state.optimizer.state.get(p, {}).items()})
        for p in plan.state.parameters
    ]

    class RejectedKernel:
        def launch(self, *args, **kwargs):
            raise RuntimeError("cooperative launch rejected")

    with pytest.raises(RuntimeError, match="cooperative launch rejected"):
        plan.launch(RejectedKernel(), 188, 256, 66612)
    for parameter, (old, state) in zip(plan.state.parameters, before, strict=True):
        torch.testing.assert_close(parameter, old, rtol=0, atol=0)
        for key, expected in state.items():
            torch.testing.assert_close(plan.state.optimizer.state[parameter][key], expected, rtol=0, atol=0)
    if not initialized:
        assert len(plan.state.optimizer.state) == 0
    with pytest.raises(RuntimeError, match="consumed"):
        plan.commit()


@pytest.mark.parametrize("replacement", [False, True])
def test_prepared_update_rejects_changed_checkpoint(monkeypatch, replacement):
    plan, _ = _prepared(monkeypatch)
    parameter = plan.state.parameters[0]
    if replacement:
        plan.state.optimizer.state[parameter] = {k: v.clone() for k, v in plan.state.optimizer.state[parameter].items()}
    else:
        plan.state.optimizer.state[parameter]["step"].add_(1)
    try:
        with pytest.raises(RuntimeError, match="after preparation"):
            plan.mark_launched()
    finally:
        plan.cancel()


def test_stream_switch_rejected_and_failed_prepare_releases_reservation(monkeypatch):
    plan, stream = _prepared(monkeypatch)
    stream.cuda_stream = 8
    try:
        with pytest.raises(RuntimeError, match="same CUDA stream"):
            plan.mark_launched()
    finally:
        plan.cancel()
    with pytest.raises(RuntimeError, match="first CUDA stream"):
        plan.guard.begin()
    stream.cuda_stream = 7
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    # Use the actual production entry point; invalid shape fails before any CUDA
    # runtime or optimizer work. Repeating it must not become a reservation error.
    critic = object.__new__(PersistentCritic)
    critic.device = torch.device("cpu")
    critic.kernel = object()
    critic._update_guard = plan.guard
    for _ in range(2):
        with pytest.raises(ValueError, match="two-dimensional"):
            critic.update(torch.zeros(3), torch.zeros(3, 1), torch.zeros(3, 1), torch.arange(3))

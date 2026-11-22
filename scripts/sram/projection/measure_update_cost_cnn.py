# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cost of the RSL-RL PPO learner of the DexSuite Kuka-Allegro lift recipes, stock vs. one CUDA graph.

Recipes (agent configs resolved exactly as the stock RSL-RL train entrypoint resolves them):

``dex_rgb64``   ``Isaac-Lift-KukaAllegro-Camera`` with ``presets=single_camera,rgb64``: spatial-softmax CNN actor on the
                raw uint8 3x64x64 base-camera image + the 1-D ``policy``/``proprio`` state (running normalization),
                and a privileged MLP critic on ``policy``/``proprio``/``perception`` (object point cloud). 5 epochs x 8
                minibatches, fixed LR 7e-5.
``dex_state``   ``Isaac-Lift-KukaAllegro``: MLP actor and critic on ``policy``/``proprio``/``perception``. 5 x 4,
                adaptive-KL LR.
``dex_depth64`` (optional) the camera recipe with ``presets=single_camera,depth64`` (float32 1x64x64 depth image).

No simulator is launched. ``PPO.construct_algorithm`` (the call ``OnPolicyRunner.__init__`` makes) builds the real
actor, critic, Adam and ``RolloutStorage`` from synthetic observations of the exact group shapes and dtypes the
environments emit (group dims observed by the labperf benchmark of the same Isaac Lab commit, see ``TASKS``). The
storage is then filled with ``num_envs`` x ``num_steps_per_env`` synthetic transitions by ``PPO.act`` and
``PPO.process_env_step`` (random-normal 1-D observations, uniform uint8 images, random rewards / dones / time-outs)
and GAE is computed by ``PPO.compute_returns`` as in training.

Adapted from ``measure_update_cost.py`` (MLP recipes without observation normalization). Additions: several
observation groups incl. uint8 / float images, CNN actors (``model.get_latent``), running observation normalization
(``obs_normalization=True``: the normalizer statistics update that ``PPO.update`` runs over the whole rollout after the
minibatch loop is reproduced graph-safely), the fixed-LR schedule, ``use_mixed_precision`` (bf16 autocast of the
forward + loss, the recipe knob ``agent.algorithm.use_mixed_precision``), kernel-time categories (conv / GEMM / ...),
and activation-memory measurements.

Precision modes (``--modes``):

``tf32``  stock backend flags of the RSL-RL train entrypoint (TF32 matmul + cuDNN, cudnn.benchmark), fp32 params.
``bf16``  ``tf32`` + ``use_mixed_precision=True`` (bf16 autocast of the update forward + loss; ``PPO.act`` and
          ``compute_returns`` stay fp32/TF32 as in rsl_rl). An extra *what-if* act graph runs under bf16 autocast too.
``fp32``  TF32 off (the setting in which the equivalence check is a tight test).

Variants timed per mode (stock and graphed interleaved, ``--repeats`` runs each, state restored before every run):

``stock``    ``PPO.compute_returns`` (under ``inference_mode``) + ``PPO.update()``.
``graphed``  the same math captured in two CUDA graphs: GAE, and the whole ``num_learning_epochs`` x
             ``num_mini_batches`` update (minibatch permutation, gathers, actor CNN/MLP + critic forward, clipped
             surrogate + clipped value loss + entropy bonus, backward, per-model ``clip_grad_norm_``, fused capturable
             Adam with a device-tensor LR, KL-adaptive LR on the device when the schedule is adaptive, and the
             running-normalizer update). Not reproduced: the action sample that ``PPO.update`` draws in every actor
             forward and discards (``Normal.sample`` is not capturable), and the per-minibatch ``.item()`` host syncs.
``act``      rollout-side policy inference per env step for all envs: stock ``PPO.act`` vs the same ops from a graph.

Per variant: synchronized host wall time, CUDA-event GPU span, ``torch.profiler`` kernel time / count / categories,
host kernel / graph launches, host syncs. Analytic FLOPs: 2 x MACs of every ``Conv2d`` / ``Linear`` (and the
spatial-softmax expectation matmul) of actor and critic, forward + backward (dW always; dX only where the layer input
requires grad: not for the first conv on the image or the critic's first layer), for all samples of all epochs.
Activation memory: bytes autograd saves for backward in one minibatch forward (walk of the autograd graph,
parameters and buffers excluded, deduplicated by storage), the gathered minibatch inputs, and the measured
caching-allocator peak above the pre-update baseline of every stock minibatch step and of the end-of-update
normalizer pass.

Equivalence (as in ``measure_update_cost.py``): graph replay vs eager run of the same code (whole update, incl.
normalizer statistics), and every one of the epochs x minibatches steps of a stock ``PPO.update`` reproduced by the
graphed code teacher-forced from the stock state before that step. With convolutions, cuDNN's non-deterministic
algorithms (stock ``cudnn.deterministic=False``, e.g. atomic wgrad) make even two eager runs of a whole update differ
(~1e-9 per step, amplified over the sequential steps): graph vs eager is then judged against that eager-vs-eager floor.

``--labperf_dir`` adds the context of the audited labperf benchmark (read-only): the environment-only throughput at
``num_envs`` and the per-iteration ``Collection time`` / ``Learning time`` split of the production training logs, and
the learner share of an iteration with the stock and the graphed learner.

Timings depend on host load and on other GPU work: quote numbers only from a quiet host and GPU.

Usage::

    OMP_NUM_THREADS=1 python scripts/sram/projection/measure_update_cost_cnn.py \
        --tasks dex_rgb64 dex_state --modes tf32 bf16 --repeats 20 \
        --labperf_dir /path/to/labperf --out /tmp/update_cost_cnn.json
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import glob
import io
import json
import math
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from itertools import chain

_CAMERA_OVERRIDES = ["physics=newton_mjwarp", "renderer=newton_renderer"]
# Observation groups, shapes (batch omitted) and dtypes of the environments. Dims as observed by the labperf benchmark
# on the same Isaac Lab commit (labperf reports/benchmark_report.md "Workload definitions": kuka_rgb_64 "3x64x64; 960;
# 170; 465", kuka_state "960; 170; 465"; uint8 64x64 camera observation in
# reports/rgb_render_sensor_closed_review_20261110.md) and consistent with the observation config
# (core/lift/lift_env_cfg.py ObservationsCfg: policy 34 x 5 history, proprio 93 x 5, perception 64 points x 3 x 5;
# kuka_allegro/camera_cfg.py SingleCameraObservationsCfg: base_image, normalize=False, channel_first=True).
_STATE_GROUPS = {"policy": ((170,), "float32"), "proprio": ((465,), "float32"), "perception": ((960,), "float32")}
TASKS = {
    "dex_rgb64": {
        "task": "Isaac-Lift-KukaAllegro-Camera",
        "overrides": [*_CAMERA_OVERRIDES, "presets=single_camera,rgb64"],
        "act_dim": 23,
        "obs": {**_STATE_GROUPS, "base_image": ((3, 64, 64), "uint8")},
        "expected_mlp_inputs": {"actor": 699, "critic": 1595},  # labperf native_manipulation_comparison.md
        "labperf": {"throughput": "kuka_rgb_64", "train": "kuka_rgb_native"},
    },
    "dex_state": {
        "task": "Isaac-Lift-KukaAllegro",
        "overrides": ["physics=newton_mjwarp"],
        "act_dim": 23,
        "obs": dict(_STATE_GROUPS),
        "expected_mlp_inputs": {"actor": 1595, "critic": 1595},
        "labperf": {"throughput": "kuka_state_native", "train": "kuka_state_native"},
    },
    "dex_depth64": {
        "task": "Isaac-Lift-KukaAllegro-Camera",
        "overrides": [*_CAMERA_OVERRIDES, "presets=single_camera,depth64"],
        "act_dim": 23,
        "obs": {**_STATE_GROUPS, "base_image": ((1, 64, 64), "depth")},  # float32 depth [m]
        "expected_mlp_inputs": {"actor": 699, "critic": 1595},
        "labperf": {"throughput": "kuka_depth_64", "train": None},
    },
}
MODES = ("tf32", "bf16", "fp32")
DONE_PROB = 0.005  # per env and step
TIMEOUT_FRACTION = 0.5  # share of the dones that are time-outs (bootstrapped by process_env_step)
CHECK_RTOL, CHECK_ATOL = 1e-4, 1e-5  # stock vs graphed parameters (see check_equivalence)
NONDET_CONTROL_RUNS, NONDET_FACTOR = 2, 4.0  # eager-vs-eager controls of a whole update (see check_equivalence)
# Adam variants for the graphed update; both capturable with a device-tensor lr.
GRAPHED_ADAM = {"fused": {"fused": True, "capturable": True}, "foreach": {"foreach": True, "capturable": True}}

_KERNEL_LAUNCH_APIS = {
    "cudaLaunchKernel",
    "cudaLaunchKernelExC",
    "cuLaunchKernel",
    "cuLaunchKernelEx",
    "cudaLaunchCooperativeKernel",
    "cuLaunchCooperativeKernel",
}
_GRAPH_LAUNCH_APIS = {"cudaGraphLaunch", "cuGraphLaunch"}
_SYNC_APIS = {"cudaStreamSynchronize", "cuStreamSynchronize", "cudaEventSynchronize", "cuEventSynchronize"}
_GEMM_TOKENS = ("gemm", "gemv", "xmma", "cutlass", "splitk", "magma", "nvjet")
# first match wins: cuDNN conv kernels often also carry GEMM tokens (implicit GEMM)
_KERNEL_CATEGORIES = (
    ("conv", ("conv", "fprop", "dgrad", "wgrad", "cudnn", "winograd")),
    ("gemm", _GEMM_TOKENS),
    ("optimizer_foreach", ("adam", "multi_tensor")),
    ("gather_index", ("index", "gather", "scatter")),
    ("reduce", ("reduce", "softmax", "welford", "norm")),
    ("elementwise", ("elementwise", "elu", "copy", "fill", "cat", "where", "clamp")),
)
_REGION = "measure_update_cost_cnn::region"

torch = None  # imported in main() so that --help stays fast
nn = None


def _import_torch():
    global torch, nn
    import torch as _torch
    import torch.nn as _nn

    torch, nn = _torch, _nn


# --------------------------------------------------------------------------------------------------------------------
# Helpers: statistics, timing, profiling
# --------------------------------------------------------------------------------------------------------------------


def _stats(xs: list[float]) -> dict:
    xs = sorted(xs)
    n = len(xs)
    return {
        "median": statistics.median(xs),
        "min": xs[0],
        "mean": statistics.fmean(xs),
        "p90": xs[min(n - 1, round(0.9 * (n - 1)))],
        "max": xs[-1],
        "n": n,
    }


def _timed(fn, n_marks: int = 0) -> tuple[float, list[float]]:
    """Run ``fn(marks)`` once between two device syncs.

    Returns the host wall time [ms] and the CUDA-event GPU span [ms] of each segment delimited by the ``n_marks``
    events that ``fn`` records itself.
    """
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(n_marks + 2)]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ev[0].record()
    fn(ev[1:-1])
    ev[-1].record()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1e3
    return wall, [ev[i].elapsed_time(ev[i + 1]) for i in range(len(ev) - 1)]


def _kernel_category(name: str) -> str:
    n = name.lower()
    for cat, tokens in _KERNEL_CATEGORIES:
        if any(t in n for t in tokens):
            return cat
    return "other"


def _profile(fn, runs: int, setup=None) -> dict:
    """Profile ``fn()`` (one call per profiler session) and summarize GPU activity and host CUDA API calls.

    ``setup()`` runs before every session, outside the profiled region. Times are medians over the sessions.
    """
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile, record_function

    per_run = []
    for _ in range(runs):
        if setup is not None:
            setup()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with record_function(_REGION):
                fn()
                torch.cuda.synchronize()
        events = prof.events()
        region = next(e for e in events if e.name == _REGION and e.device_type == DeviceType.CPU)
        r0, r1 = region.time_range.start, region.time_range.end
        api, sync_src, memcpy_src = Counter(), Counter(), Counter()
        kernels, kernel_counts, gemm_names = Counter(), Counter(), Counter()
        cat_us, cat_n = Counter(), Counter()
        k_us = gemm_us = cp_us = ms_us = 0.0
        n_k = n_gemm = n_cp = n_ms = 0
        g0, g1 = math.inf, -math.inf
        for e in events:
            if e.device_type == DeviceType.CUDA:
                if e.is_user_annotation:  # record_function ranges (region, Optimizer.step#...), not kernels
                    continue
                d = e.time_range.elapsed_us()
                g0, g1 = min(g0, e.time_range.start), max(g1, e.time_range.end)
                if e.name.startswith("Memcpy"):
                    n_cp += 1
                    cp_us += d
                elif e.name.startswith("Memset"):
                    n_ms += 1
                    ms_us += d
                else:
                    n_k += 1
                    k_us += d
                    kernels[e.name] += d
                    kernel_counts[e.name] += 1
                    cat = _kernel_category(e.name)
                    cat_us[cat] += d
                    cat_n[cat] += 1
                    if cat == "gemm":
                        n_gemm += 1
                        gemm_us += d
                        gemm_names[e.name] += 1
            elif e.name.startswith("cu") and r0 <= e.time_range.start <= r1:
                api[e.name] += 1
                if e.name in _SYNC_APIS or e.name.startswith(("cudaMemcpy", "cuMemcpy")):
                    chain_, parent = [], e.cpu_parent
                    while parent is not None and parent.name != _REGION:
                        chain_.append(parent.name)
                        parent = parent.cpu_parent
                    src = " > ".join(reversed(chain_[-4:])) or "<harness>"
                    (sync_src if e.name in _SYNC_APIS else memcpy_src)[src] += 1
        per_run.append(
            {
                "kernel_ms": k_us / 1e3,
                "gemm_kernel_ms": gemm_us / 1e3,
                "conv_kernel_ms": cat_us["conv"] / 1e3,
                "memcpy_ms": cp_us / 1e3,
                "memset_ms": ms_us / 1e3,
                "gpu_active_span_ms": (g1 - g0) / 1e3 if n_k + n_cp + n_ms else 0.0,
                "gpu_kernels": n_k,
                "gemm_kernels": n_gemm,
                "conv_kernels": cat_n["conv"],
                "gpu_memcpy": n_cp,
                "gpu_memset": n_ms,
                "host_kernel_launches": sum(v for k, v in api.items() if k in _KERNEL_LAUNCH_APIS),
                "host_graph_launches": sum(v for k, v in api.items() if k in _GRAPH_LAUNCH_APIS),
                "host_memcpy_calls": sum(v for k, v in api.items() if k.startswith(("cudaMemcpy", "cuMemcpy"))),
                "host_memset_calls": sum(v for k, v in api.items() if k.startswith(("cudaMemset", "cuMemset"))),
                "host_syncs": sum(v for k, v in api.items() if k in _SYNC_APIS),
                "host_api_calls": dict(api),
                "kernel_ms_by_category": {k: v / 1e3 for k, v in cat_us.most_common()},
                "kernels_by_category": dict(cat_n.most_common()),
                "sync_sources": dict(sync_src.most_common(8)),
                "memcpy_sources": dict(memcpy_src.most_common(8)),
                "top_kernels_ms": {k[:140]: v / 1e3 for k, v in kernels.most_common(12)},
                "top_kernels_by_count": {k[:140]: v for k, v in kernel_counts.most_common(10)},
                "gemm_kernel_names": {k[:160]: v for k, v in gemm_names.most_common(4)},
            }
        )
    out = dict(per_run[-1])
    for key in ("kernel_ms", "gemm_kernel_ms", "conv_kernel_ms", "memcpy_ms", "memset_ms", "gpu_active_span_ms"):
        out[key] = statistics.median(r[key] for r in per_run)
    out["kernel_ms_runs"] = [r["kernel_ms"] for r in per_run]
    counts = {(r["gpu_kernels"], r["host_kernel_launches"], r["host_graph_launches"], r["host_syncs"]) for r in per_run}
    out["counts_stable_across_runs"] = len(counts) == 1
    out["profiler_runs"] = runs
    return out


def _sum_profiles(*profs: dict) -> dict:
    keys = [k for k, v in profs[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    out = {k: sum(p[k] for p in profs) for k in keys if k != "profiler_runs"}
    cats = Counter()
    for p in profs:
        cats.update(p.get("kernel_ms_by_category", {}))
    out["kernel_ms_by_category"] = dict(cats.most_common())
    out["gemm_kernel_names"] = profs[-1].get("gemm_kernel_names", {})
    return out


def _nvidia_smi() -> dict:
    def q(*a):
        return subprocess.run(["nvidia-smi", *a], capture_output=True, text=True, timeout=30).stdout.strip()

    try:
        return {
            "gpu": q(
                "--query-gpu=name,utilization.gpu,memory.used,clocks.sm,clocks.max.sm,power.draw",
                "--format=csv,noheader",
            ),
            "compute_apps": q(
                "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"
            ).splitlines(),
        }
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


@contextlib.contextmanager
def _backend_flags(tf32: bool):
    """Backend flags of the stock RSL-RL train entrypoint (``train_rsl_rl.run``); ``tf32=False`` flips only TF32."""
    flags = dict(cuda_matmul_allow_tf32=tf32, cudnn_allow_tf32=tf32, cudnn_deterministic=False, cudnn_benchmark=True)
    try:
        from isaaclab_rl.entrypoints.common import scoped_torch_backend_flags
    except Exception:  # noqa: BLE001  - same flags, set by hand
        scoped_torch_backend_flags = None
    if scoped_torch_backend_flags is not None:
        with scoped_torch_backend_flags(**flags):
            yield
        return
    saved = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = False, True
    try:
        yield
    finally:
        (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark,
        ) = saved


# --------------------------------------------------------------------------------------------------------------------
# Stock RSL-RL PPO, built from the task's agent config
# --------------------------------------------------------------------------------------------------------------------


def resolve_agent_cfg(task_id: str, overrides: list[str]) -> dict:
    """Agent cfg as the RSL-RL train entrypoint resolves it, as the dict OnPolicyRunner receives."""
    from importlib.metadata import version

    from isaaclab.utils import to_dict

    from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg

    import isaaclab_tasks  # noqa: F401  (registers the tasks)
    from isaaclab_tasks.utils import resolve_task_config

    _, agent_cfg = resolve_task_config(task_id, "rsl_rl_cfg_entry_point", overrides=list(overrides))
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))
    cfg = to_dict(agent_cfg)
    cfg["multi_gpu"] = None  # what OnPolicyRunner._configure_multi_gpu() sets for a single-GPU run
    return cfg


class _SyntheticVecEnv:
    """The two VecEnv attributes ``PPO.construct_algorithm`` reads."""

    def __init__(self, num_envs: int, num_actions: int):
        self.num_envs = num_envs
        self.num_actions = num_actions


def synthetic_obs(spec: dict, num_envs: int, device: str):
    """One env-step observation TensorDict with the task's groups, shapes and dtypes."""
    from tensordict import TensorDict

    data = {}
    for group, (shape, kind) in spec["obs"].items():
        if kind == "uint8":
            data[group] = torch.randint(0, 256, (num_envs, *shape), dtype=torch.uint8, device=device)
        elif kind == "depth":
            data[group] = 0.01 + 2.49 * torch.rand(num_envs, *shape, device=device)
        else:
            data[group] = torch.randn(num_envs, *shape, device=device)
    return TensorDict(data, batch_size=[num_envs], device=device)


def _rand_like(t):
    return torch.randint_like(t, 0, 256) if t.dtype == torch.uint8 else torch.randn_like(t)


def build_stock_ppo(cfg: dict, spec: dict, num_envs: int, device: str):
    from rsl_rl.utils import resolve_callable

    alg_class = resolve_callable(cfg["algorithm"]["class_name"])
    with contextlib.redirect_stdout(io.StringIO()):  # construct_algorithm prints both models
        alg = alg_class.construct_algorithm(
            synthetic_obs(spec, num_envs, device),
            _SyntheticVecEnv(num_envs, spec["act_dim"]),
            copy.deepcopy(cfg),
            device,
        )
    alg.train_mode()
    return alg


def stock_params(alg) -> list:
    return list(chain(alg.actor.parameters(), alg.critic.parameters()))  # = optimizer param order


def synthetic_rollout(alg, spec: dict, num_envs: int, device: str, act_t=None, store_t=None):
    """Fill ``alg.storage`` like ``OnPolicyRunner.learn()`` does, with a synthetic env; return the last observation.

    ``act_t`` / ``store_t`` collect per-step (wall ms, GPU span ms) of ``PPO.act`` / ``PPO.process_env_step``.
    """
    alg.storage.clear()
    obs = synthetic_obs(spec, num_envs, device)
    for _ in range(alg.storage.num_transitions_per_env):
        # env.step() outputs, generated before the timed regions
        next_obs = synthetic_obs(spec, num_envs, device)
        rewards = torch.randn(num_envs, device=device)
        terminal = torch.rand(num_envs, device=device) < DONE_PROB
        time_outs = terminal & (torch.rand(num_envs, device=device) < TIMEOUT_FRACTION)
        dones = terminal.to(torch.long)  # RslRlVecEnvWrapper.step() returns long dones and bool time_outs
        with torch.inference_mode():
            wall, span = _timed(lambda _m: alg.act(obs))
            if act_t is not None:
                act_t.append((wall, span[0]))
            wall, span = _timed(lambda _m: alg.process_env_step(next_obs, rewards, dones, {"time_outs": time_outs}))
            if store_t is not None:
                store_t.append((wall, span[0]))
        obs = next_obs
    return obs


def stock_learning_phase(alg, last_obs, mark=None) -> dict:
    """What OnPolicyRunner.learn() runs after the rollout: compute_returns (inference mode), then update()."""
    with torch.inference_mode():
        alg.compute_returns(last_obs)
    if mark is not None:
        mark.record()
    return alg.update()


def _model_buffers(model) -> dict:
    return {n: b.detach().clone() for n, b in model.named_buffers()}


def snapshot(alg) -> dict:
    """Parameters, Adam state, LR and model buffers (running observation-normalizer statistics)."""
    params = stock_params(alg)
    with torch.no_grad():
        return {
            "params": [p.detach().clone() for p in params],
            "opt": [
                {k: v.detach().clone() for k, v in alg.optimizer.state[p].items() if torch.is_tensor(v)} for p in params
            ],
            "lr": float(alg.learning_rate),
            "buffers": {"actor": _model_buffers(alg.actor), "critic": _model_buffers(alg.critic)},
        }


def _restore(snap: dict, params: list, optimizer) -> None:
    with torch.no_grad():
        for p, p0 in zip(params, snap["params"], strict=True):
            p.copy_(p0)
        for p, s0 in zip(params, snap["opt"], strict=True):
            state = optimizer.state[p]
            for k, v0 in s0.items():
                state[k].copy_(v0)  # also CPU <-> CUDA for Adam's `step`


def _restore_buffers(model, bufs: dict) -> None:
    """In place: the graphed copies read their normalizer buffers at fixed addresses."""
    with torch.no_grad():
        for name, b in model.named_buffers():
            b.copy_(bufs[name])


def restore_stock(alg, snap: dict) -> None:
    _restore(snap, stock_params(alg), alg.optimizer)
    _restore_buffers(alg.actor, snap["buffers"]["actor"])
    _restore_buffers(alg.critic, snap["buffers"]["critic"])
    alg.learning_rate = snap["lr"]
    for group in alg.optimizer.param_groups:
        group["lr"] = snap["lr"]


# --------------------------------------------------------------------------------------------------------------------
# Graphed: same math as PPO.compute_returns + PPO.update, captured in CUDA graphs
# --------------------------------------------------------------------------------------------------------------------


def _adam_hyperparams(alg) -> dict:
    return {k: v for k, v in alg.optimizer.defaults.items() if k in ("betas", "eps", "weight_decay", "amsgrad")}


def _clone_model(model):
    """deepcopy without the transient output distribution of the last forward pass (holds non-leaf tensors)."""
    memo = {}
    dist = getattr(model, "distribution", None)
    if dist is not None:
        for value in vars(dist).values():
            transient = isinstance(value, torch.distributions.Distribution)
            transient |= torch.is_tensor(value) and value.grad_fn is not None
            if transient:
                memo[id(value)] = None
    return copy.deepcopy(model, memo)


def _normalizer_update_(norm, x) -> None:
    """``EmpiricalNormalization.update`` (rsl_rl 5.5.1), graph-safe: the batch count is created on the device
    (``torch.full_like``, no host->device copy) and ``_std`` is written in place (stock rebinds it)."""
    if not norm.training:
        return
    count_x = torch.full_like(norm.count, x.shape[0])
    var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
    mean_x = torch.mean(x, dim=0, keepdim=True)
    norm.count += count_x
    rate = count_x / norm.count
    delta_mean = mean_x - norm._mean
    norm._mean += rate * delta_mean
    norm._var += rate * (var_x - norm._var + delta_mean * (mean_x - norm._mean))
    norm._std.copy_(torch.sqrt(norm._var))


class GraphedPPOLearner:
    """``PPO.compute_returns`` + ``PPO.update`` for feed-forward MLP / (spatial-softmax) CNN actor-critics, replayed
    from 2 CUDA graphs.

    The actor/critic are deep copies of the stock ones and use the rsl_rl / task modules for everything:
    ``get_latent`` (normalizer, image normalizer, CNN, spatial softmax), ``mlp``, ``distribution.update / log_prob /
    entropy / params / kl_divergence``. Only the control flow differs from stock: the rollout lives in static buffers
    (same layout as ``RolloutStorage``), the minibatch permutation is drawn in the graph, there is no host sync, the
    LR is a device tensor that drives ``Adam(capturable=True)`` (KL-adapted on the device if the schedule is adaptive),
    and the running-normalizer update writes its buffers in place.
    """

    def __init__(self, alg, device: str, adam_impl: str = "fused"):
        from rsl_rl.modules import EmpiricalNormalization
        from rsl_rl.utils import resolve_optimizer

        unsupported = [n for n in ("rnd", "symmetry") if getattr(alg, n)]
        unsupported += ["recurrent"] if alg.actor.is_recurrent or alg.critic.is_recurrent else []
        unsupported += ["torch.compile"] if alg.actor is not alg._raw_actor or alg.critic is not alg._raw_critic else []
        for model in (alg.actor, alg.critic):
            norm = model.obs_normalizer
            if model.obs_normalization and not (isinstance(norm, EmpiricalNormalization) and norm.until is None):
                unsupported.append(f"normalizer {type(norm).__name__}(until={getattr(norm, 'until', None)})")
        if unsupported:
            raise NotImplementedError(f"graphed update does not implement: {unsupported}")
        opt_class = type(alg.optimizer)
        if opt_class not in (torch.optim.Adam, torch.optim.AdamW):
            raise NotImplementedError(f"graphed update needs a capturable Adam/AdamW, cfg has {opt_class.__name__}")

        st = alg.storage
        self.device = device
        self.actor, self.critic = _clone_model(alg._raw_actor), _clone_model(alg._raw_critic)
        self.params = list(chain(self.actor.parameters(), self.critic.parameters()))
        self.T, self.N = st.num_transitions_per_env, st.num_envs
        self.E, self.M = alg.num_learning_epochs, alg.num_mini_batches
        self.mb = self.T * self.N // self.M
        self.clip, self.gamma, self.lam = alg.clip_param, alg.gamma, alg.lam
        self.value_loss_coef, self.entropy_coef = alg.value_loss_coef, alg.entropy_coef
        self.max_grad_norm, self.use_clipped_value_loss = alg.max_grad_norm, alg.use_clipped_value_loss
        self.adaptive = alg.desired_kl is not None and alg.schedule == "adaptive"
        self.desired_kl = alg.desired_kl
        self.norm_adv_per_mb = alg.normalize_advantage_per_mini_batch
        self.mixed = bool(alg.use_mixed_precision)  # bf16 autocast of forward + loss, as PPO.update
        self.optimizer = resolve_optimizer(opt_class.__name__)(
            self.params,
            lr=torch.tensor(alg.learning_rate, device=device),
            **GRAPHED_ADAM[adam_impl],
            **_adam_hyperparams(alg),
        )
        self.lr = self.optimizer.param_groups[0]["lr"]  # device tensor read by the captured Adam kernels
        assert torch.is_tensor(self.lr) and self.lr.is_cuda

        # static rollout buffers, RolloutStorage layout; all stored groups are gathered, as the TensorDict indexing of
        # RolloutStorage.mini_batch_generator does
        self.groups = list(st.observations.keys())
        self.obs = {g: torch.zeros_like(st.observations[g]) for g in self.groups}
        self.last_obs = {g: torch.zeros_like(st.observations[g][0]) for g in self.groups}
        for name in ("actions", "rewards", "dones", "values", "actions_log_prob", "returns", "advantages"):
            setattr(self, name, torch.zeros_like(getattr(st, name)))
        self.dist_params = tuple(torch.zeros_like(p) for p in st.distribution_params)
        self.perm = torch.empty(self.M * self.mb, dtype=torch.long, device=device)
        self.loss_sums = torch.zeros(3, device=device)
        self.g_returns = self.g_update = None

    # -- data / state --------------------------------------------------------------------------------------------------

    def load_rollout(self, storage, last_obs) -> None:
        with torch.no_grad():
            for g in self.groups:
                self.obs[g].copy_(storage.observations[g])
                self.last_obs[g].copy_(last_obs[g])
            for name in ("actions", "rewards", "dones", "values", "actions_log_prob"):
                getattr(self, name).copy_(getattr(storage, name))
            for p, q in zip(self.dist_params, storage.distribution_params, strict=True):
                p.copy_(q)

    def set_state(self, snap: dict) -> None:
        _restore(snap, self.params, self.optimizer)
        _restore_buffers(self.actor, snap["buffers"]["actor"])
        _restore_buffers(self.critic, snap["buffers"]["critic"])
        self.lr.fill_(snap["lr"])

    def normalizer_state(self) -> dict:
        return {"actor": _model_buffers(self.actor), "critic": _model_buffers(self.critic)}

    # -- the math (eager-callable; captured below) ---------------------------------------------------------------------

    def _returns_body(self) -> None:
        """== PPO.compute_returns."""
        with torch.no_grad():
            last_values = self.critic.mlp(self.critic.get_latent(self.last_obs))
            advantage = 0
            for step in reversed(range(self.T)):
                next_values = last_values if step == self.T - 1 else self.values[step + 1]
                next_is_not_terminal = 1.0 - self.dones[step].float()
                delta = self.rewards[step] + next_is_not_terminal * self.gamma * next_values - self.values[step]
                advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
                self.returns[step] = advantage + self.values[step]
            advantages = self.returns - self.values
            if not self.norm_adv_per_mb:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            self.advantages.copy_(advantages)

    def _update_body(self) -> None:
        """== PPO.update for a feed-forward actor/critic without RND or symmetry."""
        # one permutation per update, shared by all epochs (RolloutStorage.mini_batch_generator)
        torch.randperm(self.M * self.mb, device=self.device, out=self.perm)
        self.loss_sums.zero_()
        for _ in range(self.E):
            for i in range(self.M):
                self._minibatch_step(self.perm[i * self.mb : (i + 1) * self.mb])
        with torch.no_grad():
            self.loss_sums /= self.E * self.M
            # PPO.update: running-normalizer statistics from the whole rollout, after the minibatch loop
            flat = {g: o.flatten(0, 1) for g, o in self.obs.items()}
            for model in (self.actor, self.critic):
                if model.obs_normalization:
                    _normalizer_update_(model.obs_normalizer, torch.cat([flat[g] for g in model.obs_groups], dim=-1))
        # release the last autograd graph: it would keep the warm-up's AccumulateGrad nodes (side stream) alive
        self.actor.distribution._distribution = None

    def _gather(self, idx):
        b_obs = {g: o.flatten(0, 1)[idx] for g, o in self.obs.items()}
        rest = tuple(
            t.flatten(0, 1)[idx]
            for t in (self.actions, self.values, self.returns, self.advantages, self.actions_log_prob)
        )
        old_dist = tuple(p.flatten(0, 1)[idx] for p in self.dist_params)
        return b_obs, rest, old_dist

    def _forward_loss(self, b_obs, rest, b_old_dist_params):
        """Forward + losses of one minibatch (and the on-device KL-adaptive LR), as PPO.update."""
        b_actions, b_values, b_returns, b_adv, b_old_log_prob = rest
        dist = self.actor.distribution
        if self.norm_adv_per_mb:
            with torch.no_grad():
                b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.mixed):
            dist.update(self.actor.mlp(self.actor.get_latent(b_obs)))
            actions_log_prob = dist.log_prob(b_actions)
            value_pred = self.critic.mlp(self.critic.get_latent(b_obs))
            entropy = dist.entropy

            if self.adaptive:  # KL-adaptive LR, on the device
                with torch.no_grad():
                    kl_mean = torch.mean(dist.kl_divergence(b_old_dist_params, dist.params))
                    lr = self.lr
                    lr_down = torch.clamp(lr / 1.5, min=1e-5)
                    lr_up = torch.clamp(lr * 1.5, max=1e-2)
                    up = (kl_mean < self.desired_kl / 2.0) & (kl_mean > 0.0)
                    lr.copy_(torch.where(kl_mean > self.desired_kl * 2.0, lr_down, torch.where(up, lr_up, lr)))

            ratio = torch.exp(actions_log_prob - torch.squeeze(b_old_log_prob))
            surrogate = -torch.squeeze(b_adv) * ratio
            surrogate_clipped = -torch.squeeze(b_adv) * torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            if self.use_clipped_value_loss:
                value_clipped = b_values + (value_pred - b_values).clamp(-self.clip, self.clip)
                value_losses = (value_pred - b_returns).pow(2)
                value_losses_clipped = (value_clipped - b_returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (b_returns - value_pred).pow(2).mean()
            entropy_mean = entropy.mean()
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_mean
        return loss, value_loss, surrogate_loss, entropy_mean

    def _minibatch_step(self, idx) -> None:
        """One iteration of the minibatch loop of PPO.update: gather, loss, backward, clipping, Adam."""
        b_obs, rest, b_old_dist_params = self._gather(idx)
        loss, value_loss, surrogate_loss, entropy_mean = self._forward_loss(b_obs, rest, b_old_dist_params)
        self.optimizer.zero_grad(set_to_none=False)  # grads stay static tensors
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optimizer.step()
        with torch.no_grad():
            self.loss_sums += torch.stack([x.detach().float() for x in (value_loss, surrogate_loss, entropy_mean)])

    # -- capture / replay ----------------------------------------------------------------------------------------------

    def capture(self, warmup: int = 2) -> dict:
        """Warm up eagerly on a side stream (creates grads, Adam state, cuBLAS/cuDNN workspaces, cuDNN autotune),
        then capture. Returns the caching-allocator growth of the capture (the graphs' private memory pools).

        Warm-up changes parameters, optimizer state and normalizer statistics: call ``set_state`` afterwards.
        """
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                self._returns_body()
                self._update_body()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()  # torch.cuda.graph() empties the cache on entry: same baseline
        reserved0 = torch.cuda.memory_reserved()
        self.g_returns = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g_returns):
            self._returns_body()
        self.g_update = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g_update):
            self._update_body()
        torch.cuda.synchronize()
        return {"graph_pools_reserved_bytes": torch.cuda.memory_reserved() - reserved0}

    def replay(self, mark=None) -> dict:
        self.g_returns.replay()
        if mark is not None:
            mark.record()
        self.g_update.replay()
        value, surrogate, entropy = self.loss_sums.tolist()  # the loss dict PPO.update returns (1 sync)
        return {"value": value, "surrogate": surrogate, "entropy": entropy}


class GraphedAct:
    """The ops of ``PPO.act`` (actor forward + sample + log-prob, critic forward) for all envs, as one CUDA graph.

    Sampling uses ``rsample()`` (``loc + scale * eps``, same distribution): ``Normal.sample()`` calls
    ``torch.normal(loc, scale)``, whose ``scale >= 0`` check is a host sync and cannot be captured. ``bf16=True`` runs
    the forward passes under bf16 autocast (a what-if: rsl_rl's ``PPO.act`` never autocasts).
    """

    def __init__(self, actor, critic, obs_like: dict, bf16: bool = False):
        from rsl_rl.modules.distribution import GaussianDistribution

        if not isinstance(actor.distribution, GaussianDistribution):
            raise NotImplementedError("graphed act implements Gaussian action distributions only")
        self.actor, self.critic, self.bf16 = actor, critic, bf16
        self.obs = {g: _rand_like(o) for g, o in obs_like.items()}
        self.graph = None

    def _body(self) -> None:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16):
            dist = self.actor.distribution
            dist.update(self.actor.mlp(self.actor.get_latent(self.obs)))
            self.actions = dist._distribution.rsample()
            self.values = self.critic.mlp(self.critic.get_latent(self.obs))
            self.actions_log_prob = dist.log_prob(self.actions)
            self.dist_params = dist.params

    def capture(self) -> None:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._body()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._body()
        torch.cuda.synchronize()


# --------------------------------------------------------------------------------------------------------------------
# FLOPs, activation memory, equivalence check, per-mode measurement
# --------------------------------------------------------------------------------------------------------------------


def layer_census(model, obs) -> list[dict]:
    """Conv2d / Linear layers (shapes from one forward pass via hooks) and spatial-softmax matmuls, with per-sample
    forward MACs and which backward GEMMs they need (dW: weight requires grad; dX: layer input requires grad)."""
    rows, hooks = [], []
    names = {m: n for n, m in model.named_modules()}

    def hook(mod, inp, out):
        x = inp[0]
        if isinstance(mod, nn.Conv2d):
            macs = int(out.shape[1] * out.shape[2] * out.shape[3]) * (mod.in_channels // mod.groups)
            macs *= mod.kernel_size[0] * mod.kernel_size[1]
            kind = "conv2d"
        else:
            macs, kind = mod.in_features * mod.out_features, "linear"
        dw, dx = bool(mod.weight.requires_grad), bool(x.requires_grad)
        rows.append(
            {
                "layer": names[mod],
                "kind": kind,
                "in_shape": list(x.shape[1:]),
                "out_shape": list(out.shape[1:]),
                "fwd_macs": macs,
                "dW": dw,
                "dX": dx,
                "train_macs": macs * (1 + dw + dx),
            }
        )

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(hook))
    try:
        with torch.enable_grad():
            model.mlp(model.get_latent(obs))
    finally:
        for h in hooks:
            h.remove()
    for n, m in model.named_modules():
        if type(m).__name__ == "SpatialSoftmax":  # weights [B, C, H*W] @ [H*W, 2]; pos buffers need no dW
            macs = m.num_channels * m.height * m.width * 2
            rows.append(
                {
                    "layer": f"{n}.expectation_matmul",
                    "kind": "matmul",
                    "in_shape": [m.num_channels, m.height * m.width],
                    "out_shape": [m.num_channels, 2],
                    "fwd_macs": macs,
                    "dW": False,
                    "dX": True,
                    "train_macs": 2 * macs,
                }
            )
    return rows


def flop_counts(alg, spec: dict, num_envs: int, num_steps: int, device: str) -> dict:
    """Analytic FLOPs = 2 x MACs of conv / linear / matmul layers; biases, activations, normalization, losses, Adam
    and the image normalizer are not counted."""
    obs = synthetic_obs(spec, 2, device)
    out = {}
    fwd_total = train_total = 0
    kind_fwd, kind_train = Counter(), Counter()
    for name, model in (("actor", alg.actor), ("critic", alg.critic)):
        rows = layer_census(model, obs)
        out[f"{name}_layers"] = rows
        f, t = sum(r["fwd_macs"] for r in rows), sum(r["train_macs"] for r in rows)
        out[f"{name}_fwd_macs_per_sample"], out[f"{name}_train_macs_per_sample"] = f, t
        fwd_total, train_total = fwd_total + f, train_total + t
        for r in rows:
            kind_fwd[r["kind"]] += r["fwd_macs"]
            kind_train[r["kind"]] += r["train_macs"]
    samples, epochs = num_envs * num_steps, alg.num_learning_epochs
    out["fwd_flops_per_sample"] = 2 * fwd_total
    out["train_flops_per_sample"] = 2 * train_total
    out["fwd_flops_per_sample_by_kind"] = {k: 2 * v for k, v in kind_fwd.items()}
    out["train_flops_per_sample_by_kind"] = {k: 2 * v for k, v in kind_train.items()}
    out["samples_per_update"] = samples
    out["samples_processed_per_update"] = samples * epochs
    out["update_flops"] = 2 * train_total * samples * epochs
    out["update_flops_by_kind"] = {k: 2 * v * samples * epochs for k, v in kind_train.items()}
    out["update_flops_6x_fwd_macs"] = 2 * 3 * fwd_total * samples * epochs  # convention of project_locomotion.py
    out["returns_flops"] = 2 * out["critic_fwd_macs_per_sample"] * num_envs  # last-value critic forward
    out["learning_phase_flops"] = out["update_flops"] + out["returns_flops"]
    out["act_flops_per_env_step"] = 2 * fwd_total * num_envs  # actor + critic forward, all envs
    return out


def _storage_ptrs(*models) -> set:
    return {t.untyped_storage().data_ptr() for m in models for t in chain(m.parameters(), m.buffers())}


def saved_activation_census(loss, exclude_ptrs: set) -> dict:
    """Bytes of the tensors autograd saved for the backward of ``loss`` (walk of the autograd graph; parameters and
    buffers excluded; deduplicated by storage), by autograd node type."""
    by_node, storages, seen = Counter(), {}, set()
    keep = []  # node wrappers are created on access: keep them alive so that id() stays unique
    stack = [loss.grad_fn]
    while stack:
        fn = stack.pop()
        if fn is None or id(fn) in seen:
            continue
        seen.add(id(fn))
        keep.append(fn)
        for attr in dir(fn):
            if not attr.startswith("_saved_"):
                continue
            try:
                val = getattr(fn, attr)
            except Exception:  # noqa: BLE001  (freed / non-tensor saved values)
                continue
            for t in val if isinstance(val, (tuple, list)) else (val,):
                if not (torch.is_tensor(t) and t.is_cuda):
                    continue
                st = t.untyped_storage()
                ptr = st.data_ptr()
                if ptr in exclude_ptrs or ptr in storages:
                    continue
                storages[ptr] = st.nbytes()
                by_node[type(fn).__name__] += st.nbytes()
        stack.extend(nxt for nxt, _ in fn.next_functions)
    return {"bytes": sum(storages.values()), "tensors": len(storages), "by_node_bytes": dict(by_node.most_common())}


def activation_memory(alg, learner: GraphedPPOLearner, snap: dict, last_obs) -> dict:
    """Per-minibatch memory: autograd-saved activations, gathered inputs, measured allocator peaks (stock, graphed
    eager), and the stock end-of-update normalizer pass. Leaves stock and learner state restored to ``snap``."""
    mb = learner.mb
    out = {"mini_batch_size": mb}
    # -- census on one minibatch with the graphed (== stock model) code, eager ----------------------------------------
    learner.set_state(snap)
    learner._returns_body()
    idx = learner.perm[:mb].clone()  # a minibatch of the last drawn permutation
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    b_obs, rest, old = learner._gather(idx)
    inputs = sum(t.numel() * t.element_size() for t in chain(b_obs.values(), rest, old))
    loss, *_ = learner._forward_loss(b_obs, rest, old)
    torch.cuda.synchronize()
    after_fwd = torch.cuda.memory_allocated() - base
    census = saved_activation_census(loss, _storage_ptrs(learner.actor, learner.critic))
    learner.optimizer.zero_grad(set_to_none=False)
    loss.backward()
    nn.utils.clip_grad_norm_(learner.actor.parameters(), learner.max_grad_norm)
    nn.utils.clip_grad_norm_(learner.critic.parameters(), learner.max_grad_norm)
    learner.optimizer.step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del loss, b_obs, rest, old
    learner.actor.distribution._distribution = None
    out["gathered_inputs_bytes"] = inputs
    out["gathered_inputs_bytes_by_group"] = {
        g: mb * math.prod(o.shape[2:]) * o.element_size() for g, o in learner.obs.items()
    }
    out["saved_activations"] = census
    out["graphed_eager_step"] = {"allocated_after_forward_bytes": after_fwd, "peak_bytes": peak}
    out["per_sample"] = {
        "gathered_inputs_bytes": inputs / mb,
        "saved_activation_bytes": census["bytes"] / mb,
        "graphed_eager_step_peak_bytes": peak / mb,
    }
    learner.set_state(snap)

    # -- stock PPO.update: allocator peak of every minibatch step and of the end-of-update normalizer pass -------------
    restore_stock(alg, snap)
    with torch.inference_mode():
        alg.compute_returns(last_obs)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    peaks = []

    def post_hook(opt, *_):  # allocator statistics are host-side: no sync
        peaks.append(torch.cuda.max_memory_allocated() - base)
        torch.cuda.reset_peak_memory_stats()

    hook = alg.optimizer.register_step_post_hook(post_hook)
    torch.cuda.reset_peak_memory_stats()
    try:
        alg.update()
    finally:
        hook.remove()
    torch.cuda.synchronize()
    tail = torch.cuda.max_memory_allocated() - base
    out["stock_update"] = {
        "minibatch_step_peak_bytes_max": max(peaks),
        "minibatch_step_peak_bytes_first": peaks[0],
        "minibatch_step_peak_bytes_median": statistics.median(peaks),
        "normalizer_pass_peak_bytes": tail,
        "baseline_allocated_bytes": base,
        "note": "peaks above the allocation before update(); a stock step also holds the previous minibatch "
        "(the loop variable) while the next one is gathered",
    }
    out["per_sample"]["stock_minibatch_step_peak_bytes"] = max(peaks) / mb
    restore_stock(alg, snap)
    torch.cuda.synchronize()
    return out


def _param_diff(pa: list, pb: list, p0: list) -> dict:
    diff = math.sqrt(sum(float((a - b).double().pow(2).sum()) for a, b in zip(pa, pb, strict=True)))
    update = math.sqrt(sum(float((a - c).double().pow(2).sum()) for a, c in zip(pa, p0, strict=True)))
    return {
        "max_abs_param_diff": max(float((a - b).abs().max()) for a, b in zip(pa, pb)),
        "max_abs_param_change_by_update": max(float((a - c).abs().max()) for a, c in zip(pa, p0)),
        "param_diff_norm_over_update_norm": diff / max(update, 1e-30),
        "allclose": all(torch.allclose(a, b, rtol=CHECK_RTOL, atol=CHECK_ATOL) for a, b in zip(pa, pb)),
    }


def _buffer_diff(a: dict, b: dict) -> dict:
    out = {}
    for model in ("actor", "critic"):
        for name, t in a[model].items():
            if t.is_floating_point() or t.dtype == torch.long:
                d = float((t.double() - b[model][name].double()).abs().max()) if t.numel() else 0.0
                out[f"{model}.{name}"] = d
    return out


def _params_of(params: list) -> list:
    return [p.detach().clone() for p in params]


def check_equivalence(alg, learner: GraphedPPOLearner, snap: dict, last_obs, seed: int, adam_impl: str) -> dict:
    """Stock ``PPO.compute_returns`` + ``PPO.update`` vs the graphed update, same state, data and minibatch order.

    1. ``graph_vs_eager``: graph replay vs the same code run eagerly, whole update (parameters and running-normalizer
       statistics) -> capture changes nothing.
    2. ``per_step_vs_stock`` (teacher-forced): for each of the epochs x minibatches steps of a stock ``PPO.update``,
       the state before the step (parameters, Adam moments, step count, LR) is recorded via optimizer step hooks; one
       step of the graphed code runs from that state on the same minibatch and must reproduce the stock parameters
       after the step and the stock KL-adapted LR.
    3. ``normalizer_vs_stock``: running-normalizer statistics after the whole update, stock vs graph replay.
    Pass = (1), (2) within allclose(rtol, atol) with the same LR at every step, and (3) within 1e-5 relative.
    ``free_running`` reports whole-update differences (stock vs graphed, and stock vs stock with the graph's Adam as
    the control) for information: 20-40 sequential steps amplify any rounding difference.
    """
    params = stock_params(alg)
    mb, M = learner.mb, learner.M

    # -- stock update, recording the state around every optimizer step -------------------------------------------------
    pre, post = [], []

    def pre_hook(opt, *_):
        pre.append(
            {
                "params": _params_of(params),
                "opt": [{k: v.detach().clone() for k, v in opt.state[p].items() if torch.is_tensor(v)} for p in params],
                "lr": float(opt.param_groups[0]["lr"]),  # already KL-adapted for this step
            }
        )

    def post_hook(opt, *_):
        post.append(_params_of(params))

    restore_stock(alg, snap)
    hooks = [alg.optimizer.register_step_pre_hook(pre_hook), alg.optimizer.register_step_post_hook(post_hook)]
    try:
        torch.cuda.manual_seed(seed)  # randperm in update() is the first CUDA RNG draw
        losses_stock = stock_learning_phase(alg, last_obs)
    finally:
        for h in hooks:
            h.remove()
    p_stock, lr_stock, adv_stock = _params_of(params), alg.learning_rate, alg.storage.advantages.clone()
    norm_stock = {"actor": _model_buffers(alg.actor), "critic": _model_buffers(alg.critic)}
    torch.cuda.manual_seed(seed)
    perm = torch.randperm(M * mb, device=alg.device)  # the stock permutation (same seed, same call)

    # -- (1) graph replay vs eager run of the same code ----------------------------------------------------------------
    learner.set_state(snap)
    torch.cuda.manual_seed(seed)  # randperm in the update graph is its first RNG draw -> same permutation
    losses_graphed = learner.replay()
    p_graphed, lr_graphed, norm_graphed = _params_of(learner.params), float(learner.lr), learner.normalizer_state()
    learner.set_state(snap)
    torch.cuda.manual_seed(seed)
    learner._returns_body()
    learner._update_body()
    p_eager = _params_of(learner.params)
    graph_vs_eager = _param_diff(p_graphed, p_eager, snap["params"])
    graph_vs_eager["same_lr"] = float(learner.lr) == lr_graphed
    graph_vs_eager["normalizer_max_abs_diff"] = _buffer_diff(norm_graphed, learner.normalizer_state())
    # control: the same eager code twice from the same state and permutation = run-to-run nondeterminism floor
    # (cuDNN picks non-deterministic algorithms, e.g. atomic wgrad, under the stock cudnn.deterministic=False; the
    # epochs x minibatches sequential steps amplify ~1e-9 per-step differences)
    floor = 0.0
    eager_controls = []
    for _ in range(NONDET_CONTROL_RUNS):
        learner.set_state(snap)
        torch.cuda.manual_seed(seed)
        learner._returns_body()
        learner._update_body()
        eager_controls.append(_param_diff(p_eager, _params_of(learner.params), snap["params"]))
        floor = max(floor, eager_controls[-1]["max_abs_param_diff"])
    graph_vs_eager["eager_vs_eager_controls"] = eager_controls
    graph_vs_eager["nondeterminism_floor_max_abs"] = floor
    graph_vs_eager["within_floor"] = bool(
        graph_vs_eager["allclose"] or graph_vs_eager["max_abs_param_diff"] <= NONDET_FACTOR * floor
    )

    # -- (2) teacher-forced per-step comparison with stock -------------------------------------------------------------
    learner.set_state(snap)
    learner._returns_body()  # returns / advantages with the critic the update starts from (as compute_returns)
    worst, lr_mismatch, n_ok = None, 0, 0
    for k, (before, after) in enumerate(zip(pre, post, strict=True)):
        lr_before = snap["lr"] if k == 0 else pre[k - 1]["lr"]
        learner.set_state(
            {"params": before["params"], "opt": before["opt"], "lr": lr_before, "buffers": snap["buffers"]}
        )
        i = k % M
        learner._minibatch_step(perm[i * mb : (i + 1) * mb])
        d = _param_diff(after, _params_of(learner.params), before["params"])
        lr_ok = abs(float(learner.lr) - before["lr"]) <= 1e-5 * before["lr"]
        lr_mismatch += not lr_ok
        n_ok += bool(d["allclose"] and lr_ok)
        if worst is None or d["max_abs_param_diff"] > worst["max_abs_param_diff"]:
            worst = {**d, "step": k}
    learner.actor.distribution._distribution = None
    per_step = {"steps": len(pre), "steps_passed": n_ok, "lr_mismatches": lr_mismatch, "worst_step": worst}

    # -- (3) running-normalizer statistics after the whole update, stock vs graphed -----------------------------------
    norm_diff = _buffer_diff(norm_stock, norm_graphed)
    norm_ok = all(
        torch.allclose(norm_stock[m][n].double(), norm_graphed[m][n].double(), rtol=1e-5, atol=1e-6)
        for m in ("actor", "critic")
        for n in norm_stock[m]
        if n.startswith(("_mean", "_var", "_std")) or n.endswith(("._mean", "._var", "._std", ".count"))
    )

    # -- free-running whole update (informational) ---------------------------------------------------------------------
    matched = type(alg.optimizer)(params, lr=snap["lr"], **GRAPHED_ADAM[adam_impl], **_adam_hyperparams(alg))
    for p, s0 in zip(params, snap["opt"], strict=True):
        matched.state[p] = {k: v.to(p.device, copy=True) for k, v in s0.items()}
    saved, alg.optimizer = alg.optimizer, matched
    try:
        restore_stock(alg, snap)
        torch.cuda.manual_seed(seed)
        stock_learning_phase(alg, last_obs)
        p_matched = _params_of(params)
    finally:
        alg.optimizer = saved
    restore_stock(alg, snap)
    learner.set_state(snap)
    torch.cuda.synchronize()

    passed = graph_vs_eager["within_floor"] and graph_vs_eager["same_lr"] and n_ok == len(pre) == learner.E * M
    passed = passed and norm_ok and max(graph_vs_eager["normalizer_max_abs_diff"].values(), default=0.0) == 0.0
    return {
        "criterion": f"graph replay vs eager same code (whole update): allclose(rtol={CHECK_RTOL}, atol={CHECK_ATOL}) "
        f"or max|dp| <= {NONDET_FACTOR} x the eager-vs-eager nondeterminism floor, same LR, bitwise-equal normalizer "
        f"statistics; every teacher-forced step vs stock PPO.update allclose(rtol={CHECK_RTOL}, atol={CHECK_ATOL}) "
        "with the same LR; normalizer statistics vs stock allclose(rtol=1e-5, atol=1e-6)",
        "passed": bool(passed),
        "graph_vs_eager": graph_vs_eager,
        "per_step_vs_stock": per_step,
        "normalizer_vs_stock": {"passed": bool(norm_ok), "max_abs_diff": norm_diff},
        "max_abs_advantage_diff": float((adv_stock - learner.advantages).abs().max()),
        "free_running": {
            "stock_vs_graphed": _param_diff(p_stock, p_graphed, snap["params"]),
            "control_stock_vs_stock_with_graph_adam": _param_diff(p_stock, p_matched, snap["params"]),
            "lr_stock": lr_stock,
            "lr_graphed": lr_graphed,
            "losses_stock": losses_stock,
            "losses_graphed": losses_graphed,
        },
    }


def measure_mode(alg, spec: dict, flops: dict, args, mode: str, lr0: float) -> dict:
    N, dev = args.num_envs, args.device
    T = alg.storage.num_transitions_per_env
    alg.use_mixed_precision = mode == "bf16"
    res = {
        "mode": mode,
        "torch_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "torch_cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "use_mixed_precision": alg.use_mixed_precision,
        "nvidia_smi_before": _nvidia_smi(),  # other GPU work during this measurement
        "loadavg_before": os.getloadavg(),
    }

    # -- warm-up: rollout + learning phase, as in training (also runs cuDNN autotuning) --------------------------------
    for _ in range(max(1, args.warmup)):
        last_obs = synthetic_rollout(alg, spec, N, dev)
        stock_learning_phase(alg, last_obs)

    # -- stock rollout-side inference: rollouts of the current policy, no update in between ----------------------------
    act_t, store_t = [], []
    for _ in range(max(3, math.ceil(args.repeats / 4))):
        last_obs = synthetic_rollout(alg, spec, N, dev, act_t, store_t)

    # -- the state every timed learning phase starts from: the policy that generated the stored rollout (so the first
    #    minibatch sees KL ~ 0, as in training), its Adam moments, normalizer statistics, the configured LR ------------
    alg.learning_rate = lr0
    for group in alg.optimizer.param_groups:
        group["lr"] = lr0
    snap = snapshot(alg)

    # -- graphed learner + graphed act ---------------------------------------------------------------------------------
    t0 = time.perf_counter()
    learner = GraphedPPOLearner(alg, dev, args.graphed_adam)
    res["graphed_adam"] = {"class": type(learner.optimizer).__name__, **GRAPHED_ADAM[args.graphed_adam]}
    learner.load_rollout(alg.storage, last_obs)
    res["graph_memory"] = learner.capture()
    learner.set_state(snap)
    act_graph = GraphedAct(learner.actor, learner.critic, last_obs)
    act_graph.capture()
    act_graph_bf16 = None
    if mode == "bf16":
        act_graph_bf16 = GraphedAct(learner.actor, learner.critic, last_obs, bf16=True)
        act_graph_bf16.capture()
    res["graph_capture_s"] = time.perf_counter() - t0
    for _ in range(args.warmup):
        learner.set_state(snap)
        learner.replay()

    # -- interleaved timed runs ----------------------------------------------------------------------------------------
    st_wall, st_ret, st_upd, gr_wall, gr_ret, gr_upd = [], [], [], [], [], []
    for _ in range(args.repeats):
        restore_stock(alg, snap)
        wall, (ret, upd) = _timed(lambda m: stock_learning_phase(alg, last_obs, m[0]), n_marks=1)
        st_wall.append(wall), st_ret.append(ret), st_upd.append(upd)
        learner.set_state(snap)
        wall, (ret, upd) = _timed(lambda m: learner.replay(m[0]), n_marks=1)
        gr_wall.append(wall), gr_ret.append(ret), gr_upd.append(upd)
    gact_t = [(w, s[0]) for w, s in (_timed(lambda _m: act_graph.graph.replay()) for _ in range(args.repeats * T))]
    gact16_t = []
    if act_graph_bf16 is not None:
        gact16_t = [
            (w, s[0]) for w, s in (_timed(lambda _m: act_graph_bf16.graph.replay()) for _ in range(args.repeats * T))
        ]

    # -- profiler passes (state restored outside the profiled region) --------------------------------------------------
    def stock_returns():
        with torch.inference_mode():
            alg.compute_returns(last_obs)

    def stock_act():
        with torch.inference_mode():
            alg.act(last_obs)

    def graphed_update():  # second half of GraphedPPOLearner.replay()
        learner.g_update.replay()
        learner.loss_sums.tolist()

    def graphed_setup():
        learner.set_state(snap)
        learner.g_returns.replay()

    runs = args.profile_runs
    prof = {
        "stock_returns": _profile(stock_returns, runs, setup=lambda: restore_stock(alg, snap)),
        "stock_update": _profile(alg.update, runs, setup=lambda: (restore_stock(alg, snap), stock_returns())),
        "graphed_returns": _profile(learner.g_returns.replay, runs),
        "graphed_update": _profile(graphed_update, runs, setup=graphed_setup),
        "stock_act": _profile(stock_act, runs),
        "graphed_act": _profile(act_graph.graph.replay, runs),
    }
    if act_graph_bf16 is not None:
        prof["graphed_act_bf16"] = _profile(act_graph_bf16.graph.replay, runs)
    prof["stock_learning_phase"] = _sum_profiles(prof["stock_returns"], prof["stock_update"])
    prof["graphed_learning_phase"] = _sum_profiles(prof["graphed_returns"], prof["graphed_update"])

    # -- activation memory ---------------------------------------------------------------------------------------------
    res["memory"] = activation_memory(alg, learner, snap, last_obs)

    # -- equivalence (tight with TF32 off; informational with TF32 on / bf16) ------------------------------------------
    res["equivalence"] = check_equivalence(alg, learner, snap, last_obs, args.seed + 1, args.graphed_adam)
    restore_stock(alg, snap)

    lp_flops = flops["learning_phase_flops"]
    stock_ms, graphed_ms = _stats(st_wall), _stats(gr_wall)
    sp, gp = prof["stock_learning_phase"], prof["graphed_learning_phase"]

    def _rates(p: dict, wall_ms: float) -> dict:
        conv_fl = flops["update_flops_by_kind"].get("conv2d", 0)
        lin_fl = flops["update_flops_by_kind"].get("linear", 0) + flops["returns_flops"]
        return {
            "tflops_wall": lp_flops / (wall_ms * 1e-3) / 1e12,
            "tflops_kernel": lp_flops / (p["kernel_ms"] * 1e-3) / 1e12,
            "conv_tflops_on_conv_kernels": (
                conv_fl / (p["conv_kernel_ms"] * 1e-3) / 1e12 if p["conv_kernel_ms"] else None
            ),
            "gemm_tflops_on_gemm_kernels": (
                lin_fl / (p["gemm_kernel_ms"] * 1e-3) / 1e12 if p["gemm_kernel_ms"] else None
            ),
        }

    res["stock"] = {
        "learning_phase_wall_ms": stock_ms,
        "returns_gpu_span_ms": _stats(st_ret),
        "update_gpu_span_ms": _stats(st_upd),
        "kernel_ms": sp["kernel_ms"],
        **_rates(sp, stock_ms["median"]),
    }
    res["graphed"] = {
        "learning_phase_wall_ms": graphed_ms,
        "returns_gpu_span_ms": _stats(gr_ret),
        "update_gpu_span_ms": _stats(gr_upd),
        "kernel_ms": gp["kernel_ms"],
        **_rates(gp, graphed_ms["median"]),
        "speedup_vs_stock_wall": stock_ms["median"] / graphed_ms["median"],
    }
    act_flops = flops["act_flops_per_env_step"]
    res["act"] = {
        "stock_act_wall_ms": _stats([w for w, _ in act_t]),
        "stock_act_gpu_span_ms": _stats([s for _, s in act_t]),
        "stock_process_env_step_wall_ms": _stats([w for w, _ in store_t]),
        "graphed_act_wall_ms": _stats([w for w, _ in gact_t]),
        "graphed_act_gpu_span_ms": _stats([s for _, s in gact_t]),
        "flops_per_env_step": act_flops,
    }
    if gact16_t:
        res["act"]["graphed_act_bf16_wall_ms"] = _stats([w for w, _ in gact16_t])
        res["act"]["graphed_act_bf16_gpu_span_ms"] = _stats([s for _, s in gact16_t])
    res["act"]["stock_tflops_wall"] = act_flops / (res["act"]["stock_act_wall_ms"]["median"] * 1e-3) / 1e12
    res["act"]["graphed_tflops_wall"] = act_flops / (res["act"]["graphed_act_wall_ms"]["median"] * 1e-3) / 1e12
    res["profile"] = prof
    res["nvidia_smi_after"] = _nvidia_smi()
    res["loadavg_after"] = os.getloadavg()

    learner = act_graph = act_graph_bf16 = None  # release the captured CUDA graphs
    alg.use_mixed_precision = False
    gc.collect()
    torch.cuda.empty_cache()
    return res


# --------------------------------------------------------------------------------------------------------------------
# labperf context (read-only): environment-only throughput and the production iteration split
# --------------------------------------------------------------------------------------------------------------------

_LOG_LINE = re.compile(rb"^\s*(Collection time|Learning time): ([0-9.]+)s")


def labperf_context(labperf_dir: str, spec: dict, num_envs: int, num_steps: int) -> dict:
    """Environment-only throughput (results/throughput) and the RSL-RL per-iteration Collection / Learning time of the
    production training logs (results/train/*.log) of the labperf benchmark."""
    out = {"labperf_dir": labperf_dir}
    names = spec.get("labperf", {})
    if names.get("throughput"):
        name = f"isaaclab_{names['throughput']}_{num_envs}_s42.jsonl"
        path = os.path.join(labperf_dir, "results", "throughput", name)
        if os.path.isfile(path):
            with open(path) as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            sps = [r["steps_per_second"] for r in rows if r.get("record_type") == "throughput"]
            out["env_only_throughput"] = {
                "file": path,
                "transitions_per_s_windows": sps,
                "median_transitions_per_s": statistics.median(sps),
                "env_only_s_per_iteration": num_envs * num_steps / statistics.median(sps),
            }
    if names.get("train"):
        coll, learn, files = [], [], []
        pattern = os.path.join(labperf_dir, "results", "train", f"isaaclab_{names['train']}_{num_envs}_s*.log")
        for path in sorted(glob.glob(pattern)):
            files.append(path)
            with open(path, "rb") as fh:
                for line in fh:
                    if b" time: " not in line:
                        continue
                    m = _LOG_LINE.match(line)
                    if m:
                        (coll if m.group(1) == b"Collection time" else learn).append(float(m.group(2)))
        if coll and learn:
            n = min(len(coll), len(learn))
            frac = [learn[i] / (coll[i] + learn[i]) for i in range(n)]
            out["production_iteration_split"] = {
                "files": files,
                "iterations": n,
                "collection_s": _stats(coll[:n]),
                "learning_s": _stats(learn[:n]),
                "learning_fraction": _stats(frac),
                "note": "rsl_rl OnPolicyRunner time.time() split: collection = num_steps x (act + env.step + "
                "process_env_step + logger); learning = compute_returns + update (until its last .item())",
            }
    return out


def iteration_projection(ctx: dict, modes: dict, T: int) -> dict:
    """Learner share of a production iteration (labperf medians) with the stock and the graphed learner."""
    split = ctx.get("production_iteration_split")
    if not split:
        return {}
    coll, learn = split["collection_s"]["median"], split["learning_s"]["median"]
    out = {
        "production_collection_s": coll,
        "production_learning_s": learn,
        "production_learner_fraction": learn / (coll + learn),
    }
    for mode, m in modes.items():
        stock_s = m["stock"]["learning_phase_wall_ms"]["median"] / 1e3
        graphed_s = m["graphed"]["learning_phase_wall_ms"]["median"] / 1e3
        act_saving = T * (m["act"]["stock_act_wall_ms"]["median"] - m["act"]["graphed_act_wall_ms"]["median"]) / 1e3
        coll_g = coll - act_saving
        out[mode] = {
            "measured_stock_learner_s": stock_s,
            "measured_stock_over_production": stock_s / learn,
            "graphed_learner_s": graphed_s,
            "learner_fraction_with_graphed_learner": graphed_s / (coll + graphed_s),
            "iteration_speedup_graphed_learner": (coll + learn) / (coll + graphed_s),
            "act_saving_per_iteration_s": act_saving,
            "learner_fraction_with_graphed_learner_and_act": graphed_s / (coll_g + graphed_s),
            "iteration_speedup_graphed_learner_and_act": (coll + learn) / (coll_g + graphed_s),
        }
    return out


# --------------------------------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------------------------------


def _recipe_check(alg, spec: dict) -> dict:
    got = {"actor": alg.actor.mlp[0].in_features, "critic": alg.critic.mlp[0].in_features}
    expected = spec["expected_mlp_inputs"]
    return {"mlp_input_widths": got, "expected": expected, "match": got == expected}


def run_task(key: str, args) -> dict:
    spec = TASKS[key]
    t0 = time.perf_counter()
    cfg = resolve_agent_cfg(spec["task"], spec["overrides"])
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    alg = build_stock_ppo(cfg, spec, args.num_envs, args.device)
    T = cfg["num_steps_per_env"]
    flops = flop_counts(alg, spec, args.num_envs, T, args.device)
    a = cfg["algorithm"]
    storage_bytes = {g: t.numel() * t.element_size() for g, t in alg.storage.observations.items()}
    out = {
        "task": spec["task"],
        "overrides": spec["overrides"],
        "num_envs": args.num_envs,
        "num_steps_per_env": T,
        "obs_groups_spec": {g: {"shape": list(s), "dtype": d} for g, (s, d) in spec["obs"].items()},
        "act_dim": spec["act_dim"],
        "actor": {
            k: cfg["actor"].get(k)
            for k in ("class_name", "hidden_dims", "activation", "obs_normalization", "cnn_cfg", "init_temperature")
        },
        "critic": {k: cfg["critic"].get(k) for k in ("class_name", "hidden_dims", "activation", "obs_normalization")},
        "actor_distribution": cfg["actor"].get("distribution_cfg"),
        "obs_groups": cfg["obs_groups"],
        "algorithm": {
            k: a.get(k)
            for k in (
                "class_name",
                "num_learning_epochs",
                "num_mini_batches",
                "learning_rate",
                "schedule",
                "desired_kl",
                "clip_param",
                "gamma",
                "lam",
                "entropy_coef",
                "value_loss_coef",
                "max_grad_norm",
                "optimizer",
                "use_clipped_value_loss",
                "normalize_advantage_per_mini_batch",
                "use_mixed_precision",
                "share_cnn_encoders",
            )
        },
        "mini_batch_size": args.num_envs * T // a["num_mini_batches"],
        "actor_params": sum(p.numel() for p in alg.actor.parameters()),
        "critic_params": sum(p.numel() for p in alg.critic.parameters()),
        "recipe_check": _recipe_check(alg, spec),
        "rollout_storage_obs_bytes": storage_bytes,
        "rollout_storage_obs_bytes_total": sum(storage_bytes.values()),
        "flops": flops,
        "modes": {},
    }
    lr0 = a["learning_rate"]
    for mode in args.modes:
        print(f"[{key}] {mode} ...", flush=True)
        with _backend_flags(tf32=mode != "fp32"):
            out["modes"][mode] = measure_mode(alg, spec, flops, args, mode, lr0)
        _print_mode_line(key, out, mode)
    if args.labperf_dir:
        out["labperf"] = labperf_context(args.labperf_dir, spec, args.num_envs, T)
        out["iteration_projection"] = iteration_projection(out["labperf"], out["modes"], T)
    out["setup_and_measure_s"] = time.perf_counter() - t0
    del alg
    gc.collect()
    torch.cuda.empty_cache()
    return out


def _compact(d: dict) -> dict:
    """Per-task summary in seconds / bytes."""
    f = d["flops"]
    c = {
        "task": d["task"],
        "num_envs": d["num_envs"],
        "num_steps_per_env": d["num_steps_per_env"],
        "mini_batch_size": d["mini_batch_size"],
        "samples_per_update": f["samples_per_update"],
        "update_gflop": f["update_flops"] / 1e9,
        "update_gflop_by_kind": {k: v / 1e9 for k, v in f["update_flops_by_kind"].items()},
        "train_mflop_per_sample": f["train_flops_per_sample"] / 1e6,
        "act_gflop_per_env_step": f["act_flops_per_env_step"] / 1e9,
        "timing": "median synchronized wall time, compute_returns + update(); act = PPO.act per env step",
    }
    for mode, m in d["modes"].items():
        p = m["profile"]
        mem = m["memory"]
        c[mode] = {
            "stock_s": m["stock"]["learning_phase_wall_ms"]["median"] / 1e3,
            "stock_min_s": m["stock"]["learning_phase_wall_ms"]["min"] / 1e3,
            "stock_kernel_s": m["stock"]["kernel_ms"] / 1e3,
            "graphed_s": m["graphed"]["learning_phase_wall_ms"]["median"] / 1e3,
            "graphed_min_s": m["graphed"]["learning_phase_wall_ms"]["min"] / 1e3,
            "graphed_kernel_s": m["graphed"]["kernel_ms"] / 1e3,
            "speedup": m["graphed"]["speedup_vs_stock_wall"],
            "stock_tflops": m["stock"]["tflops_wall"],
            "graphed_tflops": m["graphed"]["tflops_wall"],
            "stock_kernel_ms_by_category": p["stock_learning_phase"]["kernel_ms_by_category"],
            "graphed_kernel_ms_by_category": p["graphed_learning_phase"]["kernel_ms_by_category"],
            "stock_gpu_kernels": p["stock_learning_phase"]["gpu_kernels"],
            "graphed_gpu_kernels": p["graphed_learning_phase"]["gpu_kernels"],
            "stock_host_launches": p["stock_learning_phase"]["host_kernel_launches"]
            + p["stock_learning_phase"]["host_graph_launches"],
            "graphed_host_launches": p["graphed_learning_phase"]["host_kernel_launches"]
            + p["graphed_learning_phase"]["host_graph_launches"],
            "stock_host_syncs": p["stock_learning_phase"]["host_syncs"],
            "graphed_host_syncs": p["graphed_learning_phase"]["host_syncs"],
            "act_stock_s_per_step": m["act"]["stock_act_wall_ms"]["median"] / 1e3,
            "act_graphed_s_per_step": m["act"]["graphed_act_wall_ms"]["median"] / 1e3,
            "act_graphed_bf16_s_per_step": m["act"].get("graphed_act_bf16_wall_ms", {}).get("median", float("nan"))
            / 1e3,
            "act_stock_kernels": p["stock_act"]["gpu_kernels"],
            "act_graphed_kernels": p["graphed_act"]["gpu_kernels"],
            "act_stock_host_syncs": p["stock_act"]["host_syncs"],
            "process_env_step_s": m["act"]["stock_process_env_step_wall_ms"]["median"] / 1e3,
            "saved_activation_bytes_per_minibatch": mem["saved_activations"]["bytes"],
            "saved_activation_bytes_per_sample": mem["per_sample"]["saved_activation_bytes"],
            "gathered_input_bytes_per_sample": mem["per_sample"]["gathered_inputs_bytes"],
            "stock_minibatch_peak_bytes": mem["stock_update"]["minibatch_step_peak_bytes_max"],
            "stock_normalizer_pass_peak_bytes": mem["stock_update"]["normalizer_pass_peak_bytes"],
            "graphed_eager_step_peak_bytes": mem["graphed_eager_step"]["peak_bytes"],
            "graph_pools_reserved_bytes": m["graph_memory"]["graph_pools_reserved_bytes"],
            "equivalence_passed": m["equivalence"]["passed"],
        }
    if "iteration_projection" in d:
        c["iteration_projection"] = d["iteration_projection"]
    return c


def _print_mode_line(key: str, d: dict, mode: str) -> None:
    m = d["modes"][mode]
    e = m["equivalence"]
    ps, fr = e["per_step_vs_stock"], e["free_running"]
    mem = m["memory"]
    print(
        f"[{key}] {mode}: stock {m['stock']['learning_phase_wall_ms']['median']:.2f} ms, graphed "
        f"{m['graphed']['learning_phase_wall_ms']['median']:.3f} ms ({m['graphed']['speedup_vs_stock_wall']:.2f}x), "
        f"act stock {m['act']['stock_act_wall_ms']['median']:.3f} / graphed "
        f"{m['act']['graphed_act_wall_ms']['median']:.3f} ms/step; saved act. "
        f"{mem['per_sample']['saved_activation_bytes'] / 1e3:.1f} kB/sample, stock mb peak "
        f"{mem['stock_update']['minibatch_step_peak_bytes_max'] / 1e9:.2f} GB"
        f"\n    check {'PASS' if e['passed'] else 'FAIL'}: graph vs eager max|dp| "
        f"{e['graph_vs_eager']['max_abs_param_diff']:.1e} (eager vs eager floor "
        f"{e['graph_vs_eager']['nondeterminism_floor_max_abs']:.1e}); per-step vs stock "
        f"{ps['steps_passed']}/{ps['steps']} steps, worst max|dp| {ps['worst_step']['max_abs_param_diff']:.1e} (step "
        f"moved {ps['worst_step']['max_abs_param_change_by_update']:.1e}), lr mismatches {ps['lr_mismatches']}; "
        f"normalizer vs "
        f"stock {'ok' if e['normalizer_vs_stock']['passed'] else 'DIFF'}; free-running stock vs graphed "
        f"{fr['stock_vs_graphed']['max_abs_param_diff']:.1e}, stock vs stock(graph's Adam) "
        f"{fr['control_stock_vs_stock_with_graph_adam']['max_abs_param_diff']:.1e} "
        f"(update moved {fr['stock_vs_graphed']['max_abs_param_change_by_update']:.1e})",
        flush=True,
    )


def _table(results: dict) -> str:
    hdr = (
        "task",
        "mode",
        "samples/upd",
        "GFLOP/upd",
        "stock ms",
        "graphed ms",
        "x",
        "TFLOP/s st/gr",
        "kernel ms st/gr",
        "conv ms st/gr",
        "GPU kernels st/gr",
        "host launches st/gr",
        "syncs st/gr",
        "act ms/step st/gr",
        "check",
    )
    rows = [hdr]
    for key, d in results.items():
        for mode, m in d["modes"].items():
            p = m["profile"]
            sp, gp = p["stock_learning_phase"], p["graphed_learning_phase"]
            rows.append(
                (
                    key,
                    mode,
                    f"{d['flops']['samples_per_update']}",
                    f"{d['flops']['learning_phase_flops'] / 1e9:.1f}",
                    f"{m['stock']['learning_phase_wall_ms']['median']:.2f}",
                    f"{m['graphed']['learning_phase_wall_ms']['median']:.2f}",
                    f"{m['graphed']['speedup_vs_stock_wall']:.2f}",
                    f"{m['stock']['tflops_wall']:.1f}/{m['graphed']['tflops_wall']:.1f}",
                    f"{m['stock']['kernel_ms']:.1f}/{m['graphed']['kernel_ms']:.1f}",
                    f"{sp['conv_kernel_ms']:.1f}/{gp['conv_kernel_ms']:.1f}",
                    f"{sp['gpu_kernels']}/{gp['gpu_kernels']}",
                    f"{sp['host_kernel_launches'] + sp['host_graph_launches']}/"
                    f"{gp['host_kernel_launches'] + gp['host_graph_launches']}",
                    f"{sp['host_syncs']}/{gp['host_syncs']}",
                    f"{m['act']['stock_act_wall_ms']['median']:.3f}/{m['act']['graphed_act_wall_ms']['median']:.3f}",
                    "PASS" if m["equivalence"]["passed"] else "FAIL",
                )
            )
    widths = [max(len(r[i]) for r in rows) for i in range(len(hdr))]
    lines = ["  ".join(c.rjust(w) for c, w in zip(r, widths)) for r in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    lines += [
        "ms = median synchronized wall time of compute_returns + update(); kernel ms = summed GPU kernel time;",
        "GFLOP/TFLOP/s = conv + GEMM FLOPs of the learning phase; host launches = kernel + graph launch API calls;",
        f"check: allclose(rtol={CHECK_RTOL}, atol={CHECK_ATOL}) + same LR for graph replay vs eager (whole update) and"
        " for every step of a stock PPO.update, teacher-forced from the stock state before the step; normalizer"
        " statistics vs stock.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tasks", nargs="+", default=["dex_rgb64", "dex_state"], choices=list(TASKS))
    ap.add_argument("--modes", nargs="+", default=["tf32", "bf16"], choices=MODES)
    ap.add_argument("--repeats", type=int, default=20, help="timed runs per variant (stock / graphed interleaved)")
    ap.add_argument("--out", default=None, help="output JSON path")
    ap.add_argument("--num_envs", "--num-envs", type=int, default=4096)
    ap.add_argument("--warmup", type=int, default=3, help="untimed runs per variant before timing")
    ap.add_argument("--profile_runs", "--profile-runs", type=int, default=2, help="torch.profiler passes per variant")
    ap.add_argument(
        "--graphed_adam",
        "--graphed-adam",
        choices=tuple(GRAPHED_ADAM),
        default="fused",
        help="Adam of the graphed update (both capturable, device-tensor lr; same math as stock)",
    )
    ap.add_argument("--labperf_dir", "--labperf-dir", default=None, help="labperf checkout (read-only) for context")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    _import_torch()
    torch.cuda.set_device(args.device)
    warnings.filterwarnings("ignore", message=".*Profiler clears events at the end of each cycle.*")
    from importlib.metadata import version

    meta = {
        "script": os.path.relpath(__file__, os.path.join(os.path.dirname(__file__), "..", "..", "..")),
        "argv": sys.argv,
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "loadavg": os.getloadavg(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "rsl_rl_lib": version("rsl-rl-lib"),
        "gpu": torch.cuda.get_device_name(args.device),
        "sm_count": torch.cuda.get_device_properties(args.device).multi_processor_count,
        "nvidia_smi_start": _nvidia_smi(),
        "args": vars(args),
        "notes": [
            "wall = synchronized host wall time (includes launch overhead and host syncs); gpu_span = CUDA events",
            "kernel_ms = sum of GPU kernel durations from torch.profiler (busy time, excludes launch gaps)",
            "host launches = cudaLaunchKernel-family + cudaGraphLaunch API calls seen by the profiler",
            "FLOPs = 2 x MACs of all Conv2d / Linear layers (+ spatial-softmax matmul), fwd + bwd (dW; dX where the "
            "layer input requires grad), actor + critic",
            "kernel categories are by kernel-name tokens (first match): conv, gemm, optimizer_foreach, gather_index, "
            "reduce, elementwise, other",
        ],
    }
    results = {}
    for key in args.tasks:
        results[key] = run_task(key, args)
    meta["nvidia_smi_end"] = _nvidia_smi()
    meta["loadavg_end"] = os.getloadavg()
    meta["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    table = _table(results)
    print("\n" + table)
    for key, d in results.items():
        if d.get("iteration_projection"):
            print(f"[{key}] iteration projection: {json.dumps(d['iteration_projection'], default=float)}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        doc = {key: _compact(d) for key, d in results.items()}
        doc["meta"] = meta
        doc["table"] = table.splitlines()
        doc["details"] = results
        with open(args.out, "w") as fh:
            json.dump(doc, fh, indent=1, default=str)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

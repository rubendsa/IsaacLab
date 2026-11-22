# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cost of the RSL-RL PPO learning phase, stock vs. one CUDA graph, on synthetic rollouts of real task shapes.

The agent config of each task is resolved the way the stock RSL-RL train entrypoint resolves it
(``resolve_task_config`` -> ``handle_deprecated_rsl_rl_cfg`` -> ``to_dict``), and the algorithm is built with
``PPO.construct_algorithm``, the call ``OnPolicyRunner.__init__`` makes. No simulator is launched: the real
``RolloutStorage`` is filled with ``num_envs`` x ``num_steps_per_env`` synthetic transitions (random-normal
observations, actions sampled from the policy by ``PPO.act``, random rewards / dones / time-outs) and GAE is computed
by ``PPO.compute_returns`` as usual.

Variants, each timed with TF32 on (the backend flags of the stock train entrypoint) and/or off:

``stock``    ``PPO.compute_returns`` (under ``inference_mode``, as the runner calls it) + ``PPO.update()``.
``graphed``  the same math: GAE, then ``num_learning_epochs`` x ``num_mini_batches`` of actor/critic forward,
             clipped surrogate + clipped value loss + entropy bonus, backward, separate actor and critic
             ``clip_grad_norm_``, Adam and the KL-adaptive learning rate. The minibatch permutation
             (``torch.randperm``, graph-safe RNG) and the whole epochs x minibatches update are captured in ONE CUDA
             graph and GAE in a second one. The learning rate is a device tensor updated with ``torch.where``
             and read by ``Adam(capturable=True)`` (``fused=True`` by default, ``--graphed_adam foreach`` for the
             foreach implementation). Not reproduced: the action sample that ``PPO.update`` draws in every actor
             forward and then discards, and the ``torch.cat`` copies of single observation groups.
``act``      rollout-side policy inference once per env step for all envs: stock ``PPO.act`` (actor forward,
             sample, log-prob, critic forward) and the same ops replayed from a CUDA graph.

Each variant is timed over ``--repeats`` runs, with stock and graphed runs interleaved. Parameters, optimizer state
and learning rate are restored before every run, so every repeat does identical work. Each run records synchronized
host wall time and the CUDA-event GPU span. ``torch.profiler`` passes then give the summed GPU kernel time, the
number of kernels executed and the host-side kernel-launch, graph-launch, memcpy and sync API calls. Analytic GEMM
FLOPs (2 x MACs of every ``nn.Linear`` of actor and critic, forward and backward, over all samples of all epochs;
the first layer needs no input gradient) convert the times into achieved TFLOP/s.

Equivalence (``check_equivalence``) is tested on one update from identical parameters, optimizer state, learning
rate, rollout data and minibatch permutation:

1. The graph replay must match an eager run of the same code.
2. Every one of the epochs x minibatches steps of a stock ``PPO.update`` must be reproduced by the graphed code
   started from the stock state before that step (teacher-forced).

Free-running whole updates are reported but not tested: starting from the policy that collected the rollout, the 20
sequential steps amplify any ~1e-7 rounding difference chaotically. Two stock runs that differ only in the Adam
implementation diverge just as much.

The output JSON has one compact entry per task key (times in seconds, read by ``project_locomotion.py
--update_cost``), plus ``meta`` and ``details`` (everything measured).

Timings depend on host load and on other GPU work: quote numbers only from a quiet host and GPU.

Usage::

    OMP_NUM_THREADS=1 python scripts/sram/projection/measure_update_cost.py \
        --tasks cartpole anymal_c g1 --repeats 20 --tf32 both --out /tmp/update_cost.json
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import io
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from itertools import chain

# obs / action dims as measured from the live environments (understand/locomotion-footprint/data/rl_*.json)
TASKS = {
    "cartpole": {"task": "Isaac-Cartpole-Direct", "obs_dim": 4, "act_dim": 1, "footprint": "rl_cartpole.json"},
    "anymal_c": {
        "task": "IsaacContrib-Velocity-Flat-AnymalC",
        "obs_dim": 48,
        "act_dim": 12,
        "footprint": "rl_anymalc.json",
    },
    "g1": {"task": "Isaac-Velocity-Flat-G1", "obs_dim": 123, "act_dim": 37, "footprint": "rl_g1.json"},
}
DEFAULT_FOOTPRINT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "docs", "sram", "data", "rl"
)
DONE_PROB = 0.005  # per env and step
TIMEOUT_FRACTION = 0.5  # share of the dones that are time-outs (bootstrapped by process_env_step)
CHECK_RTOL, CHECK_ATOL = 1e-4, 1e-5  # stock vs graphed parameters (see check_equivalence)
# Adam variants for the graphed update; both capturable with a device-tensor lr. The capturable foreach Adam of
# torch 2.11 divides by 0-dim tensors off the foreach fast path (~3 kernels per parameter and step); fused: 2 kernels.
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
_GEMM_TOKENS = ("gemm", "gemv", "xmma", "cutlass", "splitk", "magma")
_REGION = "measure_update_cost::region"

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
                    if any(t in e.name.lower() for t in _GEMM_TOKENS):
                        n_gemm += 1
                        gemm_us += d
                        gemm_names[e.name] += 1
            elif e.name.startswith("cu") and r0 <= e.time_range.start <= r1:
                api[e.name] += 1
                if e.name in _SYNC_APIS or e.name.startswith(("cudaMemcpy", "cuMemcpy")):
                    chain, parent = [], e.cpu_parent
                    while parent is not None and parent.name != _REGION:
                        chain.append(parent.name)
                        parent = parent.cpu_parent
                    src = " > ".join(reversed(chain[-4:])) or "<harness>"
                    (sync_src if e.name in _SYNC_APIS else memcpy_src)[src] += 1
        per_run.append(
            {
                "kernel_ms": k_us / 1e3,
                "gemm_kernel_ms": gemm_us / 1e3,
                "memcpy_ms": cp_us / 1e3,
                "memset_ms": ms_us / 1e3,
                "gpu_active_span_ms": (g1 - g0) / 1e3 if n_k + n_cp + n_ms else 0.0,
                "gpu_kernels": n_k,
                "gemm_kernels": n_gemm,
                "gpu_memcpy": n_cp,
                "gpu_memset": n_ms,
                "host_kernel_launches": sum(v for k, v in api.items() if k in _KERNEL_LAUNCH_APIS),
                "host_graph_launches": sum(v for k, v in api.items() if k in _GRAPH_LAUNCH_APIS),
                "host_memcpy_calls": sum(v for k, v in api.items() if k.startswith(("cudaMemcpy", "cuMemcpy"))),
                "host_memset_calls": sum(v for k, v in api.items() if k.startswith(("cudaMemset", "cuMemset"))),
                "host_syncs": sum(v for k, v in api.items() if k in _SYNC_APIS),
                "host_api_calls": dict(api),
                "sync_sources": dict(sync_src.most_common(8)),
                "memcpy_sources": dict(memcpy_src.most_common(8)),
                "top_kernels_ms": {k[:120]: v / 1e3 for k, v in kernels.most_common(8)},
                "top_kernels_by_count": {k[:120]: v for k, v in kernel_counts.most_common(10)},
                "gemm_kernel_names": {k[:160]: v for k, v in gemm_names.most_common(4)},
            }
        )
    out = dict(per_run[-1])
    for key in ("kernel_ms", "gemm_kernel_ms", "memcpy_ms", "memset_ms", "gpu_active_span_ms"):
        out[key] = statistics.median(r[key] for r in per_run)
    out["kernel_ms_runs"] = [r["kernel_ms"] for r in per_run]
    counts = {(r["gpu_kernels"], r["host_kernel_launches"], r["host_graph_launches"], r["host_syncs"]) for r in per_run}
    out["counts_stable_across_runs"] = len(counts) == 1
    out["profiler_runs"] = runs
    return out


def _sum_profiles(*profs: dict) -> dict:
    keys = [k for k, v in profs[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    out = {k: sum(p[k] for p in profs) for k in keys if k != "profiler_runs"}
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
def _tf32(on: bool):
    """Backend flags of the stock RSL-RL train entrypoint (``train_rsl_rl.run``); "off" flips only the TF32 flags."""
    flags = dict(cuda_matmul_allow_tf32=on, cudnn_allow_tf32=on, cudnn_deterministic=False, cudnn_benchmark=True)
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
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = on
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


def resolve_agent_cfg(task_id: str, physics: str) -> dict:
    """Agent cfg as the RSL-RL train entrypoint resolves it, as the dict OnPolicyRunner receives."""
    from importlib.metadata import version

    from isaaclab.utils import to_dict

    from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg

    import isaaclab_tasks  # noqa: F401  (registers the tasks)
    from isaaclab_tasks.utils import resolve_task_config

    _, agent_cfg = resolve_task_config(task_id, "rsl_rl_cfg_entry_point", overrides=[f"physics={physics}"])
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))
    cfg = to_dict(agent_cfg)
    cfg["multi_gpu"] = None  # what OnPolicyRunner._configure_multi_gpu() sets for a single-GPU run
    return cfg


class _SyntheticVecEnv:
    """The two VecEnv attributes ``PPO.construct_algorithm`` reads."""

    def __init__(self, num_envs: int, num_actions: int):
        self.num_envs = num_envs
        self.num_actions = num_actions


def _synthetic_obs(num_envs: int, obs_dim: int, device: str):
    from tensordict import TensorDict

    return TensorDict({"policy": torch.randn(num_envs, obs_dim, device=device)}, batch_size=[num_envs], device=device)


def build_stock_ppo(cfg: dict, num_envs: int, obs_dim: int, num_actions: int, device: str):
    from rsl_rl.utils import resolve_callable

    groups = {g for gs in cfg.get("obs_groups", {}).values() for g in gs}
    if groups - {"policy"}:
        raise NotImplementedError(f"synthetic rollouts only provide the 'policy' obs group, cfg needs {groups}")
    alg_class = resolve_callable(cfg["algorithm"]["class_name"])
    with contextlib.redirect_stdout(io.StringIO()):  # construct_algorithm prints both models
        alg = alg_class.construct_algorithm(
            _synthetic_obs(num_envs, obs_dim, device),
            _SyntheticVecEnv(num_envs, num_actions),
            copy.deepcopy(cfg),
            device,
        )
    alg.train_mode()
    return alg


def stock_params(alg) -> list:
    return list(chain(alg.actor.parameters(), alg.critic.parameters()))  # = optimizer param order


def synthetic_rollout(alg, num_envs: int, obs_dim: int, device: str, act_t=None, store_t=None):
    """Fill ``alg.storage`` like ``OnPolicyRunner.learn()`` does, with a synthetic env; return the last observation.

    ``act_t`` / ``store_t`` collect per-step (wall ms, GPU span ms) of ``PPO.act`` / ``PPO.process_env_step``.
    """
    alg.storage.clear()
    obs = _synthetic_obs(num_envs, obs_dim, device)
    for _ in range(alg.storage.num_transitions_per_env):
        # env.step() outputs, generated before the timed regions
        next_obs = _synthetic_obs(num_envs, obs_dim, device)
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


def snapshot(params: list, optimizer, lr: float) -> dict:
    with torch.no_grad():
        return {
            "params": [p.detach().clone() for p in params],
            "opt": [
                {k: v.detach().clone() for k, v in optimizer.state[p].items() if torch.is_tensor(v)} for p in params
            ],
            "lr": float(lr),
        }


def _restore(snap: dict, params: list, optimizer) -> None:
    with torch.no_grad():
        for p, p0 in zip(params, snap["params"], strict=True):
            p.copy_(p0)
        for p, s0 in zip(params, snap["opt"], strict=True):
            state = optimizer.state[p]
            for k, v0 in s0.items():
                state[k].copy_(v0)  # also CPU <-> CUDA for Adam's `step`


def restore_stock(alg, snap: dict) -> None:
    _restore(snap, stock_params(alg), alg.optimizer)
    alg.learning_rate = snap["lr"]
    for group in alg.optimizer.param_groups:
        group["lr"] = snap["lr"]


# --------------------------------------------------------------------------------------------------------------------
# Graphed: same math as PPO.compute_returns + PPO.update, captured in CUDA graphs
# --------------------------------------------------------------------------------------------------------------------


def _latent(model, obs: dict):
    groups = model.obs_groups
    x = obs[groups[0]] if len(groups) == 1 else torch.cat([obs[g] for g in groups], dim=-1)
    return model.obs_normalizer(x)


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


class GraphedPPOLearner:
    """``PPO.compute_returns`` + ``PPO.update`` for feed-forward ``MLPModel`` actor/critic, replayed from 2 graphs.

    The actor/critic are deep copies of the stock ones and use the rsl_rl modules for everything: ``mlp``,
    ``distribution.update / log_prob / entropy / params / kl_divergence``. Only the control flow differs from stock:
    the rollout lives in static buffers (same layout as ``RolloutStorage``), the minibatch permutation is drawn in
    the graph, there is no host sync, and the KL-adaptive LR is a device tensor that drives ``Adam(capturable=True)``.
    """

    def __init__(self, alg, device: str, adam_impl: str = "fused"):
        from rsl_rl.utils import resolve_optimizer

        unsupported = [n for n in ("rnd", "symmetry") if getattr(alg, n)]
        unsupported += ["use_mixed_precision"] if alg.use_mixed_precision else []
        unsupported += ["recurrent"] if alg.actor.is_recurrent or alg.critic.is_recurrent else []
        unsupported += ["obs_normalization"] if alg.actor.obs_normalization or alg.critic.obs_normalization else []
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
        self.optimizer = resolve_optimizer(opt_class.__name__)(
            self.params,
            lr=torch.tensor(alg.learning_rate, device=device),
            **GRAPHED_ADAM[adam_impl],
            **_adam_hyperparams(alg),
        )
        self.lr = self.optimizer.param_groups[0]["lr"]  # device tensor read by the captured Adam kernels
        assert torch.is_tensor(self.lr) and self.lr.is_cuda

        # static rollout buffers, RolloutStorage layout
        self.groups = sorted(set(self.actor.obs_groups) | set(self.critic.obs_groups))
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
        self.lr.fill_(snap["lr"])

    # -- the math (eager-callable; captured below) ---------------------------------------------------------------------

    def _returns_body(self) -> None:
        """== PPO.compute_returns."""
        with torch.no_grad():
            last_values = self.critic.mlp(_latent(self.critic, self.last_obs))
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
        """== PPO.update for a feed-forward actor/critic without RND, symmetry or mixed precision."""
        # one permutation per update, shared by all epochs (RolloutStorage.mini_batch_generator)
        torch.randperm(self.M * self.mb, device=self.device, out=self.perm)
        self.loss_sums.zero_()
        for _ in range(self.E):
            for i in range(self.M):
                self._minibatch_step(self.perm[i * self.mb : (i + 1) * self.mb])
        with torch.no_grad():
            self.loss_sums /= self.E * self.M
        # release the last autograd graph: it would keep the warm-up's AccumulateGrad nodes (side stream) alive
        self.actor.distribution._distribution = None

    def _minibatch_step(self, idx) -> None:
        """One iteration of the minibatch loop of PPO.update: KL-adaptive LR, loss, backward, clipping, Adam."""
        dist = self.actor.distribution
        b_obs = {g: o.flatten(0, 1)[idx] for g, o in self.obs.items()}
        b_actions, b_values, b_returns, b_adv, b_old_log_prob = (
            t.flatten(0, 1)[idx]
            for t in (self.actions, self.values, self.returns, self.advantages, self.actions_log_prob)
        )
        b_old_dist_params = tuple(p.flatten(0, 1)[idx] for p in self.dist_params)
        if self.norm_adv_per_mb:
            with torch.no_grad():
                b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        dist.update(self.actor.mlp(_latent(self.actor, b_obs)))
        actions_log_prob = dist.log_prob(b_actions)
        value_pred = self.critic.mlp(_latent(self.critic, b_obs))
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

        self.optimizer.zero_grad(set_to_none=False)  # grads stay static tensors
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optimizer.step()
        with torch.no_grad():
            self.loss_sums += torch.stack((value_loss, surrogate_loss, entropy_mean)).detach()

    # -- capture / replay ----------------------------------------------------------------------------------------------

    def capture(self, warmup: int = 2) -> None:
        """Warm up eagerly on a side stream (creates grads, Adam state, cuBLAS workspaces), then capture.

        Warm-up changes parameters and optimizer state: call ``set_state`` afterwards.
        """
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                self._returns_body()
                self._update_body()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self.g_returns = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g_returns):
            self._returns_body()
        self.g_update = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g_update):
            self._update_body()
        torch.cuda.synchronize()

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
    ``torch.normal(loc, scale)``, whose ``scale >= 0`` check is a host sync and cannot be captured.
    """

    def __init__(self, actor, critic, obs_like: dict):
        from rsl_rl.modules.distribution import GaussianDistribution

        if not isinstance(actor.distribution, GaussianDistribution):
            raise NotImplementedError("graphed act implements Gaussian action distributions only")
        self.actor, self.critic = actor, critic
        self.obs = {g: torch.randn_like(o) for g, o in obs_like.items()}
        self.graph = None

    def _body(self) -> None:
        with torch.no_grad():
            dist = self.actor.distribution
            dist.update(self.actor.mlp(_latent(self.actor, self.obs)))
            self.actions = dist._distribution.rsample()
            self.values = self.critic.mlp(_latent(self.critic, self.obs))
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
# FLOPs, equivalence check, per-mode measurement
# --------------------------------------------------------------------------------------------------------------------


def flop_counts(alg, num_envs: int, num_steps: int) -> dict:
    """Analytic GEMM FLOPs (2 x MACs of the nn.Linear layers; biases, activations, losses and Adam not counted)."""

    def layers(model):
        return [(m.in_features, m.out_features) for m in model.mlp if isinstance(m, nn.Linear)]

    out = {}
    train_macs = fwd_macs = naive_macs = 0
    for name, model in (("actor", alg.actor), ("critic", alg.critic)):
        ls = layers(model)
        fwd = sum(i * o for i, o in ls)
        train = 3 * fwd - ls[0][0] * ls[0][1]  # fwd + dW + dX, no dX for the first layer (obs need no grad)
        out[f"{name}_layers"] = ls
        out[f"{name}_fwd_macs_per_sample"] = fwd
        out[f"{name}_train_macs_per_sample"] = train
        fwd_macs, train_macs, naive_macs = fwd_macs + fwd, train_macs + train, naive_macs + 3 * fwd
    samples = num_envs * num_steps
    epochs = alg.num_learning_epochs
    out["samples_per_update"] = samples
    out["samples_processed_per_update"] = samples * epochs
    out["update_flops"] = 2 * train_macs * samples * epochs
    out["update_flops_6x_fwd_macs"] = 2 * naive_macs * samples * epochs  # convention of project_locomotion.py
    out["returns_flops"] = 2 * out["critic_fwd_macs_per_sample"] * num_envs  # last-value critic forward
    out["learning_phase_flops"] = out["update_flops"] + out["returns_flops"]
    out["act_flops_per_env_step"] = 2 * fwd_macs * num_envs
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


def _params_of(params: list) -> list:
    return [p.detach().clone() for p in params]


def check_equivalence(alg, learner: GraphedPPOLearner, snap: dict, last_obs, seed: int, adam_impl: str) -> dict:
    """Stock ``PPO.compute_returns`` + ``PPO.update`` vs the graphed update, same state, data and minibatch order.

    A free-running comparison of whole updates is not a usable test: starting from the policy that collected the
    rollout, the 20 sequential steps (PPO's clipping ``max``, Adam's normalisation of near-zero gradients) amplify any
    ~1e-7 rounding difference -- e.g. stock's foreach Adam vs a fused Adam, both run by the stock code -- to ~1e-3 of
    the update. So the check is split:

    1. ``graph_vs_eager``: graph replay vs the same code run eagerly, whole update -> capture changes nothing.
    2. ``per_step_vs_stock`` (teacher-forced): for each of the epochs x minibatches steps of a stock ``PPO.update``,
       the state before the step (parameters, Adam moments, step count, LR) is recorded via optimizer step hooks; one
       step of the graphed code runs from that state on the same minibatch and must reproduce the stock parameters
       after the step and the stock KL-adapted LR.
    Pass = (1) and (2) within allclose(rtol, atol) and same LR at every step. ``free_running`` reports whole-update
    differences (stock vs graphed, and stock vs stock with the graph's Adam variant as the control) for information.
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
    torch.cuda.manual_seed(seed)
    perm = torch.randperm(M * mb, device=alg.device)  # the stock permutation (same seed, same call)

    # -- (1) graph replay vs eager run of the same code ----------------------------------------------------------------
    learner.set_state(snap)
    torch.cuda.manual_seed(seed)  # randperm in the update graph is its first RNG draw -> same permutation
    losses_graphed = learner.replay()
    p_graphed, lr_graphed = _params_of(learner.params), float(learner.lr)
    learner.set_state(snap)
    torch.cuda.manual_seed(seed)
    learner._returns_body()
    learner._update_body()
    graph_vs_eager = _param_diff(p_graphed, _params_of(learner.params), snap["params"])
    graph_vs_eager["same_lr"] = float(learner.lr) == lr_graphed

    # -- (2) teacher-forced per-step comparison with stock -------------------------------------------------------------
    learner.set_state(snap)
    learner._returns_body()  # returns / advantages with the critic the update starts from (as compute_returns)
    worst, lr_mismatch, n_ok = None, 0, 0
    for k, (before, after) in enumerate(zip(pre, post, strict=True)):
        lr_before = snap["lr"] if k == 0 else pre[k - 1]["lr"]
        learner.set_state({"params": before["params"], "opt": before["opt"], "lr": lr_before})
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
    torch.cuda.synchronize()

    passed = graph_vs_eager["allclose"] and graph_vs_eager["same_lr"] and n_ok == len(pre) == learner.E * M
    return {
        "criterion": f"allclose(rtol={CHECK_RTOL}, atol={CHECK_ATOL}) and same LR: graph replay vs eager same code "
        "(whole update) and every teacher-forced step vs stock PPO.update",
        "passed": bool(passed),
        "graph_vs_eager": graph_vs_eager,
        "per_step_vs_stock": per_step,
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


def measure_mode(alg, spec: dict, flops: dict, args, tf32: bool, lr0: float) -> dict:
    N, dev = args.num_envs, args.device
    T = alg.storage.num_transitions_per_env
    res = {
        "tf32": tf32,
        "torch_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "torch_matmul_fp32_precision": getattr(torch.backends.cuda.matmul, "fp32_precision", None),
        "nvidia_smi_before": _nvidia_smi(),  # other GPU work during this measurement
        "loadavg_before": os.getloadavg(),
    }

    # -- warm-up: rollout + learning phase, as in training -------------------------------------------------------------
    for _ in range(max(1, args.warmup)):
        last_obs = synthetic_rollout(alg, N, spec["obs_dim"], dev)
        stock_learning_phase(alg, last_obs)

    # -- (d) stock rollout-side inference: rollouts of the current policy, no update in between ------------------------
    act_t, store_t = [], []
    for _ in range(max(3, math.ceil(args.repeats / 4))):
        last_obs = synthetic_rollout(alg, N, spec["obs_dim"], dev, act_t, store_t)

    # -- the state every timed learning phase starts from: the policy that generated the stored rollout (so the first
    #    minibatch sees KL ~ 0, as in training), its Adam moments, and the configured learning rate --------------------
    alg.learning_rate = lr0
    for group in alg.optimizer.param_groups:
        group["lr"] = lr0
    snap = snapshot(stock_params(alg), alg.optimizer, alg.learning_rate)

    # -- (b) graphed learner + graphed act -----------------------------------------------------------------------------
    t0 = time.perf_counter()
    learner = GraphedPPOLearner(alg, dev, args.graphed_adam)
    res["graphed_adam"] = {"class": type(learner.optimizer).__name__, **GRAPHED_ADAM[args.graphed_adam]}
    learner.load_rollout(alg.storage, last_obs)
    learner.capture()
    learner.set_state(snap)
    act_graph = GraphedAct(learner.actor, learner.critic, last_obs)
    act_graph.capture()
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
    gact_t = [_timed(lambda _m: act_graph.graph.replay()) for _ in range(args.repeats * T)]
    gact_t = [(w, s[0]) for w, s in gact_t]

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
    prof["stock_learning_phase"] = _sum_profiles(prof["stock_returns"], prof["stock_update"])
    prof["graphed_learning_phase"] = _sum_profiles(prof["graphed_returns"], prof["graphed_update"])

    # -- equivalence (meaningful with TF32 off; informational with TF32 on) --------------------------------------------
    res["equivalence"] = check_equivalence(alg, learner, snap, last_obs, args.seed + 1, args.graphed_adam)
    restore_stock(alg, snap)

    lp_flops = flops["learning_phase_flops"]
    stock_ms, graphed_ms = _stats(st_wall), _stats(gr_wall)
    k_st, k_gr = prof["stock_learning_phase"]["kernel_ms"], prof["graphed_learning_phase"]["kernel_ms"]
    res["stock"] = {
        "learning_phase_wall_ms": stock_ms,
        "returns_gpu_span_ms": _stats(st_ret),
        "update_gpu_span_ms": _stats(st_upd),
        "kernel_ms": k_st,
        "tflops_wall": lp_flops / (stock_ms["median"] * 1e-3) / 1e12,
        "tflops_kernel": lp_flops / (k_st * 1e-3) / 1e12,
    }
    res["graphed"] = {
        "learning_phase_wall_ms": graphed_ms,
        "returns_gpu_span_ms": _stats(gr_ret),
        "update_gpu_span_ms": _stats(gr_upd),
        "kernel_ms": k_gr,
        "tflops_wall": lp_flops / (graphed_ms["median"] * 1e-3) / 1e12,
        "tflops_kernel": lp_flops / (k_gr * 1e-3) / 1e12,
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
    res["act"]["stock_tflops_wall"] = act_flops / (res["act"]["stock_act_wall_ms"]["median"] * 1e-3) / 1e12
    res["act"]["graphed_tflops_wall"] = act_flops / (res["act"]["graphed_act_wall_ms"]["median"] * 1e-3) / 1e12
    res["profile"] = prof

    learner = act_graph = None  # release the captured CUDA graphs
    gc.collect()
    torch.cuda.empty_cache()
    return res


# --------------------------------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------------------------------


def _footprint_crosscheck(path: str, cfg: dict, spec: dict, alg) -> dict:
    if not os.path.isfile(path):
        return {"file": path, "checked": False}
    with open(path) as fh:
        ref = json.load(fh)
    mine = {
        "task": spec["task"],
        "obs_dim": spec["obs_dim"],
        "num_actions": spec["act_dim"],
        "num_steps_per_env": cfg["num_steps_per_env"],
        "actor_params": sum(p.numel() for p in alg.actor.parameters()),
        "critic_params": sum(p.numel() for p in alg.critic.parameters()),
    }
    mismatches = {k: (v, ref.get(k)) for k, v in mine.items() if ref.get(k) != v}
    for model in ("actor", "critic"):
        if ref.get(f"{model}_cfg", {}).get("hidden_dims") != cfg[model]["hidden_dims"]:
            mismatches[f"{model}_hidden_dims"] = (
                cfg[model]["hidden_dims"],
                ref.get(f"{model}_cfg", {}).get("hidden_dims"),
            )
    return {"file": path, "checked": True, "match": not mismatches, "mismatches": mismatches}


def run_task(key: str, args) -> dict:
    spec = TASKS[key]
    t0 = time.perf_counter()
    cfg = resolve_agent_cfg(spec["task"], args.physics)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    alg = build_stock_ppo(cfg, args.num_envs, spec["obs_dim"], spec["act_dim"], args.device)
    T = cfg["num_steps_per_env"]
    flops = flop_counts(alg, args.num_envs, T)
    a = cfg["algorithm"]
    out = {
        "task": spec["task"],
        "physics": args.physics,
        "num_envs": args.num_envs,
        "num_steps_per_env": T,
        "obs_dim": spec["obs_dim"],
        "act_dim": spec["act_dim"],
        "actor": {k: cfg["actor"].get(k) for k in ("class_name", "hidden_dims", "activation", "obs_normalization")},
        "critic": {k: cfg["critic"].get(k) for k in ("class_name", "hidden_dims", "activation", "obs_normalization")},
        "actor_distribution": cfg["actor"].get("distribution_cfg"),
        "obs_groups": {"actor": alg.actor.obs_groups, "critic": alg.critic.obs_groups},
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
            )
        },
        "mini_batch_size": args.num_envs * T // a["num_mini_batches"],
        "actor_params": sum(p.numel() for p in alg.actor.parameters()),
        "critic_params": sum(p.numel() for p in alg.critic.parameters()),
        "flops": flops,
        "footprint_crosscheck": _footprint_crosscheck(
            os.path.join(args.footprint_dir, spec["footprint"]), cfg, spec, alg
        ),
        "modes": {},
    }
    for mode in ("on", "off") if args.tf32 == "both" else (args.tf32,):
        print(f"[{key}] TF32 {mode} ...", flush=True)
        with _tf32(mode == "on"):
            out["modes"][f"tf32_{mode}"] = measure_mode(alg, spec, flops, args, mode == "on", a["learning_rate"])
        _print_mode_line(key, out, f"tf32_{mode}")
    out["setup_and_measure_s"] = time.perf_counter() - t0
    del alg
    gc.collect()
    torch.cuda.empty_cache()
    return out


def _compact(d: dict) -> dict:
    """Per-task summary in seconds (what project_locomotion.py --update_cost embeds)."""
    c = {
        "task": d["task"],
        "num_envs": d["num_envs"],
        "num_steps_per_env": d["num_steps_per_env"],
        "samples_per_update": d["flops"]["samples_per_update"],
        "update_gflop": d["flops"]["update_flops"] / 1e9,
        "update_gflop_6x_fwd_macs": d["flops"]["update_flops_6x_fwd_macs"] / 1e9,
        "timing": "median synchronized wall time, compute_returns + update(); act = PPO.act per env step",
    }
    for mode, m in d["modes"].items():
        p = m["profile"]
        c[mode] = {
            "stock_s": m["stock"]["learning_phase_wall_ms"]["median"] / 1e3,
            "stock_returns_s": m["stock"]["returns_gpu_span_ms"]["median"] / 1e3,
            "stock_update_s": m["stock"]["update_gpu_span_ms"]["median"] / 1e3,
            "stock_kernel_s": m["stock"]["kernel_ms"] / 1e3,
            "graphed_s": m["graphed"]["learning_phase_wall_ms"]["median"] / 1e3,
            "graphed_kernel_s": m["graphed"]["kernel_ms"] / 1e3,
            "stock_tflops": m["stock"]["tflops_wall"],
            "graphed_tflops": m["graphed"]["tflops_wall"],
            "stock_gpu_kernels": p["stock_learning_phase"]["gpu_kernels"],
            "graphed_gpu_kernels": p["graphed_learning_phase"]["gpu_kernels"],
            "stock_host_launches": p["stock_learning_phase"]["host_kernel_launches"]
            + p["stock_learning_phase"]["host_graph_launches"],
            "graphed_host_launches": p["graphed_learning_phase"]["host_kernel_launches"]
            + p["graphed_learning_phase"]["host_graph_launches"],
            "act_stock_s_per_step": m["act"]["stock_act_wall_ms"]["median"] / 1e3,
            "act_graphed_s_per_step": m["act"]["graphed_act_wall_ms"]["median"] / 1e3,
            "equivalence_passed": m["equivalence"]["passed"],
        }
    return c


def _print_mode_line(key: str, d: dict, mode: str) -> None:
    m = d["modes"][mode]
    e = m["equivalence"]
    ps, fr = e["per_step_vs_stock"], e["free_running"]
    print(
        f"[{key}] {mode}: stock {m['stock']['learning_phase_wall_ms']['median']:.2f} ms, graphed "
        f"{m['graphed']['learning_phase_wall_ms']['median']:.3f} ms ({m['graphed']['speedup_vs_stock_wall']:.1f}x), "
        f"act stock {m['act']['stock_act_wall_ms']['median']:.3f} "
        f"/ graphed {m['act']['graphed_act_wall_ms']['median']:.3f}"
        f" ms/step\n    check {'PASS' if e['passed'] else 'FAIL'}: graph vs eager max|dp| "
        f"{e['graph_vs_eager']['max_abs_param_diff']:.1e}; per-step vs stock {ps['steps_passed']}/{ps['steps']} steps, "
        f"worst max|dp| {ps['worst_step']['max_abs_param_diff']:.1e} (step moved "
        f"{ps['worst_step']['max_abs_param_change_by_update']:.1e}), lr mismatches {ps['lr_mismatches']}; free-running "
        f"stock vs graphed {fr['stock_vs_graphed']['max_abs_param_diff']:.1e}, stock vs stock(graph's Adam) "
        f"{fr['control_stock_vs_stock_with_graph_adam']['max_abs_param_diff']:.1e} "
        f"(update moved {fr['stock_vs_graphed']['max_abs_param_change_by_update']:.1e})",
        flush=True,
    )


def _table(results: dict) -> str:
    hdr = (
        "task",
        "TF32",
        "samples/upd",
        "GFLOP/upd",
        "stock ms",
        "graphed ms",
        "x",
        "TFLOP/s st/gr",
        "kernel ms st/gr",
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
                    mode.split("_")[1],
                    f"{d['flops']['samples_per_update']}",
                    f"{d['flops']['learning_phase_flops'] / 1e9:.1f}",
                    f"{m['stock']['learning_phase_wall_ms']['median']:.2f}",
                    f"{m['graphed']['learning_phase_wall_ms']['median']:.3f}",
                    f"{m['graphed']['speedup_vs_stock_wall']:.1f}",
                    f"{m['stock']['tflops_wall']:.2f}/{m['graphed']['tflops_wall']:.1f}",
                    f"{m['stock']['kernel_ms']:.2f}/{m['graphed']['kernel_ms']:.2f}",
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
        "GFLOP/TFLOP/s = GEMM FLOPs of the learning phase; host launches = kernel + graph launch API calls;",
        f"check: allclose(rtol={CHECK_RTOL}, atol={CHECK_ATOL}) + same LR for graph replay vs eager (whole update) and"
        " for every step of a stock PPO.update, teacher-forced from the stock state before the step.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--repeats", type=int, default=20, help="timed runs per variant (stock / graphed interleaved)")
    ap.add_argument("--tf32", choices=("on", "off", "both"), default="both")
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
    ap.add_argument("--physics", default="newton_mjwarp", help="physics preset used to resolve the task config")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--footprint_dir",
        "--footprint-dir",
        default=DEFAULT_FOOTPRINT_DIR,
        help="rl_<task>.json files to cross-check dims/params against (skipped if missing)",
    )
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
        "rsl_rl_lib": version("rsl-rl-lib"),
        "gpu": torch.cuda.get_device_name(args.device),
        "sm_count": torch.cuda.get_device_properties(args.device).multi_processor_count,
        "nvidia_smi_start": _nvidia_smi(),
        "args": vars(args),
        "notes": [
            "wall = synchronized host wall time (includes launch overhead and host syncs); gpu_span = CUDA events",
            "kernel_ms = sum of GPU kernel durations from torch.profiler (busy time, excludes launch gaps)",
            "host launches = cudaLaunchKernel-family + cudaGraphLaunch API calls seen by the profiler",
            "FLOPs = 2 x MACs of all nn.Linear layers, fwd + bwd (dW, dX except first layer), actor + critic",
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

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure fused PPO and spatial-softmax hot paths at the native Kuka camera batch size.

This microbenchmark excludes simulation, rendering, convolutions, optimizer updates,
and rollout gathering. Its speedups must not be presented as whole-training speedups.
Eager CUDA event intervals include CPU launch gaps; captured CUDA Graph replay is
also measured for both implementations to separate those gaps from device execution.
Run on an idle GPU after the independent numerical parity tests pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import triton

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "isaaclab_sram"))

from isaaclab_sram import ppo_kernels, spatial_softmax

from isaaclab_tasks.core.lift.config.kuka_allegro.agents.models import SpatialSoftmax


def torch_gae(rewards, dones, values, last_values, gamma, lam):
    returns = torch.empty_like(values)
    advantage = torch.zeros_like(last_values)
    for step in reversed(range(rewards.shape[0])):
        next_value = last_values if step == rewards.shape[0] - 1 else values[step + 1]
        mask = 1.0 - dones[step].float()
        delta = rewards[step] + mask * gamma * next_value - values[step]
        advantage = delta + mask * gamma * lam * advantage
        returns[step] = advantage + values[step]
    advantages = returns - values
    return returns, (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)


def torch_ppo_loss(mean, std, values, actions, old_logp, old_values, returns, advantages):
    distribution = torch.distributions.Normal(mean, std, validate_args=False)
    ratio = torch.exp(distribution.log_prob(actions).sum(-1) - old_logp)
    surrogate = torch.maximum(-advantages * ratio, -advantages * ratio.clamp(0.8, 1.2)).mean()
    clipped_values = old_values + (values - old_values).clamp(-0.2, 0.2)
    value_loss = torch.maximum((values - returns).square(), (clipped_values - returns).square()).mean()
    entropy = distribution.entropy().sum(-1).mean()
    return surrogate + value_loss - 0.005 * entropy


def measure(call, warmup, repeats):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    gpu_ms, wall_ms = [], []
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        tick = time.perf_counter()
        start.record()
        call()
        stop.record()
        stop.synchronize()
        wall_ms.append(1000.0 * (time.perf_counter() - tick))
        gpu_ms.append(start.elapsed_time(stop))
    return {
        "gpu_ms_mean": statistics.mean(gpu_ms),
        "gpu_ms_median": statistics.median(gpu_ms),
        "gpu_ms_min": min(gpu_ms),
        "wall_ms_mean": statistics.mean(wall_ms),
        "wall_ms_median": statistics.median(wall_ms),
        "repeats": repeats,
    }


def capture(call):
    """Warm up on a side stream before capturing the complete forward/backward call."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    return graph


def kernel_metadata():
    """Record compiler resource usage, including spills that invalidate SRAM residency claims."""
    kernels = {}
    for module in (ppo_kernels, spatial_softmax):
        for name, function in vars(module).items():
            if not isinstance(function, triton.runtime.jit.JITFunction):
                continue
            variants = []
            for cache, *_ in function.device_caches.values():
                for compiled in cache.values():
                    variants.append(
                        {
                            "registers_per_thread": compiled.n_regs,
                            "spill_slots": compiled.n_spills,
                            "shared_memory_bytes": compiled.metadata.shared,
                            "warps": compiled.metadata.num_warps,
                        }
                    )
            if variants:
                kernels[name] = variants
    return kernels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--num-envs", type=int, default=4096)
    args = parser.parse_args()
    if min(args.warmup, args.repeats, args.batch_size, args.num_envs) < 1:
        parser.error("all sizes, warmup, and repeats must be positive")
    torch.manual_seed(11)
    torch.backends.cuda.matmul.allow_tf32 = False
    batch, actions_n = args.batch_size, 23
    mean = torch.randn(batch, actions_n, device="cuda", requires_grad=True)
    std = torch.ones(actions_n, device="cuda", requires_grad=True)
    values = torch.randn(batch, device="cuda", requires_grad=True)
    actions = mean.detach() + torch.randn_like(mean)
    old_logp = torch.distributions.Normal(mean.detach(), std.detach()).log_prob(actions).sum(-1)
    old_values, returns, advantages = (torch.randn(batch, device="cuda") for _ in range(3))
    ppo_inputs = (mean, std, values, actions, old_logp, old_values, returns, advantages)

    def ppo_backward(fused):
        mean.grad = std.grad = values.grad = None
        loss = ppo_kernels.fused_ppo_loss(*ppo_inputs, 0.2, 1.0, 0.005)[0] if fused else torch_ppo_loss(*ppo_inputs)
        loss.backward()

    spatial = SpatialSoftmax(32, 4, 4).cuda()
    features = torch.randn(batch, 32, 4, 4, device="cuda", requires_grad=True)
    spatial_gradient = torch.randn(batch, 64, device="cuda")

    def spatial_backward(fused):
        features.grad = spatial.log_temperature.grad = None
        output = (
            spatial_softmax.fused_spatial_softmax(features, spatial.log_temperature, spatial.pos_x, spatial.pos_y)
            if fused
            else spatial(features)
        )
        output.backward(spatial_gradient)

    rewards = torch.randn(32, args.num_envs, 1, device="cuda")
    dones = (torch.rand_like(rewards) < 0.05).to(torch.uint8)
    rollout_values = torch.randn_like(rewards)
    last_values = torch.randn(args.num_envs, 1, device="cuda")
    gae_inputs = (rewards, dones, rollout_values, last_values, 0.99, 0.95)
    cases = {
        "ppo_forward_backward": (lambda: ppo_backward(False), lambda: ppo_backward(True)),
        "spatial_softmax_forward_backward": (lambda: spatial_backward(False), lambda: spatial_backward(True)),
        "gae_and_global_normalization": (lambda: torch_gae(*gae_inputs), lambda: ppo_kernels.fused_gae(*gae_inputs)),
    }
    results = {}
    for name, (native, fused) in cases.items():
        native_result = measure(native, args.warmup, args.repeats)
        fused_result = measure(fused, args.warmup, args.repeats)
        results[name] = {
            "torch": native_result,
            "fused": fused_result,
            "gpu_median_speedup": native_result["gpu_ms_median"] / fused_result["gpu_ms_median"],
            "wall_median_speedup": native_result["wall_ms_median"] / fused_result["wall_ms_median"],
        }
        native_graph, fused_graph = capture(native), capture(fused)
        native_replay = measure(native_graph.replay, args.warmup, args.repeats)
        fused_replay = measure(fused_graph.replay, args.warmup, args.repeats)
        results[name]["captured_graph"] = {
            "torch": native_replay,
            "fused": fused_replay,
            "gpu_median_speedup": native_replay["gpu_ms_median"] / fused_replay["gpu_ms_median"],
            "wall_median_speedup": native_replay["wall_ms_median"] / fused_replay["wall_ms_median"],
        }
        print(name, json.dumps(results[name]), flush=True)
    paths = (Path(__file__), Path(ppo_kernels.__file__), Path(spatial_softmax.__file__))
    report = {
        "scope": "Isolated hot paths, not end-to-end training; compile and capture excluded by warmup",
        "timing_note": "Eager CUDA events include CPU launch gaps. Captured replay removes most host scheduling cost.",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "shapes": {"ppo": [batch, actions_n], "spatial": list(features.shape), "gae": list(rewards.shape)},
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "results": results,
        "kernel_resources": kernel_metadata(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

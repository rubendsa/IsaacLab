# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure Warp native RGB64 CNN/keypoint experiment, with an independent Torch oracle.

Run in the Isaac Lab environment. The default CPU mode creates no CUDA tensors::

    CUDA_VISIBLE_DEVICES='' uv run --no-sync python scripts/sram/experiment_warp_perception.py \
        --device cpu --batch 2 --output warp_cpu.json

The candidate uses Warp scalar kernels and Tape-generated convolution adjoints.
Its learned spatial-softmax adjoint is explicit. Torch only constructs the actual
native model, supplies frozen inputs/parameters, and evaluates the oracle. This
does not implement PPO, Adam, random sampling, or persistent model-state storage.
No custom CUDA source, native-code injection, or third-party neural library is used.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

try:
    import warp as wp
except ImportError as error:
    raise SystemExit("This optional experiment needs the repository's warp-lang environment.") from error


@wp.kernel(enable_backward=False)
def normalize_rgb(raw: wp.array4d[wp.uint8], normalized: wp.array4d[float]):
    n, c, y, x = wp.tid()
    normalized[n, c, y, x] = float(raw[n, c, y, x]) * (1.0 / 255.0) - 0.5


@wp.kernel(enable_backward=True)
def convolution(
    image: wp.array4d[float],
    weight: wp.array4d[float],
    bias: wp.array[float],
    output: wp.array4d[float],
    stride: int,
):
    n, channel, y, x = wp.tid()
    total = bias[channel]
    for source in range(weight.shape[1]):
        for ky in range(weight.shape[2]):
            for kx in range(weight.shape[3]):
                total += weight[channel, source, ky, kx] * image[n, source, y * stride + ky, x * stride + kx]
    output[n, channel, y, x] = total


@wp.kernel(enable_backward=True)
def elu(image: wp.array4d[float], output: wp.array4d[float]):
    n, channel, y, x = wp.tid()
    value = image[n, channel, y, x]
    if value > 0.0:
        output[n, channel, y, x] = value
    else:
        output[n, channel, y, x] = wp.exp(value) - 1.0


@wp.kernel(enable_backward=False)
def spatial_forward(
    features: wp.array4d[float],
    log_temperature: wp.array[float],
    pos_x: wp.array[float],
    pos_y: wp.array[float],
    probabilities: wp.array3d[float],
    keypoints: wp.array2d[float],
):
    n, c = wp.tid()
    temperature = wp.max(wp.exp(log_temperature[c]), 0.001)
    maximum = float(-1.0e30)
    for k in range(16):
        maximum = wp.max(maximum, features[n, c, k // 4, k % 4] / temperature)
    denominator = float(0.0)
    for k in range(16):
        denominator += wp.exp(features[n, c, k // 4, k % 4] / temperature - maximum)
    expected_x = float(0.0)
    expected_y = float(0.0)
    for k in range(16):
        probability = wp.exp(features[n, c, k // 4, k % 4] / temperature - maximum) / denominator
        probabilities[n, c, k] = probability
        expected_x += probability * pos_x[k]
        expected_y += probability * pos_y[k]
    keypoints[n, c] = expected_x
    keypoints[n, c + 32] = expected_y


@wp.kernel(enable_backward=False)
def spatial_backward(
    features: wp.array4d[float],
    log_temperature: wp.array[float],
    pos_x: wp.array[float],
    pos_y: wp.array[float],
    probabilities: wp.array3d[float],
    keypoints: wp.array2d[float],
    upstream: wp.array2d[float],
    feature_gradient: wp.array4d[float],
    temperature_gradient: wp.array[float],
):
    n, c = wp.tid()
    unclamped = wp.exp(log_temperature[c])
    temperature = wp.max(unclamped, 0.001)
    log_gradient = float(0.0)
    for k in range(16):
        gradient = (
            probabilities[n, c, k]
            * (upstream[n, c] * (pos_x[k] - keypoints[n, c]) + upstream[n, c + 32] * (pos_y[k] - keypoints[n, c + 32]))
            / temperature
        )
        feature_gradient[n, c, k // 4, k % 4] = gradient
        log_gradient -= gradient * features[n, c, k // 4, k % 4]
    # Torch clamp's derivative includes equality at the lower bound. Explicit
    # differentiation avoids Warp's different generic min/max tie convention.
    if unclamped >= 0.001:
        wp.atomic_add(temperature_gradient, c, log_gradient)


def warp_array(tensor, device, *, requires_grad=False):
    """Copy only at the frozen input/output boundary; no candidate Torch compute."""
    data = tensor.detach().cpu().numpy()
    return wp.array(
        data, dtype=wp.uint8 if data.dtype == np.uint8 else wp.float32, device=device, requires_grad=requires_grad
    )


def check_close(name, actual, expected, *, atol=2e-5, rtol=2e-4):
    """Independent value/gradient gates, including finite and nonzero checks."""
    reference = expected.detach().cpu().numpy()
    candidate = actual.numpy()
    np.testing.assert_allclose(candidate, reference, atol=atol, rtol=rtol, err_msg=name)
    if not np.isfinite(candidate).all():
        raise AssertionError(f"Nonfinite {name}")
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "relative_l2": float(np.linalg.norm(difference) / max(np.linalg.norm(reference), 1e-30)),
        "reference_l2": float(np.linalg.norm(reference)),
        "atol": atol,
        "rtol": rtol,
    }


def check_temperature_clamp(softmax, device):
    """Exercise inactive/active temperature clamping near the actual FP32 boundary."""
    import torch

    reference = copy.deepcopy(softmax)
    reference.zero_grad(set_to_none=True)
    threshold = torch.tensor(0.001, device=reference.log_temperature.device)
    center = threshold.log()
    lower = torch.nextafter(center, torch.full_like(center, -torch.inf))
    upper = torch.nextafter(center, torch.full_like(center, torch.inf))
    levels = torch.stack((center - 1.0, lower, center, upper, center + 1.0, torch.zeros_like(center)))
    with torch.no_grad():
        reference.log_temperature.copy_(levels[torch.arange(32, device=levels.device) % len(levels)])
    features = (torch.randn(2, 32, 4, 4, device=levels.device) * 0.002).requires_grad_()
    points = reference(features)
    upstream = torch.randn_like(points)
    (points * upstream).sum().backward()
    values = warp_array(features, device)
    temperature = warp_array(reference.log_temperature, device)
    pos_x = warp_array(reference.pos_x.flatten(), device)
    pos_y = warp_array(reference.pos_y.flatten(), device)
    probabilities = wp.empty((2, 32, 16), dtype=wp.float32, device=device)
    output = wp.empty((2, 64), dtype=wp.float32, device=device)
    gradient = wp.empty(features.shape, dtype=wp.float32, device=device)
    grad_temperature = wp.zeros(32, dtype=wp.float32, device=device)
    wp.launch(
        spatial_forward, dim=(2, 32), inputs=[values, temperature, pos_x, pos_y, probabilities, output], device=device
    )
    wp.launch(
        spatial_backward,
        dim=(2, 32),
        inputs=[
            values,
            temperature,
            pos_x,
            pos_y,
            probabilities,
            output,
            warp_array(upstream, device),
            gradient,
            grad_temperature,
        ],
        device=device,
    )
    unclamped = reference.log_temperature.detach().exp()
    report = {
        "keypoints": check_close("clamped_keypoints", output, points),
        "feature_gradient": check_close("clamped_feature_gradient", gradient, features.grad, atol=2e-4, rtol=3e-4),
        "temperature_gradient": check_close(
            "clamped_temperature_gradient", grad_temperature, reference.log_temperature.grad
        ),
        "tested_log_temperatures": levels.detach().cpu().tolist(),
        "native_unclamped_temperatures": levels.exp().detach().cpu().tolist(),
        "note": "FP32 exp(log_temperature) may skip exact temperature equality; adjacent log values are included.",
    }
    inactive = (unclamped < threshold).cpu().numpy()
    assert inactive.any() and (~inactive).any()
    assert np.count_nonzero(grad_temperature.numpy()[inactive]) == 0
    assert np.linalg.norm(grad_temperature.numpy()[~inactive]) > 0
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu or a coordinated cuda:N device")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.batch <= 256:
        parser.error("batch must be between 1 and the native N64 minibatch size 256")

    import torch
    from rsl_rl.utils import resolve_class
    from tensordict import TensorDict

    from isaaclab.utils import to_dict

    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

    from isaaclab_tasks.core.lift.config.kuka_allegro.agents.rsl_rl_ppo_cfg import (
        KukaAllegroSingleCameraPPORunnerCfg,
    )

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    if args.device != "cpu":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    cfg = to_dict(handle_deprecated_rsl_rl_cfg(KukaAllegroSingleCameraPPORunnerCfg(), version("rsl-rl-lib")))
    observations = TensorDict(
        {
            "policy": torch.zeros(args.batch, 170),
            "proprio": torch.zeros(args.batch, 465),
            "base_image": torch.randint(0, 256, (args.batch, 3, 64, 64), dtype=torch.uint8),
        },
        batch_size=[args.batch],
    )
    actor_class, kwargs = resolve_class(cfg["actor"])
    actor = actor_class(observations[:1], cfg["obs_groups"], "actor", 23, **kwargs).to(args.device)
    cnn = actor.cnns["base_image"]
    softmax = actor.softmaxes["base_image"]
    source_paths = [
        Path(__file__),
        Path(inspect.getfile(actor_class)),
        Path(inspect.getfile(type(cnn))),
        Path(inspect.getfile(KukaAllegroSingleCameraPPORunnerCfg)),
    ]
    source_paths += [
        Path(wp.__file__).parent / name
        for name in ("native/builtin.h", "native/warp.cu", "_src/context.py", "_src/builtins.py", "_src/tape.py")
    ]
    source_before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    convolutions = [layer for layer in cnn if isinstance(layer, torch.nn.Conv2d)]
    assert [(c.in_channels, c.out_channels, c.kernel_size, c.stride) for c in convolutions] == [
        (3, 16, (8, 8), (4, 4)),
        (16, 32, (4, 4), (2, 2)),
        (32, 32, (3, 3), (1, 1)),
    ]
    assert softmax.log_temperature.shape == (32,)
    with torch.no_grad():
        softmax.log_temperature.copy_(torch.linspace(-0.7, 0.7, 32, device=args.device))
    raw = observations["base_image"].to(args.device)
    image = actor.image_normalizers["base_image"](raw).detach().requires_grad_()
    features = cnn(image)
    features.retain_grad()
    keypoints = softmax(features)
    upstream = torch.randn_like(keypoints)
    (keypoints * upstream).sum().backward()

    wp.init()
    # Module-local enable_backward overrides Isaac Lab's simulation-wide switch.
    wp.set_module_options({"enable_backward": True, "fast_math": False})
    device = wp.get_device(args.device)
    wraw = warp_array(raw, device)
    wimage = wp.empty(raw.shape, dtype=wp.float32, device=device, requires_grad=True)
    wp.launch(normalize_rgb, dim=raw.shape, inputs=[wraw, wimage], device=device)
    buffers = [wimage]
    preactivations = []
    weights, biases = [], []
    for layer, side in zip(convolutions, (15, 6, 4), strict=True):
        weights.append(warp_array(layer.weight, device, requires_grad=True))
        biases.append(warp_array(layer.bias, device, requires_grad=True))
        buffers.append(
            wp.empty((args.batch, layer.out_channels, side, side), dtype=wp.float32, device=device, requires_grad=True)
        )
        preactivations.append(wp.empty(buffers[-1].shape, dtype=wp.float32, device=device, requires_grad=True))
    tape = wp.Tape()
    with tape:
        for i, stride in enumerate((4, 2, 1)):
            # Warp's reverse replay does not retain the accumulator of a runtime
            # reduction loop for a following nonlinear expression. Keep the
            # preactivation explicitly so Tape differentiates ELU at its real input.
            wp.launch(
                convolution,
                dim=preactivations[i].shape,
                inputs=[buffers[i], weights[i], biases[i], preactivations[i], stride],
                device=device,
            )
            wp.launch(elu, dim=buffers[i + 1].shape, inputs=[preactivations[i], buffers[i + 1]], device=device)
    temperature = warp_array(softmax.log_temperature, device)
    pos_x = warp_array(softmax.pos_x.flatten(), device)
    pos_y = warp_array(softmax.pos_y.flatten(), device)
    probabilities = wp.empty((args.batch, 32, 16), dtype=wp.float32, device=device)
    wkeypoints = wp.empty((args.batch, 64), dtype=wp.float32, device=device)
    wupstream = warp_array(upstream, device)
    grad_features = wp.empty(buffers[-1].shape, dtype=wp.float32, device=device)
    grad_temperature = wp.zeros(32, dtype=wp.float32, device=device)
    wp.launch(
        spatial_forward,
        dim=(args.batch, 32),
        inputs=[buffers[-1], temperature, pos_x, pos_y, probabilities, wkeypoints],
        device=device,
    )
    wp.launch(
        spatial_backward,
        dim=(args.batch, 32),
        inputs=[
            buffers[-1],
            temperature,
            pos_x,
            pos_y,
            probabilities,
            wkeypoints,
            wupstream,
            grad_features,
            grad_temperature,
        ],
        device=device,
    )
    tape.backward(grads={buffers[-1]: grad_features})
    metrics = {
        "normalized_rgb": check_close("normalized_rgb", wimage, image, atol=1e-7, rtol=1e-6),
        "cnn_features": check_close("cnn_features", buffers[-1], features),
        "keypoints": check_close("keypoints", wkeypoints, keypoints),
        "feature_gradient": check_close("feature_gradient", grad_features, features.grad),
        "normalized_rgb_gradient": check_close("normalized_rgb_gradient", wimage.grad, image.grad),
        "log_temperature_gradient": check_close(
            "log_temperature_gradient", grad_temperature, softmax.log_temperature.grad
        ),
    }
    for i, layer in enumerate(convolutions):
        metrics[f"conv{i}_weight_gradient"] = check_close(
            f"conv{i}_weight_gradient", weights[i].grad, layer.weight.grad
        )
        metrics[f"conv{i}_bias_gradient"] = check_close(f"conv{i}_bias_gradient", biases[i].grad, layer.bias.grad)
    assert all(result["reference_l2"] > 0 for result in metrics.values())
    clamp_metrics = check_temperature_clamp(softmax, device)
    source_after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    assert source_before == source_after, "Sources changed during the experiment"
    report = {
        "status": "PASS",
        "device": str(device),
        "batch": args.batch,
        "seed": args.seed,
        "versions": {name: version(name) for name in ("warp-lang", "torch", "rsl-rl-lib")},
        "native_class": f"{actor_class.__module__}.{actor_class.__qualname__}",
        "candidate": "Pure Warp scalar convolution/ELU forward + Tape adjoints + explicit spatial-softmax adjoint",
        "scope": "Native RGB64 CNN/keypoint VJP: three convolution weights/biases and learned temperature; no PPO",
        "residency": "Global inputs, weights, intermediate activations and gradients; no persistent-state claim",
        "performance_claim": False,
        "metrics": metrics,
        "temperature_clamp": clamp_metrics,
        "source_sha256": source_after,
        "sources_unchanged": source_before == source_after,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(args.output),
                "max_absolute_error": max(item["max_abs"] for item in metrics.values()),
            }
        )
    )


if __name__ == "__main__":
    main()

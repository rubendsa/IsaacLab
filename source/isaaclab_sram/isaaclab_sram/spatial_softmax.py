# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fused spatial keypoints for the native Kuka visual policy.

Each feature map's temperature scaling, softmax and coordinate expectations live
in registers. Backward recomputes the probabilities instead of saving an image
of probabilities in device memory. CNN inputs/outputs and temperature-gradient
partials still cross global memory; this is not an all-SRAM visual learner.
"""

from __future__ import annotations

from types import MethodType

import torch
import triton
import triton.language as tl


@triton.jit
def _keypoints_forward(
    X, T, PX, PY, Y, N: tl.constexpr, C: tl.constexpr, S: tl.constexpr, R: tl.constexpr, P: tl.constexpr
):
    row = tl.program_id(0) * R + tl.arange(0, R)
    pixel = tl.arange(0, P)
    temperature = tl.maximum(tl.exp(tl.load(T + row % C)), 1.0e-3)
    z = (
        tl.load(X + row[:, None] * S + pixel[None, :], (row[:, None] < N) & (pixel[None, :] < S), other=0)
        / temperature[:, None]
    )
    z = tl.where(pixel[None, :] < S, z, -float("inf"))
    w = tl.exp(z - tl.max(z, 1)[:, None])
    w = w / tl.sum(w, 1)[:, None]
    px = tl.load(PX + pixel, pixel < S, other=0)
    py = tl.load(PY + pixel, pixel < S, other=0)
    out = row // C * (2 * C) + row % C
    tl.store(Y + out, tl.sum(w * px[None, :], 1), row < N)
    tl.store(Y + out + C, tl.sum(w * py[None, :], 1), row < N)


@triton.jit
def _keypoints_backward(
    X, T, PX, PY, DY, DX, DT, N: tl.constexpr, C: tl.constexpr, S: tl.constexpr, R: tl.constexpr, P: tl.constexpr
):
    row = tl.program_id(0) * R + tl.arange(0, R)
    pixel = tl.arange(0, P)
    raw_temperature = tl.exp(tl.load(T + row % C))
    temperature = tl.maximum(raw_temperature, 1.0e-3)
    x = tl.load(X + row[:, None] * S + pixel[None, :], (row[:, None] < N) & (pixel[None, :] < S), other=0)
    logits = x / temperature[:, None]
    z = tl.where(pixel[None, :] < S, logits, -float("inf"))
    w = tl.exp(z - tl.max(z, 1)[:, None])
    w = w / tl.sum(w, 1)[:, None]
    px = tl.load(PX + pixel, pixel < S, other=0)
    py = tl.load(PY + pixel, pixel < S, other=0)
    ex = tl.sum(w * px[None, :], 1)
    ey = tl.sum(w * py[None, :], 1)
    out = row // C * (2 * C) + row % C
    gx = tl.load(DY + out, row < N, other=0)
    gy = tl.load(DY + out + C, row < N, other=0)
    dz = w * (gx[:, None] * (px[None, :] - ex[:, None]) + gy[:, None] * (py[None, :] - ey[:, None]))
    tl.store(
        DX + row[:, None] * S + pixel[None, :], dz / temperature[:, None], (row[:, None] < N) & (pixel[None, :] < S)
    )
    dt = -tl.sum(dz * logits, 1)
    tl.store(DT + row, tl.where(raw_temperature >= 1.0e-3, dt, 0.0), row < N)


class _Keypoints(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, log_temperature, pos_x, pos_y):
        if features.ndim != 4 or features.dtype != torch.float32 or not features.is_cuda:
            raise ValueError("Fused spatial softmax requires CUDA float32 [N,C,H,W] features")
        batch, channels, height, width = features.shape
        if batch < 1 or channels < 1 or height * width < 1:
            raise ValueError("Fused spatial softmax requires nonempty feature maps")
        if log_temperature.shape != (channels,) or pos_x.numel() != height * width or pos_y.numel() != height * width:
            raise ValueError("Temperature and position shapes must match the feature map")
        tensors = (log_temperature, pos_x, pos_y)
        if any(t.device != features.device or t.dtype != features.dtype for t in tensors):
            raise ValueError("Features, temperatures and coordinates must share device and float32 dtype")
        features, log_temperature, pos_x, pos_y = (t.contiguous() for t in (features, *tensors))
        pixels = height * width
        rows = max(1, min(16, 256 // triton.next_power_of_2(pixels)))
        output = torch.empty((batch, 2 * channels), dtype=features.dtype, device=features.device)
        with torch.cuda.device(features.device):
            _keypoints_forward[(triton.cdiv(batch * channels, rows),)](
                features,
                log_temperature,
                pos_x,
                pos_y,
                output,
                batch * channels,
                channels,
                pixels,
                rows,
                triton.next_power_of_2(pixels),
                num_warps=4,
            )
        ctx.save_for_backward(features, log_temperature, pos_x, pos_y)
        ctx.rows = rows
        return output

    @staticmethod
    def backward(ctx, grad_output):
        features, log_temperature, pos_x, pos_y = ctx.saved_tensors
        batch, channels, height, width = features.shape
        pixels = height * width
        grad_features = torch.empty_like(features)
        temperature_partials = torch.empty((batch, channels), device=features.device, dtype=torch.float32)
        with torch.cuda.device(features.device):
            _keypoints_backward[(triton.cdiv(batch * channels, ctx.rows),)](
                features,
                log_temperature,
                pos_x,
                pos_y,
                grad_output.contiguous(),
                grad_features,
                temperature_partials,
                batch * channels,
                channels,
                pixels,
                ctx.rows,
                triton.next_power_of_2(pixels),
                num_warps=4,
            )
        return grad_features, temperature_partials.sum(dim=0), None, None


def fused_spatial_softmax(
    features: torch.Tensor, log_temperature: torch.Tensor, pos_x: torch.Tensor, pos_y: torch.Tensor
) -> torch.Tensor:
    """Return native x-then-y keypoints with gradients for features and log temperatures."""
    return _Keypoints.apply(features, log_temperature, pos_x, pos_y)


def _fused_forward(self, features):
    return fused_spatial_softmax(features, self.log_temperature, self.pos_x, self.pos_y)


def install_fused_spatial_softmax(actor: torch.nn.Module) -> int:
    """Install fused forwards while retaining native parameters and checkpoint keys."""
    from isaaclab_tasks.core.lift.config.kuka_allegro.agents.models import SpatialSoftmax

    count = 0
    for module in actor.modules():
        if isinstance(module, SpatialSoftmax):
            module.forward = MethodType(_fused_forward, module)
            count += 1
    return count

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the megakernel's device functions, each checked against its Isaac Lab / torch twin.

* ``wrap_to_pi``        vs ``isaaclab.utils.math.wrap_to_pi``
* ``elu`` / derivative  vs ``torch.nn.functional.elu`` + autograd
* ``reduce8``           vs numpy sums (butterfly reduce-scatter lane assignment)
* ``feistel_perm``      is a bijection on [0, n) for many n
* ``normal01``          Philox + Box-Muller moments
* ``cartpole_substep``  vs a float64 closed-form MuJoCo-Warp ``implicitfast`` reference
"""

from __future__ import annotations

import ctypes
import math
import os
import pathlib
import sys

import numpy as np
import pytest
import torch

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")

KDIR = pathlib.Path(os.environ.get("ISAACLAB_SRAM_KERNEL_DIR", PKG / "isaaclab_sram" / "kernels"))


def _module(src: str, defines: dict | None = None):
    from isaaclab_sram import cuda_rt as rt

    rt.ensure_context(0)
    headers = {n: (KDIR / n).read_text() for n in ("sram_device.cuh", "cartpole_physics.cuh")}
    return rt, rt.compile_module(src, headers=headers, defines=defines or {}, name="test_device.cu")


TEST_SRC = r"""
#include "sram_device.cuh"
#include "cartpole_physics.cuh"
extern "C" __global__ void k_wrap(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = wrap_to_pi(x[i]); }
extern "C" __global__ void k_elu(const float* x, float* y, float* dy, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float a = elu(x[i]); y[i] = a; dy[i] = delu_from_act(a); } }
extern "C" __global__ void k_reduce8(const float* v, float* out) {   // one warp; v[lane*8 + j]
    int lane = threadIdx.x & 31; float r[8];
    for (int j = 0; j < 8; ++j) r[j] = v[lane * 8 + j];
    out[lane] = reduce8(r, lane); }
extern "C" __global__ void k_perm(u32* out, u32 n, u32 half, uint4 key) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = feistel_perm(i, n, half, key); }
extern "C" __global__ void k_normal(float* out, int n, u32 s0, u32 s1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { uint4 r = philox(make_uint4((u32)i, 7u, 3u, 0x41435431u), s0, s1); out[i] = normal01(r.x, r.y); } }
extern "C" __global__ void k_substep(float* q, const float* f, int n, int steps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float s = q[4*i], th = q[4*i+1], sd = q[4*i+2], thd = q[4*i+3];
        for (int k = 0; k < steps; ++k) cartpole_substep(s, th, sd, thd, f[i]);
        q[4*i] = s; q[4*i+1] = th; q[4*i+2] = sd; q[4*i+3] = thd; } }
"""


def _phys_defines():
    from isaaclab_sram import megakernel as mk
    from isaaclab_sram.cartpole import CartpoleTask, PPOConfig

    task, ppo = CartpoleTask(), PPOConfig()
    d = mk.kernel_defines(task, ppo, mk.LaunchPlan(4096, 188, 22, True, 0, 16384))
    return {k: v for k, v in d.items() if k.startswith("P_")}, task


def test_wrap_to_pi_matches_isaaclab():
    from isaaclab.utils.math import wrap_to_pi as wrap_ref

    rt, mod = _module(TEST_SRC, _phys_defines()[0])
    x = (
        torch.cat(
            [
                torch.linspace(-40, 40, 200001),
                torch.tensor([0.0, math.pi, -math.pi, 2 * math.pi, -2 * math.pi, 3 * math.pi]),
            ]
        )
        .float()
        .cuda()
    )
    y = torch.empty_like(x)
    mod.kernel("k_wrap").launch((x.numel() + 255) // 256, 256, [x, y, rt.i32(x.numel())])
    torch.cuda.synchronize()
    ref = wrap_ref(x)
    # identical up to float32 rounding of the remainder (|err| <= a few ulp of 2*pi)
    assert (y - ref).abs().max().item() < 2e-6


def test_elu_and_derivative_match_torch():
    rt, mod = _module(TEST_SRC, _phys_defines()[0])
    x = torch.linspace(-12, 12, 100001).cuda()
    y, dy = torch.empty_like(x), torch.empty_like(x)
    mod.kernel("k_elu").launch((x.numel() + 255) // 256, 256, [x, y, dy, rt.i32(x.numel())])
    torch.cuda.synchronize()
    xr = x.clone().requires_grad_(True)
    yr = torch.nn.functional.elu(xr)
    yr.sum().backward()
    assert (y - yr.detach()).abs().max().item() < 1e-6
    mask = x.abs() > 1e-6  # derivative at exactly 0 is a convention
    assert (dy - xr.grad)[mask].abs().max().item() < 1e-6


def test_reduce8_lane_assignment():
    rt, mod = _module(TEST_SRC, _phys_defines()[0])
    v = torch.randn(32, 8, dtype=torch.float32).cuda()
    out = torch.empty(32, device="cuda")
    mod.kernel("k_reduce8").launch(1, 32, [v.reshape(-1), out])
    torch.cuda.synchronize()
    sums = v.sum(0).cpu()
    expect = torch.stack([sums[(lane >> 2) & 7] for lane in range(32)])
    assert torch.allclose(out.cpu(), expect, atol=1e-5)


class _U4(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint32), ("y", ctypes.c_uint32), ("z", ctypes.c_uint32), ("w", ctypes.c_uint32)]


@pytest.mark.parametrize("n", [1, 2, 3, 5, 84, 88, 336, 352, 1000, 5576, 22320, 65536, 100003])
def test_feistel_perm_is_a_bijection(n):
    rt, mod = _module(TEST_SRC, _phys_defines()[0])
    half = 1
    while (1 << (2 * half)) < n:
        half += 1
    rng = np.random.default_rng(n)
    for _ in range(3):
        key = _U4(*[int(k) for k in rng.integers(0, 2**32, 4, dtype=np.uint64)])
        out = torch.empty(n, dtype=torch.int32, device="cuda")
        mod.kernel("k_perm").launch((n + 255) // 256, 256, [out, rt.u32(n), rt.u32(half), key])
        torch.cuda.synchronize()
        assert torch.equal(torch.sort(out.cpu().long()).values, torch.arange(n))


def test_normal01_moments():
    rt, mod = _module(TEST_SRC, _phys_defines()[0])
    n = 1 << 21
    out = torch.empty(n, device="cuda")
    mod.kernel("k_normal").launch(n // 256, 256, [out, rt.i32(n), rt.u32(123), rt.u32(456)])
    torch.cuda.synchronize()
    x = out.double().cpu()
    assert abs(x.mean().item()) < 3e-3 and abs(x.std().item() - 1.0) < 3e-3
    assert abs(((x**4).mean() - 3.0).item()) < 3e-2  # kurtosis of a Gaussian


def test_cartpole_substep_matches_float64_reference():
    defines, task = _phys_defines()
    rt, mod = _module(TEST_SRC, defines)
    ph = task.physics
    dt = float(np.float32(task.physics_dt))
    n, steps = 65536, 2
    rng = np.random.default_rng(0)
    q0 = np.stack(
        [rng.uniform(-3, 3, n), rng.uniform(-math.pi, math.pi, n), rng.uniform(-5, 5, n), rng.uniform(-8, 8, n)], 1
    ).astype(np.float32)
    f = rng.uniform(-500, 500, n).astype(np.float32)
    q = torch.tensor(q0).cuda().reshape(-1).contiguous()
    mod.kernel("k_substep").launch(n // 256, 256, [q, torch.tensor(f).cuda(), rt.i32(n), rt.i32(steps)])
    torch.cuda.synchronize()
    s, th, sd, thd = (q0[:, i].astype(np.float64) for i in range(4))
    for _ in range(steps):
        c, sn = np.cos(th), np.sin(th)
        M00, M01 = ph.cart_mass + ph.pole_mass, -ph.pole_mass * ph.pole_com * c
        M11 = ph.pole_inertia + ph.pole_mass * ph.pole_com**2
        b0, b1 = ph.pole_mass * ph.pole_com * sn * thd**2, -ph.pole_mass * ph.gravity * ph.pole_com * sn
        fa = np.clip(-ph.cart_damping * sd, -ph.actuator_force_limit, ph.actuator_force_limit)
        q0_, q1_ = -b0 + fa + f, -b1
        Mh = M00 + dt * ph.cart_damping
        det = Mh * M11 - M01**2
        a0, a1 = (M11 * q0_ - M01 * q1_) / det, (Mh * q1_ - M01 * q0_) / det
        sd, thd = sd + dt * a0, thd + dt * a1
        s, th = s + dt * sd, th + dt * thd
    ref = np.stack([s, th, sd, thd], 1)
    got = q.reshape(n, 4).double().cpu().numpy()
    assert np.abs(got - ref).max() < 5e-5

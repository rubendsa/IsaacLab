# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Protect native visual coordinates and both gradients against an independent torch model.

Unlike the Cartpole tests, this exercises a learned visual temperature, its clamp,
non-square feature coordinates, masked tail rows and strided feature input.
"""

import copy
import sys
from pathlib import Path

import pytest
import torch

from isaaclab.test.utils import DeviceScope, test_devices

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_native_spatial_softmax_forward_and_gradients(device):
    from isaaclab_sram.spatial_softmax import fused_spatial_softmax

    from isaaclab_tasks.core.lift.config.kuka_allegro.agents.models import SpatialSoftmax

    torch.manual_seed(17)
    native = SpatialSoftmax(3, 5, 3).to(device)
    with torch.no_grad():
        native.log_temperature.copy_(torch.tensor([-8.0, -0.7, 0.5], device=device))
    features = (torch.randn(7, 3, 5, 6, device=device) * 0.01)[..., ::2]
    fused_features = features.detach().requires_grad_()
    temperature = native.log_temperature.detach().clone().requires_grad_()
    # The temperature floor magnifies float32 rounding by 1000. A double
    # precision native reference distinguishes kernel error from reference
    # cancellation when the softmax is almost one-hot.
    reference = copy.deepcopy(native).double()
    reference_features = features.detach().double().requires_grad_()
    expected = reference(reference_features)
    actual = fused_spatial_softmax(fused_features, temperature, native.pos_x, native.pos_y)
    gradient = torch.randn_like(actual)
    expected.backward(gradient.double())
    actual.backward(gradient)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(fused_features.grad.double(), reference_features.grad, rtol=2e-4, atol=1e-4)
    torch.testing.assert_close(temperature.grad.double(), reference.log_temperature.grad, rtol=2e-4, atol=2e-5)
    assert temperature.grad[0].item() == 0.0

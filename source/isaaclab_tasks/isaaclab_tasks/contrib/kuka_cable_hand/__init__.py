# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kuka iiwa 7 with the cable-driven hand of Newton's ``exp/cable-driven-hand-series`` branch: lift a cube.

A Newton-native vectorized task (Newton VBD with massless routed tendons) modeled on
``Isaac-Lift-KukaAllegro``. It needs the Newton build of that branch; see ``README.md``.
Train with ``python -m isaaclab_tasks.contrib.kuka_cable_hand.train``.
"""

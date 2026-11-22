// Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// Closed-form Isaac-Cartpole-Direct physics, integrated exactly like Newton's MuJoCo-Warp
// `implicitfast` integrator with Isaac Lab 3.0's cartpole model.  Constants (P_*) are -D defines.
#pragma once

// One MuJoCo-Warp `implicitfast` substep of the Isaac Lab cart-pole (float32).
//   M = [[mc+mp, -mp l cos th], [-mp l cos th, Ip + mp l^2]]
//   qfrc_bias = [mp l sin th thd^2, -mp g l sin th]
//   actuator (velocity servo, kv, target 0) = clamp(-kv sd, -400, 400); feed-forward effort f_app
//   implicitfast: (M + dt diag(kv, 0)) qacc = qfrc_smooth ; qd += dt qacc ; q += dt qd
__device__ __forceinline__ void cartpole_substep(float& s, float& th, float& sd, float& thd, float f_app) {
    float sth, cth;
    sincosf(th, &sth, &cth);
    const float M00 = P_MC + P_MP;
    const float M01 = -(P_MP * P_L) * cth;
    const float M11 = P_IP + P_MP * P_L * P_L;
    const float bias0 = (P_MP * P_L) * sth * thd * thd;
    const float bias1 = -(P_MP * P_G * P_L) * sth;
    float f_act = -P_KV * sd;
    f_act = fminf(fmaxf(f_act, -P_ACT_FRC), P_ACT_FRC);
    const float qf0 = -bias0 + f_act + f_app;
    const float qf1 = -bias1;
    const float Mh00 = M00 + P_DT * P_KV;
    const float det = Mh00 * M11 - M01 * M01;
    const float qacc0 = (M11 * qf0 - M01 * qf1) / det;
    const float qacc1 = (Mh00 * qf1 - M01 * qf0) / det;
    sd = sd + qacc0 * P_DT;
    thd = thd + qacc1 * P_DT;
    s = s + P_DT * sd;
    th = th + P_DT * thd;
}

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Numerical verification of the SRAM-resident PPO megakernel against independent references.

Runs one training iteration of the megakernel in its debug build (which dumps the SRAM-resident
rollout buffer, the post-physics states, the all-reduced gradient of the first mini-batch and the
parameters around the first Adam step) and re-derives every quantity on the host:

1. physics      - every transition re-integrated with a float64 closed-form MuJoCo-Warp
                  ``implicitfast`` reference (independent of the kernel's float32 code),
2. MDP          - rewards, terminations, time-out bootstrapping and resets,
3. networks     - mu / value / log-prob recomputed with RSL-RL's own ``MLPModel`` classes,
4. GAE          - advantages / returns recomputed with RSL-RL's ``compute_returns`` recursion and
                  global advantage normalisation,
5. gradients    - the first mini-batch loss evaluated with RSL-RL's PPO loss code and
                  differentiated by torch autograd, compared with the kernel's all-reduced
                  gradient,
6. optimiser    - KL-adaptive LR, separate actor/critic ``clip_grad_norm_`` and ``torch.optim.Adam``
                  compared with the kernel's first parameter update.

Usage::

    ./isaaclab.sh -p scripts/sram/verify_megakernel.py [--num_envs 4096] [--seed 3]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "source", "isaaclab_sram"))

from isaaclab_sram import megakernel as mk  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--num_envs", type=int, default=4096)
ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--storage", default="auto", choices=["auto", "smem", "global"])
ap.add_argument("--warmup_iters", type=int, default=0, help="train this many iterations before the checked one")
ap.add_argument("--out", default=None)
args = ap.parse_args()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

trainer = mk.SramPPOCartpole(num_envs=args.num_envs, seed=args.seed, storage=args.storage, debug=True)
plan, task, ppo = trainer.plan, trainer.task, trainer.ppo
if args.warmup_iters:
    trainer.train(args.warmup_iters)
# the debug dumps are written for the first iteration *of a launch* (it == 0)
P0 = trainer.params.detach().clone().cpu()
lr0 = float(trainer.scal[0].item())
adam_t0 = int(trainer.scal[1].item())
m0 = trainer.adam_m.detach().clone().cpu()
v0 = trainer.adam_v.detach().clone().cpu()
env_before = trainer.env_state.detach().clone().cpu().double().numpy()
eplen_before = trainer.ep_len.detach().clone().cpu().numpy()
trainer.dbg.zero_()
trainer.train(1)
torch.cuda.synchronize()
env_after = trainer.env_state.detach().cpu().double().numpy()

T, G, EM = ppo.num_steps_per_env, plan.num_ctas, plan.envs_per_cta_max
FS = T * EM
dbg = trainer.dbg.detach().cpu().double().numpy()
off_post = 0
off_store = G * 4 * FS
off_grad = off_store + G * (mk.NFIELD + 1) * FS
off_p1 = off_grad + mk.ACC_F_STRIDE
off_p0 = off_p1 + mk.N_PARAMS_PAD
post = dbg[off_post:off_store].reshape(G, 4, FS)
store = dbg[off_store:off_grad].reshape(G, mk.NFIELD + 1, FS)
grad_k = dbg[off_grad : off_grad + mk.ACC_F_STRIDE]
p1_k = dbg[off_p1 : off_p1 + mk.N_PARAMS_PAD]
p0_k = dbg[off_p0 : off_p0 + mk.N_PARAMS_PAD]

report: dict = {
    "num_envs": args.num_envs,
    "num_ctas": G,
    "envs_per_cta_max": EM,
    "seed": args.seed,
    "storage": "smem" if plan.store_in_smem else "global",
}
fails = []


def check(name, err, tol):
    ok = bool(err <= tol)
    report[name] = {"max_abs_err": float(err), "tol": tol, "ok": ok}
    print(f"[{'OK' if ok else 'FAIL'}] {name}: max err {err:.3e} (tol {tol:.0e})")
    if not ok:
        fails.append(name)


# ------------------------------------------------------------------------------------------------
# float64 physics reference (written independently from the kernel)
# ------------------------------------------------------------------------------------------------
ph = task.physics
DT = float(np.float32(task.physics_dt))


def substep64(s, th, sd, thd, f):
    c, sn = np.cos(th), np.sin(th)
    M00 = ph.cart_mass + ph.pole_mass
    M01 = -ph.pole_mass * ph.pole_com * c
    M11 = ph.pole_inertia + ph.pole_mass * ph.pole_com**2
    b0 = ph.pole_mass * ph.pole_com * sn * thd**2
    b1 = -ph.pole_mass * ph.gravity * ph.pole_com * sn
    fa = np.clip(-ph.cart_damping * sd, -ph.actuator_force_limit, ph.actuator_force_limit)
    q0, q1 = -b0 + fa + f, -b1
    Mh = M00 + DT * ph.cart_damping
    det = Mh * M11 - M01**2
    a0 = (M11 * q0 - M01 * q1) / det
    a1 = (Mh * q1 - M01 * q0) / det
    sd = sd + DT * a0
    thd = thd + DT * a1
    return s + DT * sd, th + DT * thd, sd, thd


def wrap_to_pi(a):
    w = np.mod(a + np.pi, 2 * np.pi)
    return np.where((w == 0) & (a > 0), np.pi, w - np.pi)


# ------------------------------------------------------------------------------------------------
# networks with RSL-RL's own classes
# ------------------------------------------------------------------------------------------------
actor, critic = mk.rsl_rl_models(ppo, 0)
mk.unpack_params(P0, actor, critic)
check("init_params_roundtrip", float(np.abs(p0_k[: mk.N_PARAMS] - P0.numpy()[: mk.N_PARAMS]).max()), 0.0)
sigma0 = float(np.clip(P0[mk.OFF_STD].item(), 1e-6, 1e6))

phys_err, rew_err, obs_link_err, reset_range_viol, mu_err, v_err, logp_err = 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0
done_mismatches, episode_length_mismatches = 0, 0
gae_err, ret_err = 0.0, 0.0
all_adv_raw = []
per_cta = []
f_obs, f_act, f_logp, f_val, f_mu, f_rew, f_done, f_adv, f_ret = 0, 4, 5, 6, 7, 8, 9, 10, 11
gamma, lam = ppo.gamma, ppo.lam
for c in range(G):
    e0, e1 = (c * plan.num_envs) // G, ((c + 1) * plan.num_envs) // G
    E = e1 - e0
    n = E * T
    st = store[c, :, :n]
    obs = st[0:4].reshape(4, T, E).transpose(1, 2, 0)  # [T, E, 4]
    act = st[f_act].reshape(T, E)
    pst = post[c, :, :n].reshape(4, T, E).transpose(1, 2, 0)
    # physics: obs[t] + act[t] -> post[t]
    s, th, sd, thd = (obs[..., i].copy() for i in range(4))
    f = np.float32(task.action_scale) * act.astype(np.float32)
    for _ in range(task.decimation):
        s, th, sd, thd = substep64(s, th, sd, thd, f.astype(np.float64))
    ref = np.stack([s, th, sd, thd], -1)
    phys_err = max(phys_err, float(np.abs(ref - pst).max()))
    # transitions link: obs[t+1] == post[t] unless done
    done = st[f_done].reshape(T, E) > 0.5
    link = np.abs(obs[1:] - pst[:-1]).max(-1)
    obs_link_err = max(obs_link_err, float(np.where(done[:-1], 0.0, link).max()) if T > 1 else 0.0)
    # resets land inside the configured ranges
    rs = obs[1:][done[:-1]]
    if rs.size:
        lo = np.array(
            [
                task.initial_cart_position_range[0],
                task.initial_pole_angle_range[0],
                task.initial_cart_velocity_range[0],
                task.initial_pole_velocity_range[0],
            ]
        )
        hi = np.array(
            [
                task.initial_cart_position_range[1],
                task.initial_pole_angle_range[1],
                task.initial_cart_velocity_range[1],
                task.initial_pole_velocity_range[1],
            ]
        )
        reset_range_viol += int(((rs < lo - 1e-6) | (rs > hi + 1e-6)).sum())
    # reward / termination / time-out bootstrap
    term = np.abs(pst[..., 0]) > task.max_cart_pos
    # Reconstruct timeouts independently from the pre-launch episode counters. Deriving them
    # from the kernel's done bits would allow a missing or early reset to validate itself.
    lengths = eplen_before[e0:e1].copy()
    tout = np.zeros_like(term)
    for t in range(T):
        lengths += 1
        tout[t] = lengths >= task.max_episode_length
        expected_done = term[t] | tout[t]
        done_mismatches += int(np.count_nonzero(done[t] != expected_done))
        lengths[expected_done] = 0
    episode_length_mismatches += int(np.count_nonzero(lengths != trainer.ep_len[e0:e1].cpu().numpy()))
    tf = term.astype(np.float64)
    r = (
        task.rew_scale_alive * (1 - tf)
        + task.rew_scale_terminated * tf
        + task.rew_scale_pole_pos * wrap_to_pi(pst[..., 1]) ** 2
        + task.rew_scale_cart_vel * np.abs(pst[..., 2])
        + task.rew_scale_pole_vel * np.abs(pst[..., 3])
    ) * task.step_dt
    val = st[f_val].reshape(T, E)
    r_boot = r + np.where(tout, gamma * val, 0.0)
    rr = st[f_rew].reshape(T, E)
    res = np.abs(rr - r_boot)
    rew_err = max(rew_err, float(res.max()))
    # networks
    with torch.no_grad():
        o = torch.tensor(obs.reshape(-1, 4), dtype=torch.float32)
        mu = actor.mlp(o).double().numpy().reshape(T, E)
        v = critic.mlp(o).double().numpy().reshape(T, E)
    mu_err = max(mu_err, float(np.abs(mu - st[f_mu].reshape(T, E)).max()))
    v_err = max(v_err, float(np.abs(v - val).max()))
    d = act - st[f_mu].reshape(T, E)
    lp = -(d**2) / (2 * sigma0**2) - math.log(sigma0) - 0.5 * math.log(2 * math.pi)
    logp_err = max(logp_err, float(np.abs(lp - st[f_logp].reshape(T, E)).max()))
    # GAE with RSL-RL's recursion (last value from the post-rollout environment state)
    with torch.no_grad():
        last_v = critic.mlp(torch.tensor(env_after[e0:e1], dtype=torch.float32)).double().numpy().reshape(E)
    adv = np.zeros(E)
    rets = np.zeros((T, E))
    for t in reversed(range(T)):
        nv = last_v if t == T - 1 else val[t + 1]
        nnt = 1.0 - done[t]
        delta = rr[t] + nnt * gamma * nv - val[t]
        adv = delta + nnt * gamma * lam * adv
        rets[t] = adv + val[t]
    ret_err = max(ret_err, float(np.abs(rets - st[f_ret].reshape(T, E)).max()))
    all_adv_raw.append((rets - val).reshape(-1))
    per_cta.append(dict(c=c, E=E, n=n, st=st, perm=store[c, mk.NFIELD, :n].astype(np.int64)))

check("physics_vs_float64_reference", phys_err, 2e-4)
check("transition_linkage", obs_link_err, 0.0)
report["reset_range_violations"] = reset_range_viol
if reset_range_viol:
    fails.append("reset_ranges")
check("reward_termination_timeout_bootstrap", rew_err, 1e-5)
check("termination_and_timeout_flags", done_mismatches, 0)
check("episode_counters_after_rollout", episode_length_mismatches, 0)
check("actor_mean_vs_rsl_rl_MLPModel", mu_err, 1e-5)
check("critic_value_vs_rsl_rl_MLPModel", v_err, 1e-5)
check("log_prob", logp_err, 1e-5)
check("gae_returns", ret_err, 1e-4)
adv_raw = np.concatenate(all_adv_raw)
mean, std = adv_raw.mean(), adv_raw.std(ddof=1)
adv_norm_err = 0.0
for pc in per_cta:
    T_E = pc["n"]
    st = pc["st"]
    raw = st[f_ret] - st[f_val]
    adv_norm_err = max(adv_norm_err, float(np.abs((raw - mean) / (std + 1e-8) - st[f_adv]).max()))
check("global_advantage_normalisation", adv_norm_err, 1e-4)
perm_ok = all(sorted(pc["perm"].tolist()) == list(range(pc["n"])) for pc in per_cta)
report["permutations_are_bijections"] = perm_ok
if not perm_ok:
    fails.append("permutation")

# ------------------------------------------------------------------------------------------------
# first mini-batch: RSL-RL loss + autograd vs the kernel's all-reduced gradient
# ------------------------------------------------------------------------------------------------
rows = []
for pc in per_cta:
    mbl = pc["n"] // ppo.num_mini_batches
    idx = pc["perm"][:mbl]
    rows.append(pc["st"][:, idx])
B = np.concatenate(rows, axis=1)  # [NFIELD, mb_global]
assert B.shape[1] == plan.mb_global
tt = lambda x: torch.tensor(x, dtype=torch.float32)  # noqa: E731
obs_b = tt(B[0:4].T)
act_b, oldlogp_b, oldmu_b = tt(B[f_act])[:, None], tt(B[f_logp]), tt(B[f_mu])[:, None]
adv_b, ret_b, val_b = tt(B[f_adv]), tt(B[f_ret])[:, None], tt(B[f_val])[:, None]
from tensordict import TensorDict  # noqa: E402

td = TensorDict({"policy": obs_b}, batch_size=[obs_b.shape[0]])
actor.zero_grad(set_to_none=True)
critic.zero_grad(set_to_none=True)
# --- verbatim RSL-RL 5.5.1 PPO.update() loss (rsl_rl/algorithms/ppo.py) ---
actor(td, stochastic_output=True)
actions_log_prob = actor.get_output_log_prob(act_b)
values = critic(td)
distribution_params = tuple(p for p in actor.output_distribution_params)
entropy = actor.output_entropy
with torch.no_grad():
    old_sigma = torch.full_like(oldmu_b, sigma0)
    kl = actor.get_kl_divergence((oldmu_b, old_sigma), distribution_params)
    kl_mean = torch.mean(kl)
ratio = torch.exp(actions_log_prob - torch.squeeze(oldlogp_b))
surrogate = -torch.squeeze(adv_b) * ratio
surrogate_clipped = -torch.squeeze(adv_b) * torch.clamp(ratio, 1.0 - ppo.clip_param, 1.0 + ppo.clip_param)
surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
value_clipped = val_b + (values - val_b).clamp(-ppo.clip_param, ppo.clip_param)
value_losses = (values - ret_b).pow(2)
value_losses_clipped = (value_clipped - ret_b).pow(2)
value_loss = torch.max(value_losses, value_losses_clipped).mean()
loss = surrogate_loss + ppo.value_loss_coef * value_loss - ppo.entropy_coef * entropy.mean()
loss.backward()
# ---
g_ref = torch.zeros(mk.N_PARAMS_PAD, dtype=torch.float64)
for n_i, model in enumerate((actor, critic)):
    off = n_i * mk.NET_P
    for m in (mm for mm in model.mlp if isinstance(mm, torch.nn.Linear)):
        k = m.weight.numel()
        g_ref[off : off + k] = m.weight.grad.reshape(-1).double()
        off += k
        k = m.bias.numel()
        g_ref[off : off + k] = m.bias.grad.reshape(-1).double()
        off += k
g_ref[mk.OFF_STD] = actor.distribution.std_param.grad.reshape(-1)[0].double()
g_ref_float32 = g_ref
# A monolithic CPU float32 bias-gradient reduction loses accuracy at large mini-batches.
# At 65,536 environments its error is 2.38e-6, while the kernel is within 1.39e-8 of
# float64 autograd. Use the stronger float64 oracle without widening the tolerance.
# Keep the original float32 networks/gradients for the torch Adam replay below.
actor64, critic64 = mk.rsl_rl_models(ppo, 0)
mk.unpack_params(P0, actor64, critic64)
actor64, critic64 = actor64.double(), critic64.double()
mean64, value64 = actor64.mlp(obs_b.double()), critic64.mlp(obs_b.double())
distribution64 = torch.distributions.Normal(mean64, actor64.distribution.std_param.clamp(1e-6, 1e6))
ratio64 = torch.exp(distribution64.log_prob(act_b.double()).sum(-1) - oldlogp_b.double())
surrogate64 = torch.maximum(
    -adv_b.double() * ratio64,
    -adv_b.double() * ratio64.clamp(1.0 - ppo.clip_param, 1.0 + ppo.clip_param),
).mean()
clipped64 = val_b.double() + (value64 - val_b.double()).clamp(-ppo.clip_param, ppo.clip_param)
value_loss64 = torch.maximum((value64 - ret_b.double()).square(), (clipped64 - ret_b.double()).square()).mean()
loss64 = surrogate64 + ppo.value_loss_coef * value_loss64 - ppo.entropy_coef * distribution64.entropy().sum(-1).mean()
loss64.backward()
g_ref = torch.zeros_like(g_ref_float32)
for n_i, model64 in enumerate((actor64, critic64)):
    off = n_i * mk.NET_P
    for layer in (m for m in model64.mlp if isinstance(m, torch.nn.Linear)):
        for parameter in (layer.weight, layer.bias):
            count = parameter.numel()
            g_ref[off : off + count] = parameter.grad.reshape(-1)
            off += count
g_ref[mk.OFF_STD] = actor64.distribution.std_param.grad.reshape(-1)[0]
report["first_minibatch_gradient_reference_dtype"] = "float64"
report["float32_reference_gradient_error_vs_float64"] = float((g_ref_float32 - g_ref).abs().max())
gk = torch.tensor(grad_k[: mk.N_PARAMS_PAD])
abs_err = (gk[: mk.N_PARAMS] - g_ref[: mk.N_PARAMS]).abs()
scale = g_ref[: mk.N_PARAMS].abs().max().item()
check(
    "first_minibatch_gradient_vs_autograd (abs, all 2499 params)", float(abs_err.max()), 2e-6 * max(1.0, scale) + 1e-7
)
rel = float((gk[: mk.N_PARAMS] - g_ref[: mk.N_PARAMS]).norm() / g_ref[: mk.N_PARAMS].norm())
report["first_minibatch_gradient_relative_l2_err"] = rel
print(f"       gradient relative L2 error {rel:.3e}; |g|max {scale:.3e}")
nmb = plan.mb_global
check("kl_mean", abs(grad_k[mk.N_PARAMS_PAD + 0] / nmb - kl_mean.item()), 1e-6)
check("surrogate_loss", abs(grad_k[mk.N_PARAMS_PAD + 1] / nmb - surrogate_loss.item()), 1e-6)
check("value_loss", abs(grad_k[mk.N_PARAMS_PAD + 2] / nmb - value_loss.item()), 1e-5 * max(1, value_loss.item()))


# ------------------------------------------------------------------------------------------------
# first optimiser step: adaptive LR + per-model clip_grad_norm_ + torch.optim.Adam
# ------------------------------------------------------------------------------------------------
def adapt(lr, kl_mean):
    if ppo.schedule == "adaptive" and ppo.desired_kl > 0:
        if kl_mean > ppo.desired_kl * 2.0:
            lr = max(1e-5, lr / 1.5)
        elif kl_mean < ppo.desired_kl / 2.0 and kl_mean > 0.0:
            lr = min(1e-2, lr * 1.5)
    return lr


lr = adapt(lr0, kl_mean.item())
params = list(actor.parameters()) + list(critic.parameters())
opt = torch.optim.Adam(params, lr=lr)
if adam_t0 > 0:
    # continue from the kernel's optimiser state (warm-up runs): same flat layout as the parameters
    for n_i, model in enumerate((actor, critic)):
        off = n_i * mk.NET_P
        for m in (mm for mm in model.mlp if isinstance(mm, torch.nn.Linear)):
            for p in (m.weight, m.bias):
                k = p.numel()
                opt.state[p] = {
                    "step": torch.tensor(float(adam_t0)),
                    "exp_avg": m0[off : off + k].reshape(p.shape).clone(),
                    "exp_avg_sq": v0[off : off + k].reshape(p.shape).clone(),
                }
                off += k
    sp = actor.distribution.std_param
    opt.state[sp] = {
        "step": torch.tensor(float(adam_t0)),
        "exp_avg": m0[mk.OFF_STD : mk.OFF_STD + 1].reshape(sp.shape).clone(),
        "exp_avg_sq": v0[mk.OFF_STD : mk.OFF_STD + 1].reshape(sp.shape).clone(),
    }
torch.nn.utils.clip_grad_norm_(actor.parameters(), ppo.max_grad_norm)
torch.nn.utils.clip_grad_norm_(critic.parameters(), ppo.max_grad_norm)
opt.step()
p1_ref = mk.pack_params(actor, critic).double()
err = float((torch.tensor(p1_k[: mk.N_PARAMS]) - p1_ref[: mk.N_PARAMS]).abs().max())
check("first_adam_step_params_vs_torch", err, 1e-6)

# ------------------------------------------------------------------------------------------------
# every update of the iteration, teacher-forced: start torch from the kernel's exact state before update q
# (parameters, Adam moments, learning rate), run RSL-RL's mini-batch step on the same samples, and compare
# the state after it.  This checks all num_epochs x num_mini_batches updates at float32 rounding level.
# ------------------------------------------------------------------------------------------------
NP = mk.N_PARAMS_PAD
nq = ppo.num_learning_epochs * ppo.num_mini_batches
off_steps = off_p0 + NP
step_stride = 3 * NP + 4
raw = trainer.dbg.detach().cpu().numpy()


def kernel_step(q):
    b = off_steps + q * step_stride
    lr_q = float(raw[b + 3 * NP : b + 3 * NP + 2].view(np.float64)[0])
    return dbg[b : b + NP], dbg[b + NP : b + 2 * NP], dbg[b + 2 * NP : b + 3 * NP], lr_q, float(raw[b + 3 * NP + 2])


def minibatch(mb):
    rows = []
    for pc in per_cta:
        mbl = pc["n"] // ppo.num_mini_batches
        rows.append(pc["st"][:, pc["perm"][mb * mbl : (mb + 1) * mbl]])
    return np.concatenate(rows, axis=1)


def adam_state(opt, m_flat, v_flat, step):
    """torch.optim.Adam state from the kernel's flat Adam moments (same layout as the parameters)."""
    m_flat, v_flat = torch.as_tensor(m_flat, dtype=torch.float32), torch.as_tensor(v_flat, dtype=torch.float32)
    for n_i, model in enumerate((actor, critic)):
        off = n_i * mk.NET_P
        for m in (mm for mm in model.mlp if isinstance(mm, torch.nn.Linear)):
            for p in (m.weight, m.bias):
                k = p.numel()
                opt.state[p] = {
                    "step": torch.tensor(float(step)),
                    "exp_avg": m_flat[off : off + k].reshape(p.shape).clone(),
                    "exp_avg_sq": v_flat[off : off + k].reshape(p.shape).clone(),
                }
                off += k
    sp = actor.distribution.std_param
    opt.state[sp] = {
        "step": torch.tensor(float(step)),
        "exp_avg": m_flat[mk.OFF_STD : mk.OFF_STD + 1].reshape(sp.shape).clone(),
        "exp_avg_sq": v_flat[mk.OFF_STD : mk.OFF_STD + 1].reshape(sp.shape).clone(),
    }


def adam_flat(opt):
    m_flat, v_flat = torch.zeros(NP, dtype=torch.float64), torch.zeros(NP, dtype=torch.float64)
    for n_i, model in enumerate((actor, critic)):
        off = n_i * mk.NET_P
        for m in (mm for mm in model.mlp if isinstance(mm, torch.nn.Linear)):
            for p in (m.weight, m.bias):
                k = p.numel()
                m_flat[off : off + k] = opt.state[p]["exp_avg"].reshape(-1).double()
                v_flat[off : off + k] = opt.state[p]["exp_avg_sq"].reshape(-1).double()
                off += k
    sp = actor.distribution.std_param
    m_flat[mk.OFF_STD] = opt.state[sp]["exp_avg"].reshape(-1)[0].double()
    v_flat[mk.OFF_STD] = opt.state[sp]["exp_avg_sq"].reshape(-1)[0].double()
    return m_flat, v_flat


def rsl_rl_update(opt, B, lr_before):
    """One RSL-RL 5.5.1 PPO mini-batch update (ppo.py update(), verbatim math) -> (kl_mean, lr_used)."""
    obs_b = tt(B[0:4].T)
    act_b, oldlogp_b, oldmu_b = tt(B[f_act])[:, None], tt(B[f_logp]), tt(B[f_mu])[:, None]
    adv_b, ret_b, val_b = tt(B[f_adv]), tt(B[f_ret])[:, None], tt(B[f_val])[:, None]
    td = TensorDict({"policy": obs_b}, batch_size=[obs_b.shape[0]])
    actor(td, stochastic_output=True)
    actions_log_prob = actor.get_output_log_prob(act_b)
    values = critic(td)
    distribution_params = tuple(p for p in actor.output_distribution_params)
    entropy = actor.output_entropy
    with torch.no_grad():
        kl = actor.get_kl_divergence((oldmu_b, torch.full_like(oldmu_b, sigma0)), distribution_params)
        km = torch.mean(kl).item()
    lr_used = adapt(lr_before, km)
    for g in opt.param_groups:
        g["lr"] = lr_used
    ratio = torch.exp(actions_log_prob - torch.squeeze(oldlogp_b))
    surrogate = -torch.squeeze(adv_b) * ratio
    surrogate_clipped = -torch.squeeze(adv_b) * torch.clamp(ratio, 1.0 - ppo.clip_param, 1.0 + ppo.clip_param)
    surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
    value_clipped = val_b + (values - val_b).clamp(-ppo.clip_param, ppo.clip_param)
    value_loss = torch.max((values - ret_b).pow(2), (value_clipped - ret_b).pow(2)).mean()
    loss = surrogate_loss + ppo.value_loss_coef * value_loss - ppo.entropy_coef * entropy.mean()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), ppo.max_grad_norm)
    torch.nn.utils.clip_grad_norm_(critic.parameters(), ppo.max_grad_norm)
    opt.step()
    return km, lr_used


tf_param, tf_m, tf_v, tf_kl, tf_lr = 0.0, 0.0, 0.0, 0.0, 0
kls = []
before = (P0.double().numpy(), m0.double().numpy(), v0.double().numpy(), lr0)
for q in range(nq):
    p_b, m_b, v_b, lr_b = before
    mk.unpack_params(torch.as_tensor(p_b, dtype=torch.float32), actor, critic)
    opt_q = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr_b)
    if adam_t0 + q > 0:
        adam_state(opt_q, m_b, v_b, adam_t0 + q)
    km, lr_used = rsl_rl_update(opt_q, minibatch(q % ppo.num_mini_batches), lr_b)
    kls.append(km)
    p_k, m_k, v_k, lr_k, kl_k = kernel_step(q)
    p_t = mk.pack_params(actor, critic).double().numpy()
    m_t, v_t = (x.numpy() for x in adam_flat(opt_q))
    tf_param = max(tf_param, float(np.abs(p_t[: mk.N_PARAMS] - p_k[: mk.N_PARAMS]).max()))
    # moments relative to the largest moment of the vector (tiny moments carry no rounding headroom)
    tf_m = max(
        tf_m, float(np.abs(m_t[: mk.N_PARAMS] - m_k[: mk.N_PARAMS]).max() / (np.abs(m_t[: mk.N_PARAMS]).max() + 1e-30))
    )
    tf_v = max(
        tf_v, float(np.abs(v_t[: mk.N_PARAMS] - v_k[: mk.N_PARAMS]).max() / (np.abs(v_t[: mk.N_PARAMS]).max() + 1e-30))
    )
    tf_kl = max(tf_kl, abs(km - kl_k))
    tf_lr += int(lr_used != lr_k)
    before = (p_k, m_k, v_k, lr_k)  # teacher forcing: the next update starts from the kernel's state
check(f"teacher_forced_update_params ({nq} updates, max abs)", tf_param, 1e-6)
# A sample whose probability ratio lies within float rounding of the PPO clip boundary (1 +- clip) can fall on
# different sides in torch and in the kernel; that changes the mini-batch gradient by ~one sample (1/N_mb), i.e.
# up to ~1e-4 of the largest Adam moment at large mini-batches.  Systematic errors would show up at >= 1e-2.
check("teacher_forced_update_adam_m (max abs / max |m|)", tf_m, 1e-3)
check("teacher_forced_update_adam_v (max abs / max |v|)", tf_v, 1e-3)
check("teacher_forced_update_kl_mean (max abs)", tf_kl, 1e-6)
check("teacher_forced_update_learning_rate (mismatches)", float(tf_lr), 0.0)

# ------------------------------------------------------------------------------------------------
# free-running replay of the whole update phase (informational): rounding differences of ~1e-7 per
# update are amplified by 20 sequential non-linear steps, so this is reported, with a loose bound
# that only catches gross errors.
# ------------------------------------------------------------------------------------------------
mk.unpack_params(P0, actor, critic)
opt_f = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr0)
if adam_t0 > 0:
    adam_state(opt_f, m0.numpy(), v0.numpy(), adam_t0)
lr = lr0
for q in range(nq):
    _, lr = rsl_rl_update(opt_f, minibatch(q % ppo.num_mini_batches), lr)
p_final_ref = mk.pack_params(actor, critic).double()
p_final_k = trainer.params.detach().cpu().double()
rel_final = float((p_final_k[: mk.N_PARAMS] - p_final_ref[: mk.N_PARAMS]).norm() / p_final_ref[: mk.N_PARAMS].norm())
check("iteration_final_params_vs_rsl_rl_replay (relative L2, free-running)", rel_final, 1e-3)
report["iteration_final_params_max_abs_err_free_running"] = float(
    (p_final_k[: mk.N_PARAMS] - p_final_ref[: mk.N_PARAMS]).abs().max()
)
report["iteration_final_params_relative_l2_err"] = rel_final
report["replay_kl_means"] = kls
check("iteration_final_learning_rate", abs(float(trainer.scal[0].item()) - kernel_step(nq - 1)[3]), 0.0)
print(
    f"       teacher-forced: params {tf_param:.2e}, adam m {tf_m:.2e}, v(rel) {tf_v:.2e}, kl {tf_kl:.2e}, "
    f"lr mismatches {tf_lr}; free-running final-param rel L2 {rel_final:.3e}; KL range {min(kls):.2e}..{max(kls):.2e}"
)
report["kernel_minus_rsl_rl_note"] = (
    "mini-batches are stratified per SM (each SM contributes 1/num_mini_batches of its own "
    "transitions); the loss/gradient math on a given mini-batch is checked exactly above"
)
report["failures"] = fails
print(json.dumps({k: v for k, v in report.items() if not isinstance(v, dict)}, indent=1))
if args.out:
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
sys.exit(1 if fails else 0)

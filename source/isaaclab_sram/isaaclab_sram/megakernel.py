# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Host driver for the SRAM-resident PPO megakernel (Isaac-Cartpole-Direct).

The whole RSL-RL PPO training loop (rollout, GAE, epochs x mini-batches of PPO updates, Adam) runs
inside one cooperative CUDA kernel compiled at runtime with NVRTC.  The host only

1. resolves the task / agent configuration from Isaac Lab's registry,
2. initialises the networks exactly like RSL-RL (``rsl_rl.models.MLPModel``),
3. allocates a few KB of persistent state and launches the kernel, and
4. reads back per-iteration statistics and the trained weights.
"""

from __future__ import annotations

import math
import pathlib
import time
from dataclasses import dataclass

import numpy as np
import torch

from . import cuda_rt as rt
from .cartpole import CartpoleTask, PPOConfig

KERNEL_SRC = pathlib.Path(__file__).resolve().parent / "kernels" / "ppo_cartpole.cu"
KERNEL_HEADERS = ("sram_device.cuh", "cartpole_physics.cuh")


def kernel_headers() -> dict:
    """In-memory NVRTC headers shipped next to the kernel."""
    return {name: (KERNEL_SRC.parent / name).read_text() for name in KERNEL_HEADERS}


# must match the kernel
THREADS = 256
HID = 32
OBS = 4
CHUNK = 32
NET_P = HID * OBS + HID + HID * HID + HID + HID + 1  # 1249
N_PARAMS = 2 * NET_P + 1
N_PARAMS_PAD = 2500
OFF_STD = 2 * NET_P
VEC_STRIDE = 2560  # all-reduce vector stride (gradient + loss statistics)
ACC_F_STRIDE = VEC_STRIDE  # (debug-dump layout name)
NFIELD = 12
NSA = 10
NSTAT = 20
STAT_NAMES = [
    "episodes",
    "episode_return_sum",
    "episode_length_sum",
    "terminated",
    "timed_out",
    "step_reward_sum",
    "value_loss",
    "surrogate_loss",
    "entropy",
    "kl",
    "learning_rate",
    "action_std",
    "grad_norm_actor",
    "grad_norm_critic",
    "cart_limit_hits",
    "adv_mean",
    "adv_std",
]
PROF_STRIDE = 80  # clock64 marks per iteration in PROF builds (kernel PROF_STRIDE)
PROF_MAX_ITERATIONS = 4096
SMEM_OPTIN_SM120 = 101376  # bytes per block on sm_120; the planner queries the device when it can


def _f(x: float) -> str:
    """float32 C literal with full precision."""
    v = float(np.float32(x))
    if not math.isfinite(v):
        raise ValueError(x)
    return f"({v!r}f)"


@dataclass
class LaunchPlan:
    num_envs: int
    num_ctas: int
    envs_per_cta_max: int
    store_in_smem: bool
    smem_bytes: int
    mb_global: int
    perm_in_smem: bool = True


def plan_launch(
    num_envs: int, ppo: PPOConfig, num_ctas: int | None, storage: str, num_sms: int, smem_optin: int = SMEM_OPTIN_SM120
) -> LaunchPlan:
    if num_envs <= 0 or num_sms <= 0:
        raise ValueError("num_envs and num_sms must be positive")
    if storage not in {"auto", "smem", "global"}:
        raise ValueError(f"unknown rollout storage {storage!r}; expected auto, smem, or global")
    if num_ctas is not None and num_ctas <= 0:
        raise ValueError("num_ctas must be positive")
    if min(ppo.num_steps_per_env, ppo.num_learning_epochs, ppo.num_mini_batches) <= 0:
        raise ValueError("PPO rollout steps, learning epochs, and mini-batches must be positive")
    n_ctas = int(num_ctas if num_ctas is not None else min(num_sms, num_envs))
    if n_ctas > num_sms:
        raise ValueError(f"num_ctas={n_ctas} exceeds the {num_sms} SMs (one persistent CTA per SM)")
    e_max = math.ceil(num_envs / n_ctas)
    T = ppo.num_steps_per_env
    fstride = T * e_max
    base = (
        N_PARAMS_PAD + 2 * HID * 33 + CHUNK * OBS + 4 * 2 * CHUNK * HID + 2 * 2 * CHUNK + NSA * CHUNK + 64 + e_max * 7
    )
    store = NFIELD * fstride
    static_smem = 256  # s_redd + alignment slack
    fits = 4 * (base + store) + static_smem <= smem_optin
    if 4 * base + static_smem > smem_optin:
        raise ValueError(f"{num_envs} envs over {n_ctas} CTAs do not fit even with global rollout storage")
    if storage == "smem" and not fits:
        raise ValueError(f"rollout storage for {e_max} envs/CTA does not fit in shared memory")
    store_smem = fits if storage == "auto" else storage == "smem"
    used = base + (store if store_smem else 0)
    # the per-iteration mini-batch permutation goes to SMEM with whatever space the rollout leaves
    perm = fstride if 4 * (used + fstride) + static_smem <= smem_optin else 0
    smem_bytes = 4 * (used + perm)
    # every CTA contributes floor(E_c * T / N_MB) samples to each mini-batch
    mb_global = 0
    for c in range(n_ctas):
        e0, e1 = (c * num_envs) // n_ctas, ((c + 1) * num_envs) // n_ctas
        mb_global += ((e1 - e0) * T) // ppo.num_mini_batches
    if mb_global == 0:
        raise ValueError("each PPO mini-batch is empty; increase rollout steps or decrease mini-batches or CTAs")
    return LaunchPlan(num_envs, n_ctas, e_max, store_smem, smem_bytes, mb_global, perm_in_smem=perm > 0)


def kernel_defines(task: CartpoleTask, ppo: PPOConfig, plan: LaunchPlan, debug: bool = False) -> dict:
    p = task.physics
    return {
        "N_ENVS": plan.num_envs,
        "N_CTAS": plan.num_ctas,
        "E_MAX": plan.envs_per_cta_max,
        "STORE_SMEM": int(plan.store_in_smem),
        "PERM_SMEM": int(plan.perm_in_smem),
        "DBG": int(debug),
        "T_STEPS": ppo.num_steps_per_env,
        "N_EPOCHS": ppo.num_learning_epochs,
        "N_MB": ppo.num_mini_batches,
        "INV_NMB": _f(1.0 / plan.mb_global),
        # physics (Newton / MuJoCo-Warp implicitfast)
        "P_MC": _f(p.cart_mass),
        "P_MP": _f(p.pole_mass),
        "P_L": _f(p.pole_com),
        "P_IP": _f(p.pole_inertia),
        "P_G": _f(p.gravity),
        "P_KV": _f(p.cart_damping),
        "P_ACT_FRC": _f(p.actuator_force_limit),
        "P_DT": _f(task.physics_dt),
        "P_DECIM": int(task.decimation),
        "P_LIMIT": _f(p.cart_limit),
        "P_VLIM_CART": _f(p.cart_vel_limit),
        "P_VLIM_POLE": _f(p.pole_vel_limit),
        # MDP
        "P_ACTION_SCALE": _f(task.action_scale),
        "P_MAX_CART": _f(task.max_cart_pos),
        "P_MAX_EP_LEN": int(task.max_episode_length),
        "P_STEP_DT": _f(task.step_dt),
        "P_REW_ALIVE": _f(task.rew_scale_alive),
        "P_REW_TERM": _f(task.rew_scale_terminated),
        "P_REW_POLE_POS": _f(task.rew_scale_pole_pos),
        "P_REW_CART_VEL": _f(task.rew_scale_cart_vel),
        "P_REW_POLE_VEL": _f(task.rew_scale_pole_vel),
        "P_RST_S_LO": _f(task.initial_cart_position_range[0]),
        "P_RST_S_HI": _f(task.initial_cart_position_range[1]),
        "P_RST_SD_LO": _f(task.initial_cart_velocity_range[0]),
        "P_RST_SD_HI": _f(task.initial_cart_velocity_range[1]),
        "P_RST_TH_LO": _f(task.initial_pole_angle_range[0]),
        "P_RST_TH_HI": _f(task.initial_pole_angle_range[1]),
        "P_RST_THD_LO": _f(task.initial_pole_velocity_range[0]),
        "P_RST_THD_HI": _f(task.initial_pole_velocity_range[1]),
        # PPO
        "P_GAMMA": _f(ppo.gamma),
        "P_LAMBDA": _f(ppo.lam),
        "P_CLIP": _f(ppo.clip_param),
        "P_ENT_COEF": _f(ppo.entropy_coef),
        "P_VF_COEF": _f(ppo.value_loss_coef),
        "P_DESIRED_KL": _f(ppo.desired_kl if ppo.desired_kl > 0 else 1.0),
        "P_ADAPTIVE": int(ppo.schedule == "adaptive" and ppo.desired_kl > 0),
        "P_MAX_GRAD_NORM": _f(ppo.max_grad_norm),
        # torch.optim.Adam (foreach) passes Python floats: beta2 and (1 - beta) are rounded to float32 once,
        # the bias corrections 1 - beta ** step are computed in float64
        "ADAM_B1": _f(0.9),
        "ADAM_B2": _f(0.999),
        "ADAM_1MB1": _f(1.0 - 0.9),
        "ADAM_1MB2": _f(1.0 - 0.999),
        "ADAM_B1_D": "(0.9)",
        "ADAM_B2_D": "(0.999)",
        "ADAM_EPS": _f(1e-8),
    }


def rsl_rl_models(ppo: PPOConfig, seed: int):
    """Actor / critic modules exactly as RSL-RL 5.x builds them for this task (default nn.Linear init)."""
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    torch.manual_seed(seed)
    obs = TensorDict({"policy": torch.zeros(1, OBS)}, batch_size=[1])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = MLPModel(
        obs,
        groups,
        "actor",
        1,
        hidden_dims=list(ppo.hidden_dims),
        activation="elu",
        obs_normalization=False,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": ppo.init_noise_std},
    )
    critic = MLPModel(
        obs, groups, "critic", 1, hidden_dims=list(ppo.hidden_dims), activation="elu", obs_normalization=False
    )
    return actor, critic


def pack_params(actor, critic) -> torch.Tensor:
    flat = torch.zeros(N_PARAMS_PAD, dtype=torch.float32)
    for n, model in enumerate((actor, critic)):
        lin = [m for m in model.mlp if isinstance(m, torch.nn.Linear)]
        chunks = []
        for m in lin:
            chunks += [m.weight.detach().reshape(-1), m.bias.detach().reshape(-1)]
        v = torch.cat(chunks)
        assert v.numel() == NET_P
        flat[n * NET_P : (n + 1) * NET_P] = v
    flat[OFF_STD] = actor.distribution.std_param.detach().reshape(-1)[0]
    return flat


def unpack_params(flat: torch.Tensor, actor, critic) -> None:
    flat = flat.detach().float().cpu()
    with torch.no_grad():
        for n, model in enumerate((actor, critic)):
            off = n * NET_P
            for m in (mm for mm in model.mlp if isinstance(mm, torch.nn.Linear)):
                k = m.weight.numel()
                m.weight.copy_(flat[off : off + k].reshape(m.weight.shape))
                off += k
                k = m.bias.numel()
                m.bias.copy_(flat[off : off + k].reshape(m.bias.shape))
                off += k
        actor.distribution.std_param.copy_(flat[OFF_STD : OFF_STD + 1])


class SramPPOCartpole:
    """SRAM-resident PPO trainer for Isaac-Cartpole-Direct (one persistent CUDA kernel)."""

    def __init__(
        self,
        task: CartpoleTask | None = None,
        ppo: PPOConfig | None = None,
        *,
        num_envs: int = 4096,
        seed: int = 42,
        num_ctas: int | None = None,
        storage: str = "auto",
        debug: bool = False,
        profile: bool = False,
        device: str = "cuda:0",
        extra_defines: dict | None = None,
    ):
        self.task = task or CartpoleTask()
        self.ppo = ppo or PPOConfig()
        self.seed = int(seed)
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        rt.ensure_context(self.device.index or 0)
        self.num_sms = rt.num_sms()
        self.smem_optin = rt.device_attr("MAX_SHARED_MEMORY_PER_BLOCK_OPTIN")
        self.plan = plan_launch(num_envs, self.ppo, num_ctas, storage, self.num_sms, self.smem_optin)
        if debug and profile:
            raise ValueError("debug and profile builds are mutually exclusive")
        self.debug = debug
        self.profile = profile
        t0 = time.perf_counter()
        self.defines = kernel_defines(self.task, self.ppo, self.plan, debug)
        self.defines["PROF"] = int(profile)
        self.defines["PROF_MAX_IT"] = PROF_MAX_ITERATIONS
        self.defines.update(extra_defines or {})
        self.module = rt.compile_module(
            KERNEL_SRC.read_text(), defines=self.defines, headers=kernel_headers(), name="ppo_cartpole.cu"
        )
        self.compile_s = time.perf_counter() - t0
        self.kernel = self.module.kernel("sram_ppo_cartpole")
        self.kernel.set_max_dynamic_smem(self.plan.smem_bytes)
        occ = self.kernel.occupancy(THREADS, self.plan.smem_bytes)
        if occ < 1:
            raise RuntimeError(f"kernel does not fit on an SM: {self.kernel.info()} smem={self.plan.smem_bytes}")
        self.kernel_info = self.kernel.info()
        self._alloc()
        self.reset()

    # ------------------------------------------------------------------------------------------
    def _alloc(self):
        d, f32 = self.device, torch.float32
        p = self.plan
        T = self.ppo.num_steps_per_env
        fstride = T * p.envs_per_cta_max
        self.params = torch.zeros(N_PARAMS_PAD, device=d, dtype=f32)
        self.adam_m = torch.zeros(N_PARAMS_PAD, device=d, dtype=f32)
        self.adam_v = torch.zeros(N_PARAMS_PAD, device=d, dtype=f32)
        self.scal = torch.zeros(4, device=d, dtype=torch.float64)
        self.env_state = torch.zeros(p.num_envs, 4, device=d, dtype=f32)
        self.ep_len = torch.zeros(p.num_envs, device=d, dtype=torch.int32)
        self.ep_ret = torch.zeros(p.num_envs, device=d, dtype=f32)
        # per-CTA partial vectors + the reduced vector (pad slots stay zero forever)
        self.part = torch.zeros((p.num_ctas + 1) * VEC_STRIDE, device=d, dtype=f32)
        self.partd = torch.zeros(2 * p.num_ctas, device=d, dtype=torch.float64)
        self.bar = torch.zeros(64, device=d, dtype=torch.int32)
        self.store = torch.zeros(1 if p.store_in_smem else p.num_ctas * NFIELD * fstride, device=d, dtype=f32)
        if self.debug:
            n = p.num_ctas * 4 * fstride + p.num_ctas * (NFIELD + 1) * fstride + ACC_F_STRIDE + 2 * N_PARAMS_PAD
            # + the state after every update of the first iteration (params, Adam m, Adam v, lr, KL)
            n += self.ppo.num_learning_epochs * self.ppo.num_mini_batches * (3 * N_PARAMS_PAD + 4)
            self.dbg = torch.zeros(n, device=d, dtype=f32)
        elif self.profile:
            # int64 clock64 marks [iteration][PROF_STRIDE]; the kernel stops recording past PROF_MAX_ITERATIONS
            self.dbg = torch.zeros(2 * PROF_STRIDE * PROF_MAX_ITERATIONS, device=d, dtype=f32)
        else:
            self.dbg = torch.zeros(1, device=d, dtype=f32)
        self.stats = torch.zeros(0, NSTAT, device=d, dtype=f32)

    def reset(self):
        """Initialise networks (RSL-RL init), optimiser state and environments."""
        self.actor, self.critic = rsl_rl_models(self.ppo, self.seed)
        self.params.copy_(pack_params(self.actor, self.critic).to(self.device))
        self.adam_m.zero_()
        self.adam_v.zero_()
        self.scal.zero_()
        self.scal[0] = self.ppo.learning_rate
        self.scal[2] = 1.0  # beta1 ** step
        self.scal[3] = 1.0  # beta2 ** step
        g = torch.Generator(device="cpu").manual_seed(self.seed + 1)
        n = self.plan.num_envs
        t = self.task

        def u(lo_hi):
            return lo_hi[0] + (lo_hi[1] - lo_hi[0]) * torch.rand(n, generator=g)

        st = torch.zeros(n, 4)
        st[:, 0] = u(t.initial_cart_position_range)
        st[:, 1] = u(t.initial_pole_angle_range)
        # Isaac Lab 3.0 + Newton: the very first reset happens before the soft velocity limits are
        # populated, so initial joint velocities are clamped to 0 (observed in the stock env).
        self.env_state.copy_(st.to(self.device))
        if self.ppo.init_at_random_ep_len:
            self.ep_len.copy_(
                torch.randint(0, t.max_episode_length, (n,), generator=g, dtype=torch.int32).to(self.device)
            )
        else:
            self.ep_len.zero_()
        self.ep_ret.zero_()
        self.iteration = 0
        self.stats = torch.zeros(0, NSTAT, device=self.device, dtype=torch.float32)

    # ------------------------------------------------------------------------------------------
    def _launch(self, iter0: int, n_iters: int):
        seed = self.seed & 0xFFFFFFFFFFFFFFFF
        args = [
            self.params,
            self.adam_m,
            self.adam_v,
            self.scal,
            self.env_state,
            self.ep_len,
            self.ep_ret,
            self.part,
            self.partd,
            self.bar,
            self.stats,
            self.store,
            self.dbg if (self.debug or self.profile) else rt.ptr(0),
            rt.i32(iter0),
            rt.i32(n_iters),
            rt.u32(seed & 0xFFFFFFFF),
            rt.u32((seed >> 32) ^ 0x5EED5EED),
        ]
        self.kernel.launch(self.plan.num_ctas, THREADS, args, smem=self.plan.smem_bytes, cooperative=True)

    def train(self, num_iterations: int | None = None, iters_per_launch: int | None = None) -> dict:
        """Run ``num_iterations`` PPO iterations on the GPU; returns timing + per-iteration statistics."""
        n = int(num_iterations if num_iterations is not None else self.ppo.max_iterations)
        if n < 0:
            raise ValueError("num_iterations must be nonnegative")
        if iters_per_launch is not None and iters_per_launch <= 0:
            raise ValueError("iters_per_launch must be positive")
        if n == 0:
            return {
                "iterations": 0,
                "first_iteration": self.iteration,
                "launches": 0,
                "gpu_time_s": 0.0,
                "wall_time_s": 0.0,
                "env_steps": 0,
                "env_steps_per_s": 0.0,
                "stats": self.stats_table(self.iteration, self.iteration),
            }
        per = int(iters_per_launch if iters_per_launch is not None else n)
        if per <= 0:
            raise ValueError("iters_per_launch must be a positive integer")
        need = self.iteration + n
        if self.stats.shape[0] < need:
            grown = torch.zeros(need, NSTAT, device=self.device, dtype=torch.float32)
            grown[: self.stats.shape[0]] = self.stats
            self.stats = grown
        torch.cuda.synchronize()
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall0 = time.perf_counter()
        start.record()
        done = 0
        while done < n:
            k = min(per, n - done)
            self._launch(self.iteration + done, k)
            done += k
        stop.record()
        stop.synchronize()
        wall = time.perf_counter() - wall0
        gpu_ms = start.elapsed_time(stop)
        first = self.iteration
        self.iteration += n
        steps = n * self.plan.num_envs * self.ppo.num_steps_per_env
        return {
            "iterations": n,
            "first_iteration": first,
            "launches": math.ceil(n / per),
            "gpu_time_s": gpu_ms / 1e3,
            "wall_time_s": wall,
            "env_steps": steps,
            "env_steps_per_s": steps / (gpu_ms / 1e3),
            "stats": self.stats_table(first, first + n),
        }

    # ------------------------------------------------------------------------------------------
    def stats_table(self, i0: int = 0, i1: int | None = None) -> dict[str, np.ndarray]:
        s = self.stats[i0:i1].double().cpu().numpy()
        out = {name: s[:, k] for k, name in enumerate(STAT_NAMES)}
        eps = np.maximum(out["episodes"], 1.0)
        out["mean_episode_return"] = out["episode_return_sum"] / eps
        out["mean_episode_length"] = out["episode_length_sum"] / eps
        out["mean_step_reward"] = out["step_reward_sum"] / (self.plan.num_envs * self.ppo.num_steps_per_env)
        return out

    def trained_models(self):
        """Return RSL-RL ``MLPModel`` actor / critic holding the trained weights."""
        unpack_params(self.params, self.actor, self.critic)
        return self.actor, self.critic

    def describe(self) -> dict:
        p = self.plan
        T = self.ppo.num_steps_per_env
        rollout_bytes = p.num_envs * T * NFIELD * 4
        return {
            "num_envs": p.num_envs,
            "num_ctas": p.num_ctas,
            "num_sms": self.num_sms,
            "envs_per_cta_max": p.envs_per_cta_max,
            "threads_per_cta": THREADS,
            "rollout_storage": "shared memory (SRAM)" if p.store_in_smem else "global memory (L2/DRAM)",
            "dynamic_smem_bytes_per_cta": p.smem_bytes,
            "rollout_buffer_bytes_total": rollout_bytes,
            "params": N_PARAMS,
            "mini_batch_size": p.mb_global,
            "registers_per_thread": self.kernel_info.get("NUM_REGS"),
            "local_mem_bytes_per_thread": self.kernel_info.get("LOCAL_SIZE_BYTES"),
            "nvrtc_compile_s": self.compile_s,
        }

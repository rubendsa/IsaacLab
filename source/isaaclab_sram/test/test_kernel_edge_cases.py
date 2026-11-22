# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adversarial edge cases of the SRAM-resident PPO megakernel.

``test_megakernel_parity.py`` checks one iteration at five launch configurations and ``test_determinism.py`` checks
4096 environments. These tests cover what they leave out:

* launch configurations: environments not divisible by the CTAs, one CTA, CTAs without environments, several rollout
  chunks per step, more environments per CTA than threads, the largest shared- and global-memory launches and
  ``num_ctas`` overrides,
* the launch planner against the kernel's shared-memory carve-up,
* bitwise equality of shared- and global-memory rollout storage, of split launches at odd sizes and of repeated runs,
* every optimizer step of an iteration replayed one at a time from the kernel's own state. An instrumented copy of the
  kernel dumps the state after each step, which separates real defects from the drift of a free-running replay,
* the first-mini-batch KL divergence, the barrier after the advantage normalization, host-side bookkeeping
  (``reset``, ``train(0)``, the profiling buffer) and the random streams,
* learning at odd launch configurations.

The instrumented kernels are built from the kernel source at test time in a temporary directory; the package files are
never modified. Several tests are regression tests for defects found by an adversarial review and fixed since
(``ADV_BARRIER_NOTE``, ``KL_NOTE``, host-side bookkeeping).
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

try:
    import isaaclab_sram  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name != "isaaclab_sram":
        raise
    # The package is not part of the uv-locked workspace, so import it from this checkout when it is not installed.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("cuda.bindings", reason="the megakernel runtime needs the CUDA driver and NVRTC bindings")
pytest.importorskip("rsl_rl", reason="the trainer initializes its networks with RSL-RL")

from isaaclab_sram import megakernel as mk  # noqa: E402
from isaaclab_sram.cartpole import PPOConfig  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required"),
]

VERIFY_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "sram" / "verify_megakernel.py"

STATE_TENSORS = ("params", "adam_m", "adam_v", "scal", "env_state", "ep_len", "ep_ret")
"""Trainer tensors that carry the training state from one launch to the next."""

# rollout storage fields of the kernel (F_* in ppo_cartpole.cu)
F_ACT, F_LOGP, F_VAL, F_MU, F_DONE, F_ADV, F_RET = 4, 5, 6, 7, 9, 10, 11

STEP_RECORD = 3 * mk.N_PARAMS_PAD + 4
"""Floats per optimizer step in the instrumented kernel's dump: parameters, Adam m, Adam v, learning rate, KL."""

ADV_BARRIER_NOTE = (
    "regression: the in-place advantage normalisation needs a __syncthreads() before the debug dump and the first "
    "mini-batch gather (other warps' entries); without it, global rollout storage at >= 131072 environments was not "
    "bitwise reproducible and the debug dump could hold raw advantages"
)
KL_NOTE = (
    "regression: (sigma_old / sigma)^2 must be computed by division like torch; (sigma_old * (1 / sigma))^2 is "
    "1 - 2^-23 for ~14% of float32 std values, which made the first mini-batch KL +7e-15 instead of exactly 0 and "
    "multiplied the learning rate by 1.5 where RSL-RL keeps it"
)


# ------------------------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------------------------
def _num_sms() -> int:
    return torch.cuda.get_device_properties(0).multi_processor_count


def _plan(num_envs: int, num_ctas: int | None = None, storage: str = "auto") -> mk.LaunchPlan:
    from isaaclab_sram import cuda_rt as rt

    rt.ensure_context(0)
    opt_in = rt.device_attr("MAX_SHARED_MEMORY_PER_BLOCK_OPTIN")
    return mk.plan_launch(num_envs, PPOConfig(), num_ctas, storage, rt.num_sms(), opt_in)


def _largest(predicate) -> int:
    """Largest ``num_envs`` >= 1 for which ``predicate`` holds (it must hold for small and fail for large sizes)."""
    lo, hi = 1, 2
    while predicate(hi):
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if predicate(mid) else (lo, mid)
    return lo


def _fits(num_envs: int) -> bool:
    try:
        _plan(num_envs)
    except ValueError:
        return False
    return True


def _largest_smem_resident() -> int:
    return _largest(lambda n: _fits(n) and _plan(n).store_in_smem)


def _largest_launch() -> int:
    return _largest(_fits)


def _resolve(num_envs: int | str) -> int:
    return (
        {"max_smem": _largest_smem_resident, "max_global": _largest_launch}[num_envs]()
        if isinstance(num_envs, str)
        else num_envs
    )


def _snapshot(trainer: mk.SramPPOCartpole) -> dict[str, torch.Tensor]:
    return {name: getattr(trainer, name).clone() for name in STATE_TENSORS}


def _differing(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> list[str]:
    return [name for name in STATE_TENSORS if not torch.equal(a[name], b[name])]


def _rollout_dump(trainer: mk.SramPPOCartpole) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Per CTA of a debug launch: (transitions, storage fields ``[NFIELD, n]``, mini-batch permutation ``[n]``)."""
    plan, steps = trainer.plan, trainer.ppo.num_steps_per_env
    fstride = steps * plan.envs_per_cta_max
    off = plan.num_ctas * 4 * fstride
    store = trainer.dbg[off : off + plan.num_ctas * (mk.NFIELD + 1) * fstride].double().cpu().numpy()
    store = store.reshape(plan.num_ctas, mk.NFIELD + 1, fstride)
    out = []
    for c in range(plan.num_ctas):
        e0, e1 = (c * plan.num_envs) // plan.num_ctas, ((c + 1) * plan.num_envs) // plan.num_ctas
        n = (e1 - e0) * steps
        out.append((n, store[c, : mk.NFIELD, :n], store[c, mk.NFIELD, :n].astype(np.int64)))
    return out


def _read_device_array(trainer: mk.SramPPOCartpole, name: str, dtype) -> np.ndarray:
    from cuda.bindings import driver as cu
    from isaaclab_sram import cuda_rt as rt

    torch.cuda.synchronize()
    dptr, size = trainer.module.global_ptr(name)
    host = np.zeros(size // np.dtype(dtype).itemsize, dtype=dtype)
    rt.check(cu.cuMemcpyDtoH(host.ctypes.data, dptr, size))
    return host


_KERNEL_ENTRY = 'extern "C" __global__ void __launch_bounds__(THREADS, 1) sram_ppo_cartpole('
_GATHER_END = """                        s_sa[SA_OLDV * CHUNK + tid] = st[F_VAL * FSTRIDE + tr];
                    }
                    __syncthreads();
"""
_ADAM_END = """                if (tid == 0 && cta == 0) { it_gna = na; it_gnc = nc; }
                __syncthreads();
"""
_INSTRUMENT_GLOBALS = """__device__ unsigned int g_stale_adv[2];   // [stale advantage reads, gathered advantages]
#define STEP_RECORD (3 * N_PARAMS_PAD + 4)
__device__ float g_step_dump[2 * N_EPOCHS * N_MB * STEP_RECORD];
"""
_INSTRUMENT_GATHER = """                        s_sa[SA_OLDV * CHUNK + tid] = st[F_VAL * FSTRIDE + tr];
                    }
                    if (epoch == 0 && mb == 0) {   // test instrumentation: re-read the advantage after a barrier
                        __syncthreads();
                        if (tid < Sc) {
                            const int tr2 = (int)feistel_perm((u32)(mb * mb_local + c0 + tid), (u32)n_local,
                                                              phalf, pkey);
                            const float now = st[F_ADV * FSTRIDE + tr2];
                            if (__float_as_uint(now) != __float_as_uint(s_sa[SA_ADV * CHUNK + tid]))
                                atomicAdd(&g_stale_adv[0], 1u);
                            atomicAdd(&g_stale_adv[1], 1u);
                        }
                    }
                    __syncthreads();
"""
_INSTRUMENT_ADAM = """                if (tid == 0 && cta == 0) { it_gna = na; it_gnc = nc; }
                __syncthreads();
                if (it == 0 && (cta == 0 || cta == N_CTAS - 1)) {   // test instrumentation: state after every step
                    w2_to_flat();
                    __syncthreads();
                    const size_t rec = (size_t)(cta == 0 ? 0 : 1) * (N_EPOCHS * N_MB) + epoch * N_MB + mb;
                    float* dd = g_step_dump + rec * STEP_RECORD;
                    float* dm = dd + N_PARAMS_PAD;       // Adam m
                    float* dv = dd + 2 * N_PARAMS_PAD;   // Adam v
                    for (int i = tid; i < N_PARAMS_PAD; i += THREADS) dd[i] = s_params[i];
#pragma unroll
                    for (int kk = 0; kk < 8; ++kk) { dm[OWN_W2(kk)] = m_w2[kk]; dv[OWN_W2(kk)] = v_w2[kk]; }
                    dm[own_w1] = m_w1; dv[own_w1] = v_w1;
                    if (own_x >= 0) { dm[own_x] = m_x; dv[own_x] = v_x; }
                    if (tid == 0) {
                        dd[3 * N_PARAMS_PAD] = (float)s_opt[0];
                        dd[3 * N_PARAMS_PAD + 1] = __ldcg(&g_red[VEC_KL]) * INV_NMB;
                    }
                    __syncthreads();
                }
"""


def _instrumented_kernel(directory: Path) -> Path:
    """Copy of the megakernel that records stale advantage gathers and the state after every optimizer step."""
    src = mk.KERNEL_SRC.read_text()
    for anchor in (_KERNEL_ENTRY, _GATHER_END, _ADAM_END):
        assert src.count(anchor) == 1, f"instrumentation anchor not found in {mk.KERNEL_SRC.name}: {anchor!r}"
    src = src.replace(_KERNEL_ENTRY, _INSTRUMENT_GLOBALS + _KERNEL_ENTRY)
    src = src.replace(_GATHER_END, _INSTRUMENT_GATHER).replace(_ADAM_END, _INSTRUMENT_ADAM)
    path = directory / "ppo_cartpole.cu"
    path.write_text(src)
    for name, text in mk.kernel_headers().items():
        (directory / name).write_text(text)
    return path


@pytest.fixture
def instrumented(tmp_path, monkeypatch):
    """Make trainers built inside the test compile the instrumented kernel copy."""
    monkeypatch.setattr(mk, "KERNEL_SRC", _instrumented_kernel(tmp_path))


def _adapt(ppo: PPOConfig, lr: float, kl: float) -> float:
    """RSL-RL 5.5.1 KL-adaptive learning rate."""
    if kl > ppo.desired_kl * 2.0:
        return max(1e-5, lr / 1.5)
    if kl < ppo.desired_kl / 2.0 and kl > 0.0:
        return min(1e-2, lr * 1.5)
    return lr


# ------------------------------------------------------------------------------------------------------------------
# launch planner
# ------------------------------------------------------------------------------------------------------------------
def _kernel_carve_up(
    e_max: int, steps: int, rollout_in_smem: bool, perm_in_smem: bool = False
) -> list[tuple[str, int, bool]]:
    """Dynamic shared-memory arrays of ppo_cartpole.cu in declaration order: (name, floats, read as float4)."""
    per_net_chunk = 2 * mk.CHUNK * mk.HID  # NNET * CHUNK * HID
    return [
        ("s_params", mk.N_PARAMS_PAD, False),
        ("s_w2", 2 * mk.HID * 33, False),  # NNET * W2N, padded row stride W2P = 33
        ("s_x", mk.CHUNK * mk.OBS, True),
        ("s_a1", per_net_chunk, True),
        ("s_a2", per_net_chunk, False),
        ("s_dz2", per_net_chunk, True),
        ("s_dz1", per_net_chunk, False),
        ("s_out", 2 * mk.CHUNK, False),
        ("s_gout", 2 * mk.CHUNK, False),
        ("s_sa", mk.NSA * mk.CHUNK, False),
        ("s_red", 64, False),
        ("s_env", 4 * e_max, True),
        ("s_epret", e_max, False),
        ("s_eplen", e_max, False),
        ("s_lastv", e_max, False),
        ("s_perm", steps * e_max if perm_in_smem else 0, False),
        ("s_after_perm", 0, False),
        ("st", mk.NFIELD * steps * e_max if rollout_in_smem else 0, False),
    ]


def test_kernel_carve_up_order_matches_the_planner_model():
    """The test's carve-up model follows the kernel's pointer chain (guards the byte-level check below)."""
    import re

    src = mk.KERNEL_SRC.read_text()
    chain = re.findall(r"^\s*(?:float|int)\*\s+(s_\w+|st)\s*=\s*(?:\((?:float|int)\*\)\()?(\w+)", src, re.MULTILINE)
    names = [name for name, _ in chain]
    expected = [name for name, _, _ in _kernel_carve_up(1, 1, True)]
    assert names[: len(expected)] == expected
    for (name, base), previous in zip(chain[1 : len(expected)], expected[:-1]):
        assert base == previous, f"{name} is not carved right after {previous}"


@pytest.mark.parametrize("num_envs", [1, 189, 1000, 4096, "max_smem", 65536, "max_global"])
@pytest.mark.parametrize("storage", ["auto", "global"])
def test_planner_bytes_and_float4_alignment_match_the_kernel_carve_up(num_envs, storage):
    plan = _plan(_resolve(num_envs), storage=storage)
    carve = _kernel_carve_up(
        plan.envs_per_cta_max, PPOConfig().num_steps_per_env, plan.store_in_smem, plan.perm_in_smem
    )
    assert plan.smem_bytes == 4 * sum(floats for _, floats, _ in carve)
    offset = 0
    for name, floats, as_float4 in carve:
        if as_float4:
            assert (4 * offset) % 16 == 0, f"{name} starts at byte {4 * offset}, not 16-byte aligned"
        offset += floats


def test_kernel_static_shared_memory_fits_the_planner_slack():
    trainer = mk.SramPPOCartpole(num_envs=_num_sms(), seed=0)
    static = trainer.kernel_info["SHARED_SIZE_BYTES"]
    # the dynamic `extern __shared__ __align__(16)` array starts at the next 16-byte boundary after the static arrays
    aligned = -(-static // 16) * 16
    assert aligned <= 256, f"the planner reserves 256 bytes of static shared memory, the kernel needs {aligned}"


def test_largest_planned_launches_run():
    """The planner's shared- and global-memory limits are real: both largest launches run, one more env changes plan."""
    n_smem, n_max = _largest_smem_resident(), _largest_launch()
    assert not _plan(n_smem + 1).store_in_smem
    with pytest.raises(ValueError):
        _plan(n_max + 1)
    for n, in_smem in ((n_smem, True), (n_max, False)):
        trainer = mk.SramPPOCartpole(num_envs=n, seed=0)
        assert trainer.plan.store_in_smem == in_smem
        stats = trainer.train(2)["stats"]
        assert all(torch.isfinite(getattr(trainer, name).float()).all() for name in STATE_TENSORS)
        assert stats["episodes"].sum() > 0
        del trainer
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------------------------
# verification script at configurations the parity test does not cover
# ------------------------------------------------------------------------------------------------------------------
_VERIFY_RUNNER = r"""
import runpy, sys
from pathlib import Path
script, num_ctas = sys.argv[1], int(sys.argv[2])
sys.path.insert(0, str(Path(script).resolve().parents[2] / "source" / "isaaclab_sram"))
from isaaclab_sram import megakernel as mk
if num_ctas:
    init = mk.SramPPOCartpole.__init__
    def patched(self, *args, **kwargs):
        kwargs["num_ctas"] = num_ctas
        init(self, *args, **kwargs)
    mk.SramPPOCartpole.__init__ = patched
sys.argv = [script, *sys.argv[3:]]
runpy.run_path(script, run_name="__main__")
"""

VERIFY_CONFIGS = [
    # one CTA: the reduce-scatter loops over all 2504 columns in 157 passes of 16
    pytest.param(1, None, [], id="1_env"),
    # one CTA with two environments, all others with one
    pytest.param("sms+1", None, [], id="sms_plus_1_envs"),
    pytest.param(1000, None, [], id="1000_envs_uneven"),
    # several rollout chunks per step with shared-memory storage
    pytest.param(8192, None, [], id="8192_envs"),
    pytest.param("max_smem", None, [], id="largest_smem_resident"),
    # global rollout storage, several chunks per step
    pytest.param(20000, None, [], id="20000_envs"),
    # more environments per CTA than threads (GAE loops, normalization loops)
    pytest.param(65536, None, [], id="65536_envs"),
    pytest.param(1000, 1, [], id="1000_envs_on_1_cta"),
    pytest.param(4096, 64, [], id="4096_envs_on_64_ctas"),
    pytest.param(64, None, ["--storage", "smem"], id="64_envs_smem"),
    pytest.param(64, None, ["--storage", "global"], id="64_envs_global"),
    pytest.param(100, None, ["--storage", "global"], id="100_envs_global"),
    # the std at the 4th iteration has x * fl(1/x) != 1 on a 188-SM GPU
    pytest.param(1000, None, ["--warmup_iters", "3"], id="1000_envs_after_3_iters"),
    # every optimizer step matches to 1e-7 (see the one-step test below), but the free-running 20-step replay of
    # verify_megakernel.py drifts by 1.5e-4 > 1e-4 at learning rate 1e-2 through the PPO clipping discontinuities
    pytest.param(
        4096,
        None,
        ["--seed", "5", "--warmup_iters", "3"],
        id="4096_envs_seed_5_after_3_iters",
    ),
]


@pytest.mark.parametrize(("num_envs", "num_ctas", "extra"), VERIFY_CONFIGS)
def test_verify_megakernel_at_edge_configurations(num_envs, num_ctas, extra, tmp_path):
    if not VERIFY_SCRIPT.is_file():
        pytest.skip(f"test requires an Isaac Lab source checkout with {VERIFY_SCRIPT.name}")
    num_envs = _num_sms() + 1 if num_envs == "sms+1" else _resolve(num_envs)
    if num_ctas and num_ctas > _num_sms():
        pytest.skip(f"{num_ctas} CTAs need {num_ctas} SMs")
    report_path = tmp_path / "report.json"
    proc = subprocess.run(
        [sys.executable, "-c", _VERIFY_RUNNER, str(VERIFY_SCRIPT), str(num_ctas or 0)]
        + ["--num_envs", str(num_envs), *extra, "--out", str(report_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=900,
    )
    output = (
        f"exit code {proc.returncode}\n--- stdout ---\n{proc.stdout[-6000:]}\n--- stderr ---\n{proc.stderr[-6000:]}"
    )
    assert report_path.is_file(), output
    report = json.loads(report_path.read_text(encoding="utf-8"))
    failed = {name: r for name, r in report.items() if isinstance(r, dict) and not r["ok"]}
    assert not failed, f"failed checks: {failed}\n{output}"
    assert proc.returncode == 0, output
    assert report["num_ctas"] == (num_ctas or min(_num_sms(), num_envs))
    assert report["reset_range_violations"] == 0 and report["permutations_are_bijections"]


def test_verifier_detects_a_late_timeout(tmp_path):
    """The reference must reject incorrect timeout flags instead of deriving its oracle from them."""
    source = mk.KERNEL_SRC.read_text()
    correct = "const bool tout = eplen >= P_MAX_EP_LEN;"
    assert source.count(correct) == 1
    kernel = tmp_path / "ppo_cartpole.cu"
    kernel.write_text(source.replace(correct, "const bool tout = eplen > P_MAX_EP_LEN;"))
    for name, header in mk.kernel_headers().items():
        (tmp_path / name).write_text(header)
    report_path = tmp_path / "report.json"
    runner = """
import runpy, sys
from pathlib import Path
script, kernel, report = sys.argv[1:]
sys.path.insert(0, str(Path(script).resolve().parents[2] / "source" / "isaaclab_sram"))
from isaaclab_sram import megakernel as mk
mk.KERNEL_SRC = Path(kernel)
sys.argv = [script, "--num_envs", "512", "--out", report]
runpy.run_path(script, run_name="__main__")
"""
    proc = subprocess.run(
        [sys.executable, "-c", runner, str(VERIFY_SCRIPT), str(kernel), str(report_path)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert report_path.is_file(), proc.stdout + proc.stderr
    report = json.loads(report_path.read_text())
    assert proc.returncode != 0, "the verifier accepted a kernel with a one-step-late timeout"
    assert "termination_and_timeout_flags" in report["failures"], proc.stdout
    assert "episode_counters_after_rollout" in report["failures"], proc.stdout


# ------------------------------------------------------------------------------------------------------------------
# bitwise determinism at odd launch configurations
# ------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("num_envs", [1, 100, 1000, 8192, "max_smem"])
def test_shared_and_global_rollout_storage_are_bitwise_identical(num_envs):
    num_envs = _resolve(num_envs)
    states = {}
    for storage in ("smem", "global"):
        trainer = mk.SramPPOCartpole(num_envs=num_envs, seed=7, storage=storage)
        assert trainer.plan.store_in_smem == (storage == "smem")
        trainer.train(4)
        states[storage] = _snapshot(trainer)
    assert not _differing(states["smem"], states["global"])


SPLIT_CONFIGS = [
    pytest.param(1000, None, id="1000_envs"),
    pytest.param(20000, None, id="20000_envs_global"),
    pytest.param(50, 100, id="50_envs_on_100_ctas"),  # half of the CTAs own no environment
    pytest.param(4096, 64, id="4096_envs_on_64_ctas"),
]


@pytest.mark.parametrize(("num_envs", "num_ctas"), SPLIT_CONFIGS)
def test_split_launches_are_bitwise_identical_at_odd_configurations(num_envs, num_ctas):
    if num_ctas and num_ctas > _num_sms():
        pytest.skip(f"{num_ctas} CTAs need {num_ctas} SMs")
    single = mk.SramPPOCartpole(num_envs=num_envs, seed=8, num_ctas=num_ctas)
    single.train(6)
    split = mk.SramPPOCartpole(num_envs=num_envs, seed=8, num_ctas=num_ctas)
    assert split.train(2, iters_per_launch=1)["launches"] == 2
    assert split.train(4, iters_per_launch=3)["launches"] == 2
    assert not _differing(_snapshot(single), _snapshot(split))
    assert torch.equal(single.stats[:, :1], split.stats[:, :1])  # episode counts are integer sums: exact


# ------------------------------------------------------------------------------------------------------------------
# every optimizer step, replayed from the kernel's own state
# ------------------------------------------------------------------------------------------------------------------
def _replay_one_step(ppo, actor, critic, flat_state, adam_step, batch, sigma_old, lr):
    """One RSL-RL 5.5.1 PPO optimizer step from ``flat_state`` (params, m, v); returns (params, KL)."""
    from tensordict import TensorDict

    params, m, v = (torch.tensor(x, dtype=torch.float32) for x in flat_state)
    mk.unpack_params(params, actor, critic)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
    if adam_step > 0:
        slots = []
        for n_i, model in enumerate((actor, critic)):
            off = n_i * mk.NET_P
            for lin in (layer for layer in model.mlp if isinstance(layer, torch.nn.Linear)):
                for p in (lin.weight, lin.bias):
                    slots.append((p, off))
                    off += p.numel()
        slots.append((actor.distribution.std_param, mk.OFF_STD))
        for p, off in slots:
            k = p.numel()
            opt.state[p] = {
                "step": torch.tensor(float(adam_step)),
                "exp_avg": m[off : off + k].reshape(p.shape).clone(),
                "exp_avg_sq": v[off : off + k].reshape(p.shape).clone(),
            }
    t = lambda x: torch.tensor(x, dtype=torch.float32)  # noqa: E731
    obs = TensorDict({"policy": t(batch[0:4].T)}, batch_size=[batch.shape[1]])
    act, old_logp, old_mu = t(batch[F_ACT])[:, None], t(batch[F_LOGP]), t(batch[F_MU])[:, None]
    adv, ret, val = t(batch[F_ADV]), t(batch[F_RET])[:, None], t(batch[F_VAL])[:, None]
    actor(obs, stochastic_output=True)
    logp = actor.get_output_log_prob(act)
    values = critic(obs)
    dist = tuple(p for p in actor.output_distribution_params)
    entropy = actor.output_entropy
    with torch.no_grad():
        kl = torch.mean(actor.get_kl_divergence((old_mu, torch.full_like(old_mu, sigma_old)), dist)).item()
    ratio = torch.exp(logp - old_logp)
    surrogate = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 1.0 - ppo.clip_param, 1.0 + ppo.clip_param)).mean()
    clipped = val + (values - val).clamp(-ppo.clip_param, ppo.clip_param)
    value_loss = torch.max((values - ret).pow(2), (clipped - ret).pow(2)).mean()
    loss = surrogate + ppo.value_loss_coef * value_loss - ppo.entropy_coef * entropy.mean()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), ppo.max_grad_norm)
    torch.nn.utils.clip_grad_norm_(critic.parameters(), ppo.max_grad_norm)
    opt.step()
    return mk.pack_params(actor, critic).numpy(), kl


ONE_STEP_CONFIGS = [
    # the configuration whose free-running replay exceeds verify_megakernel.py's tolerance
    pytest.param(4096, None, 5, 3, id="4096_envs_seed_5_after_3_iters"),
    pytest.param(20000, None, 3, 2, id="20000_envs_global_after_2_iters"),
    pytest.param(50, 100, 3, 2, id="50_envs_on_100_ctas_after_2_iters"),
    pytest.param(1, None, 3, 1, id="1_env_after_1_iter"),
]


@pytest.mark.parametrize(("num_envs", "num_ctas", "seed", "warmup"), ONE_STEP_CONFIGS)
def test_every_optimizer_step_matches_rsl_rl_from_the_kernel_state(instrumented, num_envs, num_ctas, seed, warmup):
    """Replays step q of one iteration from the kernel's state after step q-1 and compares with its state after q.

    The step uses the kernel's learning rate; the KL-adaptive decision is checked separately for q >= 1 (the first
    mini-batch is covered by ``test_first_minibatch_kl_is_zero_so_the_learning_rate_is_kept``).
    """
    if num_ctas and num_ctas > _num_sms():
        pytest.skip(f"{num_ctas} CTAs need {num_ctas} SMs")
    trainer = mk.SramPPOCartpole(num_envs=num_envs, seed=seed, num_ctas=num_ctas, debug=True)
    ppo, plan = trainer.ppo, trainer.plan
    if warmup:
        trainer.train(warmup)
    start = [
        trainer.params.cpu().numpy().copy(),
        trainer.adam_m.cpu().numpy().copy(),
        trainer.adam_v.cpu().numpy().copy(),
    ]
    lr_prev, adam_t0 = float(trainer.scal[0].item()), int(trainer.scal[1].item())
    sigma_old = float(np.clip(start[0][mk.OFF_STD], 1e-6, 1e6))
    trainer.dbg.zero_()
    trainer.train(1)
    n_steps = ppo.num_learning_epochs * ppo.num_mini_batches
    record = _read_device_array(trainer, "g_step_dump", np.float32).reshape(2, -1, STEP_RECORD)[:, :n_steps]
    if plan.num_ctas > 1:
        assert np.array_equal(record[0], record[1]), "CTA 0 and the last CTA hold different parameter replicas"
    dump = _rollout_dump(trainer)
    actor, critic = mk.rsl_rl_models(ppo, 0)
    pad = mk.N_PARAMS_PAD
    state = start
    for q in range(n_steps):
        mb = q % ppo.num_mini_batches
        batch = np.concatenate(
            [
                st[:, perm[mb * (n // ppo.num_mini_batches) : (mb + 1) * (n // ppo.num_mini_batches)]]
                for n, st, perm in dump
            ],
            axis=1,
        )
        k_params, k_lr, k_kl = record[0, q, :pad], float(record[0, q, 3 * pad]), float(record[0, q, 3 * pad + 1])
        ref, kl = _replay_one_step(ppo, actor, critic, state, adam_t0 + q, batch, sigma_old, k_lr)
        err = float(np.abs(k_params[: mk.N_PARAMS] - ref[: mk.N_PARAMS]).max())
        tol = 1e-6 * max(1.0, float(np.abs(ref[: mk.N_PARAMS]).max()))
        assert err <= tol, f"optimizer step {q}: max |param error| {err:.3e} > {tol:.1e}"
        # torch evaluates 0.5 * (var_ratio + t1 - 1 - log(var_ratio)) in float32, whose cancellation leaves ~1e-8
        # absolute error; the kernel's 0.5 * ((vr - 1 - log vr) + t1) is more accurate for tiny KL
        assert abs(k_kl - kl) <= 1e-7 + 1e-4 * abs(kl), f"step {q}: kernel KL {k_kl:.6e}, RSL-RL {kl:.6e}"
        # skip the decision where it is set by rounding: next to a threshold, or `kl > 0` for a KL at torch's noise
        near_threshold = min(abs(kl - 2 * ppo.desired_kl), abs(kl - ppo.desired_kl / 2)) < 1e-4 * ppo.desired_kl
        if q > 0 and not near_threshold and abs(kl) > 1e-6:
            expected_lr = _adapt(ppo, lr_prev, kl)
            assert abs(k_lr - expected_lr) <= 1e-6 * expected_lr, f"step {q}: lr {k_lr:.6e}, RSL-RL {expected_lr:.6e}"
        lr_prev = k_lr
        state = [record[0, q, :pad], record[0, q, pad : 2 * pad], record[0, q, 2 * pad : 3 * pad]]


# ------------------------------------------------------------------------------------------------------------------
# advantage normalization barrier
# ------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("num_envs", "storage"),
    [
        pytest.param(64, "smem", id="64_envs_smem"),
        pytest.param(64, "global", id="64_envs_global"),
        pytest.param(100, "global", id="100_envs_global"),
        pytest.param(4096, "global", id="4096_envs_global"),
    ],
)
def test_debug_dump_holds_normalized_advantages(num_envs, storage):
    trainer = mk.SramPPOCartpole(num_envs=num_envs, seed=3, storage=storage, debug=True)
    trainer.train(1)
    dump = _rollout_dump(trainer)
    raw = np.concatenate([st[F_RET] - st[F_VAL] for _, st, _ in dump])
    adv = np.concatenate([st[F_ADV] for _, st, _ in dump])
    normalized = (raw - raw.mean()) / (raw.std(ddof=1) + 1e-8)
    assert np.abs(adv - normalized).max() < 1e-4


@pytest.mark.parametrize(
    ("num_envs", "storage"),
    [
        pytest.param(4096, "smem", id="4096_envs_smem"),
        pytest.param(20000, "global", id="20000_envs_global"),
        pytest.param(262144, "global", id="262144_envs_global"),
        pytest.param("max_global", "global", id="largest_launch"),
    ],
)
def test_first_minibatch_gathers_normalized_advantages(instrumented, num_envs, storage):
    """No warp may gather an advantage before the warp that normalizes it has stored it.

    Regression test for ``ADV_BARRIER_NOTE``: counts the gathers that read a value different from what a
    barrier-ordered read returns. Before the fix it fired a few times per 20 iterations at 262144 and 317720
    environments on an RTX PRO 6000.
    """
    trainer = mk.SramPPOCartpole(num_envs=_resolve(num_envs), seed=11, storage=storage)
    trainer.train(5)
    stale, gathered = _read_device_array(trainer, "g_stale_adv", np.uint32).tolist()
    assert gathered == 5 * trainer.plan.mb_global
    assert stale == 0, f"{stale} of {gathered} first-mini-batch advantages were gathered before normalization"


@pytest.mark.parametrize(
    "num_envs",
    [
        pytest.param(20000, id="20000_envs_global"),
        pytest.param(262144, id="262144_envs_global"),
    ],
)
def test_repeated_runs_are_bitwise_identical_with_global_rollout_storage(num_envs):
    """The production kernel, same seed, several runs: every run must end in the same state."""
    states = []
    for _ in range(4):
        trainer = mk.SramPPOCartpole(num_envs=num_envs, seed=0, storage="global")
        trainer.train(8)
        states.append(_snapshot(trainer))
        del trainer
        torch.cuda.empty_cache()
    differing = [_differing(states[0], s) for s in states[1:]]
    assert not any(differing), f"runs 1-3 differ from run 0 in {differing}"


# ------------------------------------------------------------------------------------------------------------------
# KL-adaptive learning rate at the first mini-batch
# ------------------------------------------------------------------------------------------------------------------
def _std_with_inexact_reciprocal(start: float = 0.7) -> float:
    """First float32 std >= start with std * fl(1 / std) != 1 (float32 multiplication and division are exact IEEE)."""
    x = np.float32(start)
    while x * (np.float32(1.0) / x) == np.float32(1.0):
        x = np.nextafter(x, np.float32(np.inf))
    return float(x)


@pytest.mark.parametrize(
    "std",
    [
        pytest.param(0.75, id="std_with_exact_reciprocal"),
        pytest.param("inexact", id="std_with_inexact_reciprocal"),
    ],
)
def test_first_minibatch_kl_is_zero_so_the_learning_rate_is_kept(std):
    """With one mini-batch per iteration RSL-RL evaluates the KL on the rollout policy itself: it is exactly 0, and the
    learning rate stays where it was."""
    std = _std_with_inexact_reciprocal() if std == "inexact" else std
    x = np.float32(std)
    assert (x * (np.float32(1.0) / x) == np.float32(1.0)) == (std == 0.75)
    ppo = PPOConfig(num_learning_epochs=1, num_mini_batches=1)
    trainer = mk.SramPPOCartpole(ppo=ppo, num_envs=512, seed=0)
    trainer.params[mk.OFF_STD] = std
    trainer.train(1)
    assert trainer.scal[0].item() == ppo.learning_rate


# ------------------------------------------------------------------------------------------------------------------
# host-side bookkeeping
# ------------------------------------------------------------------------------------------------------------------
def test_reset_restarts_training_bitwise():
    fresh = mk.SramPPOCartpole(num_envs=1000, seed=9)
    fresh.train(5)
    trainer = mk.SramPPOCartpole(num_envs=1000, seed=9)
    trainer.train(5)
    trainer.reset()
    trainer.train(5)
    assert trainer.iteration == 5
    assert not _differing(_snapshot(trainer), _snapshot(fresh))


def test_reset_restarts_the_episode_statistics():
    expected = mk.SramPPOCartpole(num_envs=1000, seed=9).train(5)["stats"]
    trainer = mk.SramPPOCartpole(num_envs=1000, seed=9)
    trainer.train(5)
    trainer.reset()
    actual = trainer.train(5)["stats"]
    for name in ("episodes", "episode_length_sum", "terminated", "timed_out"):
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
    np.testing.assert_allclose(actual["mean_step_reward"], expected["mean_step_reward"], rtol=1e-5)


def test_train_zero_iterations_is_a_no_op():
    trainer = mk.SramPPOCartpole(num_envs=_num_sms(), seed=1)
    before = _snapshot(trainer)
    result = trainer.train(0)
    assert result["iterations"] == 0 and result["launches"] == 0
    assert trainer.iteration == 0
    assert not _differing(before, _snapshot(trainer))

    # Invalid launch limits previously decreased the completed-iteration counter forever.
    for iterations, per_launch in ((-1, None), (1, -1), (1, 0)):
        with pytest.raises(ValueError):
            trainer.train(iterations, iters_per_launch=per_launch)
    assert not _differing(before, _snapshot(trainer))


@pytest.mark.parametrize(
    ("envs", "ctas", "storage", "steps", "batches"),
    [
        (0, None, "auto", 16, 4),
        (1, 0, "auto", 16, 4),
        (1, None, "unknown", 16, 4),
        (1, None, "auto", 0, 4),
        (1, None, "auto", 16, 0),
        (1, None, "auto", 1, 4),
    ],
)
def test_invalid_launch_configuration_fails_before_cuda_launch(envs, ctas, storage, steps, batches):
    ppo = replace(PPOConfig(), num_steps_per_env=steps, num_mini_batches=batches)
    with pytest.raises(ValueError):
        mk.plan_launch(envs, ppo, ctas, storage, 188)


def test_profile_marks_stay_inside_their_buffer():
    trainer = mk.SramPPOCartpole(num_envs=2, seed=1, profile=True)  # PROF_CTA = 1 needs two CTAs
    n = trainer.dbg.numel()
    guarded = torch.full((n + 4096,), -7.0, device=trainer.device)
    trainer.dbg = guarded[:n]
    trainer.train(4096 + 8, iters_per_launch=1024)
    torch.cuda.synchronize()
    overwritten = int((guarded[n:] != -7.0).sum())
    assert overwritten == 0, f"{overwritten} floats written past the {n}-float profile buffer"


# ------------------------------------------------------------------------------------------------------------------
# random streams
# ------------------------------------------------------------------------------------------------------------------
def _noise_and_resets(trainer: mk.SramPPOCartpole, sigma: float):
    """Action noise [T, num_envs], reset states and permutations of the last debug launch."""
    steps, plan = trainer.ppo.num_steps_per_env, trainer.plan
    eps = np.zeros((steps, plan.num_envs))
    resets, perms = [], []
    for c, (n, st, perm) in enumerate(_rollout_dump(trainer)):
        e0 = (c * plan.num_envs) // plan.num_ctas
        e = n // steps
        eps[:, e0 : e0 + e] = ((st[F_ACT] - st[F_MU]) / sigma).reshape(steps, e)
        obs = st[0:4].reshape(4, steps, e).transpose(1, 2, 0)
        done = st[F_DONE].reshape(steps, e) > 0.5
        resets.append(obs[1:][done[:-1]])
        perms.append(perm)
    return eps, np.concatenate(resets), perms


def test_action_noise_reset_and_permutation_streams_are_independent():
    trainer = mk.SramPPOCartpole(num_envs=4096, seed=3, debug=True)
    sigma0 = float(trainer.params[mk.OFF_STD].clamp(1e-6, 1e6))
    trainer.train(1)
    eps0, resets0, perms0 = _noise_and_resets(trainer, sigma0)
    sigma1 = float(trainer.params[mk.OFF_STD].clamp(1e-6, 1e6))
    trainer.dbg.zero_()
    trainer.train(1)
    eps1, resets1, perms1 = _noise_and_resets(trainer, sigma1)

    for eps in (eps0, eps1):
        assert abs(eps.mean()) < 0.02 and abs(eps.std() - 1.0) < 0.02
        # a counter collision would repeat whole streams; float32 rounding leaves a handful of chance duplicates
        assert np.unique(eps).size > 0.99 * eps.size
        assert abs(np.corrcoef(eps[:-1].ravel(), eps[1:].ravel())[0, 1]) < 0.03  # consecutive steps
        assert abs(np.corrcoef(eps[:, :-1].ravel(), eps[:, 1:].ravel())[0, 1]) < 0.03  # neighbouring environments
    assert abs(np.corrcoef(eps0.ravel(), eps1.ravel())[0, 1]) < 0.03  # consecutive iterations
    for resets in (resets0, resets1):
        assert resets.shape[0] > 100
        assert np.unique(resets, axis=0).shape[0] == resets.shape[0]
    # every CTA shuffles with its own key, and the key changes every iteration
    same_size = [c for c in range(len(perms0)) if perms0[c].size == perms0[0].size]
    assert len(same_size) > 1
    assert not np.array_equal(perms0[same_size[0]], perms0[same_size[1]])
    assert not np.array_equal(perms0[0], perms1[0])


# ------------------------------------------------------------------------------------------------------------------
# learning
# ------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("num_envs", "num_ctas"),
    [
        pytest.param(1000, None, id="1000_envs"),
        pytest.param(20000, None, id="20000_envs_global"),
        pytest.param(4096, 64, id="4096_envs_on_64_ctas"),
    ],
)
def test_learns_cartpole_at_odd_launch_configurations(num_envs, num_ctas):
    """150 iterations reach the ~4.9 episode return of the 4096-environment default (the maximum is about 5)."""
    if num_ctas and num_ctas > _num_sms():
        pytest.skip(f"{num_ctas} CTAs need {num_ctas} SMs")
    trainer = mk.SramPPOCartpole(num_envs=num_envs, seed=42, num_ctas=num_ctas)
    stats = trainer.train(150)["stats"]
    assert torch.isfinite(trainer.params).all()
    assert stats["mean_episode_return"][-10:].mean() > 4.8
    # the kernel does not model the cart joint limit; terminations at |x| > 3 m must keep the cart away from it
    assert stats["cart_limit_hits"].sum() == 0
    assert math.isfinite(stats["value_loss"][-1])

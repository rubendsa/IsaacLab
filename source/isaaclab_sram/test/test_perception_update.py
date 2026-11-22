# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare the complete fused visual learner with independent native RSL-RL PPO.

Kernel-level tests cannot catch detached CNNs, incorrect loss wiring, optimizer
ordering or omitted running-normalizer updates. This exercises the production
algorithm boundary with the native RGB64 network and all 40 optimizer steps.
The input rollout is synthetic; this is numerical integration coverage, not a
test of rendering, lift success or training speed.
"""

from __future__ import annotations

import copy
import sys
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils import to_dict

from isaaclab_rl.entrypoints.common import scoped_torch_backend_flags
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.core.lift.config.kuka_allegro.agents.rsl_rl_ppo_cfg import (
    KukaAllegroSingleCameraPPORunnerCfg,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _observations(num_envs: int, device: str) -> TensorDict:
    """Native RGB64 observation groups, with nontrivial normalization statistics."""
    return TensorDict(
        {
            "policy": torch.randn(num_envs, 170, device=device) * 0.7 + 0.3,
            "proprio": torch.randn(num_envs, 465, device=device) * 0.4 - 0.2,
            "perception": torch.randn(num_envs, 960, device=device) * 0.5 + 0.4,
            "base_image": torch.randint(0, 256, (num_envs, 3, 64, 64), dtype=torch.uint8, device=device),
        },
        batch_size=[num_envs],
        device=device,
    )


def _config() -> dict:
    cfg = to_dict(handle_deprecated_rsl_rl_cfg(KukaAllegroSingleCameraPPORunnerCfg(), version("rsl-rl-lib")))
    cfg["multi_gpu"] = None
    cfg["torch_compile_mode"] = None
    return cfg


def _collect_rollout(algorithm, num_envs: int, device: str, offset: float = 0.0) -> TensorDict:
    observations = _observations(num_envs, device)
    with torch.inference_mode():
        for step in range(algorithm.storage.num_transitions_per_env):
            algorithm.act(observations)
            observations = _observations(num_envs, device)
            # A different distribution on later rollouts makes stale captured
            # normalizer buffers observable in subsequent parameter updates.
            observations["policy"].add_(offset)
            observations["proprio"].mul_(1.0 + offset)
            rewards = torch.randn(num_envs, device=device) * 0.2
            dones = (torch.arange(num_envs, device=device) + step) % 11 == 0
            time_outs = dones & (torch.arange(num_envs, device=device) % 2 == 0)
            algorithm.process_env_step(observations, rewards, dones.long(), {"time_outs": time_outs})
    return observations


def _assert_algorithm_states_close(reference, actual) -> None:
    for role in ("actor", "critic"):
        reference_model, actual_model = getattr(reference, role), getattr(actual, role)
        # Include learned temperatures and running statistics; weight-only
        # comparisons would miss an omitted or stale normalizer update.
        expected_state = reference_model.state_dict()
        actual_state = actual_model.state_dict()
        assert actual_state.keys() == expected_state.keys()
        for name, expected in expected_state.items():
            torch.testing.assert_close(actual_state[name], expected, rtol=3e-4, atol=2e-5, msg=f"{role}.{name}")
        for expected, observed in zip(reference_model.parameters(), actual_model.parameters(), strict=True):
            expected_optimizer = reference.optimizer.state[expected]
            actual_optimizer = actual.optimizer.state[observed]
            for name in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    actual_optimizer[name],
                    expected_optimizer[name],
                    rtol=5e-4,
                    atol=2e-6 if name == "exp_avg" else 2e-8,
                    msg=f"{role} Adam {name}",
                )
            assert actual_optimizer["step"].item() == expected_optimizer["step"].item()


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_native_camera_ppo_update_matches_stock_parameters_moments_and_normalizers(device):
    """The fused learner preserves a native PPO update, including visual learning."""
    cfg = _config()
    env = SimpleNamespace(num_envs=8, num_actions=23)
    device_index = torch.device(device).index
    with (
        torch.random.fork_rng(devices=[device_index]),
        scoped_torch_backend_flags(
            cuda_matmul_allow_tf32=False, cudnn_allow_tf32=False, cudnn_deterministic=True, cudnn_benchmark=False
        ),
    ):
        torch.manual_seed(29)
        observations = _observations(env.num_envs, device)
        stock = PPO.construct_algorithm(observations, env, copy.deepcopy(cfg), device)
        fused_cfg = copy.deepcopy(cfg)
        fused_cfg["algorithm"]["class_name"] = "isaaclab_sram.perception:SramPerceptionPPO"
        fused = PPO.construct_algorithm(observations, env, fused_cfg, device)
        for reference_model, actual_model in ((stock.actor, fused.actor), (stock.critic, fused.critic)):
            reference_model.update_normalization(observations)
            actual_model.load_state_dict(reference_model.state_dict())
        stock.train_mode()
        fused.train_mode()
        initial_actor = {name: parameter.detach().clone() for name, parameter in stock.actor.named_parameters()}

        # Collect once through native PPO, then clone the complete rollout. Both
        # learners receive exactly the same transitions, old policy and targets.
        observations = _collect_rollout(stock, env.num_envs, device)
        fused.storage = copy.deepcopy(stock.storage)
        with torch.inference_mode():
            stock.compute_returns(observations)
            fused.compute_returns(observations)
        torch.testing.assert_close(fused.storage.returns, stock.storage.returns, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(fused.storage.advantages, stock.storage.advantages, rtol=2e-5, atol=2e-6)

        # Restore the same RNG immediately before each complete update, so the
        # native minibatch generator draws the same permutation and action RNG.
        torch.manual_seed(53)
        reference_losses = stock.update()
        torch.manual_seed(53)
        actual_losses = fused.update()
        for name, expected in reference_losses.items():
            assert actual_losses[name] == pytest.approx(expected, rel=3e-4, abs=2e-5), name

        _assert_algorithm_states_close(stock, fused)

        # Every convolution and the learned visual temperature must participate
        # in the update, not just the state MLP or the privileged critic.
        for name, parameter in fused.actor.named_parameters():
            if name.startswith(("cnns.", "softmaxes.")):
                assert torch.count_nonzero(parameter.detach() - initial_actor[name]).item() > 0, name


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize(
    "backend",
    [
        "isaaclab_sram.perception:SramGraphPerceptionPPO",
        "isaaclab_sram.persistent_perception:SramPersistentCriticPPO",
        "isaaclab_sram.persistent_full_perception:SramPersistentFullPerceptionPPO",
        "isaaclab_sram.unified_perception:SramUnifiedPerceptionPPO",
    ],
    ids=["cuda_graph", "persistent_critic", "persistent_full", "persistent_unified"],
)
def test_optimized_updates_match_eager_after_resume_and_normalizer_changes(device, backend):
    """Optimized updates preserve resumed Adam and consume later normalization."""
    cfg = _config()
    cfg["algorithm"]["class_name"] = "isaaclab_sram.perception:SramPerceptionPPO"
    unified = backend == "isaaclab_sram.unified_perception:SramUnifiedPerceptionPPO"
    env = SimpleNamespace(num_envs=16 if unified else 8, num_actions=23)
    with (
        torch.random.fork_rng(devices=[torch.device(device).index]),
        scoped_torch_backend_flags(
            cuda_matmul_allow_tf32=False, cudnn_allow_tf32=False, cudnn_deterministic=True, cudnn_benchmark=False
        ),
    ):
        torch.manual_seed(61)
        eager = PPO.construct_algorithm(_observations(env.num_envs, device), env, copy.deepcopy(cfg), device)
        eager.train_mode()
        last = _collect_rollout(eager, env.num_envs, device)
        with torch.inference_mode():
            eager.compute_returns(last)
        eager.update()

        cfg["algorithm"]["class_name"] = backend
        optimized = PPO.construct_algorithm(last, env, cfg, device)
        optimized.load(copy.deepcopy(eager.save()), load_cfg=None, strict=True)
        optimized.train_mode()
        for parameter in optimized.actor.parameters():
            state = optimized.optimizer.state[parameter]
            assert state["step"].item() > 0
            assert all(value.device == parameter.device for value in state.values())

        # The first update initializes compiled/captured state from nonzero Adam;
        # the second consumes a changed observation distribution and normalizer.
        # Unified additionally reloads before a third update, covering reused
        # scratch buffers as well as checkpoint-triggered reconstruction.
        for iteration in range(3 if unified else 2):
            if unified and iteration == 2:
                optimized.load(copy.deepcopy(optimized.save()), load_cfg=None, strict=True)
                assert optimized._unified is None
            torch.manual_seed(71 + iteration)
            last = _collect_rollout(eager, env.num_envs, device, offset=0.5 + iteration)
            optimized.storage = copy.deepcopy(eager.storage)
            with torch.inference_mode():
                eager.compute_returns(last)
                optimized.compute_returns(last)
            torch.manual_seed(81 + iteration)
            expected_losses = eager.update()
            expected_cpu_rng = torch.random.get_rng_state()
            expected_cuda_rng = torch.cuda.get_rng_state(device)
            torch.manual_seed(81 + iteration)
            actual_losses = optimized.update()
            if unified:
                assert optimized._unified.launch_count == (1 if iteration == 2 else iteration + 1)
            if backend != "isaaclab_sram.perception:SramGraphPerceptionPPO":
                # Persistent phases consume the same discarded action samples
                # as native PPO. Graph replay deliberately omits those draws.
                assert torch.equal(torch.random.get_rng_state(), expected_cpu_rng)
                assert torch.equal(torch.cuda.get_rng_state(device), expected_cuda_rng)
            for name, expected in expected_losses.items():
                assert actual_losses[name] == pytest.approx(expected, rel=3e-4, abs=2e-5), name
            _assert_algorithm_states_close(eager, optimized)

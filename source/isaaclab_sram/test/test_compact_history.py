# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify compact storage against histories produced by the actual observation manager.

The reference is the native manager's recorded snapshots, including its noise,
delay, flattening and reset implementation. These CPU contracts do not require
a simulator or duplicate the codec's reconstruction algorithm.
"""

from types import SimpleNamespace

import pytest
import torch
from isaaclab_sram.compact_history import CompactHistory, layouts_from_observation_manager, pack_history

from isaaclab.managers import ObservationGroupCfg, ObservationManager, ObservationTermCfg
from isaaclab.utils import noise
from isaaclab.utils.buffers import CircularBuffer

pytestmark = pytest.mark.unit


def _read_frame(env, name):
    return env.frames[name]


def _manager(order, *, native_sizes=False, concatenate=True):
    if native_sizes:
        shapes = {"quaternion": (4,), "target": (7,), "action": (23,)}
    else:
        shapes = {"matrix": (2, 3), "vector": (2,), "scalar": () if order == "time" else (1,)}
    env = SimpleNamespace(
        num_envs=64 if native_sizes else 5,
        device="cpu",
        sim=SimpleNamespace(is_playing=lambda: True),
        frames={},
    )
    groups = {}
    specs = {"policy": shapes}
    if native_sizes:
        specs.update(proprio={"joints": (46,), "tips": (42,), "contact": (5,)}, perception={"cloud": (64, 3)})
    for group_name, group_shapes in specs.items():
        group = ObservationGroupCfg(
            history_length=5 if order == "time" or native_sizes else None,
            history_order=order,
            concatenate_terms=concatenate,
            enable_corruption=not native_sizes,
        )
        for i, (name, shape) in enumerate(group_shapes.items()):
            key = f"{group_name}_{name}"
            env.frames[key] = torch.zeros((env.num_envs, *shape))
            term = ObservationTermCfg(
                func=_read_frame,
                params={"name": key},
                history_length=(5, 3, 0)[i % 3],
                noise=noise.UniformNoiseCfg(n_min=-0.13, n_max=0.17),
                scale=0.7,
                clip=(-3.0, 4.0),
                delay_min_lag=1 if i == 0 and not native_sizes else 0,
                delay_max_lag=2 if i == 0 and not native_sizes else 0,
            )
            setattr(group, name, term)
        groups[group_name] = group
    return env, ObservationManager(groups, env)


def _advance(env, manager, sequence):
    for i, (name, value) in enumerate(env.frames.items()):
        env.frames[name] = torch.randn_like(value) + sequence * 0.03 + i * 0.2
    return manager.compute(update_history=True)


@pytest.mark.parametrize("order", ["term", "time"])
def test_native_manager_snapshots_survive_resets_ordering_noise_delay_and_gather(order):
    """Reconstruct real manager outputs across internal and rollout-boundary resets."""
    torch.manual_seed(101)
    env, manager = _manager(order)
    layout = layouts_from_observation_manager(manager, ("policy",))["policy"]
    packed = CompactHistory(layout, 11, env.num_envs)
    for step in range(7):
        _advance(env, manager, step)

    for rollout in range(2):
        # Reusing storage must take a fresh prefix, rather than assuming the
        # previous rollout's last frame survives resets between collections.
        manager.reset(torch.tensor([1, 3]))
        packed.clear()
        snapshots, masks = [], []
        for step in range(11):
            reset = torch.zeros(env.num_envs, dtype=torch.bool)
            if step == 0:
                reset[[1, 3]] = True
            elif step in (3, 7, 8):
                ids = torch.tensor([0, 4]) if step != 7 else torch.arange(env.num_envs)
                manager.reset(ids)
                reset[ids] = True
            observations = _advance(env, manager, 7 + rollout * 11 + step)["policy"]
            snapshots.append(observations.clone())
            masks.append(reset)
            packed.append(observations)  # Exercise inferred resets.
            observations.zero_()  # Compact storage must own its frames.
        expected = torch.stack(snapshots)
        torch.testing.assert_close(packed.reconstruct(), expected, rtol=0, atol=0)
        # Non-monotone, duplicate, first and last indices protect native RSL-RL
        # flattened minibatch semantics, not just sequential reconstruction.
        indices = torch.tensor([54, 0, 7, 31, 31, 22, 1, 49])
        torch.testing.assert_close(packed.gather(indices), expected.flatten(0, 1)[indices], rtol=0, atol=0)
        assert packed.gather(torch.empty(0, dtype=torch.int64)).shape == (0, layout.observation_size)
        explicit = pack_history(expected, layout, resets=torch.stack(masks))
        torch.testing.assert_close(explicit.reconstruct(), expected, rtol=0, atol=0)
        with pytest.raises((IndexError, RuntimeError)):
            packed.gather(torch.tensor([-1]))
        with pytest.raises((IndexError, RuntimeError)):
            packed.gather(torch.tensor([55]))


def test_native64_32_step_storage_reduction_uses_resolved_manager_layout():
    """The native vector dimensions retain only four prefix frames plus 32 samples."""
    env, manager = _manager("term", native_sizes=True)
    layouts = layouts_from_observation_manager(manager)
    packed = {name: CompactHistory(layout, 32, 64) for name, layout in layouts.items()}
    native = {name: [] for name in packed}
    for step in range(32):
        if step in (4, 16):
            manager.reset(torch.tensor([1, 17, 38]))
        for name, values in _advance(env, manager, step).items():
            packed[name].append(values)
            native[name].append(values)
    for name, buffer in packed.items():
        torch.testing.assert_close(buffer.reconstruct(), torch.stack(native[name]), rtol=0, atol=0)
    # Sizes come from actual manager-created groups with native dimensions.
    assert sum(buffer.frame_bytes for buffer in packed.values()) == 36 * 64 * 319 * 4
    assert sum(torch.stack(values).numel() * 4 for values in native.values()) == 32 * 64 * 1595 * 4
    assert sum(buffer.storage_bytes for buffer in packed.values()) < 3 * 1024**2


def test_illegal_history_and_misaligned_reset_mask_fail_instead_of_corrupting_observations():
    """Skipped history updates or a done mask shifted by one step cannot silently pack."""
    env, manager = _manager("time")
    layout = layouts_from_observation_manager(manager, ("policy",))["policy"]
    snapshots = [_advance(env, manager, step)["policy"] for step in range(5)]
    dense = torch.stack(snapshots)
    invalid = dense.clone()
    invalid[3, 0, 0] += 123.0  # Mutate an old frame, not the newly sampled frame.
    with pytest.raises(ValueError, match="cannot pack losslessly"):
        pack_history(invalid, layout)
    wrong_resets = torch.zeros(5, env.num_envs, dtype=torch.bool)
    wrong_resets[3, 0] = True
    with pytest.raises(ValueError, match="cannot pack losslessly"):
        pack_history(dense, layout, resets=wrong_resets)
    # Unsupported non-flat manager groups fail at discovery before training.
    _, unsupported = _manager("time", concatenate=False)
    with pytest.raises(ValueError, match="concatenated flat"):
        layouts_from_observation_manager(unsupported, ("policy",))


def test_bitwise_history_preserves_nan_payloads_and_signed_zero():
    """Lossless means original float representations, including non-finite payloads."""
    env, manager = _manager("time")
    layout = layouts_from_observation_manager(manager, ("policy",))["policy"]
    # Use the real circular buffer as the oracle for payloads which manager noise
    # intentionally transforms. One buffer per resolved term supplies snapshots.
    buffers = [CircularBuffer(term.history_length, 2, "cpu") for term in layout.terms]
    records = []
    for step in range(6):
        terms = []
        for term, buffer in zip(layout.terms, buffers, strict=True):
            bits = torch.tensor([0x7FC00001, -2147483648, 0, 0x7FC00002, 0x3F800000, -1082130432])
            values = bits.to(torch.int32).view(torch.float32).repeat(2 * term.frame_size)
            values = values[step : step + 2 * term.frame_size].reshape(2, term.frame_size)
            if step == 3:
                buffer.reset([1])
            buffer.append(values)
            terms.append(buffer.buffer.clone())
        # Native time-major concatenation, independently of codec join/split.
        records.append(torch.cat(terms, dim=-1).flatten(1))
    dense = torch.stack(records)
    packed = pack_history(dense, layout)
    assert torch.equal(packed.reconstruct().view(torch.int32), dense.view(torch.int32))


def test_native_rollout_storage_adapter_matches_dense_minibatches_after_clear_and_reset():
    """The actual PPO storage/generator sees identical images, histories and auxiliary fields."""
    from isaaclab_sram.compact_storage import compact_rollout_observations
    from rsl_rl.storage import RolloutStorage
    from tensordict import TensorDict

    torch.manual_seed(303)
    env, manager = _manager("term")
    layouts = layouts_from_observation_manager(manager, ("policy",))
    initial = _advance(env, manager, 0)
    initial["base_image"] = torch.zeros(env.num_envs, 3, 64, 64, dtype=torch.uint8)
    observations = TensorDict(initial, batch_size=[env.num_envs])
    dense = RolloutStorage("rl", env.num_envs, 8, observations, [23])
    compact = RolloutStorage("rl", env.num_envs, 8, observations, [23])
    adapter = compact_rollout_observations(compact, layouts)
    for rollout in range(2):
        dense.clear()
        compact.clear()
        manager.reset(torch.tensor([0, 4]))
        for step in range(8):
            values = _advance(env, manager, rollout * 8 + step)
            values["base_image"] = torch.randint(0, 256, (env.num_envs, 3, 64, 64), dtype=torch.uint8)
            transition = RolloutStorage.Transition()
            transition.observations = TensorDict(values, batch_size=[env.num_envs])
            transition.actions = torch.randn(env.num_envs, 23)
            transition.rewards = torch.randn(env.num_envs)
            transition.dones = torch.tensor([step in (2, 5), False, False, step == 2, False])
            transition.values = torch.randn(env.num_envs, 1)
            transition.actions_log_prob = torch.randn(env.num_envs)
            transition.distribution_params = (torch.randn(env.num_envs, 23), torch.rand(env.num_envs, 23) + 0.1)
            dense.add_transition(transition)
            compact.add_transition(transition)
            manager.reset(transition.dones.nonzero().flatten())
        for name in ("returns", "advantages"):
            value = torch.randn_like(getattr(dense, name))
            getattr(dense, name).copy_(value)
            getattr(compact, name).copy_(value)
        torch.manual_seed(707 + rollout)
        expected_batches = list(dense.mini_batch_generator(8, 5))
        torch.manual_seed(707 + rollout)
        actual_batches = list(compact.mini_batch_generator(8, 5))
        assert len(actual_batches) == 40
        for expected, actual in zip(expected_batches, actual_batches, strict=True):
            assert isinstance(actual.observations, TensorDict)
            assert actual.observations["base_image"].dtype == torch.uint8
            for name in expected.observations.keys():
                torch.testing.assert_close(actual.observations[name], expected.observations[name], rtol=0, atol=0)
            for name in ("actions", "values", "returns", "advantages", "old_actions_log_prob"):
                torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
            for expected_param, actual_param in zip(
                expected.old_distribution_params, actual.old_distribution_params, strict=True
            ):
                torch.testing.assert_close(actual_param, expected_param, rtol=0, atol=0)
        for name in observations.keys():
            torch.testing.assert_close(
                adapter.flatten(0, 1)[name], dense.observations[name].flatten(0, 1), rtol=0, atol=0
            )
        assert adapter.storage_bytes < adapter.dense_bytes

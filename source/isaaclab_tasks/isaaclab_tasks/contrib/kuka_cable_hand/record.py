# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record policy rollouts of the Kuka cable-hand lift task for offline rendering.

Runs the deterministic policy and stores, for every policy step, the world pose of every body, the cable
pieces of every tendon with their tension, the lift target and the episode boundaries. A renderer (for
example ``render_ovrtx.py``) can then replay the motion with any camera, look or renderer without
simulating again.

Example::

    python -m isaaclab_tasks.contrib.kuka_cable_hand.record --checkpoint pretrained/model_1499.pt \\
        --num_envs 4 --episodes 1 --out rollout.npz
"""

from __future__ import annotations

import argparse
import dataclasses
import json


def record(checkpoint: str, out: str, num_envs: int, episodes: int, seed: int, device: str) -> dict:
    import numpy as np
    import torch
    from newton.examples.cable import cable_hand as ch

    from .env import TABLE_TOP, KukaCableHandLiftCfg, KukaCableHandLiftEnv
    from .play import _load_policy

    cfg = KukaCableHandLiftCfg(num_envs=num_envs, device=device, seed=seed)
    env = KukaCableHandLiftEnv(cfg)
    policy = _load_policy(env, checkpoint, device)
    lay, spec, model = env.layout, env.spec, env.model
    per_world = len(lay.tendons)
    links_per_world = model.tendon_link_count // num_envs
    segs_per_world = model.tendon_segment_count // num_envs
    keys = list(lay.tendons)
    roles = {
        key: spec.digits[d].cable(c).role
        for key, (d, c) in zip(lay.hand.actuators, lay.hand.actuator_cables, strict=True)
    }
    world_tendons = [
        [
            dataclasses.replace(
                lay.tendons[key],
                tendon=lay.tendons[key].tendon + w * per_world,
                link_start=lay.tendons[key].link_start + w * links_per_world,
                seg_start=lay.tendons[key].seg_start + w * segs_per_world,
            )
            for key in keys
        ]
        for w in range(num_envs)
    ]
    steps = episodes * env.max_episode_length
    body_q = np.zeros((steps, num_envs, env.B, 7), dtype=np.float32)
    targets = np.zeros((steps, num_envs, 3), dtype=np.float32)
    dones = np.zeros((steps, num_envs), dtype=bool)
    lifted = np.zeros((steps, num_envs), dtype=bool)
    success = np.zeros((steps, num_envs), dtype=bool)
    piece_start, piece_end, piece_tension, piece_world, piece_cable, piece_offset = [], [], [], [], [], [0]
    obs = env.get_observations()
    with torch.inference_mode():
        for i in range(steps):
            obs, _, done, _ = env.step(policy(obs))
            body_q[i] = env.body_q.cpu().numpy()
            targets[i] = env.target_pos.cpu().numpy()
            dones[i] = done.cpu().numpy()
            lifted[i] = env._step_lifted.cpu().numpy()
            success[i] = env._step_success.cpu().numpy()
            count = 0
            for w in range(num_envs):
                for c, tendon in enumerate(world_tendons[w]):
                    s, e, t = ch.tendon_cable_pieces(env.solver, model, env.state_0, tendon)
                    piece_start.append(s.astype(np.float32))
                    piece_end.append(e.astype(np.float32))
                    piece_tension.append(t.astype(np.float32))
                    piece_world.append(np.full(len(s), w, dtype=np.int16))
                    piece_cable.append(np.full(len(s), c, dtype=np.int16))
                    count += len(s)
            piece_offset.append(piece_offset[-1] + count)
    hand_bodies = {key: int(index) for key, index in lay.hand.bodies.items()}
    meta = {
        "checkpoint": checkpoint,
        "num_envs": num_envs,
        "episodes": episodes,
        "seed": seed,
        "step_dt": env.step_dt,
        "bodies_per_world": env.B,
        "body_labels": [str(label) for label in list(model.body_label)[: env.B]],
        "arm_links": [int(b) for b in lay.arm_links],
        "hand_bodies": hand_bodies,
        "object_body": int(lay.object_body),
        "object_half_extent": cfg.object_half_extent,
        "table_center": list(cfg.table_center),
        "table_half_extents": list(cfg.table_half_extents),
        "table_top": TABLE_TOP,
        "cable_keys": keys,
        "cable_roles": [roles[k] for k in keys],
        "cable_radius": 0.5 * spec.cable.diameter,
        "tension_max": spec.cable.tension_max,
    }
    np.savez_compressed(
        out,
        body_q=body_q,
        targets=targets,
        dones=dones,
        lifted=lifted,
        success=success,
        piece_start=np.concatenate(piece_start) if piece_start else np.zeros((0, 3), np.float32),
        piece_end=np.concatenate(piece_end) if piece_end else np.zeros((0, 3), np.float32),
        piece_tension=np.concatenate(piece_tension) if piece_tension else np.zeros(0, np.float32),
        piece_world=np.concatenate(piece_world) if piece_world else np.zeros(0, np.int16),
        piece_cable=np.concatenate(piece_cable) if piece_cable else np.zeros(0, np.int16),
        piece_offset=np.asarray(piece_offset, dtype=np.int64),
        meta=json.dumps(meta),
    )
    env.close()
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True, help="Output .npz path.")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = args.out if args.out.endswith(".npz") else args.out + ".npz"  # numpy appends the suffix anyway
    meta = record(args.checkpoint, out, args.num_envs, args.episodes, args.seed, args.device)
    print(f"recorded {meta['episodes']} episode(s) of {meta['num_envs']} environment(s) to {out}")


if __name__ == "__main__":
    main()

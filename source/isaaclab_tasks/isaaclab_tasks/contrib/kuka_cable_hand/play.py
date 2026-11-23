# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained Kuka cable-hand lift policy and optionally record a video.

Evaluation runs one episode in every environment with the deterministic policy and reports how many
episodes lifted the cube and brought it to the target. The video renders a few environments with
the Newton OpenGL viewer (headless) and writes an MP4 with ``ffmpeg``. Cables are drawn as cylinders,
colored by role (curl / flexor cyan-blue, knuckle magenta) and brightened by tension.

Example::

    python -m isaaclab_tasks.contrib.kuka_cable_hand.play --checkpoint logs/.../model_1999.pt \\
        --eval_envs 1024 --video kuka_cable_hand.mp4 --video_envs 4
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import re
import subprocess


def _latest_checkpoint(path: str) -> str:
    if os.path.isfile(path):
        return path
    files = glob.glob(os.path.join(path, "model_*.pt"))
    if not files:
        raise FileNotFoundError(f"no model_*.pt in {path}")
    return max(files, key=lambda f: int(re.findall(r"model_(\d+)\.pt", f)[0]))


def _load_policy(env, checkpoint: str, device: str):
    from rsl_rl.runners import OnPolicyRunner

    from .agents import ppo_runner_cfg

    runner = OnPolicyRunner(env, ppo_runner_cfg(), log_dir=None, device=device)
    runner.load(checkpoint)
    return runner.get_inference_policy(device=device)


def evaluate(checkpoint: str, num_envs: int, seed: int, device: str) -> dict:
    """One deterministic episode per environment; returns the success statistics."""
    import torch

    from .env import TABLE_TOP, KukaCableHandLiftCfg, KukaCableHandLiftEnv

    cfg = KukaCableHandLiftCfg(num_envs=num_envs, device=device, seed=seed)
    env = KukaCableHandLiftEnv(cfg)
    policy = _load_policy(env, checkpoint, device)
    obs = env.get_observations()
    active = torch.ones(num_envs, dtype=torch.bool, device=device)
    lifted_any = torch.zeros_like(active)
    success_any = torch.zeros_like(active)
    success_final = torch.zeros_like(active)
    held_final = torch.zeros_like(active)
    max_height = torch.zeros(num_envs, device=device)
    final_dist = torch.zeros(num_envs, device=device)
    early = torch.zeros_like(active)
    reasons = {}
    rest_z = TABLE_TOP + cfg.object_half_extent
    with torch.inference_mode():
        for _ in range(env.max_episode_length):
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
            # the env has already reset finished worlds: read their terminal values from the step's state buffers
            lifted_any |= active & env._step_lifted
            success_any |= active & env._step_success
            max_height = torch.where(active, torch.maximum(max_height, env._step_height - rest_z), max_height)
            ended = active & dones
            if ended.any():
                success_final |= ended & env._step_success
                held_final |= ended & env._step_lifted
                final_dist = torch.where(ended, env._step_goal_dist, final_dist)
                early |= ended & ~extras["time_outs"]
                for key, flag in env._term_reasons.items():
                    reasons[key] = reasons.get(key, 0) + int((ended & flag).sum())
            active &= ~dones
            if not active.any():
                break
    n = float(num_envs)
    stats = {
        "checkpoint": checkpoint,
        "episodes": num_envs,
        "lifted_any": float(lifted_any.sum()) / n,
        "success_any": float(success_any.sum()) / n,
        "success_at_end": float(success_final.sum()) / n,
        "held_at_end": float(held_final.sum()) / n,
        "early_termination": float(early.sum()) / n,
        "termination_reasons": reasons,
        "mean_final_goal_distance_m": float(final_dist.mean()),
        "mean_max_lift_m": float(max_height.mean()),
    }
    env.close()
    return stats


def record_video(
    checkpoint: str, path: str, num_envs: int, episodes: int, seed: int, device: str, size=(1280, 720), steps=None
):
    """Render ``episodes`` episodes of ``num_envs`` environments side by side into an MP4."""
    import math

    import newton
    import newton.viewer
    import numpy as np
    import torch
    import warp as wp
    from newton.examples.cable import cable_hand as ch

    from .env import KukaCableHandLiftCfg, KukaCableHandLiftEnv

    cfg = KukaCableHandLiftCfg(num_envs=num_envs, device=device, seed=seed, visuals=True)
    env = KukaCableHandLiftEnv(cfg)
    policy = _load_policy(env, checkpoint, device)
    viewer = newton.viewer.ViewerGL(width=size[0], height=size[1], headless=True)
    viewer.set_model(env.model)
    spacing = 0.9
    viewer.set_world_offsets((0.0, spacing, 0.0))
    offsets = viewer.world_offsets.numpy() if viewer.world_offsets is not None else np.zeros((num_envs, 3))
    center = offsets.mean(axis=0)
    # camera beyond the far edge of the table (the robot reaches toward -x), looking back at the hands
    span = spacing * max(num_envs - 1, 0)
    dist = 0.75 + 0.75 * span
    pitch = 14.0
    height = 0.42 + dist * math.tan(math.radians(pitch))
    viewer.set_camera(pos=wp.vec3(-0.55 - dist, float(center[1]), height), pitch=-pitch, yaw=0.0)
    piece = newton.Mesh.create_cylinder(1.0, 1.0, up_axis=newton.Axis.Z, segments=12, compute_inertia=False)
    viewer.log_mesh(
        "/cable_piece", wp.array(piece.vertices, dtype=wp.vec3), wp.array(piece.indices, dtype=wp.int32), hidden=True
    )
    spec = env.spec
    lay = env.layout
    per_world = len(lay.tendons)
    links_per_world = env.model.tendon_link_count // num_envs
    segs_per_world = env.model.tendon_segment_count // num_envs
    roles = {
        key: spec.digits[d].cable(c).role
        for key, (d, c) in zip(lay.hand.actuators, lay.hand.actuator_cables, strict=True)
    }
    world_tendons = [
        {
            key: dataclasses.replace(
                t,
                tendon=t.tendon + w * per_world,
                link_start=t.link_start + w * links_per_world,
                seg_start=t.seg_start + w * segs_per_world,
            )
            for key, t in lay.tendons.items()
        }
        for w in range(num_envs)
    ]
    radius = 0.5 * spec.cable.diameter
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{size[0]}x{size[1]}",
            "-r",
            str(round(1.0 / env.step_dt)),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            path,
        ],
        stdin=subprocess.PIPE,
    )
    obs = env.get_observations()
    t = 0.0
    with torch.inference_mode():
        for _ in range(steps or episodes * env.max_episode_length):
            obs, _, _, _ = env.step(policy(obs))
            t += env.step_dt
            viewer.begin_frame(t)
            viewer.log_state(env.state_0)
            starts, ends, colors = [], [], []
            for w in range(num_envs):
                for key, tendon in world_tendons[w].items():
                    s, e, tension = ch.tendon_cable_pieces(env.solver, env.model, env.state_0, tendon)
                    starts.append(s + offsets[w])
                    ends.append(e + offsets[w])
                    colors.append(ch.tension_colors(tension, spec.cable.tension_max, roles[key]))
            xforms, half = ch.cable_piece_transforms(np.concatenate(starts), np.concatenate(ends))
            scales = np.stack([np.full(len(xforms), radius), np.full(len(xforms), radius), half], axis=1)
            viewer.log_instances(
                "/cables",
                "/cable_piece",
                wp.array(xforms, dtype=wp.transform),
                wp.array(scales.astype(np.float32), dtype=wp.vec3),
                wp.array(np.concatenate(colors).astype(np.float32), dtype=wp.vec3),
                None,
            )
            target = env.target_pos.cpu().numpy() + offsets[:num_envs]
            viewer.log_shapes(
                "/targets",
                newton.GeoType.SPHERE,
                0.012,
                wp.array([wp.transform(wp.vec3(*p.tolist()), wp.quat_identity()) for p in target], dtype=wp.transform),
                wp.array(np.tile([[0.1, 0.85, 0.2]], (num_envs, 1)).astype(np.float32), dtype=wp.vec3),
            )
            viewer.end_frame()
            frame = viewer.get_frame().numpy()
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    proc.stdin.close()
    proc.wait()
    viewer.close()
    env.close()
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file or run directory (latest model_*.pt).")
    parser.add_argument("--eval_envs", type=int, default=1024, help="Episodes to evaluate; 0 skips the evaluation.")
    parser.add_argument("--video", default=None, help="Output MP4 path.")
    parser.add_argument("--video_envs", type=int, default=4)
    parser.add_argument("--video_episodes", type=int, default=1)
    parser.add_argument(
        "--video_steps", type=int, default=None, help="Policy steps to record (default: whole episodes)."
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--json", default=None, help="Write the evaluation statistics to this file.")
    args = parser.parse_args()
    checkpoint = _latest_checkpoint(args.checkpoint)
    if args.eval_envs > 0:
        stats = evaluate(checkpoint, args.eval_envs, args.seed, args.device)
        print(json.dumps(stats, indent=2), flush=True)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(stats, f, indent=2)
    if args.video:
        path = record_video(
            checkpoint, args.video, args.video_envs, args.video_episodes, args.seed, args.device, steps=args.video_steps
        )
        print("video:", path)


if __name__ == "__main__":
    main()

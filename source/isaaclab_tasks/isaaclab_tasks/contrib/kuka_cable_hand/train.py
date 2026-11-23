# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the Kuka iiwa 7 + cable-driven hand lift policy with RSL-RL PPO.

Example::

    python -m isaaclab_tasks.contrib.kuka_cable_hand.train --num_envs 4096 --max_iterations 1500
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log_root", default="logs/rsl_rl/kuka_cable_hand_lift")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--resume", default=None, help="Checkpoint to load before training.")
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument(
        "--held_reset_fraction",
        type=float,
        default=0.3,
        help="Share of resets that start with the cube held by a scripted pinch (spawn-in-hand curriculum).",
    )
    args = parser.parse_args()

    import torch
    from rsl_rl.runners import OnPolicyRunner

    from .agents import ppo_runner_cfg
    from .env import KukaCableHandLiftCfg, KukaCableHandLiftEnv

    torch.manual_seed(args.seed)
    env_cfg = KukaCableHandLiftCfg(
        num_envs=args.num_envs, device=args.device, seed=args.seed, held_reset_fraction=args.held_reset_fraction
    )
    t0 = time.time()
    env = KukaCableHandLiftEnv(env_cfg)
    print(f"[kuka_cable_hand] {args.num_envs} environments ready in {time.time() - t0:.1f} s", flush=True)

    train_cfg = ppo_runner_cfg(seed=args.seed, max_iterations=args.max_iterations, save_interval=args.save_interval)
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.abspath(os.path.join(args.log_root, stamp + (f"_{args.run_name}" if args.run_name else "")))
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    with open(os.path.join(log_dir, "params", "env.json"), "w") as f:
        json.dump(dataclasses.asdict(env_cfg), f, indent=2)
    with open(os.path.join(log_dir, "params", "agent.json"), "w") as f:
        json.dump(train_cfg, f, indent=2)

    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=args.device)
    if args.resume:
        runner.load(args.resume)
    print(f"[kuka_cable_hand] logging to {log_dir}", flush=True)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()

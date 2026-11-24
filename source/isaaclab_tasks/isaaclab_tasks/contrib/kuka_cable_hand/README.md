# Kuka iiwa 7 + cable-driven hand: lift a cube

The cable-driven hand of Newton's
[`exp/cable-driven-hand-series`](https://github.com/skulkar22/newton/tree/exp/cable-driven-hand-series)
branch (`newton.examples.cable.cable_hand`) on the KUKA LBR iiwa 7 of the Kuka-Allegro lift task
(`Isaac-Lift-KukaAllegro`), with a PPO policy that picks a cube up from the table and carries it to a
target in the air.

![The policy reaches, pinches and lifts the cube, then holds it at the target (OVRTX render)](docs/kuka_cable_hand_rtx_preview.gif)

Videos rendered with OVRTX (RTX path tracing, the arm in the original Kuka-Allegro materials):
[highlights](docs/kuka_cable_hand_rtx_highlights.mp4),
[side view, 3 episodes](docs/kuka_cable_hand_rtx_closeup.mp4),
[x-ray with the cables](docs/kuka_cable_hand_rtx_xray.mp4),
[4 environments](docs/kuka_cable_hand_rtx_4envs.mp4).
GitHub does not play MP4s in the page: each link opens the file view, which has a download button.

## What is simulated

| | Kuka-Allegro lift (`Isaac-Lift-KukaAllegro`) | This task |
|---|---|---|
| Arm | iiwa 7 from `Robots/KukaAllegro/kuka.usd`, PD joint drives | same USD (Allegro left out), prescribed joint trajectories |
| Hand | Allegro, 16 joint-position actuators | cable-driven hand: 15 flexion joints on 7 cables + thumb opposition hinge |
| Physics | MuJoCo-Warp (Newton) | VBD (Newton) with massless routed tendons |
| Scene | table top at 0.255 m, robot at the origin | same table, same layout |
| Object | 5 cm cube, 0.2 kg (default: procedural shapes) | 4 cm cube, 0.07 kg |
| Actions | relative joint targets for all 23 joints (scale 0.1) | relative arm joint targets (scale 0.1) + 7 cable tensions + opposition angle |
| Policy rate | 30 Hz | 30 Hz |
| Rewards | reaching, position progress, success, penalties | reaching, lifting, goal tracking (Isaac Lab lift recipe), penalties |
| Curriculum | ADR, gravity, spawn-in-hand reset bank | spawn-in-hand reset bank (scripted pinch) |

**The hand.** Five digits with three joints each, closed by cables and opened by preloaded return
springs. The index and middle fingers have two cables (a knuckle cable that bends the first joint and a
curl cable that runs to the fingertip), the ring finger, little finger and thumb one flexor each: seven
cables on seven linear actuators in the palm. Every actuator runs behind the governor of the branch
(force limit 60 N, rest tension 2 N, rate limit 30 mm/s). The thumb sits on a direct-drive opposition
hinge (0 to 130 deg). Only the pads (one capsule per phalanx and two on the palm) collide. All of this
comes unchanged from the branch: `add_hand_skeleton`, `add_hand_tendons`, `add_hand_pads`,
`create_actuator_governor`, `create_spring_preload` and the solver settings of
`make_tendon_hand_grasp_solver`. The hand is mounted in line with the iiwa flange (palm +x along the
flange axis, palmar side along link-7 -y), its wrist flange 71 mm from the link-7 frame (the Allegro
mount plane).

**The arm.** The links and the palm follow the forward kinematics of a joint reference that moves
toward the commanded target under the iiwa 7 speed limits (x0.5) and a 5 rad/s^2 acceleration limit,
the way the branch's grasp examples drive a kinematic wrist. This models a stiff position-controlled
industrial arm; the arm has no dynamics of its own and its links do not collide. A safety floor in the
driver refuses steps that would bring the palm pads, the palm origin, the forearm stub or the wrist
flange to the table (a stronger-than-any-contact arm would otherwise crush the fingers into the table
and past their stops), or the origins of links 4 to 7 within 6.5 cm of it while they are over the table
(the first policy, trained without that rule, rested the wrist housings in the table).

**Physics settings.** The branch tunes the tendon hand at 20 substeps x 16 iterations per 60 Hz frame.
Training uses 10 substeps x 8 iterations, at most 8 capstan sweeps per iteration and a contact
stiffness of 2.5e4 N/m (instead of 1e5), about 10x cheaper per frame. With these settings the branch's
own scripted grasps still work (the kinematic wrist of `example_cable_driven_hand`, `--scene grasp`):

| Grasp plan of the branch | Lift in the hold (plan 100 mm), default settings | fast settings, 1e5 N/m | fast settings, 2.5e4 N/m |
|---|---|---|---|
| can (side power grasp) | 100.8 mm | can thrown out (unstable) | 101.0 mm |
| block (three-digit pinch) | 99.3 mm | 102.1 mm | 102.1 mm |

At 4096 environments the task runs at about 20,000 environment steps per second on one RTX PRO 6000
Blackwell (0.19 s per policy step, 4.8 s per PPO iteration of 24 steps); the cost is set by the number
of kernel launches per substep, so it is nearly flat from 1 to 4096 worlds.

**Resets.** All worlds are identical replicas. After 0.5 s of settling the complete per-world state is
captured once: body and joint state, governor and opposition state, the arm reference and every
per-tendon, per-tendon-link and per-segment array of the solver. A reset copies the capture back for the
selected worlds, clears the solver's rigid history (joint duals and penalties, contact warm start) and
the contact matching there, and draws a new cube pose (+-6 cm, any yaw) and lift target.
`SolverVBD.reset` refuses models with tendons (the branch rebuilds the solver per episode); the rigid
part of it is called with the tendon check bypassed and the tendon history comes from the capture.

**Spawn-in-hand curriculum.** At start-up every world runs a scripted pinch (the block pinch of the
branch's grasp plans: thumb opposed to 118 deg, index and middle knuckle cables at 45 N, curl cables at
12 N) on one of 8 cube poses and lifts the cube by 10 cm; the worlds that hold it (about half) are
captured a second time. `held_reset_fraction` (default 0.3) of the resets start from those states. A
policy trained without this curriculum learned to bring the pinch point to the cube but did not lift it
once in 106 iterations (10 M steps).

## Observations, actions, rewards

Observations (103): arm joint reference and its rate, the 15 flexion angles and the opposition angle,
slider travels and cable tensions, palm pose, index/middle/ring/little/thumb tip pads relative to the
cube, cube pose and velocity, cube relative to the pinch point, target and target relative to the cube,
last action.

Actions (15, in [-1, 1]): 7 relative arm joint targets (x0.1 rad), 7 cable tensions (2 N to 45 N; the
governor enforces the rate and force limits), thumb opposition (60 deg to 130 deg).

Rewards (each multiplied by the 1/30 s policy step): pinch point to cube `1 - tanh(d/0.1)` (1.0) and
`1 - tanh(d/0.02)` (1.0); index, middle and thumb pads to cube `1 - tanh(d/0.05)` (1.0); lifted by 4 cm
while between the pinch digits (15); goal tracking `1 - tanh(d/0.3)` (16) and `1 - tanh(d/0.05)` (5)
while lifted; action rate (-1e-3); arm joint speed (-1e-3); palm pads below the table top (-200 per m);
early termination (-10). Episodes last 7 s and end early when the cube leaves the table, a flexion joint
is pushed more than 0.15 rad out of its range, or the state is not finite. Success: cube within 5 cm
of the target.

## Running

The branch pins Newton to the cable-driven hand commit (`pyproject.toml`, `uv.lock`):

```bash
uv sync
# train (4096 environments, 30 % spawn-in-hand resets; about 4.9 s per iteration)
.venv/bin/python -m isaaclab_tasks.contrib.kuka_cable_hand.train --num_envs 4096 --max_iterations 1500
# evaluate 1024 episodes and record a video of 4 environments (needs an X display for the GL viewer)
DISPLAY=:0 .venv/bin/python -m isaaclab_tasks.contrib.kuka_cable_hand.play \
    --checkpoint logs/rsl_rl/kuka_cable_hand_lift/<run> --eval_envs 1024 --video lift.mp4 --video_envs 4
# close-up from a three-quarter view, shells translucent so the cables (colored by role, brighter with tension) show
DISPLAY=:0 .venv/bin/python -m isaaclab_tasks.contrib.kuka_cable_hand.play \
    --checkpoint logs/rsl_rl/kuka_cable_hand_lift/<run> --eval_envs 0 --video xray.mp4 --video_envs 1 \
    --view angled --xray
```

Run the scripts with the environment's Python directly; `uv run` re-syncs the environment first.

## Rendering with OVRTX

`record.py` replays the policy and stores every body pose, the cable pieces with their tensions and the targets;
`render_ovrtx.py` renders such a recording with OVRTX, the kitless Omniverse RTX renderer (the `ovrtx` extra:
`uv sync --extra ovrtx`). The scene is authored as USD: the iiwa 7 link meshes with their original OmniPBR
materials, copied from the Kuka-Allegro asset into an arm-only USD (no Allegro hand); the hand from
`cable_hand_visuals.usd` of the Newton branch; the cables as `BasisCurves`, in the x-ray view colored by role (cyan
flexor and curl cables, magenta knuckle cables) and glowing with tension (none at the 2 N rest, full at 45 N, the
policy's largest command), otherwise neutral gray; the task table at its simulated size and place (a light-oak top
on legs; as in Isaac Lab's Kuka-Allegro lift task the arm stands on the floor beside it, here on a mounting plate);
and a light set (light floor and backdrop; dome, key, fill and rim lights, of which the fill and rim cast no
shadows). Each frame writes the recorded world poses to the prims (`omni:xform`) and renders a converged
path-traced image (history cleared, so moving parts leave no ghosting). On one RTX PRO 6000 Blackwell a 1920x1080
frame takes about 1.05 s at 48 spp and 1.95 s for the x-ray at 96 spp. The target marker hides once the cube
reaches it (with hysteresis, so it does not flicker).

```bash
python -m isaaclab_tasks.contrib.kuka_cable_hand.record \
    --checkpoint source/isaaclab_tasks/isaaclab_tasks/contrib/kuka_cable_hand/pretrained/model_1499.pt \
    --num_envs 1 --episodes 3 --seed 3 --out closeup.npz
python -m isaaclab_tasks.contrib.kuka_cable_hand.render_ovrtx --rollout closeup.npz --out closeup.mp4 --view side
python -m isaaclab_tasks.contrib.kuka_cable_hand.record \
    --checkpoint source/isaaclab_tasks/isaaclab_tasks/contrib/kuka_cable_hand/pretrained/model_1499.pt \
    --num_envs 1 --episodes 2 --seed 11 --out xray.npz
python -m isaaclab_tasks.contrib.kuka_cable_hand.render_ovrtx --rollout xray.npz --out xray.mp4 \
    --view track --distance 0.48 --xray --cable_scale 3.0 --spp 96 --max_frames 353
python -m isaaclab_tasks.contrib.kuka_cable_hand.record \
    --checkpoint source/isaaclab_tasks/isaaclab_tasks/contrib/kuka_cable_hand/pretrained/model_1499.pt \
    --num_envs 4 --episodes 1 --seed 5 --out overview.npz
python -m isaaclab_tasks.contrib.kuka_cable_hand.render_ovrtx --rollout overview.npz --out overview.mp4 \
    --view wide --elevation 55
```

Views: `side` (the whole robot from beside the table: its base stands on the floor next to the table, 4 cm behind
the table's far edge and below its top, so the low views over the table (`angled`, `front`, `track`, the default
`wide`) hide it; the 4-environment video looks down from 55 deg, where the bases show behind the table), `angled`
(fixed three-quarter view over the table), `track` (follows the pinch point, raised or lowered to keep the cube and
the lift target in view; both fit while the target is less than about 24 cm above the cube), `wide` (several
environments), `front`; `--distance`, `--elevation` and `--azimuth` override a view's camera. The first OVRTX run
compiles the RTX shader cache (`~/.cache/ovrtx`), which can take many minutes; later runs set up in about 2.5 s.
The look was checked by two rounds of review of full-resolution stills and a frame-by-frame review of the final
videos.

## Results

One training run: seed 42, 4096 environments, 1500 PPO iterations (147 M environment steps), 2 h 14 min
on one RTX PRO 6000 Blackwell, 30 % of the resets from the held-cube bank (2355 of 4096 worlds held the cube
after the scripted pinch). Every evaluation below starts **from the table** (no held starts): the
deterministic policy, one 7 s episode per environment, cube pose and target drawn as in training.

![Learning curve](docs/learning_curve.png)

| Checkpoint | Episodes | Lifted the cube | Reached the target (5 cm) | At the target at 7 s | Holding at 7 s | Ended early |
|---|---|---|---|---|---|---|
| 100 | 1024 | 76.8 % | 12.8 % | 0.0 % | 37.4 % | 9.4 % |
| 300 | 1024 | 91.9 % | 39.1 % | 13.2 % | 78.3 % | 7.8 % |
| 500 | 1024 | 92.8 % | 52.9 % | 30.5 % | 83.2 % | 6.0 % |
| 700 | 1024 | 96.5 % | 93.4 % | 87.0 % | 92.2 % | 5.1 % |
| 900 | 1024 | 99.4 % | 98.7 % | 95.2 % | 95.7 % | 3.1 % |
| 1200 | 1024 | 99.8 % | 99.2 % | 96.3 % | 96.5 % | 2.6 % |
| **1499** | **4096** | **99.8 %** | **99.4 %** | **97.5 %** | **97.5 %** | **2.1 %** |

In the final evaluation the cube ends 16.5 mm from the target on average. Of the 85 early endings, 73 are
a cube knocked off the table and 12 a digit pushed out of its range. All checkpoints are in
[`pretrained/sweep.jsonl`](pretrained/sweep.jsonl), the final evaluation in
[`pretrained/final_eval.json`](pretrained/final_eval.json).

The policy pinches the cube between the opposed thumb and the index and middle fingers (both cables of
those fingers pulled), lifts it and holds it at the target; when it drops the cube it goes back for it.

![One episode: reach, pre-grasp, pinch, lift, at the target, hold](docs/kuka_cable_hand_rtx.jpg)

Earlier previews (rendered with `play.py`, Newton OpenGL viewer):
[highlights](docs/kuka_cable_hand_highlights.mp4) (all three below, 42 s),
[close-up, 3 episodes](docs/kuka_cable_hand_closeup.mp4),
[x-ray, cables colored by role](docs/kuka_cable_hand_xray.mp4),
[4 environments](docs/kuka_cable_hand_4envs.mp4).

Two earlier runs shaped the recipe: without the held-cube resets PPO brought the pinch point to the cube but
did not lift it once in 106 iterations; a first run with the bank lifted the cube from the table in 87 % of
the episodes after 100 iterations but rested the wrist housings in the table, which the link rule of the
safety floor now prevents (that run was stopped and the recipe retrained from scratch).

The trained policy and its configuration are in [`pretrained/`](pretrained/):

```bash
DISPLAY=:0 .venv/bin/python -m isaaclab_tasks.contrib.kuka_cable_hand.play \
    --checkpoint source/isaaclab_tasks/isaaclab_tasks/contrib/kuka_cable_hand/pretrained/model_1499.pt \
    --eval_envs 4096 --video rollout.mp4 --video_envs 1 --view angled
```

## Limitations

- The arm is prescribed (kinematic). Contact cannot push it back, so the safety floor keeps the palm
  above the table; the fingers may touch the table, and a digit pushed far past its stop ends the episode.
- When a digit is crushed the solver can report that a cable crossed the supported wrap range of a
  sheave (`ERROR: Tendon ... crossed the supported wrap range`). These messages appear while the policy
  still flails; `train.py` output can be filtered to count them.
- The fast solver settings are validated on the branch's scripted can and block grasps, not on the
  hand calculation the branch checks its digits against (0.07 deg at 20 x 16).
- Only the massless-tendon cable model is used; the rod cable model is far more expensive.
- The task is a Newton-native vectorized environment with the `rsl_rl` `VecEnv` interface, placed in
  `isaaclab_tasks.contrib` and built on the Kuka-Allegro asset; it is not an Isaac Lab
  `ManagerBasedRLEnv`/`DirectRLEnv`. The hand is built procedurally by the branch's helpers rather than
  from USD, and a per-world reset needs the solver's tendon history restored, which Isaac Lab's Newton
  scene path (USD cloner, `NewtonManager`) has no hook for. Porting it means adding the hand through
  `newton_builder_world_hook`, the governor and opposition drive as state-force callbacks, and the
  tendon reset to the environment's `_reset_idx`.

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kuka iiwa 7 with the cable-driven hand: lift an object from a table (Newton VBD, massless tendons).

The scene follows the Kuka-Allegro lift task (``Isaac-Lift-KukaAllegro``): the arm stands at the
origin, the object rests on the same 0.8 m x 1.5 m table and the policy acts on relative arm joint
targets at 30 Hz. The Allegro hand is replaced by the cable-driven hand of the Newton
``exp/cable-driven-hand-series`` branch (``newton.examples.cable.cable_hand``): five underactuated
digits, seven cables on seven linear actuators behind a force/speed-limited governor, and a thumb on a
direct-drive opposition hinge. The policy commands one tension per cable and the opposition angle.

Physics. The hand, its tendons and the object are solved by :class:`newton.solvers.SolverVBD` with the
tendon settings of the branch, run at 10 substeps x 8 iterations per 60 Hz frame with at most 8
capstan sweeps and a contact stiffness of 2.5e4 N/m (the branch tunes the hand at 20 x 16; the
cheaper setting lifts the scripted can and block grasps of the branch by 101 mm and 102 mm against
the planned 100 mm, see the README). The arm is a stiff position-controlled
robot: its links and the palm follow the forward kinematics of a joint reference that moves toward
the commanded targets within the iiwa 7 joint speed limits and an acceleration limit (prescribed
motion, like the kinematic wrist of the branch's grasp examples). A safety floor keeps the palm and
the wrist off the table. Gravity acts as body forces on the hand and the object.

Resets. All worlds are identical replicas. After a settling phase the full state and the per-world
tendon history of the solver are captured once; a reset copies them back for the selected worlds,
clears the solver's rigid history and the contact matching there, and draws a new object pose and
lift target. (``SolverVBD.reset`` refuses models with tendons, so the rigid part of it is called with
the tendon check bypassed and the tendon arrays are restored from the capture.) A share of the resets
can start from a second capture in which a scripted pinch holds the cube (spawn-in-hand curriculum).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import newton
import numpy as np
import torch
import warp as wp
from newton.examples.cable import cable_hand as ch
from tensordict import TensorDict

from . import arm as arm_lib

TABLE_TOP = 0.255
"""Height of the table top [m] (the Kuka-Allegro lift table)."""

DIGITS = ("index", "middle", "ring", "little", "thumb")


@dataclass
class RewardWeights:
    """Reward weights; every term is also multiplied by the policy step [s] (as in Isaac Lab)."""

    reaching: float = 1.0
    """``1 - tanh(d / 0.1)`` of the distance between the pinch point and the object [m]."""
    reaching_fine: float = 1.0
    """``1 - tanh(d / 0.02)`` of the same distance."""
    fingertips: float = 1.0
    """``1 - tanh(d / 0.05)`` of the mean distance of the index, middle and thumb tip pads to the object [m]."""
    lifting: float = 15.0
    """1 while the object is more than :attr:`KukaCableHandLiftCfg.lift_height` above its resting height."""
    goal_tracking: float = 16.0
    """``lifted * (1 - tanh(d / 0.3))`` of the object-to-target distance [m]."""
    goal_tracking_fine: float = 5.0
    """``lifted * (1 - tanh(d / 0.05))`` of the object-to-target distance [m]."""
    action_rate: float = -1.0e-3
    """Squared change of the actions between policy steps."""
    arm_velocity: float = -1.0e-3
    """Squared joint velocity of the arm reference [rad^2/s^2]."""
    table_penetration: float = -200.0
    """Depth of the lowest palm pad below the table top [m] (the palm is prescribed and can enter the table)."""
    termination: float = -10.0
    """Paid once when an episode ends early (object off the table, palm deep in the table, invalid state)."""


@dataclass
class KukaCableHandLiftCfg:
    """Configuration of the lift task."""

    num_envs: int = 4096
    device: str = "cuda:0"
    seed: int = 42

    # --- physics ---
    frame_dt: float = 1.0 / 60.0
    """Physics frame [s]."""
    substeps: int = 10
    """VBD substeps per frame."""
    iterations: int = 8
    """VBD iterations per substep."""
    tendon_max_sweeps: int = 8
    """Largest number of capstan relaxation sweeps per VBD iteration."""
    contact_ke: float = 2.5e4
    """Contact stiffness of pads, object and table [N/m]."""
    decimation: int = 2
    """Physics frames per policy step (30 Hz policy)."""
    episode_length_s: float = 7.0
    settle_time: float = 0.5
    """Time the hand settles at rest before the reset state is captured [s]."""

    # --- scene ---
    table_center: tuple[float, float, float] = (-0.55, 0.0, TABLE_TOP - 0.02)
    table_half_extents: tuple[float, float, float] = (0.4, 0.75, 0.02)
    object_half_extent: float = 0.02
    """Half edge of the cube [m]: a 4 cm cube (the Kuka-Allegro task lifts a 5 cm cube with the larger Allegro
    hand; this human-size hand pinches a 4 cm cube between the thumb and the index and middle fingers)."""
    object_mass: float = 0.07
    """Cube mass [kg]."""
    object_nominal_xy: tuple[float, float] = (-0.60, 0.0)
    object_xy_range: float = 0.06
    """Half range of the uniform object position on the table [m]."""
    object_yaw_range: float = math.pi
    """Half range of the uniform object yaw [rad]."""
    arm_home: tuple[float, ...] = (-0.0578, -0.3568, 0.6659, 1.901, -1.8673, -1.5234, 0.5995)
    """Arm joints at reset [rad]: the palm faces the table with its grasp site 17 cm above the object center
    (inverse kinematics from the Kuka-Allegro home pose)."""
    arm_home_noise: float = 0.0
    """Half range of a uniform offset added to every arm joint at reset [rad]."""
    visuals: bool = False
    """Attach the visual meshes of the arm and the hand (for rendering)."""

    # --- actions ---
    arm_action_scale: float = 0.1
    """Arm action: target = reference + scale * action [rad] (the Kuka-Allegro relative joint action)."""
    arm_speed_fraction: float = 0.5
    """Fraction of the iiwa 7 joint speed limits the arm reference may move at."""
    arm_acceleration_limit: float = 5.0
    """Largest joint acceleration of the arm reference [rad/s^2]. The palm then accelerates by well under
    the 1 g the hand's spring preload is sized for; faster changes fling the fingers past their stops."""
    palm_floor_margin: float = 0.002
    """The arm driver does not take a step that brings a palm pad within this distance of the table top [m]
    (a stiff arm would otherwise crush the fingers into the table and past their stops)."""
    finger_angle_margin: float = 0.15
    """An episode ends when a flexion joint leaves its range by more than this [rad]: a digit crushed past its
    stop (the arm is prescribed and stronger than any contact) can turn a cable around its sheave."""
    tension_command_max: float = 45.0
    """Cable tension at action +1 [N]; action -1 is the rest tension. The governor limit is 60 N."""
    opposition_range: tuple[float, float] = (math.radians(60.0), math.radians(130.0))
    """Thumb opposition angle at action -1 and +1 [rad] (the hinge range is 0 to 130 deg)."""
    grasp_site_in_palm: tuple[float, float, float] = (0.0535, 0.095, 0.022)
    """Pinch point in the palm frame [m]: between the opposed thumb and the index and middle fingertips
    (the block pinch of the branch's grasp plans). The reaching reward brings it to the object."""

    # --- task ---
    lift_height: float = 0.04
    """Height above the resting height at which the object counts as lifted [m]."""
    held_distance: float = 0.04
    """The object counts as lifted only while the mean distance of the index, middle and thumb tip pads to its
    center is below this [m]."""
    target_x_range: tuple[float, float] = (-0.70, -0.50)
    target_y_range: tuple[float, float] = (-0.15, 0.15)
    target_z_range: tuple[float, float] = (0.40, 0.55)
    success_threshold: float = 0.05
    """Object-to-target distance that counts as success [m]."""
    rewards: RewardWeights = field(default_factory=RewardWeights)

    # --- held-object reset bank (the spawn-in-hand curriculum of the Kuka-Allegro task) ---
    held_reset_fraction: float = 0.0
    """Share of resets that start from a state in which a scripted pinch holds the cube 10 cm above the table.
    The states are made once at start-up (see :meth:`KukaCableHandLiftEnv._build_held_bank`)."""
    held_bank_groups: int = 8
    """Number of different cube poses the scripted pinch is run from (the worlds are split into groups)."""

    @property
    def step_dt(self) -> float:
        return self.frame_dt * self.decimation

    @property
    def max_episode_length(self) -> int:
        return int(math.ceil(self.episode_length_s / self.step_dt))


# ----------------------------------------------------------------------------------------------- kernels


@wp.kernel
def _body_weight_kernel(
    body_mass: wp.array[float],
    body_flags: wp.array[wp.int32],
    gravity: wp.vec3,
    body_f: wp.array[wp.spatial_vector],
):
    """Add the weight of every dynamic body to its wrench (the model has no gravity of its own)."""
    b = wp.tid()
    if body_flags[b] & int(newton.BodyFlags.KINEMATIC):
        return
    body_f[b] = body_f[b] + wp.spatial_vector(body_mass[b] * gravity, wp.vec3(0.0, 0.0, 0.0))


@wp.kernel
def _opposition_drive_kernel(
    body_q: wp.array[wp.transform],
    palm: wp.array[int],
    column: wp.array[int],
    parent_xform: wp.transform,
    child_xform: wp.transform,
    axis: wp.vec3,
    target: wp.array[float],
    lower: float,
    upper: float,
    ke: float,
    torque_limit: float,
    speed_limit: float,
    dt: float,
    target_index: wp.array[int],
    reference: wp.array[float],
    angle: wp.array[float],
    joint_target_q: wp.array[float],
):
    """Rate- and torque-limited reference of the thumb opposition drive, one thread per hand.

    The same rule as ``cable_hand.opposition_drive_kernel`` of the branch, for many hands.
    """
    w = wp.tid()
    q_parent = wp.mul(wp.transform_get_rotation(body_q[palm[w]]), wp.transform_get_rotation(parent_xform))
    q_child = wp.mul(wp.transform_get_rotation(body_q[column[w]]), wp.transform_get_rotation(child_xform))
    rel = wp.mul(wp.quat_inverse(q_parent), q_child)
    phi = 2.0 * wp.atan2(wp.dot(wp.vec3(rel[0], rel[1], rel[2]), axis), rel[3])
    if phi > wp.pi:
        phi = phi - 2.0 * wp.pi
    if phi < -wp.pi:
        phi = phi + 2.0 * wp.pi
    goal = wp.clamp(target[w], lower, upper)
    ref = reference[w] + wp.clamp(goal - reference[w], -speed_limit * dt, speed_limit * dt)
    ref = wp.clamp(ref, phi - torque_limit / ke, phi + torque_limit / ke)
    reference[w] = ref
    angle[w] = phi
    joint_target_q[target_index[w]] = ref


# ------------------------------------------------------------------------------------------------- scene


@dataclass
class WorldLayout:
    """Indices of one world of the scene (world 0; world ``w`` adds ``w`` times the per-world counts)."""

    arm_links: list[int]
    hand: ch.HandSkeleton
    tendons: dict[str, ch.DigitTendon]
    pads: ch.HandPads
    object_body: int
    object_shape: int
    table_shape: int
    fingertips: list[int]
    palm_pad_bodies: list[int]
    object_rest: np.ndarray
    """Pose ``(x, y, z, qx, qy, qz, qw)`` of the object at build time."""


def _offset_skeleton(sk: ch.DigitSkeleton, db: int, dj: int, dq: int, dd: int) -> ch.DigitSkeleton:
    """Copy of a digit skeleton with every body, joint, coordinate and DOF index shifted by one world."""
    return dataclasses.replace(
        sk,
        bodies={k: v + db for k, v in sk.bodies.items()},
        joint_flexion=tuple(j + dj for j in sk.joint_flexion),
        joint_sliders=tuple(j + dj for j in sk.joint_sliders),
        q_flexion=tuple(q + dq for q in sk.q_flexion),
        q_sliders=tuple(q + dq for q in sk.q_sliders),
        dof_flexion=tuple(d + dd for d in sk.dof_flexion),
        dof_sliders=tuple(d + dd for d in sk.dof_sliders),
        base_joint=None if sk.base_joint is None else sk.base_joint + dj,
    )


def build_world(
    cfg: KukaCableHandLiftCfg, spec: ch.HandSpec, arm: arm_lib.ArmModel
) -> tuple[newton.ModelBuilder, WorldLayout]:
    """Build one world: the kinematic iiwa 7, the tendon hand on its flange, the table and the cube."""
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    q_home = np.asarray(cfg.arm_home, dtype=float)
    links = arm.link_transforms(q_home)
    visual_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0, has_shape_collision=False, has_particle_collision=False, is_visible=True
    )
    arm_links = []
    for i in range(8):
        body = b.add_body(
            xform=wp.transform(*arm_lib.transform_to_pq(links[i]).tolist()),
            mass=float(arm.link_mass[i]),
            com=wp.vec3(*arm.link_com[i].tolist()),
            inertia=wp.mat33(*arm.link_inertia[i].flatten().tolist()),
            is_kinematic=True,
            label=f"iiwa7_link_{i}",
        )
        arm_links.append(body)
    if cfg.visuals:
        for v in arm.visuals:
            b.add_shape_mesh(
                body=arm_links[v.link],
                xform=v.xform,
                mesh=v.mesh,
                scale=v.scale,
                cfg=visual_cfg,
                color=(0.92, 0.42, 0.10) if v.link not in (0,) else (0.25, 0.25, 0.27),
                label=f"iiwa7_link_{v.link}/visual",
            )

    palm_xform = wp.transform(*arm_lib.transform_to_pq(links[-1] @ arm.mount).tolist())
    hand = ch.add_hand_skeleton(b, spec, "kinematic", palm_xform, opposition=0.0)
    tendons = ch.add_hand_tendons(b, spec, hand)
    ch.add_hand_articulation(b, hand)
    pads = ch.add_hand_pads(b, spec, hand, visible=not cfg.visuals, contact_ke=cfg.contact_ke)
    if cfg.visuals:
        ch.add_hand_visuals(b, hand, spec=spec)

    surface_cfg = ch.grasp_shape_config(spec, spec.pads.table_mu**2 / spec.pads.mu, contact_ke=cfg.contact_ke)
    table_shape = b.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(*cfg.table_center), wp.quat_identity()),
        hx=cfg.table_half_extents[0],
        hy=cfg.table_half_extents[1],
        hz=cfg.table_half_extents[2],
        cfg=surface_cfg,
        color=(0.25, 0.15, 0.15),
        label="table",
    )
    h = cfg.object_half_extent
    rest = np.array([*cfg.object_nominal_xy, TABLE_TOP + h, 0.0, 0.0, 0.0, 1.0])
    inertia = (2.0 / 3.0) * cfg.object_mass * h * h
    object_body = b.add_body(
        xform=wp.transform(*rest.tolist()),
        mass=cfg.object_mass,
        inertia=wp.mat33(inertia, 0.0, 0.0, 0.0, inertia, 0.0, 0.0, 0.0, inertia),
        label="object",
    )
    object_cfg = ch.grasp_shape_config(spec, spec.pads.mu, contact_ke=cfg.contact_ke)
    object_shape = b.add_shape_box(
        body=object_body, hx=h, hy=h, hz=h, cfg=object_cfg, color=(0.58, 0.47, 0.82), label="object"
    )
    layout = WorldLayout(
        arm_links=arm_links,
        hand=hand,
        tendons=tendons,
        pads=pads,
        object_body=object_body,
        object_shape=object_shape,
        table_shape=table_shape,
        fingertips=[hand.bodies[f"{d}/link3"] for d in DIGITS],
        palm_pad_bodies=[int(x) for x in pads.body],
        object_rest=rest,
    )
    return b, layout


# --------------------------------------------------------------------------------------------------- env


class KukaCableHandLiftEnv:
    """Vectorized lift task; implements the ``rsl_rl.env.VecEnv`` interface.

    Observations are returned as a :class:`tensordict.TensorDict` with one group, ``"policy"``.
    """

    def __init__(self, cfg: KukaCableHandLiftCfg):
        self.cfg = cfg
        self.num_envs = n = cfg.num_envs
        self.device = cfg.device
        self.wp_device = wp.get_device(cfg.device)
        self.max_episode_length = cfg.max_episode_length
        self.step_dt = cfg.step_dt
        self.sim_dt = cfg.frame_dt / cfg.substeps
        if (cfg.decimation * cfg.substeps) % 2:
            raise ValueError("decimation * substeps must be even so the state buffers keep their roles")
        self._torch_gen = torch.Generator(device=self.device)
        self._torch_gen.manual_seed(cfg.seed)

        # --- model ---------------------------------------------------------------------------------
        self.spec = spec = ch.default_hand_spec()
        self.arm = arm = arm_lib.load_arm(arm_lib.resolve_kuka_usd(), with_visuals=cfg.visuals)
        single, lay = build_world(cfg, spec, arm)
        self.layout = lay
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for _ in range(n):
            builder.add_world(single)
        builder.color()
        with wp.ScopedDevice(self.wp_device):
            self.model = model = builder.finalize(device=self.wp_device)
        self.B, self.J = single.body_count, single.joint_count
        self.Q, self.D = single.joint_coord_count, single.joint_dof_count
        if model.body_count != n * self.B or model.joint_count != n * self.J or model.joint_coord_count != n * self.Q:
            raise RuntimeError("unexpected world replication layout")
        if model.tendon_count != n * len(lay.tendons):
            raise RuntimeError(f"expected {n * len(lay.tendons)} tendons, the model has {model.tendon_count}")

        # --- collision, solver, states ---------------------------------------------------------------
        self.pipeline = newton.CollisionPipeline(model, contact_matching="latest")
        self.contacts = self.pipeline.contacts()
        self.solver = ch.make_tendon_hand_grasp_solver(
            model, iterations=cfg.iterations, tendon_max_sweeps=cfg.tendon_max_sweeps
        )
        self.state_0, self.state_1 = model.state(), model.state()
        self.control = model.control()

        # --- hand control for all worlds ------------------------------------------------------------
        skeletons = [
            _offset_skeleton(sk, w * self.B, w * self.J, w * self.Q, w * self.D)
            for w in range(n)
            for sk in lay.hand.skeletons
        ]
        self.governor = ch.create_actuator_governor(model, spec, skeletons)
        lead = np.array([spec.cable.tension_rest / sk.slider_kp for sk in skeletons for _ in sk.cable_names])
        self.governor.s_ref.assign(self.governor.s_ref.numpy() + lead.astype(np.float32))
        self.preload = ch.create_spring_preload(model, skeletons)
        self.preload.apply(self.control)
        self.num_cables = len(lay.hand.actuators)
        hand = lay.hand
        joint_q_start = model.joint_q_start.numpy()
        self._opp_palm = wp.array([hand.palm + w * self.B for w in range(n)], dtype=int, device=self.wp_device)
        self._opp_column = wp.array(
            [hand.thumb_column + w * self.B for w in range(n)], dtype=int, device=self.wp_device
        )
        self._opp_target_index = wp.array(
            [int(joint_q_start[hand.joint_opposition + w * self.J]) for w in range(n)], dtype=int, device=self.wp_device
        )
        self.opp_target = wp.zeros(n, dtype=float, device=self.wp_device)
        self.opp_reference = wp.zeros(n, dtype=float, device=self.wp_device)
        self.opp_angle = wp.zeros(n, dtype=float, device=self.wp_device)
        self.gravity = wp.vec3(0.0, 0.0, -ch.GRAVITY)

        # --- arm driver ------------------------------------------------------------------------------
        self._link_body = wp.array(
            np.array([[b + w * self.B for b in lay.arm_links] for w in range(n)], dtype=np.int32),
            dtype=int,
            device=self.wp_device,
        )
        self._palm_body = wp.array([hand.palm + w * self.B for w in range(n)], dtype=int, device=self.wp_device)
        self._joint_xforms = wp.array(
            [wp.transform(*arm_lib.transform_to_pq(x).tolist()) for x in arm.joint_xforms],
            dtype=wp.transform,
            device=self.wp_device,
        )
        self._mount = wp.transform(*arm_lib.transform_to_pq(arm.mount).tolist())
        self._q_lower = wp.array(arm.lower, dtype=float, device=self.wp_device)
        self._q_upper = wp.array(arm.upper, dtype=float, device=self.wp_device)
        self._q_speed = wp.array(
            arm_lib.IIWA7_VELOCITY_LIMIT * cfg.arm_speed_fraction, dtype=float, device=self.wp_device
        )
        home = np.tile(np.asarray(cfg.arm_home, dtype=np.float32), (n, 1))
        self.q_target = wp.array(home, dtype=float, device=self.wp_device)
        self.q_ref = wp.array(home, dtype=float, device=self.wp_device)
        self.qd_ref = wp.zeros((n, 7), dtype=float, device=self.wp_device)
        self.arm_blocked = wp.zeros(n, dtype=int, device=self.wp_device)
        # safety floor of the arm driver: palm pad ends, palm origin and wrist flange above the table top
        # (palm pad ends, palm origin and wrist flange; the digits are free to touch the table)
        pads = lay.pads
        palm_pads = [i for i, k in enumerate(pads.keys) if k.startswith("palm/")]
        points = [*pads.p0[palm_pads], *pads.p1[palm_pads], (0.0, 0.0, 0.0), arm_lib.PALM_FLANGE_IN_PALM]
        self._floor_points = wp.array(np.asarray(points, dtype=np.float32), dtype=wp.vec3, device=self.wp_device)
        self._palm_floor = TABLE_TOP + float(pads.radius[palm_pads].max()) + cfg.palm_floor_margin
        self._link_floor = TABLE_TOP

        # --- torch views -----------------------------------------------------------------------------
        self._views()
        self.num_actions = 7 + self.num_cables + 1
        self.actions = torch.zeros(n, self.num_actions, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.episode_length_buf = torch.zeros(n, dtype=torch.long, device=self.device)
        self.target_pos = torch.zeros(n, 3, device=self.device)
        self.tension_rest = float(spec.cable.tension_rest)
        self.opp_upper = float(hand.opposition_upper)
        self._episode_sums: dict[str, torch.Tensor] = {}
        self.extras: dict = {}

        # --- settle, capture graph, take the reset snapshot -------------------------------------------
        self._graph = None
        self._capture_graph()
        settle_steps = max(1, round(cfg.settle_time / cfg.step_dt))
        for _ in range(settle_steps):
            self._simulate_step()
        self._take_snapshot()
        self._snap_held = None
        self.held_valid = torch.zeros(n, dtype=torch.bool, device=self.device)
        if cfg.held_reset_fraction > 0.0:
            self._build_held_bank()
        self.reset_idx(torch.arange(n, device=self.device))
        self._compute_state()
        self.obs_buf = self._observations()

    # ------------------------------------------------------------------------------------- internals

    def _views(self):
        n, B = self.num_envs, self.B
        to_t = wp.to_torch
        self.body_q = to_t(self.state_0.body_q).view(n, B, 7)
        self.body_qd = to_t(self.state_0.body_qd).view(n, B, 6)
        self.joint_q = to_t(self.state_0.joint_q).view(n, self.Q)
        self.t_q_target = to_t(self.q_target)
        self.t_q_ref = to_t(self.q_ref)
        self.t_qd_ref = to_t(self.qd_ref)
        self.t_command = to_t(self.governor.command).view(n, self.num_cables)
        self.t_drive_force = to_t(self.governor.drive_force).view(n, -1)
        self.t_travel = to_t(self.governor.travel).view(n, -1)
        self.t_opp_target = to_t(self.opp_target)
        self.t_opp_angle = to_t(self.opp_angle)
        lay = self.layout
        hand = lay.hand
        self.q_flex_idx = torch.tensor(
            [q for sk in hand.skeletons for q in sk.q_flexion], dtype=torch.long, device=self.device
        )
        self.q_opp_idx = int(hand.q_opposition)
        self.q_flex_lower = torch.tensor(
            np.concatenate([sk.flexion_lower for sk in hand.skeletons]), dtype=torch.float32, device=self.device
        )
        self.q_flex_upper = torch.tensor(
            np.concatenate([sk.flexion_upper for sk in hand.skeletons]), dtype=torch.float32, device=self.device
        )
        self.fingertip_idx = torch.tensor(lay.fingertips, dtype=torch.long, device=self.device)
        # fingertip pad centers in the distal link frames, and the palm pads (for the table check)
        keys = lay.pads.keys
        tip_pad = [keys.index(f"{d}/link3") for d in DIGITS]
        self.tip_offset = torch.tensor(
            0.5 * (lay.pads.p0[tip_pad] + lay.pads.p1[tip_pad]), dtype=torch.float32, device=self.device
        )
        palm_pads = [i for i, k in enumerate(keys) if k.startswith("palm/")]
        self.palm_pad_points = torch.tensor(
            np.concatenate([lay.pads.p0[palm_pads], lay.pads.p1[palm_pads]]), dtype=torch.float32, device=self.device
        )
        self.palm_pad_radius = float(lay.pads.radius[palm_pads].max())
        self.grasp_site = torch.tensor(self.cfg.grasp_site_in_palm, dtype=torch.float32, device=self.device)
        self.pinch_tip_idx = torch.tensor([DIGITS.index(d) for d in ("index", "middle", "thumb")], device=self.device)

    def _simulate_substeps(self):
        """One policy step of physics; every call is a kernel launch (captured into a CUDA graph)."""
        model, control, dt = self.model, self.control, self.sim_dt
        hand = self.layout.hand
        for _ in range(self.cfg.decimation * self.cfg.substeps):
            s0, s1 = self.state_0, self.state_1
            s0.clear_forces()
            wp.launch(
                _body_weight_kernel,
                dim=model.body_count,
                inputs=[model.body_mass, model.body_flags, self.gravity],
                outputs=[s0.body_f],
            )
            wp.launch(
                _opposition_drive_kernel,
                dim=self.num_envs,
                inputs=[
                    s0.body_q,
                    self._opp_palm,
                    self._opp_column,
                    hand.opposition_parent_xform,
                    hand.opposition_child_xform,
                    wp.vec3(*hand.opposition_axis),
                    self.opp_target,
                    hand.opposition_lower,
                    hand.opposition_upper,
                    hand.opposition_ke,
                    hand.opposition_torque_limit,
                    hand.opposition_speed_limit,
                    dt,
                    self._opp_target_index,
                ],
                outputs=[self.opp_reference, self.opp_angle, control.joint_target_q],
            )
            self.governor.step(s0, control, dt)
            # the arm and the palm move last: the governor and the opposition drive measured the hand at the
            # poses of the start of this substep
            wp.launch(
                arm_lib.arm_drive_kernel,
                dim=self.num_envs,
                inputs=[
                    self.q_target,
                    self._q_lower,
                    self._q_upper,
                    self._q_speed,
                    self.cfg.arm_acceleration_limit,
                    self._joint_xforms,
                    self._mount,
                    self._link_body,
                    self._palm_body,
                    model.body_com,
                    self._floor_points,
                    self._palm_floor,
                    self._link_floor,
                    dt,
                ],
                outputs=[self.q_ref, self.qd_ref, self.arm_blocked, s0.body_q, s0.body_qd],
            )
            self.pipeline.collide(s0, self.contacts)
            self.solver.step(s0, s1, control, self.contacts, dt)
            self.state_0, self.state_1 = s1, s0
        newton.eval_ik(model, self.state_0, self.state_0.joint_q, self.state_0.joint_qd)

    def _capture_graph(self):
        with wp.ScopedDevice(self.wp_device):
            self._simulate_substeps()  # compile and allocate lazily created buffers
            if self.wp_device.is_cuda:
                with wp.ScopedCapture(device=self.wp_device) as capture:
                    self._simulate_substeps()
                self._graph = capture.graph

    def _simulate_step(self):
        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            with wp.ScopedDevice(self.wp_device):
                self._simulate_substeps()

    def _tendon_arrays(self) -> list[tuple[str, torch.Tensor]]:
        """Solver arrays that hold per-tendon, per-tendon-link or per-segment state, as torch views.

        Index arrays among them hold indices of their own world, so every world is restored from its
        own capture. Struct arrays (the per-step material modes) are derived each step and skipped.
        """
        model, solver = self.model, self.solver
        sizes = {int(model.tendon_count), int(model.tendon_link_count), int(model.tendon_segment_count)}
        found = []
        for name, value in vars(solver).items():
            if "tendon" not in name or not isinstance(value, wp.array) or value.ndim < 1:
                continue
            if value.shape[0] not in sizes or value.shape[0] == 0 or value.shape[0] % self.num_envs:
                continue
            try:
                found.append((name, wp.to_torch(value)))
            except RuntimeError:
                continue  # structured dtype
        return found

    def _live_arrays(self) -> dict[str, torch.Tensor]:
        """Every per-world array a reset restores, as torch views with the world as the first dimension."""
        n = self.num_envs
        arrays = {
            "body_q": wp.to_torch(self.state_0.body_q),
            "body_qd": wp.to_torch(self.state_0.body_qd),
            "joint_q": wp.to_torch(self.state_0.joint_q),
            "joint_qd": wp.to_torch(self.state_0.joint_qd),
            "q_ref": self.t_q_ref,
            "q_target": self.t_q_target,
        }
        for name in ("s_ref", "travel", "drive_force", "ref_rate", "speed_excess", "command"):
            arrays[f"governor.{name}"] = wp.to_torch(getattr(self.governor, name))
        for name in ("opp_target", "opp_reference", "opp_angle"):
            arrays[name] = wp.to_torch(getattr(self, name))
        for name, t in self._tendon_arrays():
            arrays[f"solver.{name}"] = t
        return {k: v.view(n, -1, *v.shape[1:]) if v.shape[0] != n else v for k, v in arrays.items()}

    def _capture(self) -> dict[str, torch.Tensor]:
        """Copy of every per-world array (index arrays among them hold indices of their own world)."""
        wp.synchronize_device(self.wp_device)
        return {k: v.clone() for k, v in self._live_arrays().items()}

    def _restore(self, snapshot: dict[str, torch.Tensor], ids: torch.Tensor):
        """Copy the selected worlds of a capture back into the live arrays."""
        for k, live in self._live_arrays().items():
            live[ids] = snapshot[k][ids]

    def _take_snapshot(self):
        # all worlds are identical at this point; the snapshot of every world is the reset state of that world
        self._snap_open = self._capture()

    def _script_step(self, q_goal: torch.Tensor, tensions: torch.Tensor, opposition: float):
        """One policy step of physics with commanded arm joints, cable tensions and opposition (no RL bookkeeping)."""
        self.t_q_target[:] = q_goal
        self.t_command[:] = tensions
        self.t_opp_target[:] = opposition
        self._simulate_step()

    def _build_held_bank(self):
        """Run a scripted pinch from the reset state and keep the worlds that end up holding the cube.

        The worlds are split into :attr:`KukaCableHandLiftCfg.held_bank_groups` groups, each with its own cube
        pose near the nominal position (faces turned to the pinch). Every group follows the same open-loop
        script on its own inverse-kinematics waypoints: oppose the thumb to 118 deg, bring the pinch point
        down to the cube, pull the index and middle knuckle cables, lift by 10 cm. This is the block pinch of
        the branch's grasp plans. The resulting states seed :attr:`KukaCableHandLiftCfg.held_reset_fraction`
        of the resets, like the spawn-in-hand resets of the Kuka-Allegro task.
        """
        from scipy.optimize import least_squares

        cfg, n, arm, dev = self.cfg, self.num_envs, self.arm, self.device
        all_ids = torch.arange(n, device=dev)
        self._restore(self._snap_open, all_ids)
        groups = max(1, min(cfg.held_bank_groups, n))
        rng = np.random.default_rng(cfg.seed + 12345)
        group = torch.as_tensor(np.arange(n) % groups, device=dev)
        site = np.asarray(cfg.grasp_site_in_palm, dtype=float)
        q_home = np.asarray(cfg.arm_home, dtype=float)
        home_palm = arm.palm_transform(q_home)
        rot = home_palm[:3, :3]
        hand_yaw = math.atan2(rot[1, 0], rot[0, 0])
        h = cfg.object_half_extent
        poses = [
            (
                cfg.object_nominal_xy[0] + rng.uniform(-0.03, 0.03),
                cfg.object_nominal_xy[1] + rng.uniform(-0.03, 0.03),
                hand_yaw + rng.uniform(-0.2, 0.2),
            )
            for _ in range(groups)
        ]
        ob = self.layout.object_body
        pos = torch.tensor([[x, y, TABLE_TOP + h] for x, y, _ in poses], dtype=torch.float32, device=dev)[group]
        yaw = torch.tensor([a for _, _, a in poses], dtype=torch.float32, device=dev)[group]
        self.body_q[:, ob, :3] = pos
        self.body_q[:, ob, 3:] = 0.0
        self.body_q[:, ob, 5] = torch.sin(0.5 * yaw)
        self.body_q[:, ob, 6] = torch.cos(0.5 * yaw)
        self.body_qd[:, ob] = 0.0
        self._reset_common(all_ids)

        def ik(target, q0):
            def residual(q):
                links = arm.link_transforms(q)
                t = links[-1] @ arm.mount
                p = t[:3, :3] @ site + t[:3, 3]
                low = np.clip(TABLE_TOP + 0.05 - links[5:, 2, 3], 0.0, None)
                orient = np.concatenate([t[:3, i] - rot[:, i] for i in range(3)])
                return np.concatenate([p - target, 0.1 * orient, 3.0 * low, 0.01 * (q - q0)])

            return least_squares(residual, q0, bounds=(arm.lower + 0.05, arm.upper - 0.05)).x

        def path(q0, p0, p1, count):
            out, q = [], q0
            for i in range(1, count + 1):
                q = ik(p0 + (p1 - p0) * i / count, q)
                out.append(q)
            return out

        site_home = rot @ site + home_palm[:3, 3]
        plans = []
        for x, y, _ in poses:
            obj = np.array([x, y, TABLE_TOP + h])
            approach = path(q_home, site_home, obj + np.array([0.0, 0.0, 0.08]), 6)
            descend = path(approach[-1], obj + np.array([0.0, 0.0, 0.08]), obj, 6)
            lift = path(descend[-1], obj, obj + np.array([0.0, 0.0, 0.10]), 8)
            plans.append((approach, descend, lift))

        keys = self.layout.hand.actuators
        pinch = {"index": 12.0, "index_knuckle": 45.0, "middle": 12.0, "middle_knuckle": 45.0, "thumb": 8.0}
        rest = torch.full((n, len(keys)), self.tension_rest, device=dev)
        grip = torch.tensor([pinch.get(k, self.tension_rest) for k in keys], dtype=torch.float32, device=dev)
        grip = grip.expand(n, -1).contiguous()
        oppose = math.radians(118.0)

        def follow(phase, tensions, steps_per):
            for i in range(len(plans[0][phase])):
                goal = torch.tensor(np.stack([p[phase][i] for p in plans]), dtype=torch.float32, device=dev)[group]
                for _ in range(steps_per):
                    self._script_step(goal, tensions, oppose)

        follow(0, rest, 8)
        last = torch.tensor(np.stack([p[0][-1] for p in plans]), dtype=torch.float32, device=dev)[group]
        for _ in range(20):
            self._script_step(last, rest, oppose)
        follow(1, rest, 8)
        last = torch.tensor(np.stack([p[1][-1] for p in plans]), dtype=torch.float32, device=dev)[group]
        for _ in range(40):
            self._script_step(last, grip, oppose)
        follow(2, grip, 8)
        last = torch.tensor(np.stack([p[2][-1] for p in plans]), dtype=torch.float32, device=dev)[group]
        for _ in range(20):
            self._script_step(last, grip, oppose)

        self._compute_state()
        q_flex = self.joint_q[:, self.q_flex_idx]
        m = cfg.finger_angle_margin
        hand_ok = ((q_flex >= self.q_flex_lower - m) & (q_flex <= self.q_flex_upper + m)).all(dim=-1)
        finite = torch.isfinite(self.body_q.view(n, -1)).all(dim=-1)
        self.held_valid = self._lifted & hand_ok & finite
        self._snap_held = self._capture()
        lo, hi = cfg.opposition_range
        actions = torch.zeros(n, 7 + len(keys) + 1, device=dev)
        actions[:, 7 : 7 + len(keys)] = (
            2.0 * (grip - self.tension_rest) / (cfg.tension_command_max - self.tension_rest) - 1.0
        )
        actions[:, -1] = 2.0 * (oppose - lo) / (hi - lo) - 1.0
        self._held_actions = actions
        print(
            f"[kuka_cable_hand] held-object reset bank: {int(self.held_valid.sum())}/{n} worlds hold the cube "
            f"({groups} cube poses)",
            flush=True,
        )

    def _reset_solver_worlds(self, world_mask: wp.array):
        """Clear the rigid VBD history (joint duals, penalties, contact warm start) of the selected worlds."""
        model = self.model
        count = model.tendon_count
        model.tendon_count = 0  # the tendon history is restored from the snapshot instead
        try:
            self.solver.reset(self.state_0, world_mask, flags=0)
        finally:
            model.tendon_count = count
        self.pipeline.reset_contact_matching(world_mask)

    # ---------------------------------------------------------------------------------------- public

    def get_observations(self) -> TensorDict:
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._compute_state()
        self.obs_buf = self._observations()
        return self.get_observations(), {}

    def reset_idx(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        cfg, n = self.cfg, self.num_envs
        ids = env_ids.to(self.device, torch.long)
        held = torch.zeros(len(ids), dtype=torch.bool, device=self.device)
        if self._snap_held is not None:
            u = torch.rand(len(ids), generator=self._torch_gen, device=self.device)
            held = (u < cfg.held_reset_fraction) & self.held_valid[ids]
        held_ids, ids = ids[held], ids[~held]
        if held_ids.numel() > 0:
            self._restore(self._snap_held, held_ids)
            self._reset_common(held_ids)
            self.actions[held_ids] = self._held_actions[held_ids]
            self.prev_actions[held_ids] = self._held_actions[held_ids]
        if ids.numel() == 0:
            return
        # state of the whole world
        self._restore(self._snap_open, ids)
        # arm at home (+ optional noise)
        home = torch.tensor(cfg.arm_home, dtype=torch.float32, device=self.device).expand(len(ids), 7)
        if cfg.arm_home_noise > 0.0:
            home = home + cfg.arm_home_noise * (
                2.0 * torch.rand(len(ids), 7, generator=self._torch_gen, device=self.device) - 1.0
            )
        self.t_q_target[ids] = home
        self.t_q_ref[ids] = home
        self.t_qd_ref[ids] = 0.0
        if cfg.arm_home_noise > 0.0:
            self._place_arm(ids)
        # object pose on the table
        k = len(ids)
        rnd = lambda: 2.0 * torch.rand(k, generator=self._torch_gen, device=self.device) - 1.0  # noqa: E731
        ob = self.layout.object_body
        pos = torch.zeros(k, 3, device=self.device)
        pos[:, 0] = cfg.object_nominal_xy[0] + cfg.object_xy_range * rnd()
        pos[:, 1] = cfg.object_nominal_xy[1] + cfg.object_xy_range * rnd()
        pos[:, 2] = TABLE_TOP + cfg.object_half_extent
        yaw = cfg.object_yaw_range * rnd()
        quat = torch.zeros(k, 4, device=self.device)
        quat[:, 2] = torch.sin(0.5 * yaw)
        quat[:, 3] = torch.cos(0.5 * yaw)
        self.body_q[ids, ob, :3] = pos
        self.body_q[ids, ob, 3:] = quat
        self.body_qd[ids, ob] = 0.0
        self._reset_common(ids)
        self.actions[ids] = 0.0
        self.prev_actions[ids] = 0.0
        self.t_command.view(n, -1)[ids] = self.tension_rest
        self.t_opp_target[ids] = cfg.opposition_range[0]

    def _reset_common(self, ids: torch.Tensor):
        """Lift target, scratch state, solver history and episode counters of freshly reset worlds."""
        cfg, n, k = self.cfg, self.num_envs, len(ids)
        for i, (lo, hi) in enumerate((cfg.target_x_range, cfg.target_y_range, cfg.target_z_range)):
            self.target_pos[ids, i] = lo + (hi - lo) * torch.rand(k, generator=self._torch_gen, device=self.device)
        # mirror into the scratch state and clear the solver history of these worlds
        s1q = wp.to_torch(self.state_1.body_q).view(n, self.B, 7)
        s1qd = wp.to_torch(self.state_1.body_qd).view(n, self.B, 6)
        s1q[ids] = self.body_q[ids]
        s1qd[ids] = self.body_qd[ids]
        mask = torch.zeros(n + 1, dtype=torch.bool, device=self.device)
        mask[ids] = True
        self._reset_solver_worlds(wp.from_torch(mask, dtype=wp.bool))
        self.episode_length_buf[ids] = 0

    def _place_arm(self, ids: torch.Tensor):
        """Write the link and palm poses of the arm reference (used when the reset pose is not the snapshot)."""
        # one zero-length drive call per world: the reference already equals the target
        wp.launch(
            arm_lib.arm_drive_kernel,
            dim=self.num_envs,
            inputs=[
                self.q_target,
                self._q_lower,
                self._q_upper,
                self._q_speed,
                self.cfg.arm_acceleration_limit,
                self._joint_xforms,
                self._mount,
                self._link_body,
                self._palm_body,
                self.model.body_com,
                self._floor_points,
                self._palm_floor,
                self._link_floor,
                self.sim_dt,
            ],
            outputs=[self.q_ref, self.qd_ref, self.arm_blocked, self.state_0.body_q, self.state_0.body_qd],
            device=self.wp_device,
        )

    def step(self, actions: torch.Tensor):
        cfg, n = self.cfg, self.num_envs
        a = torch.clamp(actions.to(self.device), -1.0, 1.0)
        self.prev_actions[:] = self.actions
        self.actions[:] = a
        # arm: relative joint targets around the current reference
        self.t_q_target[:] = self.t_q_ref + cfg.arm_action_scale * a[:, :7]
        # cables: absolute tension commands between the rest tension and the command ceiling
        u = 0.5 * (a[:, 7 : 7 + self.num_cables] + 1.0)
        self.t_command.view(n, -1)[:] = self.tension_rest + u * (cfg.tension_command_max - self.tension_rest)
        # thumb opposition angle
        lo, hi = cfg.opposition_range
        self.t_opp_target[:] = lo + 0.5 * (a[:, -1] + 1.0) * (hi - lo)

        self._simulate_step()
        self.episode_length_buf += 1
        self._compute_state()
        rewards, terminated, terms = self._rewards_and_terminations()
        truncated = self.episode_length_buf >= self.max_episode_length
        dones = terminated | truncated
        # episode logs
        log = {}
        for key, value in terms.items():
            self._episode_sums.setdefault(key, torch.zeros(n, device=self.device))
            self._episode_sums[key] += value
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            for key, s in self._episode_sums.items():
                log[f"Episode_Reward/{key}"] = (
                    s[done_ids] / (self.episode_length_buf[done_ids].float() * self.step_dt)
                ).mean()
                s[done_ids] = 0.0
            log["Episode_Termination/early"] = terminated[done_ids].float().mean()
            for key, flag in self._term_reasons.items():
                log[f"Episode_Termination/{key}"] = flag[done_ids].float().mean()
            log["Episode_Termination/time_out"] = (truncated & ~terminated)[done_ids].float().mean()
            log["Metrics/final_success"] = self._success[done_ids].float().mean()
            log["Metrics/final_object_height"] = (
                self._obj_pos[done_ids, 2] - TABLE_TOP - cfg.object_half_extent
            ).mean()
        log["Metrics/success"] = self._success.float().mean()
        log["Metrics/lifted"] = self._lifted.float().mean()
        log["Metrics/goal_distance"] = self._goal_dist.mean()
        self.extras = {"time_outs": truncated & ~terminated, "log": log}
        # values of this step before the finished worlds are reset (for evaluation scripts)
        self._step_lifted = self._lifted.clone()
        self._step_success = self._success.clone()
        self._step_goal_dist = self._goal_dist.clone()
        self._step_height = self._obj_pos[:, 2].clone()
        self.reset_idx(done_ids)
        if done_ids.numel() > 0:
            self._compute_state()
        self.obs_buf = self._observations()
        return self.get_observations(), rewards, dones, self.extras

    # ---------------------------------------------------------------------------------- MDP terms

    def _compute_state(self):
        """Derived quantities shared by the observations and the rewards (world frame = robot base frame)."""
        lay = self.layout
        bq = self.body_q
        self._palm_pos = bq[:, lay.hand.palm, :3]
        self._palm_quat = bq[:, lay.hand.palm, 3:]
        self._obj_pos = bq[:, lay.object_body, :3]
        self._obj_quat = bq[:, lay.object_body, 3:]
        self._obj_vel = self.body_qd[:, lay.object_body]
        self._site = self._palm_pos + _quat_apply(self._palm_quat, self.grasp_site.expand_as(self._palm_pos))
        tip_q = bq[:, self.fingertip_idx]
        self._tips = tip_q[..., :3] + _quat_apply(tip_q[..., 3:], self.tip_offset.expand(self.num_envs, -1, -1))
        pad_pts = self.palm_pad_points.expand(self.num_envs, -1, -1)
        palm_q = self._palm_quat.unsqueeze(1).expand(-1, pad_pts.shape[1], -1)
        self._palm_pad_z = (self._palm_pos.unsqueeze(1) + _quat_apply(palm_q, pad_pts))[..., 2]
        self._goal_dist = torch.linalg.norm(self._obj_pos - self.target_pos, dim=-1)
        rest_z = TABLE_TOP + self.cfg.object_half_extent
        tips = self._tips[:, self.pinch_tip_idx]
        self._pinch_dist = torch.linalg.norm(tips - self._obj_pos.unsqueeze(1), dim=-1).mean(dim=-1)
        # held: high enough and between the pinch digits (a cube batted into the air does not count)
        self._lifted = (self._obj_pos[:, 2] > rest_z + self.cfg.lift_height) & (
            self._pinch_dist < self.cfg.held_distance
        )
        self._success = self._goal_dist < self.cfg.success_threshold

    def _observations(self) -> torch.Tensor:
        jq = self.joint_q
        home = torch.tensor(self.cfg.arm_home, dtype=torch.float32, device=self.device)
        tips_rel = (self._tips - self._obj_pos.unsqueeze(1)).reshape(self.num_envs, -1)
        obs = torch.cat(
            [
                self.t_q_ref - home,  # 7
                0.1 * self.t_qd_ref,  # 7
                jq[:, self.q_flex_idx],  # 15 flexion angles [rad]
                jq[:, self.q_opp_idx : self.q_opp_idx + 1],  # 1 opposition angle [rad]
                30.0 * self.t_travel,  # 7 slider travels [m] * 30
                self.t_drive_force / 60.0,  # 7 cable tensions (drive forces) / limit
                self._palm_pos,  # 3
                self._palm_quat,  # 4
                tips_rel,  # 15 fingertip pads relative to the object [m]
                self._obj_pos,  # 3
                self._obj_quat,  # 4
                self._obj_vel,  # 6
                self._obj_pos - self._site,  # 3
                self.target_pos,  # 3
                self.target_pos - self._obj_pos,  # 3
                self.actions,  # 15
            ],
            dim=-1,
        )
        return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).clamp(-10.0, 10.0)

    def _rewards_and_terminations(self):
        cfg, w = self.cfg, self.cfg.rewards
        d_site = torch.linalg.norm(self._obj_pos - self._site, dim=-1)
        d_tips = self._pinch_dist
        lifted = self._lifted.float()
        penetration = torch.clamp(TABLE_TOP - (self._palm_pad_z.min(dim=-1).values - self.palm_pad_radius), min=0.0)
        finite = torch.isfinite(self.body_q.view(self.num_envs, -1)).all(dim=-1)
        off_table = (self._obj_pos[:, 2] < TABLE_TOP - 0.05) | (torch.abs(self._obj_pos[:, 1]) > 0.75)
        off_table |= (self._obj_pos[:, 0] < -0.95) | (self._obj_pos[:, 0] > -0.15)
        q_flex = self.joint_q[:, self.q_flex_idx]
        m = cfg.finger_angle_margin
        hand_invalid = ((q_flex < self.q_flex_lower - m) | (q_flex > self.q_flex_upper + m)).any(dim=-1)
        hand_invalid |= ~torch.isfinite(q_flex).all(dim=-1)
        deep = penetration > 0.03
        terminated = off_table | deep | hand_invalid | ~finite
        self._term_reasons = {
            "object_off_table": off_table,
            "palm_in_table": deep,
            "hand_invalid": hand_invalid,
            "non_finite": ~finite,
        }
        terms = {
            "reaching": w.reaching * (1.0 - torch.tanh(d_site / 0.1)),
            "reaching_fine": w.reaching_fine * (1.0 - torch.tanh(d_site / 0.02)),
            "fingertips": w.fingertips * (1.0 - torch.tanh(d_tips / 0.05)),
            "lifting": w.lifting * lifted,
            "goal_tracking": w.goal_tracking * lifted * (1.0 - torch.tanh(self._goal_dist / 0.3)),
            "goal_tracking_fine": w.goal_tracking_fine * lifted * (1.0 - torch.tanh(self._goal_dist / 0.05)),
            "action_rate": w.action_rate * torch.sum((self.actions - self.prev_actions) ** 2, dim=-1),
            "arm_velocity": w.arm_velocity * torch.sum(self.t_qd_ref**2, dim=-1),
            "table_penetration": w.table_penetration * penetration,
            "termination": w.termination * terminated.float(),
        }
        terms = {k: torch.nan_to_num(v, nan=0.0) * self.step_dt for k, v in terms.items()}
        reward = torch.stack(list(terms.values()), dim=0).sum(dim=0)
        return reward, terminated, terms

    def close(self):
        self._graph = None


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors ``v`` by quaternions ``q`` (``x, y, z, w``)."""
    xyz = q[..., :3]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + q[..., 3:4] * t + torch.cross(xyz, t, dim=-1)

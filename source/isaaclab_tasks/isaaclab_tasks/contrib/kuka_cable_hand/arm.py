# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""KUKA LBR iiwa 7 arm for the cable-driven hand: kinematics, mount and the prescribed-motion driver.

The arm is read from the Kuka-Allegro USD of the Isaac Lab asset server (the asset of the
``Isaac-Lift-KukaAllegro`` task) with the Allegro hand left out. Only the kinematic chain and the
visual meshes are used: the arm moves on prescribed joint trajectories (a stiff position-controlled
industrial arm), the cable-driven hand and the object are simulated by VBD.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import newton
import numpy as np
import warp as wp

KUKA_USD_RELATIVE_PATH = "Robots/KukaAllegro/kuka.usd"
"""Path of the Kuka-Allegro USD below ``ISAACLAB_NUCLEUS_DIR``."""

ARM_JOINT_COUNT = 7

IIWA7_VELOCITY_LIMIT = np.array([1.7104, 1.7104, 1.7453, 2.2689, 2.4435, 3.1416, 3.1416])
"""Joint speed limits of the LBR iiwa 7 R800 [rad/s] (the values of the Kuka-Allegro asset)."""

MOUNT_OFFSET = 0.071
"""Distance of the hand's wrist flange from the link-7 frame along the link-7 axis [m] (the Allegro mount plane)."""

PALM_FLANGE_IN_PALM = (-0.090, -0.0035, 0.0035)
"""Center of the rear face of the cable hand's wrist flange in the palm frame [m]."""

PALM_AXES_IN_LINK7 = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]).T
"""Palm axes as columns in the link-7 frame: palm +x (distal) along the flange axis, palm +y (palmar)
along link-7 -y, palm +z (thumb side) along link-7 +x."""


def quat_to_matrix(q) -> np.ndarray:
    """Rotation matrix of a quaternion ``(x, y, z, w)``."""
    x, y, z, w = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """Quaternion ``(x, y, z, w)`` of a rotation matrix."""
    q = wp.quat_from_matrix(wp.mat33(*np.asarray(m, dtype=float).flatten().tolist()))
    return np.array([q[0], q[1], q[2], q[3]], dtype=float)


def make_transform(p, r: np.ndarray) -> np.ndarray:
    """4x4 homogeneous transform."""
    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = p
    return t


def transform_to_pq(t: np.ndarray) -> np.ndarray:
    """Position and quaternion ``(x, y, z, qx, qy, qz, qw)`` of a 4x4 transform."""
    return np.concatenate([t[:3, 3], matrix_to_quat(t[:3, :3])])


@dataclass
class ArmVisual:
    """One visual mesh shape of an arm link."""

    link: int
    mesh: newton.Mesh
    xform: wp.transform
    scale: tuple[float, float, float]
    color: tuple[float, float, float] | None


@dataclass
class ArmModel:
    """Kinematic chain and visuals of the iiwa 7 read from USD."""

    joint_xforms: np.ndarray
    """Pose of every joint frame in its parent link frame, shape ``(7, 4, 4)``; the joints turn about their z axis."""
    lower: np.ndarray
    """Lower joint limits [rad], shape ``(7,)``."""
    upper: np.ndarray
    """Upper joint limits [rad], shape ``(7,)``."""
    link_mass: np.ndarray
    """Mass of links 0..7 [kg], shape ``(8,)``."""
    link_com: np.ndarray
    """Center of mass of links 0..7 in the link frame [m], shape ``(8, 3)``."""
    link_inertia: np.ndarray
    """Inertia of links 0..7 about the center of mass [kg m^2], shape ``(8, 3, 3)``."""
    visuals: list[ArmVisual] = field(default_factory=list)
    mount: np.ndarray = field(default_factory=lambda: np.eye(4))
    """Pose of the palm frame in the link-7 frame (4x4)."""

    def link_transforms(self, q, base: np.ndarray | None = None) -> np.ndarray:
        """World poses of links 0..7 for joint angles ``q``, shape ``(8, 4, 4)``."""
        x = np.eye(4) if base is None else np.array(base, dtype=float)
        out = [x.copy()]
        for j in range(ARM_JOINT_COUNT):
            c, s = np.cos(q[j]), np.sin(q[j])
            rz = np.array([[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
            x = x @ self.joint_xforms[j] @ rz
            out.append(x.copy())
        return np.array(out)

    def palm_transform(self, q, base: np.ndarray | None = None) -> np.ndarray:
        """World pose of the palm frame (4x4)."""
        return self.link_transforms(q, base)[-1] @ self.mount


def default_mount() -> np.ndarray:
    """Palm frame in the link-7 frame: in line with the flange, palmar side along link-7 -y."""
    r = PALM_AXES_IN_LINK7
    flange_in_link7 = np.array([0.0, 0.0, MOUNT_OFFSET])
    t = flange_in_link7 - r @ np.asarray(PALM_FLANGE_IN_PALM)
    return make_transform(t, r)


def load_arm(usd_path: str, with_visuals: bool = True) -> ArmModel:
    """Read the iiwa 7 chain (and optionally its visual meshes) from the Kuka-Allegro USD.

    Args:
        usd_path: Local path of ``kuka.usd`` (see :func:`resolve_kuka_usd`).
        with_visuals: Whether to keep the visual meshes of the links.

    Returns:
        The arm model.
    """
    b = newton.ModelBuilder()
    b.add_usd(usd_path, ignore_paths=[".*/ee_link.*", ".*allegro_mount.*"])
    labels = list(b.body_label)
    links = [labels.index(f"/kuka/iiwa7_link_{i}") for i in range(8)]
    joint_xforms, lower, upper = [], [], []
    for j in range(1, 8):
        joint = list(b.joint_label).index(f"/kuka/iiwa7_link_{j - 1}/iiwa7_joint_{j}")
        if b.joint_parent[joint] != links[j - 1] or b.joint_child[joint] != links[j]:
            raise RuntimeError(f"unexpected topology of iiwa7_joint_{j}")
        xp = np.array(b.joint_X_p[joint], dtype=float)
        xc = np.array(b.joint_X_c[joint], dtype=float)
        if not np.allclose(xc, [0, 0, 0, 0, 0, 0, 1], atol=1e-6):
            raise RuntimeError("the iiwa joints are expected to have the child frame at the link origin")
        dof = b.joint_qd_start[joint]
        if not np.allclose(b.joint_axis[dof], (0.0, 0.0, 1.0)):
            raise RuntimeError("the iiwa joints are expected to turn about their z axis")
        joint_xforms.append(make_transform(xp[:3], quat_to_matrix(xp[3:])))
        lower.append(b.joint_limit_lower[dof])
        upper.append(b.joint_limit_upper[dof])
    mass = np.array([b.body_mass[i] for i in links])
    com = np.array([np.array(b.body_com[i], dtype=float) for i in links])
    inertia = np.array([np.array(b.body_inertia[i], dtype=float).reshape(3, 3) for i in links])
    visuals = []
    if with_visuals:
        for s in range(b.shape_count):
            body = b.shape_body[s]
            if body not in links or b.shape_type[s] != newton.GeoType.MESH:
                continue
            if not (int(b.shape_flags[s]) & int(newton.ShapeFlags.VISIBLE)):
                continue
            color = b.shape_color[s] if hasattr(b, "shape_color") else None
            visuals.append(
                ArmVisual(
                    link=links.index(body),
                    mesh=b.shape_source[s],
                    xform=b.shape_transform[s],
                    scale=tuple(float(v) for v in b.shape_scale[s]),
                    color=None if color is None else tuple(float(c) for c in color),
                )
            )
    return ArmModel(
        joint_xforms=np.array(joint_xforms),
        lower=np.array(lower),
        upper=np.array(upper),
        link_mass=mass,
        link_com=com,
        link_inertia=inertia,
        visuals=visuals,
        mount=default_mount(),
    )


def resolve_kuka_usd() -> str:
    """Download (or reuse the cached copy of) the Kuka-Allegro USD from the Isaac Lab asset server."""
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

    return retrieve_file_path(f"{ISAACLAB_NUCLEUS_DIR}/{KUKA_USD_RELATIVE_PATH}")


def solve_home_pose(
    arm: ArmModel,
    site_in_palm: np.ndarray,
    site_target: np.ndarray,
    palm_rotation: np.ndarray,
    q0,
    rotation_weight: float = 0.3,
) -> np.ndarray:
    """Joint angles that put a palm site at a world point with a given palm orientation (least squares).

    Args:
        arm: The arm.
        site_in_palm: Point in the palm frame [m].
        site_target: World position of that point [m].
        palm_rotation: Desired world orientation of the palm frame (3x3).
        q0: Initial guess [rad].
        rotation_weight: Weight of the orientation residual [m/rad].

    Returns:
        The joint angles [rad], within the limits.
    """
    from scipy.optimize import least_squares

    def residual(q):
        t = arm.palm_transform(q)
        p = t[:3, :3] @ site_in_palm + t[:3, 3]
        r_err = 0.5 * (np.cross(t[:3, 0], palm_rotation[:, 0]) + np.cross(t[:3, 1], palm_rotation[:, 1]))
        r_err += 0.5 * np.cross(t[:3, 2], palm_rotation[:, 2])
        return np.concatenate([p - site_target, rotation_weight * r_err, 1e-3 * (np.asarray(q) - np.asarray(q0))])

    res = least_squares(residual, np.asarray(q0, dtype=float), bounds=(arm.lower + 0.05, arm.upper - 0.05))
    return res.x


# --- prescribed-motion driver ----------------------------------------------------------------------------


vec7 = wp.types.vector(length=7, dtype=float)


@wp.func
def _rot_z(angle: float) -> wp.transform:
    return wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle))


@wp.func
def _write_kinematic_pose(
    body: int,
    pose: wp.transform,
    dt: float,
    body_com: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Write a kinematic body pose and the velocity that moved it there within ``dt``."""
    old = body_q[body]
    com = body_com[body]
    r0 = wp.transform_get_rotation(old)
    r1 = wp.transform_get_rotation(pose)
    c0 = wp.transform_point(old, com)
    c1 = wp.transform_point(pose, com)
    turn = wp.mul(r1, wp.quat_inverse(r0))
    if turn[3] < 0.0:
        turn = wp.quat(-turn[0], -turn[1], -turn[2], -turn[3])
    axis = wp.vec3(0.0, 0.0, 0.0)
    angle = float(0.0)
    wp.quat_to_axis_angle(turn, axis, angle)
    body_q[body] = pose
    body_qd[body] = wp.spatial_vector((c1 - c0) / dt, axis * (angle / dt))


@wp.kernel
def arm_drive_kernel(
    q_target: wp.array2d[float],
    q_lower: wp.array[float],
    q_upper: wp.array[float],
    speed_limit: wp.array[float],
    acceleration_limit: float,
    joint_xforms: wp.array[wp.transform],
    mount: wp.transform,
    link_body: wp.array2d[int],
    palm_body: wp.array[int],
    body_com: wp.array[wp.vec3],
    palm_points: wp.array[wp.vec3],
    palm_clearance: wp.array[float],
    table_top: float,
    table_x: wp.vec2,
    table_half_y: float,
    link_clearance: float,
    dt: float,
    q_ref: wp.array2d[float],
    qd_ref: wp.array2d[float],
    blocked: wp.array[int],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Advance the joint reference of one arm toward its target and place the links and the palm.

    The reference follows a trapezoidal profile: its speed stays below ``speed_limit``, its
    acceleration below ``acceleration_limit``, and it brakes to stop at the target; it stays within
    the joint limits. The hand rides on the palm, and a reference that jumped in speed would fling
    the fingers past their soft stops. The link and palm poses follow from the forward kinematics of
    the reference; their velocities are the motion of this call over ``dt``. Run once per substep,
    after the hand control has measured the slider travels against the palm (all poses of a substep
    must belong to the same instant).

    Safety floor: a step is not taken when it would bring a palm point closer to the table top than
    its clearance (the arm is stronger than any contact and would crush the hand into the table), or
    the origin of link 4, 5, 6 or 7 closer than ``link_clearance`` while it is above the table (the
    arm links do not collide).
    """
    w = wp.tid()
    base = body_q[link_body[w, 0]]  # the base stays where it was built
    dv = acceleration_limit * dt
    q_new = vec7()
    v_new = vec7()
    for j in range(7):
        goal = wp.clamp(q_target[w, j], q_lower[j], q_upper[j])
        q = q_ref[w, j]
        v = qd_ref[w, j]
        e = goal - q
        v_des = wp.sign(e) * wp.min(speed_limit[j], wp.sqrt(2.0 * acceleration_limit * wp.abs(e)))
        v = v + wp.clamp(v_des - v, -dv, dv)
        if wp.abs(e) <= wp.abs(v) * dt and wp.abs(v) <= dv:
            v = e / dt  # settle on the target
        q_new[j] = wp.clamp(q + v * dt, q_lower[j], q_upper[j])
        v_new[j] = (q_new[j] - q) / dt
    # safety floor
    x = base
    ok = True
    for j in range(7):
        x = wp.transform_multiply(wp.transform_multiply(x, joint_xforms[j]), _rot_z(q_new[j]))
        p = wp.transform_get_translation(x)
        over_table = p[0] > table_x[0] and p[0] < table_x[1] and wp.abs(p[1]) < table_half_y
        if j >= 3 and over_table and p[2] < table_top + link_clearance:
            ok = False
    palm = wp.transform_multiply(x, mount)
    for k in range(palm_points.shape[0]):
        if wp.transform_point(palm, palm_points[k])[2] < table_top + palm_clearance[k]:
            ok = False
    if not ok:
        for j in range(7):
            q_new[j] = q_ref[w, j]
            v_new[j] = 0.0
        blocked[w] = blocked[w] + 1
    x = base
    for j in range(7):
        q_ref[w, j] = q_new[j]
        qd_ref[w, j] = v_new[j]
        x = wp.transform_multiply(wp.transform_multiply(x, joint_xforms[j]), _rot_z(q_new[j]))
        _write_kinematic_pose(link_body[w, j + 1], x, dt, body_com, body_q, body_qd)
    _write_kinematic_pose(palm_body[w], wp.transform_multiply(x, mount), dt, body_com, body_q, body_qd)

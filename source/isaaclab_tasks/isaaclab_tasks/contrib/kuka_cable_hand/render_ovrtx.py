# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render recorded rollouts (``record.py``) with OVRTX, the kitless Omniverse RTX renderer.

The scene is authored as USD and rendered by ``ovrtx`` through an ``ovstage`` stage:

- the KUKA LBR iiwa 7 with the original OmniPBR (MDL) materials of the Isaac Lab Kuka-Allegro asset: the eight
  link meshes and their materials are copied from ``Robots/KukaAllegro/kuka.usd`` into an arm-only USD (no
  Allegro hand, no physics) that is referenced once per environment;
- the cable-driven hand from ``cable_hand_visuals.usd`` of the Newton branch (its own looks), one prim per body;
- the cube, the lift target, the task table at its simulated size and place (a light-oak top on legs, in front
  of the arms, which stand on the floor on a mounting plate, as in Isaac Lab's Kuka-Allegro lift task), and a
  light set (light floor and backdrop; dome, key, fill and rim lights, of which the fill and rim cast no shadows);
- the cables as linear ``BasisCurves``: in the x-ray view colored by role like the branch's examples and glowing
  with tension, otherwise neutral gray (they show through the openings of the hand).

Every frame writes the recorded world pose of every body to its prim (``omni:xform`` with
``omni:resetXformStack``), rewrites the cable curves, and renders a converged path-traced frame (the history is
cleared before each frame, so moving parts leave no ghosting). Frames are piped to ``ffmpeg``.

Needs the ``ovrtx`` extra (``ovrtx`` 0.7 and ``ovstage`` 0.4). The first run compiles the RTX shader cache in
``~/.cache/ovrtx`` and can take many minutes; later runs start in seconds.

Example::

    python -m isaaclab_tasks.contrib.kuka_cable_hand.record --checkpoint pretrained/model_1499.pt --out roll.npz
    python -m isaaclab_tasks.contrib.kuka_cable_hand.render_ovrtx --rollout roll.npz --out roll.mp4 --view angled
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time

import numpy as np

KUKA_LINKS = 8
"""Number of iiwa link meshes (links 0 to 7)."""

XRAY_OPACITY = {"cover": 0.15, "housing": 0.15, "tip": 0.35, "pad": 0.5}
"""Opacity of the hand looks drawn as clear shells in the x-ray view (the black link frames and the actuators
stay opaque; the pads keep enough body to show the contacts on the cube)."""

THEME = {
    "floor": (0.80, 0.81, 0.83),
    "table": (0.45, 0.36, 0.26),
    "cube": (0.03, 0.18, 0.70),
    "target": (0.02, 0.42, 0.10),
    "background": (0.94, 0.94, 0.94),
    "dome": 450.0,
    "key": 1500.0,
    "fill": 300.0,
    "rim": 350.0,
    "mount": (0.15, 0.15, 0.16),
}
"""Light theme: colors (linear RGB) and light intensities. Near-white floor and backdrop, a light-oak table, a
key light from the upper front left, a soft fill from the right and a weak rim light from behind that keeps the
white hand shells apart from the white set."""

SIDE_VIEW = (100.0, 20.0, 1.55, (-0.33, 0.0, 0.38))
"""Azimuth and elevation [deg], distance [m] and look-at point [m] of the ``side`` view."""

TABLE_LEG_HALF_WIDTH = 0.025
"""Half width of the table legs [m] (render only; the simulated table is the top slab)."""

TABLE_LEG_INSET = 0.035
"""Distance of the legs from the edges of the table top [m]."""

MOUNT_PLATE_RADIUS = 0.18
"""Radius of the floor plate under each arm base [m]; the iiwa base flange reaches 0.13 m."""

TARGET_HIDE_DISTANCE = 0.03
"""The target marker hides once the cube center comes this close to it [m] (the goal is reached)."""

TARGET_SHOW_DISTANCE = 0.06
"""A hidden target marker shows again only if the cube moves this far away from it [m] or a new episode starts."""


def target_visibility(data, meta: dict, frames: int) -> np.ndarray:
    """Per frame and environment, whether the target marker is drawn (with hysteresis, so it never blinks)."""
    cube = data["body_q"][:frames, :, meta["object_body"], :3]
    dist = np.linalg.norm(cube - data["targets"][:frames], axis=-1)
    visible = np.ones(dist.shape, dtype=bool)
    shown = np.ones(dist.shape[1], dtype=bool)
    for i in range(frames):
        if i > 0:
            shown |= np.any(data["targets"][i] != data["targets"][i - 1], axis=-1)  # a new episode
        shown = np.where(shown, dist[i] >= TARGET_HIDE_DISTANCE, dist[i] > TARGET_SHOW_DISTANCE)
        visible[i] = shown
    return visible


# ----------------------------------------------------------------------------------------------- arm USD


def build_arm_usd(kuka_usd: str, out_path: str) -> str:
    """Write an arm-only USD of the iiwa 7 with the asset's own materials.

    ``/kuka_arm`` (default prim) holds ``Looks/<material>`` (copied verbatim) and ``iiwa7_link_i/visuals/mesh``
    for links 0 to 7 (meshes copied verbatim, bindings re-targeted to the copied materials). Drive
    ``/kuka_arm/iiwa7_link_i`` with the world pose of Newton body ``iiwa7_link_i``.

    Args:
        kuka_usd: Local path of the Kuka-Allegro ``kuka.usd`` (see :func:`arm.resolve_kuka_usd`).
        out_path: Output file (``.usd``).

    Returns:
        ``out_path``.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    src = Usd.Stage.Open(kuka_usd, Usd.Stage.LoadAll)
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())
    base_inv = xcache.GetLocalToWorldTransform(src.GetPrimAtPath("/kuka")).GetInverse()

    def defining_spec(prim):
        specs = [s for s in prim.GetPrimStack() if s.specifier == Sdf.SpecifierDef]
        if len(specs) != 1:
            raise RuntimeError(f"{prim.GetPath()}: expected one defining spec, got {len(specs)}")
        return specs[0]

    if os.path.exists(out_path):
        os.remove(out_path)
    dst = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(dst, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(dst, 1.0)
    root = UsdGeom.Xform.Define(dst, "/kuka_arm")
    dst.SetDefaultPrim(root.GetPrim())
    layer = dst.GetRootLayer()
    UsdGeom.Scope.Define(dst, "/kuka_arm/Looks")
    copied = {}
    for i in range(KUKA_LINKS):
        link = src.GetPrimAtPath(f"/kuka/iiwa7_link_{i}")
        vis = link.GetChild("visuals")
        meshes = [p for p in Usd.PrimRange(vis, Usd.TraverseInstanceProxies()) if p.IsA(UsdGeom.Mesh)]
        if not meshes:
            raise RuntimeError(f"{link.GetPath()} has no visual mesh")
        link_dst = UsdGeom.Xform.Define(dst, f"/kuka_arm/iiwa7_link_{i}")
        link_dst.AddTransformOp().Set(xcache.GetLocalToWorldTransform(link) * base_inv)
        vis_dst = UsdGeom.Xform.Define(dst, f"/kuka_arm/iiwa7_link_{i}/visuals")
        local = UsdGeom.Xformable(vis).GetLocalTransformation()
        if not _is_identity(local):
            vis_dst.AddTransformOp().Set(local)
        for k, mesh in enumerate(meshes):
            material, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
            if not material:
                raise RuntimeError(f"{mesh.GetPath()} has no material in the source asset")
            name = material.GetPath().name
            if name not in copied:
                spec = defining_spec(material.GetPrim())
                copied[name] = Sdf.Path(f"/kuka_arm/Looks/{name}")
                if not Sdf.CopySpec(spec.layer, spec.path, layer, copied[name]):
                    raise RuntimeError(f"copying material {spec.path} failed")
            spec = defining_spec(mesh)
            mesh_path = vis_dst.GetPath().AppendChild("mesh" if len(meshes) == 1 else f"mesh_{k}")
            if not Sdf.CopySpec(spec.layer, spec.path, layer, mesh_path):
                raise RuntimeError(f"copying mesh {spec.path} failed")
            prim = dst.GetPrimAtPath(mesh_path)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(dst.GetPrimAtPath(copied[name])))
    layer.Save()
    return out_path


def _is_identity(m, tol: float = 1e-12) -> bool:
    return all(abs(m[r][c] - (1.0 if r == c else 0.0)) <= tol for r in range(4) for c in range(4))


# --------------------------------------------------------------------------------------------- scene USD


def usd_matrices(poses: np.ndarray) -> np.ndarray:
    """``(N, 7)`` poses ``(px, py, pz, qx, qy, qz, qw)`` -> ``(N, 4, 4)`` USD row-vector matrices (float64)."""
    poses = np.asarray(poses, dtype=np.float64)
    x, y, z, w = poses[:, 3], poses[:, 4], poses[:, 5], poses[:, 6]
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    rot = np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )
    out = np.zeros((len(poses), 4, 4))
    out[:, :3, :3] = np.swapaxes(rot, -1, -2)
    out[:, 3, :3] = poses[:, :3]
    out[:, 3, 3] = 1.0
    return out


def look_at(eye, target) -> np.ndarray:
    """Camera-to-world USD row-vector matrix of a z-up camera at ``eye`` looking at ``target``."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    f = target - eye
    f /= np.linalg.norm(f)
    x = np.cross(f, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3], m[3, :3] = x, y, -f, eye
    return m


PINCH_POINT_IN_PALM = (0.0535, 0.095, 0.022)
"""Pinch point of the task in the palm frame [m]; the tracking camera follows it."""

TRACK_MARGIN = 0.12
"""Largest height [m] of the cube below, and of the shown lift target above, the tracking camera's look point
(half the frame height at the track distance of the x-ray clip, 0.48 m, is 0.17 m)."""

CABLE_GLOW_TENSION = 45.0
"""Tension [N] at which a cable glows fully: the largest tension the policy commands (``tension_command_max``)."""

CABLE_NEUTRAL = (0.46, 0.47, 0.49)
"""Color of the cables outside the x-ray view, where only short stretches show through the hand's openings."""

CABLE_GLOW_XRAY = 1.0
"""Default glow of a fully loaded cable in the x-ray view, relative to its base color (no glow otherwise)."""


def camera_path(
    view: str,
    data,
    meta: dict,
    frames: int,
    spacing: float,
    orbit: float,
    distance: float | None,
    elevation: float | None = None,
    azimuth: float | None = None,
):
    """Eye and target of every frame for a camera preset.

    ``angled``, ``side``, ``front`` and ``wide`` are fixed views of the whole scene. ``track`` follows the pinch
    point of environment 0 from a fixed direction, raised or lowered to keep the cube and the shown lift target in
    view (:data:`TRACK_MARGIN`) and smoothed over about one second per episode. ``orbit`` sweeps the azimuth by that
    many degrees over the clip; ``distance`` [m], ``elevation`` and ``azimuth`` [deg] override the preset's.
    """
    n = meta["num_envs"]
    span = spacing * max(n - 1, 0)
    targets = None
    if view == "angled":
        # 22 deg keeps the far floor, where grazing path-tracing noise crawls, out of the top of the frame
        az, el, dist, target = 36.0, 22.0, 1.05 + 0.7 * span, np.array([-0.56, -0.02, 0.50])
    elif view == "side":
        # the whole robot from beside the table: the base stands on the floor next to the table, which hides it from
        # any view over the table (it is 4 cm behind the table's far edge, below the top)
        az, el, dist, target = SIDE_VIEW[0], SIDE_VIEW[1], SIDE_VIEW[2] + 0.7 * span, np.array(SIDE_VIEW[3])
    elif view == "front":
        az, el, dist, target = 0.0, 12.0, 1.05 + 0.75 * span, np.array([-0.56, 0.0, 0.38])
    elif view == "wide":
        az, el, dist, target = 30.0, 22.0, 0.975 + 0.75 * span, np.array([-0.36, -0.10 * span, 0.44])
    elif view == "track":
        az, el, dist, target = 40.0, 22.0, 0.55, None
        palm = data["body_q"][:frames, 0, meta["hand_bodies"]["palm"]].astype(np.float64)
        rot = usd_matrices(palm)[:, :3, :3]  # row vectors: point_world = point_local @ R + p
        look = palm[:, :3] + np.asarray(PINCH_POINT_IN_PALM) @ rot
        look[:, 2] += 0.06  # frame the hand above the pinch point
        # keep the cube, and the lift target while it shows, within TRACK_MARGIN of the look point's height
        shown = target_visibility(data, meta, frames)[:, 0]
        goal_z = np.where(shown, data["targets"][:frames, 0, 2], -np.inf)
        cube_z = data["body_q"][:frames, 0, meta["object_body"], 2]
        look[:, 2] = np.minimum(np.maximum(look[:, 2], goal_z - TRACK_MARGIN), cube_z + TRACK_MARGIN)
        off = env_offsets(n, spacing)[0]
        k = 15  # half window [frames]
        kernel = np.hanning(2 * k + 3)[1:-1]
        kernel /= kernel.sum()
        # smooth each episode on its own, so the camera cuts together with the scene at a reset
        # a done frame already shows the next episode (record.py stores the state after the auto-reset)
        cuts = [0, *(int(c) for c in np.flatnonzero(data["dones"][1:frames, 0]) + 1), frames]
        smooth = []
        for a, b in zip(cuts[:-1], cuts[1:], strict=True):
            seg = look[a:b]
            pad = np.concatenate([np.repeat(seg[:1], k, 0), seg, np.repeat(seg[-1:], k, 0)])
            smooth.append(np.stack([np.convolve(pad[:, j], kernel, mode="valid") for j in range(3)], -1))
        targets = np.concatenate(smooth) + off
    else:
        raise ValueError(f"unknown view {view!r}")
    if distance is not None:
        dist = distance
    if elevation is not None:
        el = elevation
    if azimuth is not None:
        az = azimuth
    el = min(max(el, -89.0), 89.0)  # look_at needs a forward direction that is not vertical
    path = []
    for i in range(frames):
        a = az + (orbit * (i / (frames - 1) - 0.5) if orbit and frames > 1 else 0.0)
        a, e = math.radians(a), math.radians(el)
        t = targets[i] if targets is not None else target
        eye = t + dist * np.array([-math.cos(e) * math.cos(a), -math.cos(e) * math.sin(a), math.sin(e)])
        path.append((eye, t))
    return path


def define_disc(stage, path: str, radius: float, height: float, segments: int = 128):
    """A closed cylinder along z, centered at the origin, as a polygon mesh with flat caps and a smooth side.

    Implicit ``UsdGeom.Cylinder`` prims are tessellated coarsely by the renderer (visible facets on a plate).
    """
    from pxr import Gf, UsdGeom, Vt

    ang = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    rim = np.stack([np.cos(ang), np.sin(ang), np.zeros(segments)], -1)
    h = 0.5 * height
    points = np.concatenate([radius * rim + [0.0, 0.0, h], radius * rim - [0.0, 0.0, h]])
    nxt = (np.arange(segments) + 1) % segments
    top, bottom = np.arange(segments), np.arange(segments)[::-1] + segments
    side = np.stack([np.arange(segments) + segments, nxt + segments, nxt, np.arange(segments)], -1)
    side_normals = np.stack([rim, rim[nxt], rim[nxt], rim], 1)
    normals = np.concatenate(
        [np.tile([0.0, 0.0, 1.0], (segments, 1)), np.tile([0.0, 0.0, -1.0], (segments, 1)), side_normals.reshape(-1, 3)]
    )
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([segments, segments] + [4] * segments))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(np.concatenate([top, bottom, side.reshape(-1)]).tolist()))
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(-radius, -radius, -h), Gf.Vec3f(radius, radius, h)]))
    return mesh


def env_offsets(num_envs: int, spacing: float) -> np.ndarray:
    """Display offsets of the environments: a row along y, centered on y = 0."""
    out = np.zeros((num_envs, 3))
    out[:, 1] = (np.arange(num_envs) - 0.5 * (num_envs - 1)) * spacing
    return out


def cable_curves(
    frame_pieces,
    num_envs: int,
    num_cables: int,
    roles,
    tension_max: float,
    radius: float,
    glow: float = 0.0,
    colored: bool = True,
):
    """Per environment: curve vertex counts, points, widths, colors and glow of the cables of one frame.

    The color is the branch's (hue by role, brightness by tension), or :data:`CABLE_NEUTRAL` if not ``colored``.
    The glow (emission) is ``glow`` times the base color of the role, scaled linearly from 0 at the rest tension
    to 1 at :data:`CABLE_GLOW_TENSION`, so loaded cables stand out regardless of the lighting.
    """
    from newton.examples.cable import cable_hand as ch

    start, end, tension, world, cable = frame_pieces
    out = []
    for w in range(num_envs):
        counts, points, colors, glows = [], [], [], []
        for c in range(num_cables):
            sel = (world == w) & (cable == c)
            if not np.any(sel):
                continue
            s, e, t = start[sel], end[sel], tension[sel]
            if colored:
                col = np.asarray(ch.tension_colors(t, tension_max, roles[c]), dtype=np.float32)
            else:
                col = np.tile(np.asarray(CABLE_NEUTRAL, dtype=np.float32), (len(t), 1))
            rest = ch.CABLE_COLOR_REST_TENSION
            u = np.clip((t - rest) / (CABLE_GLOW_TENSION - rest), 0.0, 1.0)
            base = np.asarray(ch.CABLE_COLORS.get(roles[c], ch.CABLE_COLORS["curl"]), dtype=np.float32)
            emit = (glow * u[:, None] * base).astype(np.float32)
            points.append(np.concatenate([s[:1], e]).astype(np.float32))
            colors.append(np.concatenate([col[:1], col]).astype(np.float32))
            glows.append(np.concatenate([emit[:1], emit]))
            counts.append(len(s) + 1)
        pts = np.concatenate(points) if points else np.zeros((0, 3), np.float32)
        cols = np.concatenate(colors) if colors else np.zeros((0, 3), np.float32)
        emits = np.concatenate(glows) if glows else np.zeros((0, 3), np.float32)
        out.append(
            (
                np.asarray(counts, np.int32),
                pts,
                np.full(len(pts), 2.0 * radius, np.float32),
                cols,
                emits,
            )
        )
    return out


def build_scene(
    path: str,
    arm_usd: str,
    hand_usd: str,
    meta: dict,
    first_curves,
    size: tuple[int, int],
    spp: int,
    xray: bool,
    spacing: float,
    cable_scale: float,
    target_radius: float = 0.008,
) -> dict:
    """Author the scene layer; returns the prim paths that are driven per frame."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdRender, UsdShade, Vt

    n = meta["num_envs"]
    offsets = env_offsets(n, spacing)
    st = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    st.SetDefaultPrim(UsdGeom.Xform.Define(st, "/World").GetPrim())
    T = Sdf.ValueTypeNames

    def preview(path, rgb, roughness=0.6, metallic=0.0, emissive=None, ior=None):
        mat = UsdShade.Material.Define(st, path)
        sh = UsdShade.Shader.Define(st, path + "/surface")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", T.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("roughness", T.Float).Set(float(roughness))
        sh.CreateInput("metallic", T.Float).Set(float(metallic))
        if ior is not None:
            sh.CreateInput("ior", T.Float).Set(float(ior))
        if emissive is not None:
            sh.CreateInput("emissiveColor", T.Color3f).Set(Gf.Vec3f(*emissive))
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return mat

    def box(path, center, half, material):
        cube = UsdGeom.Cube.Define(st, path)
        cube.CreateSizeAttr(2.0)
        xf = UsdGeom.Xformable(cube)
        xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
        xf.AddScaleOp().Set(Gf.Vec3f(*map(float, half)))
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
        return cube

    UsdGeom.Scope.Define(st, "/World/Looks")
    floor_mat = preview("/World/Looks/Floor", THEME["floor"], 1.0, ior=1.0)  # no grazing reflection
    mount_mat = preview("/World/Looks/Mount", THEME["mount"], 0.4, metallic=0.5)
    table_mat = preview("/World/Looks/Table", THEME["table"], 0.5)
    cube_mat = preview("/World/Looks/Cube", THEME["cube"], 0.35)
    target_mat = preview(
        "/World/Looks/Target", THEME["target"], 0.35, emissive=tuple(0.15 * c for c in THEME["target"])
    )
    cable_mat = UsdShade.Material.Define(st, "/World/Looks/Cable")
    reader = UsdShade.Shader.Define(st, "/World/Looks/Cable/DisplayColor")
    reader.CreateIdAttr("UsdPrimvarReader_float3")
    reader.CreateInput("varname", T.Token).Set("displayColor")
    reader.CreateInput("fallback", T.Float3).Set(Gf.Vec3f(0.2, 0.6, 0.9))
    reader.CreateOutput("result", T.Float3)
    cable_sh = UsdShade.Shader.Define(st, "/World/Looks/Cable/surface")
    cable_sh.CreateIdAttr("UsdPreviewSurface")
    cable_sh.CreateInput("diffuseColor", T.Color3f).ConnectToSource(reader.ConnectableAPI(), "result")
    cable_sh.CreateInput("roughness", T.Float).Set(0.45)
    glow_reader = UsdShade.Shader.Define(st, "/World/Looks/Cable/TensionGlow")
    glow_reader.CreateIdAttr("UsdPrimvarReader_float3")
    glow_reader.CreateInput("varname", T.Token).Set("tensionGlow")
    glow_reader.CreateInput("fallback", T.Float3).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    glow_reader.CreateOutput("result", T.Float3)
    cable_sh.CreateInput("emissiveColor", T.Color3f).ConnectToSource(glow_reader.ConnectableAPI(), "result")
    cable_mat.CreateSurfaceOutput().ConnectToSource(cable_sh.ConnectableAPI(), "surface")

    # a floor large enough that its far edge sits on the horizon (no visible corner)
    box("/World/Floor", (0.0, 0.0, -0.01), (200.0, 200.0, 0.01), floor_mat)
    # the task table as simulated (Isaac Lab's Kuka-Allegro lift table): its top slab at the true footprint, in front
    # of the arms, which stand on the floor; one long top under all environments, on legs
    tc, th = np.asarray(meta["table_center"]), np.asarray(meta["table_half_extents"])
    top = float(meta["table_top"])
    x0, x1 = tc[0] - th[0], tc[0] + th[0]
    y0, y1 = offsets[:, 1].min() - th[1], offsets[:, 1].max() + th[1]
    UsdGeom.Xform.Define(st, "/World/Table")
    box("/World/Table/top", (tc[0], 0.5 * (y0 + y1), top - th[2]), (th[0], 0.5 * (y1 - y0), th[2]), table_mat)
    under = top - 2.0 * th[2]  # underside of the top
    inset = TABLE_LEG_INSET + TABLE_LEG_HALF_WIDTH
    leg_y = np.linspace(y0 + inset, y1 - inset, max(2, int(np.ceil((y1 - y0 - 2 * inset) / 1.4)) + 1))
    for i, leg_x in enumerate((x0 + inset, x1 - inset)):
        for j, y in enumerate(leg_y):
            box(
                f"/World/Table/leg_{i}_{j}",
                (leg_x, y, 0.5 * under),
                (TABLE_LEG_HALF_WIDTH, TABLE_LEG_HALF_WIDTH, 0.5 * under),
                table_mat,
            )

    driven = {"bodies": [], "body_index": [], "env": [], "targets": [], "curves": []}
    half = float(meta["object_half_extent"])
    for w in range(n):
        env = f"/World/envs/env_{w}"
        UsdGeom.Xform.Define(st, env)
        arm = UsdGeom.Xform.Define(st, f"{env}/arm")
        arm.GetPrim().GetReferences().AddReference(arm_usd)
        for i, b in enumerate(meta["arm_links"]):
            driven["bodies"].append(f"{env}/arm/iiwa7_link_{i}")
            driven["body_index"].append(b)
            driven["env"].append(w)
        hand = UsdGeom.Xform.Define(st, f"{env}/hand")
        hand.GetPrim().GetReferences().AddReference(hand_usd)
        for key, b in meta["hand_bodies"].items():
            driven["bodies"].append(f"{env}/hand/{key}")
            driven["body_index"].append(b)
            driven["env"].append(w)
        if xray:  # clear, non-refracting shells
            for tag, opacity in XRAY_OPACITY.items():
                sh = UsdShade.Shader(st.OverridePrim(f"{env}/hand/looks/{tag}/surface"))
                sh.CreateInput("opacity", T.Float).Set(opacity)
                sh.CreateInput("roughness", T.Float).Set(0.1)
                sh.CreateInput("ior", T.Float).Set(1.0)
        plate = define_disc(st, f"{env}/mount", radius=MOUNT_PLATE_RADIUS, height=0.01)  # the arm's base on the floor
        UsdGeom.Xformable(plate).AddTranslateOp().Set(Gf.Vec3d(0.0, float(offsets[w, 1]), 0.005))
        UsdShade.MaterialBindingAPI.Apply(plate.GetPrim()).Bind(mount_mat)
        cube = UsdGeom.Cube.Define(st, f"{env}/cube")
        cube.CreateSizeAttr(2.0 * half)
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(cube_mat)
        driven["bodies"].append(f"{env}/cube")
        driven["body_index"].append(meta["object_body"])
        driven["env"].append(w)
        target = UsdGeom.Sphere.Define(st, f"{env}/target")
        target.CreateRadiusAttr(float(target_radius))
        UsdShade.MaterialBindingAPI.Apply(target.GetPrim()).Bind(target_mat)
        UsdGeom.PrimvarsAPI(target).CreatePrimvar("doNotCastShadows", T.Bool).Set(True)
        driven["targets"].append(f"{env}/target")
        counts, pts, widths, cols, emits = first_curves[w]
        curves = UsdGeom.BasisCurves.Define(st, f"{env}/cables")
        curves.CreateTypeAttr(UsdGeom.Tokens.linear)
        curves.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
        curves.CreateCurveVertexCountsAttr(Vt.IntArray(counts.tolist()))
        curves.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts + offsets[w].astype(np.float32)))
        curves.CreateWidthsAttr(Vt.FloatArray.FromNumpy(cable_scale * widths))
        curves.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
        pv = UsdGeom.PrimvarsAPI(curves).CreatePrimvar("displayColor", T.Color3fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec3fArray.FromNumpy(cols))
        glow_pv = UsdGeom.PrimvarsAPI(curves).CreatePrimvar("tensionGlow", T.Color3fArray, UsdGeom.Tokens.vertex)
        glow_pv.Set(Vt.Vec3fArray.FromNumpy(emits))
        UsdShade.MaterialBindingAPI.Apply(curves.GetPrim()).Bind(cable_mat)
        driven["curves"].append(f"{env}/cables")

    # lights: white dome (soft fill from everywhere), a soft key light from the front left and a weak fill
    UsdGeom.Scope.Define(st, "/World/Lights")
    dome = UsdLux.DomeLight.Define(st, "/World/Lights/Dome")
    dome.CreateIntensityAttr(THEME["dome"])
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    key = UsdLux.DistantLight.Define(st, "/World/Lights/Key")  # upper front left of the camera presets
    key.CreateIntensityAttr(THEME["key"])
    key.CreateAngleAttr(4.0)
    key.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))
    key.AddRotateXYZOp().Set(Gf.Vec3f(-40.0, 0.0, 81.0))
    fill = UsdLux.DistantLight.Define(st, "/World/Lights/Fill")
    fill.CreateIntensityAttr(THEME["fill"])
    fill.CreateAngleAttr(25.0)
    fill.AddRotateXYZOp().Set(Gf.Vec3f(-30.0, -40.0, 180.0))
    rim = UsdLux.DistantLight.Define(st, "/World/Lights/Rim")  # from behind the subjects
    rim.CreateIntensityAttr(THEME["rim"])
    rim.CreateAngleAttr(6.0)
    for light in (fill, rim):  # one shadow direction only: the key's
        UsdLux.ShadowAPI.Apply(light.GetPrim()).CreateShadowEnableAttr(False)
    rim.AddRotateXYZOp().Set(Gf.Vec3f(-38.0, 24.0, 0.0))

    # camera (posed per frame) and the render product
    cam = UsdGeom.Camera.Define(st, "/World/Camera")
    cam.CreateFocalLengthAttr(17.0)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateVerticalApertureAttr(20.955 * size[1] / size[0])
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    UsdGeom.Xformable(cam).AddTransformOp().Set(Gf.Matrix4d(1.0))
    product = UsdRender.Product.Define(st, "/Render/Product")
    product.CreateResolutionAttr(Gf.Vec2i(*size))
    product.CreateCameraRel().SetTargets([cam.GetPath()])
    var = UsdRender.Var.Define(st, "/Render/Vars/LdrColor")
    var.CreateSourceNameAttr("LdrColor")
    product.CreateOrderedVarsRel().SetTargets([var.GetPath()])
    p = product.GetPrim()
    for name, typ, value in [
        ("omni:rtx:rendermode", T.Token, "PathTracing"),
        ("omni:rtx:pt:samplesPerPixel", T.UInt, int(spp)),
        ("omni:rtx:pt:samplesPerIteration", T.Int, 8),
        ("omni:rtx:pt:denoising:enabled", T.Bool, True),
        ("omni:rtx:waitForEvents", T.TokenArray, ["AllLoadingFinished", "OnlyOnFirstRequest"]),
        ("omni:rtx:background:source:type", T.Token, "color"),
        ("omni:rtx:background:source:color", T.Color3f, Gf.Vec3f(*THEME["background"])),
        ("omni:rtx:rt:fractionalOpacity", T.Bool, bool(xray)),
        ("omni:rtx:pt:fractionalCutoutOpacity", T.Bool, bool(xray)),
        ("omni:rtx:dlss:frameGeneration", T.Bool, False),
    ]:
        p.CreateAttribute(name, typ).Set(value)
    st.GetRootLayer().Save()
    driven["camera"] = "/World/Camera"
    driven["product"] = "/Render/Product"
    driven["offsets"] = offsets
    return driven


# ---------------------------------------------------------------------------------------------- renderer


def render(
    rollout: str,
    out: str,
    view: str = "angled",
    size: tuple[int, int] = (1920, 1080),
    spp: int = 48,
    xray: bool = False,
    orbit: float = 0.0,
    spacing: float = 0.9,
    cable_scale: float = 1.5,
    max_frames: int | None = None,
    workdir: str | None = None,
    keep_frames: int = 0,
    distance: float | None = None,
    only_frames: list[int] | None = None,
    elevation: float | None = None,
    cable_glow: float | None = None,
    azimuth: float | None = None,
) -> dict:
    """Render a recorded rollout to an MP4; returns timing statistics."""
    import ovrtx
    import ovstage
    from newton.examples.cable import cable_hand_visuals

    from .arm import resolve_kuka_usd

    data = np.load(rollout)
    meta = json.loads(str(data["meta"]))
    n = meta["num_envs"]
    frames = len(data["body_q"]) if max_frames is None else min(max_frames, len(data["body_q"]))
    if data["dones"][frames - 1].any():
        frames -= 1  # the last recorded step already shows the next episode
    workdir = workdir or os.path.splitext(out)[0] + "_usd"
    os.makedirs(workdir, exist_ok=True)
    offs = data["piece_offset"]
    pieces = (data["piece_start"], data["piece_end"], data["piece_tension"], data["piece_world"], data["piece_cable"])

    def frame_pieces(i):
        a, b = offs[i], offs[i + 1]
        return tuple(x[a:b] for x in pieces)

    num_cables = len(meta["cable_keys"])
    glow = cable_glow if cable_glow is not None else (CABLE_GLOW_XRAY if xray else 0.0)
    curves_args = (n, num_cables, meta["cable_roles"], meta["tension_max"], meta["cable_radius"], glow, xray)
    workdir = os.path.abspath(workdir)
    out = os.path.abspath(out)
    arm_usd = os.path.join(os.path.expanduser("~/.cache/kuka_cable_hand"), "kuka_arm.usd")
    if not os.path.isfile(arm_usd):
        os.makedirs(os.path.dirname(arm_usd), exist_ok=True)
        build_arm_usd(resolve_kuka_usd(), arm_usd)
    hand_usd = os.path.join(
        os.path.dirname(cable_hand_visuals.__file__), "..", "assets", "cable_hand", "cable_hand_visuals.usd"
    )
    hand_usd = os.path.abspath(hand_usd)
    scene_path = os.path.join(workdir, "scene.usda")
    driven = build_scene(
        scene_path,
        arm_usd,
        hand_usd,
        meta,
        cable_curves(frame_pieces(0), *curves_args),
        size,
        spp,
        xray,
        spacing,
        cable_scale,
        target_radius=0.015 if view == "wide" else 0.008,
    )

    config = ovrtx.RendererConfig()
    config.log_level = "error"
    config.log_file_path = os.path.join(workdir, "ovrtx.log")
    config.texture_streaming_mode = ovrtx.TextureStreamingMode.SYNCHRONOUS
    t0 = time.perf_counter()
    renderer = ovrtx.Renderer(config=config)
    stage = ovstage.Stage("kuka_cable_hand_render")
    renderer.attach_ovstage(stage)
    if ovstage.population.open_usd_async(stage, scene_path, 1, 0.0).wait(int(600e9)) is None:
        raise RuntimeError(f"ovstage could not open {scene_path}")
    stage.advance_write_floor(1, ovstage.Scope.ALL).wait()
    paths = ovstage.PathDictionary(stage)
    lists = []

    def query(prim_paths):
        plist = paths.create_path_list_from_strings(prim_paths)
        lists.append(plist)
        return stage.query_from_path_list(plist)

    moving = driven["bodies"] + driven["targets"] + [driven["camera"]]
    q_moving = query(moving)
    q_curves = [query([p]) for p in driven["curves"]]
    xform_type = ovstage.numpy_to_dldatatype(np.dtype(np.float64), lanes=16)

    def write_array(q, name, arr, ordinal):
        arr = np.ascontiguousarray(arr)
        if arr.ndim > 1:
            lanes = int(np.prod(arr.shape[1:]))
            flat = arr.reshape(-1)
            arr = ovstage.make_dltensor(
                flat, dtype=ovstage.numpy_to_dldatatype(flat.dtype, lanes=lanes), shape=[len(flat) // lanes], ndim=1
            )
        stage.write_attribute(q, name, ordinal=ordinal, tensors=arr, is_array=True).wait()

    stage.write_attribute(
        q_moving, "omni:resetXformStack", ordinal=2, tensors=np.ones(len(moving), np.bool_), is_array=False
    ).wait()
    body_index = np.asarray(driven["body_index"])
    body_env = np.asarray(driven["env"])
    offsets = driven["offsets"]
    product = driven["product"]
    stills = bool(only_frames)
    ffmpeg = (
        None
        if stills
        else subprocess.Popen(
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
                str(round(1.0 / meta["step_dt"])),
                "-i",
                "-",
                "-vf",
                "scale=out_color_matrix=bt709:out_range=tv",  # HD video is BT.709; tag it so players agree
                "-colorspace",
                "bt709",
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "17",
                "-preset",
                "slow",
                "-movflags",
                "+faststart",  # playable while it downloads
                out,
            ],
            stdin=subprocess.PIPE,
        )
    )
    cameras = camera_path(view, data, meta, frames, spacing, orbit, distance, elevation, azimuth)
    target_visible = target_visibility(data, meta, frames)
    t_setup = time.perf_counter() - t0
    t_frames = []
    still_dir = os.path.join(workdir, "frames")
    if keep_frames or only_frames:
        os.makedirs(still_dir, exist_ok=True)
    selected = range(frames) if not stills else sorted({f for f in only_frames if 0 <= f < frames})
    if stills and len(selected) < len(set(only_frames)):
        print(f"[render_ovrtx] frames outside 0..{frames - 1} skipped", flush=True)
    for i in selected:
        t = time.perf_counter()
        ordinal = i + 2
        bq = data["body_q"][i]  # (envs, bodies, 7)
        poses = bq[body_env, body_index].astype(np.float64)
        poses[:, :3] += offsets[body_env]
        mats = usd_matrices(poses)
        tgt = np.zeros((n, 7))
        tgt[:, :3] = data["targets"][i] + offsets
        tgt[:, 6] = 1.0
        tgt[~target_visible[i], 2] = -1.0  # hidden below the floor while the goal is reached
        eye, look = cameras[i]
        cam = look_at(eye, look)[None]
        allm = np.concatenate([mats, usd_matrices(tgt), cam]).reshape(-1, 16)
        tensor = ovstage.make_dltensor(
            np.ascontiguousarray(allm).reshape(-1), dtype=xform_type, shape=[len(allm)], ndim=1
        )
        stage.write_attribute(
            q_moving,
            "omni:xform",
            ordinal=ordinal,
            tensors=tensor,
            is_array=False,
            semantic=ovstage.AttributeSemantic.MATRIX,
        ).wait()
        for w, (counts, pts, widths, cols, emits) in enumerate(cable_curves(frame_pieces(i), *curves_args)):
            write_array(q_curves[w], "curveVertexCounts", counts, ordinal)
            write_array(q_curves[w], "points", (pts + offsets[w]).astype(np.float32), ordinal)
            write_array(q_curves[w], "widths", (cable_scale * widths).astype(np.float32), ordinal)
            write_array(q_curves[w], "primvars:displayColor", cols, ordinal)
            write_array(q_curves[w], "primvars:tensionGlow", emits, ordinal)
        stage.advance_write_floor(ordinal, ovstage.Scope.ALL).wait()
        renderer.clear_history([product], first_frame_quality=True)
        result = renderer.step({product}, meta["step_dt"], ordinal=ordinal)
        frame = max(result[product].frames, key=lambda f: f.end_time)
        var = frame.render_vars["LdrColor"]
        with var.map(device=ovrtx.Device.CPU) as mapping:
            pixels = np.from_dlpack(mapping)
            rgb = pixels.reshape(size[1], size[0], -1)[..., :3].copy()
            del pixels  # no DLPack view may outlive the mapping
        del mapping, var, frame, result
        if ffmpeg is not None:
            ffmpeg.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
        if (keep_frames and i % keep_frames == 0) or only_frames:
            from PIL import Image

            Image.fromarray(rgb.astype(np.uint8)).save(os.path.join(still_dir, f"frame_{i:04d}.png"))
        t_frames.append(time.perf_counter() - t)
        if i % 30 == 0:
            print(f"[render_ovrtx] frame {i}/{frames}: {t_frames[-1]:.2f} s", flush=True)
    if ffmpeg is not None:
        ffmpeg.stdin.close()
        if ffmpeg.wait() != 0:
            raise RuntimeError(f"ffmpeg failed to write {out}")
    for q in [q_moving, *q_curves]:
        q.release().wait()
    for plist in lists:
        paths.destroy_path_list(plist)
    renderer.detach_ovstage()
    renderer.destroy()
    paths.destroy()
    stage.destroy()
    return {
        "frames": frames,
        "setup_s": round(t_setup, 2),
        "mean_frame_s": round(float(np.mean(t_frames)), 3) if t_frames else None,
        "out": out,
        "scene": scene_path,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rollout", required=True, help=".npz written by record.py")
    parser.add_argument("--out", required=True, help="Output .mp4")
    parser.add_argument("--view", choices=("angled", "side", "front", "wide", "track"), default="angled")
    parser.add_argument("--distance", type=float, default=None, help="Camera distance override [m].")
    parser.add_argument("--elevation", type=float, default=None, help="Camera elevation override [deg].")
    parser.add_argument("--azimuth", type=float, default=None, help="Camera azimuth override [deg].")
    parser.add_argument("--only_frames", type=int, nargs="+", default=None, help="Render only these frames (stills).")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--spp", type=int, default=48, help="Path-tracing samples per pixel per frame.")
    parser.add_argument("--xray", action="store_true", help="Translucent hand shells so the cables show.")
    parser.add_argument("--orbit", type=float, default=0.0, help="Camera azimuth sweep over the clip [deg].")
    parser.add_argument("--spacing", type=float, default=0.9, help="Distance between environments [m].")
    parser.add_argument("--cable_scale", type=float, default=1.5, help="Drawn cable width / real width.")
    parser.add_argument(
        "--cable_glow",
        type=float,
        default=None,
        help=f"Cable glow at full load (default {CABLE_GLOW_XRAY} with --xray, else 0).",
    )
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--keep_frames", type=int, default=0, help="Also save every k-th frame as PNG.")
    args = parser.parse_args()
    stats = render(
        args.rollout,
        args.out,
        view=args.view,
        size=(args.width, args.height),
        spp=args.spp,
        xray=args.xray,
        orbit=args.orbit,
        spacing=args.spacing,
        cable_scale=args.cable_scale,
        max_frames=args.max_frames,
        keep_frames=args.keep_frames,
        distance=args.distance,
        only_frames=args.only_frames,
        elevation=args.elevation,
        cable_glow=args.cable_glow,
        azimuth=args.azimuth,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

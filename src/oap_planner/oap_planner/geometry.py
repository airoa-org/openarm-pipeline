# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""The table and pillar clearance of a joint configuration: FK against the URDF's own collision geometry, with the
gripper as the planner's ellipsoid. Workspace values: config/workspace.yaml."""
from __future__ import annotations

import os

import yaml
from ament_index_python.packages import get_package_share_directory

SIDES = ("left", "right")
REFUSED = 3  # plan's exit code when it finds no safe path: the GUI tells the user to move the arms by hand
with open(os.path.join(get_package_share_directory("oap_planner"), "config", "workspace.yaml")) as _f:
    WORKSPACE = yaml.safe_load(_f)
SAFETY_MARGIN_M = float(WORKSPACE["safety_margin"])  # pillar
TABLE_MARGIN_M = float(WORKSPACE["table_margin"])  # table
TABLE_TOP = float(WORKSPACE["table"]["top_z"])
TABLE_X = tuple(float(v) for v in WORKSPACE["table"]["x"])
TABLE_Y = tuple(float(v) for v in WORKSPACE["table"]["y"])

# OpenArm v1.0 joint limits, kept independent of the control node.
LIMITS = {
    "left": [(-3.490659, +1.396263), (-3.316125, +0.174533), (-1.570796, +1.570796),
             (+0.000000, +2.443461), (-1.570796, +1.570796), (-0.785398, +0.785398),
             (-1.570796, +1.570796)],
    "right": [(-1.396263, +3.490659), (-0.174533, +3.316125), (-1.570796, +1.570796),
              (+0.000000, +2.443461), (-1.570796, +1.570796), (-0.785398, +0.785398),
              (-1.570796, +1.570796)],
}
READY_POSE = {
    "left":  [-1.0749, -0.9773, +1.1562, +1.6124, -1.3011, +0.0539, +0.5752],
    "right": [+1.0749, +0.9773, -1.1562, +1.6124, +1.3011, -0.0539, -0.5752],
}
# Conservative piecewise box approximation of the central pillar geometry.
PILLAR_SLABS = [
    (-1.0000, 0.0322, -0.1550, +0.0950, -0.0950, +0.0950),
    (0.0322, 0.0644, -0.0299, +0.0299, -0.0530, +0.0530),
    (0.0644, 0.6120, -0.0300, +0.0299, -0.0300, +0.0300),
    (0.6120, 0.6442, -0.0694, +0.0528, -0.0796, +0.0796),
    (0.6442, 0.6764, -0.0766, +0.0600, -0.0723, +0.0723),
    (0.6764, 0.7086, -0.0850, +0.0650, -0.0627, +0.0627),
    (0.7086, 0.7408, -0.0825, +0.0626, -0.0585, +0.0585),
    (0.7408, 0.7730, -0.0581, +0.0433, -0.0493, +0.0493),
]
PILLAR_LINKS = ("link3", "link4", "link5", "link6", "link7", "hand_root")  # checked against the pillar, the gripper too
CHECK_STEPS = 25  # interpolation samples per trajectory segment
FINGER_CLOSED, FINGER_OPEN = 0.0, 0.7854  # the finger range; 0 rad is mechanically closed


def joint_names(side):
    return [f"openarm_{side}_joint{i}" for i in range(1, 8)]


class PlanChecker:
    """FK against the URDF's own <collision> geometry."""

    def __init__(self):
        import os
        import struct
        from xml.etree import ElementTree as ET
        import numpy as np
        from ament_index_python.packages import get_package_share_directory
        self.np = np

        def rpy(r, p, y):
            cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p),
                                      np.sin(p), np.cos(y), np.sin(y))
            return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
                    @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
                    @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))
        self._rpy = rpy

        with open(os.path.join(get_package_share_directory("oap_description"), "urdf",
                               "openarm_bimanual.urdf")) as f:
            root = ET.fromstring(f.read())

        self.edges = {}
        for j in root.iter("joint"):
            if j.get("type") is None:
                continue
            par, ch = j.find("parent"), j.find("child")
            if par is None or ch is None:
                continue
            o = j.find("origin")
            g = (lambda k, d: (o.get(k, d).split() if o is not None else d.split()))
            xyz = [float(v) for v in g("xyz", "0 0 0")]
            r, p, y = [float(v) for v in g("rpy", "0 0 0")]
            ax = j.find("axis")
            axis = [float(v) for v in (ax.get("xyz").split() if ax is not None else "0 0 1".split())]
            self.edges[ch.get("link")] = (par.get("link"), np.array(xyz), rpy(r, p, y),
                                          np.array(axis, float), j.get("type"), j.get("name"))

        def stl(fn):
            pkg, rel = fn[len("package://"):].split("/", 1)
            with open(os.path.join(get_package_share_directory(pkg), rel), "rb") as f:
                head = f.read(84)
                if head[:5] == b"solid" and b"facet" in head:
                    f.seek(0)
                    v = [[float(x) for x in ln.split()[1:4]]
                         for ln in f.read().decode("ascii", "ignore").splitlines()
                         if ln.strip().startswith("vertex")]
                    return np.array(v)
                n = struct.unpack("<I", head[80:84])[0]
                buf = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50)
                return buf[:, 12:48].copy().view(np.float32).reshape(n, 3, 3) \
                    .reshape(-1, 3).astype(float)

        self.geo = {s: {} for s in SIDES}
        for l in root.iter("link"):
            name = l.get("name") or ""
            side = next((s for s in SIDES if name.startswith(f"openarm_{s}_")), None)
            if side is None:
                continue
            pts = []
            for c in l.findall("collision"):
                gm = c.find("geometry")
                if gm is None:
                    continue
                o = c.find("origin")
                gg = (lambda k, d: (o.get(k, d).split() if o is not None else d.split()))
                off = np.array([float(v) for v in gg("xyz", "0 0 0")])
                R = rpy(*[float(v) for v in gg("rpy", "0 0 0")])
                m, b = gm.find("mesh"), gm.find("box")
                if m is not None:
                    q = stl(m.get("filename"))[::6]
                    sc = m.get("scale")          
                    if sc:                       
                        # Apply the mesh scale declared in the URDF.
                        q = q * np.array([float(v) for v in sc.split()])
                elif b is not None:
                    sx, sy, sz = np.array([float(v) for v in b.get("size").split()]) / 2.0
                    q = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy)
                                  for z in (-sz, sz)])
                else:
                    continue
                pts.append(q @ R.T + off)
            if pts:
                self.geo[side][name] = np.vstack(pts)

        # The gripper: the planner's ellipsoid (trajopt.yaml) as surface points on <side>_hand_root.
        with open(os.path.join(get_package_share_directory("oap_planner"), "config", "trajopt.yaml")) as f:
            trajopt = yaml.safe_load(f)
        u, v = np.meshgrid(np.linspace(0.0, np.pi, 13), np.linspace(0.0, 2.0 * np.pi, 25))
        sphere = np.stack([np.sin(u) * np.cos(v), np.sin(u) * np.sin(v), np.cos(u)], -1).reshape(-1, 3)
        hand = sphere * np.array(trajopt["hand_semi_axes"]) + np.array(trajopt["hand_center"])
        for side in SIDES:
            self.geo[side][trajopt["hand_frame"].format(side=side)] = hand

    def _fk(self, leaf, angles):
        np = self.np
        T, node, hops = np.eye(4), leaf, 0
        while node != "world":
            e = self.edges.get(node)
            if e is None or hops > 128:
                return None
            parent, xyz, R0, axis, jtype, jname = e
            if jtype == "fixed":
                R = R0
            else:
                a = axis / np.linalg.norm(axis)
                K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
                q = angles.get(jname, 0.0)
                R = R0 @ (np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K))
            M = np.eye(4)
            M[:3, :3], M[:3, 3] = R, xyz
            T, node, hops = M @ T, parent, hops + 1
        return T

    def _box_sd(self, W, bx, by, bz):
        np = self.np
        d = np.stack([np.maximum(bx[0] - W[:, 0], W[:, 0] - bx[1]),
                      np.maximum(by[0] - W[:, 1], W[:, 1] - by[1]),
                      np.maximum(bz[0] - W[:, 2], W[:, 2] - bz[1])], 1)
        return np.linalg.norm(np.maximum(d, 0.0), axis=1) + np.minimum(d.max(axis=1), 0.0)

    def clearance(self, side, q):
        """Return signed table and pillar clearances in meters. Negative values intersect geometry."""
        np = self.np
        d = dict(zip(joint_names(side), q))
        tbl, pil = 1e9, 0.15
        for link, P in self.geo[side].items():
            T = self._fk(link, d)
            if T is None:
                continue
            W = P @ T[:3, :3].T + T[:3, 3]
            tbl = min(tbl, self._box_sd(W, TABLE_X, TABLE_Y, (-1.0, TABLE_TOP)).min())
            if link.endswith(PILLAR_LINKS):
                for z0, z1, x0, x1, y0, y1 in PILLAR_SLABS:
                    pil = min(pil, self._box_sd(W, (x0, x1), (y0, y1), (z0, z1)).min())
        return tbl, pil

    def check(self, side, start, waypoints):
        """Worst (table, pillar) clearance over the whole commanded motion."""
        np = self.np
        tbl, pil = 1e9, 1e9
        pts = [np.asarray(start, float)] + [np.asarray(w, float) for w in waypoints]
        for a, b in zip(pts[:-1], pts[1:]):
            steps = [0.0] if np.allclose(a, b) else np.linspace(0, 1, CHECK_STEPS)
            for u in steps:
                t, p = self.clearance(side, a + u * (b - a))
                tbl, pil = min(tbl, t), min(pil, p)
        return tbl, pil

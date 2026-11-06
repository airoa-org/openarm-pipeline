#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""A bag's wrist cameras side by side (left | right), in real time, with the target path in force at each frame (cyan)
and the hand's measured path over the same time (orange), both as fingertip points. Writes <bag>/overlay.mp4.

    ros2 run oap_gui overlay output/<bag>
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

import cv2
import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions
from rosidl_runtime_py.utilities import get_message

SIDES = ("left", "right")
FPS = 30.0
W, H, BAR = 640, 480, 24
TIP = np.array([0.088, 0.0, 0.0])  # [m] fingertip in the hand_root frame
K = (240.0, 240.0, 320.0, 240.0)  # fx, fy, cx, cy at 640x480: the wrist camera's nominal pose and field of view
CAM_R = np.array([[0.0, -0.173648, 0.984808], [-1.0, 0.0, 0.0], [0.0, -0.984808, -0.173648]])  # in hand_root
CAM_T = np.array([-0.001, 0.0, 0.0585])
TARGET, MEASURED = (255, 229, 0), (0, 61, 255)  # BGR: cyan, orange
EXECUTED_STEP_S = 0.033


def stamp(h) -> float:
    return h.stamp.sec + 1e-9 * h.stamp.nanosec


def rotation(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def tip(p, q) -> np.ndarray:
    return np.asarray(p, float) + rotation(q) @ TIP


def reader(bag, topics):
    r = SequentialReader()
    r.open(StorageOptions(uri=bag, storage_id=""), ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    r.set_filter(StorageFilter(topics=[t for t in topics if t in types]))
    return r, types


def load_paths(bag):
    """Per arm: the measured hand (times, rotations, positions, tips) and each chunk (its stamp, knot times, tips)."""
    hand = {s: ([], [], [], []) for s in SIDES}
    plans = {s: [] for s in SIDES}
    topics = {f"/arm/{s}/hand_pose": (s, "hand") for s in SIDES}
    topics.update({f"/arm/{s}/cmd_pose_trajectory": (s, "plan") for s in SIDES})
    r, types = reader(bag, topics)
    while r.has_next():
        topic, raw, _ = r.read_next()
        s, kind = topics[topic]
        m = deserialize_message(raw, get_message(types[topic]))
        if kind == "hand":
            p, o = m.pose.position, m.pose.orientation
            q = (o.x, o.y, o.z, o.w)
            xyz = (p.x, p.y, p.z)
            for lst, v in zip(hand[s], (stamp(m.header), rotation(q), np.array(xyz), tip(xyz, q))):
                lst.append(v)
        elif m.points:
            t0 = stamp(m.header)
            tf = [pt.transforms[0] for pt in m.points]
            plans[s].append((t0, np.array([t0 + pt.time_from_start.sec + 1e-9 * pt.time_from_start.nanosec
                                           for pt in m.points]),
                             np.array([tip((t.translation.x, t.translation.y, t.translation.z),
                                           (t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w)) for t in tf])))
    return {s: tuple(np.array(v) for v in hand[s]) for s in SIDES}, plans


def paths_at(t, hand, plans):
    """The target path in force at t (from where it puts the tip now) and the measured tip path over the same time,
    both in the hand frame at t."""
    ht, hr, hp, htip = hand
    if not len(ht) or t < ht[0] or t > ht[-1]:
        return None, None
    i = min(int(np.searchsorted(ht, t)), len(ht) - 1)
    to_hand = lambda pts: (np.asarray(pts) - hp[i]) @ hr[i]  # R^T (p - t), row-wise
    k = int(np.searchsorted([p[0] for p in plans], t, "right")) - 1
    if k < 0 or plans[k][1][-1] <= t:
        return np.zeros((0, 3)), np.zeros((0, 3))
    kt, pts = plans[k][1], plans[k][2]
    now = np.array([np.interp(t, kt, pts[:, j]) for j in range(3)])
    target = np.vstack([now, pts[kt > t]])
    executed = htip[np.searchsorted(ht, np.arange(t, min(kt[-1], ht[-1]), EXECUTED_STEP_S))]
    return to_hand(target), to_hand(executed)


def project(points):
    if not len(points):
        return np.zeros((0, 2)), np.zeros(0)
    c = (np.asarray(points) - CAM_T) @ CAM_R  # into the camera frame
    c = c[c[:, 2] > 1e-3]
    fx, fy, cx, cy = K
    return np.stack([fx * c[:, 0] / c[:, 2] + cx, fy * c[:, 1] / c[:, 2] + cy], 1), c[:, 2]


def draw(jpg, target, executed):
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR) if jpg is not None else None
    img = np.full((H, W, 3), 60, np.uint8) if img is None else cv2.resize(img, (W, H))
    if target is None:
        return img
    for pts, color in ((executed, MEASURED), (target, TARGET)):
        uv, z = project(pts)
        if len(uv) > 1:
            cv2.polylines(img, [np.round(uv).astype(np.int32)], False, color, 2, cv2.LINE_AA)
        for (u, v), zz in zip(uv, z):
            rad = int(np.clip(0.3 / zz, 2, 12))  # bigger = closer
            cv2.circle(img, (int(round(u)), int(round(v))), rad, color, -1, cv2.LINE_AA)
            cv2.circle(img, (int(round(u)), int(round(v))), rad, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def render(bag, out):
    hand, plans = load_paths(bag)
    cams = {f"/{s}/{s}_arm_cam/image_raw/compressed": s for s in SIDES}
    r, types = reader(bag, cams)
    missing = [t for t in cams if t not in types]
    if missing:
        raise SystemExit(f"the bag has no {', '.join(missing)}")
    tmp = out + ".part.mp4"
    video = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (2 * W, H + BAR))
    recent = {s: collections.deque(maxlen=60) for s in SIDES}  # (stamp, jpg), newest last
    t_next, t0, n = None, None, 0
    # Bag order is receive order, near stamp order: a frame at t is drawn once both cameras have a frame after t.
    while r.has_next():
        topic, raw, _ = r.read_next()
        m = deserialize_message(raw, get_message(types[topic]))
        recent[cams[topic]].append((stamp(m.header), bytes(m.data)))
        if not all(recent[s] for s in SIDES):
            continue
        if t_next is None:
            t0 = t_next = max(recent[s][0][0] for s in SIDES)
        while all(recent[s][-1][0] > t_next for s in SIDES):
            canvas = np.zeros((H + BAR, 2 * W, 3), np.uint8)
            for col, s in enumerate(SIDES):
                jpg = next((j for ts, j in reversed(recent[s]) if ts <= t_next), None)
                canvas[BAR:, col * W:(col + 1) * W] = draw(jpg, *paths_at(t_next, hand[s], plans[s]))
            cv2.putText(canvas, f"left | right   t = {t_next - t0:6.2f} s   cyan: target   orange: measured", (6, 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            video.write(canvas)
            n += 1
            t_next += 1.0 / FPS
    video.release()
    if not n:
        os.remove(tmp)
        raise SystemExit("no frame where both cameras have images")
    os.replace(tmp, out)  # Play finds only a finished video
    print(f"{out}: {n} frames, {n / FPS:.1f} s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    args = ap.parse_args()
    bag = os.path.abspath(args.bag)
    print(f"[overlay] {bag}: reading the paths, then the cameras", flush=True)
    render(bag, os.path.join(bag, "overlay.mp4"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Synthetic ground-truth pose for tests and `--demo` (a throwing lunge, like the prototype demo)."""
from __future__ import annotations
import numpy as np
from .skeleton import JOINTS, IDX


def _u(*v):
    v = np.array(v, float)
    return v / np.linalg.norm(v)


def lunge_pose(height=1.70, yaw_deg=35.0, pelvis_cam=(0.1, 0.05, 4.0)):
    """Return (J,3) camera-frame joints (x right, y down, z forward), metres."""
    H = height
    P = {}
    # local frame: R = person's right, U = up, F = forward ; rows of vectors (R, U, F)
    P["pelvis"] = np.zeros(3)
    P["lhip"], P["rhip"] = np.array([-0.065 * H, 0, 0]), np.array([0.065 * H, 0, 0])
    P["neck"] = 0.28 * H * _u(0.15, 1, 0.18)
    P["lsho"] = P["neck"] + np.array([-0.1075 * H, 0, 0])
    P["rsho"] = P["neck"] + np.array([0.1075 * H, 0, 0])
    P["nose"] = P["neck"] + 0.11 * H * _u(0.05, 0.75, 0.66)
    P["head"] = P["neck"] + 0.17 * H * _u(0.05, 1, 0.25)
    P["leye"] = P["nose"] + np.array([-0.018, 0.035, -0.01]) * H / 1.7 * 1.0
    P["reye"] = P["nose"] + np.array([0.018, 0.035, -0.01]) * H / 1.7 * 1.0
    P["lear"] = P["leye"] + 0.045 * H * _u(-0.6, -0.05, -0.8)
    P["rear"] = P["reye"] + 0.045 * H * _u(0.6, -0.05, -0.8)
    # left arm reaches forward/up, right arm cocked back (throw)
    P["lelb"] = P["lsho"] + 0.186 * H * _u(-0.6, 0.5, 0.6)
    P["lwri"] = P["lelb"] + 0.146 * H * _u(-0.2, 1, 0.3)
    P["lhand"] = P["lwri"] + 0.05 * H * _u(-0.2, 1, 0.3)
    P["relb"] = P["rsho"] + 0.186 * H * _u(0.7, -0.3, -0.5)
    P["rwri"] = P["relb"] + 0.146 * H * _u(0.5, 0.5, -0.5)
    P["rhand"] = P["rwri"] + 0.05 * H * _u(0.5, 0.5, -0.5)
    # legs: left forward lunge, right back on the ball of the foot
    P["lknee"] = P["lhip"] + 0.245 * H * _u(-0.05, -0.7, 0.7)
    P["lank"] = P["lknee"] + 0.246 * H * _u(0, -1, 0)
    P["rknee"] = P["rhip"] + 0.245 * H * _u(0.05, -1, -0.4)
    P["rank"] = P["rknee"] + 0.246 * H * _u(0, -0.6, -0.8)
    for s, med, toe_dir in (("l", 1, _u(0, -0.3, 1)), ("r", -1, _u(0, -0.85, 0.5))):
        a = P[s + "ank"]
        P[s + "heel"] = a + 0.045 * H * _u(0, -0.5, -0.85)
        P[s + "bigtoe"] = a + 0.13 * H * (toe_dir + np.array([0.12 * med, 0, 0]))
        P[s + "smalltoe"] = a + 0.115 * H * (toe_dir + np.array([-0.3 * med, 0, 0]))
    L = np.stack([P[n] for n in JOINTS])             # local (R, U, F)
    y = np.radians(yaw_deg)
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rc, Uc, Fc = Ry @ [-1, 0, 0], np.array([0, -1, 0.0]), Ry @ [0, 0, -1]
    X = L[:, :1] * Rc + L[:, 1:2] * Uc + L[:, 2:3] * Fc
    return X + np.array(pelvis_cam)


def procrustes(A, B):
    """Similarity-align A onto B; return aligned A."""
    mA, mB = A.mean(0), B.mean(0)
    A0, B0 = A - mA, B - mB
    U, S, Vt = np.linalg.svd(A0.T @ B0)
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (A0 ** 2).sum()
    return s * A0 @ R + mB


def _rot(axis, deg):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a); t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def stand_pose(height=1.70, lean_deg=0.0, side_bend_deg=0.0, pelvis_cam=(0.0, 0.1, 4.0)):
    """Upright stance, feet apart, arms hanging. lean = forward hip flexion of the trunk;
    side_bend = lateral bend of the trunk to the person's LEFT (left side compressed)."""
    H = height
    P = {}
    P["pelvis"] = np.zeros(3)
    P["lhip"], P["rhip"] = np.array([-0.065 * H, 0, 0]), np.array([0.065 * H, 0, 0])
    # upper body built upright, then rotated: lean about the right axis, bend about forward
    up = {}
    up["neck"] = np.array([0, 0.28 * H, 0])
    up["lsho"], up["rsho"] = up["neck"] + [-0.1075 * H, 0, 0], up["neck"] + [0.1075 * H, 0, 0]
    up["nose"] = up["neck"] + 0.11 * H * _u(0, 0.75, 0.66)
    up["head"] = up["neck"] + [0, 0.17 * H, 0.02 * H]
    up["leye"], up["reye"] = up["nose"] + [-0.018, 0.035, -0.01], up["nose"] + [0.018, 0.035, -0.01]
    up["lear"], up["rear"] = up["leye"] + [-0.03, 0, -0.06], up["reye"] + [0.03, 0, -0.06]
    Rb = _rot([0, 0, 1], side_bend_deg)          # about +F by +θ: U -> -R, i.e. the neck tips to the left
    Rl = _rot([1, 0, 0], lean_deg)               # about +R by +θ: U -> +F, the trunk leans forward
    if side_bend_deg:                            # lateral bend is distributed: the shoulder line tilts by θ
        for k in up:
            up[k] = Rb @ up[k]
    for k in up:
        P[k] = Rl @ up[k]
    for s, sg in (("l", -1), ("r", 1)):          # arms hang vertically from the shoulders
        P[s + "elb"] = P[s + "sho"] + [0, -0.186 * H, 0]
        P[s + "wri"] = P[s + "elb"] + [0, -0.146 * H, 0]
        P[s + "hand"] = P[s + "wri"] + [0, -0.05 * H, 0]
        hip = P[s + "hip"]
        P[s + "knee"] = hip + [sg * 0.02 * H, -0.245 * H, 0]
        P[s + "ank"] = P[s + "knee"] + [0, -0.246 * H, 0]
        a = P[s + "ank"]
        P[s + "heel"] = a + 0.045 * H * _u(0, -0.6, -0.8)
        P[s + "bigtoe"] = a + 0.13 * H * _u(sg * 0.15, -0.35, 1)       # relaxed ~10° toe-out
        P[s + "smalltoe"] = a + 0.115 * H * _u(sg * 0.5, -0.35, 1)
    L = np.stack([P[n] for n in JOINTS])
    Rc, Uc, Fc = np.array([-1, 0, 0.0]), np.array([0, -1, 0.0]), np.array([0, 0, -1.0])
    return L[:, :1] * Rc + L[:, 1:2] * Uc + L[:, 2:3] * Fc + np.array(pelvis_cam)

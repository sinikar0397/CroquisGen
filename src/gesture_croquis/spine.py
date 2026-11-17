"""
Spine curve from the 3D torso box (keypoint-only; no vertebra detections exist in COCO/Halpe/RTMW).

Anatomical assumption: the spine leaves the pelvis perpendicular to the hip line and enters
the thorax perpendicular to the shoulder line (both within the plane that contains the
neck–pelvis chord). A cubic Bézier with those end tangents gives:
  * lateral bend  -> C-curve, concave on the compressed side (shoulder/hip lines tilted)
  * twist         -> no extra bend (rotation about the chord), recorded as an angle
A quadratic cervical segment continues C1-smoothly from the neck to the head (crown).

Observable from keypoints:  lateral bend, twist, overall lean.
NOT observable:             sagittal (front/back) curvature — needs a body model (SKEL/MHR) or
                            a silhouette; `sagittal` lets you add an assumed curvature.
"""
from __future__ import annotations
import numpy as np

from .skeleton import IDX

N_TRUNK, N_NECK = 17, 6          # samples: pelvis -> neck (≈ L5..T1), neck -> head (cervical + skull)
LANDMARKS = {"L5": 0.0, "L3": 0.12, "L1": 0.25, "T12": 0.30, "T8": 0.55, "T4": 0.80, "C7": 1.0}


def _u(v):
    return v / (np.linalg.norm(v) + 1e-9)


def _perp_tangent(d, lateral):
    """Direction of d with its component along the lateral axis removed."""
    l = _u(lateral)
    t = d - (d @ l) * l
    return _u(t) if np.linalg.norm(t) > 1e-6 else _u(d)


def spine_curve(X: np.ndarray, sagittal: float = 0.0, k: float = 1 / 3):
    """X: (J,3) joints (any right-handed frame). Returns dict:
    trunk (N_TRUNK,3) pelvis->neck, neck (N_NECK,3) neck->head, s (N_TRUNK,) arc fraction,
    landmarks {name: (3,)}, lateral_bend_deg, twist_deg, chord_len."""
    i = IDX
    P0, P3, H = X[i["pelvis"]], X[i["neck"]], X[i["head"]]
    d = P3 - P0
    L = np.linalg.norm(d)
    rp, rt = X[i["rhip"]] - X[i["lhip"]], X[i["rsho"]] - X[i["lsho"]]
    t0, t1 = _perp_tangent(d, rp), _perp_tangent(d, rt)
    B1, B2 = P0 + t0 * L * k, P3 - t1 * L * k
    if sagittal:
        # assumed front/back curvature: push control points along the chest-forward axis
        fwd = _u(np.cross(_u(d), _u(rp + rt)))
        B1, B2 = B1 + fwd * sagittal * L, B2 + fwd * sagittal * L
    t = np.linspace(0, 1, 400)[:, None]
    dense = (1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * B1 + 3 * (1 - t) * t ** 2 * B2 + t ** 3 * P3
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    arc /= arc[-1]

    def at(s):           # arc-length parameterised sample
        return np.stack([np.interp(s, arc, dense[:, c]) for c in range(3)], -1)

    s = np.linspace(0, 1, N_TRUNK)
    trunk = at(s)
    # cervical: quadratic from the neck, leaving along the trunk's end tangent
    hn = H - P3
    C = P3 + _u(P3 - B2) * np.linalg.norm(hn) * 0.45
    u = np.linspace(0, 1, N_NECK + 1)[1:, None]
    neck = (1 - u) ** 2 * P3 + 2 * (1 - u) * u * C + u ** 2 * H

    # angles between hip and shoulder lines, measured about the trunk's own axes
    up = _u(d)
    rp_, rt_ = _u(rp - (rp @ up) * up), _u(rt - (rt @ up) * up)
    twist = np.degrees(np.arctan2(np.cross(rp_, rt_) @ up, rp_ @ rt_))
    fwd = _u(np.cross(up, _u(rp + rt)))
    rpf, rtf = _u(rp - (rp @ fwd) * fwd), _u(rt - (rt @ fwd) * fwd)       # frontal-plane
    lat = np.degrees(np.arctan2(np.cross(rpf, rtf) @ fwd, rpf @ rtf))
    return dict(trunk=trunk, neck=neck, s=s, chord_len=float(L),
                landmarks={n: at(np.array([v]))[0] for n, v in LANDMARKS.items()},
                lateral_bend_deg=float(lat), twist_deg=float(twist),
                max_offset=float(np.max(np.linalg.norm(np.cross(trunk - P0, up), axis=1))))


def polyline(sp) -> np.ndarray:
    """Full pelvis -> head polyline."""
    return np.vstack([sp["trunk"], sp["neck"]])

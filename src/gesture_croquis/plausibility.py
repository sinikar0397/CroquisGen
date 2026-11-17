"""
Pose plausibility: score a 3D skeleton for anatomical / physical sense, and search over the
per-bone depth-direction ambiguity that 2D keypoints cannot resolve.

Why: from one image every bone has two depth solutions (toward / away from the camera) with the
same 2D projection. A depth network breaks the tie only when it is reliable; on distant or
telephoto shots it is not, and single limbs or the torso yaw come out mirrored — producing
knees that bend sideways, arms twisted past their range, limbs passing through the body.

Score (energy, lower = more plausible), each term dimensionless:
  image     reprojection error + bone lengths vs the reference fit (2D consistency)
  depth     agreement with the depth network along bones (Cauchy-robust, low weight)
  limits    joint-angle ranges (elbow/knee flexion, hip/shoulder extension, trunk twist/bend, head)
  twist     axial rotation of upper arm and thigh, read from the plane the elbow / knee bends in
            (swing-twist decomposition). A knee bending sideways = thigh rotated ~90° = impossible.
  collision limb / torso / head capsules interpenetrating
  balance   COM off the ankle support line (skipped for dynamic poses)
  effort    total static joint effort from S3 (people adopt low-effort poses), final ranking only

Search: flip the depth direction of bone groups (torso yaw/lean, each upper/lower limb segment,
feet, head) keeping the 2D projection fixed; limbs are searched per torso configuration
(coordinate descent), the best hypotheses are re-fitted and re-scored.
"""
from __future__ import annotations
import itertools
import numpy as np

from .skeleton import IDX, JOINTS, J
from .s2_lift import (project, backproject, body_frame, _sp, _TREE, pose_prior_terms, balance_term,
                      _LOW_TRUST, _cross)

# ----------------------------------------------------------------------------------------
# Axial rotation (swing-twist) of upper arm / thigh
# ----------------------------------------------------------------------------------------
# side: (proximal, middle, distal, neutral distal-segment direction in the body frame, sign)
#   arm:  elbow flexed 90° with the humerus hanging -> forearm points FORWARD; + = external rot.
#   leg:  knee flexed 90° with the thigh hanging    -> shank points BACKWARD
_TWIST_LIMBS = {
    "l_arm": ("lsho", "lelb", "lwri", "fwd", -1), "r_arm": ("rsho", "relb", "rwri", "fwd", +1),
    "l_leg": ("lhip", "lknee", "lank", "back", -1), "r_leg": ("rhip", "rknee", "rank", "back", +1),
}
# soft walls (deg): arm external +150 (throwing layback) / internal -110 ; hip rotation ±50
_TWIST_RANGE = {"arm": (-110.0, 150.0), "leg": (-50.0, 50.0)}   # throwing layback reaches ~150° ER


def limb_twists(X):
    """{limb: (angle_deg, weight)}; weight ~0 when the middle joint is straight (twist undefined)
    or the proximal segment points straight up (swing singularity)."""
    frames = {"arm": body_frame(X, "thorax"), "leg": body_frame(X, "pelvis")}
    out = {}
    for name, (p, m, d, ref, sg) in _TWIST_LIMBS.items():
        up, r, fwd = frames[name[2:]]
        P, M, D = X[IDX[p]], X[IDX[m]], X[IDX[d]]
        u = M - P; u = u / (np.sqrt(u @ u) + 1e-9)
        a = -up                                          # rest direction of the segment
        c = float(np.clip(a @ u, -1, 1))
        v = _cross(a, u)
        refv = fwd if ref == "fwd" else -fwd
        if c > -0.999:                                   # Rodrigues: minimal rotation a -> u
            f_ref = refv * c + _cross(v, refv) + v * (v @ refv) / (1 + c)
        else:
            f_ref = -refv
        low = D - M
        f = low - (low @ u) * u
        fr = f_ref - (f_ref @ u) * u
        ang = np.degrees(np.arctan2(_cross(fr, f) @ u, fr @ f))
        flex_w = np.sqrt(f @ f) / (np.sqrt(low @ low) + 1e-9)               # sin(flexion)
        swing = np.degrees(np.arccos(c))
        w = float(np.clip((flex_w - 0.15) / 0.3, 0, 1) * np.clip((165 - swing) / 30, 0, 1))   # off below ~9° flexion
        out[name] = (float(sg * ang), w)
    return out


def twist_terms(X):
    """Soft-wall residuals for axial rotation beyond the anatomical range."""
    res = []
    for name, (ang, w) in limb_twists(X).items():
        lo, hi = _TWIST_RANGE[name[2:]]
        res += [w * 2 * _sp((ang - hi) / 8), w * 2 * _sp((lo - ang) / 8)]
    return np.array(res)


# ----------------------------------------------------------------------------------------
# Self-collision (capsules; radii in metres for a 1.70 m body, scaled with height)
# ----------------------------------------------------------------------------------------
_CAPS = {
    "torso": ("pelvis", "neck", 0.10), "head": ("neck", "head", 0.085),
    "l_uarm": ("lsho", "lelb", 0.04), "r_uarm": ("rsho", "relb", 0.04),
    "l_farm": ("lelb", "lwri", 0.035), "r_farm": ("relb", "rwri", 0.035),
    "l_thigh": ("lhip", "lknee", 0.05), "r_thigh": ("rhip", "rknee", 0.05),
    "l_shank": ("lknee", "lank", 0.04), "r_shank": ("rknee", "rank", 0.04),
    "l_foot": ("lheel", "lbigtoe", 0.03), "r_foot": ("rheel", "rbigtoe", 0.03),
}
# (capsule A, capsule B, fraction of A kept from its distal end) — segments attached to each
# other are excluded; the upper arm only tests its distal half against the trunk
_PAIRS = []
for s, o in (("l", "r"), ("r", "l")):
    _PAIRS += [(f"{s}_farm", "torso", 1.0), (f"{s}_uarm", "torso", 0.5), (f"{s}_farm", "head", 1.0),
               (f"{s}_farm", f"{o}_thigh", 1.0),       # same-side hand on thigh = normal contact
               (f"{s}_shank", f"{o}_thigh", 1.0), (f"{s}_foot", f"{o}_shank", 1.0)]
_PAIRS += [("l_thigh", "r_thigh", 0.66), ("l_shank", "r_shank", 1.0), ("l_foot", "r_foot", 1.0),
           ("l_farm", "r_farm", 1.0)]


_PA = np.array([[IDX[_CAPS[A][0]], IDX[_CAPS[A][1]]] for A, B, f in _PAIRS])
_PB = np.array([[IDX[_CAPS[B][0]], IDX[_CAPS[B][1]]] for A, B, f in _PAIRS])
_PR = np.array([_CAPS[A][2] + _CAPS[B][2] for A, B, f in _PAIRS])
_PF = np.array([f for A, B, f in _PAIRS])


def seg_dist(p1, q1, p2, q2):
    """Vectorised closest distance between segments p1q1[n] and p2q2[n] (n,3 arrays)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = np.einsum("ij,ij->i", d1, d1) + 1e-12
    e = np.einsum("ij,ij->i", d2, d2) + 1e-12
    b = np.einsum("ij,ij->i", d1, d2)
    c = np.einsum("ij,ij->i", d1, r)
    f = np.einsum("ij,ij->i", d2, r)
    den = a * e - b * b
    s = np.where(den > 1e-12, np.clip((b * f - c * e) / np.maximum(den, 1e-12), 0, 1), 0.0)
    t = (b * s + f) / e
    lo, hi = t < 0, t > 1
    s = np.where(lo, np.clip(-c / a, 0, 1), np.where(hi, np.clip((b - c) / a, 0, 1), s))
    t = np.clip(t, 0, 1)
    return np.linalg.norm((p1 + d1 * s[:, None]) - (p2 + d2 * t[:, None]), axis=1)


def collision_terms(X, height=1.70, valid=None, smooth=True):
    """Penetration depth per capsule pair (metres; ~0 when apart). smooth -> softplus for the fit."""
    qa, pa0 = X[_PA[:, 1]], X[_PA[:, 0]]
    pa = qa + (pa0 - qa) * _PF[:, None]
    pen = _PR * (height / 1.70) - seg_dist(pa, qa, X[_PB[:, 0]], X[_PB[:, 1]])
    if valid is not None:
        ok = valid[_PA].all(1) & valid[_PB].all(1)
        pen = np.where(ok, pen, -1.0)
    return 0.01 * _sp(pen / 0.01) if smooth else np.maximum(pen, 0.0)


# ----------------------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------------------
class Scorer:
    """Holds the observation (2D keypoints, camera, depth cue, reference bone lengths)."""

    W = dict(image=1.0, depth=0.5, limits=1.0, twist=1.0, collision=1.0, balance=1.0, effort=3.0, occlusion=1.0)

    def __init__(self, kpts2d, conf, K, X_ref, valid, height, z_rel=None, z_conf=None, dynamic=False,
                 order_cons=None):
        self.order_cons = order_cons or []
        self.k2, self.conf, self.K, self.valid, self.height = kpts2d, conf, K, valid, height
        self.vi = np.where(valid)[0]
        self.sig_px = max(2.0, 0.012 * np.ptp(kpts2d[self.vi], axis=0).max())
        self.E = np.array([(IDX[a], IDX[b]) for a, b in _TREE if valid[IDX[a]] and valid[IDX[b]]])
        self.Lref = np.linalg.norm(X_ref[self.E[:, 0]] - X_ref[self.E[:, 1]], axis=1)
        self.dynamic = dynamic
        if dynamic:   # athletic / moving poses: don't pull toward low-effort, balanced stances
            self.W = dict(self.W, balance=0.0, effort=0.5)
        self.z_rel = z_rel
        if z_rel is not None:
            zc = np.clip(z_conf if z_conf is not None else conf, 0, 1)
            self.ew = np.array([np.sqrt(min(zc[a], zc[b])) * (0.35 if JOINTS[b] in _LOW_TRUST else 1.0)
                                for a, b in self.E])
            self.ezr = z_rel[self.E[:, 1]] - z_rel[self.E[:, 0]]

    def terms(self, X, depth_scale=1.0, with_effort=False):
        vi, E = self.vi, self.E
        t = {}
        rp = (project(X[vi], self.K) - self.k2[vi]) / self.sig_px
        L = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
        # proportions, not absolute size: at long range size and distance trade off freely
        k = (L @ self.Lref) / (L @ L + 1e-12)
        t["image"] = float((self.conf[vi][:, None] * rp ** 2).sum()
                           + (((k * L - self.Lref) / (0.05 * self.Lref)) ** 2).sum())
        if self.z_rel is not None:
            r = ((X[E[:, 1], 2] - X[E[:, 0], 2]) - depth_scale * self.ezr) / 0.07
            t["depth"] = float((self.ew ** 2 * np.log1p(r * r)).sum())
        else:
            t["depth"] = 0.0
        t["limits"] = float((pose_prior_terms(X) ** 2).sum())
        t["twist"] = float((twist_terms(X) ** 2).sum())
        t["collision"] = float(((collision_terms(X, self.height, self.valid, smooth=False) / 0.03) ** 2).sum())
        t["balance"] = 0.0 if self.dynamic else float((2 * balance_term(X)) ** 2)
        t["effort"] = self.effort(X) if with_effort else 0.0
        if self.order_cons:
            from .occlusion import order_terms
            t["occlusion"] = float((order_terms(X, self.order_cons) ** 2).sum())
        else:
            t["occlusion"] = 0.0
        t["total"] = float(sum(self.W[k] * v for k, v in t.items()))
        return t

    def effort(self, X):
        """Sum of squared static joint effort (fraction of capacity) from the S3 T0 model."""
        from .poseframe import PoseFrame
        from .s3_load import static_load
        f = PoseFrame(img_hw=(1, 1), kpts2d=self.k2, conf=self.conf)
        f.joints3d, f.valid3d, f.body_height_m = X, self.valid, self.height
        try:
            static_load(f)
        except Exception:
            return 0.0
        e = np.array(list(f.s3_info["effort_fraction_of_capacity"].values()))
        return float((e ** 2).sum())


# ----------------------------------------------------------------------------------------
# Depth-flip hypotheses
# ----------------------------------------------------------------------------------------
GROUPS = {   # each group: list of "bits"; a bit = set of tree edges flipped together
    "torso": [[("pelvis", "neck")], [("neck", "lsho"), ("neck", "rsho")], [("pelvis", "lhip"), ("pelvis", "rhip")]],
    "l_leg": [[("lhip", "lknee")], [("lknee", "lank")]],
    "r_leg": [[("rhip", "rknee")], [("rknee", "rank")]],
    "l_arm": [[("lsho", "lelb")], [("lelb", "lwri")]],
    "r_arm": [[("rsho", "relb")], [("relb", "rwri")]],
    "l_foot": [[("lank", "lheel"), ("lank", "lbigtoe"), ("lank", "lsmalltoe")]],
    "r_foot": [[("rank", "rheel"), ("rank", "rbigtoe"), ("rank", "rsmalltoe")]],
    "head": [[("neck", "nose"), ("neck", "head")]],
}
LIMB_GROUPS = ["l_leg", "r_leg", "l_arm", "r_arm", "l_foot", "r_foot", "head"]


def flip_edges(X, K, edges):
    """Exact perspective flip: walk the kinematic tree from the pelvis; each child joint stays on
    its camera ray at its original bone length from its (possibly moved) parent. That gives a
    quadratic in depth with two roots — the two depth solutions of the bone. Flipped edges take
    the other root; all others keep the same side. 2D projection and bone lengths are preserved."""
    if not edges:
        return X
    flip = set(edges)
    uv = project(X, K)
    rays = np.c_[(uv - K[:2, 2]) / K[[0, 1], [0, 1]], np.ones(len(X))]      # X = Z * ray

    def roots(r, P, L):
        A, B, C = r @ r, -2 * (r @ P), P @ P - L * L
        disc = max(B * B - 4 * A * C, 0.0)                  # can't reach: both = closest point
        return (-B - np.sqrt(disc)) / (2 * A), (-B + np.sqrt(disc)) / (2 * A)

    Xn = X.copy()
    for a, b in _TREE:
        ia, ib = IDX[a], IDX[b]
        L = np.linalg.norm(X[ib] - X[ia])
        r = rays[ib]
        near, far = roots(r, X[ia], L)                       # which root is the original on?
        on_far = abs(X[ib, 2] - far) < abs(X[ib, 2] - near)
        if (a, b) in flip:
            on_far = not on_far
        near, far = roots(r, Xn[ia], L)                      # same choice from the moved parent
        Xn[ib] = max(far if on_far else near, 0.3) * r
    return Xn


def _combos(group):
    bits = GROUPS[group]
    for mask in itertools.product((0, 1), repeat=len(bits)):
        yield mask, [e for m, b in zip(mask, bits) if m for e in b]


def search_flips(X, K, scorer, sweeps=2, depth_scale=1.0):
    """For each torso configuration, coordinate-descend over limb flips.
    Returns a list of (score, X, flips_description) sorted best first (one per torso config)."""
    results = []
    for tmask, tedges in _combos("torso"):
        Xc = flip_edges(X, K, tedges)
        cur = scorer.terms(Xc, depth_scale)["total"]
        chosen = {"torso": tmask}
        for _ in range(sweeps):
            improved = False
            for g in LIMB_GROUPS:
                best = (cur, None, None)
                for mask, edges in _combos(g):
                    if not any(mask):
                        continue
                    Xn = flip_edges(Xc, K, edges)
                    s = scorer.terms(Xn, depth_scale)["total"]
                    if s < best[0] - 1e-6:
                        best = (s, Xn, mask)
                if best[1] is not None:
                    cur, Xc = best[0], best[1]
                    prev = chosen.get(g, (0,) * len(GROUPS[g]))
                    chosen[g] = tuple(int(p ^ m) for p, m in zip(prev, best[2]))
                    improved = True
            if not improved:
                break
        desc = {g: m for g, m in chosen.items() if any(m)}
        results.append((cur, Xc, desc))
    results.sort(key=lambda r: r[0])
    return results

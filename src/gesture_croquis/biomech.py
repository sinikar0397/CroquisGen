"""
Biomechanical energy used INSIDE the 3D fit (S2+S3 integrated):

  passive    each joint's deviation from its RELAXED angle, measured in the anatomically
             correct local frame (legs vs pelvis, arms vs thorax). Muscle-tendon units are
             slack near the relaxed posture and resist increasingly with stretch, so the
             cost is (deviation / stiffness-sigma)^2 with asymmetric sigmas (e.g. hip adduction
             across the midline is far stiffer than abduction).
  effort     static gravity torques: segment masses (de Leva) + ground reactions that balance
             the body, torque at each joint / its isometric capacity.
  balance    centre of mass outside the support polygon.
  floor      planted contact points lie on ONE horizontal plane (gravity-perpendicular). Under
             perspective this is a genuine depth cue: a foot further back projects higher.
  pitch      camera pitch is a free parameter (prior 0 ± 8°), so "horizontal" is right.

Everything is vectorised numpy so it can run inside least_squares' finite-difference Jacobian.
Frames: camera (x right, y down, z forward); gravity = (0, cos φ, sin φ) for camera pitch φ.
"""
from __future__ import annotations
import numpy as np

from .skeleton import IDX, JOINTS, J
from .s3_load import LIMB_SEGMENTS, TRUNK_PARTS, CAPACITY

G = 9.81
D = np.degrees


def _u(v):
    return v / (np.sqrt(v @ v) + 1e-9)


def _cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def gravity_dir(pitch):
    """World-down expressed in camera coordinates for a camera pitched down by `pitch` (rad)."""
    return np.array([0.0, np.cos(pitch), np.sin(pitch)])


def frames(X, down):
    """Pelvis and thorax frames (up, right, fwd) with 'up' = trunk axis."""
    i = IDX
    up = _u(X[i["neck"]] - X[i["pelvis"]])
    out = {}
    for name, (a, b) in (("pelvis", ("lhip", "rhip")), ("thorax", ("lsho", "rsho"))):
        r = X[i[b]] - X[i[a]]
        r = _u(r - (r @ up) * up)
        out[name] = (up, r, _u(_cross(up, r)))
    return out


# ----------------------------------------------------------------------------------------
# Relaxed posture + passive stiffness
# ----------------------------------------------------------------------------------------
# name: (neutral deg, sigma below neutral, sigma above neutral)
RELAXED = {
    "hip_flex": (5, 15, 40),          # extension is resisted early (hip flexors / capsule)
    "hip_abd": (6, 9, 22),            # adduction across the midline is stiff
    "hip_rot": (10, 25, 25),          # slight external rotation (thigh twist, when knee bent)
    "knee_flex": (5, 4, 45),          # hyperextension blocked, flexion cheap-ish
    "ankle_dorsi": (0, 30, 18),       # plantarflexion (on toes) easier than dorsiflexion
    "foot_out": (14, 12, 22),         # feet naturally toe out ~10-15° vs the pelvis
    "sho_flex": (8, 20, 70),
    "sho_abd": (10, 25, 60),
    "sho_rot": (-15, 50, 50),
    "elbow_flex": (15, 20, 60),       # a hanging arm rests slightly bent
    "trunk_twist": (0, 25, 25),
    "trunk_bend": (0, 14, 14),
    "head_yaw": (0, 40, 40),
}


def elevation_split(seg, up, lat, fwd):
    """Elevation (angle from hanging) split by its plane: (flexion, abduction) in degrees.
    flex = elev*sin(plane), abd = elev*cos(plane), plane = azimuth from lateral toward forward.
    Well defined for every direction (unlike separate projected angles, which break at 90°)."""
    elev = D(np.arccos(np.clip(-(seg @ up), -1, 1)))
    plane = np.arctan2(seg @ fwd, seg @ lat)
    return elev * np.sin(plane), elev * np.cos(plane)


def joint_angles(X, down):
    """Anatomical angles (deg) and a weight per angle (0 when undefined)."""
    i = IDX
    F = frames(X, down)
    ang, w = {}, {}
    for s, lat_sign in (("l", -1.0), ("r", 1.0)):
        up, r, fwd = F["pelvis"]
        lat = r * lat_sign                                         # lateral = away from midline
        hip, knee, ank = X[i[s + "hip"]], X[i[s + "knee"]], X[i[s + "ank"]]
        heel, toe = X[i[s + "heel"]], X[i[s + "bigtoe"]]
        th, sh, ft = _u(knee - hip), _u(ank - knee), _u(toe - heel)
        ang[s + "hip_flex"], ang[s + "hip_abd"] = elevation_split(th, up, lat, fwd)
        kf = D(np.arccos(np.clip(th @ sh, -1, 1)))
        # sign the knee angle: flexion = shank swings backward relative to the thigh
        ang[s + "knee_flex"] = kf if (_cross(th, sh) @ (-r)) >= -1e-6 else -kf
        ang[s + "ankle_dorsi"] = 90 - D(np.arccos(np.clip((-sh) @ ft, -1, 1)))
        fh = ft - (ft @ up) * up
        ang[s + "foot_out"] = D(np.arctan2(fh @ lat, fh @ fwd))
        w[s + "foot_out"] = float(np.clip((np.sqrt(fh @ fh) - 0.3) / 0.4, 0, 1))
        # thigh axial rotation from the knee's bending plane (undefined for a straight knee)
        f_ref = -fwd - (-fwd @ th) * th
        f_sh = sh - (sh @ th) * th
        n1, n2 = np.sqrt(f_ref @ f_ref), np.sqrt(f_sh @ f_sh)
        ang[s + "hip_rot"] = lat_sign * D(np.arctan2(_cross(f_ref, f_sh) @ th, f_ref @ f_sh)) if n2 > 1e-6 else 0.0
        w[s + "hip_rot"] = float(np.clip((n2 - 0.15) / 0.3, 0, 1) * (n1 > 0.3))

        up, r, fwd = F["thorax"]
        lat = r * lat_sign
        sho, elb, wri = X[i[s + "sho"]], X[i[s + "elb"]], X[i[s + "wri"]]
        ua, fa = _u(elb - sho), _u(wri - elb)
        ang[s + "sho_flex"], ang[s + "sho_abd"] = elevation_split(ua, up, lat, fwd)
        ang[s + "elbow_flex"] = D(np.arccos(np.clip(ua @ fa, -1, 1)))
        f_ref = fwd - (fwd @ ua) * ua
        f_fa = fa - (fa @ ua) * ua
        n1, n2 = np.sqrt(f_ref @ f_ref), np.sqrt(f_fa @ f_fa)
        ang[s + "sho_rot"] = -lat_sign * D(np.arctan2(_cross(f_ref, f_fa) @ ua, f_ref @ f_fa)) if n2 > 1e-6 else 0.0
        w[s + "sho_rot"] = float(np.clip((n2 - 0.15) / 0.3, 0, 1) * np.clip((n1 - 0.2) / 0.3, 0, 1))

    (upp, rp, fp), (upt, rt, ft_) = F["pelvis"], F["thorax"]
    ang["trunk_twist"] = D(np.arctan2(_cross(rp, rt) @ upp, rp @ rt))
    ang["trunk_bend"] = D(np.arctan2((X[i["neck"]] - X[i["pelvis"]]) @ rp,
                                     (X[i["neck"]] - X[i["pelvis"]]) @ (-down) + 1e-9))
    nose = X[i["nose"]] - X[i["neck"]]
    ang["head_yaw"] = D(np.arctan2(nose @ rt, nose @ ft_))
    return ang, w


def passive_residuals(X, down):
    ang, w = joint_angles(X, down)
    res = []
    for k, a in ang.items():
        base = k[1:] if k[0] in "lr" and k[1:] in RELAXED else k
        n, lo, hi = RELAXED[base]
        d = a - n
        res.append(w.get(k, 1.0) * d / (lo if d < 0 else hi))
    return np.array(res), ang


# ----------------------------------------------------------------------------------------
# Gravity effort (vectorised S3)
# ----------------------------------------------------------------------------------------
_SEG = list(LIMB_SEGMENTS.items())
_SA = np.array([IDX[a] for _, (a, b, m, r) in _SEG]); _SB = np.array([IDX[b] for _, (a, b, m, r) in _SEG])
_SR = np.array([r for _, (a, b, m, r) in _SEG]); _SM = np.array([m for _, (a, b, m, r) in _SEG])
_SEG_NAMES = [n for n, _ in _SEG] + list(TRUNK_PARTS)
_TM = np.array([m for m, s in TRUNK_PARTS.values()]); _TS = np.array([s for m, s in TRUNK_PARTS.values()])

_DISTAL = {}
for _s in "lr":
    _DISTAL[_s + "hip"] = [f"{_s}_thigh", f"{_s}_shank", f"{_s}_foot"]
    _DISTAL[_s + "knee"] = [f"{_s}_shank", f"{_s}_foot"]
    _DISTAL[_s + "ank"] = [f"{_s}_foot"]
    _DISTAL[_s + "sho"] = [f"{_s}_uarm", f"{_s}_farm", f"{_s}_hand"]
    _DISTAL[_s + "elb"] = [f"{_s}_farm", f"{_s}_hand"]
_DISTAL["neck"] = ["head"]
_DISTAL["pelvis"] = ["head", "l_uarm", "l_farm", "l_hand", "r_uarm", "r_farm", "r_hand",
                     "middle_trunk", "upper_trunk"]                     # lumbar (L5) load
EFFORT_JOINTS = list(_DISTAL)
_EJ = np.array([IDX[j] for j in EFFORT_JOINTS])
_CAP = np.array([CAPACITY.get(j[1:] if j[0] in "lr" and j[1:] in CAPACITY else j, CAPACITY["L5"])
                 for j in EFFORT_JOINTS])
_MASK_SEG = np.array([[1.0 if n in _DISTAL[j] else 0.0 for n in _SEG_NAMES] for j in EFFORT_JOINTS])
_CONTACT_OWNER = {"lheel": "l_foot", "lbigtoe": "l_foot", "lsmalltoe": "l_foot", "lank": "l_shank",
                  "rheel": "r_foot", "rbigtoe": "r_foot", "rsmalltoe": "r_foot", "rank": "r_shank",
                  "lknee": "l_thigh", "rknee": "r_thigh", "lwri": "l_hand", "rwri": "r_hand",
                  "lhand": "l_hand", "rhand": "r_hand", "pelvis": "lower_trunk"}
FOOT_POINTS = {"l": ["lheel", "lbigtoe", "lsmalltoe"], "r": ["rheel", "rbigtoe", "rsmalltoe"]}


def _support_basis(down):
    e1 = _u(np.array([1.0, 0, 0]) - down * down[0])
    return e1, _u(_cross(down, e1))


class Physics:
    """Extra residual block + parameters (camera pitch) for fit_skeleton."""

    def __init__(self, contacts, height, mass=None, dynamic=False, w=None):
        self.contacts = list(contacts)
        self.ci = np.array([IDX[c] for c in self.contacts], int)
        self.height = height
        self.mass = mass or 22.0 * height ** 2
        self.dynamic = dynamic
        # stress: unified muscle effort-stress (gravity demand + length state, muscles.py)
        # passive: relaxed-posture prior (statistical naturalness; also covers axial rotations)
        self.w = dict(passive=0.7, stress=3.0, balance=0.0 if dynamic else 1.0,
                      floor=0.0 if dynamic else 1.0, pitch=1.0)
        if w:
            self.w.update(w)
        owner = [_CONTACT_OWNER.get(c, "lower_trunk") for c in self.contacts]
        self.mask_c = np.array([[1.0 if o in _DISTAL[j] else 0.0 for o in owner] for j in EFFORT_JOINTS]).reshape(len(EFFORT_JOINTS), -1)
        self.n_params = 1

    # ---- pieces ------------------------------------------------------------------------
    def com_points(self, X):
        P = X[_SA] + _SR[:, None] * (X[_SB] - X[_SA])
        pel, nk = X[IDX["pelvis"]], X[IDX["neck"]]
        T = pel + _TS[:, None] * (nk - pel)
        return np.vstack([P, T]), np.r_[_SM, _TM] * self.mass

    def solve(self, X, down):
        """COM, centre of pressure, reaction weights, joint torques (camera frame)."""
        P, m = self.com_points(X)
        com = (m @ P) / m.sum()
        e1, e2 = _support_basis(down)
        if len(self.ci):
            C = X[self.ci]
            A = np.vstack([C @ e1, C @ e2, np.ones(len(C))])
            b = np.array([com @ e1, com @ e2, 1.0])
            lam = 1e-3
            n = len(C)
            w = np.linalg.solve(A.T @ A * np.array(1.0) + lam * np.eye(n), A.T @ b + lam / n)
            w = np.clip(w, 0, None); w = w / max(w.sum(), 1e-9)
            cop = w @ C
        else:
            C, w, cop = np.zeros((0, 3)), np.zeros(0), com
        Fg = np.outer(m * G, down)                                   # gravity, down
        Fc = -np.outer(w * m.sum() * G, down)                        # reactions, up
        cg = np.cross(P, Fg); cc = np.cross(C, Fc) if len(C) else np.zeros((0, 3))
        Jx = X[_EJ]
        Fsum = _MASK_SEG @ Fg + (self.mask_c @ Fc if len(C) else 0)
        tau = _MASK_SEG @ cg + (self.mask_c @ cc if len(C) else 0) - np.cross(Jx, Fsum)
        return com, cop, w, tau

    def residuals(self, X, p):
        pitch = p[0]
        down = gravity_dir(pitch)
        res = []
        pr, ang = passive_residuals(X, down)
        res.append(self.w["passive"] * pr)
        com, cop, w, tau = self.solve(X, down)
        from .muscles import stress_residuals
        sr, _ = stress_residuals(X, down, frames(X, down), ang, dict(zip(EFFORT_JOINTS, tau)))
        res.append(self.w["stress"] * sr)
        if len(self.ci):
            off = (com - cop) - ((com - cop) @ down) * down
            res.append(self.w["balance"] * np.array([np.sqrt(off @ off) / 0.04]))
            h = -(X[self.ci] @ down)                                   # heights along "up"
            res.append(self.w["floor"] * (h - h.mean()) / 0.025)
        res.append(self.w["pitch"] * np.array([np.degrees(pitch) / 8.0]))
        return np.concatenate(res)

    def breakdown(self, X, p):
        down = gravity_dir(p[0])
        pr, ang = passive_residuals(X, down)
        com, cop, w, tau = self.solve(X, down)
        eff = np.sqrt((tau * tau).sum(1)) / _CAP
        from .muscles import muscle_stress
        ms = muscle_stress(X, down, frames(X, down), ang, dict(zip(EFFORT_JOINTS, tau)))
        out = dict(passive=float((pr ** 2).sum()), effort=float((eff ** 2).sum()),
                   stress=float(sum(g["stress"] ** 2 for g in ms["groups"].values())),
                   stress_index=ms["index"], stress_peak=ms["peak"],
                   stress_top=dict(sorted(((k, round(g["stress"], 3)) for k, g in ms["groups"].items()),
                                          key=lambda kv: -kv[1])[:5]),
                   joint_stress={j: round(v, 3) for j, v in ms["joints"].items()})
        if len(self.ci):
            off = (com - cop) - ((com - cop) @ down) * down
            h = -(X[self.ci] @ down)
            out.update(balance=float((np.sqrt(off @ off) / 0.04) ** 2), floor=float((((h - h.mean()) / 0.025) ** 2).sum()))
        names = [k for k in ang]
        dev = {}
        for k, r in zip(names, pr):
            dev[k] = float(r)
        out["worst_joints"] = dict(sorted(((k, round(v, 2)) for k, v in dev.items()), key=lambda kv: -abs(kv[1]))[:5])
        out["joint_effort"] = {j: round(float(e), 3) for j, e in zip(EFFORT_JOINTS, eff)}
        out["pitch_deg"] = float(np.degrees(p[0]))
        out["angles"] = {k: round(float(v), 1) for k, v in ang.items()}
        return out


def choose_contacts(X, valid, height, down=np.array([0.0, 1.0, 0.0]), dynamic=False):
    """Foot points near the floor; in static mode both feet are assumed planted when the higher
    foot is within 15 cm (a noisy fit often leaves a planted foot hovering)."""
    tol = max(0.04, 0.025 * height)
    h = -(X @ down)
    lowest = {s: min(h[IDX[p]] for p in pts if valid[IDX[p]]) for s, pts in FOOT_POINTS.items()
              if any(valid[IDX[p]] for p in pts)}
    if not lowest:
        return []
    floor = min(lowest.values())
    out = []
    for s, pts in FOOT_POINTS.items():
        if s not in lowest:
            continue
        if lowest[s] - floor < tol or (not dynamic and lowest[s] - floor < 0.15):
            lo = lowest[s]
            out += [p for p in pts if valid[IDX[p]] and h[IDX[p]] - lo < 0.06]
    return out

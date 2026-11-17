"""
Unified muscle effort-stress: gravity load + length state in one Hill-type measure.

For every anatomical degree of freedom an agonist/antagonist pair of lumped muscle groups is
modelled with a moment arm r, an optimal fibre length L0 and a maximal isometric moment M_max.

  length       l~ = 1 + r * (theta - theta_rest) / L0 * side   (normalised fibre length; the
               relaxed posture is l~ = 1; flexing shortens flexors and stretches extensors)
  passive      F_p(l~) = (exp(k (l~-1)/e0) - 1) / (exp(k) - 1) for l~ > 1, else 0
               (Thelen 2003, k = 4, e0 = 0.6): tension of a stretched muscle-tendon unit
  force-length f_l(l~) = exp(-(l~-1)^2 / 0.45)                 (active capacity drops when the
               muscle is very short or very long)
  demand       M_req = -tau_ext . axis   (moment the muscles must supply against gravity +
               ground reaction, from the static solve)
  active       the passive moments of both groups help or hinder; the remainder is supplied by the
               group on the needed side:  a = M_act / (M_max * f_l)
  stress       s = sqrt(a^2 + F_p^2)      per group, 0 = slack and idle, 1 = maximal

So holding a load at a joint angle where the working muscle is short (weak) costs more, and a
pose that stretches a muscle costs even without load: the two effects the artist reads as
"tension" are one number. Rotational DoFs without a modelled pair (hip/shoulder axial rotation,
foot turn-out, head yaw) keep the plain relaxed-angle prior in biomech.py.

Parameter values are lumped literature-order estimates (Delp 1990, Holzbaur 2005, Arnold 2010
via OpenSim models); they set relative costs, not exact forces.
"""
from __future__ import annotations
import numpy as np

from .skeleton import IDX

D2R = np.pi / 180
_E0, _KPE, _GAMMA = 0.6, 4.0, 0.45

# DoF: (angle key, torque joint, axis key, rest angle deg,
#       [(group, side +1/-1 = moves the angle up/down, moment arm m, L0 m, M_max N·m), ...])
DOFS = {
    "hip_flex":   ("hip_flex", "hip", "flex_p", 5.0,
                   [("hip_flexors", +1, 0.030, 0.10, 150), ("hip_extensors", -1, 0.050, 0.12, 250)]),
    "hip_abd":    ("hip_abd", "hip", "abd_p", 6.0,
                   [("hip_abductors", +1, 0.050, 0.07, 120), ("hip_adductors", -1, 0.040, 0.12, 140)]),
    "knee_flex":  ("knee_flex", "knee", "knee", 5.0,
                   [("knee_flexors", +1, 0.030, 0.10, 120), ("knee_extensors", -1, 0.040, 0.09, 250)]),
    "ankle":      ("ankle_dorsi", "ank", "ankle", 0.0,
                   [("dorsiflexors", +1, 0.040, 0.07, 50), ("plantarflexors", -1, 0.050, 0.05, 150)]),
    "sho_flex":   ("sho_flex", "sho", "flex_t", 8.0,
                   [("shoulder_flexors", +1, 0.030, 0.10, 70), ("shoulder_extensors", -1, 0.030, 0.12, 80)]),
    "sho_abd":    ("sho_abd", "sho", "abd_t", 10.0,
                   [("shoulder_abductors", +1, 0.030, 0.10, 60), ("shoulder_adductors", -1, 0.040, 0.15, 90)]),
    "elbow_flex": ("elbow_flex", "elb", "elbow", 15.0,
                   [("elbow_flexors", +1, 0.035, 0.11, 70), ("elbow_extensors", -1, 0.020, 0.09, 50)]),
}
TRUNK_DOFS = {
    "trunk_flex": ("trunk_flex", "pelvis", "trunk_flex", 0.0,
                   [("abdominals", +1, 0.080, 0.30, 150), ("back_extensors", -1, 0.050, 0.12, 230)]),
    "trunk_bend": ("trunk_bend", "pelvis", "trunk_bend", 0.0,
                   [("right_lateral_flexors", +1, 0.100, 0.15, 150), ("left_lateral_flexors", -1, 0.100, 0.15, 150)]),
    "neck_flex":  ("neck_flex", "neck", "neck_flex", 0.0,
                   [("neck_flexors", +1, 0.020, 0.08, 20), ("neck_extensors", -1, 0.030, 0.10, 40)]),
}


def passive(lt):
    x = np.maximum(lt - 1.0, 0.0)
    return (np.exp(_KPE * x / _E0) - 1.0) / (np.exp(_KPE) - 1.0)


def force_length(lt):
    return np.exp(-(lt - 1.0) ** 2 / _GAMMA)


def _u(v):
    return v / (np.sqrt(v @ v) + 1e-9)


def axes(X, down, frames):
    """Rotation axes (camera frame) for each DoF, oriented so +rotation = +angle."""
    (upp, rp, fp), (upt, rt, ft) = frames["pelvis"], frames["thorax"]
    up = -down
    ax = {}
    for s, lat_sign in (("l", -1.0), ("r", 1.0)):
        latp, latt = rp * lat_sign, rt * lat_sign
        ax[s + "flex_p"] = rp                                   # (-up) -> fwd rotates about +r
        ax[s + "abd_p"] = _u(np.cross(-upp, latp))
        ax[s + "knee"] = -rp                                    # shank swings back about -r
        ax[s + "ankle"] = rp                                    # toes up about +r
        ax[s + "flex_t"] = rt
        ax[s + "abd_t"] = _u(np.cross(-upt, latt))
        ua = X[IDX[s + "elb"]] - X[IDX[s + "sho"]]
        fa = X[IDX[s + "wri"]] - X[IDX[s + "elb"]]
        n = np.cross(ua, fa)
        ax[s + "elbow"] = _u(n) if np.sqrt(n @ n) > 0.2 * np.sqrt(ua @ ua) * np.sqrt(fa @ fa) else rt
    ax["trunk_flex"] = -rp                                      # up -> fwd rotates about -r
    ax["trunk_bend"] = fp                                       # up -> right rotates about fwd
    ax["neck_flex"] = -rt
    return ax


def extra_angles(X, down, frames):
    """Trunk and neck flexion (deg) not provided by biomech.joint_angles."""
    (upp, rp, fp), (upt, rt, ft) = frames["pelvis"], frames["thorax"]
    trunk = _u(X[IDX["neck"]] - X[IDX["pelvis"]])
    lean = np.degrees(np.arctan2(trunk @ fp, trunk @ (-down)))
    head = _u(X[IDX["head"]] - X[IDX["neck"]])
    neck = np.degrees(np.arctan2(head @ ft, head @ trunk)) - 10.0   # head carried ~10° forward
    # lumbar flexion ~ 60% of the trunk's sagittal lean (the pelvis tilts for the rest)
    return {"trunk_flex": 0.6 * lean, "neck_flex": neck}


def muscle_stress(X, down, frames, angles, tau_by_joint):
    """Per-group dict(stress, activation, passive, length) + per-joint and total summaries.
    tau_by_joint: external moment (camera frame) of distal loads about each joint."""
    ang = dict(angles)
    ang.update(extra_angles(X, down, frames))
    ax = axes(X, down, frames)
    groups, joints = {}, {}

    def one(prefix, dof, spec):
        key, jnt, axk, rest, grp = spec
        a_key = (prefix + key) if prefix else key
        if a_key not in ang:
            return
        th = (ang[a_key] - rest) * D2R
        jname = (prefix + jnt) if prefix else jnt
        tau = tau_by_joint.get(jname)
        m_req = 0.0 if tau is None else float(-(tau @ ax[(prefix + axk) if prefix else axk]))
        lts = [1.0 - side * r * th / L0 for _, side, r, L0, _ in grp]      # agonist shortens
        fps = [float(passive(lt)) for lt in lts]
        m_pas = sum(side * fp * Mmax for (_, side, r, L0, Mmax), fp in zip(grp, fps))
        m_act = m_req - m_pas
        for (gname, side, r, L0, Mmax), lt, fp in zip(grp, lts, fps):
            a = max(0.0, side * m_act) / (Mmax * float(force_length(lt)))
            s = float(np.sqrt(a * a + fp * fp))
            name = (prefix + gname) if prefix else gname
            groups[name] = dict(stress=s, activation=a, passive=fp, length=float(lt), joint=jname, dof=dof)
            joints[jname] = max(joints.get(jname, 0.0), s)

    for s in "lr":
        for dof, spec in DOFS.items():
            one(s, dof, spec)
    for dof, spec in TRUNK_DOFS.items():
        one("", dof, spec)
    vals = np.array([g["stress"] for g in groups.values()])
    return dict(groups=groups, joints=joints, index=float(np.sqrt((vals ** 2).mean())) if len(vals) else 0.0,
                peak=float(vals.max()) if len(vals) else 0.0)


def stress_residuals(X, down, frames, angles, tau_by_joint):
    ms = muscle_stress(X, down, frames, angles, tau_by_joint)
    return np.array([g["stress"] for g in ms["groups"].values()]), ms


def frame_stress(frame, contacts=None, mass=None):
    """Muscle stress of a PoseFrame (camera frame, its gravity direction and contacts)."""
    from .biomech import Physics, frames as body_frames, joint_angles, EFFORT_JOINTS
    X = frame.joints3d
    down = np.asarray(frame.gravity_dir, float)
    cs = [c for c in (contacts if contacts is not None else frame.contacts or []) if c in IDX]
    ph = Physics(cs, frame.body_height_m, mass=mass)
    _, _, _, tau = ph.solve(X, down)
    ang = joint_angles(X, down)
    ang = ang[0] if isinstance(ang, tuple) else ang
    return muscle_stress(X, down, body_frames(X, down), ang, dict(zip(EFFORT_JOINTS, tau)))

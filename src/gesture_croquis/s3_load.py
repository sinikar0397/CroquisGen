"""
S3 tier T0: static load on a 3D skeleton (world frame, y up, gravity -y).

1. Segment masses: de Leva (1996) fractions × body mass. The trunk is split into
   lower/middle/upper parts placed ON THE SPINE CURVE, so spine load reflects bending.
2. Contacts: points near the ground (feet, and knees/hands/pelvis for kneeling, crawling or
   sitting on the floor), or given explicitly.
3. Ground reactions: vertical forces at the contacts whose centre of pressure matches the
   body COM's ground projection (bounded least squares, smoothed toward an even split).
   If the COM lies outside the support polygon, the residual is reported as `com_offset_m`:
   the pose is falling/moving, which is a motion cue for S4 rather than a load.
4. Joint torque = moment about the joint of every gravity and reaction force distal to it
   (the pelvis is the root). Effort = |torque| / isometric capacity.
5. Stretch proxy: joint flexion angles + torso-side strain (short side = compressed).

All of this is a static approximation (no accelerations, no muscle redundancy) — tier T1
(OpenSim / MyoSuite static optimisation) replaces steps 4–5 later.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import lsq_linear

from .skeleton import IDX, JOINTS
from .spine import spine_curve

G = 9.81
UP = np.array([0.0, 1.0, 0.0])

# name: (proximal, distal, mass fraction, COM ratio from proximal)   — de Leva 1996, male
LIMB_SEGMENTS = {
    "l_uarm": ("lsho", "lelb", 0.0271, 0.577), "r_uarm": ("rsho", "relb", 0.0271, 0.577),
    "l_farm": ("lelb", "lwri", 0.0162, 0.457), "r_farm": ("relb", "rwri", 0.0162, 0.457),
    "l_hand": ("lwri", "lhand", 0.0061, 0.790), "r_hand": ("rwri", "rhand", 0.0061, 0.790),
    "l_thigh": ("lhip", "lknee", 0.1416, 0.410), "r_thigh": ("rhip", "rknee", 0.1416, 0.410),
    "l_shank": ("lknee", "lank", 0.0433, 0.446), "r_shank": ("rknee", "rank", 0.0433, 0.446),
    "l_foot": ("lheel", "lbigtoe", 0.0137, 0.442), "r_foot": ("rheel", "rbigtoe", 0.0137, 0.442),
    "head": ("neck", "head", 0.0694, 0.500),
}
TRUNK_PARTS = {"lower_trunk": (0.1117, 0.08), "middle_trunk": (0.1633, 0.42), "upper_trunk": (0.1596, 0.78)}

# Distal body regions (which segments / contact points sit beyond each joint)
_LEG = lambda s: dict(hip=[f"{s}_thigh", f"{s}_shank", f"{s}_foot"], knee=[f"{s}_shank", f"{s}_foot"],
                      ank=[f"{s}_foot"])
_ARM = lambda s: dict(sho=[f"{s}_uarm", f"{s}_farm", f"{s}_hand"], elb=[f"{s}_farm", f"{s}_hand"],
                      wri=[f"{s}_hand"])
SEG_POINTS = {  # contact-capable joints belonging to each segment
    "l_thigh": ["lknee"], "r_thigh": ["rknee"], "l_shank": ["lank"], "r_shank": ["rank"],
    "l_foot": ["lheel", "lbigtoe", "lsmalltoe"], "r_foot": ["rheel", "rbigtoe", "rsmalltoe"],
    "l_farm": ["lelb"], "r_farm": ["relb"], "l_hand": ["lwri", "lhand"], "r_hand": ["rwri", "rhand"],
    "head": ["head"], "lower_trunk": ["pelvis", "lhip", "rhip"], "upper_trunk": ["lsho", "rsho"],
}
# Rough adult isometric torque capacity (N·m); only the ratios matter for drawing.
CAPACITY = dict(hip=200, knee=180, ank=150, sho=70, elb=60, wri=15, neck=40,
                L5=220, L3=220, L1=200, T12=200, T8=180, T4=150)
CONTACT_CANDIDATES = ["lheel", "rheel", "lbigtoe", "rbigtoe", "lsmalltoe", "rsmalltoe", "lank", "rank",
                      "lknee", "rknee", "lhand", "rhand", "lwri", "rwri", "pelvis"]


def _u(v):
    return v / (np.linalg.norm(v) + 1e-9)


def _angle(a, b, c):
    """Interior angle at b (deg)."""
    u, v = _u(a - b), _u(c - b)
    return float(np.degrees(np.arccos(np.clip(u @ v, -1, 1))))


def mass_points(W, valid, sp, body_mass):
    """Return {segment: (com (3,), mass kg)} in world frame."""
    out = {}
    for name, (a, b, f, r) in LIMB_SEGMENTS.items():
        ia, ib = IDX[a], IDX[b]
        if name.endswith("foot") and not (valid[ia] and valid[ib]):      # no foot points: use ankle
            ia = ib = IDX[a[0] + "ank"]
        if name.endswith("hand") and not valid[ib]:
            out[name] = (W[ia], f * body_mass); continue
        out[name] = (W[ia] + r * (W[ib] - W[ia]), f * body_mass)
    for name, (f, s) in TRUNK_PARTS.items():
        out[name] = (np.array([np.interp(s, sp["s"], sp["trunk"][:, c]) for c in range(3)]), f * body_mass)
    return out


def detect_contacts(W, valid, height, tol=None):
    tol = tol or max(0.04, 0.025 * height)
    feet = [IDX[n] for n in CONTACT_CANDIDATES[:8] if valid[IDX[n]]]
    ground = W[feet, 1].min() if feet else W[valid, 1].min()
    return [n for n in CONTACT_CANDIDATES if valid[IDX[n]] and W[IDX[n], 1] - ground < tol]


def distribute_reactions(W, contacts, com, total_weight, smooth=0.05):
    """Vertical reaction per contact: sum = weight, centre of pressure ≈ COM ground projection."""
    if not contacts:
        return {}, None
    P = np.stack([W[IDX[c]] for c in contacts])
    n = len(contacts)
    A = np.vstack([P[:, 0], P[:, 2], np.ones(n) * 1.0])               # x, z moments + force sum
    b = np.array([com[0], com[2], 1.0])
    A_aug = np.vstack([A * [[1], [1], [10]], smooth * np.eye(n)])      # force sum ~ hard
    b_aug = np.r_[b * [1, 1, 10], smooth * np.ones(n) / n]
    w = lsq_linear(A_aug, b_aug, bounds=(0, np.inf)).x
    w = w / max(w.sum(), 1e-9)
    cop = w @ P
    return {c: float(wi * total_weight) for c, wi in zip(contacts, w)}, cop


def _support_offset(points_xz, p):
    """Distance from p to the convex hull of points (0 if inside)."""
    from scipy.spatial import ConvexHull
    pts = np.unique(np.round(points_xz, 4), axis=0)
    if len(pts) == 1:
        return float(np.linalg.norm(p - pts[0]))
    if len(pts) == 2:
        a, b = pts; t = np.clip((p - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1)
        return float(np.linalg.norm(p - (a + t * (b - a))))
    try:
        h = ConvexHull(pts)
    except Exception:
        return 0.0
    eq = h.equations                                  # n·x + c <= 0 inside
    if np.all(eq[:, :2] @ p + eq[:, 2] <= 1e-9):
        return 0.0
    d = []
    for s in h.simplices:
        a, b = pts[s]; t = np.clip((p - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1)
        d.append(np.linalg.norm(p - (a + t * (b - a))))
    return float(min(d))


def static_load(frame, body_mass=None, contacts=None, sagittal=0.0, g_scale=1.0):
    """Fill frame.spine3d / com3d / contacts / joint_torque / joint_effort / muscle_stretch / s3_info.
    contacts=None: detected here (lowest foot points). contacts=[] with g_scale=0: flight (video,
    tracked floor): no ground reaction and no gravity load in the body frame (free fall), so the
    remaining stress is the passive length state alone."""
    H = frame.body_height_m
    body_mass = body_mass or 22.0 * H * H                       # BMI 22 default
    v = frame.valid3d
    W = frame.world_joints()
    sp = spine_curve(W, sagittal=sagittal)
    m = mass_points(W, v, sp, body_mass)
    com = sum(p * mk for p, mk in m.values()) / sum(mk for _, mk in m.values())
    auto = contacts is None
    contacts = detect_contacts(W, v, H) if auto else [c for c in contacts if v[IDX[c]]]
    off = lambda cs: _support_offset(np.stack([W[IDX[c], [0, 2]] for c in cs]), com[[0, 2]]) if cs else np.inf
    added = []
    if auto:
        # static assumption: if the COM is outside the support, the 3D fit probably left a
        # planted foot slightly in the air -> add the next-lowest foot points (<= 15 cm up)
        ground = min(W[IDX[c], 1] for c in contacts) if contacts else 0.0
        rest = sorted((n for n in CONTACT_CANDIDATES[:8] if v[IDX[n]] and n not in contacts),
                      key=lambda n: W[IDX[n], 1])
        while off(contacts) > 0.05 and rest and W[IDX[rest[0]], 1] - ground < 0.15:
            nxt = rest.pop(0); contacts.append(nxt); added.append(nxt)
    R, cop = distribute_reactions(W, contacts, com, body_mass * G)
    offset = off(contacts) if contacts else None

    ge = G * g_scale                                                # effective gravity
    if R:
        R = {c_: f * g_scale for c_, f in R.items()}
    forces = [(p, -UP * mk * ge, seg) for seg, (p, mk) in m.items()]          # gravity
    owner = {pt: seg for seg, pts in SEG_POINTS.items() for pt in pts}
    forces += [(W[IDX[c]], UP * f, owner.get(c, "lower_trunk")) for c, f in R.items()]   # reactions

    def moment(about, segs):
        return sum((np.cross(p - about, F) for p, F, s in forces if s in segs), np.zeros(3))

    torque = {}
    for s in "lr":
        for j, segs in _LEG(s).items():
            torque[s + j] = moment(W[IDX[s + j]], segs)
        for j, segs in _ARM(s).items():
            torque[s + j] = moment(W[IDX[s + j]], segs)
    torque["neck"] = moment(W[IDX["neck"]], ["head"])
    upper_body = ["head", "l_uarm", "l_farm", "l_hand", "r_uarm", "r_farm", "r_hand"]
    trunk_s = {n: s for n, (_, s) in TRUNK_PARTS.items()}
    from .spine import LANDMARKS
    for lm in ("L5", "L3", "L1", "T12", "T8", "T4"):
        segs = upper_body + [n for n, s in trunk_s.items() if s > LANDMARKS[lm]]
        torque[lm] = moment(sp["landmarks"][lm], segs)

    # express torques in the body frame for interpretability (flexion / lateral / twist)
    up = _u(W[IDX["neck"]] - W[IDX["pelvis"]])
    right = _u((W[IDX["rhip"]] - W[IDX["lhip"]]) + (W[IDX["rsho"]] - W[IDX["lsho"]]))
    fwd = _u(np.cross(up, right)); right = _u(np.cross(fwd, up))
    cap = lambda j: CAPACITY.get(j[1:] if j[0] in "lr" and j[1:] in CAPACITY else j, 100)
    effort = {j: float(np.linalg.norm(t) / cap(j)) for j, t in torque.items()}
    mx = max(effort.values()) or 1.0

    # ---- stretch / compression proxy -------------------------------------------------
    st = {}
    for s in "lr":
        st[f"{s}_knee_flex"] = 180 - _angle(W[IDX[s + "hip"]], W[IDX[s + "knee"]], W[IDX[s + "ank"]])
        st[f"{s}_elbow_flex"] = 180 - _angle(W[IDX[s + "sho"]], W[IDX[s + "elb"]], W[IDX[s + "wri"]])
        st[f"{s}_hip_flex"] = 180 - _angle(W[IDX["neck"]], W[IDX[s + "hip"]], W[IDX[s + "knee"]])
        st[f"{s}_shoulder_elev"] = _angle(W[IDX[s + "hip"]], W[IDX[s + "sho"]], W[IDX[s + "elb"]])
        side_len = np.linalg.norm(W[IDX[s + "sho"]] - W[IDX[s + "hip"]])
        st[f"{s}_torso_strain"] = float(side_len / (0.29 * H) - 1)       # <0 compressed
    st["spine_lateral_bend"] = sp["lateral_bend_deg"]
    st["spine_twist"] = sp["twist_deg"]
    st["compressed_torso_side"] = "l" if st["l_torso_strain"] < st["r_torso_strain"] else "r"

    frame.body_mass_kg = float(body_mass)
    frame.spine3d = frame.from_world(np.vstack([sp["trunk"], sp["neck"]]))
    frame.com3d = frame.from_world(com[None])[0]
    frame.contacts = list(R.keys())
    frame.joint_torque = {j: dict(flex=float(t @ right), lateral=float(t @ fwd), twist=float(t @ up),
                                  norm=float(np.linalg.norm(t))) for j, t in torque.items()}
    frame.joint_effort = {j: e / mx for j, e in effort.items()}
    frame.muscle_stretch = {k: (float(v_) if not isinstance(v_, str) else v_) for k, v_ in st.items()}
    frame.s3_info = dict(body_mass_kg=float(body_mass), contacts_auto=auto, g_scale=float(g_scale),
                         airborne=bool(not contacts and g_scale == 0.0), contacts_added_for_balance=added,
                         reactions_N={c: round(f, 1) for c, f in R.items()},
                         com_world=com.round(3).tolist(), cop_world=None if cop is None else cop.round(3).tolist(),
                         com_offset_m=offset, balanced=(offset is not None and offset < 0.03),
                         effort_fraction_of_capacity={j: round(e, 3) for j, e in effort.items()},
                         peak_effort_joint=max(effort, key=effort.get),
                         spine_max_offset_m=sp["max_offset"])
    # ---- unified muscle effort-stress (gravity demand + length state) ------------------
    try:
        from .muscles import frame_stress
        ms = frame_stress(frame, mass=body_mass, g_scale=g_scale)
        frame.s3_info.update(stress_index=round(ms["index"], 3), stress_peak=round(ms["peak"], 3),
                             muscle_stress={k: {kk: round(vv, 3) if isinstance(vv, float) else vv
                                                for kk, vv in g.items()} for k, g in ms["groups"].items()},
                             joint_stress={j: round(v_, 3) for j, v_ in ms["joints"].items()},
                             peak_stress_muscle=max(ms["groups"], key=lambda k: ms["groups"][k]["stress"]))
        # joint markers: the unified stress replaces |torque|/capacity when available
        js = dict(ms["joints"])
        if "pelvis" in js:
            js["L5"] = js.pop("pelvis")
        top = max(js.values()) if js else 0.0
        if top > 0:
            frame.joint_effort = {j: v_ / max(top, 0.5) for j, v_ in js.items()}
    except Exception as e:                                   # keep S3 usable on odd frames
        frame.s3_info["stress_error"] = repr(e)
    return frame

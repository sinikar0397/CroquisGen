"""
S4: abstraction. PoseFrame (S2 3D skeleton + spine, S3 load / muscle state) -> gesture strokes.

Layers (as in the reference sheet):
  A  line of action   one C/S curve through the longest, most loaded chain, exaggerated
  B  gesture/rhythm   one curve per limb, torso squash/stretch sides, shoulder + hip lines,
                      head, feet
  C  construction     ribcage / pelvis volumes and joint circles (off by default)

Nothing here moves a joint. Every stroke is built from the projected skeleton; the only freedom
is which chain to use, a cubic-Bezier simplification, a bounded bow (exaggeration and the
muscle-side bias) and the line weight. A later learned stylisation may only perturb stroke
parameters (`Stroke.pts` offsets / `Stroke.width`), never `PoseFrame.joints3d`.

Coordinates: strokes are in VIEW pixels. For the photo view (yaw = pitch = 0) that is the
photo's pixel frame, so strokes overlay the photo directly. A novel view rotates the body about
the vertical axis through the pelvis (yaw, + = the body turns to its own left) and about the
horizontal axis through the pelvis (pitch, + = seen more from above), then reuses the photo
camera K at the same distance. Without S2 (joints3d is None) S4 falls back to the 2D keypoints
(photo view only, no depth cues).

Line weight per stroke sample (px):
    w = w0[role] * S * (1 + a*effort + b*compression + c*near + e*support) * taper(t)
          * (1 + d*front) * (1 - (1 - lost)*behind)
  S        = unit width: torso length in px / 120
  effort   = S3 joint effort-stress interpolated along the stroke
  compression = joint flexion (knee / elbow / hip) or the squash of a torso side
  near     = 1 at the nearest body point, 0 at the farthest (view depth)
  support  = fraction of body weight carried by that limb (S3 ground reactions)
  front / behind = where two layer-B strokes cross: the nearer one gets an accent, the farther
             one is thinned to a "lost line" (T-junction / overlap cue). The LOA takes no part.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json
import time
import numpy as np

from .skeleton import IDX
from .spine import spine_curve, polyline, N_TRUNK, N_NECK, LANDMARKS


# ============================================================================== parameters
@dataclass
class S4Params:
    exaggeration: float = 0.35          # LOA push off the chord (plan: 0.2 .. 0.5)
    limb_exaggeration: float = 0.5      # fraction of `exaggeration` applied to rhythm lines
    limb_bias: float = 0.12             # max muscle-side bow, fraction of limb chord
    torso_bow: float = 0.15             # extra inward bow of a fully compressed torso side
    # weight model
    a_effort: float = 0.8
    b_compression: float = 0.6
    c_near: float = 0.5
    d_overlap: float = 0.6
    e_support: float = 0.5
    lost: float = 0.2                   # width factor where a stroke passes behind another
    depth_margin_m: float = 0.03        # depth difference that counts as "in front"
    taper_min: float = 0.15
    taper_pow: float = 0.6
    weight_smooth: float = 0.08         # Gaussian sigma of the weight factor, fraction of stroke samples
    w0: dict = field(default_factory=lambda: dict(
        loa=7.0, limb=3.0, limb_on_loa=1.4, torso_comp=3.5, torso_str=1.6, girdle=2.2,
        head=2.2, face=1.4, foot=2.2, construction=1.1))
    # LOA score = (L / body_px) * (1 + load_w*load) * (1 + support_w*support) / (1 + resid_w*resid)
    load_w: float = 1.0
    support_w: float = 0.6
    resid_w: float = 25.0
    backtrack_w: float = 10.0


@dataclass
class View:
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0

    @property
    def is_photo(self) -> bool:
        return abs(self.yaw_deg) < 1e-9 and abs(self.pitch_deg) < 1e-9

    @property
    def tag(self) -> str:
        if self.is_photo:
            return ""
        return f"_y{self.yaw_deg:+.0f}" + (f"_p{self.pitch_deg:+.0f}" if self.pitch_deg else "")

    @classmethod
    def parse(cls, s: str) -> "View":
        """'90' -> yaw 90 ; '-45:20' -> yaw -45, pitch 20."""
        y, _, p = str(s).partition(":")
        return cls(float(y or 0), float(p or 0))


@dataclass
class Stroke:
    name: str
    layer: str                          # 'A' | 'B' | 'C'
    role: str                           # key of S4Params.w0
    pts: np.ndarray                     # (N,2) view px
    depth: np.ndarray                   # (N,) metres along the view axis (0 in 2D mode)
    effort: np.ndarray                  # (N,) 0..1
    compression: np.ndarray             # (N,) 0..1
    support: float = 0.0
    closed: bool = False
    group: str = ""                     # strokes of one group don't create overlap cues on each other
    width: np.ndarray | None = None     # (N,) px, filled by the weight pass
    front: np.ndarray | None = None     # (N,) 0..1 accent mask
    behind: np.ndarray | None = None    # (N,) 0..1 lost-line mask
    meta: dict = field(default_factory=dict)

    @property
    def t(self) -> np.ndarray:
        return _arc_t(self.pts)

    def to_dict(self):
        d = dict(name=self.name, layer=self.layer, role=self.role, closed=self.closed,
                 support=round(float(self.support), 3), meta=self.meta,
                 pts=np.round(self.pts, 1).tolist(),
                 width=None if self.width is None else np.round(self.width, 2).tolist(),
                 depth=np.round(self.depth, 3).tolist())
        return d


@dataclass
class Croquis:
    canvas_hw: tuple                    # (H, W) of the view (the photo size)
    view: View
    unit_px: float                      # S
    strokes: list
    loa: dict
    info: dict

    def layer(self, L: str) -> list:
        return [s for s in self.strokes if s.layer == L]

    def get(self, name: str) -> Stroke | None:
        return next((s for s in self.strokes if s.name == name), None)

    def to_dict(self):
        return dict(canvas_hw=list(self.canvas_hw), view=asdict(self.view), unit_px=self.unit_px,
                    loa=self.loa, info=self.info, strokes=[s.to_dict() for s in self.strokes])

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, default=_json_default)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


# ============================================================================== geometry
def _u(v):
    return v / (np.linalg.norm(v) + 1e-12)


def _arc_t(P):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    return d / d[-1] if d[-1] > 1e-9 else np.linspace(0, 1, len(P))


def resample(P, n, *attrs):
    """Arc-length resample P (M,D) to n points; attrs (M,) are interpolated along. Returns
    (P', [attrs'], length)."""
    P = np.asarray(P, float)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    L = float(d[-1])
    t = (d + 1e-9 * np.arange(len(d))) / (L + 1e-9 * len(d))     # strictly increasing
    tt = np.linspace(0, 1, n)
    Q = np.stack([np.interp(tt, t, P[:, c]) for c in range(P.shape[1])], -1)
    return Q, [np.interp(tt, t, np.asarray(a, float)) for a in attrs], L


def _bez_basis(t):
    t = np.asarray(t)[:, None]
    return np.hstack([(1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2, t ** 3])


def bezier(ctrl, n=100):
    return _bez_basis(np.linspace(0, 1, n)) @ ctrl


def bezier_even(ctrl, n=100):
    """n points evenly spaced in arc length."""
    Q, _, _ = resample(bezier(ctrl, 400), n)
    return Q


def fit_cubic_bezier(pts, iters=8):
    """Least-squares cubic Bezier with fixed end points; chord-length start, then `iters`
    rounds of nearest-point re-parameterisation. Returns (ctrl (4,D), residual / arc length)."""
    pts = np.asarray(pts, float)
    L = max(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum(), 1e-9)
    t = _arc_t(pts)
    P0, P3 = pts[0], pts[-1]
    for it in range(iters + 1):
        B = _bez_basis(t)
        rhs = pts - np.outer(B[:, 0], P0) - np.outer(B[:, 3], P3)
        P12, *_ = np.linalg.lstsq(B[:, 1:3], rhs, rcond=None)
        ctrl = np.vstack([P0, P12, P3])
        if it == iters:
            break
        tt = np.linspace(0, 1, 256)
        dense = _bez_basis(tt) @ ctrl
        t = tt[np.argmin(((pts[:, None] - dense[None]) ** 2).sum(-1), axis=1)]
        for _ in range(2):                                         # Newton steps on |B(t) - p|^2
            Bt = _bez_basis(t) @ ctrl
            d1 = 3 * ((1 - t)[:, None] ** 2 * (ctrl[1] - ctrl[0]) + 2 * ((1 - t) * t)[:, None] * (ctrl[2] - ctrl[1])
                      + (t ** 2)[:, None] * (ctrl[3] - ctrl[2]))
            d2 = 6 * ((1 - t)[:, None] * (ctrl[2] - 2 * ctrl[1] + ctrl[0]) + t[:, None] * (ctrl[3] - 2 * ctrl[2] + ctrl[1]))
            r = Bt - pts
            den = (d1 * d1).sum(1) + (r * d2).sum(1)
            t = np.clip(t - (r * d1).sum(1) / np.where(np.abs(den) > 1e-9, den, 1e-9), 0, 1)
        t = np.maximum.accumulate(t); t[0], t[-1] = 0.0, 1.0
    resid = float(np.linalg.norm(_bez_basis(t) @ ctrl - pts, axis=1).mean() / L)
    return ctrl, resid


def _normal2(chord):
    c = _u(chord)
    return np.array([-c[1], c[0]])


def exaggerate(ctrl, k):
    """Push the inner control points off the chord: perpendicular offsets x (1 + k)."""
    a, b = ctrl[0], ctrl[3]
    n = _normal2(b - a)
    out = ctrl.copy()
    for i in (1, 2):
        out[i] = ctrl[i] + n * ((ctrl[i] - a) @ n) * k
    return out


def straighten(ctrl, f):
    """Pull the inner control points toward the chord by fraction f (0 = keep, 1 = straight)."""
    a, b = ctrl[0], ctrl[3]
    n = _normal2(b - a)
    out = ctrl.copy()
    for i in (1, 2):
        out[i] = ctrl[i] - n * ((ctrl[i] - a) @ n) * f
    return out


def bezier_offsets(ctrl):
    """Signed max distance of the curve from its chord (px, + along the chord's left normal)."""
    a, b = ctrl[0], ctrl[3]
    d = (bezier(ctrl, 100) - a) @ _normal2(b - a)
    return float(d[np.argmax(np.abs(d))])


def quad_curve(a, b, bow, n):
    """Quadratic from a to b whose midpoint is displaced by the 2D vector `bow`."""
    c = (a + b) / 2 + 2 * bow
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * c + t ** 2 * b


def ellipse(c, a, b, axis, n=72):
    """Closed ellipse: centre c, semi-axis a along `axis` (2D), b across."""
    e1 = _u(np.asarray(axis, float)); e2 = np.array([-e1[1], e1[0]])
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return c + np.outer(np.cos(th) * a, e1) + np.outer(np.sin(th) * b, e2)


def _rodrigues(a, b):
    """Minimal rotation taking unit vector a onto unit vector b."""
    a, b = _u(a), _u(b)
    v, c = np.cross(a, b), float(a @ b)
    if c < -0.999999:                                  # opposite: rotate 180 deg about any normal
        p = _u(np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0]))
        return 2 * np.outer(p, p) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def _angle(a, b, c):
    u, v = _u(a - b), _u(c - b)
    return float(np.degrees(np.arccos(np.clip(u @ v, -1, 1))))


def default_K(hw, fov_deg=55.0):
    h, w = hw
    f = 0.5 * w / np.tan(np.radians(fov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


# ============================================================================== body in view
# Lumped muscle groups -> the side of the limb they sit on, in the limb's anatomical frame
# (anatomical position: limb hanging, anterior = body forward, lateral = away from midline).
LIMB_MUSCLE_SIDES = {
    "leg": {"hip_flexors": ("fwd", +1), "hip_extensors": ("fwd", -1),
            "knee_extensors": ("fwd", +1), "knee_flexors": ("fwd", -1),
            "hip_abductors": ("lat", +1), "hip_adductors": ("lat", -1)},
    "arm": {"shoulder_flexors": ("fwd", +1), "shoulder_extensors": ("fwd", -1),
            "elbow_flexors": ("fwd", +1), "elbow_extensors": ("fwd", -1),
            "shoulder_abductors": ("lat", +1), "shoulder_adductors": ("lat", -1)},
}
_SPINE_KEYS = ["L5", "L3", "L1", "T12", "T8", "T4", "neck"]
_SPINE_S = [LANDMARKS[k] for k in _SPINE_KEYS[:-1]] + [1.0]
_LEG_CONTACT = ("heel", "bigtoe", "smalltoe", "ank", "knee")
_ARM_CONTACT = ("hand", "wri", "elb")


class BodyView:
    """Skeleton, spine and S3 state of one PoseFrame, seen from one View."""

    def __init__(self, frame, view: View = View(), conf_thr: float = 0.2):
        self.frame, self.view = frame, view
        self.H = float(frame.body_height_m or 1.70)
        self.eff = {k: float(v) for k, v in (frame.joint_effort or {}).items()}
        s3 = frame.s3_info or {}
        self.groups = s3.get("muscle_stress", {}) or {}
        if frame.joints3d is not None:
            self.mode = "3d"
            X = np.asarray(frame.joints3d, float)
            self.valid = (np.asarray(frame.valid3d, bool) if frame.valid3d is not None
                          else np.ones(len(X), bool))
            self.K = (np.asarray(frame.camera_K, float) if frame.camera_K is not None
                      else default_K(frame.img_hw))
            if frame.spine3d is not None and len(frame.spine3d) == N_TRUNK + N_NECK:
                Sc = np.asarray(frame.spine3d, float)
            else:
                Sc = frame.from_world(polyline(spine_curve(frame.world_joints())))
            self.X, self.S = self._rotate(X), self._rotate(Sc)
        else:
            if not view.is_photo:
                raise ValueError("novel views need S2 3D joints (frame.joints3d is None)")
            self.mode = "2d"
            k = np.asarray(frame.kpts2d, float)
            self.valid = np.asarray(frame.conf, float) > conf_thr
            self.K = None
            self.X = np.c_[k, np.zeros(len(k))]
            self.S = polyline(spine_curve(self.X))
        self.s_param = np.r_[np.linspace(0, 1, N_TRUNK), 1 + np.linspace(0, 1, N_NECK + 1)[1:]]
        self.uv, self.z = self.proj(self.X)
        self.Suv, self.Sz = self.proj(self.S)
        if self.mode == "2d":
            self._ppm = np.linalg.norm(self.uv[IDX["neck"]] - self.uv[IDX["pelvis"]]) / (0.28 * self.H)
        # body frame (any right-handed frame: camera after the view rotation, or image + z)
        up = _u(self.X[IDX["neck"]] - self.X[IDX["pelvis"]])
        right = _u((self.X[IDX["rhip"]] - self.X[IDX["lhip"]]) + (self.X[IDX["rsho"]] - self.X[IDX["lsho"]]))
        self.fwd = _u(np.cross(up, right))
        self.right = _u(np.cross(self.fwd, up))
        self.up = up
        # weight carried per limb (S3 ground reactions)
        R = s3.get("reactions_N", {}) or {}
        tot = sum(R.values()) or 0.0
        sup = dict(lleg=0.0, rleg=0.0, larm=0.0, rarm=0.0, trunk=0.0)
        for c, f in R.items():
            if c[:1] in "lr" and c[1:] in _LEG_CONTACT:
                sup[c[0] + "leg"] += f
            elif c[:1] in "lr" and c[1:] in _ARM_CONTACT:
                sup[c[0] + "arm"] += f
            else:
                sup["trunk"] += f
        self.support = {k: (v / tot if tot > 0 else 0.0) for k, v in sup.items()}
        vz = self.z[self.valid]
        self.zmin, self.zmax = (float(vz.min()), float(vz.max())) if len(vz) else (0.0, 0.0)

    # ---------------------------------------------------------------- projection
    def _rotate(self, Xcam):
        v = self.view
        if v.is_photo:
            return Xcam
        f = self.frame
        W = f.to_world(Xcam)
        piv = f.to_world(np.asarray(f.joints3d, float))[IDX["pelvis"]]
        y, p = np.radians(v.yaw_deg), np.radians(v.pitch_deg)
        Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
        return f.from_world((W - piv) @ (Rx @ Ry).T + piv)

    def proj(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        if self.mode == "3d":
            z = np.maximum(P[:, 2], 1e-3)
            return P[:, :2] / z[:, None] * self.K[[0, 1], [0, 1]] + self.K[:2, 2], P[:, 2].copy()
        return P[:, :2].copy(), np.zeros(len(P))

    def pxpm(self, P) -> float:
        """Pixels per metre at 3D point P."""
        if self.mode == "3d":
            return float(self.K[0, 0] / max(float(np.asarray(P)[2]), 1e-3))
        return float(self._ppm)

    def project_dir(self, P, d) -> np.ndarray:
        """2D image of a unit 3D direction d at P (length 1 = parallel to the image plane)."""
        eps = 0.05
        a, _ = self.proj(P[None]); b, _ = self.proj((P + eps * d)[None])
        return (b[0] - a[0]) / (eps * self.pxpm(P))

    # ---------------------------------------------------------------- state lookups
    def ok(self, j) -> bool:
        return bool(self.valid[IDX[j]])

    def P(self, j):
        return self.X[IDX[j]]

    def spine_effort(self, s) -> float:
        vals = [self.eff.get(k, 0.0) for k in _SPINE_KEYS]
        return float(np.interp(min(s, 1.0), _SPINE_S, vals)) if s <= 1 else self.eff.get("neck", 0.0)

    def compression(self, j) -> float:
        """Flexion of a hinge as a 0..1 squash cue."""
        tri = {"knee": ("hip", "knee", "ank"), "elb": ("sho", "elb", "wri")}
        s, n = j[:1], j[1:]
        if n in tri:
            a, b, c = (s + x for x in tri[n])
        elif n == "hip":
            a, b, c = "neck", j, s + "knee"
        else:
            return 0.0
        if not all(self.ok(x) for x in (a, b, c)):
            return 0.0
        flex = 180 - _angle(self.P(a), self.P(b), self.P(c))
        return float(np.clip(flex / 120.0, 0, 1))

    def body_px(self) -> float:
        return self.H * self.pxpm(self.P("pelvis"))

    def unit(self) -> float:
        return 0.28 * self.body_px() / 120.0

    def near(self, z) -> np.ndarray:
        rng = self.zmax - self.zmin
        if rng < 0.05:
            return np.full(np.shape(z), 0.5)
        return np.clip((self.zmax - np.asarray(z)) / rng, 0, 1)

    # ---------------------------------------------------------------- chains
    def chain(self, parts):
        """parts: joint names and ('spine', s_from, s_to) tokens (s: 0 pelvis, 1 neck, 2 head).
        Returns dict(P3, uv, z, effort, compression, labels) or None if a joint is missing."""
        P, E, C, lab = [], [], [], []
        for p in parts:
            if isinstance(p, tuple):
                _, a, b = p
                lo, hi = min(a, b) - 1e-9, max(a, b) + 1e-9
                idx = np.where((self.s_param >= lo) & (self.s_param <= hi))[0]
                for i in (idx[::-1] if a > b else idx):
                    P.append(self.S[i]); E.append(self.spine_effort(self.s_param[i])); C.append(0.0)
                    lab.append(None)
            else:
                if not self.ok(p):
                    return None
                P.append(self.P(p)); E.append(self.eff.get(p, 0.0)); C.append(self.compression(p))
                lab.append(p)
        P = np.array(P)
        uv, z = self.proj(P)
        return dict(P3=P, uv=uv, z=z, effort=np.array(E), compression=np.array(C), labels=lab)


# ============================================================================== S4 steps
SPINE_EXIT = 0.3                         # arc fraction pelvis->neck where leg chains leave the spine


def _loa_candidates(b: BodyView):
    head_end = 2.0 if b.ok("head") else 1.0
    cands = {}
    for s in "lr":
        o = "r" if s == "l" else "l"
        leg = lambda x: [x + "hip", x + "knee", x + "ank"]
        arm = ([s + "hand"] if b.ok(s + "hand") else []) + [s + "wri", s + "elb", s + "sho"]
        # the line leaves the spine at ~L1 and runs into the leg: going via the pelvis centre
        # would add a sideways hip jog that reads as a false S-curve
        cands[f"head->{s.upper()} leg"] = ([("spine", head_end, SPINE_EXIT)] + leg(s), s + "leg")
        cands[f"{s.upper()} hand->{o.upper()} leg"] = (arm + [("spine", 0.9, SPINE_EXIT)] + leg(o), o + "leg")
        cands[f"{s.upper()} hand->{s.upper()} leg"] = (arm + [("spine", 0.9, SPINE_EXIT)] + leg(s), s + "leg")
    cands["spine"] = ([("spine", head_end, 0.0)], None)
    return cands


def line_of_action(b: BodyView, prm: S4Params):
    """Score every candidate chain; fit, exaggerate and return the winner as a stroke."""
    ref = b.body_px()
    rows, best = [], None
    for name, (parts, leg) in _loa_candidates(b).items():
        ch = b.chain(parts)
        if ch is None or len(ch["uv"]) < 3:
            continue
        P, (E, C, Z), L = resample(ch["uv"], 64, ch["effort"], ch["compression"], ch["z"])
        if L < 1e-6:
            continue
        ctrl, resid = fit_cubic_bezier(P)
        load = float(E.mean())
        sup = b.support.get(leg, 0.0) if leg else 0.0
        back = _backtrack(P)
        score = ((L / ref) * (1 + prm.load_w * load) * (1 + prm.support_w * sup)
                 / (1 + prm.resid_w * resid) / (1 + prm.backtrack_w * back))
        rows.append(dict(name=name, score=round(score, 4), length_px=round(L, 1), load=round(load, 3),
                         support=round(sup, 3), resid=round(resid, 4), backtrack=round(back, 3)))
        if best is None or score > best[0]:
            best = (score, name, ch, ctrl, resid, E, C, Z, sup, parts)
    if best is None:
        raise ValueError("S4: no valid line-of-action chain (too few joints)")
    score, name, ch, ctrl, resid, E, C, Z, sup, parts = best
    ctrl_x = exaggerate(ctrl, prm.exaggeration)
    pts = bezier_even(ctrl_x, 120)
    t = np.linspace(0, 1, 64)
    tt = np.linspace(0, 1, 120)
    st = Stroke("line_of_action", "A", "loa", pts, np.interp(tt, t, Z), np.interp(tt, t, E),
                np.interp(tt, t, C), support=sup, group="loa")
    # where each chain joint sits along the LOA (arc fraction), for downstream posing/UI
    ta = _arc_t(ch["uv"])
    joint_t = {j: round(float(ta[i]), 3) for i, j in enumerate(ch["labels"]) if j}
    shape = "S" if _is_s_curve(ctrl_x) else "C"
    st.meta = dict(chain=name, shape=shape, bow_px=round(bezier_offsets(ctrl_x), 1))
    info = dict(name=name, shape=shape, score=round(score, 4), resid=round(resid, 4),
                joints=[j for j in ch["labels"] if j], joint_t=joint_t,
                ctrl=np.round(ctrl, 2).tolist(), ctrl_exaggerated=np.round(ctrl_x, 2).tolist(),
                candidates=sorted(rows, key=lambda r: -r["score"]))
    return st, info


def _backtrack(P):
    """Fraction of the path that runs backwards along its own chord (a hanging arm folded onto
    the trunk). A single C or S line of action never doubles back."""
    c = _u(P[-1] - P[0])
    prog = np.diff(P, axis=0) @ c
    L = np.linalg.norm(np.diff(P, axis=0), axis=1).sum() + 1e-9
    return float(np.clip(-prog, 0, None).sum() / L)


def _is_s_curve(ctrl):
    """S if the curve's turning changes sign with a real amount of turn on both sides."""
    P = bezier(ctrl, 200)
    d = np.diff(P, axis=0)
    ang = np.diff(np.unwrap(np.arctan2(d[:, 1], d[:, 0])))       # signed turn per step
    pos, neg = ang[ang > 0].sum(), -ang[ang < 0].sum()
    return bool(min(pos, neg) > np.radians(8))


def _limb_frame(b: BodyView, chord3, side):
    """Anatomical fwd / lateral directions of a limb: the trunk's, carried by the minimal rotation
    that takes 'hanging down' onto the limb's chord."""
    R = _rodrigues(-b.up, chord3)
    lat = -b.right if side == "l" else b.right
    return R @ b.fwd, R @ lat


def rhythm_lines(b: BodyView, prm: S4Params, loa_joints):
    """One curve per limb. Bow = the limb's own bend (exaggerated) + a bias toward the side of the
    working / shortened muscles (S3), reduced on a limb whose muscles are passively stretched."""
    out = []
    k = prm.exaggeration * prm.limb_exaggeration
    for s in "lr":
        for kind, js in (("arm", [s + "sho", s + "elb", s + "wri", s + "hand"]),
                         ("leg", [s + "hip", s + "knee", s + "ank"])):
            valid = []
            for j in js:
                if not b.ok(j):
                    break
                valid.append(j)
            if len(valid) < 2:
                continue
            ch = b.chain(valid)
            P, (E, C, Z), L = resample(ch["uv"], 48, ch["effort"], ch["compression"], ch["z"])
            if L < 1e-6:
                continue
            ctrl, resid = fit_cubic_bezier(P) if len(valid) > 2 else (np.vstack([P[0], P[16], P[32], P[-1]]), 0.0)
            ctrl = exaggerate(ctrl, k)
            # muscle-side bias
            bias3 = np.zeros(3); passive = 0.0
            fwd, lat = _limb_frame(b, ch["P3"][-1] - ch["P3"][0], s)
            dirs = dict(fwd=fwd, lat=lat)
            for g, (axis, sg) in LIMB_MUSCLE_SIDES[kind].items():
                m = b.groups.get(s + g)
                if not m:
                    continue
                q = float(m.get("activation", 0.0)) + 0.5 * float(np.clip((1 - m.get("length", 1.0)) / 0.3, 0, 1))
                bias3 += sg * q * dirs[axis]
                passive = max(passive, float(m.get("passive", 0.0)))
            mid3 = 0.5 * (ch["P3"][0] + ch["P3"][-1])
            n2 = _normal2(P[-1] - P[0])
            vis = float(np.clip(b.project_dir(mid3, bias3) @ n2, -1, 1)) if np.linalg.norm(bias3) > 1e-9 else 0.0
            bow = prm.limb_bias * L * vis * (1 + prm.exaggeration)
            ctrl[1:3] += n2 * bow / 0.75                       # cubic midpoint moves 0.75 x the offset
            straight = float(np.clip(2 * passive, 0, 1)) * 0.8
            ctrl = straighten(ctrl, straight)
            on_loa = sum(j in loa_joints for j in valid) >= 2
            pts = bezier_even(ctrl, 48)
            name = f"{s}_{kind}"
            st = Stroke(name, "B", "limb_on_loa" if on_loa else "limb", pts, Z, E, C,
                        support=b.support.get(s + kind, 0.0), group="loa" if on_loa else name)
            st.meta = dict(joints=valid, on_loa=on_loa, bias_bow_px=round(float(bow), 1),
                           straighten=round(straight, 2), fit_resid=round(float(resid), 4),
                           bow_px=round(bezier_offsets(ctrl), 1))
            out.append(st)
        # foot: heel (or ankle) -> ball of the foot
        a = s + "heel" if b.ok(s + "heel") else s + "ank"
        if b.ok(s + "ank") and b.ok(s + "bigtoe") and b.ok(a):
            ch = b.chain([a, s + "bigtoe"])
            P, (Z,), L = resample(ch["uv"], 16, ch["z"])
            if L > 1e-6:
                e = b.eff.get(s + "ank", 0.0)
                out.append(Stroke(f"{s}_foot", "B", "foot", P, Z, np.full(16, e), np.zeros(16),
                                  support=b.support.get(s + "leg", 0.0), group=f"{s}_leg"))
    return out


def torso_lines(b: BodyView, prm: S4Params):
    """Squash/stretch sides of the torso + shoulder and hip lines.

    The two torso contour lines drawn are the silhouette sides for this view: a blend of the
    left/right sides (frontal view) and the front/back sides (profile view). The squashed side is
    curved (bowed inward, following the spine's concavity) and heavy; the stretched side straight
    and light."""
    need = ("lsho", "rsho", "lhip", "rhip", "neck", "pelvis")
    if not all(b.ok(j) for j in need):
        return [], {}
    H = b.H
    hs = 0.5 * (b.P("rsho") - b.P("lsho")); hh = 0.5 * (b.P("rhip") - b.P("lhip"))
    neck, pel = b.P("neck"), b.P("pelvis")
    # lateral squash from the side lengths
    Ll = np.linalg.norm(b.P("lsho") - b.P("lhip")); Lr = np.linalg.norm(b.P("rsho") - b.P("rhip"))
    lat_d = (Lr - Ll) / (0.5 * (Ll + Lr) + 1e-9)                 # > 0: left side shorter
    comp = dict(l=float(np.clip(lat_d / 0.08, 0, 1)), r=float(np.clip(-lat_d / 0.08, 0, 1)), f=0.0, k=0.0)
    ab, bk = b.groups.get("abdominals"), b.groups.get("back_extensors")
    if ab and bk:
        sag_d = float(bk.get("length", 1)) - float(ab.get("length", 1))   # > 0: front shorter
        comp["f"], comp["k"] = float(np.clip(sag_d / 0.1, 0, 1)), float(np.clip(-sag_d / 0.1, 0, 1))
    # silhouette direction e (in the torso's horizontal plane, across the line of sight)
    cen = 0.5 * (neck + pel)
    if b.mode == "3d":
        e = np.cross(b.up, _u(cen)); e = _u(e - (e @ b.up) * b.up)
    else:
        e = b.right
    ce, se = float(e @ b.right), float(e @ b.fwd)
    if ce < 0:
        ce, se = -ce, -se                                         # +e side = the person's right-ish
    dt, db = 0.055 * H, 0.05 * H                                  # chest / pelvis half depth
    th_t = np.arctan2(dt * se, np.linalg.norm(hs) * ce)
    th_b = np.arctan2(db * se, np.linalg.norm(hh) * ce)
    spine_uv = b.Suv[:N_TRUNK]
    dev = spine_uv[N_TRUNK // 2] - 0.5 * (spine_uv[0] + spine_uv[-1])  # spine bow in the view
    cen_uv, _ = b.proj(cen[None])
    strokes, sides = [], {}
    wc, ws = ce * ce, se * se
    for sgn, lab in ((+1, "r"), (-1, "l")):
        top = neck + sgn * (np.cos(th_t) * hs + np.sin(th_t) * dt * b.fwd)
        bot = pel + sgn * (np.cos(th_b) * hh + np.sin(th_b) * db * b.fwd)
        sag = "f" if sgn * se > 0 else "k"
        c_side = wc * comp[lab] + ws * comp[sag]
        sides[lab] = dict(top=top, bot=bot, comp=c_side, sag=sag)
    for lab, o in (("r", "l"), ("l", "r")):
        sd, od = sides[lab], sides[o]
        uv, z = b.proj(np.stack([sd["top"], sd["bot"]]))
        L = np.linalg.norm(uv[1] - uv[0])
        n_in = _normal2(uv[1] - uv[0])
        if (cen_uv[0] - 0.5 * (uv[0] + uv[1])) @ n_in < 0:
            n_in = -n_in
        rel = sd["comp"] - od["comp"]
        follow = max(0.0, float(dev @ n_in)) * (1 + prm.exaggeration)      # spine concavity
        bow_len = follow + prm.torso_bow * L * max(rel, 0.0)
        bow_len *= 1 - float(np.clip(-rel * 3, 0, 1))                       # stretched -> straight
        pts = quad_curve(uv[0], uv[1], n_in * bow_len, 40)
        tt = np.linspace(0, 1, 40)
        eff = np.array([b.spine_effort(1 - t) for t in tt])
        squashed = rel > 0.05
        role = "torso_comp" if squashed else "torso_str"
        sup = wc * b.support.get(lab + "leg", 0.0)
        st = Stroke(f"torso_{lab}", "B", role, pts, np.interp(tt, [0, 1], z), eff,
                    np.full(40, max(sd["comp"], 0.0)), support=sup, group="torso")
        st.meta = dict(squashed=bool(squashed), compression=round(sd["comp"], 3), bow_px=round(bow_len, 1),
                       anatomical_side="right" if lab == "r" else "left",
                       mix=dict(lateral=round(wc, 2), sagittal=round(ws, 2), sagittal_side=sd["sag"]))
        strokes.append(st)
    # shoulder and hip lines (contrapposto)
    tilt = {}
    for nm, a, c in (("shoulder_line", "lsho", "rsho"), ("hip_line", "lhip", "rhip")):
        ch = b.chain([a, c])
        P, (Z,), L = resample(ch["uv"], 24, ch["z"])
        e = 0.5 * (b.eff.get(a, 0) + b.eff.get(c, 0))
        strokes.append(Stroke(nm, "B", "girdle", P, Z, np.full(24, e), np.zeros(24), group="torso"))
        d = ch["uv"][1] - ch["uv"][0]
        if d[0] < 0:
            d = -d
        seen = np.linalg.norm(d) / (np.linalg.norm(ch["P3"][1] - ch["P3"][0]) * b.pxpm(ch["P3"].mean(0)) + 1e-9)
        # + = rises to the image right; None when the line points into the view (no readable tilt)
        tilt[nm] = float(np.degrees(np.arctan2(-d[1], d[0]))) if seen > 0.35 else None
    squashed = [s.meta["anatomical_side"] for s in strokes[:2] if s.meta["squashed"]]
    ts, th = tilt["shoulder_line"], tilt["hip_line"]
    info = dict(shoulder_tilt_deg=None if ts is None else round(ts, 1),
                hip_tilt_deg=None if th is None else round(th, 1),
                contrapposto=bool(ts is not None and th is not None and ts * th < 0 and min(abs(ts), abs(th)) > 2),
                squashed_side=squashed[0] if squashed else None,
                lateral_squash=round(float(lat_d), 3), view_mix=dict(lateral=round(wc, 2), sagittal=round(ws, 2)))
    return strokes, info


def head_strokes(b: BodyView):
    if not (b.ok("neck") and b.ok("head")):
        return []
    H = b.H
    c3 = b.P("neck") + 0.62 * (b.P("head") - b.P("neck"))
    (cuv,), (cz,) = b.proj(c3[None])
    ppm = b.pxpm(c3)
    ax = b.proj(b.P("head")[None])[0][0] - b.proj(b.P("neck")[None])[0][0]
    if np.linalg.norm(ax) < 0.03 * H * ppm:
        ax = b.project_dir(c3, b.up)
        if np.linalg.norm(ax) < 1e-6:
            ax = np.array([0.0, -1.0])
    a, bb = 0.068 * H * ppm, 0.055 * H * ppm
    pts = ellipse(cuv, a, bb, ax, 72)
    e = b.eff.get("neck", 0.0)
    out = [Stroke("head", "B", "head", pts, np.full(72, cz), np.full(72, e), np.zeros(72), closed=True,
                  group="head")]
    if b.ok("nose"):
        (nuv,), (nz,) = b.proj(b.P("nose")[None])
        d = nuv - cuv
        if np.linalg.norm(d) > 0.12 * bb:                         # face turned: show its direction
            seg = np.linspace(cuv + 0.4 * d, nuv + 0.25 * d, 10)
            out.append(Stroke("face_dir", "B", "face", seg, np.linspace(cz, nz, 10), np.full(10, e),
                              np.zeros(10), group="head"))
    return out


def construction(b: BodyView):
    """Layer C: ribcage / pelvis ellipses and joint circles (2D approximations of the volumes)."""
    out, H = [], b.H
    if all(b.ok(j) for j in ("lsho", "rsho", "lhip", "rhip", "neck", "pelvis")):
        i_rib = int(round(0.62 * (N_TRUNK - 1)))
        for nm, c3, half, wpx in (
                ("ribcage", b.S[i_rib], 0.11 * H,
                 max(0.85 * np.linalg.norm(b.uv[IDX["rsho"]] - b.uv[IDX["lsho"]]) / 2, 0.06 * H * b.pxpm(b.S[i_rib]))),
                ("pelvis_vol", b.P("pelvis") + 0.02 * H * b.up, 0.06 * H,
                 max(1.5 * np.linalg.norm(b.uv[IDX["rhip"]] - b.uv[IDX["lhip"]]) / 2, 0.055 * H * b.pxpm(b.P("pelvis"))))):
            (cuv,), (cz,) = b.proj(c3[None])
            ends, _ = b.proj(np.stack([c3 - half * b.up, c3 + half * b.up]))
            axis = ends[1] - ends[0]
            a = max(np.linalg.norm(axis) / 2, 0.035 * H * b.pxpm(c3))
            if np.linalg.norm(axis) < 1e-6:
                axis = np.array([0.0, -1.0])
            pts = ellipse(cuv, a, wpx, axis, 64)
            out.append(Stroke(nm, "C", "construction", pts, np.full(64, cz), np.zeros(64), np.zeros(64),
                              closed=True, group="construction"))
    for j in ("lsho", "rsho", "lelb", "relb", "lwri", "rwri", "lhip", "rhip", "lknee", "rknee", "lank", "rank"):
        if b.ok(j):
            (cuv,), (cz,) = b.proj(b.P(j)[None])
            r = 0.022 * H * b.pxpm(b.P(j))
            out.append(Stroke(f"joint_{j}", "C", "construction", ellipse(cuv, r, r, [1, 0], 24),
                              np.full(24, cz), np.zeros(24), np.zeros(24), closed=True, group="construction"))
    return out


# ============================================================================== weights
def _taper(t, closed, prm):
    if closed:
        return np.ones_like(t)
    return prm.taper_min + (1 - prm.taper_min) * np.sin(np.pi * t) ** prm.taper_pow


def _gauss_smooth(v, sigma, closed=False):
    if sigma < 0.5 or len(v) < 3:
        return v
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    pad = np.r_[v[-r:], v, v[:r]] if closed and r < len(v) else np.pad(v, r, mode="edge")
    return np.convolve(pad, k, mode="valid")


def _smooth(m, k=5):
    if len(m) < k:
        return m
    return np.clip(np.convolve(np.pad(m, k // 2, mode="edge"), np.ones(k) / k, mode="valid"), 0, 1)


def assign_weights(strokes, b: BodyView, prm: S4Params, S: float):
    base = []
    for st in strokes:
        t = st.t
        if st.layer == "C":
            st.width = np.full(len(st.pts), prm.w0["construction"] * S)
            base.append(st.width); continue
        f = (1 + prm.a_effort * st.effort + prm.b_compression * st.compression
             + prm.c_near * b.near(st.depth) + prm.e_support * st.support)
        f = _gauss_smooth(f, prm.weight_smooth * len(f), st.closed)     # one swell, not joint-by-joint lumps
        base.append(prm.w0[st.role] * S * f * _taper(t, st.closed, prm))
    # overlap / occlusion junctions: body contours only (layer B; the LOA is an abstract line and
    # neither occludes nor is occluded). Needs depth.
    ink = [i for i, st in enumerate(strokes) if st.layer == "B"]
    for st in strokes:
        st.front = np.zeros(len(st.pts)); st.behind = np.zeros(len(st.pts))
    for i in ink:
        si = strokes[i]
        si.front = np.zeros(len(si.pts)); si.behind = np.zeros(len(si.pts))
    if b.mode == "3d":
        ti = {i: strokes[i].t for i in ink}
        for i in ink:
            si = strokes[i]
            fr = np.zeros(len(si.pts), bool); bh = np.zeros(len(si.pts), bool)
            end_i = (ti[i] < 0.08) | (ti[i] > 0.92) if not si.closed else np.zeros(len(si.pts), bool)
            for j in ink:
                sj = strokes[j]
                if j == i or (si.group and si.group == sj.group):
                    continue
                d = np.linalg.norm(si.pts[:, None] - sj.pts[None], axis=-1)
                r = 0.5 * (base[i][:, None] + base[j][None]) + 1.5 * S
                end_j = (ti[j] < 0.08) | (ti[j] > 0.92) if not sj.closed else np.zeros(len(sj.pts), bool)
                near = (d < r) & ~(end_i[:, None] & end_j[None])
                dz = si.depth[:, None] - sj.depth[None]
                fr |= (near & (dz < -prm.depth_margin_m)).any(1)
                bh |= (near & (dz > prm.depth_margin_m)).any(1)
            sg = 0.04 * len(si.pts)
            si.front = np.clip(_gauss_smooth(_smooth(fr.astype(float)), sg, si.closed) * 1.5, 0, 1)
            si.behind = np.clip(_gauss_smooth(_smooth(bh.astype(float)), sg, si.closed) * 1.5, 0, 1)
    for i, st in enumerate(strokes):
        if st.layer == "C":
            continue
        st.width = base[i] * (1 + prm.d_overlap * st.front) * (1 - (1 - prm.lost) * st.behind)


# ============================================================================== entry point
def build_croquis(frame, view: View | None = None, params: S4Params | None = None,
                  layers: str = "AB", exaggeration: float | None = None) -> Croquis:
    """PoseFrame -> Croquis (strokes in view px with per-sample widths)."""
    view = view or View()
    prm = params or S4Params()
    if exaggeration is not None:
        prm.exaggeration = float(exaggeration)
    tm, t0 = {}, time.perf_counter()

    def lap(k):
        nonlocal t0
        t = time.perf_counter(); tm[k] = round(1e3 * (t - t0), 2); t0 = t

    b = BodyView(frame, view); lap("view")
    S = b.unit()
    loa, loa_info = line_of_action(b, prm); lap("loa")
    strokes = [loa]
    limbs = rhythm_lines(b, prm, set(loa_info["joints"])); lap("limbs")
    torso, torso_info = torso_lines(b, prm); lap("torso")
    strokes += torso + limbs + head_strokes(b); lap("head")
    if "C" in layers:
        strokes += construction(b)
    lap("construction")
    strokes = [s for s in strokes if s.layer in layers]
    assign_weights(strokes, b, prm, S); lap("weights")
    tm["total"] = round(sum(tm.values()), 2)
    s3 = frame.s3_info or {}
    info = dict(mode=b.mode, unit_px=round(S, 3), support={k: round(v, 3) for k, v in b.support.items()},
                com_offset_m=s3.get("com_offset_m"), balanced=s3.get("balanced"),
                airborne=not bool(frame.contacts) if b.mode == "3d" else None,
                torso=torso_info, layers=layers, timings_ms=tm, exaggeration=prm.exaggeration,
                overlaps={s.name: dict(front=round(float(s.front.mean()), 2), behind=round(float(s.behind.mean()), 2))
                          for s in strokes if s.front is not None and (s.front.any() or s.behind.any())})
    return Croquis(tuple(frame.img_hw), view, float(S), strokes, loa_info, info)


def summary(c: Croquis) -> str:
    t = c.info["torso"]
    top = c.loa["candidates"][:3]
    deg = lambda v: "n/a" if v is None else "%+.0f°" % v
    return ("  S4%s: LOA %s (%s-curve, score %.2f; next %s) | squashed torso side %s | shoulder %s hip %s%s"
            % (" view " + c.view.tag if c.view.tag else "", c.loa["name"], c.loa["shape"], c.loa["score"],
               ", ".join("%s %.2f" % (r["name"], r["score"]) for r in top[1:]) or "-",
               t.get("squashed_side") or "none", deg(t.get("shoulder_tilt_deg")), deg(t.get("hip_tilt_deg")),
               " (contrapposto)" if t.get("contrapposto") else ""))


# ============================================================================== preview (MVP S5)
INK = {"A": (25, 25, 25), "B": (35, 35, 35), "C": (200, 170, 140)}   # BGR
LOA_COLOR = (40, 40, 220)            # BGR red: the line of action stands out from the rest


def _draw_stroke(img, pts, w, color, closed):
    """Variable-width stroke as quads + round joins (handles closed and self-touching strokes)."""
    import cv2
    P = np.asarray(pts, float)
    if closed:
        P = np.vstack([P, P[:1]]); w = np.r_[w, w[:1]]
    tang = np.gradient(P, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1) * (np.asarray(w)[:, None] / 2)
    for k in range(len(P) - 1):
        q = np.array([P[k] + nrm[k], P[k + 1] + nrm[k + 1], P[k + 1] - nrm[k + 1], P[k] - nrm[k]])
        cv2.fillConvexPoly(img, np.round(q * 16).astype(np.int32), color, cv2.LINE_AA, shift=4)
    for k in range(len(P)):
        r = w[k] / 2
        if r > 0.6:
            cv2.circle(img, tuple(np.round(P[k] * 16).astype(int)), int(round(r * 16)), color, -1,
                       cv2.LINE_AA, shift=4)


def render_preview(c: Croquis, size=None, bg=None, fit=True, margin=0.08, supersample=2,
                   layers=None, color=None, loa_color=LOA_COLOR):
    """Rasterise the strokes. size=(H,W): output canvas (default: the view canvas). fit=True scales
    the drawing into the canvas; fit=False keeps view px (overlay on the photo, pass it as bg).
    color recolours the other strokes; the line of action (layer A) is drawn in loa_color
    (red by default; loa_color=None gives it the same colour as the rest)."""
    import cv2
    layers = layers or "ABC"
    H, W = size or c.canvas_hw
    ss = supersample
    strokes = [s for s in c.strokes if s.layer in layers]
    if fit:
        allp = np.vstack([s.pts for s in strokes])
        lo, hi = allp.min(0), allp.max(0)
        span = np.maximum(hi - lo, 1e-6)
        sc = min((W * (1 - 2 * margin)) / span[0], (H * (1 - 2 * margin)) / span[1])
        off = np.array([W, H]) / 2 - sc * (lo + hi) / 2
    else:
        sc, off = 1.0, np.zeros(2)
    if bg is None:
        canvas = np.full((H * ss, W * ss, 3), 250, np.uint8)
    else:
        canvas = cv2.resize(bg, (W * ss, H * ss), interpolation=cv2.INTER_LINEAR)
    order = sorted(strokes, key=lambda s: {"C": 0, "B": 1, "A": 2}[s.layer])
    for s in order:
        col = (loa_color or color or INK["A"]) if s.layer == "A" else (color or INK[s.layer])
        _draw_stroke(canvas, (s.pts * sc + off) * ss, np.maximum(s.width * sc * ss, 0.0), col, s.closed)
    return cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA)


def preview_panel(c: Croquis, photo=None, height=900, layers=None):
    """Debug panel: [faded photo + strokes in red] | [clean croquis]. Photo only for the photo view."""
    import cv2
    h0, w0 = c.canvas_hw
    W = int(round(height * w0 / h0))
    clean = render_preview(c, (height, max(W, int(height * 0.75))), layers=layers)
    tiles = []
    if photo is not None and c.view.is_photo:
        faded = (photo.astype(np.float32) * 0.45 + 140).clip(0, 255).astype(np.uint8)
        ov = render_preview(c, (h0, w0), bg=faded, fit=False, color=(150, 80, 20), layers=layers)   # blue + red LOA
        tiles.append(cv2.resize(ov, (W, height), interpolation=cv2.INTER_AREA))
    tiles.append(clean)
    out = np.hstack(tiles)
    txt = f"LOA: {c.loa['name']} ({c.loa['shape']})" + (f"  view {c.view.tag}" if c.view.tag else "")
    cv2.putText(out, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return out

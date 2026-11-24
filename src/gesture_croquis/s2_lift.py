"""
S2 (M1-lite): 2D keypoints -> metric 3D skeleton in the camera frame.

All 3D comes from one transparent least-squares fit of the canonical skeleton:
    reprojection (perspective camera)    — match S1 keypoints, confidence-weighted
  + bone lengths  (anthropometric × body height, per-bone tolerance)
  + left/right symmetry
  + anatomical hinge priors (knees and ankles only bend one way, nose in front of the neck)
  + [optional] per-joint relative depth from RTMW3D (RTMPose-family whole-body 3D model)

Backends:
  "rtmw3d"    : depth cue from RTMW3D-x (rtmlib.RTMPose3d), sign auto-checked.
  "geometric" : no extra network. Taylor (2000) scaled-orthographic lifting, depth-sign
                ambiguity resolved by enumerating limb flips against the hinge priors.

A heavier body-model backend (HSMR/SKEL, SAM 3D Body) slots in later by producing the
same PoseFrame fields; this fit then becomes its initialiser / sanity check.
"""
from __future__ import annotations
import itertools
import time
import numpy as np
from scipy.optimize import least_squares

from .skeleton import (JOINTS, J, IDX, BONES, BONE_IDX, BONE_FRAC, BONE_SIG, SYM_PAIRS,
                       CONVERTERS)

CONF_MIN = 0.05          # below this a joint is treated as missing


# ----------------------------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------------------------
def default_K(img_hw, fov_deg: float = 55.0, focal_35mm: float | None = None):
    """Pinhole intrinsics. Uses the EXIF 35mm-equivalent focal length if available,
    else a typical phone/standard-lens horizontal FOV."""
    h, w = img_hw
    if focal_35mm:
        f = focal_35mm / 36.0 * max(w, h)
    else:
        f = 0.5 * max(w, h) / np.tan(np.radians(fov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)


def exif_focal_35mm(path: str) -> float | None:
    try:
        from PIL import Image
        ex = Image.open(path).getexif()
        v = ex.get_ifd(0x8769).get(0xA405) or ex.get(0xA405)   # FocalLengthIn35mmFilm
        return float(v) if v else None
    except Exception:
        return None


def project(X, K):
    z = np.maximum(X[:, 2:3], 1e-3)
    return (X[:, :2] / z) * K[[0, 1], [0, 1]] + K[:2, 2]


def backproject(uv, Z, K):
    return np.c_[(uv - K[:2, 2]) / K[[0, 1], [0, 1]] * Z[:, None], Z]


# ----------------------------------------------------------------------------------------
# Body frame + hinge priors (camera frame: x right, y down, z forward)
# ----------------------------------------------------------------------------------------
def _cross(a, b):
    """3-vector cross product without np.cross's per-call overhead (~30x faster for single vectors)."""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def _unit(v):
    if v.ndim == 1:
        return v / (np.sqrt(v @ v) + 1e-9)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def body_frame(X, part: str = "both"):
    """(up, right, forward) unit vectors. part = 'pelvis' (hip line; use for legs),
    'thorax' (shoulder line; use for arms/head) or 'both' (average; whole-body facing)."""
    i = IDX
    up = _unit(X[i["neck"]] - X[i["pelvis"]])
    rp, rt = X[i["rhip"]] - X[i["lhip"]], X[i["rsho"]] - X[i["lsho"]]
    r = {"pelvis": rp, "thorax": rt}.get(part, _unit(rp) + _unit(rt))
    r = _unit(r - (r @ up) * up)
    fwd = _unit(_cross(up, r))        # toward where the chest / pelvis faces
    return up, r, fwd


def hinge_terms(X):
    """Signed values that should be >= 0 for an anatomically plausible pose (metres-ish).
    knee:  (thigh x shank) . (-right)  >= 0   — flexion axis ~ pelvis lateral axis
    ankle: (shank x foot)  . right     >= 0   — toes point anterior to the shank
    head:  (nose - neck)   . fwd       >= 0
    """
    i = IDX
    up, r, _ = body_frame(X, "pelvis")
    _, _, fwd = body_frame(X, "thorax")
    out = []
    for s in "lr":
        hip, knee, ank, toe = (X[i[s + n]] for n in ("hip", "knee", "ank", "bigtoe"))
        thigh, shank, foot = knee - hip, ank - knee, toe - ank
        out.append(-_cross(thigh, shank) @ r / (np.linalg.norm(thigh) + 1e-6))
        out.append(_cross(shank, foot) @ r / (np.linalg.norm(shank) + 1e-6))
    out.append((X[i["nose"]] - X[i["neck"]]) @ fwd)
    return np.array(out)       # [l_knee, l_ankle, r_knee, r_ankle, head]


def _sp(x):
    """softplus barrier: ~0 for x << 0, ~x for x >> 0."""
    return np.logaddexp(0, x)


def _ang(u, v):
    c = (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
    return np.degrees(np.arccos(np.clip(c, -0.999999, 0.999999)))


def pose_prior_terms(X):
    """Soft anatomical-range residuals (dimensionless). Keeps the fit out of impossible or
    very unusual configurations when the depth cue is weak or noisy (telephoto, extremities).
    Ranges are deliberately generous: they bound, they do not pull toward a mean pose."""
    i = IDX
    up, r, fwd = body_frame(X, "thorax")
    _, _, fwd_p = body_frame(X, "pelvis")
    res = []
    for s in "lr":
        sho, elb, wri = X[i[s + "sho"]], X[i[s + "elb"]], X[i[s + "wri"]]
        hip, knee, ank = X[i[s + "hip"]], X[i[s + "knee"]], X[i[s + "ank"]]
        elbow_flex = 180 - _ang(sho - elb, wri - elb)
        knee_flex = 180 - _ang(hip - knee, ank - knee)
        res += [3 * _sp((elbow_flex - 140) / 5), 3 * _sp((knee_flex - 155) / 5)]   # anatomical max ~145-150°
        # hyperextension behind the trunk, measured in the sagittal plane
        thigh, uarm = knee - hip, elb - sho
        hip_ext = np.degrees(np.arctan2(-(thigh @ fwd_p), -(thigh @ up)))    # + = leg behind pelvis
        sho_ext = np.degrees(np.arctan2(-(uarm @ fwd), -(uarm @ up)))
        res += [_sp((hip_ext - 30) / 8), _sp((sho_ext - 60) / 10)]
    # trunk: twist and side bend between hip line and shoulder line, about the trunk axis
    rp, rt = X[i["rhip"]] - X[i["lhip"]], X[i["rsho"]] - X[i["lsho"]]
    rp_, rt_ = rp - (rp @ up) * up, rt - (rt @ up) * up
    twist = np.degrees(np.arctan2(_cross(rp_, rt_) @ up, rp_ @ rt_))
    rpf, rtf = rp - (rp @ fwd) * fwd, rt - (rt @ fwd) * fwd
    bend = np.degrees(np.arctan2(_cross(rpf, rtf) @ fwd, rpf @ rtf))
    # (walking while turning the upper body, e.g. to wave, needs ~40-50°; axial max ~60°)
    res += [twist / 45, _sp((abs(twist) - 60) / 6), bend / 40, _sp((abs(bend) - 45) / 6)]
    # head: ears behind the eyes, nose and crown above the neck (in the trunk frame)
    eyes, ears = (X[i["leye"]] + X[i["reye"]]) / 2, (X[i["lear"]] + X[i["rear"]]) / 2
    nose, neck, head = X[i["nose"]], X[i["neck"]], X[i["head"]]
    hv = head - neck
    res += [2 * _sp((0.02 - (eyes - ears) @ fwd) / 0.01),
            2 * _sp((0.04 - (nose - neck) @ up) / 0.01),
            2 * _sp((0.5 - hv @ up / (np.linalg.norm(hv) + 1e-9)) / 0.08)]
    return np.array(res)


def approx_com(X):
    """Cheap whole-body COM from joints (de Leva fractions lumped onto joint midpoints)."""
    i = IDX
    m = lambda a, b: (X[i[a]] + X[i[b]]) / 2
    parts = [(0.435, 0.6 * X[i["pelvis"]] + 0.4 * X[i["neck"]]), (0.07, X[i["head"]] * 0.5 + X[i["neck"]] * 0.5)]
    for s in "lr":
        parts += [(0.142, m(s + "hip", s + "knee")), (0.057, m(s + "knee", s + "ank")),
                  (0.027, m(s + "sho", s + "elb")), (0.022, m(s + "elb", s + "wri"))]
    w = sum(p[0] for p in parts)
    return sum(p[0] * p[1] for p in parts) / w


def balance_term(X, gravity=np.array([0.0, 1.0, 0.0]), margin=0.12):
    """Horizontal distance (⊥ gravity) of the COM from the ankle-to-ankle support segment,
    beyond `margin`, as a softplus barrier (metres / 4 cm). Static stance assumption."""
    def horiz(v):
        return v - (v @ gravity) * gravity
    c = horiz(approx_com(X))
    a, b = horiz(X[IDX["lank"]]), horiz(X[IDX["rank"]])
    ab = b - a
    t = np.clip((c - a) @ ab / (ab @ ab + 1e-9), 0, 1)
    d = np.linalg.norm(c - (a + t * ab))
    return _sp((d - margin) / 0.04)


# kinematic edges for the depth cue, with trust weights (RTMW3D extremities are unreliable)
_LOW_TRUST = {"lheel", "rheel", "lbigtoe", "rbigtoe", "lsmalltoe", "rsmalltoe", "lhand", "rhand",
              "leye", "reye", "lear", "rear", "nose", "head"}


HINGE_NEEDS = [("lhip", "lknee", "lank"), ("lknee", "lank", "lbigtoe"),
               ("rhip", "rknee", "rank"), ("rknee", "rank", "rbigtoe"), ("nose", "neck")]


# ----------------------------------------------------------------------------------------
# Least-squares fit
# ----------------------------------------------------------------------------------------
# per-person proportion groups: bones whose length scales together
_GROUP_OF = {}
for _a, _b, *_ in BONES:
    _k = {_a, _b}
    if _k & {"lknee", "rknee"} or {"lank", "rank"} & _k and _k & {"lknee", "rknee"}:
        _GROUP_OF[(_a, _b)] = 0                                    # legs
    elif _k & {"lelb", "relb"}:
        _GROUP_OF[(_a, _b)] = 1                                    # arms
    elif _k <= {"lsho", "rsho", "lhip", "rhip", "neck", "pelvis"}:
        _GROUP_OF[(_a, _b)] = 2                                    # torso box
    else:
        _GROUP_OF[(_a, _b)] = 3                                    # head, hands, feet: global only
BONE_GROUP = np.array([_GROUP_OF[(a, b)] for a, b, *_ in BONES])
GROUP_NAMES = ["legs", "arms", "torso"]


def fit_skeleton(kpts2d, conf, K, height=1.70, X0=None, z_rel=None, z_conf=None,
                 w_hinge=1.0, w_depth=1.0, w_prior=1.0, iters=200, tol=1e-4,
                 height_sigma=0.07, prop_sigma=0.10, w_balance=1.0, physics=None, physics_p0=None,
                 order_cons=None, w_order=1.0, jac="auto", solver="auto", _probe=False):
    """Return (X (J,3) camera-frame metres, valid (J,), info).
    physics: optional biomech.Physics block — adds its residuals (passive muscle, gravity effort,
    balance, floor) and its parameters (camera pitch) to the same least-squares problem.
    jac: "auto" = exact Jacobian via JAX (s2_jax.py) when jax is installed, else finite
    differences; "jax" = require JAX; "fd" = SciPy finite differences (the original path).
    solver: "auto"/"lm" = Levenberg-Marquardt on the exact Jacobian (s2_jax.lm_solve, needs JAX);
    "scipy" = scipy.optimize.least_squares (trf). Same robust objective either way."""
    from .plausibility import twist_terms, collision_terms
    from .occlusion import order_terms
    valid = conf > CONF_MIN
    # a joint the fit cannot see at all is dropped (and bones touching it)
    bmask = valid[BONE_IDX[:, 0]] & valid[BONE_IDX[:, 1]]
    bi, bL, bS = BONE_IDX[bmask], BONE_FRAC[bmask] * height, BONE_SIG[bmask]
    bg = BONE_GROUP[bmask]
    sym = np.array([(a, b) for a, b in SYM_PAIRS if bmask[a] and bmask[b]]).reshape(-1, 2)
    remap = -np.ones(len(BONES), int); remap[np.where(bmask)[0]] = np.arange(bmask.sum())
    sym = remap[sym] if len(sym) else sym
    hinge_ok = np.array([all(valid[IDX[n]] for n in need) for need in HINGE_NEEDS])

    vi = np.where(valid)[0]
    ext = np.ptp(kpts2d[vi], axis=0).max()
    sig_px = max(2.0, 0.012 * ext)                  # ~1% of the person's pixel extent
    wr = np.sqrt(conf[vi])[:, None]
    use_depth = z_rel is not None
    if use_depth:
        # depth cue as per-bone depth DIFFERENCES along the kinematic tree: a global or
        # per-limb bias in the network's root-relative depth cancels out
        zcf = np.clip(z_conf if z_conf is not None else conf, 0, 1)
        E = np.array([(IDX[a], IDX[b]) for a, b in _TREE if valid[IDX[a]] and valid[IDX[b]]])
        ew = np.array([np.sqrt(min(zcf[a], zcf[b])) * (0.35 if JOINTS[b] in _LOW_TRUST else 1.0)
                       for a, b in E])
        ezr = z_rel[E[:, 1]] - z_rel[E[:, 0]]

    nX = 3 * len(vi)
    n_base = nX + 4 + (1 if use_depth else 0)
    # joints the fit can't see ride along with a fitted neighbour (keeps mass points finite)
    park = []
    for j in np.where(~valid)[0]:
        nb = [b if a == j else a for a, b in BONE_IDX if j in (a, b) and valid[b if a == j else a]]
        park.append((j, nb[0] if nb else IDX["pelvis"]))

    def unpack(p):
        X = np.zeros((J, 3)); X[vi] = p[:nX].reshape(-1, 3)
        for j, nb in park:
            X[j] = X[nb]
        log_h, g = p[nX], np.r_[p[nX + 1:nX + 4], 0.0]       # global height, group scales
        return X, (p[nX + 4] if use_depth else 0.0), log_h, g

    def residual_groups(p):
        X, log_sz, log_h, g = unpack(p)
        res = []
        res.append(('reprojection', ((project(X[vi], K) - kpts2d[vi]) / sig_px * wr).ravel()))
        L = np.linalg.norm(X[bi[:, 0]] - X[bi[:, 1]], axis=1)
        Lt = bL * np.exp(log_h + g[bg])
        res.append(('bones', (L - Lt) / (bS * Lt)))
        res.append(('proportions', np.r_[log_h / height_sigma, g[:3] / prop_sigma]))
        if len(sym):
            res.append(('symmetry', (L[sym[:, 0]] - L[sym[:, 1]]) / (0.04 * Lt[sym[:, 0]])))
        if w_hinge and hinge_ok.any():
            h = hinge_terms(X)[hinge_ok]
            res.append(('hinges', w_hinge * np.logaddexp(0, -h / 0.02) * 0.02 / 0.01))  # softplus barrier
        if w_prior:
            res.append(('anatomy', w_prior * pose_prior_terms(X)))
            res.append(('twist', w_prior * twist_terms(X)))
            res.append(('collision', w_prior * collision_terms(X, height, valid) / 0.03))
        if order_cons:
            res.append(('occlusion', w_order * order_terms(X, order_cons)))
        if physics is not None:
            res.append(('physics', physics.residuals(X, p[n_base:])))
        elif w_balance and valid[IDX["lank"]] and valid[IDX["rank"]]:
            res.append(('balance', np.array([2.0 * w_balance * balance_term(X)])))
        if use_depth:
            dz = X[E[:, 1], 2] - X[E[:, 0], 2]
            r = (dz - np.exp(log_sz) * ezr) / 0.07
            # Cauchy-style: quadratic near 0, saturating for outliers (a 0.6 m-wrong toe or a
            # biased limb must not drag the pose or shrink the global depth scale)
            res.append(('depth', w_depth * ew * np.sign(r) * np.sqrt(np.log1p(r * r))))
            res.append(('depth_scale', np.array([log_sz / 0.3])))                        # scale prior (≈ metres)
        return res

    def residuals(p):
        return np.concatenate([r for _, r in residual_groups(p)])

    p0 = np.r_[X0[vi].ravel(), np.zeros(4)]
    if use_depth:
        p0 = np.r_[p0, 0.0]
    if physics is not None:
        p0 = np.r_[p0, physics_p0 if physics_p0 is not None else np.zeros(physics.n_params)]
    fun, jac_fn, jac_used, spec = residuals, "2-point", "fd", None
    if jac in ("auto", "jax"):
        from . import s2_jax
        if s2_jax.available():
            spec = s2_jax.build_spec(
                kpts2d, conf, K, height, valid, park, sig_px, use_depth,
                E=E if use_depth else None, ew=ew if use_depth else None, ezr=ezr if use_depth else None,
                w_hinge=w_hinge, w_depth=w_depth, w_prior=w_prior, height_sigma=height_sigma,
                prop_sigma=prop_sigma, w_balance=w_balance, physics=physics, order_cons=order_cons,
                w_order=w_order if order_cons else 0.0, hinge_ok=hinge_ok if w_hinge else np.zeros(5, bool))
            cols = [3 * j + k for j in vi for k in range(3)] + [3 * J + k for k in range(4)]
            if use_depth:
                cols.append(3 * J + 4)
            if physics is not None:
                cols.append(3 * J + 5)
            fun, jac_fn = s2_jax.make_fun_jac(spec, cols, np.zeros(s2_jax.NP))
            jac_used = "jax"
        elif jac == "jax":
            raise ImportError("fit_skeleton(jac='jax') needs `pip install jax`")
    if _probe:                                         # for tests: the problem, not the solution
        return dict(residuals=residuals, p0=p0, fun=fun, jac=jac_fn, jac_used=jac_used, spec=spec,
                    cols=cols if spec is not None else None)
    t0 = time.perf_counter()
    if jac_used == "jax" and solver in ("auto", "lm"):
        from types import SimpleNamespace
        px, cost, nfev, njev, status = s2_jax.lm_solve(fun, jac_fn, p0, C=3.0, max_nfev=iters,
                                                       ftol=tol, xtol=1e-7, gtol=tol)
        sol = SimpleNamespace(x=px, cost=cost, nfev=nfev, njev=njev, status=status)
        jac_used = "jax-lm"
    else:
        sol = least_squares(fun, p0, jac=jac_fn, loss="soft_l1", f_scale=3.0, max_nfev=iters,
                            x_scale="jac", method="trf", ftol=tol, xtol=1e-7, gtol=tol)   # xtol is relative to |x| (metres at 2-20 m): keep it tight
    X, log_sz, log_h, g = unpack(sol.x)
    reproj = np.linalg.norm(project(X[vi], K) - kpts2d[vi], axis=1)
    Lf = np.linalg.norm(X[bi[:, 0]] - X[bi[:, 1]], axis=1)
    bL = bL * np.exp(log_h + g[bg])
    info = dict(cost=float(sol.cost), nfev=int(sol.nfev), njev=int(sol.njev or 0), jac=jac_used,
                fit_ms=1e3 * (time.perf_counter() - t0),
                reproj_px_median=float(np.median(reproj)), reproj_px_max=float(reproj.max()),
                bone_err_pct_median=float(np.median(np.abs(Lf - bL) / bL) * 100),
                hinge=dict(zip(["l_knee", "l_ankle", "r_knee", "r_ankle", "head"],
                               map(float, hinge_terms(X)))),
                depth_scale=float(np.exp(log_sz)) if use_depth else None,
                prior_terms_max=float(pose_prior_terms(X).max()),
                balance_term=float(balance_term(X)),
                fitted_height_m=float(height * np.exp(log_h)),
                proportions={n: float(np.exp(g[k])) for k, n in enumerate(GROUP_NAMES)},
                subject_distance_m=float(X[IDX["pelvis"], 2]))
    groups = {}
    for name, r in residual_groups(sol.x):            # cost per term, same robust loss as the fit
        z2 = (np.asarray(r) / 3.0) ** 2
        groups[name] = groups.get(name, 0.0) + float(0.5 * 9.0 * (2 * (np.sqrt(1 + z2) - 1)).sum())
    info["cost_groups"] = {k: round(v, 3) for k, v in groups.items()}
    if physics is not None:
        info["physics_p"] = sol.x[n_base:].tolist()
        info["physics"] = physics.breakdown(X, sol.x[n_base:])
    return X, valid, info


# ----------------------------------------------------------------------------------------
# Initialisation
# ----------------------------------------------------------------------------------------
_TREE = [  # parent -> child, traversal order from the pelvis
    ("pelvis", "lhip"), ("pelvis", "rhip"), ("pelvis", "neck"),
    ("neck", "lsho"), ("neck", "rsho"), ("neck", "nose"), ("neck", "head"),
    ("nose", "leye"), ("nose", "reye"), ("leye", "lear"), ("reye", "rear"),
    ("lsho", "lelb"), ("lelb", "lwri"), ("lwri", "lhand"),
    ("rsho", "relb"), ("relb", "rwri"), ("rwri", "rhand"),
    ("lhip", "lknee"), ("lknee", "lank"), ("lank", "lheel"), ("lank", "lbigtoe"), ("lank", "lsmalltoe"),
    ("rhip", "rknee"), ("rknee", "rank"), ("rank", "rheel"), ("rank", "rbigtoe"), ("rank", "rsmalltoe"),
]
_BLEN = {frozenset((a, b)): f for a, b, f, _ in BONES}
# bones whose depth sign is enumerated (others start at dz = 0 and are left to the optimiser)
_SIGN_BONES = [("pelvis", "neck"), ("neck", "lsho"), ("pelvis", "lhip"),
               ("lsho", "lelb"), ("lelb", "lwri"), ("rsho", "relb"), ("relb", "rwri"),
               ("lhip", "lknee"), ("lknee", "lank"), ("lank", "lbigtoe"),
               ("rhip", "rknee"), ("rknee", "rank"), ("rank", "rbigtoe")]
_MIRROR = {("neck", "rsho"): ("neck", "lsho"), ("pelvis", "rhip"): ("pelvis", "lhip")}


def _distance_from_scale(kpts2d, conf, K, height):
    """Taylor: px-per-metre s is at least l_px / L for every bone (foreshortening only shrinks)."""
    big = [(a, b) for a, b, f, _ in BONES if f > 0.1]
    r = [np.linalg.norm(kpts2d[IDX[a]] - kpts2d[IDX[b]]) / (_BLEN[frozenset((a, b))] * height)
         for a, b in big if min(conf[IDX[a]], conf[IDX[b]]) > 0.3]
    s = np.percentile(r, 90) if r else 0.25 * np.ptp(kpts2d, 0).max() / height
    return K[0, 0] / s, s


def _taylor(kpts2d, conf, K, height, signs):
    Z0, s = _distance_from_scale(kpts2d, conf, K, height)
    dz = {"pelvis": 0.0}
    for a, b in _TREE:
        L = _BLEN[frozenset((a, b))] * height
        l = np.linalg.norm(kpts2d[IDX[a]] - kpts2d[IDX[b]]) / s
        mag = np.sqrt(max(L * L - l * l, 0.0))
        key = _MIRROR.get((a, b), (a, b))
        sg = signs.get(key, 0.0) * (-1 if (a, b) in _MIRROR else 1)
        dz[b] = dz[a] + sg * mag
    Z = np.array([Z0 + dz[n] for n in JOINTS])
    return backproject(kpts2d.astype(np.float64), Z, K)


def init_geometric(kpts2d, conf, K, height):
    """Enumerate depth signs: torso bits jointly, then each limb independently (cheap)."""
    valid = conf > CONF_MIN
    hinge_ok = np.array([all(valid[IDX[n]] for n in need) for need in HINGE_NEEDS])
    torso_b = _SIGN_BONES[:3]
    limbs = [_SIGN_BONES[3:5], _SIGN_BONES[5:7], _SIGN_BONES[7:10], _SIGN_BONES[10:13]]
    hinge_of_limb = [[], [], [0, 1], [2, 3]]
    best = None
    for tb in itertools.product((-1, 1), repeat=3):
        signs = dict(zip(torso_b, tb))
        total = 0.0
        for limb, hidx in zip(limbs, hinge_of_limb):
            scores = []
            for lb in itertools.product((-1, 1), repeat=len(limb)):
                sg = {**signs, **dict(zip(limb, lb))}
                X = _taylor(kpts2d, conf, K, height, sg)
                h = hinge_terms(X)
                pen = sum(max(0, -h[k]) for k in hidx if hinge_ok[k])
                if not hidx:   # arms: weak prior that hands sit in front of the chest plane
                    _, _, fwd = body_frame(X)
                    pen += 0.1 * max(0, -(X[IDX[limb[-1][1]]] - X[IDX["neck"]]) @ fwd - 0.15)
                scores.append((pen, lb))
            pen, lb = min(scores, key=lambda x: x[0])
            signs.update(dict(zip(limb, lb))); total += pen
        X = _taylor(kpts2d, conf, K, height, signs)
        if hinge_ok[4]:
            total += 2 * max(0, -hinge_terms(X)[4])
        if best is None or total < best[0]:
            best = (total, signs, X)
    return best[2], best[1]


def init_signed_taylor(kpts2d, conf, K, height, z_rel, sign):
    """Robust init: each bone's |depth change| from 2D foreshortening vs its known length
    (Taylor), only its DIRECTION from the depth network. Immune to depth-magnitude outliers
    and per-limb bias; a wrong direction on a near-frontal bone costs little."""
    Z0, s = _distance_from_scale(kpts2d, conf, K, height)
    dz = {"pelvis": 0.0}
    for a, b in _TREE:
        L = _BLEN[frozenset((a, b))] * height
        l = np.linalg.norm(kpts2d[IDX[a]] - kpts2d[IDX[b]]) / s
        mag = np.sqrt(max(L * L - l * l, 0.0))
        d = sign * (z_rel[IDX[b]] - z_rel[IDX[a]])
        dz[b] = dz[a] + (np.sign(d) * mag if abs(d) > 0.02 else 0.0)
    Z = np.array([Z0 + dz[n] for n in JOINTS])
    return backproject(kpts2d.astype(np.float64), Z, K)


def init_from_depth(kpts2d, conf, K, height, z_rel, sign):
    Z0, _ = _distance_from_scale(kpts2d, conf, K, height)
    return backproject(kpts2d.astype(np.float64), Z0 + sign * z_rel, K)


# ----------------------------------------------------------------------------------------
# RTMW3D depth cue
# ----------------------------------------------------------------------------------------
class RTMW3DDepth:
    """RTMW3D-x (whole-body, 133 kpts, SimCC x/y/z). Runs top-down on S1's bbox."""

    def __init__(self, device: str = "auto", backend: str = "onnxruntime"):
        import rtmlib
        from .s1_keypoints import pick_device
        cfg = rtmlib.Wholebody3d.MODE["balanced"]
        self.dev = pick_device(device)
        self._mk = lambda d: rtmlib.RTMPose3d(cfg["pose"], model_input_size=cfg["pose_input_size"],
                                              backend=backend, device=d)
        try:
            self.model = self._mk(self.dev)
        except Exception as e:
            print(f"[S2] RTMW3D on '{self.dev}' failed ({e!s:.120}); using CPU")
            self.dev, self.model = "cpu", self._mk("cpu")

    def __call__(self, img_bgr, bbox):
        t0 = time.perf_counter()
        try:
            k3, sc, _, k2 = self.model(img_bgr, bboxes=[list(bbox)])
        except Exception as e:
            if self.dev == "cpu":
                raise
            print(f"[S2] RTMW3D inference on '{self.dev}' failed ({e!s:.100}); rebuilding on CPU")
            self.dev, self.model = "cpu", self._mk("cpu")
            k3, sc, _, k2 = self.model(img_bgr, bboxes=[list(bbox)])
        k3, sc, k2 = k3[0], np.clip(sc[0], 0, 1), k2[0]
        z, zc = CONVERTERS["wholebody133"](k3[:, 2:3].astype(np.float32), sc.astype(np.float32))
        uv, _ = CONVERTERS["wholebody133"](k2.astype(np.float32), sc.astype(np.float32))
        z = z[:, 0] - z[IDX["pelvis"], 0]
        return dict(z_rel=z, z_conf=zc, kpts2d=uv, raw=k3, raw2d=k2, raw_conf=sc,
                    ms=1e3 * (time.perf_counter() - t0))


# ----------------------------------------------------------------------------------------
# Stage entry point
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# Left/right label hypotheses: where limbs overlap in 2D, the image cannot say which thigh
# continues into which shin (or upper arm into which forearm); 2D detectors then often swap.
# ----------------------------------------------------------------------------------------
SWAP_SETS = {
    "legs below knees": ["knee", "ank", "heel", "bigtoe", "smalltoe"],
    "feet+ankles": ["ank", "heel", "bigtoe", "smalltoe"],
    "arms below elbows": ["elb", "wri", "hand"],
    "hands+wrists": ["wri", "hand"],
}


def _segments_cross(p1, q1, p2, q2):
    """True if 2D segments p1q1 and p2q2 properly intersect."""
    def orient(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    return (orient(p1, q1, p2) != orient(p1, q1, q2)) and (orient(p2, q2, p1) != orient(p2, q2, q1))


def label_hypotheses(k2, c, ratio=0.35, max_hyp=4):
    """[(name, [joint suffixes to swap L<->R])] — identity first; swaps only where the 2D is
    ambiguous: the two knees/ankles/elbows/wrists nearly overlap, or the L and R segments ending
    there cross each other in the image."""
    i = IDX
    d = lambda a, b: float(np.linalg.norm(k2[i[a]] - k2[i[b]]))
    seg = lambda p, q: 0.5 * (d("l" + p, "l" + q) + d("r" + p, "r" + q))
    ok = lambda *js: all(c[i[s + j]] > CONF_MIN for j in js for s in "lr")

    def ambiguous(prox, j):
        if not ok(prox, j):
            return False
        near = d("l" + j, "r" + j) < ratio * seg(prox, j)
        cross = _segments_cross(k2[i["l" + prox]], k2[i["l" + j]], k2[i["r" + prox]], k2[i["r" + j]])
        return near or cross

    legs, arms = [None], [None]
    if ambiguous("hip", "knee"):
        legs.append("legs below knees")
    if ambiguous("knee", "ank"):
        legs.append("feet+ankles")
    if ambiguous("sho", "elb"):
        arms.append("arms below elbows")
    if ambiguous("elb", "wri"):
        arms.append("hands+wrists")
    hyps = []
    for lg in legs:
        for am in arms:
            names = [n for n in (lg, am) if n]
            hyps.append((" + ".join(names) or "as detected", [j for n in names for j in SWAP_SETS[n]]))
    return hyps[:max_hyp]


def swap_lr(arrs, suffixes):
    """Swap left/right entries of every (J, ...) array for the given joint suffixes."""
    out = []
    for a in arrs:
        if a is None:
            out.append(None); continue
        a = a.copy()
        for sfx in suffixes:
            li, ri = IDX["l" + sfx], IDX["r" + sfx]
            a[[li, ri]] = a[[ri, li]]
        out.append(a)
    return out


_POOL = None


def _fit_job(kw):
    """Picklable wrapper for running fit_skeleton in a worker process."""
    return fit_skeleton(**kw)


def run_fits(jobs, workers=None):
    """Run independent fit_skeleton calls in parallel (persistent process pool; serial fallback)."""
    global _POOL
    import os
    workers = workers if workers is not None else min(8, os.cpu_count() or 1)
    if any(kw.get("jac", "auto") != "fd" for kw in jobs):
        from . import s2_jax
        if s2_jax.available():          # exact-Jacobian fits are fast: no process pool (each worker
            workers = 1                 # would recompile, and JAX is not fork-safe)
    if workers <= 1 or len(jobs) <= 1:
        return [fit_skeleton(**kw) for kw in jobs]
    try:
        if _POOL is None:
            from concurrent.futures import ProcessPoolExecutor
            _POOL = ProcessPoolExecutor(max_workers=workers)
        return list(_POOL.map(_fit_job, jobs))
    except Exception as e:                                 # e.g. sandboxed / no fork: stay serial
        print(f"[S2] parallel fits unavailable ({e!s:.80}); running serially")
        _POOL = None
        return [fit_skeleton(**kw) for kw in jobs]


class Lifter3D:
    """S2 (+ integrated S3): 2D keypoints -> physically plausible metric 3D skeleton.

    One procedure:
      1. label hypotheses   identity + L/R swaps where limbs overlap in 2D
      2. geometric fits     per hypothesis and depth sign (depth-network init + signed-Taylor init)
      3. flip search        exact depth-direction flips scored for plausibility (no refits)
      4. physical fit       the best candidates of every hypothesis are refined under ONE
                            objective: image + depth cue + anatomy + passive muscle
                            relaxation + gravity effort + balance + planted feet on a level floor
                            (camera pitch fitted)
      5. winner             lowest physical energy
    """

    def __init__(self, backend: str = "rtmw3d", height: float = 1.70, z_sign: str = "auto",
                 device: str = "auto", dynamic: bool = False, search: bool = True,
                 physical: bool = True, mass: float | None = None, n_physical: int = 2,
                 label_swaps: bool = True, swap_prior: float = 3.0, workers: int | None = None,
                 occlusion: bool = True, jac: str = "auto", solver: str = "auto"):
        self.backend, self.height, self.z_sign = backend, height, z_sign
        self.dynamic, self.search = dynamic, search
        self.physical, self.mass, self.n_physical = physical, mass, n_physical
        self.label_swaps, self.swap_prior, self.workers = label_swaps, swap_prior, workers
        self.occlusion = occlusion
        self.phys_tol = 1e-3
        self.fit_kw = dict(w_balance=0.0 if dynamic else 1.0, jac=jac, solver=solver)
        self.depth = RTMW3DDepth(device) if backend == "rtmw3d" else None
        self.jac = "fd"
        if jac != "fd":
            from . import s2_jax
            if s2_jax.available():
                t = time.perf_counter()
                s2_jax.warmup()                       # compile once (or load the on-disk cache)
                self.jac = "jax"
                print(f"[S2] exact Jacobian via JAX, solver {'LM' if solver != 'scipy' else 'SciPy trf'} "
                      f"(compile/cache load {time.perf_counter() - t:.1f}s)")
            elif jac == "jax":
                raise ImportError("--jac jax needs `pip install jax`")
            else:
                print("[S2] jax not installed: finite-difference Jacobian (pip install jax for ~10x faster S2)")

    # ------------------------------------------------------------------
    def __call__(self, frame, img_bgr=None, K=None, X_prev=None):
        """Fill frame.joints3d / valid3d / camera_K / camera_R / s2_info in place and return it."""
        t0 = time.perf_counter()
        K = default_K(frame.img_hw) if K is None else K
        k2, c = frame.kpts2d.astype(np.float64), frame.conf.copy()
        info = {}
        z = zc = None
        if self.depth is not None:
            d = self.depth(img_bgr, frame.bbox)
            info["rtmw3d_ms"] = d["ms"]
            for n in ("lhand", "rhand"):          # hands only exist in the whole-body model
                if c[IDX[n]] <= CONF_MIN:
                    k2[IDX[n]], c[IDX[n]] = d["kpts2d"][IDX[n]], d["z_conf"][IDX[n]]
            frame.raw_kpts3d_rtmw3d = d["raw"]
            z, zc = d["z_rel"], d["z_conf"]

        if X_prev is not None:                    # tracking: one warm-started physical fit
            hyps = [("as detected", [])]
        else:
            hyps = label_hypotheses(k2, c) if self.label_swaps else [("as detected", [])]
        edges = None
        if self.occlusion and img_bgr is not None and frame.bbox is not None:
            from .occlusion import EdgeImage
            te = time.perf_counter()
            edges = EdgeImage(img_bgr, frame.bbox)
            info["edge_ms"] = 1e3 * (time.perf_counter() - te)
        data = {h: swap_lr([k2, c, z, zc], sw) for h, sw in hyps}
        jobs, meta = [], []
        for hname, sw in hyps:
            k2h, ch, zh, zch = data[hname]
            for job, m in self._fit_jobs(k2h, ch, K, zh, zch, X_prev):
                jobs.append(job); meta.append(dict(m, hyp=hname))
        fits = [dict(m, X=X, v=v, info=inf, cost=inf["cost"])
                for m, (X, v, inf) in zip(meta, run_fits(jobs, self.workers))]
        pool, occ_report = [], {}
        for hname, sw in hyps:
            k2h, ch, zh, zch = data[hname]
            mine = [f for f in fits if f["hyp"] == hname]
            cons = []
            if edges is not None:              # occlusion order from body-part contours
                from .occlusion import occlusion_constraints
                ref = min(mine, key=lambda f: f["cost"])
                cons, rep = occlusion_constraints(img_bgr, frame.bbox, k2h, ch, ref["X"], K, self.height, edges=edges)
                occ_report[hname] = rep
            for cd in self._candidates(k2h, ch, K, zh, zch, mine, X_prev, cons):
                cd.update(hyp=hname, swap=sw, k2=k2h, c=ch, z=zh, zc=zch, cons=cons)
                pool.append(cd)

        if self.physical:
            # refine the best candidates of EVERY label hypothesis (so each gets a fair chance)
            sel = []
            for hname, _ in hyps:
                mine = sorted((cd for cd in pool if cd["hyp"] == hname), key=lambda cd: cd["pre"])
                sel += mine[:self.n_physical]
            phys = self._physical_batch(sel, K)
            for cd in phys:                          # detector labels are usually right: a swap
                if cd["swap"]:                       # must be clearly better to win
                    cd["cost"] += self.swap_prior
            phys.sort(key=lambda cd: cd["cost"])
            win, ranking = phys[0], phys
        else:
            pool.sort(key=lambda cd: cd["pre"])
            win, ranking = pool[0], pool

        X, v, inf = win["X"], win["v"], dict(win["info"])
        reproj = np.linalg.norm(project(X[v], K) - win["k2"][v], axis=1)
        from .occlusion import order_terms
        inf.update(occlusion={r["seg"]: r["verdict"] for r in occ_report.get(win["hyp"], []) if r["verdict"]},
                   occlusion_residual=float((order_terms(X, win.get("cons") or []) ** 2).sum()))
        inf.update(z_sign=win["sign"], init=win["src"], label_hypothesis=win["hyp"],
                   label_swaps_tested=[h for h, _ in hyps],
                   flips={g: list(m) for g, m in win["flips"].items()},
                   prior_terms_max=float(pose_prior_terms(X).max()), balance_term=float(balance_term(X)),
                   reproj_px_median=float(np.median(reproj)), reproj_px_max=float(reproj.max()),
                   subject_distance_m=float(X[IDX["pelvis"], 2]),
                   candidates=[dict(hyp=cd["hyp"], sign=cd["sign"], src=cd["src"],
                                    flips={g: list(m) for g, m in cd["flips"].items()},
                                    energy=round(cd.get("cost", cd["pre"]), 2)) for cd in ranking[:8]],
                   s2_ms=1e3 * (time.perf_counter() - t0))
        if len(ranking) > 1:
            e0 = ranking[0].get("cost", ranking[0]["pre"]); e1 = ranking[1].get("cost", ranking[1]["pre"])
            inf["energy_margin"] = float(e1 / max(e0, 1e-9))
        info.update(inf)

        frame.kpts2d, frame.conf = win["k2"].astype(np.float32), win["c"]
        frame.joints3d, frame.valid3d, frame.camera_K = X, v, K
        if info.get("physics_p"):
            ph = info["physics_p"][0]                      # fitted camera pitch: level the world
            frame.camera_R = np.array([[1, 0, 0], [0, np.cos(ph), np.sin(ph)], [0, -np.sin(ph), np.cos(ph)]])
        frame.body_height_m, frame.s2_backend, frame.s2_info = self.height, self.backend, info
        from .spine import spine_curve, polyline
        sp = spine_curve(frame.world_joints())
        frame.spine3d = frame.from_world(polyline(sp))
        info.update(spine_lateral_bend_deg=sp["lateral_bend_deg"], spine_twist_deg=sp["twist_deg"],
                    spine_note="interpolated from torso box")
        return frame

    # ------------------------------------------------------------------
    def _fit_jobs(self, k2, c, K, z, zc, X_prev=None):
        """Geometric fits per depth sign (depth-network init + signed-Taylor init)."""
        quick = dict(self.fit_kw, tol=1e-3, iters=80)        # initialisers only: coarse is fine
        base = dict(kpts2d=k2, conf=c, K=K, height=self.height, **quick)
        out = []
        if z is not None:
            for sg in {"auto": (1, -1), "+": (1,), "-": (-1,)}[self.z_sign]:
                inits = ([X_prev] if X_prev is not None else
                         [init_from_depth(k2, c, K, self.height, sg * z, 1),
                          init_signed_taylor(k2, c, K, self.height, z, sg)])
                for X0 in inits:
                    out.append((dict(base, X0=X0, z_rel=sg * z, z_conf=zc), dict(sign=sg)))
        else:
            X0 = X_prev if X_prev is not None else init_geometric(k2, c, K, self.height)[0]
            out.append((dict(base, X0=X0), dict(sign=0)))
        return out

    def _candidates(self, k2, c, K, z, zc, fits, X_prev=None, cons=None):
        """Best fit per depth sign + plausibility-ranked depth-flip hypotheses.
        Each candidate gets 'pre' = plausibility energy (cheap: no refit, no effort)."""
        from .plausibility import Scorer, search_flips
        best = {}
        for f in fits:
            if f["sign"] not in best or f["cost"] < best[f["sign"]]["cost"]:
                best[f["sign"]] = f
        ref = min(fits, key=lambda f: f["cost"])
        out = []
        for sg, f in best.items():
            zr = None if z is None else sg * z
            sc = Scorer(k2, c, K, ref["X"], ref["v"], self.height, zr, zc, dynamic=self.dynamic, order_cons=cons)
            ds = f["info"].get("depth_scale") or 1.0
            out.append(dict(X=f["X"], v=f["v"], info=f["info"], sign=sg, flips={}, src="fit",
                            pre=sc.terms(f["X"], ds)["total"]))
            if self.search and X_prev is None:
                for score, Xh, flips in search_flips(f["X"], K, sc, depth_scale=ds)[:2]:
                    if flips:
                        out.append(dict(X=Xh, v=f["v"], info=dict(f["info"]), sign=sg, flips=flips,
                                        src="flip", pre=score))
        return out

    def _physical_batch(self, cands, K):
        from .biomech import Physics, choose_contacts
        jobs = []
        for cd in cands:
            zr = None if cd["z"] is None else (cd["sign"] or 1) * cd["z"]
            cd["contacts"] = choose_contacts(cd["X"], cd["v"], self.height, dynamic=self.dynamic)
            ph = Physics(cd["contacts"], self.height, mass=self.mass, dynamic=self.dynamic)
            jobs.append(dict(kpts2d=cd["k2"], conf=cd["c"], K=K, height=self.height, X0=cd["X"], z_rel=zr,
                             z_conf=cd["zc"], physics=ph, tol=self.phys_tol, iters=120,
                             order_cons=cd.get("cons"), **self.fit_kw))
        out = []
        for cd, (X, v, inf) in zip(cands, run_fits(jobs, self.workers)):
            inf["contacts"] = cd["contacts"]
            out.append(dict(cd, X=X, v=v, info=inf, cost=inf["cost"], src=cd["src"] + "+physical"))
        return out

    def _physical(self, cd, K):
        return self._physical_batch([cd], K)[0]

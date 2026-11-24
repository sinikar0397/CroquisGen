"""
Exact Jacobian for the S2 / S2+S3 least-squares fit (`s2_lift.fit_skeleton`) with JAX.

Why: the fit spent ~90% of its time in SciPy's finite-difference Jacobian (one extra residual
evaluation per parameter, ~90 per solver step). Here the same residuals are written in
jax.numpy, compiled once with XLA, and differentiated exactly with forward-mode autodiff.

Design
  * Fixed shapes. All joints, bones, tree edges, symmetry pairs, contacts (padded to
    MAX_CONTACTS) and occlusion constraints (padded) are always present; what the numpy version
    drops is multiplied by a 0/1 mask here. So ONE compilation serves every fit of every image
    (a persistent XLA cache in ~/.cache/croquisgen/jax keeps it across runs).
  * The optimiser still sees the original parameter vector (X[valid], scales, [depth scale],
    [camera pitch]); it is embedded into a fixed-length vector and the Jacobian's columns are
    selected back. Results match the numpy path up to optimiser round-off.
  * The numpy residuals (s2_lift, biomech, muscles, plausibility, occlusion) remain the
    reference: `test_s2_jax.py` checks residuals and J^T r / J^T J against finite differences.
    If you change a residual there, change it here too (the parity test will tell you).
  * Gradient-safe versions of the few non-smooth spots (sqrt / arctan2 / arccos at 0 or ±1,
    value-dependent branches as jnp.where) — values are identical away from those points.
"""
from __future__ import annotations
import os
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    try:
        _cache = os.path.expanduser("~/.cache/croquisgen/jax")
        os.makedirs(_cache, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", _cache)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
    except Exception:
        pass
    HAVE_JAX = True
except Exception:                                    # pragma: no cover - optional dependency
    jax = jnp = None
    HAVE_JAX = False

from .skeleton import J, IDX, BONE_IDX, BONE_FRAC, BONE_SIG, SYM_PAIRS, JOINTS
from . import biomech as BM
from . import muscles as MU

MAX_CONTACTS = 16
MAX_SEG_PTS = 160          # occlusion: sample points on limb-vs-torso segments
MAX_PAIRS = 24             # occlusion: limb-limb crossings
NP = 3 * J + 6             # full parameter vector: X (3J), log_h, g0..2, log_sz, pitch

_EPS = 1e-30
_CL = 1 - 1e-10            # clip for arccos (finite gradient)


def available() -> bool:
    return HAVE_JAX


# ============================================================================ safe primitives
def _norm(v):
    return jnp.sqrt(jnp.sum(v * v, -1) + _EPS)


def _unit(v):
    return v / (_norm(v)[..., None] + 1e-9) if v.ndim > 1 else v / (_norm(v) + 1e-9)


def _sp(x):
    return jnp.logaddexp(0.0, x)


def _sqrt0(x):
    """sqrt with zero gradient at 0 (instead of NaN)."""
    pos = x > 0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, x, 1.0)), 0.0)


def _atan2(y, x):
    small = (jnp.abs(x) + jnp.abs(y)) < 1e-30
    return jnp.where(small, 0.0, jnp.arctan2(jnp.where(small, 0.0, y), jnp.where(small, 1.0, x)))


def _acos(c):
    return jnp.arccos(jnp.clip(c, -_CL, _CL))


def _deg(x):
    return x * (180.0 / np.pi)


def _project(X, K):
    z = jnp.maximum(X[:, 2:3], 1e-3)
    return (X[:, :2] / z) * jnp.stack([K[0, 0], K[1, 1]]) + K[:2, 2]


# ============================================================================ s2_lift terms
def body_frame(X, part="both"):
    i = IDX
    up = _unit(X[i["neck"]] - X[i["pelvis"]])
    rp, rt = X[i["rhip"]] - X[i["lhip"]], X[i["rsho"]] - X[i["lsho"]]
    r = {"pelvis": rp, "thorax": rt}.get(part, _unit(rp) + _unit(rt)) if part != "both" else _unit(rp) + _unit(rt)
    r = _unit(r - (r @ up) * up)
    fwd = _unit(jnp.cross(up, r))
    return up, r, fwd


def hinge_terms(X):
    i = IDX
    up, r, _ = body_frame(X, "pelvis")
    _, _, fwd = body_frame(X, "thorax")
    out = []
    for s in "lr":
        hip, knee, ank, toe = (X[i[s + n]] for n in ("hip", "knee", "ank", "bigtoe"))
        thigh, shank, foot = knee - hip, ank - knee, toe - ank
        out.append(-jnp.cross(thigh, shank) @ r / (_norm(thigh) + 1e-6))
        out.append(jnp.cross(shank, foot) @ r / (_norm(shank) + 1e-6))
    out.append((X[i["nose"]] - X[i["neck"]]) @ fwd)
    return jnp.stack(out)


def _ang(u, v):
    c = (u @ v) / (_norm(u) * _norm(v) + 1e-9)
    return _deg(jnp.arccos(jnp.clip(c, -0.999999, 0.999999)))


def pose_prior_terms(X):
    i = IDX
    up, r, fwd = body_frame(X, "thorax")
    _, _, fwd_p = body_frame(X, "pelvis")
    res = []
    for s in "lr":
        sho, elb, wri = X[i[s + "sho"]], X[i[s + "elb"]], X[i[s + "wri"]]
        hip, knee, ank = X[i[s + "hip"]], X[i[s + "knee"]], X[i[s + "ank"]]
        elbow_flex = 180 - _ang(sho - elb, wri - elb)
        knee_flex = 180 - _ang(hip - knee, ank - knee)
        res += [3 * _sp((elbow_flex - 140) / 5), 3 * _sp((knee_flex - 155) / 5)]
        thigh, uarm = knee - hip, elb - sho
        hip_ext = _deg(_atan2(-(thigh @ fwd_p), -(thigh @ up)))
        sho_ext = _deg(_atan2(-(uarm @ fwd), -(uarm @ up)))
        res += [_sp((hip_ext - 30) / 8), _sp((sho_ext - 60) / 10)]
    rp, rt = X[i["rhip"]] - X[i["lhip"]], X[i["rsho"]] - X[i["lsho"]]
    rp_, rt_ = rp - (rp @ up) * up, rt - (rt @ up) * up
    twist = _deg(_atan2(jnp.cross(rp_, rt_) @ up, rp_ @ rt_))
    rpf, rtf = rp - (rp @ fwd) * fwd, rt - (rt @ fwd) * fwd
    bend = _deg(_atan2(jnp.cross(rpf, rtf) @ fwd, rpf @ rtf))
    res += [twist / 45, _sp((jnp.abs(twist) - 60) / 6), bend / 40, _sp((jnp.abs(bend) - 45) / 6)]
    eyes, ears = (X[i["leye"]] + X[i["reye"]]) / 2, (X[i["lear"]] + X[i["rear"]]) / 2
    nose, neck, head = X[i["nose"]], X[i["neck"]], X[i["head"]]
    hv = head - neck
    res += [2 * _sp((0.02 - (eyes - ears) @ fwd) / 0.01),
            2 * _sp((0.04 - (nose - neck) @ up) / 0.01),
            2 * _sp((0.5 - hv @ up / (_norm(hv) + 1e-9)) / 0.08)]
    return jnp.stack(res)


def approx_com(X):
    i = IDX
    m = lambda a, b: (X[i[a]] + X[i[b]]) / 2
    parts = [(0.435, 0.6 * X[i["pelvis"]] + 0.4 * X[i["neck"]]), (0.07, X[i["head"]] * 0.5 + X[i["neck"]] * 0.5)]
    for s in "lr":
        parts += [(0.142, m(s + "hip", s + "knee")), (0.057, m(s + "knee", s + "ank")),
                  (0.027, m(s + "sho", s + "elb")), (0.022, m(s + "elb", s + "wri"))]
    w = sum(p[0] for p in parts)
    return sum(p[0] * p[1] for p in parts) / w


def balance_term(X, margin=0.12):
    g = jnp.array([0.0, 1.0, 0.0])
    horiz = lambda v: v - (v @ g) * g
    c = horiz(approx_com(X))
    a, b = horiz(X[IDX["lank"]]), horiz(X[IDX["rank"]])
    ab = b - a
    t = jnp.clip((c - a) @ ab / (ab @ ab + 1e-9), 0, 1)
    d = _norm(c - (a + t * ab))
    return _sp((d - margin) / 0.04)


# ============================================================================ plausibility terms
from .plausibility import _TWIST_LIMBS, _TWIST_RANGE, _PA, _PB, _PR, _PF   # noqa: E402


def twist_terms(X):
    frames = {"arm": body_frame(X, "thorax"), "leg": body_frame(X, "pelvis")}
    res = []
    for name, (p, m, d, ref, sg) in _TWIST_LIMBS.items():
        up, r, fwd = frames[name[2:]]
        P, M, D = X[IDX[p]], X[IDX[m]], X[IDX[d]]
        u = M - P
        u = u / (_norm(u) + 1e-9)
        a = -up
        c = jnp.clip(a @ u, -1, 1)
        v = jnp.cross(a, u)
        refv = fwd if ref == "fwd" else -fwd
        ok = c > -0.999
        f_ref = jnp.where(ok, refv * c + jnp.cross(v, refv) + v * (v @ refv) / jnp.where(ok, 1 + c, 1.0), -refv)
        low = D - M
        f = low - (low @ u) * u
        fr = f_ref - (f_ref @ u) * u
        ang = _deg(_atan2(jnp.cross(fr, f) @ u, fr @ f))
        flex_w = _norm(f) / (_norm(low) + 1e-9)
        swing = _deg(_acos(c))
        w = jnp.clip((flex_w - 0.15) / 0.3, 0, 1) * jnp.clip((165 - swing) / 30, 0, 1)
        ang = sg * ang
        lo, hi = _TWIST_RANGE[name[2:]]
        res += [w * 2 * _sp((ang - hi) / 8), w * 2 * _sp((lo - ang) / 8)]
    return jnp.stack(res)


def _seg_dist(p1, q1, p2, q2):
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = jnp.sum(d1 * d1, 1) + 1e-12
    e = jnp.sum(d2 * d2, 1) + 1e-12
    b = jnp.sum(d1 * d2, 1)
    c = jnp.sum(d1 * r, 1)
    f = jnp.sum(d2 * r, 1)
    den = a * e - b * b
    s = jnp.where(den > 1e-12, jnp.clip((b * f - c * e) / jnp.maximum(den, 1e-12), 0, 1), 0.0)
    t = (b * s + f) / e
    lo, hi = t < 0, t > 1
    s = jnp.where(lo, jnp.clip(-c / a, 0, 1), jnp.where(hi, jnp.clip((b - c) / a, 0, 1), s))
    t = jnp.clip(t, 0, 1)
    return _norm((p1 + d1 * s[:, None]) - (p2 + d2 * t[:, None]))


def collision_terms(X, height, ok):
    qa, pa0 = X[_PA[:, 1]], X[_PA[:, 0]]
    pa = qa + (pa0 - qa) * _PF[:, None]
    pen = _PR * (height / 1.70) - _seg_dist(pa, qa, X[_PB[:, 0]], X[_PB[:, 1]])
    pen = jnp.where(ok, pen, -1.0)
    return 0.01 * _sp(pen / 0.01)


# ============================================================================ occlusion terms
def order_terms(X, s, sigma=0.03, chest=0.08):
    i = IDX
    pel, nk = X[i["pelvis"]], X[i["neck"]]
    up = nk - pel
    lat = X[i["rsho"]] - X[i["lsho"]]
    nrm = jnp.cross(up, lat)
    nrm = nrm / (_norm(nrm) + 1e-9)
    c0 = 0.5 * (pel + nk)
    # limb vs torso plane
    A, B = X[s["oa"]], X[s["ob"]]
    P = A + s["ot"][:, None] * (B - A)
    Pz = jnp.where(s["om"] > 0, P[:, 2], 1.0)
    v = P / Pz[:, None]
    den = v @ nrm
    ok = jnp.abs(den) > 0.15
    lam = jnp.where(ok, (nrm @ c0) / jnp.where(ok, den, 1.0), P[:, 2])
    x = jnp.where(s["okind"] > 0, (lam - P[:, 2]) / sigma, (P[:, 2] - (lam - chest)) / sigma)
    seg = jnp.where(ok, s["ow"] * jnp.logaddexp(0.0, x), 0.0) * s["om"]
    # limb-limb crossings
    zf = X[s["pfa"], 2] + s["ptf"] * (X[s["pfb"], 2] - X[s["pfa"], 2])
    zb = X[s["pba"], 2] + s["ptb"] * (X[s["pbb"], 2] - X[s["pba"], 2])
    pair = s["pweight"] * jnp.logaddexp(0.0, (zf + s["pmargin"] - zb) / sigma) * s["pm"]
    return jnp.concatenate([seg, pair])


# ============================================================================ biomech + muscles
def gravity_dir(pitch):
    return jnp.stack([jnp.zeros_like(pitch), jnp.cos(pitch), jnp.sin(pitch)])


def frames(X):
    i = IDX
    up = _unit(X[i["neck"]] - X[i["pelvis"]])
    out = {}
    for name, (a, b) in (("pelvis", ("lhip", "rhip")), ("thorax", ("lsho", "rsho"))):
        r = X[i[b]] - X[i[a]]
        r = _unit(r - (r @ up) * up)
        out[name] = (up, r, _unit(jnp.cross(up, r)))
    return out


def _elevation_split(seg, up, lat, fwd):
    elev = _deg(_acos(-(seg @ up)))
    plane = _atan2(seg @ fwd, seg @ lat)
    return elev * jnp.sin(plane), elev * jnp.cos(plane)


def joint_angles(X, down, F):
    i = IDX
    ang, w = {}, {}
    for s, lat_sign in (("l", -1.0), ("r", 1.0)):
        up, r, fwd = F["pelvis"]
        lat = r * lat_sign
        hip, knee, ank = X[i[s + "hip"]], X[i[s + "knee"]], X[i[s + "ank"]]
        heel, toe = X[i[s + "heel"]], X[i[s + "bigtoe"]]
        th, sh, ft = _unit(knee - hip), _unit(ank - knee), _unit(toe - heel)
        ang[s + "hip_flex"], ang[s + "hip_abd"] = _elevation_split(th, up, lat, fwd)
        kf = _deg(_acos(th @ sh))
        ang[s + "knee_flex"] = jnp.where(jnp.cross(th, sh) @ (-r) >= -1e-6, kf, -kf)
        ang[s + "ankle_dorsi"] = 90 - _deg(_acos((-sh) @ ft))
        fh = ft - (ft @ up) * up
        ang[s + "foot_out"] = _deg(_atan2(fh @ lat, fh @ fwd))
        w[s + "foot_out"] = jnp.clip((_sqrt0(fh @ fh) - 0.3) / 0.4, 0, 1)
        f_ref = -fwd - (-fwd @ th) * th
        f_sh = sh - (sh @ th) * th
        n1, n2 = _sqrt0(f_ref @ f_ref), _sqrt0(f_sh @ f_sh)
        ang[s + "hip_rot"] = jnp.where(n2 > 1e-6, lat_sign * _deg(_atan2(jnp.cross(f_ref, f_sh) @ th, f_ref @ f_sh)), 0.0)
        w[s + "hip_rot"] = jnp.clip((n2 - 0.15) / 0.3, 0, 1) * (n1 > 0.3)

        up, r, fwd = F["thorax"]
        lat = r * lat_sign
        sho, elb, wri = X[i[s + "sho"]], X[i[s + "elb"]], X[i[s + "wri"]]
        ua, fa = _unit(elb - sho), _unit(wri - elb)
        ang[s + "sho_flex"], ang[s + "sho_abd"] = _elevation_split(ua, up, lat, fwd)
        ang[s + "elbow_flex"] = _deg(_acos(ua @ fa))
        f_ref = fwd - (fwd @ ua) * ua
        f_fa = fa - (fa @ ua) * ua
        n1, n2 = _sqrt0(f_ref @ f_ref), _sqrt0(f_fa @ f_fa)
        ang[s + "sho_rot"] = jnp.where(n2 > 1e-6, -lat_sign * _deg(_atan2(jnp.cross(f_ref, f_fa) @ ua, f_ref @ f_fa)), 0.0)
        w[s + "sho_rot"] = jnp.clip((n2 - 0.15) / 0.3, 0, 1) * jnp.clip((n1 - 0.2) / 0.3, 0, 1)

    (upp, rp, fp), (upt, rt, ft_) = F["pelvis"], F["thorax"]
    ang["trunk_twist"] = _deg(_atan2(jnp.cross(rp, rt) @ upp, rp @ rt))
    tr = X[i["neck"]] - X[i["pelvis"]]
    ang["trunk_bend"] = _deg(_atan2(tr @ rp, tr @ (-down) + 1e-9))
    nose = X[i["nose"]] - X[i["neck"]]
    ang["head_yaw"] = _deg(_atan2(nose @ rt, nose @ ft_))
    return ang, w


def passive_residuals(ang, w):
    res = []
    for k, a in ang.items():
        base = k[1:] if k[0] in "lr" and k[1:] in BM.RELAXED else k
        n, lo, hi = BM.RELAXED[base]
        d = a - n
        res.append(w.get(k, 1.0) * d / jnp.where(d < 0, lo, hi))
    return jnp.stack(res)


def _m_passive(lt):
    x = jnp.maximum(lt - 1.0, 0.0)
    return (jnp.exp(MU._KPE * x / MU._E0) - 1.0) / (np.exp(MU._KPE) - 1.0)


def _m_force_length(lt):
    return jnp.exp(-(lt - 1.0) ** 2 / MU._GAMMA)


def _m_axes(X, down, F):
    (upp, rp, fp), (upt, rt, ft) = F["pelvis"], F["thorax"]
    ax = {}
    for s, lat_sign in (("l", -1.0), ("r", 1.0)):
        latp, latt = rp * lat_sign, rt * lat_sign
        ax[s + "flex_p"] = rp
        ax[s + "abd_p"] = _unit(jnp.cross(-upp, latp))
        ax[s + "knee"] = -rp
        ax[s + "ankle"] = rp
        ax[s + "flex_t"] = rt
        ax[s + "abd_t"] = _unit(jnp.cross(-upt, latt))
        ua = X[IDX[s + "elb"]] - X[IDX[s + "sho"]]
        fa = X[IDX[s + "wri"]] - X[IDX[s + "elb"]]
        n = jnp.cross(ua, fa)
        big = _sqrt0(n @ n) > 0.2 * _sqrt0(ua @ ua) * _sqrt0(fa @ fa)
        ax[s + "elbow"] = jnp.where(big, _unit(n), rt)
    ax["trunk_flex"] = -rp
    ax["trunk_bend"] = fp
    ax["neck_flex"] = -rt
    return ax


def _m_extra_angles(X, down, F):
    (upp, rp, fp), (upt, rt, ft) = F["pelvis"], F["thorax"]
    trunk = _unit(X[IDX["neck"]] - X[IDX["pelvis"]])
    lean = _deg(_atan2(trunk @ fp, trunk @ (-down)))
    head = _unit(X[IDX["head"]] - X[IDX["neck"]])
    neck = _deg(_atan2(head @ ft, head @ trunk)) - 10.0
    return {"trunk_flex": 0.6 * lean, "neck_flex": neck}


def stress_residuals(X, down, F, angles, tau_by_joint):
    ang = dict(angles)
    ang.update(_m_extra_angles(X, down, F))
    ax = _m_axes(X, down, F)
    out = []

    def one(prefix, spec):
        key, jnt, axk, rest, grp = spec
        a_key = (prefix + key) if prefix else key
        if a_key not in ang:
            return
        th = (ang[a_key] - rest) * (np.pi / 180)
        jname = (prefix + jnt) if prefix else jnt
        tau = tau_by_joint.get(jname)
        m_req = 0.0 if tau is None else -(tau @ ax[(prefix + axk) if prefix else axk])
        lts = [1.0 - side * r * th / L0 for _, side, r, L0, _ in grp]
        fps = [_m_passive(lt) for lt in lts]
        m_pas = sum(side * fp * Mmax for (_, side, r, L0, Mmax), fp in zip(grp, fps))
        m_act = m_req - m_pas
        for (gname, side, r, L0, Mmax), lt, fp in zip(grp, lts, fps):
            a = jnp.maximum(0.0, side * m_act) / (Mmax * _m_force_length(lt))
            out.append(_sqrt0(a * a + fp * fp))

    for s in "lr":
        for dof, spec in MU.DOFS.items():
            one(s, spec)
    for dof, spec in MU.TRUNK_DOFS.items():
        one("", spec)
    return jnp.stack(out)


def physics_residuals(X, pitch, s):
    down = gravity_dir(pitch)
    F = frames(X)
    ang, w = joint_angles(X, down, F)
    pr = passive_residuals(ang, w)
    # --- static solve (biomech.Physics.solve with padded contacts) ---
    mass = s["mass"]
    P = X[BM._SA] + BM._SR[:, None] * (X[BM._SB] - X[BM._SA])
    pel, nk = X[IDX["pelvis"]], X[IDX["neck"]]
    T = pel + BM._TS[:, None] * (nk - pel)
    P = jnp.vstack([P, T])
    m = jnp.asarray(np.r_[BM._SM, BM._TM]) * mass
    com = (m @ P) / m.sum()
    e1 = _unit(jnp.array([1.0, 0.0, 0.0]) - down * down[0])
    e2 = _unit(jnp.cross(down, e1))
    cm = s["cmask"]
    nc = cm.sum()
    C = X[s["ci"]] * cm[:, None]
    A = jnp.stack([C @ e1, C @ e2, cm])
    b = jnp.stack([com @ e1, com @ e2, 1.0])
    lam = 1e-3
    M = A.T @ A + lam * jnp.eye(MAX_CONTACTS)
    wv = jnp.linalg.solve(M, A.T @ b + lam / jnp.maximum(nc, 1.0) * cm)
    wv = jnp.clip(wv, 0, None) * cm
    wv = wv / jnp.maximum(wv.sum(), 1e-9)
    cop = jnp.where(nc > 0, wv @ C, com)
    G = BM.G
    Fg = jnp.outer(m * G, down)
    Fc = -jnp.outer(wv * m.sum() * G, down)
    cg = jnp.cross(P, Fg)
    cc = jnp.cross(C, Fc)
    Jx = X[BM._EJ]
    Fsum = BM._MASK_SEG @ Fg + s["mask_c"] @ Fc
    tau = BM._MASK_SEG @ cg + s["mask_c"] @ cc - jnp.cross(Jx, Fsum)
    sr = stress_residuals(X, down, F, ang, dict(zip(BM.EFFORT_JOINTS, tau)))
    pw = s["pw"]                                     # passive, stress, balance, floor, pitch
    off = (com - cop) - ((com - cop) @ down) * down
    bal = jnp.where(nc > 0, _sqrt0(off @ off) / 0.04, 0.0)
    h = -(C @ down)
    hmean = (h * cm).sum() / jnp.maximum(nc, 1.0)
    floor = (h - hmean) / 0.025 * cm
    return jnp.concatenate([pw[0] * pr, pw[1] * sr, (pw[2] * bal)[None], pw[3] * floor,
                            (pw[4] * _deg(pitch) / 8.0)[None]])


# ============================================================================ full residual
def residuals_full(p, s):
    Xp = p[:3 * J].reshape(J, 3)
    X = Xp[s["src"]]
    log_h, g = p[3 * J], jnp.concatenate([p[3 * J + 1:3 * J + 4], jnp.zeros(1)])
    log_sz, pitch = p[3 * J + 4], p[3 * J + 5]
    res = []
    res.append(((_project(X, s["K"]) - s["k2"]) / s["sig_px"] * s["wr"][:, None]).ravel())
    bi = BONE_IDX
    L = _norm(X[bi[:, 0]] - X[bi[:, 1]])
    Lt = s["bL"] * jnp.exp(log_h + g[s["bg"]])
    res.append((L - Lt) / (s["bS"] * Lt) * s["bmask"])
    res.append(jnp.concatenate([jnp.stack([log_h / s["height_sigma"]]), g[:3] / s["prop_sigma"]]))
    sp = SYM_PAIRS
    res.append((L[sp[:, 0]] - L[sp[:, 1]]) / (0.04 * Lt[sp[:, 0]]) * s["smask"])
    h = hinge_terms(X)
    res.append(s["w_hinge"] * jnp.logaddexp(0.0, -h / 0.02) * 0.02 / 0.01 * s["hinge_ok"])
    wp = s["w_prior"]
    res.append(wp * pose_prior_terms(X))
    res.append(wp * twist_terms(X))
    res.append(wp * collision_terms(X, s["height"], s["coll_ok"]) / 0.03)
    res.append(s["w_order"] * order_terms(X, s))
    res.append(s["phys_on"] * physics_residuals(X, pitch, s))
    res.append(jnp.stack([s["w_bal"] * balance_term(X)]))
    E = s["E"]
    dz = X[E[:, 1], 2] - X[E[:, 0], 2]
    r = (dz - jnp.exp(log_sz) * s["ezr"]) / 0.07
    r2 = r * r
    small = r2 < 1e-12
    cauchy = r * jnp.where(small, 1.0, jnp.sqrt(jnp.log1p(jnp.where(small, 1.0, r2)) / jnp.where(small, 1.0, r2)))
    res.append(s["w_depth"] * s["ew"] * cauchy)
    res.append(jnp.stack([s["use_depth"] * log_sz / 0.3]))
    return jnp.concatenate(res)


_F = _JAC = None


def _compiled():
    global _F, _JAC
    if _F is None:
        _F = jax.jit(residuals_full)
        _JAC = jax.jit(jax.jacfwd(residuals_full))
    return _F, _JAC


# ============================================================================ spec from fit_skeleton
def build_spec(kpts2d, conf, K, height, valid, park, sig_px, use_depth, E=None, ew=None, ezr=None,
               w_hinge=1.0, w_depth=1.0, w_prior=1.0, height_sigma=0.07, prop_sigma=0.10,
               w_balance=1.0, physics=None, order_cons=None, w_order=1.0, hinge_ok=None):
    """Fixed-shape arrays describing one fit_skeleton problem."""
    from .s2_lift import _TREE
    f8 = lambda x: np.asarray(x, np.float64)
    src = np.arange(J)
    for j, nb in park:
        src[j] = nb
    bmask = valid[BONE_IDX[:, 0]] & valid[BONE_IDX[:, 1]]
    from .s2_lift import BONE_GROUP
    s = dict(
        src=src.astype(np.int32), K=f8(K), k2=f8(kpts2d), sig_px=f8(sig_px),
        wr=f8(np.where(valid, np.sqrt(np.clip(conf, 0, None)), 0.0)),
        bL=f8(BONE_FRAC * height), bS=f8(BONE_SIG), bg=BONE_GROUP.astype(np.int32), bmask=f8(bmask),
        smask=f8(bmask[SYM_PAIRS[:, 0]] & bmask[SYM_PAIRS[:, 1]]),
        height_sigma=f8(height_sigma), prop_sigma=f8(prop_sigma),
        w_hinge=f8(w_hinge), hinge_ok=f8(hinge_ok), w_prior=f8(w_prior), height=f8(height),
        coll_ok=valid[_PA].all(1) & valid[_PB].all(1), w_order=f8(w_order),
        w_bal=f8(2.0 * w_balance if (physics is None and w_balance and valid[IDX["lank"]] and valid[IDX["rank"]]) else 0.0),
        use_depth=f8(1.0 if use_depth else 0.0), w_depth=f8(w_depth),
    )
    # depth edges on the full tree (masked)
    Et = np.array([(IDX[a], IDX[b]) for a, b in _TREE], np.int32)
    ewf, ezf = np.zeros(len(Et)), np.zeros(len(Et))
    if use_depth:
        key = {(int(a), int(b)): k for k, (a, b) in enumerate(Et)}
        for (a, b), w_, z_ in zip(E, ew, ezr):
            k = key[(int(a), int(b))]
            ewf[k], ezf[k] = w_, z_
    s.update(E=Et, ew=f8(ewf), ezr=f8(ezf))
    # occlusion constraints (padded)
    oa, ob, ot, okind, ow, om = (np.zeros(MAX_SEG_PTS, np.int32), np.zeros(MAX_SEG_PTS, np.int32),
                                 np.zeros(MAX_SEG_PTS), np.zeros(MAX_SEG_PTS), np.zeros(MAX_SEG_PTS),
                                 np.zeros(MAX_SEG_PTS))
    pa_ = {k: np.zeros(MAX_PAIRS, np.int32) for k in ("pfa", "pfb", "pba", "pbb")}
    pf_ = {k: np.zeros(MAX_PAIRS) for k in ("ptf", "ptb", "pweight", "pmargin", "pm")}
    oa[:] = ob[:] = IDX["pelvis"]
    for k in pa_:
        pa_[k][:] = IDX["pelvis"]
    n = q = 0
    for c in (order_cons or []):
        if c["kind"] == "pair":
            if q >= MAX_PAIRS:
                raise ValueError("too many occlusion pair constraints for s2_jax")
            pa_["pfa"][q], pa_["pfb"][q], pa_["pba"][q], pa_["pbb"][q] = c["fa"], c["fb"], c["ba"], c["bb"]
            pf_["ptf"][q], pf_["ptb"][q], pf_["pweight"][q], pf_["pmargin"][q], pf_["pm"][q] = (
                c["tf"], c["tb"], c["weight"], c["margin"], 1.0)
            q += 1
        else:
            ts = np.asarray(c["ts"], float)
            if n + len(ts) > MAX_SEG_PTS:
                raise ValueError("too many occlusion sample points for s2_jax")
            sl = slice(n, n + len(ts))
            oa[sl], ob[sl], ot[sl] = c["a"], c["b"], ts
            okind[sl] = 1.0 if c["kind"] == "behind" else -1.0
            ow[sl], om[sl] = c["weight"], 1.0
            n += len(ts)
    s.update(oa=oa, ob=ob, ot=ot, okind=okind, ow=ow, om=om, **pa_, **pf_)
    # physics (padded contacts)
    ci, cm = np.full(MAX_CONTACTS, IDX["pelvis"], np.int32), np.zeros(MAX_CONTACTS)
    mask_c = np.zeros((len(BM.EFFORT_JOINTS), MAX_CONTACTS))
    pw = np.zeros(5)
    mass = 0.0
    if physics is not None:
        nci = len(physics.ci)
        if nci > MAX_CONTACTS:
            raise ValueError("too many contacts for s2_jax")
        ci[:nci], cm[:nci] = physics.ci, 1.0
        if nci:
            mask_c[:, :nci] = physics.mask_c
        w = physics.w
        pw = np.array([w["passive"], w["stress"], w["balance"], w["floor"], w["pitch"]], float)
        mass = physics.mass
    s.update(ci=ci, cmask=cm, mask_c=mask_c, pw=pw, mass=f8(mass), phys_on=f8(1.0 if physics is not None else 0.0))
    return s


def make_fun_jac(spec, cols, template):
    """Callables for scipy.optimize.least_squares on the reduced parameter vector."""
    F, JAC = _compiled()
    s = {k: jnp.asarray(v) for k, v in spec.items()}
    cols = np.asarray(cols)

    def embed(p):
        full = template.copy()
        full[cols] = p
        return full

    def fun(p):                       # np.array (copy): SciPy's robust loss scales these in place
        return np.array(F(embed(p), s))

    def jac(p):
        return np.array(JAC(embed(p), s)[:, cols])
    return fun, jac


def warmup():
    """Compile (or load from the persistent cache) the residual and its Jacobian once."""
    if not HAVE_JAX:
        return False
    from .synthetic import lunge_pose
    from .s2_lift import default_K, project
    X = lunge_pose()
    K = default_K((960, 1280))
    valid = np.ones(J, bool)
    spec = build_spec(project(X, K), np.full(J, 0.9), K, 1.70, valid, [], 5.0, False,
                      physics=BM.Physics(["lheel", "rheel"], 1.70), hinge_ok=np.ones(5, bool))
    F, JAC = _compiled()
    p = np.r_[X.ravel(), np.zeros(6)]
    s = {k: jnp.asarray(v) for k, v in spec.items()}
    F(p, s).block_until_ready()
    JAC(p, s).block_until_ready()
    return True

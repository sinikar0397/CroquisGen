"""
Occlusion order from body-part contours (cheap, no extra network).

Joints alone cannot say whether a limb passes in FRONT of the torso or BEHIND it. Contours can:

  * a limb in front of the torso owns its full silhouette: two borders, full expected width;
  * a limb behind it is cut by the torso's contour: on the torso side the visible band ends at
    an edge as strong as the limb's outer border (the occluding contour), so the visible width
    is truncated.
  * if the inner border is weak (same colour on same colour, e.g. a navy sleeve on a navy
    jacket) nothing can be concluded and no constraint is produced.

Cost: one Sobel pass on a downscaled crop of the person box + ~10 cross-section profiles per
arm segment (~ a few ms per image). Verdicts become soft depth-order constraints in the fit:
"behind" -> the segment points lie behind the torso's coronal plane; "front" -> in front of the
chest surface.

Limb-limb crossings (arms, forearms, thighs, shanks, any side): where two limbs cross in the
image, the limb in FRONT keeps both of its borders through the crossing, while the borders of the
limb BEHIND are interrupted (hidden under the front one). Comparing the border continuity of the
two limbs inside the overlap, each normalised by the same limb's border strength outside it,
gives the depth order -> a pairwise residual z_front + margin < z_back at the crossing.
"""
from __future__ import annotations
import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from .skeleton import IDX

# segment: (proximal, distal, radius m) — sleeve/limb half-thickness for a 1.70 m body
SEGMENTS = {
    "r_uarm": ("rsho", "relb", 0.055), "r_farm": ("relb", "rwri", 0.045),
    "l_uarm": ("lsho", "lelb", 0.055), "l_farm": ("lelb", "lwri", 0.045),
}
# limb segments used for limb-limb crossings
LIMBS = dict(SEGMENTS, **{
    "r_thigh": ("rhip", "rknee", 0.075), "r_shank": ("rknee", "rank", 0.055),
    "l_thigh": ("lhip", "lknee", 0.075), "l_shank": ("lknee", "lank", 0.055),
})
ADJACENT = {frozenset(p) for p in (("r_uarm", "r_farm"), ("l_uarm", "l_farm"),
                                   ("r_thigh", "r_shank"), ("l_thigh", "l_shank"))}
# (thighs meet at the crotch at t ~ 0, which the crossing test already ignores)
MAX_SIDE = 800          # px: downscale the person crop for speed


class EdgeImage:
    """Oriented gradient of the person crop (strongest Lab channel per pixel), downscaled."""

    def __init__(self, img_bgr, bbox, pad=0.15):
        h, w = img_bgr.shape[:2]
        x0, y0, x1, y1 = bbox
        px, py = (x1 - x0) * pad, (y1 - y0) * pad
        self.x0, self.y0 = int(max(0, x0 - px)), int(max(0, y0 - py))
        x1, y1 = int(min(w, x1 + px)), int(min(h, y1 + py))
        crop = img_bgr[self.y0:y1, self.x0:x1]
        self.s = min(1.0, MAX_SIDE / max(crop.shape[:2]))
        if self.s < 1.0:
            crop = cv2.resize(crop, None, fx=self.s, fy=self.s, interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab = cv2.GaussianBlur(lab, (3, 3), 0)
        gx = np.zeros(lab.shape[:2], np.float32); gy = gx.copy(); best = gx.copy()
        for c in range(3):
            sx = cv2.Sobel(lab[..., c], cv2.CV_32F, 1, 0, ksize=3)
            sy = cv2.Sobel(lab[..., c], cv2.CV_32F, 0, 1, ksize=3)
            m = np.abs(sx) + np.abs(sy)
            take = m > best
            gx, gy, best = np.where(take, sx, gx), np.where(take, sy, gy), np.where(take, m, best)
        self.gx, self.gy = gx, gy
        self.p90 = float(np.percentile(best, 90)) + 1e-6

    def to_crop(self, p):
        return (np.asarray(p, float) - [self.x0, self.y0]) * self.s

    def profile(self, p, n, s):
        """|gradient . n| sampled at p + s*n (crop coords)."""
        pts = p[None] + s[:, None] * n[None]
        vx = map_coordinates(self.gx, [pts[:, 1], pts[:, 0]], order=1, mode="nearest")
        vy = map_coordinates(self.gy, [pts[:, 1], pts[:, 0]], order=1, mode="nearest")
        return np.abs(vx * n[0] + vy * n[1])


def torso_hull_2d(X, K, height):
    """2D outline of the torso volume (elliptic cylinder around pelvis->neck) for the current pose."""
    from .s2_lift import project
    i = IDX
    pel, nk = X[i["pelvis"]], X[i["neck"]]
    up = nk - pel; Lt = np.linalg.norm(up); up = up / (Lt + 1e-9)
    r = (X[i["rsho"]] - X[i["lsho"]]) + (X[i["rhip"]] - X[i["lhip"]])
    r = r - (r @ up) * up; r = r / (np.linalg.norm(r) + 1e-9)
    f = np.cross(up, r)
    k = height / 1.70
    a, b = 0.16 * k, 0.11 * k                      # half chest width / half depth
    th = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    ring = np.cos(th)[:, None] * r * a + np.sin(th)[:, None] * f * b
    P = np.vstack([pel + h * up * Lt + ring for h in (0.0, 0.5, 1.0)])
    uv = project(P, K).astype(np.float32)
    return cv2.convexHull(uv).reshape(-1, 2)


def _inside(poly, p, margin):
    return cv2.pointPolygonTest(poly.astype(np.float32), (float(p[0]), float(p[1])), True) > -margin


def segment_evidence(E, k2, X, K, height, seg, hull):
    """Cross-section analysis of one limb segment against the torso. Returns dict or None."""
    a, b, r_m = SEGMENTS[seg]
    A, B = k2[IDX[a]], k2[IDX[b]]
    Ac, Bc = E.to_crop(A), E.to_crop(B)
    d = Bc - Ac; L = np.linalg.norm(d)
    if L < 8:
        return None
    u = d / L; n = np.array([-u[1], u[0]])
    cen = E.to_crop((k2[IDX["neck"]] + k2[IDX["pelvis"]]) / 2)
    if (cen - Ac) @ n < 0:
        n = -n                                       # n points toward the torso
    z = 0.5 * (X[IDX[a], 2] + X[IDX[b], 2])
    r = K[0, 0] * r_m * (height / 1.70) / max(z, 0.3) * E.s
    if r < 3:
        return None
    s = np.linspace(-2.2 * r, 2.2 * r, 89)
    hc = E.to_crop(hull).astype(np.float32)
    rows = []
    for t in np.linspace(0.2, 0.9, 8):
        pc = Ac + t * d
        dc = cv2.pointPolygonTest(hc, (float(pc[0]), float(pc[1])), True) / r      # + = inside torso
        pr = E.profile(pc, n, s)
        out = s < -0.25 * r
        so, eo = s[out][np.argmax(pr[out])], pr[out].max() / E.p90
        sel = s > so + 0.4 * r
        si, ei = s[sel][np.argmax(pr[sel])], pr[sel].max() / E.p90
        de = cv2.pointPolygonTest(hc, (float(pc[0] + si * n[0]), float(pc[1] + si * n[1])), True) / r
        width = (si - so) / (2 * r)
        valid = dc > -1.5
        # cut by the torso: strong inner edge ON the torso outline truncates the limb's width
        truncated = valid and width < 0.65 and ei >= max(0.6 * eo, 1.5) and abs(de) <= 1.0
        # in front of the torso: limb well inside the torso region, full width, both borders
        full = dc > 0.5 and 0.8 <= width <= 1.6 and min(eo, ei) >= 1.0
        rows.append((t, valid, truncated, full, width, eo, ei, dc, de))
    rows = np.array(rows, dtype=float)
    v, tr, fu = rows[:, 1] > 0, rows[:, 2] > 0, rows[:, 3] > 0
    inside = rows[:, 7] > 0.5
    verdict, conf, ts = None, 0.0, []
    if tr.sum() >= 2 and tr.sum() >= 0.3 * max(v.sum(), 1) and tr.sum() > fu.sum():
        verdict, conf, ts = "behind", float(min(1.0, tr.sum() / 3)), rows[tr, 0]
    elif fu.sum() >= 3 and fu.sum() >= 0.6 * max(inside.sum(), 1) and tr.sum() == 0:
        verdict, conf, ts = "front", float(min(1.0, fu.sum() / 4)), rows[fu, 0]
    return dict(seg=seg, verdict=verdict, confidence=conf, ts=list(map(float, ts)),
                n_truncated=int(tr.sum()), n_full=int(fu.sum()), n_near=int(v.sum()), rows=rows)


def _radius_px(E, K, X, height, seg):
    a, b, r_m = LIMBS[seg]
    z = 0.5 * (X[IDX[a], 2] + X[IDX[b], 2])
    return K[0, 0] * r_m * (height / 1.70) / max(z, 0.3) * E.s


def _continuity(E, P0, P1, r, Q0, Q1, rq):
    """Border continuity of limb P (crop coords P0->P1, half-width r px) inside its overlap with
    limb Q, relative to P's own border strength outside the overlap. None if undecidable."""
    d = P1 - P0; L = np.linalg.norm(d); u = d / L; n = np.array([-u[1], u[0]])
    dq = Q1 - Q0; Lq = np.linalg.norm(dq); uq = dq / Lq; nq = np.array([-uq[1], uq[0]])
    s = np.linspace(-1.9 * r, 1.9 * r, 77)
    ts = np.linspace(0.08, 0.95, max(24, int(L / max(r * 0.35, 1.0))))
    C = P0[None] + ts[:, None] * d[None]
    dist_q = np.abs((C - Q0) @ nq)
    along_q = (C - Q0) @ uq
    outside = (dist_q > rq + 1.2 * r) | (along_q < -r) | (along_q > Lq + r)
    if outside.sum() < 3:
        return None
    # border offsets and strength from outside cross-sections
    offs, refs = {+1: [], -1: []}, {+1: [], -1: []}
    for c in C[outside][:: max(1, outside.sum() // 8)]:
        pr = E.profile(c, n, s)
        for sg in (+1, -1):
            m = (sg * s > 0.5 * r) & (sg * s < 1.7 * r)
            k = np.argmax(np.where(m, pr, -1))
            offs[sg].append(s[k]); refs[sg].append(pr[k])
    res = []
    for sg in (+1, -1):
        o, ref = float(np.median(offs[sg])), float(np.median(refs[sg]))
        if ref < 0.7 * E.p90:                         # this border is invisible anyway
            continue
        B = C + o * n[None]                           # points on this border line
        db = np.abs((B - Q0) @ nq); ab = (B - Q0) @ uq
        inside = (db < 0.65 * rq) & (ab > 0) & (ab < Lq)
        if inside.sum() < 3:
            continue
        loc = np.linspace(o - 0.3 * r, o + 0.3 * r, 9)
        vals = [E.profile(c, n, loc).max() for c in C[inside]]
        res.append(float(np.median(vals)) / ref)
    return float(np.mean(res)) if res else None


def pair_evidence(E, k2, X, K, height, sa, sb):
    """Depth order of two limbs crossing in the image. dict(front, back, t_front, t_back, ...)."""
    (a0, a1, _), (b0, b1, _) = LIMBS[sa], LIMBS[sb]
    A0, A1, B0, B1 = (E.to_crop(k2[IDX[j]]) for j in (a0, a1, b0, b1))
    dA, dB = A1 - A0, B1 - B0
    M = np.array([dA, -dB]).T
    if abs(np.linalg.det(M)) < 1e-6 or min(np.linalg.norm(dA), np.linalg.norm(dB)) < 10:
        return None
    t, u = np.linalg.solve(M, B0 - A0)
    if not (0.05 < t < 1.0 and 0.05 < u < 1.0):
        return None
    cosang = abs(dA @ dB) / (np.linalg.norm(dA) * np.linalg.norm(dB))
    if cosang > np.cos(np.radians(15)):              # near-parallel: overlap ill-defined
        return None
    ra, rb = _radius_px(E, K, X, height, sa), _radius_px(E, K, X, height, sb)
    if min(ra, rb) < 3:
        return None
    ca = _continuity(E, A0, A1, ra, B0, B1, rb)
    cb = _continuity(E, B0, B1, rb, A0, A1, ra)
    out = dict(seg=f"{sa}/{sb}", verdict=None, confidence=0.0, ca=ca, cb=cb, t=float(t), u=float(u))
    if ca is None or cb is None:
        return out
    if ca >= 0.55 and cb <= 0.5 and ca - cb >= 0.25:
        front, back, tf, tb, conf = sa, sb, t, u, ca - cb
    elif cb >= 0.55 and ca <= 0.5 and cb - ca >= 0.25:
        front, back, tf, tb, conf = sb, sa, u, t, cb - ca
    else:
        return out
    out.update(verdict=f"{front} in front", front=front, back=back, t_front=float(tf), t_back=float(tb),
               confidence=float(min(1.0, conf / 0.5)))
    return out


def occlusion_constraints(img_bgr, bbox, k2, conf, X, K, height, edges=None):
    """Depth-order constraints for the fit: list of dict(seg, a, b, ts, kind, weight)."""
    if img_bgr is None or bbox is None:
        return [], []
    E = edges or EdgeImage(img_bgr, bbox)
    hull = torso_hull_2d(X, K, height)
    cons, report = [], []
    for seg, (a, b, _) in SEGMENTS.items():
        if min(conf[IDX[a]], conf[IDX[b]]) < 0.3:
            continue
        ev = segment_evidence(E, k2, X, K, height, seg, hull)
        if ev is None:
            continue
        report.append({k: v for k, v in ev.items() if k != "rows"})
        if ev["verdict"] and ev["ts"]:
            cons.append(dict(seg=seg, a=IDX[a], b=IDX[b], ts=np.array(ev["ts"]), kind=ev["verdict"],
                             weight=float(ev["confidence"])))
    # limb-limb crossings
    names = [n for n, (a, b, _) in LIMBS.items() if min(conf[IDX[a]], conf[IDX[b]]) >= 0.3]
    for i_, sa in enumerate(names):
        for sb in names[i_ + 1:]:
            if frozenset((sa, sb)) in ADJACENT:
                continue
            ev = pair_evidence(E, k2, X, K, height, sa, sb)
            if ev is None:
                continue
            report.append(ev)
            if ev["verdict"]:
                f, b = LIMBS[ev["front"]], LIMBS[ev["back"]]
                margin = 0.5 * (f[2] + b[2]) * height / 1.70
                cons.append(dict(seg=ev["seg"], kind="pair", weight=ev["confidence"], margin=margin,
                                 fa=IDX[f[0]], fb=IDX[f[1]], tf=ev["t_front"],
                                 ba=IDX[b[0]], bb=IDX[b[1]], tb=ev["t_back"]))
    return cons, report


def order_terms(X, cons, sigma=0.03, chest=0.08):
    """Soft depth-order residuals (camera frame, z forward). 'behind': segment points lie behind
    the torso's coronal plane along their pixel rays; 'front': in front of the chest surface."""
    if not cons:
        return np.zeros(0)
    i = IDX
    pel, nk = X[i["pelvis"]], X[i["neck"]]
    up = nk - pel
    lat = X[i["rsho"]] - X[i["lsho"]]
    nrm = np.cross(up, lat); nrm = nrm / (np.linalg.norm(nrm) + 1e-9)
    c0 = 0.5 * (pel + nk)
    out = []
    for c in cons:
        if c["kind"] == "pair":                                 # front limb closer at the crossing
            zf = X[c["fa"], 2] + c["tf"] * (X[c["fb"], 2] - X[c["fa"], 2])
            zb = X[c["ba"], 2] + c["tb"] * (X[c["bb"], 2] - X[c["ba"], 2])
            out.append(np.array([c["weight"] * np.logaddexp(0, (zf + c["margin"] - zb) / sigma)]))
            continue
        P = X[c["a"]][None] + c["ts"][:, None] * (X[c["b"]] - X[c["a"]])[None]
        v = P / P[:, 2:3]                                       # pixel rays with v_z = 1
        den = v @ nrm
        ok = np.abs(den) > 0.15                                 # plane not edge-on to the ray
        lam = np.where(ok, (nrm @ c0) / np.where(ok, den, 1.0), P[:, 2])   # plane depth on the ray
        if c["kind"] == "behind":
            x = (lam - P[:, 2]) / sigma
        else:
            x = (P[:, 2] - (lam - chest)) / sigma
        out.append(np.where(ok, c["weight"] * np.logaddexp(0, x), 0.0))
    return np.concatenate(out)

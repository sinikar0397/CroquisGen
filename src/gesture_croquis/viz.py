"""Visualisation of S1 + S2 results: interactive HTML (plotly, works offline) and a static PNG."""
from __future__ import annotations
import base64
import cv2
import numpy as np

from .skeleton import JOINTS, IDX, DRAW_BONES, side
from .s2_lift import project

COL = {"L": "#2a6fdb", "R": "#e8692c", "C": "#444444"}          # left blue, right orange
COL_BGR = {"L": (219, 111, 42), "R": (44, 105, 232), "C": (68, 68, 68)}


def _bone_side(a, b):
    sa, sb = side(a), side(b)
    return sa if sa == sb else ("L" if "L" in (sa, sb) else "R" if "R" in (sa, sb) else "C")


def _bones(frame):
    """Bones to draw; the straight neck-pelvis/head lines are replaced by the spine curve."""
    if frame.spine3d is None:
        return DRAW_BONES
    return [b for b in DRAW_BONES if b not in (("neck", "pelvis"), ("head", "neck"))]


def effort_points(frame):
    """[(name, world xyz, effort 0..1, N·m)] for joints + spine landmarks."""
    if not frame.joint_effort:
        return []
    from .spine import spine_curve
    W = frame.world_joints()
    lm = spine_curve(W)["landmarks"]
    out = []
    for j, e in frame.joint_effort.items():
        p = W[IDX[j]] if j in IDX else lm.get(j)
        if p is not None:
            out.append((j, p, e, frame.joint_torque.get(j, {}).get("norm", 0.0)))
    return out


def draw_2d(img_bgr, frame, show_reproj=True, thr=0.05):
    """Photo with S1 keypoints (dots) and the S2 skeleton reprojected (lines)."""
    out = img_bgr.copy()
    s = max(1, int(round(max(out.shape[:2]) / 600)))
    k, c = frame.kpts2d, frame.conf
    if frame.bbox is not None:
        x0, y0, x1, y1 = map(int, frame.bbox)
        cv2.rectangle(out, (x0, y0), (x1, y1), (120, 200, 120), s, cv2.LINE_AA)
    if show_reproj and frame.joints3d is not None:
        uv = project(frame.joints3d, frame.camera_K)
        if frame.spine3d is not None:
            cv2.polylines(out, [np.int32(project(frame.spine3d, frame.camera_K))], False,
                          COL_BGR["C"], 2 * s, cv2.LINE_AA)
        for a, b in _bones(frame):
            if frame.valid3d[IDX[a]] and frame.valid3d[IDX[b]]:
                cv2.line(out, tuple(np.int32(uv[IDX[a]])), tuple(np.int32(uv[IDX[b]])),
                         COL_BGR[_bone_side(a, b)], 2 * s, cv2.LINE_AA)
    for j, n in enumerate(JOINTS):
        if c[j] > thr:
            cv2.circle(out, tuple(np.int32(k[j])), int(2 * s + 3 * s * c[j]), (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(out, tuple(np.int32(k[j])), int(2 * s + 3 * s * c[j]), COL_BGR[side(n)], s, cv2.LINE_AA)
    return out


# ----------------------------------------------------------------------------------------
def _plotly_xyz(W):
    """world (x right, y up, z toward camera) -> plotly (x, y = depth away from camera, z = up)."""
    return W[:, 0], -W[:, 2], W[:, 1]


def make_html(frame, img_bgr, path, title="CroquisGen S1+S2"):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    W = frame.world_joints()
    v = frame.valid3d
    X, Y, Z = _plotly_xyz(W)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.38, 0.62],
                        specs=[[{"type": "xy"}, {"type": "scene"}]],
                        subplot_titles=("S1 keypoints · S2 reprojection", "S2 3D skeleton (metres)"))

    # ---- left: image -------------------------------------------------------------------
    vis = draw_2d(img_bgr, frame)
    h, w = vis.shape[:2]
    sc = min(1.0, 900 / max(h, w))
    small = cv2.resize(vis, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
    fig.add_trace(go.Image(source="data:image/jpeg;base64," + base64.b64encode(buf).decode(),
                           dx=1 / sc, dy=1 / sc, hoverinfo="skip"), 1, 1)
    m = frame.conf > 0.05
    fig.add_trace(go.Scatter(x=frame.kpts2d[m, 0], y=frame.kpts2d[m, 1], mode="markers",
                             marker=dict(size=6, color="rgba(0,0,0,0)"),
                             text=[f"{JOINTS[j]}  conf {frame.conf[j]:.2f}" for j in np.where(m)[0]],
                             hoverinfo="text", showlegend=False), 1, 1)
    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_yaxes(visible=False, row=1, col=1, scaleanchor="x", autorange="reversed")

    # ---- right: 3D -------------------------------------------------------------------
    for sname, label in (("L", "left"), ("R", "right"), ("C", "centre")):
        xs, ys, zs = [], [], []
        for a, b in _bones(frame):
            if _bone_side(a, b) == sname and v[IDX[a]] and v[IDX[b]]:
                for q in (a, b):
                    xs.append(X[IDX[q]]); ys.append(Y[IDX[q]]); zs.append(Z[IDX[q]])
                xs.append(None); ys.append(None); zs.append(None)
        if sname == "C" and frame.spine3d is not None:
            sx, sy, sz = _plotly_xyz(frame.to_world(frame.spine3d))
            xs += list(sx); ys += list(sy); zs += list(sz)
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=label,
                                   line=dict(color=COL[sname], width=7), hoverinfo="skip"), 1, 2)
    jj = np.where(v)[0]
    fig.add_trace(go.Scatter3d(
        x=X[jj], y=Y[jj], z=Z[jj], mode="markers", name="joints", showlegend=False,
        marker=dict(size=3 + 4 * frame.conf[jj], color=[COL[side(JOINTS[j])] for j in jj],
                    line=dict(color="white", width=1)),
        text=[f"{JOINTS[j]}<br>conf {frame.conf[j]:.2f}<br>world ({W[j,0]:+.2f}, {W[j,1]:+.2f}, {W[j,2]:+.2f}) m"
              for j in jj], hoverinfo="text"), 1, 2)

    # ground grid (y = 0 plane) + camera direction arrow + gravity
    cx, cz = np.median(X[jj]), np.median(Y[jj])
    R = max(1.0, 0.6 * np.ptp(np.c_[X[jj], Y[jj]], axis=0).max() + 0.5)
    g = np.linspace(-R, R, 9)
    gx, gy, gz = [], [], []
    for t in g:
        gx += [cx + t, cx + t, None, cx - R, cx + R, None]
        gy += [cz - R, cz + R, None, cz + t, cz + t, None]
        gz += [0, 0, None, 0, 0, None]
    fig.add_trace(go.Scatter3d(x=gx, y=gy, z=gz, mode="lines", line=dict(color="#cccccc", width=2),
                               name="ground", hoverinfo="skip"), 1, 2)
    pel = np.array([X[IDX["pelvis"]], Y[IDX["pelvis"]], Z[IDX["pelvis"]]])
    cam_w = -frame.joints3d[IDX["pelvis"]] * np.array([1, -1, -1])      # camera centre, pelvis-relative
    d = np.array([cam_w[0], -cam_w[2], cam_w[1]]); d = d / np.linalg.norm(d) * 0.6
    fig.add_trace(go.Scatter3d(x=[pel[0], pel[0] + d[0]], y=[pel[1], pel[1] + d[1]], z=[pel[2], pel[2] + d[2]],
                               mode="lines+text", text=["", "to camera"], textposition="top center",
                               line=dict(color="#2ca02c", width=4, dash="dash"), name="camera direction",
                               hoverinfo="skip"), 1, 2)

    _add_s3_3d(fig, frame)

    info = frame.s2_info
    prop = info.get("proportions") or {}
    sub = (f"backend: {frame.s2_backend} · height prior {frame.body_height_m:.2f} m"
           + (f" → fitted {info['fitted_height_m']:.2f} m (legs ×{prop.get('legs', 1):.2f}, arms ×{prop.get('arms', 1):.2f}, "
              f"torso ×{prop.get('torso', 1):.2f})" if "fitted_height_m" in info else "") + " · "
           f"subject ≈ {info.get('subject_distance_m', float('nan')):.2f} m from camera · "
           f"reprojection median {info.get('reproj_px_median', float('nan')):.1f}px · "
           f"bone-length error median {info.get('bone_err_pct_median', float('nan')):.1f}%")
    if "z_sign" in info and info["z_sign"]:
        sub += f" · depth sign {info['z_sign']:+d} (×{info.get('z_sign_cost_ratio', 0):.1f} vs flipped)"
    if "physics" in info:
        ph = info["physics"]
        flips = ", ".join(info.get("flips", {})) or "none"
        worst = ", ".join(f"{k} {v:+.1f}σ" for k, v in list(ph["worst_joints"].items())[:3])
        sub += (f"<br>S2+S3 fit · labels: {info.get('label_hypothesis')} · depth flips: {flips} · "
                f"energy margin ×{info.get('energy_margin', 0):.2f} · passive {ph['passive']:.1f} · muscle stress {ph.get('stress_index', 0):.2f} · "
                f"balance {ph.get('balance', 0):.2f} · camera pitch {ph['pitch_deg']:+.1f}° · least relaxed: {worst}")
        occ = info.get("occlusion") or {}
        if occ:
            sub += " · contours: " + ", ".join(v if "/" in k else f"{k} {v} torso" for k, v in occ.items())
    if frame.s3_info:
        i3, ms = frame.s3_info, frame.muscle_stretch
        off = i3.get("com_offset_m")
        bal = ("airborne" if off is None else "over support" if i3["balanced"]
               else "%.2f m outside support (moving/falling)" % off)
        side_name = "left" if ms["compressed_torso_side"] == "l" else "right"
        sub += (f"<br>S3: mass {i3['body_mass_kg']:.0f} kg · contacts {', '.join(frame.contacts) or 'none (airborne)'} · "
                f"COM {bal} · "
                + (f"muscle stress index {i3['stress_index']:.2f} (peak {i3['peak_stress_muscle'].replace('_', ' ')} "
                   f"{i3['stress_peak']:.2f}) · " if "stress_index" in i3 else f"peak effort {i3['peak_effort_joint']} · ")
                + f"spine bend {ms['spine_lateral_bend']:+.0f}° twist {ms['spine_twist']:+.0f}° · "
                f"compressed torso side: {side_name}")
    fig.update_layout(
        title=dict(text=f"{title}<br><sup>{sub}</sup>", x=0.01),
        height=800, margin=dict(l=10, r=10, t=130, b=10), paper_bgcolor="white",
        legend=dict(x=0.40, y=0.02, bgcolor="rgba(255,255,255,0.7)"),
        scene=dict(aspectmode="data",
                   xaxis=dict(title="x (right)", backgroundcolor="white", gridcolor="#eee"),
                   yaxis=dict(title="depth (away from camera)", backgroundcolor="white", gridcolor="#eee"),
                   zaxis=dict(title="up", backgroundcolor="white", gridcolor="#eee"),
                   camera=dict(eye=dict(x=0.0, y=-1.9, z=0.5), up=dict(x=0, y=0, z=1))),
    )
    fig.write_html(path, include_plotlyjs=True, full_html=True)
    return path


def _add_s3_3d(fig, frame):
    import plotly.graph_objects as go
    pts = effort_points(frame)
    if not pts:
        return
    P = np.stack([p for _, p, _, _ in pts]); x, y, z = _plotly_xyz(P)
    e = np.array([e for _, _, e, _ in pts])
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode="markers", name="muscle stress (gravity + length)",
        marker=dict(size=6 + 22 * e, color=e, colorscale="YlOrRd", cmin=0, cmax=1, opacity=0.55,
                    colorbar=dict(title="stress", x=1.0, len=0.5, thickness=12)),
        text=[f"{n}<br>stress {ee:.2f}<br>|τ| {t:.0f} N·m" for n, _, ee, t in pts], hoverinfo="text"), 1, 2)
    com = frame.to_world(frame.com3d[None])[0]
    cx, cy, cz = _plotly_xyz(np.stack([com, com * [1, 0, 1]]))
    fig.add_trace(go.Scatter3d(x=cx, y=cy, z=cz, mode="lines+markers", name="centre of mass + plumb line",
                               marker=dict(size=[7, 3], color="black", symbol=["diamond", "circle"]),
                               line=dict(color="black", width=3, dash="dot"), hoverinfo="skip"), 1, 2)
    R = frame.s3_info.get("reactions_N", {})
    if R:
        W = frame.world_joints()
        C = np.stack([W[IDX[c]] for c in R]); x, y, z = _plotly_xyz(C)
        f = np.array(list(R.values())); f = f / max(f.max(), 1e-9)
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="markers", name="ground contact (size = reaction)",
                                   marker=dict(size=4 + 12 * f, color="#2ca02c", symbol="square", opacity=0.8),
                                   text=[f"{c}: {r:.0f} N" for c, r in R.items()], hoverinfo="text"), 1, 2)


# ----------------------------------------------------------------------------------------
def make_png(frame, img_bgr, path, title=""):
    """Static overview: photo overlay + front / side / top orthographic views."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    W = frame.world_joints(); v = frame.valid3d
    fig = plt.figure(figsize=(16, 5.2), dpi=110)
    ax0 = fig.add_subplot(1, 4, 1)
    ax0.imshow(cv2.cvtColor(draw_2d(img_bgr, frame), cv2.COLOR_BGR2RGB)); ax0.axis("off")
    ax0.set_title("S1 keypoints + S2 reprojection", fontsize=10)
    views = [("front (camera view)", 0, 1, False), ("side (seen from camera right)", 2, 1, True), ("top", 0, 2, False)]
    for k, (name, a, b, flip) in enumerate(views):
        ax = fig.add_subplot(1, 4, k + 2)
        ax.axhline(0, color="#ccc", lw=1) if b == 1 else None
        if frame.spine3d is not None:
            S = frame.to_world(frame.spine3d)
            ax.plot(-S[:, a] if flip else S[:, a], S[:, b], color=COL["C"], lw=2.5)
        for name_, P_, e_, _ in effort_points(frame):
            ax.scatter(-P_[a] if flip else P_[a], P_[b], s=30 + 500 * e_, color="#d62728",
                       alpha=0.18 + 0.5 * e_, lw=0, zorder=0)
        if frame.com3d is not None:
            c = frame.to_world(frame.com3d[None])[0]
            ca = -c[a] if flip else c[a]
            ax.plot([ca, ca] if b == 1 else [ca], [c[b], 0] if b == 1 else [c[b]], "k:", marker="D", ms=5)
        for p, q in _bones(frame):
            i, j = IDX[p], IDX[q]
            if v[i] and v[j]:
                xa, xb = W[i, a], W[j, a]
                if flip:
                    xa, xb = -xa, -xb
                ax.plot([xa, xb], [W[i, b], W[j, b]], color=COL[_bone_side(p, q)], lw=2.5,
                        solid_capstyle="round")
        ax.set_aspect("equal"); ax.set_title(name, fontsize=10)
        ax.tick_params(labelsize=8); ax.grid(alpha=0.25)
        if b == 2:
            ax.set_ylabel("z (toward camera) m", fontsize=8)
    info = frame.s2_info
    fig.suptitle(title or f"{frame.s2_backend} · reproj {info.get('reproj_px_median', 0):.1f}px · "
                 f"bones {info.get('bone_err_pct_median', 0):.1f}% · dist {info.get('subject_distance_m', 0):.2f} m",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path); plt.close(fig)
    return path

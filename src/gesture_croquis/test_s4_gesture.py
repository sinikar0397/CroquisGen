"""S4 checks on synthetic poses (no models needed):  python -m pytest gesture_croquis/test_s4_gesture.py -q"""
import json
import numpy as np
import pytest

from gesture_croquis.poseframe import PoseFrame
from gesture_croquis.skeleton import J, IDX
from gesture_croquis.synthetic import lunge_pose, stand_pose
from gesture_croquis.s3_load import static_load
from gesture_croquis.s4_gesture import (build_croquis, View, default_K, fit_cubic_bezier, bezier,
                                        render_preview, preview_panel)


def make_frame(X, contacts=None, hw=(960, 1280), H=1.70, s3=True):
    K = default_K(hw)
    uv = X[:, :2] / X[:, 2:3] * K[0, 0] + K[:2, 2]
    f = PoseFrame(img_hw=hw, kpts2d=uv.astype(np.float32), conf=np.full(J, 0.9, np.float32))
    f.joints3d, f.valid3d, f.camera_K, f.body_height_m = X, np.ones(J, bool), K, H
    if s3:
        static_load(f, contacts=contacts)
    return f


def mean_w(c, name):
    return float(c.get(name).width.mean())


def test_bezier_fit_recovers_cubic():
    ctrl = np.array([[0, 0], [30, 80], [90, -40], [120, 10.0]])
    ctrl_hat, resid = fit_cubic_bezier(bezier(ctrl, 50))
    assert resid < 4e-3                                                 # < 0.4 % of arc length
    a, b = bezier(ctrl_hat, 200), bezier(ctrl, 2000)                      # same shape (parametrisation may differ)
    assert np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1)).min(1).max() < 1.0


def test_neutral_stance_loa_is_head_to_leg():
    c = build_croquis(make_frame(stand_pose()))
    assert c.loa["name"].startswith("head->"), c.loa["candidates"][:3]   # hanging arm folds back: no hand chain


def test_lunge_loa_and_weight_order():
    c = build_croquis(make_frame(lunge_pose()))
    loa = c.get("line_of_action")
    assert loa.width.max() == max(s.width.max() for s in c.strokes)       # LOA is the heaviest line
    assert any(r["name"] == c.loa["name"] for r in c.loa["candidates"])


def test_loa_endpoints_stay_on_the_skeleton():
    f = make_frame(lunge_pose())
    c = build_croquis(f, exaggeration=0.5)
    ctrl = np.array(c.loa["ctrl"]); ctrl_x = np.array(c.loa["ctrl_exaggerated"])
    assert np.allclose(ctrl[[0, 3]], ctrl_x[[0, 3]])                    # exaggeration bends, never moves ends
    end = c.loa["joints"][-1]
    uv = f.joints3d[IDX[end], :2] / f.joints3d[IDX[end], 2] * f.camera_K[0, 0] + f.camera_K[:2, 2]
    assert np.linalg.norm(c.get("line_of_action").pts[-1] - uv) < 1.0


def test_side_bend_squashes_the_left_side():
    c = build_croquis(make_frame(stand_pose(side_bend_deg=20)))
    l, r = c.get("torso_l"), c.get("torso_r")
    assert c.info["torso"]["squashed_side"] == "left"
    assert l.role == "torso_comp" and l.meta["bow_px"] > 2 * max(r.meta["bow_px"], 1.0)
    assert mean_w(c, "torso_l") > mean_w(c, "torso_r")


def test_weight_bearing_leg_is_heavier():
    f = make_frame(stand_pose(), contacts=["lheel", "lbigtoe", "lsmalltoe"])
    c = build_croquis(f)
    assert c.info["support"]["lleg"] > 0.99
    legs = {s.name: s for s in c.strokes if s.name.endswith("_leg")}
    # the support leg may be absorbed into the LOA; compare like with like
    if legs["l_leg"].role == legs["r_leg"].role:
        assert mean_w(c, "l_leg") > mean_w(c, "r_leg")
    else:
        assert c.loa["name"].endswith("L leg")


def test_novel_views():
    f = make_frame(lunge_pose())
    c0 = build_croquis(f)
    c360 = build_croquis(f, View(360.0))
    assert np.allclose(c0.get("line_of_action").pts, c360.get("line_of_action").pts, atol=0.5)
    for v in (View(90), View(-45, 20), View(180)):
        c = build_croquis(f, v, layers="ABC")
        assert all(np.isfinite(s.pts).all() and np.isfinite(s.width).all() for s in c.strokes)
        assert c.view.tag


def test_overlap_cues_thin_the_stroke_behind():
    f = make_frame(lunge_pose())
    found = False
    for yaw in (0, 45, 90, 135, -45, -90):
        c = build_croquis(f, View(yaw))
        for s in c.strokes:
            if s.behind is not None and s.behind.max() > 0.99:
                found = True
                k = int(np.argmax(s.behind))
                assert s.width[k] < 0.5 * s.width.max() + 1e-6
    assert found


def test_2d_fallback_without_s2():
    f = make_frame(lunge_pose(), s3=False)
    f.joints3d = None
    c = build_croquis(f)
    assert c.info["mode"] == "2d" and c.get("line_of_action") is not None
    with pytest.raises(ValueError):
        build_croquis(f, View(90))


def test_layers_json_and_render(tmp_path):
    f = make_frame(lunge_pose())
    c = build_croquis(f, layers="ABC")
    assert {s.layer for s in c.strokes} == {"A", "B", "C"}
    assert {s.layer for s in build_croquis(f, layers="A").strokes} == {"A"}
    p = tmp_path / "c.json"
    c.save(str(p))
    d = json.loads(p.read_text())
    assert d["loa"]["name"] == c.loa["name"] and len(d["strokes"]) == len(c.strokes)
    img = render_preview(c, (400, 300))
    assert img.shape == (400, 300, 3) and img.min() < 100                # something was drawn
    assert preview_panel(c, np.full((960, 1280, 3), 200, np.uint8), 300).shape[0] == 300


def test_batched_curve_fit_equals_single():
    """All line-of-action candidates fitted in one pass = each fitted alone."""
    from gesture_croquis.s4_gesture import fit_cubic_bezier_batch
    rng = np.random.default_rng(0)
    curves = [bezier(rng.normal(0, 50, (4, 2)), 64) + rng.normal(0, 1.0, (64, 2)) for _ in range(7)]
    CT, RS = fit_cubic_bezier_batch(np.stack(curves))
    for c, ct, rs in zip(curves, CT, RS):
        ctrl, resid = fit_cubic_bezier(c)
        assert np.allclose(ct, ctrl, atol=1e-9) and abs(rs - resid) < 1e-12

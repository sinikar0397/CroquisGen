"""Live (video) pipeline: ROI tracking, hysteresis, tracked S2.  python -m pytest gesture_croquis/test_live.py -q"""
import numpy as np

from gesture_croquis.synthetic import lunge_pose, stand_pose, procrustes
from gesture_croquis.s2_lift import default_K, project
from gesture_croquis.skeleton import J, IDX
from gesture_croquis.live import LivePipeline, LiveConfig, OneEuro, ROITracker
from gesture_croquis.biomech import choose_contacts

HW = (720, 1280)
K = default_K(HW)


def _rot_y(X, deg):
    a = np.radians(deg)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    p = X[IDX["pelvis"]]
    return (X - p) @ R.T + p


def _seq(n):
    out = []
    for i in range(n):
        a = 0.5 - 0.5 * np.cos(2 * np.pi * i / 40)
        X = (1 - a) * stand_pose(pelvis_cam=(0.1, 0.05, 4.0)) + a * lunge_pose(yaw_deg=0, pelvis_cam=(0.1, 0.05, 4.0))
        X = _rot_y(X, 40 * np.sin(2 * np.pi * i / n))
        feet = [IDX[k] for k in ("lheel", "lbigtoe", "lsmalltoe", "rheel", "rbigtoe", "rsmalltoe")]
        if out:                                  # physical: the body lowers, the floor stays put
            X = X + np.array([0.0, floor_y - X[feet, 1].max(), 0.0])
        else:
            floor_y = X[feet, 1].max()
        out.append(X)
    return out


class FakeS1:
    def __init__(self, seq):
        self.seq, self.i, self.calls = seq, 0, {"detect": 0, "roi": 0}

    def __call__(self, img, bbox=None):
        uv = project(self.seq[self.i], K) + np.random.default_rng(self.i).normal(0, 2.0, (J, 2))
        conf = np.full(J, 0.9)
        if bbox is None:
            self.calls["detect"] += 1
            box = np.r_[uv.min(0) - 30, uv.max(0) + 30]
        else:
            self.calls["roi"] += 1
            box = np.asarray(bbox)
            out = (uv[:, 0] < box[0]) | (uv[:, 0] > box[2]) | (uv[:, 1] < box[1]) | (uv[:, 1] > box[3])
            conf[out] = 0.05
        return dict(kpts=uv.astype(np.float32), conf=conf.astype(np.float32), bbox=box, layout="halpe26")


def _run(n, **cfg):
    seq = _seq(n)
    s1 = FakeS1(seq)
    pipe = LivePipeline(LiveConfig(**cfg), s1_model=s1, K=K, s2_backend="geometric",
                        lifter_kwargs=dict(occlusion=False))
    img = np.zeros((*HW, 3), np.uint8)
    res = []
    try:
        for i in range(n):
            s1.i = i
            res.append(pipe.process(img, t=i / 30.0))
    finally:
        pipe.close()
    return res, pipe, s1, seq


def test_roi_tracking_skips_the_detector_and_tracking_follows_the_motion():
    res, pipe, s1, seq = _run(40, async_keyframes=False, keyframe_every=15, redetect_every=30)
    assert s1.calls["detect"] <= 3 and s1.calls["roi"] >= 37          # detector only at start + safety net
    kinds = [r.kind for r in res]
    assert kinds[0] == "keyframe" and kinds.count("track") >= 30
    err = [np.median(np.linalg.norm(procrustes(r.frame.joints3d, X) - X, axis=1)) for r, X in zip(res, seq)]
    assert np.median(err) < 0.12                                          # same accuracy as per-photo S2
    jump = [np.median(np.linalg.norm(res[i].frame.joints3d - res[i - 1].frame.joints3d, axis=1)) for i in range(2, 40)]
    true = [np.median(np.linalg.norm(seq[i] - seq[i - 1], axis=1)) for i in range(2, 40)]
    assert np.median(jump) < 1.5 * np.median(true) + 0.005               # no added jitter


def test_keyframes_in_worker_process():
    res, pipe, s1, seq = _run(20, async_keyframes=True, keyframe_every=5)
    assert pipe.stats["keyframes"] >= 1 and all(r.croquis is not None for r in res)


def test_roi_lost_hysteresis():
    roi = ROITracker(LiveConfig())
    k = np.tile([[600.0, 300.0]], (J, 1)) + np.random.default_rng(0).normal(0, 50, (J, 2))
    good, bad, mid = np.full(J, 0.9), np.full(J, 0.1), np.full(J, 0.4)
    roi.update(k, good, HW, detected=True)
    assert not roi.lost
    roi.update(k, bad, HW, detected=False)
    assert not roi.lost                         # one bad frame is not enough
    roi.update(k, bad, HW, detected=False)
    assert roi.lost and roi.need_detect()
    roi.update(k, mid, HW, detected=True)
    assert roi.lost                             # between the thresholds: still lost
    roi.update(k, good, HW, detected=True)
    assert not roi.lost


def test_one_euro_removes_jitter_and_follows_motion():
    f = OneEuro(1.5, 0.01)
    rng = np.random.default_rng(0)
    raw, out = [], []
    for i in range(90):
        x = np.array([[300 + 2 * i, 200.0]]) + rng.normal(0, 2, (1, 2))   # 60 px/s drift + 2 px noise
        raw.append(x[0]); out.append(f(x, i / 30, np.array([True]))[0])
    raw, out = np.array(raw)[10:], np.array(out)[10:]
    truth = np.c_[300 + 2 * np.arange(10, 90), np.full(80, 200.0)]
    assert np.std(np.diff(out[:, 1])) < 0.5 * np.std(np.diff(raw[:, 1]))  # jitter down
    assert np.abs(out[:, 0] - truth[:, 0]).mean() < 6                       # little lag


def test_contact_hysteresis():
    X = stand_pose()
    v = np.ones(J, bool)
    lift = X.copy()
    for p in ("lheel", "lbigtoe", "lsmalltoe", "lank"):
        lift[IDX[p], 1] -= 0.17                  # camera y is down: the left foot 17 cm up -> not a new contact
    c0 = choose_contacts(lift, v, 1.70)
    assert not any(c.startswith("l") for c in c0)
    hover = X.copy()
    for p in ("lheel", "lbigtoe", "lsmalltoe", "lank"):
        hover[IDX[p], 1] -= 0.05
    assert any(c.startswith("l") for c in choose_contacts(hover, v, 1.70, dynamic=True,
                                                         prev=["lheel", "lbigtoe", "rheel", "rbigtoe"]))


def test_fast_motion_and_camera_pan_without_blocking():
    """Fast motion + a ±250 px camera pan + every 3rd frame: re-detection happens on the same frame,
    never through a blocking re-solve, and the aligned warm start keeps the fit on the image."""
    n = 30
    seq = []
    for j in range(n):
        i = 3 * j
        X = _seq(120)[i % 120] + np.array([0.9 * np.sin(2 * np.pi * i / 45), 0.15 * np.sin(2 * np.pi * i / 30), 0.0])
        seq.append(X)
    s1 = FakeS1(seq)
    pipe = LivePipeline(LiveConfig(async_keyframes=True), s1_model=s1, K=K,
                        s2_backend="geometric", lifter_kwargs=dict(occlusion=False))
    img = np.zeros((*HW, 3), np.uint8)
    try:
        res = []
        for i in range(n):
            s1.i = i
            res.append(pipe.process(img, t=i * 0.1))
    finally:
        pipe.close()
    assert [r.kind for r in res].count("keyframe") == 1                 # only the very first frame blocks
    err = [np.median(np.linalg.norm(procrustes(r.frame.joints3d, X) - X, axis=1)) for r, X in zip(res, seq)]
    assert np.median(err) < 0.12
    assert pipe.stats["degraded"] <= 0.3 * n


def test_frame_without_a_person_keeps_the_last_croquis():
    seq = _seq(5)
    s1 = FakeS1(seq)
    pipe = LivePipeline(LiveConfig(async_keyframes=False), s1_model=s1, K=K, s2_backend="geometric",
                        lifter_kwargs=dict(occlusion=False))
    img = np.zeros((*HW, 3), np.uint8)
    try:
        s1.i = 0
        r0 = pipe.process(img, t=0.0)
        empty = lambda im, bbox=None: dict(kpts=np.zeros((J, 2), np.float32), conf=np.zeros(J, np.float32),
                                           bbox=np.array([0, 0, 10, 10.0]), layout="halpe26")
        pipe.s1 = empty
        r1 = pipe.process(img, t=0.033)
        assert r1.kind == "no-person" and r1.croquis is r0.croquis
        pipe.s1 = s1
        s1.i = 2
        r2 = pipe.process(img, t=0.066)
        assert r2.kind.startswith("track") and r2.croquis is not None
    finally:
        pipe.close()


def test_predicted_roi_pushed_off_the_image_is_never_inverted():
    roi = ROITracker(LiveConfig())
    roi.box, roi.t = np.array([1100.0, 200.0, 1270.0, 600.0]), 0.0
    for vx in (3000.0, 6000.0, 20000.0, -20000.0):
        roi.v = np.array([vx, 0.0])
        b = roi.predict(0.1, HW)
        assert b is None or (b[2] > b[0] and b[3] > b[1] and b[0] >= 0 and b[2] <= HW[1] - 1)


def test_nonfinite_values_never_reach_the_renderer():
    from gesture_croquis.s4_gesture import render_preview
    seq = _seq(6)
    s1 = FakeS1(seq)
    pipe = LivePipeline(LiveConfig(async_keyframes=False), s1_model=s1, K=K, s2_backend="geometric",
                        lifter_kwargs=dict(occlusion=False))
    img = np.zeros((*HW, 3), np.uint8)
    try:
        s1.i = 0
        r0 = pipe.process(img, t=0.0)
        # S1 returns NaN for some joints (invalid ROI)
        def nan_s1(im, bbox=None):
            r = s1(im, bbox)
            r["kpts"] = r["kpts"].copy(); r["kpts"][[IDX["lwri"], IDX["rknee"]]] = np.nan
            return r
        pipe.s1, s1.i = nan_s1, 1
        r1 = pipe.process(img, t=0.033)
        assert np.isfinite(r1.frame.kpts2d).all() and pipe.stats["s1_nonfinite"] >= 1
        # the tracked fit diverges to NaN: rejected, last croquis kept
        pipe.s1 = s1
        orig = pipe.s2.track
        def nan_track(frame, *a, **k):
            out = orig(frame, *a, **k); frame.joints3d[:] = np.nan; return out
        pipe.s2.track, s1.i = nan_track, 2
        r2 = pipe.process(img, t=0.066)
        assert r2.kind == "fit-failed" and r2.croquis is not None and pipe.stats["fit_failed"] == 1
        pipe.s2.track, s1.i = orig, 3
        r3 = pipe.process(img, t=0.1)
        assert r3.kind.startswith("track") and np.isfinite(r3.frame.joints3d).all()
        # a stroke with NaN points is skipped by the renderer instead of crashing it
        c = r3.croquis
        c.strokes[0].pts = c.strokes[0].pts.copy(); c.strokes[0].pts[3] = np.nan
        out = render_preview(c, (360, 640), fit=False, supersample=1, view_scale=0.5)
        assert out.shape == (360, 640, 3)
    finally:
        pipe.close()


def test_fit_with_a_joint_behind_the_camera_is_rejected():
    seq = _seq(4)
    s1 = FakeS1(seq)
    pipe = LivePipeline(LiveConfig(async_keyframes=False), s1_model=s1, K=K, s2_backend="geometric",
                        lifter_kwargs=dict(occlusion=False))
    img = np.zeros((*HW, 3), np.uint8)
    try:
        s1.i = 0
        pipe.process(img, t=0.0)
        orig = pipe.s2.track
        def behind(frame, *a, **k):
            out = orig(frame, *a, **k); frame.joints3d[[IDX["lwri"], IDX["pelvis"]], 2] = -0.01; return out
        pipe.s2.track, s1.i = behind, 1
        r = pipe.process(img, t=0.033)                  # a torso joint behind the camera: not repairable
        assert r.kind == "fit-failed" and r.croquis is not None
    finally:
        pipe.close()


def test_huge_stroke_coordinates_do_not_crash_the_renderer():
    from gesture_croquis.s4_gesture import render_preview
    seq = _seq(2)
    s1 = FakeS1(seq)
    pipe = LivePipeline(LiveConfig(async_keyframes=False), s1_model=s1, K=K, s2_backend="geometric",
                        lifter_kwargs=dict(occlusion=False))
    try:
        c = pipe.process(np.zeros((*HW, 3), np.uint8), t=0.0).croquis
        for s in c.strokes[:3]:
            s.pts = s.pts.copy(); s.pts[1] = 3e9
        out = render_preview(c, (360, 640), fit=False, supersample=1, view_scale=0.5)
        assert out.shape == (360, 640, 3)
    finally:
        pipe.close()


def test_lm_solve_never_raises_on_nonfinite_numbers():
    from gesture_croquis.s2_jax import lm_solve
    p0 = np.zeros(3)
    # NaN from the start: p0 returned, cost inf, status -1
    p, cost, *_, st = lm_solve(lambda p: np.full(5, np.nan), lambda p: np.full((5, 3), np.nan), p0)
    assert st == -1 and not np.isfinite(cost) and np.allclose(p, p0)
    # finite start, the Jacobian blows up after a step: stops at the last good point
    calls = {"n": 0}
    def jac(p):
        calls["n"] += 1
        return np.eye(5, 3) if calls["n"] == 1 else np.full((5, 3), np.inf)
    p, cost, *_ = lm_solve(lambda p: np.r_[p - 1, 0, 0], jac, p0)
    assert np.isfinite(p).all() and np.isfinite(cost)


def test_s2_exception_in_tracking_is_a_failed_frame_not_a_crash():
    seq = _seq(4)
    s1 = FakeS1(seq)
    pipe = LivePipeline(LiveConfig(async_keyframes=False), s1_model=s1, K=K, s2_backend="geometric",
                        lifter_kwargs=dict(occlusion=False))
    img = np.zeros((*HW, 3), np.uint8)
    try:
        s1.i = 0
        pipe.process(img, t=0.0)
        def boom(*a, **k):
            raise np.linalg.LinAlgError("SVD did not converge in Linear Least Squares")
        orig, pipe.s2.track, s1.i = pipe.s2.track, boom, 1
        r = pipe.process(img, t=0.033)
        assert r.kind == "fit-failed" and r.croquis is not None
        pipe.s2.track, s1.i = orig, 2
        assert pipe.process(img, t=0.066).kind.startswith("track")
    finally:
        pipe.close()


def _jump_seq(n=36, t0=12, t1=22, lift=0.35):
    """Standing, then a ballistic jump (both feet leave the floor), then standing again."""
    X0 = stand_pose(pelvis_cam=(0.1, 0.05, 4.0))
    out = []
    for i in range(n):
        h = 0.0
        if t0 <= i <= t1:
            u = (i - t0) / (t1 - t0)
            h = 4 * lift * u * (1 - u)                     # parabola, peak `lift` m
        out.append(X0 + np.array([0.0, -h, 0.0]))          # camera y is down
    return out


def test_floor_tracking_detects_a_jump_and_only_the_jump():
    seq = _jump_seq()
    s1 = FakeS1(seq)
    pipe = LivePipeline(LiveConfig(async_keyframes=False, keyframe_every=100), s1_model=s1, K=K,
                        s2_backend="geometric", lifter_kwargs=dict(occlusion=False))
    img = np.zeros((*HW, 3), np.uint8)
    air = []
    try:
        for i in range(len(seq)):
            s1.i = i
            r = pipe.process(img, t=i / 30.0)
            air.append(bool(r.frame.s2_info.get("airborne")))
            if air[-1]:
                assert r.frame.s3_info["airborne"] and not r.frame.contacts      # no reactions, no gravity load
    finally:
        pipe.close()
    high = [i for i in range(len(seq)) if -seq[i][IDX["pelvis"], 1] + seq[0][IDX["pelvis"], 1] > 0.12]
    assert sum(air[i] for i in high) >= 0.6 * len(high)                           # the jump is flight
    assert not any(air[:12]) and not any(air[-8:])                                # standing is not
    assert pipe.floor.h is not None and pipe.floor.resets == 0


def test_floor_tracker_does_not_follow_a_flight_and_reseats_after_too_long():
    from gesture_croquis.live import FloorTracker
    X = stand_pose(pelvis_cam=(0.0, 0.0, 4.0))
    v = np.ones(J, bool)
    feet = choose_contacts(X, v, 1.7)
    fl = FloorTracker(LiveConfig())
    fl.update(X, v, 0.0, feet, False, 0.0)
    h0 = fl.h
    fl.update(X - [0, 0.3, 0], v, 0.0, [], True, 0.1)                # flight: floor stays
    assert fl.h == h0
    fl.update(X - [0, 0.3, 0], v, 0.0, [], True, 1.5)                # 1.4 s "flight": re-seated
    assert fl.resets == 1 and fl.h > h0 + 0.25


def test_lm_early_stop_cuts_iterations_not_accuracy():
    from gesture_croquis.s2_jax import lm_solve
    rng = np.random.default_rng(0)
    A = rng.normal(size=(40, 6)); b = rng.normal(size=40)
    f = lambda p: A @ p - b + 0.05 * np.sin(5 * p).sum()
    jac = lambda p: A + 0.25 * np.cos(5 * p)[None, :]
    p_full, c_full, n_full, *_ = lm_solve(f, jac, np.zeros(6), max_nfev=200, ftol=1e-12, gtol=1e-12)
    p_es, c_es, n_es, _, st = lm_solve(f, jac, np.zeros(6), max_nfev=200, ftol=1e-12, gtol=1e-12,
                                       max_rejects=4, stall_tol=2e-3)
    assert n_es < n_full and c_es < 1.01 * c_full


def test_a_few_diverged_limb_joints_are_repaired_and_logged():
    seq = _seq(4)
    s1 = FakeS1(seq)
    pipe = LivePipeline(LiveConfig(async_keyframes=False), s1_model=s1, K=K, s2_backend="geometric",
                        lifter_kwargs=dict(occlusion=False))
    img = np.zeros((*HW, 3), np.uint8)
    try:
        s1.i = 0
        pipe.process(img, t=0.0)
        orig = pipe.s2.track
        def behind(frame, *a, **k):
            out = orig(frame, *a, **k); frame.joints3d[IDX["lwri"], 2] = -0.2; return out
        pipe.s2.track, s1.i = behind, 1
        r = pipe.process(img, t=0.033)
        assert r.kind.startswith("track") and pipe.stats["repaired"] == 1 and pipe.stats["bad:lwri"] == 1
        assert not r.frame.valid3d[IDX["lwri"]] and r.frame.joints3d[IDX["lwri"], 2] > 1.0
        assert pipe.fail_log[-1]["joints"] == ["lwri"] and pipe.fail_log[-1]["repaired"]
    finally:
        pipe.close()


def test_debug_view_green_when_airborne_and_rotated_view():
    import cv2
    from types import SimpleNamespace
    from gesture_croquis.live import _draw, AIR_BG
    seq = _seq(2)
    s1 = FakeS1(seq)
    pipe = LivePipeline(LiveConfig(async_keyframes=False), s1_model=s1, K=K, s2_backend="geometric",
                        lifter_kwargs=dict(occlusion=False))
    img = np.zeros((*HW, 3), np.uint8)
    try:
        r = pipe.process(img, t=0.0)
    finally:
        pipe.close()
    a = SimpleNamespace(preview_width=640, overlay=False)
    ground = _draw(a, r, img, cv2)
    assert tuple(ground[2, 2]) == (250, 250, 250)
    r.frame.s2_info["airborne"] = True
    air = _draw(a, r, img, cv2)
    assert tuple(air[2, 2]) == AIR_BG
    rot = _draw(a, r, img, cv2, yaw=45.0)
    assert rot.shape == air.shape and (np.abs(rot.astype(int) - air.astype(int)).sum() > 0)
    assert (rot != np.array(AIR_BG, np.uint8)).any(axis=2).sum() > 500       # strokes were drawn


def test_theta_rotates_every_live_frame():
    import cv2
    from types import SimpleNamespace
    from gesture_croquis.live import _draw
    seq = _seq(2)
    s1 = FakeS1(seq)
    pipe = LivePipeline(LiveConfig(async_keyframes=False), s1_model=s1, K=K, s2_backend="geometric",
                        lifter_kwargs=dict(occlusion=False))
    img = np.zeros((*HW, 3), np.uint8)
    try:
        r = pipe.process(img, t=0.0)
    finally:
        pipe.close()
    a = SimpleNamespace(preview_width=640, overlay=False, theta=90.0)
    front, side = _draw(a, r, img, cv2, 0.0), _draw(a, r, img, cv2, a.theta)
    assert front.shape == side.shape and np.abs(front.astype(int) - side.astype(int)).sum() > 0


def test_a_small_hop_is_flight_planted_feet_cannot_slide_to_hide_it():
    seq = _jump_seq(n=44, t0=20, t1=24, lift=0.12)           # 12 cm hop, ~0.4 s at 10 fps
    s1 = FakeS1(seq)
    pipe = LivePipeline(LiveConfig(async_keyframes=False, keyframe_every=100), s1_model=s1, K=K,
                        s2_backend="geometric", lifter_kwargs=dict(occlusion=False))
    img = np.zeros((*HW, 3), np.uint8)
    air = []
    try:
        for i in range(len(seq)):
            s1.i = i
            air.append(bool(pipe.process(img, t=i / 10.0).frame.s2_info.get("airborne")))
    finally:
        pipe.close()
    assert sum(air[21:24]) >= 2 and not any(air[:20]) and not any(air[25:])
    assert pipe.floor.lift_m < 0.06                           # threshold adapted to the (low) noise

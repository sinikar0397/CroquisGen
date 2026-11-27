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
        out.append(_rot_y(X, 40 * np.sin(2 * np.pi * i / n)))
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

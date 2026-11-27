"""
Live (video) S1-S4: ROI tracking + hysteresis for frame-to-frame continuity and speed.

Per frame:
  S1  ROI tracking. The person detector (YOLOX) runs only when there is no track, after the
      track was lost, or every `redetect_every` frames as a safety net. Otherwise RTMPose runs on
      a box grown around the previous frame's keypoints (grown further when a joint nears its
      edge). "Lost" has hysteresis: the core-joint confidence must stay below `lost_enter` for
      two frames to count as lost, and rise above `lost_exit` to count as found again.
      A One-Euro filter on the keypoints removes jitter without lagging fast motion.
  S2  Tracking. One physical fit warm-started from the previous frame's skeleton, height,
      proportions and camera pitch (Lifter3D.track): no detector, no RTMW3D, no search. Foot
      contacts carry over with hysteresis (enter near the floor, leave only above 8 cm).
      Floor tracking (dynamic mode): the floor height is tracked over time (FloorTracker) and
      feet are judged against it, not against the lowest foot of the frame. With no foot on it
      the frame is FLIGHT: no contacts, no gravity load (free fall), and the planted feet of
      other frames are pulled onto the tracked plane.
      Keyframes. Every `keyframe_every` frames, or when the tracked fit degrades (2D error or
      energy jumps), the full re-solve runs: RTMW3D depth, both depth signs, label swaps, flip
      search. It runs in a separate worker PROCESS (its own interpreter, so it does not compete
      with the frame loop for Python's global lock; RTMW3D loads only there), and the track
      continues meanwhile. The current track
      joins it as a candidate with a `track_bonus` discount; when the result arrives, a winner
      that is not the track is re-fitted to the CURRENT frame and replaces the track only if it
      beats the track there by `track_bonus` as well. Two readings that are almost equally good
      (the "! check the side view" case) therefore do not flip back and forth.
  S3  static load on the tracked skeleton, with the tracked contacts.
  S4  strokes; the line-of-action chain has hysteresis too (S4Params.loa_hysteresis).

    python -m gesture_croquis.live --camera 0 --show
    python -m gesture_croquis.live --video clip.mp4 --out croquis.mp4
"""
from __future__ import annotations
import argparse
import collections
import copy
import time
from dataclasses import dataclass, field
import numpy as np

from .poseframe import PoseFrame
from .skeleton import IDX, J

CORE = [IDX[n] for n in ("lsho", "rsho", "lhip", "rhip", "lknee", "rknee", "lank", "rank")]


@dataclass
class LiveConfig:
    # S1 ROI
    roi_margin: float = 0.25          # ROI = confident-keypoint box grown by this fraction of its size
    roi_grow: float = 0.5             # ... by this much when a joint is near the ROI edge
    roi_min_px: int = 96
    edge_frac: float = 0.06           # "near the edge" = within this fraction of the ROI size
    redetect_every: int = 90          # run the detector at least this often (frames)
    lost_enter: float = 0.30          # core-joint mean confidence below this for 2 frames -> lost
    lost_exit: float = 0.45           # ... and above this to be tracked again
    roi_lead: float = 0.6             # predictive ROI: grow by this x the expected motion per frame
    kpt_conf: float = 0.30            # keypoints that define the ROI
    # 2D smoothing (One-Euro; px, Hz)
    smooth: bool = True
    min_cutoff: float = 1.5
    beta: float = 0.05                # speed term: fast motion is barely smoothed (no lag)
    # S2
    keyframe_every: int = 30          # full re-solve period (frames)
    keyframe_reproj_frac: float = 0.02     # ... or when the tracked 2D error (median) exceeds this
                                           # fraction of the person's height in the image,
    keyframe_reproj_max_frac: float = 0.10 # ... or one joint is this far off (a limb lost),
    keyframe_cost_jump: float = 2.5   # ... or the energy jumps by this factor from one frame to the next.
                                      # (Not vs. a running level: a harder pose, e.g. sinking into a
                                      # lunge, raises the energy legitimately.)
    track_bonus: float = 0.15         # hysteresis: a re-solve must beat the track by 15%
    track_iters: int = 20
    lm_max_rejects: int = 4           # tracked fit: stop after 4 rejected steps in a row
    lm_stall_tol: float = 2e-3        # ... or when 3 accepted steps gained < 0.2% together
    # floor tracking (dynamic mode)
    floor_alpha: float = 0.2          # EMA rate of the floor height while a foot is planted
    floor_max_rate: float = 0.25      # m/s, floor moving DOWN (feet sinking / camera rising)
    floor_max_rate_up: float = 0.05   # m/s, floor moving UP: slow, so a take-off cannot carry the floor along
    lift_k: float = 3.0               # a foot is lifted at lift_k x the noise of planted-foot heights ...
    lift_min_m: float = 0.03          # ... but at least 3 cm
    lift_max_m: float = 0.08          # ... and at most 8 cm (the fixed value before)
    floor_max_tilt_rate_deg: float = 5.0   # deg/s: how fast the tracked floor may tilt (camera tilting)
    max_flight_s: float = 1.0         # longer "flight" means the floor estimate is wrong: re-seat it
    repair_max_joints: int = 2        # a fit with <= this many diverged limb joints is repaired
    async_keyframes: bool = True      # keyframes in a worker process (False: in the frame loop, stalls)
    dynamic: bool = True              # video: no static-balance / level-floor priors (people move)
    worker_threads: int = 2           # CPU threads the keyframe worker may use (XLA + ONNX Runtime)
    # S4
    s4_layers: str = "AB"


class OneEuro:
    """One-Euro filter (Casiez et al. 2012) on (J,2) keypoints; joints with conf 0 pass through and
    reset."""

    def __init__(self, min_cutoff=1.5, beta=0.01, d_cutoff=1.0):
        self.mc, self.beta, self.dc = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t, valid):
        x = np.asarray(x, float)
        if self.x is None or self.t is None or t <= self.t:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = t - self.t
        dx = (x - self.x) / dt
        a_d = self._alpha(self.dc, dt)
        dx_hat = self.dx + a_d * (dx - self.dx)
        speed = np.linalg.norm(dx_hat, axis=-1, keepdims=True)
        a = self._alpha(self.mc + self.beta * speed, dt)
        x_hat = self.x + a * (x - self.x)
        v = np.asarray(valid, bool)[:, None]
        x_hat = np.where(v, x_hat, x)
        self.x, self.dx, self.t = x_hat, np.where(v, dx_hat, 0.0), t
        return x_hat


class ROITracker:
    """Person box for S1: detector only when needed, otherwise the previous keypoints' box."""

    def __init__(self, cfg: LiveConfig):
        self.cfg = cfg
        self.box = None
        self.since_det = 0
        self.lost = True
        self.low = 0
        self.n_detect = 0
        self.v = np.zeros(2)              # box-centre velocity, px/s
        self.t = None

    def need_detect(self) -> bool:
        return self.box is None or self.lost or self.since_det >= self.cfg.redetect_every

    def _valid_box(self, lo, hi, img_hw):
        """A box inside the image and at least roi_min_px in size, or None if its centre left the
        image (then the detector runs). Clipping alone could invert a box pushed past the edge."""
        H, W = img_hw
        c = (lo + hi) / 2
        if not (np.all(np.isfinite(c)) and 0 <= c[0] < W and 0 <= c[1] < H):
            return None
        half = np.maximum((hi - lo) / 2, self.cfg.roi_min_px / 2)
        lo, hi = np.maximum(c - half, 0), np.minimum(c + half, [W - 1, H - 1])
        if np.any(hi - lo < self.cfg.roi_min_px / 2):
            return None
        return np.array([lo[0], lo[1], hi[0], hi[1]], float)

    def predict(self, t, img_hw):
        """The ROI for time t: the last box moved by the person's velocity and grown by the expected
        motion, so fast motion (or a panning camera) does not carry the person out of it."""
        if self.box is None or self.t is None:
            return self.box
        d = self.v * max(t - self.t, 0.0)
        grow = self.cfg.roi_lead * np.abs(d)
        return self._valid_box(self.box[:2] + d - grow, self.box[2:] + d + grow, img_hw)

    def update(self, kpts, conf, img_hw, detected: bool, det_box=None, t=None):
        cfg = self.cfg
        H, W = img_hw
        self.since_det = 0 if detected else self.since_det + 1
        self.n_detect += int(detected)
        core = float(np.mean(conf[CORE]))
        if core < cfg.lost_enter:                       # hysteresis on "lost"
            self.low += 1
            if self.low >= 2:
                self.lost = True
        elif core > cfg.lost_exit:
            self.low, self.lost = 0, False
        ok = conf > cfg.kpt_conf
        if ok.sum() < 4:
            self.box = None if det_box is None else self._valid_box(np.asarray(det_box[:2], float),
                                                                    np.asarray(det_box[2:], float), img_hw)
            return
        lo, hi = kpts[ok].min(0), kpts[ok].max(0)
        size = np.maximum(hi - lo, 1.0)
        margin = cfg.roi_margin
        if self.box is not None and not detected:       # a joint near the ROI edge: grow next ROI
            bw = np.maximum(self.box[2:] - self.box[:2], 1.0)
            near = ((kpts[ok] - self.box[:2]) < cfg.edge_frac * bw) | ((self.box[2:] - kpts[ok]) < cfg.edge_frac * bw)
            if near.any():
                margin += cfg.roi_grow
        lo, hi = lo - margin * size, hi + margin * size
        c, half = (lo + hi) / 2, np.maximum((hi - lo) / 2, cfg.roi_min_px / 2)
        lo, hi = c - half, c + half
        if self.box is not None and t is not None and self.t is not None and t > self.t:
            v = (c - (self.box[:2] + self.box[2:]) / 2) / (t - self.t)
            self.v = 0.5 * self.v + 0.5 * v
        elif detected:
            self.v = np.zeros(2)
        self.t = t
        self.box = self._valid_box(lo, hi, img_hw)


def describe_models(s1="halpe26", mode="balanced", s2_backend="rtmw3d", det_mode=None):
    """Which networks run where, from rtmlib's own tables."""
    try:
        import rtmlib
        cls = {"halpe26": "BodyWithFeet", "coco17": "Body", "wholebody": "Wholebody"}[s1]
        cfg = dict(getattr(rtmlib, cls).MODE[mode])
        if det_mode:
            cfg["det"], cfg["det_input_size"] = (getattr(rtmlib, cls).MODE[det_mode][k] for k in ("det", "det_input_size"))
        name = lambda u: u.split("/")[-1].split("_8xb")[0].split("_simcc")[0]
        out = {"S1 detector (re-detection frames only)": f"{name(cfg['det'])} {cfg['det_input_size']}",
               "S1 pose (every frame, on the ROI)": f"{name(cfg['pose'])} {s1} {cfg['pose_input_size']}"}
        if s2_backend == "rtmw3d":
            c3 = rtmlib.Wholebody3d.MODE["balanced"]
            out["S2 depth (keyframes only, worker)"] = f"{name(c3['pose'])} {c3['pose_input_size']}"
        return out
    except Exception as e:
        return {"models": f"unknown ({e!r:.60})"}


def _keyframe_worker(conn, lifter_kwargs, threads=2):
    """Worker process: its own Lifter3D (with RTMW3D); solves keyframes sent over the pipe.
    Runs at low priority with capped thread pools, so a re-solve does not slow the frame loop."""
    import os
    try:
        os.nice(10)
    except Exception:
        pass
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false "
                               f"intra_op_parallelism_threads={threads}").strip()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = str(threads)
    try:                                            # RTMW3D's ONNX Runtime session: cap its threads
        import onnxruntime as ort
        _IS = ort.InferenceSession

        def _capped(*a, sess_options=None, **k):
            so = sess_options or ort.SessionOptions()
            so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
            return _IS(*a, sess_options=so, **k)
        ort.InferenceSession = _capped
    except Exception:
        pass
    from .s2_lift import Lifter3D
    lifter = Lifter3D(**lifter_kwargs)
    conn.send("ready")
    while True:
        msg = conn.recv()
        if msg is None:
            break
        frame, img, K, kw = msg
        try:
            lifter(frame, img, K=K, **kw)
            conn.send(("ok", frame))
        except Exception as e:                      # a failed re-solve must not stop the stream
            conn.send(("error", repr(e)))


class KeyframeWorker:
    """Full S2 re-solves in a separate process. submit() returns at once; poll() gives the
    finished keyframe (or None); solve() blocks (used for the very first frame)."""

    def __init__(self, lifter_kwargs, timeout=600, threads=2):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")                # JAX / ONNX Runtime are not fork-safe
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_keyframe_worker, args=(child, lifter_kwargs, threads), daemon=True)
        self.proc.start()
        self.timeout = timeout
        self.busy = False
        self._ready = False

    def _wait_ready(self):
        if not self._ready:
            try:
                if not self.conn.poll(self.timeout):
                    raise RuntimeError("keyframe worker did not start")
                assert self.conn.recv() == "ready"
            except EOFError:
                raise RuntimeError("the keyframe worker process exited at start-up. It re-imports the calling "
                                   "script, so that script must be a file with an `if __name__ == '__main__':` "
                                   "guard (or use LiveConfig(async_keyframes=False))") from None
            self._ready = True

    def submit(self, frame, img, K, kw):
        self._wait_ready()
        self.conn.send((frame, img, K, kw))
        self.busy = True

    def poll(self):
        if self.busy and self.conn.poll():
            self.busy = False
            status, out = self.conn.recv()
            return out if status == "ok" else None
        return None

    def solve(self, frame, img, K, kw):
        self.submit(frame, img, K, kw)
        if not self.conn.poll(self.timeout):
            raise RuntimeError("keyframe timed out")
        return self.poll()

    def close(self):
        try:
            self.conn.send(None)
        except Exception:
            pass
        self.proc.join(timeout=5)


@dataclass
class LiveResult:
    frame: PoseFrame
    croquis: object
    kind: str                         # "keyframe" | "track" | "track+switch"
    detected: bool
    timings: dict = field(default_factory=dict)


class FloorTracker:
    """Floor height along "up" (m; camera centre = 0, gravity from the fitted camera pitch),
    tracked over frames. Seeded by the first fit's lowest planted foot; then, while at least one
    foot is a contact, an EMA toward the lowest planted foot (rate floor_alpha, at most
    floor_max_rate m/s). During flight it stays put, so a jump reads as both feet ABOVE
    it instead of the lower foot becoming the new floor. Camera turns do not move it (height is
    measured along gravity); a camera moving up or down is followed by the EMA while a foot is
    planted. A flight longer than max_flight_s cannot be a jump: the floor is re-seated."""

    def __init__(self, cfg):
        self.cfg, self.h, self.t_air, self.resets, self.t = cfg, None, None, 0, None
        self.pitch = None                     # the floor's tilt (camera pitch), tracked as slowly
        self.res = collections.deque(maxlen=90)   # planted-foot height - floor (m): the fit's noise

    @property
    def noise_m(self):
        """Robust std of planted-foot heights about the floor (MAD), None until 15 samples."""
        if len(self.res) < 15:
            return None
        r = np.asarray(self.res)
        return float(1.4826 * np.median(np.abs(r - np.median(r))))

    @property
    def lift_m(self):
        n = self.noise_m
        c = self.cfg
        return c.lift_max_m if n is None else float(np.clip(c.lift_k * n, c.lift_min_m, c.lift_max_m))

    def update(self, X, valid, pitch, contacts, airborne, t):
        from .biomech import foot_heights, gravity_dir, FOOT_POINTS
        if self.pitch is None:
            self.pitch = float(pitch)
        fh = foot_heights(X, np.asarray(valid, bool), gravity_dir(self.pitch))
        dt = 1 / 30 if self.t is None else min(max(t - self.t, 0.0), 0.5)
        self.t = t
        if not fh:
            return
        if airborne:
            self.t_air = t if self.t_air is None else self.t_air
            if t - self.t_air > self.cfg.max_flight_s:
                self.pitch = float(pitch)
                self.h, self.t_air = min(foot_heights(X, np.asarray(valid, bool), gravity_dir(pitch)).values()), None
                self.resets += 1
            return
        self.t_air = None
        planted = [sd for sd in fh if any(p in contacts for p in FOOT_POINTS[sd])]
        if not planted:                       # supported by hands / knees only: no floor sample
            return
        lo = min(fh[sd] for sd in planted)
        if self.h is None:
            self.h = lo
        else:
            d = lo - self.h
            if abs(d) < 2 * self.cfg.lift_max_m:        # inliers only feed the noise estimate
                self.res.append(d)
            up, down = self.cfg.floor_max_rate_up * dt, self.cfg.floor_max_rate * dt
            self.h += float(np.clip(self.cfg.floor_alpha * d, -down, up))
        dp = np.radians(self.cfg.floor_max_tilt_rate_deg) * dt          # camera tilt, followed slowly
        self.pitch += float(np.clip(self.cfg.floor_alpha * (pitch - self.pitch), -dp, dp))


class LivePipeline:
    """S1-S4 on a video stream, with ROI tracking and hysteresis. Models load once.
    s1_model / lifter can be injected (tests, or a different S1)."""

    def __init__(self, cfg: LiveConfig | None = None, height=1.70, s1="halpe26", mode="balanced",
                 device="cpu", depth_device=None, K=None, s1_model=None, lifter_kwargs=None, s2_backend="rtmw3d",
                 det_mode="lightweight"):
        """lifter_kwargs: Lifter3D arguments for keyframes (default: RTMW3D depth, auto sign).
        The frame loop's own Lifter3D only tracks, so it never loads RTMW3D."""
        self.cfg = cfg or LiveConfig()
        self.height = height
        if s1_model is None:
            from .s1_keypoints import RTMPose2D
            s1_model = RTMPose2D(s1, mode=mode, device=device, det_mode=det_mode)
        kf_kw = dict(backend=s2_backend, height=height, z_sign="auto", device=device, depth_device=depth_device,
                     workers=1, dynamic=self.cfg.dynamic)
        kf_kw.update(lifter_kwargs or {})
        from .s2_lift import Lifter3D
        if self.cfg.async_keyframes:
            self.worker = KeyframeWorker(kf_kw, threads=self.cfg.worker_threads)   # loads in parallel
            self.s2 = Lifter3D(**dict(kf_kw, backend="geometric"))    # tracking only: no RTMW3D here
        else:
            self.worker = None
            self.s2 = Lifter3D(**kf_kw)
        self.s1 = s1_model
        self.models = describe_models(s1, mode, s2_backend, det_mode) if not hasattr(s1_model, "calls") else {}
        self.last_croquis = None
        self.K = K
        self.roi = ROITracker(self.cfg)
        self.euro = OneEuro(self.cfg.min_cutoff, self.cfg.beta)
        self.i = 0
        self.state = None                 # dict(X, scales, pitch, contacts, sign, cons, cost)
        self.loa_name = None
        self.costs = collections.deque(maxlen=30)
        self.last_kf = -10 ** 9
        self.stats = collections.Counter()
        self.floor = FloorTracker(self.cfg)
        self.fail_log = collections.deque(maxlen=200)   # diverged fits: which joints, and why

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _scales(info, height):
        return np.r_[np.log(info["fitted_height_m"] / height),
                     np.log([info["proportions"][k] for k in ("legs", "arms", "torso")])]

    def _state_from(self, frame):
        i = frame.s2_info
        return dict(X=frame.joints3d.copy(), valid=np.asarray(frame.valid3d, bool).copy(),
                    scales=self._scales(i, self.height),
                    pitch=(i.get("physics_p") or [0.0])[0], contacts=list(i.get("contacts", [])),
                    sign=i.get("z_sign") or (self.state or {}).get("sign"), cost=float(i["cost"]),
                    cons=(self.state or {}).get("cons"))

    @staticmethod
    def _sane_fit(X, conf=None, near_frac=0.3):
        """A usable 3D fit: finite, and no joint at/behind the camera or far closer than the body
        (a diverged joint there projects to ~1e9 px and breaks S4/drawing)."""
        X = np.asarray(X, float)
        if X.ndim != 2 or not np.isfinite(X).all():
            return False
        z = X[:, 2]
        zc = np.median(z if conf is None else z[np.asarray(conf) > 0] if (np.asarray(conf) > 0).sum() >= 4 else z)
        return bool(zc > 0.05 and z.min() > near_frac * zc)

    def _s1(self, img, bbox=None):
        """S1 with its output sanitised: a non-finite keypoint counts as missing."""
        r = dict(self.s1(img) if bbox is None else self.s1(img, bbox=bbox))
        k, c = np.asarray(r["kpts"], np.float64), np.asarray(r["conf"], np.float64)
        bad = ~np.isfinite(k).all(1) | ~np.isfinite(c)
        if bad.any():
            k, c = k.copy(), c.copy()
            k[bad], c[bad] = 0.0, 0.0
            self.stats["s1_nonfinite"] += 1
        r["kpts"], r["conf"] = k, c
        return r

    def _poor(self, conf):
        """S1 found no usable person (e.g. fast motion carried them out of the ROI)."""
        cc = conf[CORE]
        return float(np.mean(cc)) < self.cfg.lost_enter or int((cc > self.cfg.kpt_conf).sum()) < 5

    @staticmethod
    def _aligned(X, k2, conf, K, thr=0.3):
        """The previous 3D pose moved (and its depth scaled) so it projects onto the new keypoints.
        Orientation is kept. A warm start from here survives fast motion and camera pans; from the
        old position the solver needs many more steps, or fails."""
        from .s2_lift import project, backproject
        # the torso (shoulders + hips) is nearly rigid: its centroid and size track the body's
        # position and distance; limbs would make the "size" change with the pose
        idx = np.array([IDX[n] for n in ("lsho", "rsho", "lhip", "rhip") if conf[IDX[n]] > thr])
        if len(idx) < 3:
            return X
        uv = project(X[idx], K)
        c_new, c_old = k2[idx].mean(0), uv.mean(0)
        s_new = np.sqrt(((k2[idx] - c_new) ** 2).sum(1).mean())
        s_old = np.sqrt(((uv - c_old) ** 2).sum(1).mean())
        scale = float(np.clip(s_new / max(s_old, 1e-6), 0.7, 1.4))
        if abs(np.log(scale)) < 0.15:         # deadband: small size changes are pose / noise, not distance
            scale = 1.0
        cen = X[idx].mean(0)
        target = backproject(c_new[None], np.array([cen[2] / scale]), K)[0]
        return X - cen + target

    @staticmethod
    def _on_rays(X, k2, conf, K, valid_prev, frac=0.03):
        """Warm start that already matches the image: every visible joint keeps its depth from the
        (aligned) previous pose but moves onto its NEW camera ray. A joint that was not visible in
        the previous frame (its 3D position was parked on a neighbour) takes its parent's depth.
        Fast limb motion then starts the solver from the right 2D place instead of a stale or
        degenerate one, which it often could not leave within the iteration budget.
        Only joints that are really off (more than `frac` of the person's height) or newly visible
        move; small motion keeps the smooth previous pose (moving every joint copies each frame's
        keypoint noise into the start, which showed up as jitter)."""
        from .s2_lift import backproject, project, _TREE, CONF_MIN
        v = conf > CONF_MIN
        Z = X[:, 2].copy()
        vp = np.ones(J, bool) if valid_prev is None else np.asarray(valid_prev, bool)
        for a, b in _TREE:                               # parent before child
            if v[IDX[b]] and not vp[IDX[b]]:
                Z[IDX[b]] = Z[IDX[a]]
        ok = conf > 0.3
        person = float(np.ptp(k2[ok, 1])) if ok.sum() >= 4 else 0.0
        off = np.linalg.norm(project(X, K) - k2, axis=1) > frac * max(person, 1.0)
        move = v & (off | ~vp)
        X0 = X.copy()
        X0[move] = backproject(k2[move], np.maximum(Z[move], 0.3), K)
        return X0

    def _kf_kwargs(self, track):
        if track is None:
            return {}
        return dict(X_track=track["X"], track_sign=track["sign"], track_bonus=self.cfg.track_bonus,
                    track_scales=track["scales"], track_pitch=track["pitch"], contacts_prev=track["contacts"],
                    floor_h=self._floor_h())

    def _floor_h(self):
        return self.floor.h if self.cfg.dynamic else None

    def _floor_kw(self):
        return dict(floor_h=self._floor_h(), floor_pitch=self.floor.pitch if self.cfg.dynamic else None,
                    lift_m=self.floor.lift_m)

    def _lm_opts(self):
        return dict(max_rejects=self.cfg.lm_max_rejects, stall_tol=self.cfg.lm_stall_tol)

    @staticmethod
    def _bad_joints(X, conf, near_frac=0.3):
        """Indices of joints that make a fit unusable (non-finite, or at/behind the camera)."""
        X = np.asarray(X, float)
        fin = np.isfinite(X).all(1)
        if fin.sum() < 4:
            return np.arange(len(X))
        z = X[:, 2]
        good = fin & (np.asarray(conf) > 0)
        zc = np.median(z[good] if good.sum() >= 4 else z[fin])
        return np.where(~fin | (z <= near_frac * zc))[0]

    def _diagnose_and_repair(self, frame, X0, st, c, K):
        """Log which joints diverged and from what warm start; repair the fit when only a few limb
        joints went bad (put them back on their 2D rays at the parent's depth, mark them not
        valid in 3D). Returns True when the frame is usable after repair."""
        from .s2_lift import _TREE, backproject
        from .skeleton import JOINTS
        X = np.asarray(frame.joints3d, float)
        bad = self._bad_joints(X, c)
        vp = np.ones(len(c), bool) if st.get("valid") is None else np.asarray(st["valid"], bool)
        torso = {IDX[n] for n in ("pelvis", "neck", "lsho", "rsho", "lhip", "rhip")}
        rec = dict(frame=self.i, joints=[JOINTS[j] for j in bad],
                   reappeared=[JOINTS[j] for j in bad if c[j] > 0.3 and not vp[j]],
                   low_conf=[JOINTS[j] for j in bad if c[j] <= 0.3],
                   x0_z={JOINTS[j]: round(float(X0[j, 2]), 2) for j in bad if np.isfinite(X0[j, 2])},
                   lm_status=(frame.s2_info or {}).get("lm_status"),
                   error=getattr(self, "_err_this_frame", None))
        for n in rec["joints"]:
            self.stats["bad:" + n] += 1
        self.stats["bad_reappeared"] += int(bool(rec["reappeared"]))
        repairable = (0 < len(bad) <= self.cfg.repair_max_joints and not (set(bad) & torso)
                      and np.isfinite(np.delete(X, bad, 0)).all())
        rec["repaired"] = bool(repairable)
        self.fail_log.append(rec)
        if not repairable:
            return False
        parent = {IDX[b]: IDX[a] for a, b in _TREE}
        for j in sorted(bad, key=lambda j: j in parent and parent[j] in bad):   # parents first
            pa = parent.get(j)
            zp = X[pa, 2] if pa is not None and np.isfinite(X[pa, 2]) else np.median(X[np.isfinite(X[:, 2]), 2])
            X[j] = backproject(frame.kpts2d[j][None].astype(float), np.array([zp]), K)[0]
        frame.joints3d = X
        v = np.asarray(frame.valid3d, bool).copy()
        v[bad] = False
        frame.valid3d = v
        return self._sane_fit(X, c)

    def _keyframe_now(self, frame, img, K, track):
        """Blocking keyframe (first frame, re-acquisition, or async_keyframes=False)."""
        if self.worker is not None:
            out = self.worker.solve(frame, img, K, self._kf_kwargs(track))
            if out is None:
                raise RuntimeError("keyframe failed")
            return out
        self.s2(frame, img, K=K, **self._kf_kwargs(track))
        return frame

    def close(self):
        if self.worker is not None:
            self.worker.close()

    # ------------------------------------------------------------------ per frame
    def process(self, img_bgr, t=None) -> LiveResult:
        cfg = self.cfg
        t = time.perf_counter() if t is None else t
        T = {}
        hw = img_bgr.shape[:2]
        K = self.K
        if K is None:
            from .s2_lift import default_K
            K = self.K = default_K(hw)
        # ---- S1 with a predictive ROI
        t0 = time.perf_counter()
        detect = self.roi.need_detect()
        box = None if detect else self.roi.predict(t, hw)
        if box is None:                                  # no usable ROI (e.g. pushed off the image)
            detect = True
        r = self._s1(img_bgr) if detect else self._s1(img_bgr, bbox=box)
        if not detect and self._poor(r["conf"]):       # the person left the ROI: re-detect on THIS frame
            r = self._s1(img_bgr)
            detect = True
            self.stats["redetect_now"] += 1
            self.euro = OneEuro(cfg.min_cutoff, cfg.beta)    # a jump, not motion: do not smooth across it
        T["s1"] = 1e3 * (time.perf_counter() - t0)
        k2, c = r["kpts"].astype(float), r["conf"].astype(float)
        if cfg.smooth:
            k2 = self.euro(k2, t, c > 0.05)
        self.roi.update(k2, c, hw, detect, det_box=r.get("bbox"), t=t)
        self.stats["detect" if detect else "roi"] += 1
        if self._poor(c):                               # nobody usable in this frame: keep the last croquis
            self.stats["no_person"] += 1
            self.i += 1
            T["total"] = sum(T.values())
            frame = PoseFrame(img_hw=hw, kpts2d=k2.astype(np.float32), conf=c.astype(np.float32),
                              body_height_m=self.height)
            return LiveResult(frame, self.last_croquis, "no-person", detect, T)
        frame = PoseFrame(img_hw=hw, kpts2d=k2.astype(np.float32), conf=c.astype(np.float32),
                          bbox=r.get("bbox") if detect else self.roi.box, s1_layout=r.get("layout", ""),
                          body_height_m=self.height)
        # ---- S2
        t0 = time.perf_counter()
        kind = "track"
        if self.state is None:                         # no track yet: the one blocking keyframe
            try:
                frame = self._keyframe_now(frame, img_bgr, K, None)
                ok_fit = self._sane_fit(frame.joints3d, frame.conf)
            except Exception as e:
                self.last_s2_error, ok_fit = repr(e), False
            if not ok_fit:                              # try again on the next frame
                self.stats["fit_failed"] += 1
                self.i += 1
                T["s2"] = 1e3 * (time.perf_counter() - t0)
                T["total"] = sum(T.values())
                return LiveResult(frame, self.last_croquis, "fit-failed", detect, T)
            kind, self.last_kf = "keyframe", self.i
            self.state = self._state_from(frame)
        else:
            st = self.state
            X0 = self._on_rays(self._aligned(st["X"], k2, c, K), k2, c, K, st.get("valid"))   # follow motion
            if not np.isfinite(X0).all():               # a bad warm start: fall back to the last pose
                X0 = np.asarray(st["X"], float)
            self._err_this_frame = None
            try:
                self.s2.track(frame, K, X0, scales_p0=st["scales"], pitch=st["pitch"],
                              contacts_prev=st["contacts"], order_cons=st["cons"], iters=cfg.track_iters,
                              lm_opts=self._lm_opts(), **self._floor_kw())
                cost = float(frame.s2_info["cost"])
            except Exception as e:                      # numerical failure: handled as a failed fit
                self.last_s2_error = self._err_this_frame = repr(e)
                frame.joints3d = np.full((len(c), 3), np.nan)
                cost = float("inf")
            if not (self._sane_fit(frame.joints3d, c) and np.isfinite(cost)):
                if np.isfinite(cost) and self._diagnose_and_repair(frame, X0, st, c, K):
                    self.stats["repaired"] += 1
                else:
                    if not np.isfinite(cost) and frame.joints3d is not None:
                        self._diagnose_and_repair(frame, X0, st, c, K)      # log only
            if not (self._sane_fit(frame.joints3d, c) and np.isfinite(cost)):   # failed fit: keep the track
                self.stats["fit_failed"] += 1
                if self.worker is not None and not self.worker.busy:
                    self.last_kf = self.i
                    self.stats["keyframes"] += 1
                    fr = PoseFrame(img_hw=hw, kpts2d=k2.astype(np.float32), conf=c.astype(np.float32),
                                   bbox=frame.bbox, body_height_m=self.height)
                    self.worker.submit(fr, img_bgr, K, self._kf_kwargs(dict(self.state)))
                self.i += 1
                T["s2"] = 1e3 * (time.perf_counter() - t0)
                T["total"] = sum(T.values())
                return LiveResult(frame, self.last_croquis, "fit-failed", detect, T)
            # a finished background keyframe: re-fit its winner to THIS frame, switch with hysteresis
            kf = self.worker.poll() if self.worker is not None else None
            if (kf is not None and self._sane_fit(kf.joints3d, kf.conf)
                    and not str(kf.s2_info.get("init", "")).startswith("track")):
                alt = copy.deepcopy(frame)
                ki = kf.s2_info
                self.s2.track(alt, K, self._on_rays(self._aligned(kf.joints3d, k2, c, K), k2, c, K, kf.valid3d),
                              scales_p0=self._scales(ki, self.height),
                              pitch=(ki.get("physics_p") or [0.0])[0], contacts_prev=ki.get("contacts"),
                              order_cons=None, iters=cfg.track_iters,
                              lm_opts=self._lm_opts(), **self._floor_kw())
                self.stats["keyframe_proposals"] += 1
                if self._sane_fit(alt.joints3d, c) and float(alt.s2_info["cost"]) < cost * (1 - cfg.track_bonus):
                    frame, cost, kind = alt, float(alt.s2_info["cost"]), "track+switch"
                    self.stats["switches"] += 1
                    self.state = self._state_from(frame)
                    self.state["sign"] = ki.get("z_sign") or self.state["sign"]
            self.state = dict(self._state_from(frame), sign=self.state["sign"], cons=self.state["cons"])
            # schedule the next keyframe
            prev_cost = self.costs[-1] if self.costs else np.inf
            i2 = frame.s2_info
            ok = frame.conf > cfg.kpt_conf
            person_px = float(np.ptp(frame.kpts2d[ok, 1])) if ok.sum() >= 4 else float(hw[0])   # height in image
            bad = (i2["reproj_px_median"] > cfg.keyframe_reproj_frac * person_px
                   or i2["reproj_px_max"] > cfg.keyframe_reproj_max_frac * person_px
                   or cost > cfg.keyframe_cost_jump * max(prev_cost, 1.0))
            self.stats["degraded"] += int(bad)
            due = self.i - self.last_kf >= cfg.keyframe_every or bad or detect
            if due and not (self.worker is not None and self.worker.busy):
                self.last_kf = self.i
                self.stats["keyframes"] += 1
                if self.worker is not None:
                    self.worker.submit(copy.deepcopy(frame), img_bgr, K, self._kf_kwargs(dict(self.state)))
                else:
                    try:
                        kf = self._keyframe_now(copy.deepcopy(frame), img_bgr, K, dict(self.state))
                    except Exception as e:
                        self.last_s2_error, kf = repr(e), None
                    if (kf is not None and self._sane_fit(kf.joints3d, kf.conf)
                            and not str(kf.s2_info.get("init", "")).startswith("track")):
                        frame, kind = kf, "keyframe"
                        self.state = self._state_from(frame)
            self.costs.append(cost)
        T["s2"] = 1e3 * (time.perf_counter() - t0)
        # ---- S3
        t0 = time.perf_counter()
        from .s3_load import static_load
        air = bool(frame.s2_info.get("airborne"))
        static_load(frame, contacts=[] if air else (frame.s2_info.get("contacts") or None),
                    g_scale=0.0 if air else 1.0)
        self.state["contacts"] = [] if air else (list(frame.contacts) or self.state["contacts"])
        if cfg.dynamic:
            self.floor.update(frame.joints3d, frame.valid3d, self.state["pitch"], self.state["contacts"], air, t)
        self.stats["airborne"] += int(air)
        T["s3"] = 1e3 * (time.perf_counter() - t0)
        # ---- S4
        t0 = time.perf_counter()
        from .s4_gesture import build_croquis
        try:
            cq = build_croquis(frame, layers=cfg.s4_layers, prefer_loa=self.loa_name)
            self.loa_name = cq.loa["name"]
            self.last_croquis = cq
        except Exception as e:                          # an odd frame must not end the stream
            self.stats["s4_failed"] += 1
            self.last_s4_error = repr(e)
            cq, kind = self.last_croquis, "s4-failed"
        T["s4"] = 1e3 * (time.perf_counter() - t0)
        T["total"] = sum(T.values())
        self.i += 1
        self.stats[kind] += 1
        return LiveResult(frame, cq, kind, detect, T)


# ---------------------------------------------------------------------- CLI
class LatestFrame:
    """Camera reader thread that keeps only the newest frame: the pipeline always works on 'now'
    instead of a growing queue of old frames, so the delay stays ~ one processing time."""

    def __init__(self, cap):
        import threading
        self.cap, self.frame, self.n_read, self.n_taken = cap, None, 0, 0
        self.lock, self.new = threading.Lock(), threading.Event()
        self.running = True
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        while self.running:
            ok, img = self.cap.read()
            if not ok:
                self.running = False
                self.new.set()
                break
            with self.lock:
                self.frame, self.t = img, time.perf_counter()
                self.n_read += 1
            self.new.set()

    def read(self):
        self.new.wait()
        with self.lock:
            img, t = self.frame, getattr(self, "t", None)
            self.frame = None
            self.new.clear()
        if img is None:
            return False, None, None
        self.n_taken += 1
        return True, img, t

    def dropped(self):
        return self.n_read - self.n_taken

    def stop(self, timeout=2.0):
        """Stop and WAIT for the grabber thread: releasing the camera while another thread is
        inside cap.read() segfaults on macOS (AVFoundation)."""
        self.running = False
        self.th.join(timeout)
        return not self.th.is_alive()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", type=int)
    src.add_argument("--video")
    ap.add_argument("--out", help="write the croquis video here (.mp4)")
    ap.add_argument("--show", action="store_true", help="show a window (q to quit)")
    ap.add_argument("--overlay", action="store_true", help="draw over the (faded) frame instead of paper")
    ap.add_argument("--theta", type=float, default=0.0,
                    help="draw every frame from a view rotated by this many degrees about the vertical axis "
                         "through the pelvis (+ = the body turns to its own left; 90 = side view). "
                         "Debug view only; --overlay is ignored when theta != 0")
    # debug keys in the --show window: space pauses, r (while paused) rotates the view 45 deg,
    # q quits. Flight frames are drawn on a green background (this debug viewer only).
    ap.add_argument("--preview-width", type=int, default=960, help="drawing width in px (0 = full frame)")
    ap.add_argument("--height", type=float, default=1.70)
    ap.add_argument("--mode", default="balanced", choices=["lightweight", "balanced", "performance"],
                    help="S1 models (detector + RTMPose): lightweight = YOLOX-tiny + RTMPose-s, "
                         "balanced = YOLOX-m + RTMPose-m, performance = YOLOX-x + RTMPose-x")
    ap.add_argument("--det-mode", default="lightweight", choices=["lightweight", "balanced", "performance"],
                    help="detector only (re-detection frames): lightweight = YOLOX-tiny (default for video)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--depth-device", default=None)
    ap.add_argument("--static", action="store_true",
                    help="held poses: keep the static-balance and level-floor priors (default for video: off)")
    ap.add_argument("--keyframe-every", type=int, default=30)
    ap.add_argument("--redetect-every", type=int, default=90)
    ap.add_argument("--worker-threads", type=int, default=2)
    ap.add_argument("--sync-keyframes", action="store_true", help="run keyframes in the frame loop (stalls)")
    ap.add_argument("--no-smooth", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()
    import cv2
    cfg = LiveConfig(keyframe_every=a.keyframe_every, redetect_every=a.redetect_every,
                     async_keyframes=not a.sync_keyframes, smooth=not a.no_smooth, dynamic=not a.static,
                     worker_threads=a.worker_threads)
    cap = cv2.VideoCapture(a.camera if a.camera is not None else a.video)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pipe = LivePipeline(cfg, height=a.height, mode=a.mode, device=a.device, depth_device=a.depth_device,
                        det_mode=a.det_mode)
    print("[live] models:")
    for k, v in pipe.models.items():
        print(f"         {k:40s} {v}")
    print(f"[live] {'dynamic' if cfg.dynamic else 'static'} mode, keyframes "
          f"{'in a worker process' if cfg.async_keyframes else 'in the frame loop'}")
    try:
        _loop(a, cap, pipe, fps_in, cv2)
    finally:
        pipe.close()


AIR_BG = (200, 235, 200)             # BGR, light green: debug view only (flight frames)


def _draw(a, res, img, cv2, yaw=0.0):
    """Debug drawing of one result. Flight frames get a green background (live.py only: the
    renderer of the real app draws its own). yaw != 0 shows the pose from a rotated view
    (rebuilt from the frame's 3D skeleton), fitted to the canvas on plain paper."""
    from .s4_gesture import render_preview, build_croquis, View
    h, w = img.shape[:2]
    scale = min(1.0, a.preview_width / w) if a.preview_width else 1.0
    ph, pw = int(round(h * scale)), int(round(w * scale))
    air = bool((res.frame.s2_info or {}).get("airborne"))
    paper = np.full((ph, pw, 3), AIR_BG if air else (250, 250, 250), np.uint8)
    if yaw:
        try:
            prev = res.croquis.loa.get("name") if res.croquis is not None else None
            cq = build_croquis(res.frame, view=View(yaw_deg=yaw), layers="AB", prefer_loa=prev)
            out = render_preview(cq, (ph, pw), bg=paper, fit=True, supersample=1)
        except Exception as e:                       # no 3D skeleton on this frame, etc.
            out = paper
            cv2.putText(out, f"cannot rotate: {e!r}"[:80], (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (40, 40, 200), 1, cv2.LINE_AA)
        return out
    if a.overlay:                                   # blend at preview size, in uint8
        small = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA) if scale < 1 else img
        bg = cv2.addWeighted(small, 0.35, paper, 0.65, 0)
    else:
        bg = paper
    return (render_preview(res.croquis, (ph, pw), bg=bg, fit=False, supersample=1, view_scale=scale)
            if res.croquis is not None else bg)


def _label(out, res, cv2, paused=False, yaw=0.0):
    air = bool((res.frame.s2_info or {}).get("airborne"))
    txt = f"{res.kind}{' +det' if res.detected else ''}{'  AIRBORNE' if air else ''}  {res.timings['total']:.0f} ms"
    i2 = res.frame.s2_info or {}
    if i2.get("foot_gap_m") is not None:
        txt += f"  gap {100 * i2['foot_gap_m']:+.0f}/{100 * i2['lift_m']:.0f} cm"
    if paused:
        txt += f"   PAUSED  yaw {yaw:+.0f}  (space: resume, r: rotate 45, q: quit)"
    elif yaw:
        txt += f"   yaw {yaw:+.0f}"
    cv2.putText(out, txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 2, cv2.LINE_AA)
    return out


def _paused(a, res, img, cv2):
    """Space pressed: hold this frame. r rotates the view by 45 deg about the vertical axis
    through the pelvis, starting from --theta; space resumes (back to --theta), q quits.
    Returns 'quit' or None."""
    yaw = float(a.theta) % 360.0
    redraw = True
    while True:
        if redraw:
            cv2.imshow("CroquisGen live", _label(_draw(a, res, img, cv2, yaw), res, cv2, True, yaw))
            redraw = False
        k = cv2.waitKey(30) & 0xFF
        if k == ord(" "):
            return None
        if k == ord("q"):
            return "quit"
        if k == ord("r"):
            yaw = (yaw + 45.0) % 360.0
            redraw = True


def _loop(a, cap, pipe, fps_in, cv2):
    grab = LatestFrame(cap) if a.camera is not None else None      # camera: always the newest frame
    writer = None
    times = collections.defaultdict(list)
    n = 0
    paused_s = 0.0
    t_start = time.perf_counter()
    while True:
        t0 = time.perf_counter()
        if grab is not None:
            ok, img, t_cap = grab.read()
            t = t_cap
        else:
            ok, img = cap.read()
            t = n / fps_in
        if not ok or (a.max_frames and n >= a.max_frames):
            break
        times["capture"].append(1e3 * (time.perf_counter() - t0))
        res = pipe.process(img, t=t)
        for k, v in res.timings.items():
            times[k].append(v)
        t1 = time.perf_counter()
        yaw = float(a.theta) % 360.0
        out = _label(_draw(a, res, img, cv2, yaw), res, cv2, yaw=yaw)
        ph, pw = out.shape[:2]
        times["draw"].append(1e3 * (time.perf_counter() - t1))
        t2 = time.perf_counter()
        if a.out:
            if writer is None:
                writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (pw, ph))
            writer.write(out)
        if a.show:
            cv2.imshow("CroquisGen live", out)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord(" "):                           # pause (debug): hold, rotate with r
                tp = time.perf_counter()
                if _paused(a, res, img, cv2) == "quit":
                    break
                paused_s += time.perf_counter() - tp
                times["output"].append(1e3 * (tp - t2))
                n += 1
                continue                                # the paused time is not a frame delay
        times["output"].append(1e3 * (time.perf_counter() - t2))
        if grab is not None:
            times["delay"].append(1e3 * (time.perf_counter() - t))   # capture -> drawn
        n += 1
    released = grab.stop() if grab is not None else True
    if released:                     # thread still stuck in read(): let process exit free the camera
        cap.release()
    if writer is not None:
        writer.release()
    if a.show:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    wall = time.perf_counter() - t_start - paused_s
    print(f"{n} frames in {wall:.1f} s ({n / max(wall, 1e-9):.1f} fps incl. capture and drawing)"
          + (f", {grab.dropped()} stale camera frames skipped" if grab is not None else ""))
    for k in ("capture", "s1", "s2", "s3", "s4", "total", "draw", "output", "delay"):
        if times.get(k):
            v = np.array(times[k])
            label = {"total": "S1-S4", "delay": "capture->shown"}.get(k, k)
            print(f"  {label:14s} median {np.median(v):6.1f} ms   p95 {np.percentile(v, 95):6.1f} ms")
    print("  frames:", dict(pipe.stats))
    fl = pipe.fail_log
    if fl:
        top = collections.Counter(j for r in fl for j in r["joints"]).most_common(6)
        n_re = sum(bool(r["reappeared"]) for r in fl)
        n_lo = sum(bool(r["low_conf"]) for r in fl)
        n_rep = sum(r["repaired"] for r in fl)
        print(f"  diverged fits: {len(fl)} ({n_rep} repaired). joints: {dict(top)}")
        print(f"    with a joint that had just reappeared: {n_re}, with a low-confidence joint: {n_lo}, "
              f"solver status: {dict(collections.Counter(r['lm_status'] for r in fl))}")
    if pipe.cfg.dynamic:
        nz = pipe.floor.noise_m
        print(f"  floor: {'%.3f m' % pipe.floor.h if pipe.floor.h is not None else 'not seated'} "
              f"(re-seated {pipe.floor.resets}x), flight frames {pipe.stats['airborne']}, "
              f"planted-foot noise {'%.1f cm' % (100 * nz) if nz is not None else 'n/a'}, "
              f"lift threshold {100 * pipe.floor.lift_m:.1f} cm")
    for k in ("last_s2_error", "last_s4_error"):
        if getattr(pipe, k, None):
            print(f"  {k}: {getattr(pipe, k)}")


if __name__ == "__main__":
    main()

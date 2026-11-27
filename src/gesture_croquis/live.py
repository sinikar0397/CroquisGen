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
    kpt_conf: float = 0.30            # keypoints that define the ROI
    # 2D smoothing (One-Euro; px, Hz)
    smooth: bool = True
    min_cutoff: float = 1.5
    beta: float = 0.01
    # S2
    keyframe_every: int = 30          # full re-solve period (frames)
    keyframe_reproj_px: float = 12.0  # ... or when the tracked 2D error (median) exceeds this,
    keyframe_reproj_max_px: float = 48.0   # ... or one joint is this far off (a limb lost),
    keyframe_cost_jump: float = 2.5   # ... or the energy jumps by this factor from one frame to the next.
                                      # (Not vs. a running level: a harder pose, e.g. sinking into a
                                      # lunge, raises the energy legitimately.)
    track_bonus: float = 0.15         # hysteresis: a re-solve must beat the track by 15%
    track_iters: int = 40
    async_keyframes: bool = True      # keyframes in a worker process (False: in the frame loop, stalls)
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

    def need_detect(self) -> bool:
        return self.box is None or self.lost or self.since_det >= self.cfg.redetect_every

    def update(self, kpts, conf, img_hw, detected: bool, det_box=None):
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
            self.box = None if det_box is None else np.asarray(det_box, float)
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
        self.box = np.array([max(0, lo[0]), max(0, lo[1]), min(W - 1, hi[0]), min(H - 1, hi[1])], float)


def _keyframe_worker(conn, lifter_kwargs):
    """Worker process: its own Lifter3D (with RTMW3D); solves keyframes sent over the pipe."""
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

    def __init__(self, lifter_kwargs, timeout=600):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")                # JAX / ONNX Runtime are not fork-safe
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_keyframe_worker, args=(child, lifter_kwargs), daemon=True)
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


class LivePipeline:
    """S1-S4 on a video stream, with ROI tracking and hysteresis. Models load once.
    s1_model / lifter can be injected (tests, or a different S1)."""

    def __init__(self, cfg: LiveConfig | None = None, height=1.70, s1="halpe26", mode="balanced",
                 device="cpu", depth_device=None, K=None, s1_model=None, lifter_kwargs=None, s2_backend="rtmw3d"):
        """lifter_kwargs: Lifter3D arguments for keyframes (default: RTMW3D depth, auto sign).
        The frame loop's own Lifter3D only tracks, so it never loads RTMW3D."""
        self.cfg = cfg or LiveConfig()
        self.height = height
        if s1_model is None:
            from .s1_keypoints import RTMPose2D
            s1_model = RTMPose2D(s1, mode=mode, device=device)
        kf_kw = dict(backend=s2_backend, height=height, z_sign="auto", device=device, depth_device=depth_device,
                     workers=1)
        kf_kw.update(lifter_kwargs or {})
        from .s2_lift import Lifter3D
        if self.cfg.async_keyframes:
            self.worker = KeyframeWorker(kf_kw)                       # starts loading in parallel
            self.s2 = Lifter3D(**dict(kf_kw, backend="geometric"))    # tracking only: no RTMW3D here
        else:
            self.worker = None
            self.s2 = Lifter3D(**kf_kw)
        self.s1 = s1_model
        self.K = K
        self.roi = ROITracker(self.cfg)
        self.euro = OneEuro(self.cfg.min_cutoff, self.cfg.beta)
        self.i = 0
        self.state = None                 # dict(X, scales, pitch, contacts, sign, cons, cost)
        self.loa_name = None
        self.costs = collections.deque(maxlen=30)
        self.last_kf = -10 ** 9
        self.stats = collections.Counter()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _scales(info, height):
        return np.r_[np.log(info["fitted_height_m"] / height),
                     np.log([info["proportions"][k] for k in ("legs", "arms", "torso")])]

    def _state_from(self, frame):
        i = frame.s2_info
        return dict(X=frame.joints3d.copy(), scales=self._scales(i, self.height),
                    pitch=(i.get("physics_p") or [0.0])[0], contacts=list(i.get("contacts", [])),
                    sign=i.get("z_sign") or (self.state or {}).get("sign"), cost=float(i["cost"]),
                    cons=(self.state or {}).get("cons"))

    def _kf_kwargs(self, track):
        if track is None:
            return {}
        return dict(X_track=track["X"], track_sign=track["sign"], track_bonus=self.cfg.track_bonus,
                    track_scales=track["scales"], track_pitch=track["pitch"], contacts_prev=track["contacts"])

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
        # ---- S1 with ROI
        t0 = time.perf_counter()
        detect = self.roi.need_detect()
        r = self.s1(img_bgr) if detect else self.s1(img_bgr, bbox=self.roi.box)
        T["s1"] = 1e3 * (time.perf_counter() - t0)
        k2, c = r["kpts"].astype(float), r["conf"].astype(float)
        if cfg.smooth:
            k2 = self.euro(k2, t, c > 0.05)
        self.roi.update(k2, c, hw, detect, det_box=r.get("bbox"))
        self.stats["detect" if detect else "roi"] += 1
        frame = PoseFrame(img_hw=hw, kpts2d=k2.astype(np.float32), conf=c.astype(np.float32),
                          bbox=r.get("bbox") if detect else self.roi.box, s1_layout=r.get("layout", ""),
                          body_height_m=self.height)
        # ---- S2
        t0 = time.perf_counter()
        kind = "track"
        if self.state is None or (self.roi.lost and detect):      # no track: synchronous keyframe
            frame = self._keyframe_now(frame, img_bgr, K, None)
            kind, self.last_kf = "keyframe", self.i
            self.state = self._state_from(frame)
        else:
            st = self.state
            self.s2.track(frame, K, st["X"], scales_p0=st["scales"], pitch=st["pitch"],
                          contacts_prev=st["contacts"], order_cons=st["cons"], iters=cfg.track_iters)
            cost = float(frame.s2_info["cost"])
            # a finished background keyframe: re-fit its winner to THIS frame, switch with hysteresis
            kf = self.worker.poll() if self.worker is not None else None
            if kf is not None and not str(kf.s2_info.get("init", "")).startswith("track"):
                alt = copy.deepcopy(frame)
                ki = kf.s2_info
                self.s2.track(alt, K, kf.joints3d, scales_p0=self._scales(ki, self.height),
                              pitch=(ki.get("physics_p") or [0.0])[0], contacts_prev=ki.get("contacts"),
                              order_cons=None, iters=cfg.track_iters)
                self.stats["keyframe_proposals"] += 1
                if float(alt.s2_info["cost"]) < cost * (1 - cfg.track_bonus):
                    frame, cost, kind = alt, float(alt.s2_info["cost"]), "track+switch"
                    self.stats["switches"] += 1
                    self.state = self._state_from(frame)
                    self.state["sign"] = ki.get("z_sign") or self.state["sign"]
            self.state = dict(self._state_from(frame), sign=self.state["sign"], cons=self.state["cons"])
            # schedule the next keyframe
            prev_cost = self.costs[-1] if self.costs else np.inf
            i2 = frame.s2_info
            bad = (i2["reproj_px_median"] > cfg.keyframe_reproj_px or i2["reproj_px_max"] > cfg.keyframe_reproj_max_px
                   or cost > cfg.keyframe_cost_jump * max(prev_cost, 1.0))
            self.stats["degraded"] += int(bad)
            due = self.i - self.last_kf >= cfg.keyframe_every or bad or detect
            if due and not (self.worker is not None and self.worker.busy):
                self.last_kf = self.i
                self.stats["keyframes"] += 1
                if self.worker is not None:
                    self.worker.submit(copy.deepcopy(frame), img_bgr, K, self._kf_kwargs(dict(self.state)))
                else:
                    kf = self._keyframe_now(copy.deepcopy(frame), img_bgr, K, dict(self.state))
                    if not str(kf.s2_info.get("init", "")).startswith("track"):
                        frame, kind = kf, "keyframe"
                        self.state = self._state_from(frame)
            self.costs.append(cost)
        T["s2"] = 1e3 * (time.perf_counter() - t0)
        # ---- S3
        t0 = time.perf_counter()
        from .s3_load import static_load
        static_load(frame, contacts=frame.s2_info.get("contacts") or None)
        self.state["contacts"] = list(frame.contacts) or self.state["contacts"]
        T["s3"] = 1e3 * (time.perf_counter() - t0)
        # ---- S4
        t0 = time.perf_counter()
        from .s4_gesture import build_croquis
        cq = build_croquis(frame, layers=cfg.s4_layers, prefer_loa=self.loa_name)
        self.loa_name = cq.loa["name"]
        T["s4"] = 1e3 * (time.perf_counter() - t0)
        T["total"] = sum(T.values())
        self.i += 1
        self.stats[kind] += 1
        return LiveResult(frame, cq, kind, detect, T)


# ---------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", type=int)
    src.add_argument("--video")
    ap.add_argument("--out", help="write the croquis video here (.mp4)")
    ap.add_argument("--show", action="store_true", help="show a window (q to quit)")
    ap.add_argument("--overlay", action="store_true", help="draw over the (faded) frame instead of paper")
    ap.add_argument("--height", type=float, default=1.70)
    ap.add_argument("--mode", default="balanced", choices=["lightweight", "balanced", "performance"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--depth-device", default=None)
    ap.add_argument("--keyframe-every", type=int, default=30)
    ap.add_argument("--redetect-every", type=int, default=90)
    ap.add_argument("--sync-keyframes", action="store_true", help="run keyframes in the frame loop (stalls)")
    ap.add_argument("--no-smooth", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()
    import cv2
    from .s4_gesture import render_preview
    cfg = LiveConfig(keyframe_every=a.keyframe_every, redetect_every=a.redetect_every,
                     async_keyframes=not a.sync_keyframes, smooth=not a.no_smooth)
    cap = cv2.VideoCapture(a.camera if a.camera is not None else a.video)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pipe = LivePipeline(cfg, height=a.height, mode=a.mode, device=a.device, depth_device=a.depth_device)
    times = collections.defaultdict(list)
    try:
        _loop(a, cap, pipe, times, fps_in, render_preview, cv2)
    finally:
        pipe.close()


def _loop(a, cap, pipe, times, fps_in, render_preview, cv2):
    writer = None
    n = 0
    t_start = time.perf_counter()
    while True:
        ok, img = cap.read()
        if not ok or (a.max_frames and n >= a.max_frames):
            break
        t = n / fps_in if a.video else None
        res = pipe.process(img, t=t)
        for k, v in res.timings.items():
            times[k].append(v)
        h, w = img.shape[:2]
        bg = (img.astype(np.float32) * 0.35 + 165).astype(np.uint8) if a.overlay else np.full_like(img, 250)
        out = render_preview(res.croquis, (h, w), bg=bg, fit=False, supersample=1)
        cv2.putText(out, f"{res.kind}{' +det' if res.detected else ''}  {res.timings['total']:.0f} ms",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2, cv2.LINE_AA)
        if a.out:
            if writer is None:
                writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
            writer.write(out)
        if a.show:
            cv2.imshow("CroquisGen live", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        n += 1
    cap.release()
    if writer is not None:
        writer.release()
    wall = time.perf_counter() - t_start
    print(f"{n} frames in {wall:.1f} s ({n / max(wall, 1e-9):.1f} fps incl. drawing)")
    for k, v in times.items():
        v = np.array(v)
        print(f"  {k:6s} median {np.median(v):6.1f} ms   p95 {np.percentile(v, 95):6.1f} ms")
    print("  frames:", dict(pipe.stats))


if __name__ == "__main__":
    main()

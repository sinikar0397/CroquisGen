"""
S1: person detection + 2D keypoints with RTMPose (via rtmlib, no mmcv).

Models are created once and reused, so the same object serves single images now
and a webcam loop later (call .detect_bbox only every N frames and track in between).

Default = RTMPose-m Halpe-26 (body + feet), trained on the Body7 mix.
"""
from __future__ import annotations
import time
import numpy as np

from .skeleton import CONVERTERS

MODELS = {
    # name: (rtmlib solution class, canonical layout)
    "halpe26": ("BodyWithFeet", "halpe26"),
    "coco17": ("Body", "coco17"),
    "wholebody": ("Wholebody", "wholebody133"),
}


def pick_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        import onnxruntime as ort
        prov = ort.get_available_providers()
    except ImportError:
        return "cpu"
    if "CUDAExecutionProvider" in prov:
        return "cuda"
    if "CoreMLExecutionProvider" in prov:
        return "mps"            # rtmlib maps 'mps' -> CoreMLExecutionProvider
    return "cpu"


class RTMPose2D:
    """Wraps rtmlib's YOLOX detector + RTMPose top-down estimator."""

    def __init__(self, model: str = "halpe26", mode: str = "balanced",
                 device: str = "auto", backend: str = "onnxruntime", det_mode: str | None = None):
        """det_mode: a different rtmlib mode for the person detector only (e.g. "lightweight" =
        YOLOX-tiny while RTMPose stays at `mode`). Video uses it: the detector runs rarely there,
        but when it does it is on the frames with the fastest motion."""
        import rtmlib
        cls_name, self.layout = MODELS[model]
        cfg = dict(getattr(rtmlib, cls_name).MODE[mode])
        if det_mode and det_mode != mode:
            dcfg = getattr(rtmlib, cls_name).MODE[det_mode]
            cfg["det"], cfg["det_input_size"] = dcfg["det"], dcfg["det_input_size"]
        self.device = pick_device(device)
        self.mode = mode
        self._rtmlib, self._cfg, self._backend = rtmlib, cfg, backend
        try:
            self._build(rtmlib, cfg, backend, self.device)
        except Exception as e:  # CoreML EP occasionally rejects ops -> fall back to CPU
            if self.device == "cpu":
                raise
            print(f"[S1] device '{self.device}' failed ({e!s:.120}); falling back to CPU")
            self.device = "cpu"
            self._build(rtmlib, cfg, backend, "cpu")

    def _build(self, rtmlib, cfg, backend, device):
        self.det = rtmlib.YOLOX(cfg["det"], model_input_size=cfg["det_input_size"],
                                backend=backend, device=device)
        self.pose = rtmlib.RTMPose(cfg["pose"], model_input_size=cfg["pose_input_size"],
                                   to_openpose=False, backend=backend, device=device)

    # ------------------------------------------------------------------
    def detect_bboxes(self, img_bgr: np.ndarray) -> np.ndarray:
        b = np.asarray(self.det(img_bgr), np.float32)
        return b.reshape(-1, 4)

    def __call__(self, img_bgr: np.ndarray, bbox: np.ndarray | None = None):
        try:
            return self._call(img_bgr, bbox)
        except Exception as e:
            # CoreML EP can fail at first *run* (e.g. YOLOX output-shape bug), not at load
            if self.device == "cpu":
                raise
            print(f"[S1] inference on '{self.device}' failed ({e!s:.100}); rebuilding on CPU")
            self.device = "cpu"
            self._build(self._rtmlib, self._cfg, self._backend, "cpu")
            return self._call(img_bgr, bbox)

    def _call(self, img_bgr: np.ndarray, bbox: np.ndarray | None = None):
        """Return dict for the most prominent person (or the given bbox):
        kpts (J,2) canonical, conf (J,), bbox (4,), raw_kpts, raw_conf, timings."""
        t0 = time.perf_counter()
        if bbox is None:
            boxes = self.detect_bboxes(img_bgr)
            if len(boxes) == 0:
                h, w = img_bgr.shape[:2]
                boxes = np.array([[0, 0, w, h]], np.float32)   # no detection: whole image
        else:
            boxes = np.asarray(bbox, np.float32).reshape(-1, 4)
        t1 = time.perf_counter()
        kpts, scores = self.pose(img_bgr, bboxes=list(boxes))          # (N,K,2), (N,K)
        t2 = time.perf_counter()
        # SimCC scores are max logits (not probabilities); rtmlib's own drawing uses ~0.3-0.5
        # thresholds, so a plain clip keeps them comparable across images. Raw values are kept.
        conf = np.clip(scores, 0, 1)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        i = int(np.argmax(conf.mean(1) * np.sqrt(np.maximum(area, 1))))
        k, c = CONVERTERS[self.layout](kpts[i].astype(np.float32), conf[i].astype(np.float32))
        return dict(kpts=k, conf=c, bbox=boxes[i], raw_kpts=kpts[i], raw_conf=scores[i],
                    layout=self.layout, n_people=len(boxes),
                    timings=dict(det_ms=1e3 * (t1 - t0), pose_ms=1e3 * (t2 - t1)))

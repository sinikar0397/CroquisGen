"""PoseFrame: the single data contract passed between pipeline stages (cached as .npz)."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json
import numpy as np

from .skeleton import JOINTS


@dataclass
class PoseFrame:
    img_hw: tuple                      # (H, W)
    kpts2d: np.ndarray                 # (J,2) canonical layout, image px
    conf: np.ndarray                   # (J,) 0..1, 0 = missing
    bbox: np.ndarray | None = None     # (4,) xyxy of the selected person
    s1_layout: str = ""                # raw layout the S1 model produced
    raw_kpts2d: np.ndarray | None = None
    raw_conf: np.ndarray | None = None
    raw_kpts3d_rtmw3d: np.ndarray | None = None  # (133,3) RTMW3D output, if used
    # ---- S2 ----
    joints3d: np.ndarray | None = None  # (J,3) metres, CAMERA frame (x right, y down, z forward)
    valid3d: np.ndarray | None = None   # (J,) bool, joints that were actually fitted
    camera_K: np.ndarray | None = None  # (3,3)
    camera_R: np.ndarray = field(default_factory=lambda: np.eye(3))
    camera_t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gravity_dir: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))  # camera frame
    body_height_m: float = 1.70
    s2_backend: str = ""
    s2_info: dict = field(default_factory=dict)  # fit diagnostics
    spine3d: np.ndarray | None = None   # (N,3) camera frame, pelvis -> neck -> head (interpolated)
    # ---- S3 ----
    body_mass_kg: float = 0.0
    com3d: np.ndarray | None = None     # (3,) camera frame
    contacts: list = field(default_factory=list)
    joint_torque: dict = field(default_factory=dict)   # joint -> {flex, lateral, twist, norm} N·m
    joint_effort: dict = field(default_factory=dict)   # joint -> 0..1 (normalised |torque|/capacity)
    muscle_stretch: dict = field(default_factory=dict)  # joint angles + torso side strain
    s3_info: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def joint(self, name: str) -> np.ndarray:
        return self.joints3d[JOINTS.index(name)]

    _FLIP = np.array([1.0, -1.0, -1.0])                  # camera(y down, z fwd) <-> y up, z back

    def _level(self, P):
        """Camera coords -> gravity-levelled camera coords (camera_R holds the fitted pitch)."""
        R = np.asarray(self.camera_R, float).reshape(3, 3)
        return P @ R.T

    def _world_offset(self) -> np.ndarray:
        P = self._level(self.joints3d) * self._FLIP
        v = self.valid3d if self.valid3d is not None else np.ones(len(P), bool)
        feet = [JOINTS.index(n) for n in ("lank", "rank", "lheel", "rheel", "lbigtoe", "rbigtoe")]
        feet = [f for f in feet if v[f]]
        ground = P[feet, 1].min() if feet else P[v, 1].min()
        pel = P[JOINTS.index("pelvis")]
        return np.array([pel[0], ground, pel[2]])

    def to_world(self, pts_cam: np.ndarray) -> np.ndarray:
        """Y-up world frame: origin on the ground (lowest valid foot point) under the pelvis,
        x = camera right, y = up (against gravity, assumed camera-down), z = toward the camera."""
        return self._level(pts_cam) * self._FLIP - self._world_offset()

    def from_world(self, pts_w: np.ndarray) -> np.ndarray:
        R = np.asarray(self.camera_R, float).reshape(3, 3)
        return ((pts_w + self._world_offset()) * self._FLIP) @ R

    def world_joints(self) -> np.ndarray:
        return self.to_world(self.joints3d)

    def save(self, path: str):
        d = {}
        for k, v in asdict(self).items():
            if v is None:
                continue
            if isinstance(v, (dict, list)):
                d[k] = np.array(json.dumps(v, default=float))
            else:
                d[k] = np.asarray(v)
        d["joint_names"] = np.array(JOINTS)
        np.savez_compressed(path, **d)

    @classmethod
    def load(cls, path: str) -> "PoseFrame":
        z = np.load(path, allow_pickle=False)
        kw = {}
        for f in cls.__dataclass_fields__:
            if f not in z:
                continue
            v = z[f]
            if f in ("s2_info", "contacts", "joint_effort", "joint_torque", "muscle_stretch", "s3_info"):
                v = json.loads(str(v))
            elif f in ("s1_layout", "s2_backend"):
                v = str(v)
            elif f in ("body_height_m", "body_mass_kg"):
                v = float(v)
            elif f == "img_hw":
                v = tuple(int(x) for x in v)
            kw[f] = v
        return cls(**kw)

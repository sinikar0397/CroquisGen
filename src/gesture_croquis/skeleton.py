"""
Canonical CroquisGen skeleton + mappings from the RTMPose-family keypoint layouts.

Canonical joints = Halpe-26 order (so RTMPose-Halpe26 output maps 1:1) + two hand points.
All S2+ stages work on this layout, whatever backend produced the keypoints.
"""
from __future__ import annotations
import numpy as np

JOINTS = [
    "nose", "leye", "reye", "lear", "rear",            # 0-4
    "lsho", "rsho", "lelb", "relb", "lwri", "rwri",    # 5-10
    "lhip", "rhip", "lknee", "rknee", "lank", "rank",  # 11-16
    "head", "neck", "pelvis",                          # 17-19  (Halpe: head top, neck, hip centre)
    "lbigtoe", "rbigtoe", "lsmalltoe", "rsmalltoe", "lheel", "rheel",  # 20-25
    "lhand", "rhand",                                  # 26-27  (middle-finger MCP, whole-body only)
]
J = len(JOINTS)
IDX = {n: i for i, n in enumerate(JOINTS)}


def side(name: str) -> str:
    """'L', 'R' or 'C' (centre) — used for colouring and symmetry terms."""
    if name in ("head", "neck", "pelvis", "nose"):
        return "C"
    return "L" if name.startswith("l") else "R"


# ----------------------------------------------------------------------------------------
# Bones: (a, b, length as fraction of body height, relative std of that length)
# Joint-centre-to-joint-centre values adapted from Winter (2009) / Drillis & Contini
# anthropometry, adjusted to where 2D detectors actually place keypoints
# (e.g. COCO hips sit wider than true hip joint centres).
# ----------------------------------------------------------------------------------------
BONES = [
    # torso (semi-rigid box + centre line)
    ("lsho", "rsho", 0.215, 0.12), ("lhip", "rhip", 0.130, 0.15),
    ("lsho", "lhip", 0.290, 0.10), ("rsho", "rhip", 0.290, 0.10),
    ("neck", "pelvis", 0.280, 0.10),
    ("neck", "lsho", 0.108, 0.15), ("neck", "rsho", 0.108, 0.15),
    ("pelvis", "lhip", 0.065, 0.15), ("pelvis", "rhip", 0.065, 0.15),
    # head / face (loose: faces vary and keypoints are noisy)
    ("neck", "nose", 0.110, 0.20), ("neck", "head", 0.170, 0.20), ("nose", "head", 0.110, 0.25),
    ("nose", "leye", 0.022, 0.30), ("nose", "reye", 0.022, 0.30), ("leye", "reye", 0.036, 0.25),
    ("leye", "lear", 0.045, 0.30), ("reye", "rear", 0.045, 0.30), ("lear", "rear", 0.085, 0.20),
    # arms
    ("lsho", "lelb", 0.186, 0.07), ("rsho", "relb", 0.186, 0.07),
    ("lelb", "lwri", 0.146, 0.07), ("relb", "rwri", 0.146, 0.07),
    ("lwri", "lhand", 0.050, 0.25), ("rwri", "rhand", 0.050, 0.25),
    # legs
    ("lhip", "lknee", 0.245, 0.06), ("rhip", "rknee", 0.245, 0.06),
    ("lknee", "lank", 0.246, 0.06), ("rknee", "rank", 0.246, 0.06),
    # feet
    ("lank", "lheel", 0.045, 0.25), ("rank", "rheel", 0.045, 0.25),
    ("lank", "lbigtoe", 0.130, 0.15), ("rank", "rbigtoe", 0.130, 0.15),
    ("lank", "lsmalltoe", 0.115, 0.15), ("rank", "rsmalltoe", 0.115, 0.15),
    ("lheel", "lbigtoe", 0.150, 0.15), ("rheel", "rbigtoe", 0.150, 0.15),
    ("lbigtoe", "lsmalltoe", 0.045, 0.30), ("rbigtoe", "rsmalltoe", 0.045, 0.30),
]
BONE_IDX = np.array([(IDX[a], IDX[b]) for a, b, _, _ in BONES])
BONE_FRAC = np.array([f for *_, f, _ in BONES])
BONE_SIG = np.array([s for *_, s in BONES])

# Left/right bone pairs for symmetry (same index offset: every L bone has an R twin)
SYM_PAIRS = []
for i, (a, b, *_ ) in enumerate(BONES):
    if side(a) == "L" or side(b) == "L":
        ra = "r" + a[1:] if side(a) == "L" else a
        rb = "r" + b[1:] if side(b) == "L" else b
        for k, (c, d, *_ ) in enumerate(BONES):
            if (c, d) == (ra, rb):
                SYM_PAIRS.append((i, k))
SYM_PAIRS = np.array(SYM_PAIRS)

# Bones drawn in visualisations (skip face clutter / redundant torso lines)
DRAW_BONES = [
    ("head", "neck"), ("neck", "pelvis"), ("lsho", "rsho"), ("lhip", "rhip"),
    ("lsho", "lhip"), ("rsho", "rhip"),
    ("nose", "leye"), ("nose", "reye"), ("leye", "lear"), ("reye", "rear"),
    ("lsho", "lelb"), ("lelb", "lwri"), ("lwri", "lhand"),
    ("rsho", "relb"), ("relb", "rwri"), ("rwri", "rhand"),
    ("lhip", "lknee"), ("lknee", "lank"), ("lank", "lheel"), ("lank", "lbigtoe"),
    ("lheel", "lbigtoe"), ("lbigtoe", "lsmalltoe"),
    ("rhip", "rknee"), ("rknee", "rank"), ("rank", "rheel"), ("rank", "rbigtoe"),
    ("rheel", "rbigtoe"), ("rbigtoe", "rsmalltoe"),
]

# ----------------------------------------------------------------------------------------
# Layout converters -> canonical (J,2|3) + (J,) confidence
# ----------------------------------------------------------------------------------------
# COCO-WholeBody 133: 0-16 body, 17-22 feet (L big, L small, L heel, R big, R small, R heel),
# 23-90 face, 91-111 left hand (91 = root, 100 = middle MCP), 112-132 right hand (121 = MCP)
_WB_FEET = {"lbigtoe": 17, "lsmalltoe": 18, "lheel": 19, "rbigtoe": 20, "rsmalltoe": 21, "rheel": 22}
_WB_HANDS = {"lhand": 100, "rhand": 121}


def _derive_centres(k: np.ndarray, c: np.ndarray, have_neck=False, have_pelvis=False, have_head=False):
    i = IDX
    if not have_neck:
        k[i["neck"]] = (k[i["lsho"]] + k[i["rsho"]]) / 2
        c[i["neck"]] = min(c[i["lsho"]], c[i["rsho"]])
    if not have_pelvis:
        k[i["pelvis"]] = (k[i["lhip"]] + k[i["rhip"]]) / 2
        c[i["pelvis"]] = min(c[i["lhip"]], c[i["rhip"]])
    if not have_head:
        face = [i["nose"], i["leye"], i["reye"], i["lear"], i["rear"]]
        w = c[face][:, None] + 1e-6
        f = (k[face] * w).sum(0) / w.sum()
        k[i["head"]] = f + (f - k[i["neck"]]) * 0.35      # push toward the crown
        c[i["head"]] = 0.5 * float(np.mean(c[face]))       # derived: lower trust
    return k, c


def from_halpe26(kpts: np.ndarray, conf: np.ndarray):
    D = kpts.shape[1]
    k = np.zeros((J, D), np.float32); c = np.zeros(J, np.float32)
    k[:26] = kpts; c[:26] = conf
    return k, c


def from_coco17(kpts: np.ndarray, conf: np.ndarray):
    D = kpts.shape[1]
    k = np.zeros((J, D), np.float32); c = np.zeros(J, np.float32)
    k[:17] = kpts; c[:17] = conf
    return _derive_centres(k, c)


def from_wholebody133(kpts: np.ndarray, conf: np.ndarray):
    D = kpts.shape[1]
    k = np.zeros((J, D), np.float32); c = np.zeros(J, np.float32)
    k[:17] = kpts[:17]; c[:17] = conf[:17]
    for n, s in {**_WB_FEET, **_WB_HANDS}.items():
        k[IDX[n]] = kpts[s]; c[IDX[n]] = conf[s]
    return _derive_centres(k, c)


CONVERTERS = {"halpe26": from_halpe26, "coco17": from_coco17, "wholebody133": from_wholebody133}

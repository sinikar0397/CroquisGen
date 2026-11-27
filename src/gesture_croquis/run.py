"""
CroquisGen S1+S2 runner: image -> RTMPose 2D keypoints -> 3D skeleton -> visualisation.

    python -m croquisgen.run --image photo.jpg                      # RTMPose Halpe-26 + RTMW3D depth
    python -m croquisgen.run --image photo.jpg --s2 geometric       # no 3D network (rough)
    python -m croquisgen.run --images photos/ --out out/            # batch a folder
    python -m croquisgen.run --demo                                 # synthetic pose, no models
    python -m croquisgen.run --image sit.jpg --contacts lheel,rheel,pelvis   # seated: override contacts

    python -m croquisgen.run --image photo.jpg --views 0,90,-45:20 --layers ABC   # S4 extra views

Stages: S1 RTMPose 2D -> S2 3D skeleton + spine curve -> S3 static load (T0) -> S4 gesture strokes.
Outputs per image in --out:  <stem>.npz (PoseFrame); with --viz_3d also <stem>_3d.html + <stem>_views.png
                             <stem>_croquis[_y<yaw>_p<pitch>].png / .json (S4 strokes per view)
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
import cv2
import numpy as np

from .poseframe import PoseFrame
from .s2_lift import Lifter3D, default_K, exif_focal_35mm, fit_skeleton, init_from_depth, project
from .skeleton import IDX, J
from .s3_load import static_load
from .s4_gesture import build_croquis, View, preview_panel, summary as summary_s4
from . import viz

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class Pipeline:
    """Holds the loaded models so repeated calls (batch now, webcam later) skip re-loading."""

    def __init__(self, s1="halpe26", mode="balanced", s2="rtmw3d", height=1.70, device="cpu",
                 z_sign="auto", s3=True, mass=None, contacts=None, sagittal=0.0, dynamic=False,
                 search=True, physical=True, label_swaps=True, workers=None, occlusion=True, jac="auto",
                 solver="auto", depth_device=None):
        from .s1_keypoints import RTMPose2D
        t = time.perf_counter()
        self.s1 = RTMPose2D(s1, mode=mode, device=device)
        self.s2 = Lifter3D(s2, height=height, z_sign=z_sign, device=device, dynamic=dynamic, search=search,
                           physical=physical, mass=mass, label_swaps=label_swaps, workers=workers,
                           occlusion=occlusion, jac=jac, solver=solver, depth_device=depth_device)
        self.s3 = dict(body_mass=mass, contacts=contacts, sagittal=sagittal) if s3 else None
        dd = getattr(self.s2.depth, "dev", None)
        print(f"[init] models ready in {time.perf_counter() - t:.1f}s  (S1 device: {self.s1.device}"
              + (f", RTMW3D device: {dd})" if dd else ")"))

    def __call__(self, img_bgr, K=None):
        r = self.s1(img_bgr)
        f = PoseFrame(img_hw=img_bgr.shape[:2], kpts2d=r["kpts"], conf=r["conf"], bbox=r["bbox"],
                      s1_layout=r["layout"], raw_kpts2d=r["raw_kpts"], raw_conf=r["raw_conf"])
        t = time.perf_counter()
        self.s2(f, img_bgr, K=K)
        f.s2_info.update(s1_det_ms=r["timings"]["det_ms"], s1_pose_ms=r["timings"]["pose_ms"],
                         s2_total_ms=1e3 * (time.perf_counter() - t), n_people=r["n_people"])
        if self.s3 is not None:
            t = time.perf_counter()
            kw = dict(self.s3)
            if kw.get("contacts") is None and f.s2_info.get("contacts"):
                kw["contacts"] = f.s2_info["contacts"]       # same contacts the integrated fit used
            static_load(f, **kw)
            f.s3_info["s3_ms"] = 1e3 * (time.perf_counter() - t)
        return f


class Timer:
    """Named wall-clock laps in ms:  with T("s4"): ...  ->  T.ms["s4"]"""

    def __init__(self):
        self.ms = {}

    def __call__(self, key):
        timer = self

        class _Lap:
            def __enter__(self):
                self.t = time.perf_counter()

            def __exit__(self, *exc):
                timer.ms[key] = timer.ms.get(key, 0.0) + 1e3 * (time.perf_counter() - self.t)
        return _Lap()


def write_outputs(frame, img, out_dir, stem, title, s4=None, timer=None, viz_3d=False):
    """<stem>.npz always; the 3D diagnostics (<stem>_3d.html, <stem>_views.png) only with viz_3d."""
    T = timer or Timer()
    os.makedirs(out_dir, exist_ok=True)
    with T("out_npz"):
        frame.save(os.path.join(out_dir, f"{stem}.npz"))
    if viz_3d:
        with T("out_html"):
            viz.make_html(frame, img, os.path.join(out_dir, f"{stem}_3d.html"), title=title)
        with T("out_views_png"):
            viz.make_png(frame, img, os.path.join(out_dir, f"{stem}_views.png"))
    if s4 is not None:
        write_croquis(frame, img, out_dir, stem, timer=T, **s4)
    return T


def write_croquis(frame, img, out_dir, stem, views=("0",), layers="AB", exaggeration=0.35, timer=None):
    """S4 for each requested view: <stem>_croquis<tag>.json (strokes) + .png (preview panel)."""
    T = timer or Timer()
    out = []
    for v in views:
        view = View.parse(v) if not isinstance(v, View) else v
        if frame.joints3d is None and not view.is_photo:
            continue
        with T("s4_build"):
            c = build_croquis(frame, view, layers=layers, exaggeration=exaggeration)
        for k, v_ in c.info["timings_ms"].items():          # S4 internal steps, summed over views
            if k != "total":
                T.ms["s4." + k] = T.ms.get("s4." + k, 0.0) + v_
        base = os.path.join(out_dir, f"{stem}_croquis{view.tag}")
        with T("s4_json"):
            c.save(base + ".json")
        with T("s4_preview_png"):
            cv2.imwrite(base + ".png", preview_panel(c, img))
        print(summary_s4(c))
        out.append(c)
    return out


def run_demo(out_dir, height, s3_kw=None, s4_kw=None, viz_3d=False):
    """Synthetic lunge rendered as a stick-figure 'photo'; simulated depth cue. Tests S2 + viz only."""
    from .synthetic import lunge_pose
    hw = (960, 1280)
    K = default_K(hw)
    X = lunge_pose(height=height)
    rng = np.random.default_rng(0)
    uv = project(X, K) + rng.normal(0, 2, (J, 2))
    conf = np.full(J, 0.9, np.float32)
    img = np.full((*hw, 3), 235, np.uint8)
    f = PoseFrame(img_hw=hw, kpts2d=uv.astype(np.float32), conf=conf,
                  bbox=np.r_[uv.min(0) - 40, uv.max(0) + 40], s1_layout="synthetic")
    z = (X[:, 2] - X[IDX["pelvis"], 2]) * 0.85 + rng.normal(0, 0.03, J)
    Xh, v, info = fit_skeleton(uv, conf, K, height, X0=init_from_depth(uv, conf, K, height, z, 1),
                               z_rel=z, z_conf=conf)
    f.joints3d, f.valid3d, f.camera_K, f.s2_info, f.s2_backend = Xh, v, K, info, "demo (simulated depth)"
    f.body_height_m = height
    if s3_kw is not None:
        t = time.perf_counter()
        static_load(f, **s3_kw)
        f.s3_info["s3_ms"] = 1e3 * (time.perf_counter() - t)
        print(summary_s3(f))
    t = time.perf_counter()
    T = write_outputs(f, img, out_dir, "demo", "CroquisGen S1+S2 · synthetic demo",
                      s4=s4_kw if f.s3_info else None, viz_3d=viz_3d)
    print(timing_report(f, T, 1e3 * (time.perf_counter() - t) + f.s3_info.get("s3_ms", 0.0)))
    print(json.dumps({k: v for k, v in info.items() if k != "hinge"}, indent=1, default=float))


def timing_report(f, T, total_ms):
    """One table per image: every stage, its share of the total, slowest first."""
    i = f.s2_info or {}
    ms = {"s1_detect": i.get("s1_det_ms"), "s1_pose": i.get("s1_pose_ms"), "s2_lift": i.get("s2_total_ms"),
          "s3_load": (f.s3_info or {}).get("s3_ms")}
    ms = {k: v for k, v in ms.items() if v is not None}
    ms.update({k: v for k, v in T.ms.items()})
    top = {k: v for k, v in ms.items() if not k.startswith("s4.")}
    lines = ["  timing (ms, slowest first; total %.0f):" % total_ms]
    for k, v in sorted(top.items(), key=lambda kv: -kv[1]):
        lines.append("    %-16s %8.1f  %5.1f%%  %s" % (k, v, 100 * v / max(total_ms, 1e-9), "#" * int(40 * v / max(total_ms, 1e-9))))
    sub = {k[3:]: v for k, v in ms.items() if k.startswith("s4.")}
    if sub:
        lines.append("    s4_build split: " + ", ".join("%s %.1f" % kv for kv in sorted(sub.items(), key=lambda kv: -kv[1])))
    return "\n".join(lines)


def summary_s3(f):
    i, ms = f.s3_info, f.muscle_stretch
    top = sorted(i["effort_fraction_of_capacity"].items(), key=lambda kv: -kv[1])[:4]
    bal = ("airborne" if i["com_offset_m"] is None else "balanced" if i["balanced"]
           else "COM %.2f m outside support" % i["com_offset_m"])
    return ("  S3: contacts %s%s | %s | top effort %s | spine bend %+.0f° twist %+.0f° | compressed side %s"
            % (",".join(f.contacts) or "-",
               " (+%s for balance)" % ",".join(i["contacts_added_for_balance"]) if i["contacts_added_for_balance"] else "",
               bal, ", ".join("%s %.2f" % kv for kv in top),
               ms["spine_lateral_bend"], ms["spine_twist"], ms["compressed_torso_side"].upper()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image"); ap.add_argument("--images", help="folder of images")
    ap.add_argument("--out", default="out")
    ap.add_argument("--s1", default="halpe26", choices=["halpe26", "wholebody", "coco17"])
    ap.add_argument("--mode", default="balanced", choices=["lightweight", "balanced", "performance"])
    ap.add_argument("--s2", default="rtmw3d", choices=["rtmw3d", "geometric"])
    ap.add_argument("--height", type=float, default=1.70, help="subject height prior, metres")
    ap.add_argument("--fov", type=float, default=55.0, help="horizontal FOV if no EXIF focal length")
    ap.add_argument("--z-sign", default="auto", choices=["auto", "+", "-"],
                    help="RTMW3D depth sign: 'auto' (default) fits both and keeps the better; '+'/'-' fixes it "
                         "(half the fits, but a wrong sign mirrors the whole pose in depth)")
    ap.add_argument("--depth-device", default=None,
                    help="device for the RTMW3D depth model only (default: same as --device). Try 'mps' "
                         "(CoreML): RTMW3D-x is ~0.22 s on the CPU. Falls back to CPU if CoreML rejects it")
    ap.add_argument("--jac", default="auto", choices=["auto", "jax", "fd"],
                    help="S2 Jacobian: exact via JAX (auto, if installed) or finite differences (fd, slow)")
    ap.add_argument("--device", default="cpu",
                    help="cpu (default: CoreML failed on these models and fell back anyway) | auto | mps | cuda")
    ap.add_argument("--mass", type=float, default=None, help="body mass kg (default: BMI 22 from height)")
    ap.add_argument("--contacts", default=None,
                    help="comma list overriding auto contacts, e.g. lheel,rbigtoe,pelvis (pelvis = seated)")
    ap.add_argument("--sagittal", type=float, default=0.0,
                    help="assumed front/back spine curvature (fraction of trunk length, + = flexed)")
    ap.add_argument("--no-s3", action="store_true")
    ap.add_argument("--no-search", action="store_true",
                    help="skip the plausibility search over depth flips (faster, less robust)")
    ap.add_argument("--no-physical", action="store_true",
                    help="skip the integrated S2+S3 physical fit (muscle relaxation, effort, floor)")
    ap.add_argument("--no-label-swaps", action="store_true",
                    help="trust the detector's left/right labels even where limbs overlap")
    ap.add_argument("--no-occlusion", action="store_true",
                    help="ignore body-part contours (limb-in-front/behind-torso order from image edges)")
    ap.add_argument("--workers", type=int, default=None, help="parallel fit processes (default: #cores, max 8)")
    ap.add_argument("--dynamic", action="store_true",
                    help="running/jumping/falling: disable the static-balance prior in the 3D fit")
    ap.add_argument("--no-s4", action="store_true", help="skip the S4 gesture strokes")
    ap.add_argument("--views", default="0",
                    help="S4 views, comma list of yaw[:pitch] degrees, e.g. 0,90,-45:20 (0 = photo camera)")
    ap.add_argument("--layers", default="AB",
                    help="S4 layers: A line of action, B gesture/rhythm, C construction volumes")
    ap.add_argument("--exaggeration", type=float, default=0.35, help="S4 line-of-action push (0.2-0.5)")
    ap.add_argument("--viz_3d", "--viz-3d", dest="viz_3d", action="store_true",
                    help="also write the 3D diagnostics: <stem>_3d.html (interactive) and <stem>_views.png "
                         "(front/side/top). Off by default: they took 0.4-1.3 s per photo")
    ap.add_argument("--solver", default="auto", choices=["auto", "lm", "scipy"],
                    help="S2 optimiser with the JAX Jacobian: lm (default, small n x n solves) or scipy (trf, "
                         "an SVD of the full Jacobian per step)")
    ap.add_argument("--profile", action="store_true",
                    help="cProfile each image: print the 25 slowest functions, save <out>/<stem>.prof")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()

    contacts = a.contacts.split(",") if a.contacts else None
    s3_kw = None if a.no_s3 else dict(body_mass=a.mass, contacts=contacts, sagittal=a.sagittal)
    s4_kw = None if a.no_s4 else dict(views=[v for v in a.views.split(",") if v.strip()], layers=a.layers,
                                      exaggeration=a.exaggeration)
    if a.demo:
        return run_demo(a.out, a.height, s3_kw, s4_kw, viz_3d=a.viz_3d)
    paths = [a.image] if a.image else sorted(str(p) for p in Path(a.images).iterdir()
                                             if p.suffix.lower() in IMG_EXT)
    t_init = time.perf_counter()
    pipe = Pipeline(a.s1, a.mode, a.s2, a.height, a.device, a.z_sign, s3=not a.no_s3,
                    mass=a.mass, contacts=contacts, sagittal=a.sagittal, dynamic=a.dynamic,
                    search=not a.no_search, physical=not a.no_physical,
                    label_swaps=not a.no_label_swaps, workers=a.workers, occlusion=not a.no_occlusion,
                    jac=a.jac, solver=a.solver, depth_device=a.depth_device)
    print(f"[init] total {1e3 * (time.perf_counter() - t_init):.0f} ms (model loading, not counted per image)")
    for p in paths:
        t_img = time.perf_counter()
        T = Timer()
        prof = None
        if a.profile:
            import cProfile
            prof = cProfile.Profile(); prof.enable()
        with T("read_image"):
            img = cv2.imread(p)
        if img is None:
            print(f"skip {p}: unreadable"); continue
        K = default_K(img.shape[:2], fov_deg=a.fov, focal_35mm=exif_focal_35mm(p))
        f = pipe(img, K=K)
        stem = Path(p).stem
        write_outputs(f, img, a.out, stem, f"CroquisGen S1+S2 · {Path(p).name}", timer=T, viz_3d=a.viz_3d)
        i = f.s2_info
        print(f"{Path(p).name}: people {i['n_people']} | S1 det {i['s1_det_ms']:.0f} ms + pose "
              f"{i['s1_pose_ms']:.0f} ms | S2 {i['s2_total_ms']:.0f} ms | reproj {i['reproj_px_median']:.1f}px "
              f"| bones {i['bone_err_pct_median']:.1f}% | dist {i['subject_distance_m']:.2f} m"
              + (f" | z-sign {i['z_sign']:+d} (init {i.get('init')})" if "z_sign" in i else "")
              + f" | fitted height {i['fitted_height_m']:.2f} m")
        if "physics" in i:
            ph = i["physics"]
            print("  S2+S3 fit: labels '%s' | depth flips: %s | energy margin x%.2f | passive %.1f effort %.2f "
                  "balance %.2f floor %.2f | pitch %+.1f° | least relaxed: %s"
                  % (i.get("label_hypothesis"), ", ".join(i.get("flips", {})) or "none", i.get("energy_margin", 0),
                     ph["passive"], ph["effort"], ph.get("balance", 0), ph.get("floor", 0), ph["pitch_deg"],
                     ", ".join("%s %+.1f" % kv for kv in list(ph["worst_joints"].items())[:3])))
            if i.get("occlusion"):
                print("  occlusion from contours: " + ", ".join(f"{k} {v} torso" for k, v in i["occlusion"].items()))
            if i.get("energy_margin", 9) < 1.05:
                print("  ! two readings are almost equally good: check the side view")
        if f.s3_info:
            print(summary_s3(f))
        if s4_kw is not None:
            write_croquis(f, img, a.out, stem, timer=T, **s4_kw)
        print(timing_report(f, T, 1e3 * (time.perf_counter() - t_img)))
        if prof is not None:
            import pstats
            prof.disable()
            os.makedirs(a.out, exist_ok=True)
            prof.dump_stats(os.path.join(a.out, f"{stem}.prof"))
            print("  cProfile (cumulative, top 25; open the .prof with `snakeviz` for a flame view):")
            pstats.Stats(prof).sort_stats("cumulative").print_stats(25)
    print(f"outputs in {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()

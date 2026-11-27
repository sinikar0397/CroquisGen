# CroquisGen: S1–S4 (photo → 2D keypoints → physically fitted 3D skeleton + spine → static load → gesture strokes)

```
gesture_croquis/
  skeleton.py      canonical 28-joint layout (Halpe-26 order + 2 hand points), bone-length priors, layout converters
  poseframe.py     PoseFrame dataclass = the stage contract, saved as .npz
  s1_keypoints.py  S1: YOLOX person detector + RTMPose (rtmlib). Default Halpe-26 (body + feet)
  s2_lift.py       S2: metric 3D skeleton fit (RTMW3D depth cue, or geometric 2D-only fallback)
  s2_jax.py        S2: the same residuals in jax.numpy + exact Jacobian + an LM solver (10x faster)
  plausibility.py  S2: pose plausibility energy + exact depth-flip search
  occlusion.py     S2: limb-in-front/behind-torso order from body-part contours (image edges)
  biomech.py       S2+S3: relaxed-posture prior, gravity torques, balance, floor, pitch
  muscles.py       S2+S3: unified muscle effort-stress (gravity demand + muscle length, Hill-type)
  spine.py         S2: interpolated 3D spine curve (pelvis → neck → head) from the torso box
  s3_load.py       S3 tier T0: segment masses, ground contacts, reactions, joint/spine torques, stretch proxy
  s4_gesture.py    S4: line of action, rhythm lines, torso squash/stretch, line weights, novel views, preview
  viz.py           interactive 3D HTML + static PNG (photo overlay, front/side/top)
  run.py           CLI / Pipeline class (models load once; reusable for webcam later)
  synthetic.py     ground-truth lunge pose for tests and --demo
  renderer.py      the live app's renderer: S4 in 2D mode, or a cheap geometric stand-in
  realtime.py      PerceptionResult (MediaPipe) → PoseFrame, so S4 can run without S1–S3
  test_s2_jax.py, test_s4_gesture.py   (in this folder for now; move them to tests/)
```

The S2/S3 test suite from the standalone project (`test_s2_synthetic`,
`test_s3_physics`, `test_plausibility`, `test_physical_fit`, `test_occlusion`,
`test_muscles`, `test_limb_crossing`) has not been brought over yet. Neither has
the M0 prototype `croquis.py`, whose S4 is superseded by `s4_gesture.py` but
which still holds the only S5.

## In the live app (`realtime.py`, `renderer.py`)

S1–S3 are far too slow for a webcam frame, but S4 on its own is not: `BodyView`
falls back to the 2D keypoints when `frame.joints3d` is None, and the live loop
only ever wants the photo view. `realtime.pose_frame` maps MediaPipe's 33
landmarks onto the canonical 28 joints and builds a 2D-mode `PoseFrame`;
`renderer.py` calls `build_croquis` and rasterises the strokes.

This is the default engine in the app. It costs about 56 ms a frame, so the
croquis updates roughly 13 times a second while the camera and the gesture UI
keep their own rate. `python main.py --placeholder` switches to the geometric
stand-in.

What 2D mode gives up: novel views, and the depth-ordered terms in the line
weight model (near, front/behind). Running without S3 also zeroes the effort and
support terms. MediaPipe's metric `pose_world` looks like it could feed the 3D
path and skip S2, but those coordinates are hip-centred while `BodyView.proj`
divides by z as a distance from the camera — placing the body at its true depth
is a fit, and that belongs in S2.

## Setup (once)
```bash
pip install -r requirements.txt
```

## Run on a photo
Run from `src/`, since the package resolves relative to it:
```bash
cd src
python -m gesture_croquis.run --demo                          # synthetic pose, no models needed
python -m gesture_croquis.run --image photo.jpg               # RTMPose-m Halpe-26 + RTMW3D-x depth
open out/photo_3d.html
python -m gesture_croquis.run --images photos/ --out out/     # batch a folder
```
On the first run rtmlib downloads the ONNX weights (~300 MB total) into `~/.cache/rtmlib`.

| flag | default | meaning |
|---|---|---|
| `--s1` | `halpe26` | `halpe26` (body + feet), `wholebody` (133 kpts with hands and face), `coco17` |
| `--mode` | `balanced` | `lightweight` / `balanced` / `performance` (RTMPose-s / m / x) |
| `--s2` | `rtmw3d` | `rtmw3d` (depth network) or `geometric` (no 3D net, rough) |
| `--height` | `1.70` | subject height prior in metres. Sets bone lengths and therefore the distance to the camera |
| `--fov` | `55` | horizontal FOV used when the photo has no EXIF focal length |
| `--z-sign` | `auto` | RTMW3D depth sign: `auto` fits both signs and keeps the better one. `+`/`-` fix it (half the fits), but a wrong sign mirrors the whole pose in depth: `-` reversed the torso lean and the knife arm on `e3020084…` |
| `--jac` | `auto` | S2 Jacobian: exact via JAX (`auto`/`jax`, needs `pip install jax`) or finite differences (`fd`, ~10× slower) |
| `--solver` | `auto` | S2 optimiser with the JAX Jacobian: `lm` (Levenberg–Marquardt, default) or `scipy` (trf) |
| `--viz_3d` | off | also write the 3D diagnostics `<stem>_3d.html` and `<stem>_views.png` (0.4–1.3 s per photo) |
| `--dynamic` | off | running/jumping/falling: no balance prior, low effort weight |
| `--no-search` | off | skip the plausibility search over depth flips |
| `--no-physical` | off | skip the integrated S2+S3 physical fit |
| `--no-label-swaps` | off | trust the detector's L/R labels even where limbs overlap |
| `--no-occlusion` | off | ignore body-part contours (front/behind order) |
| `--workers` | #cores (≤8) | parallel fit processes |
| `--no-s4` | off | skip the gesture strokes |
| `--views` | `0` | S4 views, `yaw[:pitch]` list in degrees: `0,90,-45:20` (0 = photo camera) |
| `--layers` | `AB` | S4 layers: A line of action, B gesture/rhythm, C construction volumes |
| `--exaggeration` | `0.35` | how far the line of action is pushed off its chord (0.2–0.5) |
| `--depth-device` | same as `--device` | device for the RTMW3D depth model only, e.g. `mps` (CoreML) while S1 stays on the CPU. RTMW3D-x takes ~0.22 s on the CPU; it falls back to the CPU if CoreML rejects it |
| `--device` | `cpu` | `mps` = CoreML EP; falls back to CPU automatically if CoreML rejects the model |

## S2 speed: exact Jacobian (`s2_jax.py`)
Profiling showed about 90% of S2 in SciPy's finite-difference Jacobian: about 90 extra residual evaluations per solver step. `s2_jax.py` reimplements the same residuals in `jax.numpy`, and `fit_skeleton` passes their exact forward-mode Jacobian to `least_squares`. The covered residuals are the image terms, anatomy, twist, collision, occlusion order, physics and muscle stress.
* **Fixed shapes.** Contacts and occlusion constraints are padded and masked, so one XLA compilation serves every fit. It is cached on disk in `~/.cache/croquisgen/jax`: about 17 s the first time on a machine, about 3 s per run after that. It is compiled at `Pipeline` start-up, not during the first image.
* **Serial fits.** With JAX the fits run in the main process, so there is no process pool to start.
* **Only rows that can be non-zero.** About 200 of the ~445 residual rows are padding or switched-off terms for a given problem. `row_mask()` drops them before the solver sees the Jacobian (`test_trimmed_rows_are_always_zero`).
* **Levenberg–Marquardt solver (`--solver lm`, default).** SciPy's trf takes an SVD of the full Jacobian at every step: about 2 ms with one BLAS thread, 6.5 ms with several on an M-series Mac, which was most of S2 once the Jacobian was exact. `s2_jax.lm_solve` minimises the same robust objective (soft_l1, scale 3, column scaling like `x_scale="jac"`) by solving a small 87×87 system per step instead. It runs in Python around the compiled residual and Jacobian, so there is no extra compile time. `test_lm_solver_matches_scipy` requires the same optimum (cost within 1%, pose within 1 cm). On a side-bend pose without the depth cue, it stopped 0.5% higher in cost, with the pose 8 mm different. `--solver scipy` keeps trf.
* **Numpy stays the reference.** `test_s2_jax.py` checks that the residuals are identical and that Jᵀr and JᵀJ match finite differences within 1e-4, on several problem variants. **If you edit a residual in s2_lift / biomech / muscles / plausibility / occlusion, mirror it in s2_jax.py**; the test fails until you do.
* **Measured** on the synthetic lunge: one physics fit 2.66 s → 0.10 s (SciPy trf) → 0.06–0.08 s (LM). The whole S2 stage went 15.6 s → 1.6 s with the exact Jacobian alone, with the same winning pose and cost. Without jax installed, the old finite-difference path and SciPy are used automatically.

## How S2 works
One least-squares fit (`fit_skeleton`) of the canonical skeleton in the camera frame (metres):

* **reprojection**: the perspective projection matches the S1 keypoints, weighted by confidence and robust (soft-L1)
* **bone lengths**: anthropometric fraction × height, with per-bone tolerance (limbs ±6–7 %, face ±20–30 %)
* **left/right symmetry**
* **hinge priors**: knees and ankles bend only one way (defined in the thigh/shank frame, so this holds when sitting too); nose in front of the neck
* **depth cue** (RTMW3D): per-joint depth relative to the pelvis, with a free scale

`--z-sign auto` fits both depth signs and keeps the one with the lower cost. The ratio is printed (`z-sign +1 (x7.3)`). If real photos always pick the same sign, hard-code it with `--z-sign +`.

Synthetic check (lunge, 2 px keypoint noise, yaws 0–110°):
* **RTMW3D-style depth:** about 20–25 mm per-joint error after Procrustes alignment, distance within 1 %.
* **Geometric fallback:** 90–190 mm. It can choose the wrong depth flip for a whole limb, because 2D-only lifting is inherently ambiguous and the hinge priors don't always resolve it. Use it only when RTMW3D is unavailable.

### Robustness to bad depth (added after the first real photo)
On a telephoto press photo (about 9 m away, about 400 px wide), RTMW3D's depths were badly off: toe and ankle 1 m apart in depth, and limbs biased backwards. The first fit turned that into a 160° folded elbow and a 40° torso twist. The fit now uses:
* **Bone-wise depth:** the network's depth is used as differences along each bone, so a per-limb bias cancels. A saturating (Cauchy) loss handles outliers, hands, feet and face count less, and the depth scale prior is ≈ metres.
* **Two starts per depth sign:** one from the raw network depth, and a "signed Taylor" start (each bone's depth *magnitude* from 2D foreshortening, only its *direction* from the network). The lowest-cost fit wins.
* **Anatomical ranges** (`pose_prior_terms`), soft and generous:
  - elbow flexion ≤ ~145°, knee flexion ≤ ~160°;
  - hip extension ≤ 30°, shoulder extension ≤ 60°;
  - trunk twist and side bend gently pulled toward natural ranges, with hard walls at 50° / 45°;
  - ears behind the eyes, nose and crown above the neck.
* **Per-person proportions:** height is free (±7% around `--height`) and legs / arms / torso each scale ±10%. Hip keypoint placement varies a lot, and without this freedom a limb that looks short gets "explained" by tilting it in depth.
* **Static balance:** the approximate centre of mass is kept within ~12 cm of the ankle-to-ankle support line. Use `--dynamic` for running, jumping or falling poses.

**What it cannot fix:** with telephoto lenses and distant subjects, perspective carries no depth, so "which limbs come toward the camera" rests entirely on the network. If the two mirror solutions score within 15% (`z-sign … (x1.0x)`), the CLI warns. Look at the side view and try `--z-sign +` / `--z-sign -`. A body-model backend (SAM 3D Body / HSMR), which reads shading and occlusion from the whole image, is the real fix.

### Integrated S2+S3 fit (`Lifter3D` + `biomech.py`)
S2 and S3 are now one procedure. The 3D pose is fitted *with* the body's mechanics, not checked afterwards:

1. **Label hypotheses.** Where the left and right knees, ankles, elbows or wrists nearly overlap, or their segments cross in 2D, the image can't say which thigh continues into which shin, and 2D detectors often swap them. Each swap is tested as a hypothesis, with a +3 energy prior against swapping: the detector is usually right.
2. **Geometric fits.** Per hypothesis and depth sign, from two starts (depth network and signed-Taylor).
3. **Depth-flip search.** Exact perspective flips, ranked by plausibility (see below).
4. **Physical fit.** The best candidates of every hypothesis are refined under one objective: image + depth cue + anatomy + **passive muscle relaxation** + **gravity effort** + **balance** + **planted feet on one level floor**. Camera pitch is fitted (prior 0 ± 8°).
5. **Winner.** Lowest energy. The CLI warns when the runner-up is within 5%.

**Passive muscle term:** each joint angle, measured in its anatomical frame (legs vs pelvis, arms vs thorax), is compared with its *relaxed* value. The cost is (deviation / stiffness)² with asymmetric stiffness: tissue resists more on one side of neutral. Values (neutral, σ below, σ above, in degrees):
| angle | relaxed | σ− / σ+ | | angle | relaxed | σ− / σ+ |
|---|---|---|---|---|---|---|
| hip flexion | 5 | 15 / 40 | | shoulder flexion | 8 | 20 / 70 |
| hip abduction | 6 | **9** / 22 (adduction across the midline is stiff) | | shoulder abduction | 10 | 25 / 60 |
| hip rotation | 10 | 25 / 25 | | shoulder rotation | −15 | 50 / 50 |
| knee flexion | 5 | **4** / 45 | | elbow flexion | 15 | 20 / 60 |
| ankle dorsiflexion | 0 | 30 / 18 | | trunk twist / bend | 0 | 25 / 14 |
| foot turn-out | 14 | 12 / 22 | | head yaw vs thorax | 0 | 40 |

**Unified muscle effort-stress (`muscles.py`):** one number per muscle group that combines what gravity asks of the muscle with how far it is stretched or shortened from its relaxed length.
- Every degree of freedom (hip flex/abd, knee, ankle, shoulder flex/abd, elbow, trunk flex, trunk side-bend, neck) has an agonist/antagonist pair with a moment arm r, an optimal fibre length L0 and a maximal moment M_max (lumped OpenSim-order values).
- **Length:** normalised fibre length l~ = 1 ∓ r·(θ − θ_relaxed)/L0. Flexing shortens the flexors and stretches the extensors.
- **Passive tension** (Thelen 2003): F_p = (e^{4(l~−1)/0.6} − 1)/(e^4 − 1) for l~ > 1, computed with a separate **range-of-motion-calibrated length** `L0_PASSIVE` (see below). Active force-length keeps L0.
- **Force-length:** f_l = exp(−(l~−1)²/0.45). A very short or very long muscle is weaker.
- **Demand:** the moment the muscles must supply, from gravity + ground reaction (the static solve), minus what passive tension already provides. Activation a = M_active / (M_max·f_l) of the group on the needed side.
- **Stress** s = √(a² + F_p²): 0 = slack and idle, 1 = maximal. The **stress index** is the RMS over all groups.

So holding a load with a short (weak) muscle costs more than the same load at mid-length, and a pose that stretches a muscle costs even with no load. The fit minimises Σs² (weight 3) together with the relaxed-posture prior (weight 0.7, which also covers axial rotations, foot turn-out and head yaw that have no muscle pair). Shoulder/hip flexion and abduction use an elevation-plane split, which stays defined for arms raised to 90° and beyond.

Typical values (synthetic, ROM-calibrated): relaxed stance 0.05 · 40° forward lean 0.13 (back extensors 0.32) · throwing lunge 0.22 (hip flexors 0.62).

**Passive stiffness calibrated to range of motion (`PASSIVE_ROM` in `muscles.py`).** With the lumped L0 also used for passive tension, the quadriceps were already 1.44× their optimal length at 60° of knee flexion: about 80 N·m of passive pull that the hamstrings had to cancel. Any bent, loaded knee or hip therefore looked strenuous, and the fit straightened joints that photos show bent. Now each group's passive tension reaches 30% of maximum (`PASSIVE_AT_ROM`) exactly at the end of the normal passive range in the direction that stretches it: L0_passive = r·|limit − rest| / strain(0.3). Normal ranges bend easily, and hyperextension stays stiff (knee −5°, elbow −10°). On `e3020084…` (deep lunge): right knee 61° → 113°, right hip flexion 17° → 102°, right knee back on its keypoint (13 → 6 px), total S2 energy 47 → 32. Ranges are adult norms; widen them for dancers or other very flexible subjects.

**Floor:** the planted contact points must lie on one plane perpendicular to gravity. Under perspective this is a real depth cue: a foot farther back projects higher. In static mode both feet count as planted when the higher one is within 15 cm; use `--dynamic` for running or jumping.

Fits run in parallel (a persistent process pool with up to 8 workers, or `--workers N`). Expect about 3–6 s per photo on an 8-core Mac and 10–20 s single-core. The flags `--no-physical`, `--no-label-swaps` and `--no-search` trade robustness for speed.

### Occlusion order from body-part contours (`occlusion.py`)
Joints can't say whether a limb passes in front of the torso or behind it. Contours can:
- **A limb in front owns its full silhouette:** both borders are visible at the expected width.
- **A limb behind is cut by the torso's contour:** on the torso side, its visible band ends at an edge as strong as its outer border. That edge lies on the torso's outline (predicted by projecting an elliptic-cylinder torso volume from the current 3D fit), so the visible width is truncated.

Rules per arm segment (8 cross-sections, gradient = strongest Lab channel):
- **cut:** visible width < 0.65 × expected, *and* the inner edge ≥ 1.5 × the 90th-percentile gradient, *and* the inner edge ≥ 0.6 × the outer border, *and* the inner edge lies within one limb radius of the predicted torso outline.
- **front:** the limb is well inside the torso region with full width and both borders ≥ the 90th-percentile gradient.
- **Otherwise no verdict.** For example, a navy sleeve on a navy jacket gives no constraint rather than a guess.

A verdict becomes a soft depth-order term in both the plausibility search and the physical fit:
- **behind:** the cut part of the segment must lie behind the torso's coronal plane along its pixel rays;
- **front:** it must lie in front of the chest surface.

**Limb-limb crossings.** Where two limbs cross in the image (upper arms, forearms, thighs, shanks, either side; directly connected segments excluded), the limb **in front** keeps both borders through the crossing, while the borders of the limb **behind** are interrupted under it.
- For each limb, its border offsets and strength are measured on cross-sections outside the overlap.
- Inside the overlap (the part of the border line within 0.65 × the other limb's half-width of its centre line, so the other limb's own borders are not counted), the median edge strength at the same offset is divided by that outside strength: the **border continuity**.
- Verdict: the limb with continuity ≥ 0.55 is in front if the other has ≤ 0.5 and the gap is ≥ 0.25. Near-parallel crossings (< 15°) and limbs whose borders are invisible anyway (< 0.7 × the 90th-percentile gradient) give no verdict. Same-colour limbs (e.g. black trousers crossing) are therefore left to the other terms.
- Residual: at the crossing, z_front + margin < z_back (margin = half the summed limb radii), soft-plus with σ = 3 cm. It is used in the flip search and the physical fit, like the torso terms.

Cost: one Sobel pass on the downscaled person crop plus ~32 profiles for the torso test, and two border scans per crossing pair, about 20–40 ms per image. Disable it with `--no-occlusion`.

Example (`trump.jpg`): the far-side upper arm is cut by the jacket's front edge, which reads as **behind**. The elbow moves from 14 cm in front of the torso mid-plane to 6 cm behind it, and shoulder flexion goes from +29° to −2° (now abducted 23° to the side).

### Plausibility search over depth flips (`plausibility.py`)
From one image every bone has two depth solutions (toward or away from the camera) with identical 2D. When the depth network is unsure, single limbs or the torso yaw come out mirrored. That produces knees bending sideways, arms twisted past their range, and limbs passing through the body. S2 therefore scores whole poses and searches over these flips.

**Plausibility energy** (lower = more plausible):
| term | what it measures |
|---|---|
| image | reprojection error + bone **proportions** vs the base fit (overall scale factored out) |
| depth | agreement with RTMW3D along bones (Cauchy-robust, weight 0.5) |
| limits | joint-angle ranges (elbow/knee flexion, hip/shoulder extension, trunk twist/bend, head) |
| twist | **axial rotation** of upper arm (vs thorax) and thigh (vs pelvis), read from the plane the elbow/knee bends in (swing-twist). Ranges: arm −110…+150° (throwing layback), hip ±50°. A sideways-bending knee = ~90° hip rotation = rejected |
| collision | capsule interpenetration: forearm/upper arm vs trunk and head, legs vs each other, arms vs the opposite thigh |
| balance | COM vs ankle support line (off with `--dynamic`) |
| effort | S3 static joint effort, Σ(τ/capacity)²: people adopt low-effort poses (final ranking only; the physical fit uses the muscle stress) |

**Search:**
1. Take the best fit for each depth sign.
2. For each of the 8 torso configurations (lean, shoulder-line yaw, hip-line yaw), coordinate-descend over the flip states of each limb segment, foot and head.
3. Flips are **exact under perspective**: each child joint stays on its camera ray at its bone length from the parent. The two roots of that quadratic are the two depth solutions, and a flip picks the other one.
4. Keep the top hypotheses both as they are and after a refit, and choose the lowest energy.

The fit itself also carries the twist and collision terms as soft penalties, and legs and arms are judged in the pelvis and thorax frames respectively.

**What it can and cannot do:** it reliably removes *impossible* poses. When two readings are both possible, it prefers the more typical, balanced, low-effort one. That's right for most photos, but it can't recover an unusual athletic pose from 2D alone. The CLI warns when the top two are within 10%. Cost: about 3–7 s per photo; skip it with `--no-search`.

## Spine (S2)
No keypoint layout has spine landmarks, so the spine is **interpolated** from the 3D torso box. It is a cubic Bézier from the pelvis to the neck that leaves the pelvis perpendicular to the hip line and meets the thorax perpendicular to the shoulder line. A smooth cervical segment then continues to the head.
* **Lateral bend:** a C-curve, concave on the compressed side.
* **Twist:** recorded as an angle; it adds no bend.
* **Front/back curvature:** not observable from keypoints. `--sagittal 0.1` adds an assumed amount. A body model (SKEL/MHR) or a silhouette will replace this.

## S3 tier T0: static load
1. **Masses:** de Leva segment fractions × `--mass` (default: BMI 22 at the given height). The trunk is split into three parts placed on the spine curve.
2. **Contacts:** detected automatically as points near the lowest foot point. If the centre of mass falls outside the support area, the next-lowest foot points (up to 15 cm above the ground) are added, because a noisy 3D fit can leave a planted foot slightly in the air. Use `--contacts` for seated or leaning poses (e.g. `lheel,rheel,pelvis`).
3. **Reactions:** vertical forces whose centre of pressure matches the centre of mass's ground position (bounded least squares). If no split can balance the body, `com_offset_m` measures the imbalance. Read it as a motion or falling cue for S4.
4. **Torques:** the moment about each joint of everything distal to it (the pelvis is the root). This covers hips, knees, ankles, shoulders, elbows, wrists, neck, and the spine at L5/L3/L1/T12/T8/T4. Torques are stored as flex/lateral/twist components in the body frame. `effort_fraction_of_capacity` keeps |τ| / capacity.
5. **Muscle stress:** `s3_info.muscle_stress` holds per-group stress, activation, passive tension and normalised length; `stress_index`, `stress_peak`, `peak_stress_muscle` and `joint_stress` summarise it. `joint_effort` (the marker size in the 3D view) is now the per-joint maximum stress, divided by max(peak, 0.5) so a relaxed pose stays pale.
6. **Stretch proxy:** joint flexion angles, shoulder elevation, spine bend/twist, and per-side torso strain (negative = compressed). This drives the straight-vs-curved stroke rule in S4.

Synthetic checks (`test_s3_physics.py`, not yet brought over):
- a symmetric stance splits its weight 50/50;
- the centre of pressure equals the centre of mass's ground position;
- leaning 35° forward raises the L5 torque more than 3×;
- the floating front foot of the lunge gets planted for balance and both legs carry load;
- a side bend curves the spine toward the stretched side.

## S4: abstraction (`s4_gesture.py`)
`build_croquis(frame, View(yaw, pitch), layers="AB", exaggeration=0.35) -> Croquis`. It runs in about 30–60 ms per view.

Nothing in S4 moves a joint. Every stroke comes from the projected S2 skeleton and spine. The only freedoms are which chain to draw, a cubic-Bézier simplification, a bounded bow, and the line weight.

Output: `<stem>_croquis[<view>].json` holds the strokes in view pixels with per-sample width, depth and layer, plus the line-of-action (LOA) candidate table. `<stem>_croquis[<view>].png` is a preview panel: the strokes over the faded photo, next to the clean drawing.

**Views.** Yaw and pitch rotate the body about the vertical and horizontal axes through the pelvis, then reuse the photo camera at the same distance. The photo view (0) is in the photo's pixel frame, so strokes overlay it directly. Without S2, S4 falls back to the 2D keypoints: photo view only, no depth cues.

**Line of action (layer A).**
1. Candidate chains:
   - head → spine → either leg;
   - either hand → arm → spine → the opposite leg or the same-side leg;
   - spine only.

   Chains use the S2 **spine curve**, not the neck–pelvis line, and leave it at L1 for the hip. Going through the pelvis centre would add a sideways jog that reads as a false S.
2. Every chain is resampled by 2D arc length and fitted with one cubic Bézier: fixed ends, chord-length start, then nearest-point and Newton re-parameterisation.
3. Score = (L / body px) · (1 + load) · (1 + 0.6·support) / (1 + 25·fit residual) / (1 + 10·backtrack).
   - load is the mean S3 effort along the chain;
   - support is the fraction of body weight on the chain's leg;
   - backtrack is the share of the path that runs back along its own chord. A hanging arm folds onto the trunk and is rejected this way; a C or S curve never doubles back.
4. The winner is **exaggerated**: its inner control points are pushed off the chord by (1 + k), and the end points stay on the skeleton. It is classified as a C or S curve from the sign changes of its turning.

**Rhythm lines (layer B).** One cubic per limb (shoulder→elbow→wrist→hand, hip→knee→ankle), plus a small wedge per foot (heel→ball).
- The limb's own bend is exaggerated by half of k.
- A **muscle-side bias** bows the line toward the working or shortened muscles. Per S3 group, q = activation + 0.5·shortening. Directions come from the limb's anatomical frame: the trunk's forward and lateral axes, carried by the minimal rotation from "hanging" to the limb's chord. Examples: flexors anterior, hip abductors lateral.
- A limb whose muscles are passively stretched is straightened (×(1 − 0.8·min(1, 2·passive))). This is the artist's "straight = stretched, curved = working" rule.
- Limbs that are part of the LOA are drawn as light ghosts.

**Torso.** The two contour lines are the torso's **silhouette sides for this view**, on an elliptic cross-section: left/right in a frontal view, front/back in profile, blended in between.
- Squash per side comes from the side lengths (lateral) and from the abdominals vs back-extensor lengths (sagittal).
- The squashed side is a heavy curve, bowed inward along the spine's concavity. The stretched side is a light, straight line.
- Shoulder and hip lines record their tilt; opposite tilts are flagged as contrapposto. A tilt is not reported when the line points into the view.
- The head is an ellipse sized from body height (stable in any view), with a tick toward the nose when the face is turned.

**Construction (layer C).** Ribcage and pelvis ellipses and joint circles. They are 2D approximations, fine as a mannequin hint.

**Line weight** per sample, as in the plan:
`w = w0[role]·S·(1 + 0.8·effort + 0.6·compression + 0.5·near + 0.5·support)·taper(t)·(1 + 0.6·front)·(1 − 0.8·behind)`
- S is the torso length in px / 120. The weight factor is Gaussian-smoothed along each stroke, so a line swells once and doesn't bulge at every joint.
- **front / behind:** where two layer-B strokes cross (excluding end-to-end touches and strokes of the same group), the one more than 3 cm nearer gets an accent and the farther one thins to a "lost line". This is the overlap/T-junction cue. The LOA is abstract and takes no part.
- All weights are in `S4Params`.

**Preview.** `render_preview` and `preview_panel` draw each open stroke as one outline polygon with round end caps, adding discs only where a sharp bend could fold the outline. Closed strokes (head, construction ellipses) are drawn as quads with round joins. Default is 2× supersampling. The line of action is drawn in red (`LOA_COLOR`), even when `color=` recolours the other strokes. Pass `loa_color=None` for one colour throughout. This is a stand-in for S5 (skia/pycairo, SVG per layer), which can consume the JSON strokes as they are.

**Speed.** All line-of-action candidates and limb curves are fitted in one vectorised pass (`fit_cubic_bezier_batch`), with the nearest-point search done as a BLAS matrix product; the results are identical to fitting one curve at a time. Measured on `e3020084…` keypoints (2D mode, as the live app uses): build 46 → 17 ms, 720p render 10 → 4 ms, a whole live frame about 55 → 24 ms.

**Not yet:** the learned stroke stylisation, velocity/follow-through cues for airborne poses (`info.airborne` and `com_offset_m` are passed through for that), and a sagittal torso squash beyond what the muscle lengths give.

Tests (`test_s4_gesture.py`, synthetic poses):
- the Bézier fit;
- a neutral stance picks head→leg, not a folded arm;
- the LOA is the heaviest stroke and its ends stay on the skeleton;
- a left side-bend squashes the left torso side;
- a single-support leg is heavier;
- yaw 360 equals the photo view, and novel views are finite;
- overlap thins the stroke behind;
- the 2D fallback, layer selection, JSON and the renderer.

## Live video: S1–S4 with ROI tracking and hysteresis (`live.py`)
`LivePipeline.process(frame_bgr) -> LiveResult(frame, croquis, kind, detected, timings)`, one instance per stream. CLI:
```bash
python -m gesture_croquis.live --camera 0 --show
python -m gesture_croquis.live --video clip.mp4 --out croquis.mp4 --overlay
```
**Models** (printed at start; `--mode` picks the S1 pair):

| | `lightweight` | `balanced` (default) | `performance` | runs |
|---|---|---|---|---|
| S1 detector (`--det-mode`, default `lightweight`) | YOLOX-tiny 416² | YOLOX-m 640² | YOLOX-x 640² | re-detection frames only (start, lost, every 90) |
| S1 pose (Halpe-26) | RTMPose-s 256×192 | RTMPose-m 256×192 | RTMPose-x 384×288 | every frame, on the ROI |
| S2 depth | RTMW3D-x 384×288 (rtmlib has no other size) | same | same | keyframes only, in the worker |

**The loop** (`_loop`) times every part: capture, S1–S4, draw, output, and on a camera the capture-to-shown delay. Drawing happens at `--preview-width` (960 px), with the overlay blended in uint8 at that size. On a camera, a reader thread keeps only the newest frame (`LatestFrame`), so the delay stays about one processing time instead of a growing queue; skipped stale frames are counted. Video files are processed frame by frame.
**S1, ROI tracking.**
- The person detector (YOLOX) runs only without a track, after the track is lost, or every `redetect_every` frames (90) as a safety net. Otherwise RTMPose runs on a box grown 25% around the previous frame's confident keypoints, and by another 50% when a joint comes near the box edge.
- **"Lost" has hysteresis:** the core-joint confidence must stay below 0.30 for two frames to count as lost, and rise above 0.45 to count as found.
- A One-Euro filter on the keypoints removes jitter without lagging fast motion.

**Fast motion and camera movement** (added after the first real-camera test lagged badly):
- **Predictive ROI:** the box moves ahead by the person's velocity and grows with the expected motion.
- **Same-frame re-detection:** if the ROI result is poor (core joints missing), S1 re-runs with the detector *on the same frame*. It no longer waits two frames.
- **Small re-detection detector:** YOLOX-tiny by default (`--det-mode`), while RTMPose stays at `--mode`. The detector runs rarely, but always on the fastest frames.
- **Never blocking with a track:** re-acquisition no longer runs a synchronous full re-solve, which in the throttled worker took a second or more. It uses the aligned tracked fit and asks the worker for a re-solve in the background. Only the very first frame blocks.
- **No-person frames:** a frame with no usable person keeps the last croquis (`kind="no-person"`). Before, S4 raised and the loop crashed.

**S2, tracking.** `Lifter3D.track()` runs one physical fit warm-started from the previous frame's skeleton, fitted height and proportions (`fit_skeleton(scales_p0=...)`) and camera pitch. No detector, no RTMW3D, no label or flip search. The warm start follows the motion first:
- `_aligned`: the previous pose is moved, and its depth scaled, so its **torso** (shoulders + hips, nearly rigid) projects onto the new keypoints. Size changes under 15% are ignored. Using knees and ankles for the size made the depth wobble with the pose (frame-to-frame jitter 2.6 cm against a true 1.9 cm).
- `_on_rays`: a joint more than 3% of the person's height off its new keypoint, or one that just reappeared, keeps its depth but moves onto its new camera ray. A reappearing joint takes its parent's depth. Before this, a joint "parked" on a neighbour for one frame stayed 90–130 px off for several frames.

Synthetic stress test (fast motion, ±250 px camera pan, every 3rd frame), against the previous version:

| | degraded frames | pose error | S2 |
|---|---|---|---|
| previous warm start | 26 of 60 | 11.3 cm | 92 ms |
| aligned + on-ray warm start | 5 of 60 | 7.3 cm | 64 ms |

On the smooth clip the new warm start adds no jitter: 1.79 cm frame-to-frame motion against a true 1.90 cm; worst jump 8 cm, previously 11 cm.
- **Contact hysteresis** (`choose_contacts(prev=...)`): a foot in contact stays in contact until it is 8 cm above the floor; new contacts still need the usual entry rule.

**S2, keyframes.**
- **When:** every `keyframe_every` frames (30), after re-detection, or when the tracked fit degrades. "Degrades" means a median 2D error above 2% of the person's height in the image, one joint more than 10% off, or the energy jumping ×2.5 from one frame to the next. The energy *level* is not a trigger, because a harder pose (sinking into a lunge) raises it legitimately. (The first version used fixed pixel thresholds of 12/48 px. On a real webcam image those fired on 331 of 428 frames and kept the worker permanently busy.)
- **How:** the full re-solve (RTMW3D depth, both depth signs, label swaps, flip search) runs in a separate worker **process**, at low priority (`nice 10`) and capped to `--worker-threads` (2) XLA / ONNX Runtime threads. In a thread it shared Python's global interpreter lock with the frame loop: p95 frame time 302 ms, against 95–100 ms in a process. RTMW3D loads only in the worker.
- **Hysteresis:** the current track joins the re-solve as a candidate, with its energy discounted by `track_bonus` (15%). When the result arrives, a winner that isn't the track is re-fitted to the *current* frame. It replaces the track only if it beats the track there by 15% as well. Two nearly equal readings therefore do not flip back and forth.
- Without `track_bonus`, the synthetic test sequence switched twice, with a worst jump of 14 cm between frames; with it, never.
- Because the worker re-imports the calling script, that script needs an `if __name__ == "__main__":` guard. `LiveConfig(async_keyframes=False)` runs keyframes in the frame loop instead, with stalls.

**Dynamic by default.** Video runs without the static-balance and level-floor priors (`Lifter3D(dynamic=True)`), because a moving person is not in static equilibrium. `--static` restores them for held poses.
- **Foot contacts:** a foot counts as planted when it is within ~4 cm of the lowest foot point (in static mode, within 15 cm). In live mode it then stays planted until it rises 8 cm.
- **Limitation:** the lowest foot point always defines the floor, so a jump is not detected. Tracking the floor height over time (the camera is fixed) is the next step.

**S3 and S4** run on every frame. The line-of-action chain has hysteresis too (`S4Params.loa_hysteresis`, 15%; `build_croquis(prefer_loa=...)`).

**Measured** in my workspace (Linux, CPU) on a synthetic 120-frame clip: standing ↔ lunge, turning ±50°, 2 px keypoint noise, with an RTMPose stand-in that loses joints outside a too-tight ROI.
- **Detector:** 2 of 120 frames.
- **Per frame:** S2 tracked ~30 ms, S4 (3D mode) ~24 ms, S3 ~5 ms. Median ~60 ms, p95 ~95 ms, plus RTMPose on the ROI (about 22 ms on an M-series CPU). The first frame waits for a full solve (~0.4 s, plus ~5 s the first time the worker compiles).
- **Smoothness:** fitted joints move as much as the truth between frames (1.19 cm median), and accuracy matches per-photo S2 (8 cm median joint error after alignment).

**Not covered yet:**
- Occlusion constraints are computed on keyframes only and not carried into the track.
- All tests use synthetic keypoints; real-camera timings need a run on the Mac (the CLI prints them).
- One person per stream.
- The app's `renderer.py` still feeds MediaPipe landmarks to S4 in 2D mode and is declared `stateless`. Switching it to `LivePipeline` needs the camera frames and one long-lived pipeline instance in the app loop.

## Known limits / next steps
* **Speed:** S1 should be real-time-capable. S2 currently takes about 250–600 ms per fit (finite-difference Jacobian); `auto` z-sign × two starts means 4 fits. For webcam use: warm-start from the previous frame, fix the sign, write an analytic Jacobian, and re-detect the bbox every N frames only.
* **Gravity** is assumed to be camera-down and the ground is the lowest foot point. Floor/horizon estimation comes later.
* **The spine is interpolated:** the SKEL/HSMR backend (M1 proper) plugs into `Lifter3D` by filling the same PoseFrame fields, and this fit then becomes its initialiser and sanity check.
* **S3 is static:** no accelerations or muscle redundancy. Tier T1 (OpenSim/MyoSuite static optimisation) replaces the torque→effort step.
* RTMW3D's depth units and sign have not yet been checked on real photos. Look at the side view and the printed `z-sign`.

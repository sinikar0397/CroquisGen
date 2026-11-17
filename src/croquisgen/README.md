# CroquisGen: S1–S3 (photo → 2D keypoints → physically fitted 3D skeleton + spine → static load)

```
croquisgen/
  skeleton.py      canonical 28-joint layout (Halpe-26 order + 2 hand points), bone-length priors, layout converters
  poseframe.py     PoseFrame dataclass = the stage contract, saved as .npz
  s1_keypoints.py  S1: YOLOX person detector + RTMPose (rtmlib). Default Halpe-26 (body + feet)
  s2_lift.py       S2: metric 3D skeleton fit (RTMW3D depth cue, or geometric 2D-only fallback)
  plausibility.py  S2: pose plausibility energy + exact depth-flip search
  occlusion.py     S2: limb-in-front/behind-torso order from body-part contours (image edges)
  biomech.py       S2+S3: relaxed-posture prior, gravity torques, balance, floor, pitch
  muscles.py       S2+S3: unified muscle effort-stress (gravity demand + muscle length, Hill-type)
  spine.py         S2: interpolated 3D spine curve (pelvis → neck → head) from the torso box
  s3_load.py       S3 tier T0: segment masses, ground contacts, reactions, joint/spine torques, stretch proxy
  viz.py           interactive 3D HTML + static PNG (photo overlay, front/side/top)
  run.py           CLI / Pipeline class (models load once; reusable for webcam later)
  synthetic.py     ground-truth lunge pose for tests and --demo
tests/test_s2_synthetic.py, tests/test_s3_physics.py, tests/test_plausibility.py, tests/test_physical_fit.py, tests/test_occlusion.py,
tests/test_muscles.py, tests/test_limb_crossing.py
croquis.py         M0 prototype (unchanged; S3–S5 still live here)
```

## Setup (once)
```bash
cd ~/PycharmProjects/CroquisGen
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q          # S2 recovers a known 3D pose (no models needed)
python -m croquisgen.run --demo    # writes out/demo_3d.html + out/demo_views.png
```

## Run on a photo
```bash
python -m croquisgen.run --image photo.jpg                 # RTMPose-m Halpe-26 + RTMW3D-x depth
open out/photo_3d.html
python -m croquisgen.run --images photos/ --out out/       # batch a folder
```
On the first run rtmlib downloads the ONNX weights (~300 MB total) into `~/.cache/rtmlib`.

| flag | default | meaning |
|---|---|---|
| `--s1` | `halpe26` | `halpe26` (body + feet), `wholebody` (133 kpts with hands and face), `coco17` |
| `--mode` | `balanced` | `lightweight` / `balanced` / `performance` (RTMPose-s / m / x) |
| `--s2` | `rtmw3d` | `rtmw3d` (depth network) or `geometric` (no 3D net, rough) |
| `--height` | `1.70` | subject height prior in metres. Sets bone lengths and therefore the distance to the camera |
| `--fov` | `55` | horizontal FOV used when the photo has no EXIF focal length |
| `--z-sign` | `auto` | RTMW3D depth convention check (see below) |
| `--dynamic` | off | running/jumping/falling: no balance prior, low effort weight |
| `--no-search` | off | skip the plausibility search over depth flips |
| `--no-physical` | off | skip the integrated S2+S3 physical fit |
| `--no-label-swaps` | off | trust the detector's L/R labels even where limbs overlap |
| `--no-occlusion` | off | ignore body-part contours (front/behind order) |
| `--workers` | #cores (≤8) | parallel fit processes |
| `--device` | `auto` | `mps` = CoreML EP; falls back to CPU automatically if CoreML rejects the model |

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
- **Passive tension** (Thelen 2003): F_p = (e^{4(l~−1)/0.6} − 1)/(e^4 − 1) for l~ > 1.
- **Force-length:** f_l = exp(−(l~−1)²/0.45). A very short or very long muscle is weaker.
- **Demand:** the moment the muscles must supply, from gravity + ground reaction (the static solve), minus what passive tension already provides. Activation a = M_active / (M_max·f_l) of the group on the needed side.
- **Stress** s = √(a² + F_p²): 0 = slack and idle, 1 = maximal. The **stress index** is the RMS over all groups.

So holding a load with a short (weak) muscle costs more than the same load at mid-length, and a pose that stretches a muscle costs even with no load. The fit minimises Σs² (weight 3) together with the relaxed-posture prior (weight 0.7, which also covers axial rotations, foot turn-out and head yaw that have no muscle pair). Shoulder/hip flexion and abduction use an elevation-plane split, which stays defined for arms raised to 90° and beyond.

Typical values (synthetic): relaxed stance 0.03 · arm held forward 0.10 (shoulder flexors 0.53) · arm held sideways 0.13 (abductors 0.70) · 40° backward lean 0.12+.

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

Synthetic checks (`tests/test_s3_physics.py`):
- a symmetric stance splits its weight 50/50;
- the centre of pressure equals the centre of mass's ground position;
- leaning 35° forward raises the L5 torque more than 3×;
- the floating front foot of the lunge gets planted for balance and both legs carry load;
- a side bend curves the spine toward the stretched side.

## Known limits / next steps
* **Speed:** S1 should be real-time-capable. S2 currently takes about 250–600 ms per fit (finite-difference Jacobian); `auto` z-sign × two starts means 4 fits. For webcam use: warm-start from the previous frame, fix the sign, write an analytic Jacobian, and re-detect the bbox every N frames only.
* **Gravity** is assumed to be camera-down and the ground is the lowest foot point. Floor/horizon estimation comes later.
* **The spine is interpolated:** the SKEL/HSMR backend (M1 proper) plugs into `Lifter3D` by filling the same PoseFrame fields, and this fit then becomes its initialiser and sanity check.
* **S3 is static:** no accelerations or muscle redundancy. Tier T1 (OpenSim/MyoSuite static optimisation) replaces the torque→effort step.
* RTMW3D's depth units and sign have not yet been checked on real photos. Look at the side view and the printed `z-sign`.

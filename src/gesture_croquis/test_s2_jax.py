"""The JAX residuals / exact Jacobian must match the numpy reference (s2_lift + biomech + muscles +
plausibility + occlusion).   python -m pytest gesture_croquis/test_s2_jax.py -q -s"""
import time
import numpy as np
import pytest

from gesture_croquis import s2_jax
from gesture_croquis.skeleton import J, IDX
from gesture_croquis.synthetic import lunge_pose, stand_pose
from gesture_croquis.s2_lift import fit_skeleton, default_K, project, init_from_depth
from gesture_croquis.biomech import Physics, choose_contacts
from scipy.optimize._numdiff import approx_derivative

pytestmark = pytest.mark.skipif(not s2_jax.available(), reason="jax not installed")


def _problem(X, physics=True, depth=True, cons=True, drop=("lhand",), seed=0):
    rng = np.random.default_rng(seed)
    K = default_K((960, 1280))
    uv = project(X, K) + rng.normal(0, 2, (J, 2))
    conf = np.full(J, 0.9)
    for d in drop:
        conf[IDX[d]] = 0.0                                   # missing joint -> parked
    z = (X[:, 2] - X[IDX["pelvis"], 2]) * 0.85 + rng.normal(0, 0.03, J) if depth else None
    X0 = init_from_depth(uv, conf, K, 1.70, z, 1) if depth else X + rng.normal(0, 0.02, X.shape)
    ph = Physics(choose_contacts(X, conf > 0.05, 1.70), 1.70) if physics else None
    oc = None
    if cons:
        oc = [dict(seg="r_uarm", a=IDX["rsho"], b=IDX["relb"], ts=np.linspace(0.3, 0.9, 6), kind="behind", weight=0.8),
              dict(seg="l_farm", a=IDX["lelb"], b=IDX["lwri"], ts=np.linspace(0.2, 0.8, 4), kind="front", weight=0.6),
              dict(seg="x", kind="pair", weight=0.7, margin=0.05, fa=IDX["lknee"], fb=IDX["lank"], tf=0.4,
                   ba=IDX["rknee"], bb=IDX["rank"], tb=0.6)]
    kw = dict(kpts2d=uv, conf=conf, K=K, height=1.70, X0=X0, z_rel=z, z_conf=conf if depth else None,
              physics=ph, order_cons=oc)
    return kw


@pytest.mark.parametrize("physics,depth,cons", [(True, True, True), (False, True, False), (False, False, True),
                                                (True, False, False)])
def test_residuals_and_jacobian_match_numpy(physics, depth, cons):
    kw = _problem(lunge_pose(), physics, depth, cons)
    pb = fit_skeleton(**kw, jac="jax", _probe=True)
    rng = np.random.default_rng(1)
    p = pb["p0"] + rng.normal(0, 0.01, len(pb["p0"]))      # off any special point
    r_np, r_jx = pb["residuals"](p), pb["fun"](p)
    # same multiset of residuals (the JAX version also carries masked zeros)
    a = np.sort(np.abs(np.r_[r_np, np.zeros(len(r_jx) - len(r_np))]))
    assert np.allclose(a, np.sort(np.abs(r_jx)), atol=1e-7, rtol=1e-6)
    J_fd = approx_derivative(pb["residuals"], p, method="3-point")
    J_jx = pb["jac"](p)
    g_fd, g_jx = J_fd.T @ r_np, J_jx.T @ r_jx                # row order-free invariants
    assert np.linalg.norm(g_jx - g_fd) <= 1e-4 * np.linalg.norm(g_fd) + 1e-6
    H_fd, H_jx = J_fd.T @ J_fd, J_jx.T @ J_jx
    assert np.linalg.norm(H_jx - H_fd) <= 1e-4 * np.linalg.norm(H_fd)


def test_fit_matches_finite_differences_and_is_faster():
    kw = _problem(lunge_pose(), physics=True, depth=True, cons=True)
    fit_skeleton(**kw, jac="jax", iters=5)                  # compile outside the timing
    t = time.perf_counter(); Xf, _, inf_f = fit_skeleton(**kw, jac="fd", iters=120); t_fd = time.perf_counter() - t
    t = time.perf_counter(); Xj, _, inf_j = fit_skeleton(**kw, jac="jax", iters=120); t_jx = time.perf_counter() - t
    print(f"\n  fit: finite differences {1e3 * t_fd:.0f} ms ({inf_f['nfev']} evals) | "
          f"JAX {1e3 * t_jx:.0f} ms ({inf_j['nfev']} evals, {inf_j['njev']} jacobians) | x{t_fd / t_jx:.1f}")
    assert inf_j["jac"] == "jax" and inf_f["jac"] == "fd"
    assert inf_j["cost"] <= inf_f["cost"] * 1.02 + 1e-6      # at least as good an optimum
    v = np.ones(J, bool); v[IDX["lhand"]] = False
    assert np.median(np.linalg.norm(Xj[v] - Xf[v], axis=1)) < 0.01   # same pose within 1 cm
    assert t_jx < t_fd


def test_standing_pose_no_nans():
    kw = _problem(stand_pose(), physics=True, depth=True, cons=False, drop=())   # straight limbs: singular spots
    pb = fit_skeleton(**kw, jac="jax", _probe=True)
    Jm = pb["jac"](pb["p0"])
    assert np.isfinite(Jm).all() and np.isfinite(pb["fun"](pb["p0"])).all()

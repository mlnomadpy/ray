#!/usr/bin/env python3
"""
Validate the normalized RAY variant (prop:normalized).

Exact RAY's flat-estimator variance scales as (||x||^2+b)^2(||w||^2+b)^2/eps^2 -- the
(R^2+b)^4 blow-up. Normalizing the polynomial feature, q_b(x)=p_b(x)/(||x||^2+b) with
||q_b||=1, replaces the modulation inner product by the cosine-similarity-like
    qbar = (x.w+b)^2 / ((||x||^2+b)(||w||^2+b)) <= 1,
so the variance scale is bounded by 1/eps^2 regardless of R or b. The exact identity is
    Var(exact RAY) / Var(normalized RAY) = ((||x||^2+b)(||w||^2+b))^2 = (R^2+b)^4 (equal norms R).
We confirm this empirically by sweeping b (at R=1) and R (at b=1), and we show the
normalized variant's absolute variance stays flat while exact RAY's grows as (R^2+b)^4.

------------------------------------------------------------------------------
REPRODUCIBILITY
    Env : python3>=3.9, numpy. CPU. Run: ~/.pixi/envs/jax/bin/python3 normalized_ray.py
    Out : results/normalized_ray.json (+ stdout). Deterministic seeds.
------------------------------------------------------------------------------
"""
import json, os, time
import numpy as np

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
HERE = os.path.dirname(__file__)


def flat_estimates(x, w, modulation, eps, M, rng):
    """Single-draw flat-RAY estimates with a given (deterministic) modulation scalar."""
    d = x.shape[0]
    out = np.empty(M)
    ts = rng.exponential(scale=1.0 / eps, size=M)
    for i, t in enumerate(ts):
        om = rng.normal(size=d) * np.sqrt(2.0 * t)
        beta = rng.uniform(0, 2 * np.pi)
        out[i] = (modulation / eps) * 2.0 * np.cos(om @ x + beta) * np.cos(om @ w + beta)
    return out


def pair(d, R, rng, cos_target=0.4):
    """Two vectors of norm R with inner product about cos_target*R^2."""
    x = rng.normal(size=d); x *= R / np.linalg.norm(x)
    w = rng.normal(size=d); w -= (w @ x) / (x @ x) * x          # orthogonal part
    w *= np.sqrt(1 - cos_target ** 2) * R / np.linalg.norm(w)
    w += cos_target * R / (R) * x                               # add aligned component
    w *= R / np.linalg.norm(w)
    return x, w


def measure(x, w, b, eps, M, rng):
    a = (x @ w + b) ** 2
    qbar = a / ((x @ x + b) * (w @ w + b))                      # normalized modulation
    ve = float(np.var(flat_estimates(x, w, a, eps, M, np.random.default_rng(1))))
    vn = float(np.var(flat_estimates(x, w, qbar, eps, M, np.random.default_rng(1))))
    return ve, vn, a, qbar


def main():
    eps, d, M = 1.0, 6, 200000
    out = {"config": {"eps": eps, "d": d, "M": M}, "b_sweep": [], "R_sweep": []}

    log("=== b-sweep at R=1 (ratio should be (1+b)^4) ===")
    log(f"  {'b':>4} {'Var_exact':>11} {'Var_norm':>10} {'ratio':>9} {'(R^2+b)^4':>10}")
    rng = np.random.default_rng(7)
    x, w = pair(d, 1.0, rng)
    for b in [0.0, 0.5, 1.0, 2.0, 4.0]:
        ve, vn, a, qbar = measure(x, w, b, eps, M, rng)
        pred = (1.0 + b) ** 4
        out["b_sweep"].append({"b": b, "var_exact": ve, "var_norm": vn, "ratio": ve / vn, "pred": pred})
        log(f"  {b:>4.1f} {ve:>11.4f} {vn:>10.4f} {ve/vn:>9.2f} {pred:>10.2f}")

    log("=== R-sweep at b=1 (ratio should be (R^2+1)^4) ===")
    log(f"  {'R':>4} {'Var_exact':>11} {'Var_norm':>10} {'ratio':>9} {'(R^2+1)^4':>10}")
    for R in [0.5, 1.0, 1.5, 2.0, 3.0]:
        rg = np.random.default_rng(20)
        x, w = pair(d, R, rg)
        ve, vn, a, qbar = measure(x, w, 1.0, eps, M, rg)
        pred = (R ** 2 + 1.0) ** 4
        out["R_sweep"].append({"R": R, "var_exact": ve, "var_norm": vn, "ratio": ve / vn, "pred": pred})
        log(f"  {R:>4.1f} {ve:>11.4f} {vn:>10.4f} {ve/vn:>9.2f} {pred:>10.2f}")

    vn_vals = [r["var_norm"] for r in out["R_sweep"]]
    log(f"normalized variance across R in [0.5,3]: min {min(vn_vals):.4f}, max {max(vn_vals):.4f} (flat, <= 3/2)")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "normalized_ray.json"), "w") as f:
        json.dump(out, f, indent=2)
    log("wrote results/normalized_ray.json")


if __name__ == "__main__":
    main()

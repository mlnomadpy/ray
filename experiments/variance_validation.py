#!/usr/bin/env python3
"""
Kernel-value variance of the biased Yat-kernel random-feature estimators.

Reproduces Table `tab:variance_reduction` (Section 6.2) of
papers/01_theory/biased_random_features/main.tex.

For a single fixed pair (x, w) in d=1, estimate k_{E,b}(x,w) over many Monte
Carlo repetitions and report the empirical variance of the cosine estimator and
its variance-reduced variants:
    - Basic cosine RAY           (t ~ Exp(eps),  i.i.d. inner RFF)
    - QMC cosine RAY             (t via inverse-CDF of a 1-D Sobol sequence)
    - Orthogonal cosine RAY      (inner frequencies orthogonalized per scale)
    - QMC + orthogonal           (both)
against the O(1/D) theoretical bound  (R^2+b)^4 / (D eps)  of Theorem 3.3.

The polynomial factor is exact and scalar here (d=1), so all variance comes
from the radial Gaussian factor -- this isolates the estimator comparison.

------------------------------------------------------------------------------
REPRODUCIBILITY
    Env    : python3 >= 3.9, numpy >= 1.24, scipy >= 1.10 (scipy.stats.qmc)
    Run    : ~/.pixi/envs/jax/bin/python3 variance_validation.py
    Output : results/variance_validation.json  (+ progress to stdout)
    Wall   : seconds.  Determinism: fixed seeds per (method, D, rep-block).
------------------------------------------------------------------------------
"""
import argparse
import json
import os
import time

import numpy as np

try:
    from scipy.stats import qmc
    HAVE_QMC = True
except Exception:                                    # pragma: no cover
    HAVE_QMC = False

LOG_T0 = time.time()


def log(msg):
    print(f"[{time.time() - LOG_T0:7.1f}s] {msg}", flush=True)


def exact_k(x, w, b, eps):
    return (x * w + b) ** 2 / ((x - w) ** 2 + eps)


def _gauss_factor_cos(x, w, ts, Dp, rng, orthogonal):
    """E-unbiased cosine estimate of (1/D) sum_j g_{t_j}(x,w), per scale array ts."""
    acc = 0.0
    for t in ts:
        if orthogonal:
            # 1-D: orthogonalize by antithetic +/- pairing of standard normals.
            half = (Dp + 1) // 2
            g = rng.normal(size=half)
            w_freq = np.concatenate([g, -g])[:Dp] * np.sqrt(2.0 * t)
        else:
            w_freq = rng.normal(size=Dp) * np.sqrt(2.0 * t)
        beta = rng.uniform(0.0, 2.0 * np.pi, size=Dp)
        zx = np.sqrt(2.0 / Dp) * np.cos(w_freq * x + beta)
        zw = np.sqrt(2.0 / Dp) * np.cos(w_freq * w + beta)
        acc += float(zx @ zw)
    return acc / len(ts)


def sample_t(D, eps, rng, sobol_engine):
    if sobol_engine is None:
        return rng.exponential(scale=1.0 / eps, size=D)
    u = sobol_engine.random(D).ravel()
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    return -np.log(u) / eps                            # inverse CDF of Exp(eps)


def estimate(method, x, w, b, eps, D, Dp, rng):
    poly = (x * w + b) ** 2 / eps                      # exact scalar factor
    sob = None
    if method in ("qmc", "qmc_orth"):
        if not HAVE_QMC:
            return None
        sob = qmc.Sobol(d=1, scramble=True, seed=int(rng.integers(1 << 30)))
    ts = sample_t(D, eps, rng, sob)
    g = _gauss_factor_cos(x, w, ts, Dp, rng, orthogonal=method in ("orth", "qmc_orth"))
    return g * poly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=float, default=0.5)
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--Dp", type=int, default=50)
    ap.add_argument("--Ds", type=int, nargs="+", default=[10, 50, 100])
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "results", "variance_validation.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    methods = ["basic", "qmc", "orth", "qmc_orth"]
    moff = {m: i for i, m in enumerate(methods)}      # deterministic per-method seed offset
    R = max(abs(args.x), abs(args.w))
    if not HAVE_QMC:
        log("WARNING: scipy.stats.qmc unavailable -> QMC rows will be null.")

    k_true = exact_k(args.x, args.w, args.b, args.eps)
    log(f"eps={args.eps}  k_true={k_true:.6f}  R={R}")
    out = {"config": vars(args), "R": float(R), "k_true": float(k_true),
           "variance": {}, "bound": {}}
    for D in args.Ds:
        out["bound"][str(D)] = float((R * R + args.b) ** 4 / (D * args.eps))
        for m in methods:
            ests = []
            for r in range(args.reps):
                rng = np.random.default_rng(91 * r + 7 * D + 1000 * moff[m])
                e = estimate(m, args.x, args.w, args.b, args.eps, D, args.Dp, rng)
                if e is not None:
                    ests.append(e)
            v = float(np.var(ests)) if ests else None
            out["variance"].setdefault(m, {})[str(D)] = v
            log(f"  D={D:4d}  {m:9s}  var={v if v is None else f'{v*1e3:.4f}e-3'}")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()

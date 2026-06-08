#!/usr/bin/env python3
"""
Gram-matrix approximation error for the BIASED Yat-kernel
    k_{E,b}(x,w) = (x^T w + b)^2 / (||x - w||^2 + eps)

Reproduces Table `tab:gram_error` (Section 6.1) of
papers/01_theory/biased_random_features/main.tex.

Compares, against the EXACT biased Gram matrix, the relative Frobenius error
    ||K_approx - K||_F / ||K||_F
of two approximations:
    - Cosine Random Yat-Features (RAY): omega ~ N(0, 2 t I), trig features.
    - Nystrom: m landmark columns, K ~ K_Nm K_mm^{-1} K_mN.

RAY keeps the degree-2 biased polynomial factor (x^T w + b)^2 EXACT (it needs no
sampling) and only Monte-Carlo's the radial Gaussian factor over the
Bernstein--Widder mixing measure  t ~ Exp(eps). Data is normalized to the unit
sphere (R=1) so the dimension-independence of RAY is isolated from the R^2 growth
that raw uniform[-1,1]^d would introduce.

------------------------------------------------------------------------------
REPRODUCIBILITY
    Env    : python3 >= 3.9, numpy >= 1.24   (CPU only, no GPU needed)
    Run    : ~/.pixi/envs/jax/bin/python3 gram_approx.py
    Output : results/gram_approx.json  (+ timestamped progress to stdout)
    Seeds  : --seeds (default 5); error bars are mean +/- std over seeds.
    Wall   : a few minutes on a laptop (N=1000, the D=1000 cells dominate).
    Determinism: fully determined by --seeds; no external data.
------------------------------------------------------------------------------
"""
import argparse
import json
import os
import time

import numpy as np

LOG_T0 = time.time()


def log(msg):
    print(f"[{time.time() - LOG_T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- kernels ----
def exact_gram(X, b, eps):
    """Exact biased Yat Gram matrix."""
    G = X @ X.T
    sq = np.sum(X * X, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * G          # ||x - w||^2
    return (G + b) ** 2 / (D2 + eps)


def poly_gram(X, b, eps):
    """Exact biased polynomial factor (x^T w + b)^2 / eps."""
    return (X @ X.T + b) ** 2 / eps


# --------------------------------------------------------------- estimators --
def cosine_ray_gram(X, b, eps, D, Dp, rng):
    """Cosine RAY: t ~ Exp(eps); inner RFF omega ~ N(0, 2 t I)."""
    N = X.shape[0]
    P = poly_gram(X, b, eps)
    K = np.zeros((N, N))
    ts = rng.exponential(scale=1.0 / eps, size=D)     # density eps e^{-eps t}
    for t in ts:
        W = rng.normal(size=(X.shape[1], Dp)) * np.sqrt(2.0 * t)
        beta = rng.uniform(0.0, 2.0 * np.pi, size=Dp)
        Psi = np.sqrt(2.0 / Dp) * np.cos(X @ W + beta)   # (N, Dp)
        K += (Psi @ Psi.T) * P
    return K / D


def nystrom_gram(X, b, eps, m, rng):
    """Standard Nystrom on the exact biased kernel with m random landmarks."""
    N = X.shape[0]
    idx = rng.choice(N, size=min(m, N), replace=False)
    Kmm = exact_gram(X[idx], b, eps)
    # K_Nm: cross kernel between all points and landmarks.
    G = X @ X[idx].T
    sq = np.sum(X * X, axis=1)
    sqm = np.sum(X[idx] * X[idx], axis=1)
    D2 = sq[:, None] + sqm[None, :] - 2.0 * G
    Knm = (G + b) ** 2 / (D2 + eps)
    Kmm_pinv = np.linalg.pinv(Kmm, rcond=1e-12)
    return Knm @ Kmm_pinv @ Knm.T


def rel_fro(Kapprox, K):
    return np.linalg.norm(Kapprox - K) / np.linalg.norm(K)


# --------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 5, 10, 20])
    ap.add_argument("--Ds", type=int, nargs="+", default=[10, 50, 100, 500, 1000])
    ap.add_argument("--Dp", type=int, default=50)
    ap.add_argument("--ms", type=int, nargs="+", default=[50, 100])
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--sphere", action="store_true", default=True,
                    help="L2-normalize each point to the unit sphere (R=1)")
    ap.add_argument("--cube", dest="sphere", action="store_false",
                    help="use raw uniform[-1,1]^d instead of sphere normalization")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "results", "gram_approx.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def gen(rng, d):
        X = rng.uniform(-1.0, 1.0, size=(args.N, d))
        if args.sphere:
            X = X / np.linalg.norm(X, axis=1, keepdims=True)
        return X

    log(f"config: {vars(args)}")
    results = {"config": vars(args), "cosine": {}, "nystrom": {}}

    for d in args.dims:
        log(f"=== d={d} ===")
        for D in args.Ds:
            cos_errs = []
            for s in range(args.seeds):
                rng = np.random.default_rng(1000 * s + d + D)
                X = gen(rng, d)
                K = exact_gram(X, args.b, args.eps)
                cos_errs.append(rel_fro(cosine_ray_gram(X, args.b, args.eps, D, args.Dp, rng), K))
            results["cosine"].setdefault(str(d), {})[str(D)] = [float(np.mean(cos_errs)), float(np.std(cos_errs))]
            log(f"  d={d} D={D:4d}  cosine={np.mean(cos_errs):.4f}+/-{np.std(cos_errs):.4f}")
        for m in args.ms:
            ny_errs = []
            for s in range(args.seeds):
                rng = np.random.default_rng(7000 * s + d + m)
                X = gen(rng, d)
                K = exact_gram(X, args.b, args.eps)
                ny_errs.append(rel_fro(nystrom_gram(X, args.b, args.eps, m, rng), K))
            results["nystrom"].setdefault(str(d), {})[str(m)] = [float(np.mean(ny_errs)), float(np.std(ny_errs))]
            log(f"  d={d} Nystrom m={m:4d}  err={np.mean(ny_errs):.4f}+/-{np.std(ny_errs):.4f}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()

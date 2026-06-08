#!/usr/bin/env python3
"""
Dimension-free sample-complexity validation (#2).

Corollary 3.4 / Table 6 claim: the number of RAY *radial* samples D needed for a
delta-approximation of the Gram matrix has no explicit dependence on the input
dimension d, whereas Nystrom (curse of dimensionality, O(m^{-2s/d})) needs ever
more landmarks m as d grows.

For each d we find D*(d) = smallest D reaching relative Frobenius error <= delta,
and m*(d) = smallest Nystrom landmark count reaching the same. We expect D*(d)
roughly flat in d, m*(d) rising sharply. Data is on the unit sphere (R=1 fixed)
so the test isolates dimension from the R^2 growth of raw data.

The RAY curve is computed by INCREMENTAL accumulation: one pass adds scale blocks
and records the error at each D checkpoint, so all D values cost a single sweep.

------------------------------------------------------------------------------
REPRODUCIBILITY
    Env  : python3>=3.9, numpy>=1.24 (CPU). Run: ~/.pixi/envs/jax/bin/python3 dimension_free.py
    Out  : results/dimension_free.json (+ stdout). Wall: a few minutes. Deterministic seeds.
------------------------------------------------------------------------------
"""
import argparse, json, os, time
import numpy as np

T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def sphere(rng, N, d):
    X = rng.uniform(-1.0, 1.0, size=(N, d))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def exact_gram(X, b, eps):
    G = X @ X.T
    sq = np.sum(X * X, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * G
    return (G + b) ** 2 / (D2 + eps)


def ray_error_curve(X, b, eps, Dmax, Dp, checkpoints, rng):
    """Incremental cosine-RAY Gram; relative Frobenius error at each checkpoint D."""
    N = X.shape[0]
    P = (X @ X.T + b) ** 2 / eps
    K = exact_gram(X, b, eps)
    nK = np.linalg.norm(K)
    Kacc = np.zeros((N, N))
    cps = set(checkpoints)
    errs = {}
    for j in range(1, Dmax + 1):
        t = rng.exponential(scale=1.0 / eps)
        W = rng.normal(size=(X.shape[1], Dp)) * np.sqrt(2.0 * t)
        beta = rng.uniform(0.0, 2.0 * np.pi, size=Dp)
        Psi = np.sqrt(2.0 / Dp) * np.cos(X @ W + beta)
        Kacc += (Psi @ Psi.T) * P
        if j in cps:
            errs[j] = float(np.linalg.norm(Kacc / j - K) / nK)
    return errs


def nystrom_error(X, b, eps, m, rng):
    N = X.shape[0]
    K = exact_gram(X, b, eps)
    idx = rng.choice(N, size=min(m, N), replace=False)
    Kmm = exact_gram(X[idx], b, eps)
    G = X @ X[idx].T
    sq = np.sum(X * X, axis=1); sqm = np.sum(X[idx] * X[idx], axis=1)
    D2 = sq[:, None] + sqm[None, :] - 2.0 * G
    Knm = (G + b) ** 2 / (D2 + eps)
    Kapp = Knm @ np.linalg.pinv(Kmm, rcond=1e-12) @ Knm.T
    return float(np.linalg.norm(Kapp - K) / np.linalg.norm(K))


def first_at_or_below(grid, vals, delta):
    for g in grid:
        if vals[g] <= delta:
            return g
    return None  # not reached on the grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=500)
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 5, 10, 20, 50, 100])
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--Dmax", type=int, default=1000)
    ap.add_argument("--checkpoints", type=int, nargs="+",
                    default=[10, 20, 50, 100, 200, 350, 500, 750, 1000])
    ap.add_argument("--ms", type=int, nargs="+", default=[20, 50, 100, 200, 350])
    ap.add_argument("--Dp", type=int, default=50)
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "results", "dimension_free.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    log(f"config: {vars(args)}")

    out = {"config": vars(args), "ryf_Dstar": {}, "nystrom_mstar": {},
           "ryf_curves": {}, "nystrom_curves": {}}
    for d in args.dims:
        # RAY error curve averaged over seeds
        agg = {c: [] for c in args.checkpoints}
        for s in range(args.seeds):
            rng = np.random.default_rng(11 * s + 3 * d)
            X = sphere(rng, args.N, d)
            errs = ray_error_curve(X, args.b, args.eps, args.Dmax, args.Dp,
                                   args.checkpoints, np.random.default_rng(900 + s + d))
            for c in args.checkpoints:
                agg[c].append(errs[c])
        ray_curve = {c: float(np.mean(agg[c])) for c in args.checkpoints}
        Dstar = first_at_or_below(args.checkpoints, ray_curve, args.delta)

        # Nystrom error curve
        nyc = {}
        for m in args.ms:
            es = [nystrom_error(sphere(np.random.default_rng(70 * s + d), args.N, d),
                                args.b, args.eps, m, np.random.default_rng(800 + s + d))
                  for s in range(args.seeds)]
            nyc[m] = float(np.mean(es))
        mstar = first_at_or_below(args.ms, nyc, args.delta)

        out["ryf_curves"][str(d)] = ray_curve
        out["nystrom_curves"][str(d)] = nyc
        out["ryf_Dstar"][str(d)] = Dstar
        out["nystrom_mstar"][str(d)] = mstar
        log(f"d={d:3d}  RAY D*({args.delta})={Dstar}  Nystrom m*={mstar}  "
            f"(RAY@1000={ray_curve[args.checkpoints[-1]]:.4f}, "
            f"Nys@{args.ms[-1]}={nyc[args.ms[-1]]:.4f})")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()

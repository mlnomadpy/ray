#!/usr/bin/env python3
"""
Scalability: wall-clock and memory of fitting ridge regression as N grows (#T-E1).

Motivation check for the O(N^2) bottleneck. We fit kernel/feature ridge regression
(random targets -- this measures COST, not accuracy) three ways and time the full
fit (build representation + solve):

  - Exact kernel ridge : form K (N x N), solve (K + lam I) a = y.   O(N^2 d + N^3) time, O(N^2) memory.
  - RAY primal ridge   : form features Z (N x M), solve (Z^T Z + lam I) w = Z^T y.
                         M = D * d_b' with the SYMMETRIC polynomial feature d_b' = d(d+1)/2 + d + 1,
                         flat D'=1 (one frequency per scale, the recommended estimator). O(N M^2 + M^3), O(N M).
  - Nystrom            : m landmarks, K_Nm (N x m), solve. O(N m^2), O(N m).

Honest caveat this surfaces: RAY's feature dimension M = D * d_b' carries the d^2
polynomial blow-up, so its linear-in-N scaling has a larger CONSTANT than vanilla
RFF; the symmetric reduction (used here) and sketching shrink d_b'. The win over
exact is in the N-scaling: exact is super-linear (N^2 memory wall), RAY/Nystrom linear.

Exact is skipped once N^2 floats exceed --mem-cap-gb (the wall the paper motivates).

------------------------------------------------------------------------------
REPRODUCIBILITY
    Env  : python3>=3.9, numpy>=1.24, scipy (cho_factor). CPU.
           Run: ~/.pixi/envs/jax/bin/python3 timing_scaling.py
    Out  : results/timing_scaling.json (+ stdout). Deterministic seeds.
    Wall : a few minutes at the default grid; raise --Ns for the full sweep.
------------------------------------------------------------------------------
"""
import argparse, json, os, time
import numpy as np
from scipy.linalg import cho_factor, cho_solve

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def sym_poly(X, b, eps):
    """Symmetric biased polynomial feature p_b(x)/sqrt(eps), dim d(d+1)/2 + d + 1."""
    N, d = X.shape
    iu, ju = np.triu_indices(d)
    scale = np.where(iu == ju, 1.0, np.sqrt(2.0))           # sqrt2 on off-diagonal
    quad = (X[:, iu] * X[:, ju]) * scale                    # (N, d(d+1)/2)
    lin = np.sqrt(2.0 * b) * X if b > 0 else np.zeros((N, 0))
    const = np.full((N, 1), b) if b > 0 else np.zeros((N, 0))   # squared-augmentation constant is b
    return np.concatenate([quad, lin, const], axis=1) / np.sqrt(eps)


def exact_gram(X, b, eps):
    G = X @ X.T
    sq = np.sum(X * X, axis=1)
    return (G + b) ** 2 / (sq[:, None] + sq[None, :] - 2.0 * G + eps)


def fit_exact(X, y, b, eps, lam):
    K = exact_gram(X, b, eps)
    K[np.diag_indices_from(K)] += lam
    cho_solve(cho_factor(K, overwrite_a=True, check_finite=False), y, check_finite=False)


def fit_ray(X, y, b, eps, lam, D, rng):
    N, d = X.shape
    P = sym_poly(X, b, eps)                                 # (N, d_b')
    blocks = []
    ts = rng.exponential(scale=1.0 / eps, size=D)
    for t in ts:
        w = rng.normal(size=d) * np.sqrt(2.0 * t)
        beta = rng.uniform(0.0, 2.0 * np.pi)
        blocks.append((np.sqrt(2.0) * np.cos(X @ w + beta))[:, None] * P)  # flat D'=1: sqrt2 cos * poly
    Z = np.concatenate(blocks, axis=1) / np.sqrt(D)         # (N, M), M = D * d_b'
    M = Z.shape[1]
    A = Z.T @ Z
    A[np.diag_indices_from(A)] += lam
    cho_solve(cho_factor(A, overwrite_a=True, check_finite=False), Z.T @ y, check_finite=False)
    return M


def fit_nystrom(X, y, b, eps, lam, m, rng):
    N = X.shape[0]
    idx = rng.choice(N, size=min(m, N), replace=False)
    Xm = X[idx]
    G = X @ Xm.T
    sq = np.sum(X * X, axis=1); sqm = np.sum(Xm * Xm, axis=1)
    Knm = (G + b) ** 2 / (sq[:, None] + sqm[None, :] - 2.0 * G + eps)   # (N, m)
    A = Knm.T @ Knm
    A[np.diag_indices_from(A)] += lam
    cho_solve(cho_factor(A, overwrite_a=True, check_finite=False), Knm.T @ y, check_finite=False)


def timed(fn, *a, **k):
    t = time.perf_counter()
    out = fn(*a, **k)
    return time.perf_counter() - t, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", type=int, nargs="+",
                    default=[1000, 2000, 4000, 8000, 16000, 32000, 64000])
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--mem-cap-gb", type=float, default=4.0,
                    help="skip exact once N^2 floats (8 bytes) exceed this")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "results", "timing_scaling.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    d_b = args.d * (args.d + 1) // 2 + args.d + 1
    M = args.D * d_b
    log(f"config: {vars(args)}  d_b'={d_b}  M=D*d_b'={M}")

    out = {"config": vars(args), "M": M, "rows": []}
    for N in args.Ns:
        rng = np.random.default_rng(7)
        X = rng.standard_normal((N, args.d)).astype(np.float64)
        X /= np.linalg.norm(X, axis=1, keepdims=True)        # unit sphere
        y = rng.standard_normal(N)
        row = {"N": N,
               "mem_exact_gb": 8 * N * N / 1e9,
               "mem_ryf_gb": 8 * N * M / 1e9,
               "mem_nys_gb": 8 * N * args.m / 1e9}
        # exact (skip if too big)
        if 8 * N * N / 1e9 <= args.mem_cap_gb:
            row["t_exact"], _ = timed(fit_exact, X, y, args.b, args.eps, args.lam)
        else:
            row["t_exact"] = None
        row["t_ryf"], _ = timed(fit_ray, X, y, args.b, args.eps, args.lam, args.D,
                                np.random.default_rng(1))
        row["t_nys"], _ = timed(fit_nystrom, X, y, args.b, args.eps, args.lam, args.m,
                                np.random.default_rng(2))
        out["rows"].append(row)
        te = "skip" if row["t_exact"] is None else f"{row['t_exact']:.3f}s"
        log(f"N={N:7d}  exact={te:>8} (mem {row['mem_exact_gb']:.2f}GB)  "
            f"RAY={row['t_ryf']:.3f}s (mem {row['mem_ryf_gb']:.2f}GB)  "
            f"Nys={row['t_nys']:.3f}s (mem {row['mem_nys_gb']:.2f}GB)")

    # log-log slopes vs N (where defined)
    def slope(key):
        pts = [(r["N"], r[key]) for r in out["rows"] if r.get(key)]
        if len(pts) < 2:
            return None
        ns, ts = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
        return float(np.polyfit(ns, ts, 1)[0])
    out["slopes"] = {k: slope(k) for k in ("t_exact", "t_ryf", "t_nys")}
    log(f">> time-vs-N slopes: exact={out['slopes']['t_exact']}, "
        f"RAY={out['slopes']['t_ryf']}, Nystrom={out['slopes']['t_nys']} "
        f"(exact ~N^2..3, RAY/Nystrom ~N^1 expected)")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()

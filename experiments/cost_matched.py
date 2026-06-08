#!/usr/bin/env python3
"""
Cost-matched comparison: matched FEATURE DIMENSION, not matched draws (R3 #2, #4).

The matched-D KRR plot flatters RAY: a single radial draw is tensored with the exact
degree-2 polynomial feature, so RAY's explicit feature dimension is D * d_b with
d_b = d(d+1)/2 + d + 1 (symmetric reduction). On digits (d=64) that is 2145 PER DRAW --
RAY cannot produce a feature below 2145 dimensions. The honest axis is the explicit
scalar dimension M, the thing that sets primal memory O(N M) and solve cost.

We build EXPLICIT primal features (symmetric d(d+1)/2 polynomial, validated to reproduce
the Gram-form kernel) and compare at matched M:
    RAY (D = M // d_b draws),  Gaussian RFF (M),  IMQ-RFF (M),
    Random Maclaurin (M),  optimal rank-M oracle.
KRR is solved in the primal: w=(Phi^T Phi + lam I)^{-1} Phi^T Y, predict Phi_te w.

This exposes the d^2 tax: where it dominates (high d) and where it is benign (low d).

------------------------------------------------------------------------------
REPRODUCIBILITY
    Env  : python3>=3.9, numpy, scipy, scikit-learn. CPU.
           Run: ~/.pixi/envs/jax/bin/python3 cost_matched.py
    Out  : results/cost_matched.json (+ stdout). Deterministic seeds.
    Reuses loaders/kernels from krr_downstream.py.
------------------------------------------------------------------------------
"""
import argparse, json, os, time
import numpy as np
import krr_downstream as K

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


# ----------------------------------------------------- explicit primal features ----
def _tri_idx(d):
    iu = np.triu_indices(d, 1)
    return iu


def poly2_features(X, b, eps):
    """Exact symmetric degree-2 biased feature p_b(x), dim d(d+1)/2 + d + 1.

    q(x) = [x_i^2 (diag), sqrt2 x_i x_j (i<j)] gives q(x).q(w)=(x.w)^2; then append
    sqrt(2b) x and b so p_b(x).p_b(w) = (x.w+b)^2. Scaled by 1/sqrt(eps) to match k_yat.
    """
    n, d = X.shape
    iu = _tri_idx(d)
    diag = X * X                                   # n x d
    off = np.sqrt(2.0) * X[:, iu[0]] * X[:, iu[1]]  # n x d(d-1)/2
    lin = np.sqrt(2.0 * b) * X
    const = np.full((n, 1), b)
    P = np.concatenate([diag, off, lin, const], axis=1)
    return P / np.sqrt(eps)


def d_b(d):
    return d * (d + 1) // 2 + d + 1


def ray_primal(X, b, eps, D, rng):
    """Explicit RAY feature: D blocks of sqrt2 cos(w.x+beta) * p_b(x), dim D*d_b."""
    P = poly2_features(X, b, eps)                   # n x d_b
    n, d = X.shape
    blocks = []
    for t in rng.exponential(scale=1.0 / eps, size=D):
        w = rng.normal(size=d) * np.sqrt(2.0 * t)
        beta = rng.uniform(0.0, 2.0 * np.pi)
        c = np.sqrt(2.0) * np.cos(X @ w + beta)[:, None]
        blocks.append(c * P)
    return np.concatenate(blocks, axis=1) / np.sqrt(D)


def gauss_primal(X, gamma, M, rng):
    d = X.shape[1]
    W = rng.normal(size=(d, M)) * np.sqrt(2.0 * gamma)
    beta = rng.uniform(0.0, 2.0 * np.pi, size=M)
    return np.sqrt(2.0 / M) * np.cos(X @ W + beta)


def imq_primal(X, eps, M, rng):
    d = X.shape[1]
    ts = rng.exponential(scale=1.0 / eps, size=M)
    W = rng.normal(size=(d, M)) * np.sqrt(2.0 * ts)[None, :]
    beta = rng.uniform(0.0, 2.0 * np.pi, size=M)
    return np.sqrt(2.0 / M) * np.cos(X @ W + beta)


def randmac_primal(A, b, eps, M, rng, nmax=24):
    """Whole-kernel Random Maclaurin primal feature for the on-sphere kappa."""
    a = K._maclaurin_coeffs(b, eps, nmax)
    Z = a.sum(); p = a / Z
    d = A.shape[1]
    degs = rng.choice(nmax + 1, size=M, p=p)
    F = np.empty((A.shape[0], M))
    for r, n in enumerate(degs):
        col = np.ones(A.shape[0])
        for _ in range(int(n)):
            col *= A @ (rng.integers(0, 2, size=d) * 2 - 1)
        F[:, r] = col
    return np.sqrt(Z / M) * F


# ---------------------------------------------------------------- primal KRR ------
def krr_primal(Ptr, Pte, Y, lam):
    M = Ptr.shape[1]
    w = np.linalg.solve(Ptr.T @ Ptr + lam * np.eye(M), Ptr.T @ Y)
    return Pte @ w


def evaluate_primal(task, Ptr, Pte, Ytr, yte, lam):
    pred = krr_primal(Ptr, Pte, Ytr, lam)
    if task == "reg":
        return float(np.sqrt(np.mean((pred.ravel() - yte) ** 2)))
    return float(np.mean(np.argmax(pred, 1) == yte))


def oracle_gram(Xtr, Xte, b, eps, M, evaluate, task, Ytr, yte, lam):
    G = K.k_yat(Xtr, Xtr, b, eps)
    ev, U = np.linalg.eigh((G + G.T) / 2.0)
    idx = np.argsort(ev)[::-1][:M]
    Uu = U[:, idx]; lm = np.maximum(ev[idx], 0.0)
    Ktr = (Uu * lm) @ Uu.T
    Kte = (K.k_yat(Xte, Xtr, b, eps) @ Uu) @ Uu.T
    return evaluate(task, Ktr, Kte, Ytr, yte, lam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=1200)
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ryf-draws", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "results", "cost_matched.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    log(f"config: {vars(args)}")
    out = {"config": vars(args), "datasets": {}}

    # sanity: explicit RAY primal Gram == Gram-form ray_cross
    Xs = K._sphere(np.random.default_rng(1).normal(size=(20, 6)))
    Pa = ray_primal(Xs, 1.0, 1.0, 5, np.random.default_rng(7))
    Gp = Pa @ Pa.T
    Gg = K.ray_cross(Xs, Xs, 1.0, 1.0, 5, np.random.default_rng(7))
    log(f"primal-vs-Gram RAY max|diff| = {np.max(np.abs(Gp - Gg)):.2e} (should be ~0)")

    for loader in (K.load_digits_ds, K.load_reg_ds):
        name, task, X, y = loader(np.random.default_rng(0))
        N, d = X.shape
        ntr, nte = min(args.n_train, N * 2 // 3), min(args.n_test, N // 3)
        sub = np.random.default_rng(0).choice(N, size=min(N, 1000), replace=False)
        eps = float(np.median(K.sqdist(X[sub], X[sub])[np.triu_indices(len(sub), 1)]))
        gamma = 1.0 / eps
        dbv = d_b(d)
        Ms = [dr * dbv for dr in args.ryf_draws]   # matched explicit dimension grid
        log(f"=== {name} ({task}, N={N}, d={d}) d_b={dbv}  M grid={Ms} ===")

        agg = {}
        for s in range(args.seeds):
            rng = np.random.default_rng(100 + s)
            perm = rng.permutation(N); tr, te = perm[:ntr], perm[ntr:ntr + nte]
            Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
            Ytr = ytr if task == "reg" else np.eye(int(y.max()) + 1)[ytr]
            for dr, M in zip(args.ryf_draws, Ms):
                agg.setdefault(f"ryf@{M}", []).append(evaluate_primal(task,
                    ray_primal(Xtr, args.b, eps, dr, np.random.default_rng(900 + s)),
                    ray_primal(Xte, args.b, eps, dr, np.random.default_rng(900 + s)), Ytr, yte, args.lam))
                agg.setdefault(f"gauss@{M}", []).append(evaluate_primal(task,
                    gauss_primal(Xtr, gamma, M, np.random.default_rng(902 + s)),
                    gauss_primal(Xte, gamma, M, np.random.default_rng(902 + s)), Ytr, yte, args.lam))
                agg.setdefault(f"imq@{M}", []).append(evaluate_primal(task,
                    imq_primal(Xtr, eps, M, np.random.default_rng(901 + s)),
                    imq_primal(Xte, eps, M, np.random.default_rng(901 + s)), Ytr, yte, args.lam))
                agg.setdefault(f"randmac@{M}", []).append(evaluate_primal(task,
                    randmac_primal(Xtr, args.b, eps, M, np.random.default_rng(904 + s)),
                    randmac_primal(Xte, args.b, eps, M, np.random.default_rng(904 + s)), Ytr, yte, args.lam))
                agg.setdefault(f"oracle@{M}", []).append(
                    oracle_gram(Xtr, Xte, args.b, eps, min(M, ntr), K.evaluate, task, Ytr, yte, args.lam))
        summary = {k: [float(np.mean(v)), float(np.std(v))] for k, v in agg.items()}
        out["datasets"][name] = {"task": task, "d": d, "d_b": dbv, "eps": eps, "Ms": Ms,
                                 "metric": "RMSE" if task == "reg" else "accuracy", "results": summary}
        unit = "RMSE" if task == "reg" else "acc"
        for k in sorted(summary, key=lambda z: (int(z.split("@")[1]), z)):
            log(f"  {name:11s} {k:16s} {unit}={summary[k][0]:.4f} +/- {summary[k][1]:.4f}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()

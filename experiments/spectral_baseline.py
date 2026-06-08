#!/usr/bin/env python3
"""
Is RAY near the BEST-POSSIBLE low-rank approximation on the sphere? (R3 #1)

On the sphere k_yat,b is a zonal dot-product kernel whose Mercer eigenbasis is the
spherical harmonics (eigenvalues = Funk-Hecke / Gegenbauer coefficients). Two questions:

  (A) Optimal-rank ORACLE. The best rank-D approximation of the exact train Gram
      (Eckart-Young: top-D eigendecomposition, Nystrom-style out-of-sample) is the
      ceiling for ANY D-dimensional feature map -- harmonic, RAY, or otherwise. If RAY
      tracks this curve it is near-optimal; if it is far below, a spectral method has room.
      We run RAY, the exact-numerator hybrid, uniform Nystrom, and the oracle at matched D.

  (B) Eigen-decay via Funk-Hecke. We compute the degree-n Mercer eigenvalue lambda_n of
      kappa on S^{d-1} and the harmonic multiplicity N(d,n). Slow decay + small multiplicity
      (low d) favors a harmonic truncation; the multiplicity explosion in high d is the
      curse that forces random features. This explains the regime split.

------------------------------------------------------------------------------
REPRODUCIBILITY
    Env  : python3>=3.9, numpy, scipy, scikit-learn. CPU.
           Run: ~/.pixi/envs/jax/bin/python3 spectral_baseline.py
    Out  : results/spectral_baseline.json (+ stdout). Deterministic seeds.
    Reuses kernels/loaders from krr_downstream.py (same sphere normalization).
------------------------------------------------------------------------------
"""
import argparse, json, os, time
import numpy as np
from scipy.special import gegenbauer, gamma

import krr_downstream as K  # k_yat, ray_cross, hybrid_dp_cross, nystrom_yat, evaluate, loaders

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


# ----------------------------------------------------- (A) optimal-rank oracle ----
def optimal_rank(Xtr, Xte, b, eps, D):
    """Best rank-D approximation of the exact yat Gram (Eckart-Young ceiling).

    Features Phi_tr = U_D Lambda_D^{1/2}; out-of-sample Phi_te = K_cross U_D Lambda_D^{-1/2},
    so the train/test Grams are U_D Lambda_D U_D^T and K_cross U_D U_D^T -- the projection
    of the exact kernel onto its top-D eigenspace. No D-dimensional method can beat this.
    """
    G = K.k_yat(Xtr, Xtr, b, eps)
    evals, evecs = np.linalg.eigh((G + G.T) / 2.0)
    idx = np.argsort(evals)[::-1][:D]
    U = evecs[:, idx]; lam = np.maximum(evals[idx], 0.0)
    Ktr = (U * lam) @ U.T
    Kte = (K.k_yat(Xte, Xtr, b, eps) @ U) @ U.T
    return Ktr, Kte


# --------------------------------------------------- (B) Funk-Hecke eigen-decay ----
def funk_hecke_spectrum(kappa, d, nmax, nquad=400):
    """Mercer eigenvalues lambda_n (degree n) of a zonal kernel kappa(t), t=x.w, on S^{d-1}.

    lambda_n = c_d * integral_{-1}^1 kappa(t) [C_n^a(t)/C_n^a(1)] (1-t^2)^{(d-3)/2} dt,
    with a=(d-2)/2 and c_d the surface-measure ratio omega_{d-2}/omega_{d-1}. Returns the
    per-harmonic eigenvalue lambda_n and the multiplicity N(d,n) of degree-n harmonics.
    """
    a = (d - 2) / 2.0
    # Gauss-Legendre-style dense quadrature on [-1,1] with the (1-t^2)^{(d-3)/2} weight folded in
    t, wq = np.polynomial.legendre.leggauss(nquad)
    weight = (1.0 - t**2) ** ((d - 3) / 2.0)
    c_d = gamma((d - 1) / 2.0) / (np.sqrt(np.pi) * gamma((d - 2) / 2.0))  # omega_{d-2}/omega_{d-1}
    kt = kappa(t)
    lam, mult = [], []
    for n in range(nmax + 1):
        if d == 2:
            cn = np.cos(n * np.arccos(np.clip(t, -1, 1))); cn1 = 1.0  # Chebyshev T_n, T_n(1)=1
        else:
            G = gegenbauer(n, a)
            cn = G(t); cn1 = G(1.0)
        lam_n = c_d * np.sum(wq * kt * (cn / cn1) * weight)
        lam.append(float(lam_n))
        if d == 2:
            mult.append(1 if n == 0 else 2)
        else:
            # N(d,n) = (2n+d-2)/n * C(n+d-3, n-1), with N(d,0)=1
            if n == 0:
                mult.append(1)
            else:
                from math import comb
                mult.append(int((2 * n + d - 2) * comb(n + d - 3, n - 1) // n))
    return lam, mult


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=2500)
    ap.add_argument("--n-test", type=int, default=1500)
    ap.add_argument("--Ds", type=int, nargs="+", default=[8, 32, 128, 512])
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "results", "spectral_baseline.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    log(f"config: {vars(args)}")
    out = {"config": vars(args), "oracle": {}, "spectrum": {}}

    # ---------- (A) oracle vs RAY/hybrid/Nystrom on the real KRR tasks ----------
    for loader in (K.load_digits_ds, K.load_reg_ds):
        name, task, X, y = loader(np.random.default_rng(0))
        N, d = X.shape
        ntr, nte = min(args.n_train, N * 2 // 3), min(args.n_test, N // 3)
        sub = np.random.default_rng(0).choice(N, size=min(N, 1000), replace=False)
        eps = float(np.median(K.sqdist(X[sub], X[sub])[np.triu_indices(len(sub), 1)]))
        log(f"=== {name} ({task}, N={N}, d={d}) eps={eps:.3f} ===")
        agg = {}
        for s in range(args.seeds):
            rng = np.random.default_rng(100 + s)
            perm = rng.permutation(N); tr, te = perm[:ntr], perm[ntr:ntr + nte]
            Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
            Ytr = ytr if task == "reg" else np.eye(int(y.max()) + 1)[ytr]
            ex = K.evaluate(task, K.k_yat(Xtr, Xtr, args.b, eps), K.k_yat(Xte, Xtr, args.b, eps), Ytr, yte, args.lam)
            agg.setdefault("exact_yat", []).append(ex)
            for D in args.Ds:
                Ktr_o, Kte_o = optimal_rank(Xtr, Xte, args.b, eps, D)
                agg.setdefault(f"oracle@{D}", []).append(K.evaluate(task, Ktr_o, Kte_o, Ytr, yte, args.lam))
                agg.setdefault(f"ryf@{D}", []).append(K.evaluate(task,
                    K.ray_cross(Xtr, Xtr, args.b, eps, D, np.random.default_rng(900 + s)),
                    K.ray_cross(Xte, Xtr, args.b, eps, D, np.random.default_rng(900 + s)), Ytr, yte, args.lam))
                agg.setdefault(f"hybrid@{D}", []).append(K.evaluate(task,
                    K.hybrid_dp_cross(Xtr, Xtr, args.b, eps, D, np.random.default_rng(905 + s)),
                    K.hybrid_dp_cross(Xte, Xtr, args.b, eps, D, np.random.default_rng(905 + s)), Ytr, yte, args.lam))
                Ktr_n, Kte_n = K.nystrom_yat(Xtr, Xte, args.b, eps, D, np.random.default_rng(903 + s))
                agg.setdefault(f"nystrom@{D}", []).append(K.evaluate(task, Ktr_n, Kte_n, Ytr, yte, args.lam))
        summary = {k: [float(np.mean(v)), float(np.std(v))] for k, v in agg.items()}
        out["oracle"][name] = {"task": task, "d": d, "eps": eps,
                               "metric": "RMSE" if task == "reg" else "accuracy", "results": summary}
        unit = "RMSE" if task == "reg" else "acc"
        for k in sorted(summary):
            log(f"  {name:11s} {k:14s} {unit}={summary[k][0]:.4f} +/- {summary[k][1]:.4f}")

    # ---------- (B) Funk-Hecke eigen-decay across dimension ----------
    for d in (2, 3, 8, 64):
        eps = 1.0; b = args.b
        c = 2.0 + eps
        kappa = lambda t: (t + b) ** 2 / (c - 2.0 * t)
        lam, mult = funk_hecke_spectrum(kappa, d, nmax=12)
        # cumulative harmonic count to reach 99% of trace (sum lambda_n * mult_n)
        contrib = np.array(lam) * np.array(mult)
        contrib = np.maximum(contrib, 0)
        frac = np.cumsum(contrib) / contrib.sum()
        deg99 = int(np.searchsorted(frac, 0.99))
        cum_harm = int(np.cumsum(mult)[min(deg99, len(mult) - 1)])
        out["spectrum"][f"d={d}"] = {"lambda_n": lam, "mult_n": mult,
                                     "deg_for_99pct_trace": deg99, "harmonics_for_99pct": cum_harm}
        log(f"  d={d:3d}: lambda_0..3={['%.3e'%x for x in lam[:4]]}  "
            f"99%-trace by degree {deg99} -> {cum_harm} harmonics")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()

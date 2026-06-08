#!/usr/bin/env python3
"""
Bounded-input preprocessing ablation (reviewer gap #8).

The estimator needs bounded-norm inputs: the unbounded numerator (x.w+b)^2 destabilizes
both the exact kernel and RAY on un-normalized data. We quantify how much the preprocessing
matters by comparing five schemes on the off-sphere coupled target, for the exact yat-kernel
and deployed (sketched) RAY:
  raw      : standardize only (no norm bound)         -- expected to destabilize
  maxnorm  : standardize then divide by max row norm  -- bounded ball, varying norms
  clip99   : standardize then clip norms to 99th pct  -- robust to outliers
  sphere   : divide each row by its own norm          -- unit sphere
  normkern : normalized-kernel variant q_b = p_b/(||x||^2+b)  (bounded variance)
Metric: downstream KRR test RMSE (lower better) and the max input norm seen.

Env: ~/.pixi/envs/jax/bin/python3 (numpy). Run: preprocessing_ablation.py
REPRODUCIBILITY: results/preprocessing_ablation.json; backs the preprocessing table (sec:exp; Limitation iii).
"""
import json, os, time
import numpy as np
import krr_downstream as K
import ts_ryf_costmatched as TS

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
HERE = os.path.dirname(__file__)


def ball(N, d, rng, lo=0.3, hi=1.5):
    U = rng.normal(size=(N, d)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    return U * rng.uniform(lo, hi, size=(N, 1))


def coupled_target(X, rng, k=6):
    d = X.shape[1]
    U = rng.normal(size=(k, d)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    V = ball(k, d, rng); a = rng.normal(size=k)
    return sum(a[j] * np.tanh(2 * X @ U[j]) * np.exp(-np.linalg.norm(X - V[j], axis=1)) for j in range(k))


def prep(Xraw, scheme):
    X = (Xraw - Xraw.mean(0)) / (Xraw.std(0) + 1e-9)        # standardize first
    nr = np.linalg.norm(X, axis=1)
    if scheme == "raw":      return X
    if scheme == "maxnorm":  return X / nr.max()
    if scheme == "clip99":
        c = np.percentile(nr, 99.0); s = np.minimum(1.0, c / (nr + 1e-12))
        return X * s[:, None] / c
    if scheme == "sphere":   return X / (nr[:, None] + 1e-12)
    raise ValueError(scheme)


def k_yat_norm(A, B, b, eps):
    """normalized-kernel variant: q_b(x)=p_b(x)/(||x||^2+b); k = (x.w+b)^2/((||x||^2+b)(||w||^2+b)(||x-w||^2+eps))."""
    na = (np.sum(A * A, 1) + b)[:, None]; nb = (np.sum(B * B, 1) + b)[None, :]
    return K.k_yat(A, B, b, eps) / (na * nb) * eps  # k_yat already has /1; rescale numerator only
    # note: K.k_yat = (A.B+b)^2/(||.||^2+eps); dividing by (na*nb) normalizes the numerator.


def krr_rmse_gram(Ktr, Kte, ytr, yte, lam):
    a = np.linalg.solve(Ktr + lam * np.eye(len(ytr)), ytr)
    return float(np.sqrt(np.mean((Kte @ a - yte) ** 2)))


def main():
    b, lam, seeds, d, N, ntr = 1.0, 1e-2, 3, 16, 1200, 800
    schemes = ["raw", "maxnorm", "clip99", "sphere"]
    out = {"config": {"b": b, "lam": lam, "seeds": seeds, "d": d, "N": N, "ntr": ntr,
                       "schemes": schemes + ["normkern"]}, "rows": {}}
    Xraw0 = np.random.default_rng(0).normal(size=(N, d)) * np.random.default_rng(1).uniform(0.3, 3.0, size=(N, 1))
    agg = {}
    for s in range(seeds):
        rng = np.random.default_rng(100 + s)
        Xraw = rng.normal(size=(N, d)) * rng.uniform(0.3, 3.0, size=(N, 1))   # heavy-tailed norms (off-sphere, unbounded)
        perm = rng.permutation(N); tr, te = perm[:ntr], perm[ntr:]
        for scheme in schemes:
            X = prep(Xraw, scheme)
            y = coupled_target(X, np.random.default_rng(777))
            Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
            eps = float(np.median(K.sqdist(Xtr[:300], Xtr[:300])[np.triu_indices(300, 1)]))
            maxn = float(np.linalg.norm(X, axis=1).max())
            # exact yat
            ry = krr_rmse_gram(K.k_yat(Xtr, Xtr, b, eps), K.k_yat(Xte, Xtr, b, eps), ytr, yte, lam)
            # deployed sketched RAY
            Ztr = TS.ts_ray_primal(Xtr, b, eps, 24, 128, 1000 + s); Zte = TS.ts_ray_primal(Xte, b, eps, 24, 128, 1000 + s)
            rr = krr_rmse_gram(Ztr @ Ztr.T, Zte @ Ztr.T, ytr, yte, lam)
            agg.setdefault(scheme, {"yat": [], "ray": [], "maxnorm": []})
            agg[scheme]["yat"].append(ry); agg[scheme]["ray"].append(rr); agg[scheme]["maxnorm"].append(maxn)
        # normalized-kernel variant on the sphere-bounded data (maxnorm prep)
        X = prep(Xraw, "maxnorm"); y = coupled_target(X, np.random.default_rng(777))
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        eps = float(np.median(K.sqdist(Xtr[:300], Xtr[:300])[np.triu_indices(300, 1)]))
        rn = krr_rmse_gram(k_yat_norm(Xtr, Xtr, b, eps), k_yat_norm(Xte, Xtr, b, eps), ytr, yte, lam)
        agg.setdefault("normkern", {"yat": [], "ray": [], "maxnorm": []})
        agg["normkern"]["yat"].append(rn); agg["normkern"]["maxnorm"].append(float(np.linalg.norm(X, axis=1).max()))
    for scheme, v in agg.items():
        out["rows"][scheme] = {
            "yat_rmse": [float(np.nanmean(v["yat"])), float(np.nanstd(v["yat"]))],
            "ray_rmse": ([float(np.nanmean(v["ray"])), float(np.nanstd(v["ray"]))] if v["ray"] else None),
            "max_norm": float(np.mean(v["maxnorm"]))}
        r = out["rows"][scheme]
        rr = f"{r['ray_rmse'][0]:.3f}" if r["ray_rmse"] else "  -- "
        log(f"  {scheme:9s} max||x||={r['max_norm']:6.2f}  yat-RMSE={r['yat_rmse'][0]:.3f}  ray-RMSE={rr}")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "preprocessing_ablation.json"), "w"), indent=2)
    log("wrote results/preprocessing_ablation.json")


if __name__ == "__main__":
    main()

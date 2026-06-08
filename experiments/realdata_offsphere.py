#!/usr/bin/env python3
"""
Real-data KRR OFF the sphere (answers "all real-data KRR was sphere-normalized").

We standardize then scale the whole dataset by its max row norm, so ||x||<=1 with VARYING
norms (a bounded ball, not the sphere) -- the genuinely non-dot-product regime. We compare,
at matched draws, the exact yat-kernel and RAY / TensorSketch-RAY against Gaussian RFF and
k-means Nystrom, on digits (d=64) and a larger set (covtype subsample if available, else
california). The point: RAY tracks the exact yat-kernel off-sphere too.

Env: numpy, scipy, sklearn. Run: ~/.pixi/envs/jax/bin/python3 realdata_offsphere.py -> results/realdata_offsphere.json
"""
import json, os, time
import numpy as np
import krr_downstream as K
import ts_ryf_costmatched as TS
from sklearn.cluster import KMeans
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
HERE = os.path.dirname(__file__)


def ball(X):
    """Standardize, then scale by max row norm -> ||x||<=1 with varying norms (off-sphere)."""
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return X / (np.linalg.norm(X, axis=1).max() + 1e-12)


def load_larger(rng):
    try:
        from sklearn.datasets import fetch_covtype
        d = fetch_covtype()
        idx = rng.choice(d.data.shape[0], size=3000, replace=False)
        return "covtype", "clf", d.data[idx].astype(float), d.target[idx] - 1
    except Exception as e:
        log(f"  (covtype unavailable: {e}; falling back to california)")
        from sklearn.datasets import fetch_california_housing
        d = fetch_california_housing()
        y = (d.target - d.target.mean()) / (d.target.std() + 1e-9)
        return "california", "reg", d.data, y


def kmeans_nys(Xtr, Xte, b, eps, m, seed):
    Z = KMeans(min(m, Xtr.shape[0]), n_init=2, random_state=seed).fit(Xtr).cluster_centers_
    P = np.linalg.pinv(K.k_yat(Z, Z, b, eps), rcond=1e-10)
    return K.k_yat(Xtr, Z, b, eps) @ P @ K.k_yat(Z, Xtr, b, eps), K.k_yat(Xte, Z, b, eps) @ P @ K.k_yat(Z, Xtr, b, eps)


def main():
    b, lam, D, seeds = 1.0, 1e-2, 128, 3
    from sklearn.datasets import load_digits
    dig = load_digits()
    datasets = [("digits", "clf", dig.data.astype(float), dig.target),
                load_larger(np.random.default_rng(0))]
    out = {"config": {"b": b, "lam": lam, "D": D}, "datasets": {}}
    for name, task, Xraw, y in datasets:
        X = ball(Xraw); N, d = X.shape
        norms = np.linalg.norm(X, axis=1)
        ntr = min(2000, N * 2 // 3); nte = min(1200, N // 3)
        sub = np.random.default_rng(0).choice(N, min(N, 1000), replace=False)
        eps = float(np.median(K.sqdist(X[sub], X[sub])[np.triu_indices(len(sub), 1)]))
        gamma = 1.0 / eps
        log(f"=== {name} ({task}, N={N}, d={d}) OFF-SPHERE ||x|| in [{norms.min():.2f},{norms.max():.2f}], eps={eps:.3f} ===")
        agg = {}
        for s in range(seeds):
            rng = np.random.default_rng(100 + s); perm = rng.permutation(N)
            tr, te = perm[:ntr], perm[ntr:ntr + nte]
            Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
            Ytr = ytr if task == "reg" else np.eye(int(y.max()) + 1)[ytr]
            agg.setdefault("exact_yat", []).append(K.evaluate(task, K.k_yat(Xtr, Xtr, b, eps), K.k_yat(Xte, Xtr, b, eps), Ytr, yte, lam))
            agg.setdefault("RAY", []).append(K.evaluate(task,
                K.ray_cross(Xtr, Xtr, b, eps, D, np.random.default_rng(900 + s)),
                K.ray_cross(Xte, Xtr, b, eps, D, np.random.default_rng(900 + s)), Ytr, yte, lam))
            agg.setdefault("TS-RAY", []).append(K.evaluate(task,
                TS.ts_ray_primal(Xtr, b, eps, 4, 128, 1000 + s) @ TS.ts_ray_primal(Xtr, b, eps, 4, 128, 1000 + s).T,
                TS.ts_ray_primal(Xte, b, eps, 4, 128, 1000 + s) @ TS.ts_ray_primal(Xtr, b, eps, 4, 128, 1000 + s).T, Ytr, yte, lam))
            agg.setdefault("GaussRFF", []).append(K.evaluate(task,
                K.gaussrff_cross(Xtr, Xtr, gamma, D, np.random.default_rng(902 + s)),
                K.gaussrff_cross(Xte, Xtr, gamma, D, np.random.default_rng(902 + s)), Ytr, yte, lam))
            Ktr, Kte = kmeans_nys(Xtr, Xte, b, eps, 128, 30 + s)
            agg.setdefault("kmeansNys", []).append(K.evaluate(task, Ktr, Kte, Ytr, yte, lam))
        summ = {k: [float(np.mean(v)), float(np.std(v))] for k, v in agg.items()}
        out["datasets"][name] = {"task": task, "d": d, "metric": "RMSE" if task == "reg" else "acc",
                                 "norm_range": [float(norms.min()), float(norms.max())], "results": summ}
        unit = "RMSE" if task == "reg" else "acc"
        for k in ["exact_yat", "RAY", "TS-RAY", "GaussRFF", "kmeansNys"]:
            log(f"  {name:10s} {k:10s} {unit}={summ[k][0]:.4f} +/- {summ[k][1]:.4f}")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "realdata_offsphere.json"), "w"), indent=2)
    log("wrote results/realdata_offsphere.json")


if __name__ == "__main__":
    main()

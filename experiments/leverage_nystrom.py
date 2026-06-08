#!/usr/bin/env python3
"""
Stronger Nystrom baseline: ridge-leverage-score landmarks (R3/R5/R10/R13 ask).

Off-sphere bounded ball (varying norms). Compare RAY (flat D'=1) to Nystrom with three
landmark choices: uniform, k-means, and ridge-leverage-score (exact RLS sampling, the
strong adaptive baseline). Reports relative Frobenius Gram error vs d.

Env: numpy, sklearn. Run: ~/.pixi/envs/jax/bin/python3 leverage_nystrom.py -> results/leverage_nystrom.json
"""
import json, os, time
import numpy as np
import krr_downstream as K
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
HERE = os.path.dirname(__file__)


def ball(N, d, rng, lo=0.25, hi=1.0):
    g = rng.normal(size=(N, d)); g /= np.linalg.norm(g, axis=1, keepdims=True)
    return g * rng.uniform(lo, hi, size=(N, 1))


def nys(X, Z, b, eps):
    Kmm = np.linalg.pinv(K.k_yat(Z, Z, b, eps), rcond=1e-10)
    Knm = K.k_yat(X, Z, b, eps)
    return Knm @ Kmm @ Knm.T


def rls_landmarks(X, b, eps, m, lam, rng):
    """Exact ridge-leverage-score sampling (N small enough to form K)."""
    Kx = K.k_yat(X, X, b, eps); N = X.shape[0]
    lev = np.diag(Kx @ np.linalg.inv(Kx + lam * np.eye(N)))
    p = np.maximum(lev, 1e-12); p = p / p.sum()
    idx = rng.choice(N, size=min(m, N), replace=False, p=p)
    return X[idx]


def relf(A, B): return float(np.linalg.norm(A - B) / np.linalg.norm(B))


def main():
    b, eps, N, m = 1.0, 1.0, 800, 100
    out = {"config": {"b": b, "eps": eps, "N": N, "m": m}, "by_d": {}}
    log(f"leverage-Nystrom off-sphere: N={N}, m={m}")
    log(f"  {'d':>3} {'RAY@1000':>9} {'uniform':>8} {'k-means':>8} {'leverage':>9}")
    for d in [8, 16, 32]:
        X = ball(N, d, np.random.default_rng(0))
        eps_d = float(np.median(K.sqdist(X, X)[np.triu_indices(N, 1)]))
        Kex = K.k_yat(X, X, b, eps_d)
        ray = np.mean([relf(K.ray_cross(X, X, b, eps_d, 1000, np.random.default_rng(s)), Kex) for s in range(3)])
        uni = np.mean([relf(nys(X, X[np.random.default_rng(10 + s).choice(N, m, replace=False)], b, eps_d), Kex) for s in range(3)])
        from sklearn.cluster import KMeans
        km = np.mean([relf(nys(X, KMeans(m, n_init=2, random_state=20 + s).fit(X).cluster_centers_, b, eps_d), Kex) for s in range(3)])
        lev = np.mean([relf(nys(X, rls_landmarks(X, b, eps_d, m, 1e-2, np.random.default_rng(30 + s)), b, eps_d), Kex) for s in range(3)])
        out["by_d"][str(d)] = {"ray_1000": float(ray), "uniform": float(uni), "kmeans": float(km), "leverage": float(lev)}
        log(f"  {d:>3} {ray:>9.3f} {uni:>8.3f} {km:>8.3f} {lev:>9.3f}")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "leverage_nystrom.json"), "w"), indent=2)
    log("wrote results/leverage_nystrom.json")


if __name__ == "__main__":
    main()

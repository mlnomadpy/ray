#!/usr/bin/env python3
"""
Cost-matched inductive-bias test among RANDOM-FEATURE maps (reviewer gap #6, honest version).

Reviewer #6 asks for a deployment result where sketched RAY at matched cost is preferable.
The honest claim is not "RAY beats everyone" (adaptive Nystrom of the exact kernel wins on
accuracy; it is data-dependent). It is: among DATA-INDEPENDENT random-feature maps at a
matched feature dimension M, sketched RAY is the best on a target that needs both alignment
and proximity, because it carries the yat coupling the others lack.

Off-sphere ball (d=32), three targets at matched M:
  coupled    : tanh(alignment) x Laplace(proximity)  -- needs both
  proximity  : sum a_k / (||x-v_k||^2 + e0)           -- needs distance only
  alignment  : sum a_k (u_k.x)^2                       -- needs alignment only
Random-feature maps at M: sketched RAY, Gaussian RFF, IMQ RFF, degree-2 polynomial sketch.
References (not data-independent random features): exact yat-kernel, k-means Nystrom-yat.

Env: ~/.pixi/envs/jax/bin/python3 (numpy, sklearn). Run: coupled_matched.py
REPRODUCIBILITY: results/coupled_matched.json; backs the matched-cost inductive-bias table (sec:exp_necessity).
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


def make_targets(X, rng, k=8, e0=0.5):
    d = X.shape[1]
    U = rng.normal(size=(k, d)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    V = ball(k, d, rng); a = rng.normal(size=k)
    coupled = sum(a[j] * np.tanh(2 * X @ U[j]) * np.exp(-np.linalg.norm(X - V[j], axis=1)) for j in range(k))
    prox = sum(a[j] / (np.sum((X - V[j]) ** 2, 1) + e0) for j in range(k))
    align = sum(a[j] * (X @ U[j]) ** 2 for j in range(k))
    return {"coupled": coupled, "proximity": prox, "alignment": align}


def _gauss(X, gamma, M, seed):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(X.shape[1], M)) * np.sqrt(2.0 * gamma)
    return np.sqrt(2.0 / M) * np.cos(X @ W + rng.uniform(0, 2 * np.pi, M))


def _imq(X, eps, M, seed):
    rng = np.random.default_rng(seed)
    t = rng.exponential(1.0 / eps, size=M)
    W = rng.normal(size=(X.shape[1], M)) * np.sqrt(2.0 * t)[None, :]
    return np.sqrt(2.0 / M) * np.cos(X @ W + rng.uniform(0, 2 * np.pi, M))


def _polysketch(X, b, eps, M, seed):
    """degree-2 polynomial TensorSketch only (alignment, no radial factor), dim M."""
    return TS._ts_poly_b(X, M - X.shape[1] - 1, b, eps, np.random.default_rng(seed))


def rmse(Ftr, Fte, ytr, yte, lam):
    a = np.linalg.solve(Ftr @ Ftr.T + lam * np.eye(len(ytr)), ytr)
    return float(np.sqrt(np.mean((Fte @ (Ftr.T @ a) - yte) ** 2)))


def gram_rmse(Ktr, Kte, ytr, yte, lam):
    a = np.linalg.solve(Ktr + lam * np.eye(len(ytr)), ytr)
    return float(np.sqrt(np.mean((Kte @ a - yte) ** 2)))


def main():
    b, lam, seeds, d, N = 1.0, 1e-2, 3, 32, 2000
    ntr, nte, M, m = 1300, 600, 4096, 128
    Xall = ball(N, d, np.random.default_rng(0))
    tg = make_targets(Xall, np.random.default_rng(777))
    sub = np.random.default_rng(0).choice(N, 800, replace=False)
    eps = float(np.median(K.sqdist(Xall[sub], Xall[sub])[np.triu_indices(800, 1)]))
    gamma = 1.0 / eps; Dts = max(1, round(M / (m + d + 1)))
    log(f"off-sphere d={d} N={N} M={M} (RAY: D={Dts},m={m}); eps={eps:.3f}")
    out = {"config": {"d": d, "N": N, "M": M, "m": m, "Dts": Dts, "eps": eps, "lam": lam}, "targets": {}}
    for tname, y in tg.items():
        agg = {}
        for s in range(seeds):
            rng = np.random.default_rng(100 + s)
            perm = rng.permutation(N); tr, te = perm[:ntr], perm[ntr:ntr + nte]
            Xtr, Xte, ytr, yte = Xall[tr], Xall[te], y[tr], y[te]
            feats = {
                "RAY (sketched)": (TS.ts_ray_primal(Xtr, b, eps, Dts, m, 1000 + s),
                                   TS.ts_ray_primal(Xte, b, eps, Dts, m, 1000 + s)),
                "Gaussian-RFF": (_gauss(Xtr, gamma, M, 902 + s), _gauss(Xte, gamma, M, 902 + s)),
                "IMQ-RFF": (_imq(Xtr, eps, M, 903 + s), _imq(Xte, eps, M, 903 + s)),
                "poly-sketch": (_polysketch(Xtr, b, eps, M, 904 + s), _polysketch(Xte, b, eps, M, 904 + s)),
            }
            for mn, (Ftr, Fte) in feats.items():
                agg.setdefault(mn, []).append(rmse(Ftr, Fte, ytr, yte, lam))
            # references
            agg.setdefault("exact yat (ref)", []).append(
                gram_rmse(K.k_yat(Xtr, Xtr, b, eps), K.k_yat(Xte, Xtr, b, eps), ytr, yte, lam))
            from sklearn.cluster import KMeans
            Z = KMeans(n_clusters=min(M, ntr), n_init=2, random_state=30 + s).fit(Xtr).cluster_centers_
            Kp = np.linalg.pinv(K.k_yat(Z, Z, b, eps), rcond=1e-10)
            agg.setdefault("kmeans-Nys (ref)", []).append(gram_rmse(
                K.k_yat(Xtr, Z, b, eps) @ Kp @ K.k_yat(Z, Xtr, b, eps),
                K.k_yat(Xte, Z, b, eps) @ Kp @ K.k_yat(Z, Xtr, b, eps), ytr, yte, lam))
        out["targets"][tname] = {mn: [float(np.mean(v)), float(np.std(v))] for mn, v in agg.items()}
        order = ["RAY (sketched)", "Gaussian-RFF", "IMQ-RFF", "poly-sketch", "exact yat (ref)", "kmeans-Nys (ref)"]
        log(f"  {tname:11s}: " + "  ".join(f"{mn.split()[0]}={out['targets'][tname][mn][0]:.3f}" for mn in order))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "coupled_matched.json"), "w"), indent=2)
    log("wrote results/coupled_matched.json")


if __name__ == "__main__":
    main()

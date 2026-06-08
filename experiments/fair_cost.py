#!/usr/bin/env python3
"""
Fair-cost comparison: accuracy + memory + wall-clock at matched representation (R7 #8).

Matched draws is not enough; reviewers want matched memory/time. On sphere-normalized
digits (d=64) we fix the explicit representation dimension M=d_b=2145 (RAY's floor, one
radial draw) and report, for each method: KRR test accuracy, representation memory (N*M
floats), and feature-build wall-clock. This isolates the cost of the exact d^2 polynomial.

Methods (all at the same representation size M):
  RAY exact mod. : 1 radial draw x d_b=2145                     (exact polynomial)
  TS-RAY (m=128) : 11 radial draws x (128+d+1)                  (sketched polynomial)
  Gaussian RFF   : 2145 cosine features (approximates k_gauss)  (reference kernel)
  k-means Nystrom: m=2145 landmarks (capped at n_train)         (representation N*m)

------------------------------------------------------------------------------
REPRODUCIBILITY
    Env : python3>=3.9, numpy, scikit-learn. CPU. Run: ~/.pixi/envs/jax/bin/python3 fair_cost.py
    Out : results/fair_cost.json (+ stdout). Deterministic seeds.
    Reuses krr_downstream + cost_matched_bias + ts_ryf_costmatched.
------------------------------------------------------------------------------
"""
import json, os, time
import numpy as np
import krr_downstream as K
import cost_matched_bias as CMB
import ts_ryf_costmatched as TS

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def main():
    b, lam, seeds = 1.0, 1e-2, 3
    name, task, X, y = K.load_digits_ds(np.random.default_rng(0))
    N, d = X.shape
    ntr, nte = min(1500, N * 2 // 3), min(1000, N // 3)
    sub = np.random.default_rng(0).choice(N, size=min(N, 1000), replace=False)
    eps = float(np.median(K.sqdist(X[sub], X[sub])[np.triu_indices(len(sub), 1)]))
    gamma = 1.0 / eps
    M = CMB.d_b(d, b)              # 2145
    m_ts, D_ts = 128, max(1, round(M / (128 + d + 1)))
    log(f"digits d={d} N={N} ntr={ntr} M={M} (TS: m={m_ts}, D={D_ts}); eps={eps:.3f}")

    def timed_feat(fn):
        t0 = time.perf_counter(); F = fn(); return F, (time.perf_counter() - t0)

    agg = {}
    for s in range(seeds):
        rng = np.random.default_rng(100 + s)
        perm = rng.permutation(N); tr, te = perm[:ntr], perm[ntr:ntr + nte]
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        Ytr = np.eye(int(y.max()) + 1)[ytr]
        builders = {
            "RYF-exact": (lambda: CMB.ray_primal(Xtr, b, eps, 1, np.random.default_rng(900 + s)),
                          lambda: CMB.ray_primal(Xte, b, eps, 1, np.random.default_rng(900 + s)), M),
            "TS-RYF": (lambda: TS.ts_ray_primal(Xtr, b, eps, D_ts, m_ts, 1000 + s),
                       lambda: TS.ts_ray_primal(Xte, b, eps, D_ts, m_ts, 1000 + s), D_ts * (m_ts + d + 1)),
            "Gaussian-RFF": (lambda: _gauss(Xtr, gamma, M, 902 + s),
                             lambda: _gauss(Xte, gamma, M, 902 + s), M),
        }
        for mn, (ftr, fte, dim) in builders.items():
            (Ftr, t_build), Fte = timed_feat(ftr), None
            Fte = fte()
            acc = K.evaluate(task, Ftr @ Ftr.T, Fte @ Ftr.T, Ytr, yte, lam)
            agg.setdefault(mn, {"acc": [], "t": [], "dim": dim}); agg[mn]["acc"].append(acc); agg[mn]["t"].append(t_build)
        # k-means Nystrom at matched memory (m=min(M,ntr) landmarks)
        mland = min(M, ntr)
        t0 = time.perf_counter()
        from sklearn.cluster import KMeans
        Z = KMeans(n_clusters=mland, n_init=2, random_state=30 + s).fit(Xtr).cluster_centers_
        Kmm_pinv = np.linalg.pinv(K.k_yat(Z, Z, b, eps), rcond=1e-10)
        Ktr = K.k_yat(Xtr, Z, b, eps) @ Kmm_pinv @ K.k_yat(Z, Xtr, b, eps)
        Kte = K.k_yat(Xte, Z, b, eps) @ Kmm_pinv @ K.k_yat(Z, Xtr, b, eps)
        t_build = time.perf_counter() - t0
        acc = K.evaluate(task, Ktr, Kte, Ytr, yte, lam)
        agg.setdefault("kmeans-Nystrom", {"acc": [], "t": [], "dim": mland}); agg["kmeans-Nystrom"]["acc"].append(acc); agg["kmeans-Nystrom"]["t"].append(t_build)
        # ridge-leverage-score Nystrom (adaptive, strongest landmark baseline)
        import leverage_nystrom as LN
        t0 = time.perf_counter()
        Zl = LN.rls_landmarks(Xtr, b, eps, mland, lam, np.random.default_rng(40 + s))
        Kmm_pinv = np.linalg.pinv(K.k_yat(Zl, Zl, b, eps), rcond=1e-10)
        Ktr = K.k_yat(Xtr, Zl, b, eps) @ Kmm_pinv @ K.k_yat(Zl, Xtr, b, eps)
        Kte = K.k_yat(Xte, Zl, b, eps) @ Kmm_pinv @ K.k_yat(Zl, Xtr, b, eps)
        t_build = time.perf_counter() - t0
        acc = K.evaluate(task, Ktr, Kte, Ytr, yte, lam)
        agg.setdefault("rls-Nystrom", {"acc": [], "t": [], "dim": Zl.shape[0]}); agg["rls-Nystrom"]["acc"].append(acc); agg["rls-Nystrom"]["t"].append(t_build)

    rows = {}
    log(f"  {'method':16s} {'dim':>6s} {'acc':>7s} {'mem(MB)':>9s} {'build(s)':>9s}")
    for mn, r in agg.items():
        mem_mb = ntr * r["dim"] * 8 / 1e6
        rows[mn] = {"dim": r["dim"], "acc": [float(np.mean(r["acc"])), float(np.std(r["acc"]))],
                    "mem_mb": mem_mb, "build_s": float(np.mean(r["t"]))}
        log(f"  {mn:16s} {r['dim']:6d} {np.mean(r['acc']):7.3f} {mem_mb:9.1f} {np.mean(r['t']):9.3f}")
    out = {"config": {"d": d, "N": N, "ntr": ntr, "M": M, "eps": eps, "m_ts": m_ts, "D_ts": D_ts},
           "rows": rows}
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "results", "fair_cost.json"), "w") as f:
        json.dump(out, f, indent=2)
    log("wrote results/fair_cost.json")


def _gauss(X, gamma, M, seed):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(X.shape[1], M)) * np.sqrt(2.0 * gamma)
    beta = rng.uniform(0.0, 2.0 * np.pi, size=M)
    return np.sqrt(2.0 / M) * np.cos(X @ W + beta)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Off-sphere fair-cost comparison (reviewer R25 #4): the cost-matched downstream table in RAY's
actual niche, not on sphere-normalized data where a dot-product route exists.

Mirrors fair_cost.py (matched representation dimension; KRR test error, representation memory,
feature-build wall-clock) but on OFF-SPHERE bounded-ball data with the coupled alignment x
proximity target from necessity_demo (the regime where the yat-kernel is the right kernel).
d=64 so d_b=2145 matches the digits setting; M=d_b is RAY's floor (one radial draw).

Methods at the same representation dimension M=2145:
  RAY exact      : 1 radial draw x d_b=2145
  TS-RAY (m=128) : ~11 radial draws x (128+d+1)
  Gaussian RFF   : 2145 cosine features (reference kernel)
  k-means Nystrom: 2145 landmarks (capped at n_train)

Env: ~/.pixi/envs/jax/bin/python3 (numpy, scikit-learn). Run: off_sphere_faircost.py
REPRODUCIBILITY: results/off_sphere_faircost.json; table tab:offsphere_faircost (sec:exp_faircost).
  off-sphere d=64, N=2500, M=2145, eps=1.75, coupled target, 3 seeds (test RMSE, lower better):
    RAY-exact       dim 2145  RMSE 1.114+-0.360  mem 25.7MB  build 0.020s
    TS-RAY (m=128)  dim 2123  RMSE 0.473+-0.010  mem 25.5MB  build 0.015s
    Gaussian-RFF    dim 2145  RMSE 0.260+-0.003  mem 25.7MB  build 0.024s
    kmeans-Nystrom  dim 1500  RMSE 0.099+-0.003  mem 18.0MB  build 1.887s
  Same cost ordering as the sphere table (tab:faircost), now off-sphere: TS-RAY >> starved exact
  RAY at matched dim; k-means Nystrom of the EXACT yat-kernel is most accurate; Gaussian RFF a
  strong different-kernel reference. The O(d^2) representation floor (Limitation v) is visible at
  d=64: the yat random features need more draws; the exact kernel (via Nystrom) is the better deploy.
"""
import json, os, time
import numpy as np
import krr_downstream as K
import cost_matched_bias as CMB
import ts_ryf_costmatched as TS
from necessity_demo import sample_ball, make_targets

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def _gauss(X, gamma, M, seed):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(X.shape[1], M)) * np.sqrt(2.0 * gamma)
    beta = rng.uniform(0.0, 2.0 * np.pi, size=M)
    return np.sqrt(2.0 / M) * np.cos(X @ W + beta)


def rmse(task, Ktr, Kte, ytr, yte, lam):
    # ridge in kernel space on the continuous target; return test RMSE
    alpha = np.linalg.solve(Ktr + lam * np.eye(len(ytr)), ytr)
    pred = Kte @ alpha
    return float(np.sqrt(np.mean((pred - yte) ** 2)))


def main():
    b, lam, seeds, d, N = 1.0, 1e-2, 3, 64, 2500
    ntr, nte = 1500, 1000
    Xall = sample_ball(N, d, np.random.default_rng(0))
    targets = make_targets(Xall, np.random.default_rng(777))
    y = targets["coupled"]
    sub = np.random.default_rng(0).choice(N, size=1000, replace=False)
    eps = float(np.median(K.sqdist(Xall[sub], Xall[sub])[np.triu_indices(len(sub), 1)]))
    gamma = 1.0 / eps
    M = CMB.d_b(d, b)
    m_ts, D_ts = 128, max(1, round(M / (128 + d + 1)))
    log(f"off-sphere d={d} N={N} M={M} (TS m={m_ts} D={D_ts}); eps={eps:.3f}; "
        f"||x|| in [{np.linalg.norm(Xall,axis=1).min():.2f},{np.linalg.norm(Xall,axis=1).max():.2f}]")

    agg = {}
    for s in range(seeds):
        rng = np.random.default_rng(100 + s)
        perm = rng.permutation(N); tr, te = perm[:ntr], perm[ntr:ntr + nte]
        Xtr, Xte, ytr, yte = Xall[tr], Xall[te], y[tr], y[te]
        builders = {
            "RAY-exact": (lambda: CMB.ray_primal(Xtr, b, eps, 1, np.random.default_rng(900 + s)),
                          lambda: CMB.ray_primal(Xte, b, eps, 1, np.random.default_rng(900 + s)), M),
            "TS-RAY": (lambda: TS.ts_ray_primal(Xtr, b, eps, D_ts, m_ts, 1000 + s),
                       lambda: TS.ts_ray_primal(Xte, b, eps, D_ts, m_ts, 1000 + s), D_ts * (m_ts + d + 1)),
            "Gaussian-RFF": (lambda: _gauss(Xtr, gamma, M, 902 + s),
                             lambda: _gauss(Xte, gamma, M, 902 + s), M),
        }
        for mn, (ftr, fte, dim) in builders.items():
            t0 = time.perf_counter(); Ftr = ftr(); t_build = time.perf_counter() - t0
            Fte = fte()
            r = rmse("reg", Ftr @ Ftr.T, Fte @ Ftr.T, ytr, yte, lam)
            agg.setdefault(mn, {"rmse": [], "t": [], "dim": dim}); agg[mn]["rmse"].append(r); agg[mn]["t"].append(t_build)
        # k-means Nystrom at matched representation
        mland = min(M, ntr)
        t0 = time.perf_counter()
        from sklearn.cluster import KMeans
        Z = KMeans(n_clusters=mland, n_init=2, random_state=30 + s).fit(Xtr).cluster_centers_
        Kmm_pinv = np.linalg.pinv(K.k_yat(Z, Z, b, eps), rcond=1e-10)
        Ktr = K.k_yat(Xtr, Z, b, eps) @ Kmm_pinv @ K.k_yat(Z, Xtr, b, eps)
        Kte = K.k_yat(Xte, Z, b, eps) @ Kmm_pinv @ K.k_yat(Z, Xtr, b, eps)
        t_build = time.perf_counter() - t0
        r = rmse("reg", Ktr, Kte, ytr, yte, lam)
        agg.setdefault("kmeans-Nystrom", {"rmse": [], "t": [], "dim": mland}); agg["kmeans-Nystrom"]["rmse"].append(r); agg["kmeans-Nystrom"]["t"].append(t_build)
        # ridge-leverage-score Nystrom at matched representation (adaptive, the strongest landmark baseline)
        import leverage_nystrom as LN
        t0 = time.perf_counter()
        Zl = LN.rls_landmarks(Xtr, b, eps, mland, lam, np.random.default_rng(40 + s))
        Kmm_pinv = np.linalg.pinv(K.k_yat(Zl, Zl, b, eps), rcond=1e-10)
        Ktr = K.k_yat(Xtr, Zl, b, eps) @ Kmm_pinv @ K.k_yat(Zl, Xtr, b, eps)
        Kte = K.k_yat(Xte, Zl, b, eps) @ Kmm_pinv @ K.k_yat(Zl, Xtr, b, eps)
        t_build = time.perf_counter() - t0
        r = rmse("reg", Ktr, Kte, ytr, yte, lam)
        agg.setdefault("rls-Nystrom", {"rmse": [], "t": [], "dim": Zl.shape[0]}); agg["rls-Nystrom"]["rmse"].append(r); agg["rls-Nystrom"]["t"].append(t_build)

    rows = {}
    log(f"  {'method':16s} {'dim':>6s} {'RMSE':>14s} {'mem(MB)':>9s} {'build(s)':>9s}")
    for mn, r in agg.items():
        mem_mb = ntr * r["dim"] * 8 / 1e6
        rows[mn] = {"dim": r["dim"], "rmse": [float(np.mean(r["rmse"])), float(np.std(r["rmse"]))],
                    "mem_mb": mem_mb, "build_s": float(np.mean(r["t"]))}
        log(f"  {mn:16s} {r['dim']:6d} {np.mean(r['rmse']):.4f}+-{np.std(r['rmse']):.4f} {mem_mb:9.1f} {np.mean(r['t']):9.3f}")
    out = {"config": {"d": d, "N": N, "ntr": ntr, "M": M, "eps": eps, "m_ts": m_ts, "D_ts": D_ts, "target": "coupled"},
           "rows": rows}
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "results", "off_sphere_faircost.json"), "w"), indent=2)
    log("wrote results/off_sphere_faircost.json")


if __name__ == "__main__":
    main()

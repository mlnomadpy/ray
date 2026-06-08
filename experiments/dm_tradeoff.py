#!/usr/bin/env python3
"""
Deployed-RAY D-m allocation at fixed total feature dimension M (reviewer gap #5).

The deployed (sketched) RAY estimator has two budgets: radial draws D and sketch size m,
coupled through the explicit feature dimension M = D(m+d+1). At fixed M the practical
question is the best split. We sweep m (so D = round(M/(m+d+1))) at several fixed M and
several dimensions d in {16,64,256}, on off-sphere ball data, reporting:
  - relative Frobenius Gram error ||ZZ^T - K||_F/||K||_F  (K the exact yat Gram), and
  - downstream KRR test error on the coupled alignment x proximity target.

Small m  -> many radial draws, sharp radial term but a large sketch term (eta||P||).
Large m  -> few radial draws, small sketch term but a noisy radial term (~D^-1/2).
The minimum over m at each M is the operating point; this makes the radial-vs-sketch
decomposition (Theorem thm:ts_opnorm) operational.

Env: ~/.pixi/envs/jax/bin/python3 (numpy, sklearn). Run: dm_tradeoff.py
REPRODUCIBILITY: results/dm_tradeoff.json; backs the D-m Pareto figure/table (sec:exp_ts).
"""
import json, os, time
import numpy as np
import krr_downstream as K
import ts_ryf_costmatched as TS   # ts_ray_primal(X,b,eps,D,m,seed)

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
HERE = os.path.dirname(__file__)


def ball(N, d, rng, lo=0.3, hi=1.5):
    U = rng.normal(size=(N, d)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    return U * rng.uniform(lo, hi, size=(N, 1))


def coupled_target(X, rng, k=6):
    """tanh-alignment x Laplace-proximity: matches no candidate kernel (necessity_demo)."""
    d = X.shape[1]
    U = rng.normal(size=(k, d)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    V = ball(k, d, rng); a = rng.normal(size=k)
    y = np.zeros(X.shape[0])
    for j in range(k):
        y += a[j] * np.tanh(2 * X @ U[j]) * np.exp(-np.linalg.norm(X - V[j], axis=1))
    return y


def krr_rmse(Ftr, Fte, ytr, yte, lam):
    A = Ftr @ Ftr.T + lam * np.eye(len(ytr))
    alpha = np.linalg.solve(A, ytr)
    return float(np.sqrt(np.mean((Fte @ (Ftr.T @ alpha) - yte) ** 2)))


def main():
    b, lam, seeds = 1.0, 1e-2, 3
    N, ntr = 600, 400
    ds = [16, 64, 256]
    ms = [16, 32, 64, 128, 256, 512]
    out = {"config": {"b": b, "lam": lam, "seeds": seeds, "N": N, "ntr": ntr, "ds": ds, "ms": ms},
           "by_d": {}}
    for d in ds:
        rng0 = np.random.default_rng(0)
        Xall = ball(N, d, rng0)
        sub = rng0.choice(N, size=min(N, 400), replace=False)
        eps = float(np.median(K.sqdist(Xall[sub], Xall[sub])[np.triu_indices(len(sub), 1)]))
        d_b = d * (d + 1) // 2 + d + 1
        # fixed-M grid: a few multiples of (a representative draw cost), independent of d_b
        Ms = [m0 for m0 in (4096, 8192, 16384)]
        log(f"=== d={d} d_b={d_b} eps={eps:.3f} Ms={Ms} ===")
        rows = []
        for M in Ms:
            for m in ms:
                D = max(1, round(M / (m + d + 1)))
                gerr, rmse = [], []
                for s in range(seeds):
                    rng = np.random.default_rng(100 + s)
                    perm = rng.permutation(N); tr, te = perm[:ntr], perm[ntr:]
                    Xtr, Xte = Xall[tr], Xall[te]
                    yt = coupled_target(Xall, np.random.default_rng(777))
                    ytr, yte = yt[tr], yt[te]
                    Ztr = TS.ts_ray_primal(Xtr, b, eps, D, m, 1000 + s)
                    Zte = TS.ts_ray_primal(Xte, b, eps, D, m, 1000 + s)
                    Kex = K.k_yat(Xtr, Xtr, b, eps)
                    gerr.append(float(np.linalg.norm(Ztr @ Ztr.T - Kex) / np.linalg.norm(Kex)))
                    rmse.append(krr_rmse(Ztr, Zte, ytr, yte, lam))
                Mact = D * (m + d + 1)
                rows.append({"M": M, "m": m, "D": D, "M_actual": Mact,
                             "gram_err": [float(np.mean(gerr)), float(np.std(gerr))],
                             "krr_rmse": [float(np.mean(rmse)), float(np.std(rmse))]})
                log(f"  M={M:6d} m={m:4d} D={D:4d} (M_act={Mact:6d}): "
                    f"gram={np.mean(gerr):.3f}  krr={np.mean(rmse):.3f}")
        out["by_d"][str(d)] = {"d_b": d_b, "eps": eps, "Ms": Ms, "rows": rows}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "dm_tradeoff.json"), "w"), indent=2)
    log("wrote results/dm_tradeoff.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
FAIR random-feature comparison on CIFAR CLIP embeddings (the arena the paper actually competes in:
RF vs RF, not exact kernels). Every method gets the same careful fit so yat-RAY is not penalized
for its differently-conditioned product features:
  - per-feature standardization of Z (train stats) -> equal conditioning,
  - per-method L2 sweep picked on a held-out val split -> equal regularization (yat needs more),
  - fully-converged closed-form ridge -> no "did it train enough" confound.

Methods (random features): linear, gauss (Gaussian RFF), imq (IMQ-RFF, radial-only), poly
(polynomial TensorSketch, alignment-only), tsray (yat-RAY, the product). Matched nominal M;
actual dim reported. Reports test top-1 at each method's best val-L2.

Env: /opt/homebrew/bin/python3 (mlx, numpy). Embeddings: ~/rf_data/{cifar10,cifar100}_clip.npz.
Run: cifar_rf_fair.py --datasets cifar10 cifar100 --Ms 512 1024 2048 4096
"""
import argparse, os, time, json
import numpy as np
import mlx.core as mx
from higgs_scaling import make_params, build, poly_sketch

T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)
HERE = os.path.dirname(__file__)
SQ2 = float(np.sqrt(2.0))
LAMS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def offsphere(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    scale = np.percentile(np.linalg.norm(Xtr, axis=1), 99.9) + 1e-9
    return (Xtr / scale).astype(np.float32), (Xte / scale).astype(np.float32)


def make_single_params(method, d, M, eps, b, m_sketch, seed):
    rng = np.random.default_rng(seed)
    p = {"method": method, "d": d, "eps": eps, "b": b}
    if method == "imq":                                       # radial-only: D=M IMQ-spectral draws
        D = M; t = rng.exponential(1.0 / eps, size=D)
        p["Om"] = mx.array((rng.normal(size=(d, D)) * np.sqrt(2 * t)[None, :]).astype(np.float32))
        p["beta"] = mx.array(rng.uniform(0, 2 * np.pi, D).astype(np.float32)); p["D"] = D
    else:                                                     # poly-only sketch sized so total dim ~ M
        m = max(8, M - d - 1)
        for k in (1, 2):
            h = rng.integers(0, m, d); S = np.zeros((d, m), np.float32); S[np.arange(d), h] = 1.0
            p[f"S{k}"] = mx.array(S); p[f"s{k}"] = mx.array((rng.integers(0, 2, d) * 2 - 1).astype(np.float32))
        p["m_sketch"] = m
    return p


def features(Xnp, p, bs=8192):
    out = []
    for i in range(0, len(Xnp), bs):
        Xb = mx.array(Xnp[i:i + bs])
        if p["method"] == "imq":
            Z = SQ2 * mx.cos(Xb @ p["Om"] + p["beta"]) * float(1.0 / np.sqrt(p["D"]))
        elif p["method"] == "poly":
            Z = poly_sketch(Xb, p)
        else:
            Z = build(Xnp[i:i + bs], p)
        out.append(np.array(Z))
    return np.concatenate(out)


def znorm(Ztr, *Zs):
    mu = Ztr.mean(0); sd = Ztr.std(0) + 1e-6
    return [(Z - mu) / sd for Z in (Ztr, *Zs)]


def ridge_sweep(Ztr, ytr, Zva, yva, Zte, yte, C):
    Y = np.zeros((len(ytr), C), np.float32); Y[np.arange(len(ytr)), ytr] = 1.0
    M = Ztr.shape[1]; G = Ztr.T @ Ztr; B = Ztr.T @ Y; I = np.eye(M, dtype=np.float32)
    best = (-1, None, None)
    for lam in LAMS:
        W = np.linalg.solve(G + lam * I, B)
        va = float((np.argmax(Zva @ W, 1) == yva).mean())
        if va > best[0]:
            best = (va, lam, float((np.argmax(Zte @ W, 1) == yte).mean()))
    return best[2], best[1]   # test acc at best-val lam, the lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cifar10", "cifar100"])
    ap.add_argument("--Ms", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--m-sketch", type=int, default=128)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "cifar_rf_fair.json"))
    args = ap.parse_args()
    out = {"config": vars(args), "datasets": {}}
    for name in args.datasets:
        z = np.load(os.path.expanduser(f"~/rf_data/{name}_clip.npz"))
        X, y = z["X"], z["y"].astype(int)
        rng = np.random.default_rng(0); perm = rng.permutation(len(y))
        va = perm[:5000]; tr = perm[5000:]
        Xtr, Xva = X[tr], X[va]; ytr, yva = y[tr], y[va]
        Xte, yte = z["X_test"], z["y_test"].astype(int)
        Xtr, Xte = offsphere(Xtr, Xte); _, Xva = offsphere(X[tr], Xva)
        d = Xtr.shape[1]; C = int(y.max()) + 1; b = 1.0
        s = Xtr[:2000]
        eps = float(np.median(np.sum((s[:, None] - s[None]) ** 2, -1)[np.triu_indices(len(s), 1)]))
        gamma = 1.0 / eps
        log(f"==== {name}: Ntr={len(ytr)} d={d} C={C} eps={eps:.3f} ====")
        rows = []
        Zt, Zv, Ze = znorm(Xtr.copy(), Xva, Xte)
        acc, lam = ridge_sweep(Zt, ytr, Zv, yva, Ze, yte, C)
        rows.append({"method": "linear", "M": d, "acc": acc, "lam": lam}); log(f"  linear M={d:5d}  acc={acc:.4f} (lam={lam})")
        for M in args.Ms:
            for method in ["gauss", "imq", "poly", "tsray"]:
                p = (make_params(method, d, M, eps, gamma, b, args.m_sketch, 0)
                     if method in ("gauss", "tsray") else make_single_params(method, d, M, eps, b, args.m_sketch, 0))
                Zt, Zv, Ze = znorm(features(Xtr, p), features(Xva, p), features(Xte, p))
                acc, lam = ridge_sweep(Zt, ytr, Zv, yva, Ze, yte, C)
                rows.append({"method": method, "M": M, "dim": Zt.shape[1], "D": p.get("D"), "acc": acc, "lam": lam})
                log(f"  {method:6s} M={M:5d} dim={Zt.shape[1]:6d}  acc={acc:.4f} (lam={lam})")
        out["datasets"][name] = {"d": d, "C": C, "eps": eps, "rows": rows}
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()

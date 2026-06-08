#!/usr/bin/env python3
"""
Real-data robustness test on CIFAR CLIP embeddings: does the no-regret yat-kernel beat the
fixed single-factor baselines it should, in actual top-1 accuracy (not just the pair-AUC the
gate diagnostic measured)?

The gate diagnostic found CLIP embeddings are ALIGNMENT-dominated after standardization
(pair-AUC alignment 0.90+, radial 0.86). So the prediction is:
    Gaussian RFF (a radial kernel) should LOSE to yat-RAY / the polynomial factor,
    because the discriminative geometry here is angular, which a radial kernel cannot see.
yat-RAY carries BOTH factors, so it should track the winning (alignment) factor and beat Gaussian.

Method: build random features once per (method, M), fit a closed-form ridge classifier on one-hot
labels (W = (Z^T Z + lam I)^-1 Z^T Y), report test top-1. Reuses the MLX feature builders from
higgs_scaling.py; adds radial-only (IMQ) and polynomial-only (alignment) single-factor baselines
so the no-regret claim is visible: each baseline fails on the regime it is blind to, yat does not.

Env: /opt/homebrew/bin/python3 (mlx, numpy). Embeddings: ~/rf_data/{cifar10,cifar100}_clip.npz
(make them with cifar_embed.py). Run: cifar_classify.py --datasets cifar10 cifar100 --Ms 512 1024 2048 4096
"""
import argparse, os, time, json
import numpy as np
import mlx.core as mx
from higgs_scaling import make_params, build, poly_sketch   # reuse exact builders

T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)
HERE = os.path.dirname(__file__)
SQ2 = float(np.sqrt(2.0))


def offsphere(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    scale = np.percentile(np.linalg.norm(Xtr, axis=1), 99.9) + 1e-9
    return (Xtr / scale).astype(np.float32), (Xte / scale).astype(np.float32)


def build_single(Xb, p):
    """Single-factor baselines that share RAY's radial law / polynomial sketch."""
    if p["method"] == "imq":                                   # radial-only: the IMQ/proximity factor
        return SQ2 * mx.cos(Xb @ p["Om"] + p["beta"]) * float(1.0 / np.sqrt(p["D"]))
    return poly_sketch(Xb, p)                                  # poly-only: the alignment factor (sketched)


def features(Xnp, p, bs=8192):
    out = []
    for i in range(0, len(Xnp), bs):
        Xb = mx.array(Xnp[i:i + bs])
        if p["method"] in ("imq", "poly"):
            Z = build_single(Xb, p)
        else:
            Z = build(Xnp[i:i + bs], p)
        out.append(np.array(Z))
    return np.concatenate(out)


def make_single_params(method, d, M, eps, b, m_sketch, seed):
    """imq: D=M radial draws. poly: one TensorSketch of dim m=M (capped)."""
    rng = np.random.default_rng(seed)
    p = {"method": method, "d": d, "eps": eps, "b": b}
    if method == "imq":
        D = M; t = rng.exponential(1.0 / eps, size=D)
        p["Om"] = mx.array((rng.normal(size=(d, D)) * np.sqrt(2 * t)[None, :]).astype(np.float32))
        p["beta"] = mx.array(rng.uniform(0, 2 * np.pi, D).astype(np.float32)); p["D"] = D
    else:                                                      # poly-only sketch, dim m
        m = min(M, m_sketch * 8)
        for k in (1, 2):
            h = rng.integers(0, m, d); S = np.zeros((d, m), np.float32); S[np.arange(d), h] = 1.0
            p[f"S{k}"] = mx.array(S); p[f"s{k}"] = mx.array((rng.integers(0, 2, d) * 2 - 1).astype(np.float32))
        p["m_sketch"] = m
    return p


def ridge_acc(Ztr, ytr, Zte, yte, C, lam):
    Y = np.zeros((len(ytr), C), np.float32); Y[np.arange(len(ytr)), ytr] = 1.0
    M = Ztr.shape[1]
    A = Ztr.T @ Ztr + lam * np.eye(M, dtype=np.float32)
    W = np.linalg.solve(A, Ztr.T @ Y)
    return float((np.argmax(Zte @ W, 1) == yte).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cifar10", "cifar100"])
    ap.add_argument("--Ms", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--m-sketch", type=int, default=128)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "cifar_classify.json"))
    args = ap.parse_args()
    out = {"config": vars(args), "datasets": {}}
    for name in args.datasets:
        z = np.load(os.path.expanduser(f"~/rf_data/{name}_clip.npz"))
        Xtr, ytr, Xte, yte = z["X"], z["y"].astype(int), z["X_test"], z["y_test"].astype(int)
        Xtr, Xte = offsphere(Xtr, Xte)
        d = Xtr.shape[1]; C = int(ytr.max()) + 1; b = 1.0
        s = Xtr[:2000]
        eps = float(np.median(np.sum((s[:, None] - s[None]) ** 2, -1)[np.triu_indices(len(s), 1)]))
        gamma = 1.0 / eps
        log(f"==== {name}: N={len(ytr)} d={d} C={C} eps={eps:.3f} ====")
        rows = []
        # linear once
        acc = ridge_acc(Xtr, ytr, Xte, yte, C, args.lam)
        rows.append({"method": "linear", "M": d, "acc": acc}); log(f"  linear   M={d:5d}  acc={acc:.4f}")
        for M in args.Ms:
            for method in ["gauss", "imq", "poly", "tsray"]:
                if method in ("gauss", "tsray"):
                    p = make_params(method, d, M, eps, gamma, b, args.m_sketch, 0)
                else:
                    p = make_single_params(method, d, M, eps, b, args.m_sketch, 0)
                Ztr = features(Xtr, p); Zte = features(Xte, p)
                acc = ridge_acc(Ztr, ytr, Zte, yte, C, args.lam)
                rows.append({"method": method, "M": M, "M_actual": Ztr.shape[1],
                             "D": p.get("D"), "acc": acc})
                log(f"  {method:6s} M={M:5d} dim={Ztr.shape[1]:6d}  acc={acc:.4f}")
        out["datasets"][name] = {"d": d, "C": C, "eps": eps, "rows": rows}
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()

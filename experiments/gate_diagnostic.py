#!/usr/bin/env python3
"""
Pre-flight coupling diagnostic: does a dataset have the alignment x proximity structure that the
biased yat-kernel exploits, BEFORE spending training compute?

Idea (the cheap filter recommended for the signal-gate story): on a small subsample, score every
pair by four kernels and ask which one best separates same-class from different-class pairs and
best ranks same-class neighbors:

  radial (IMQ)        h_ij = 1 / (||xi-xj||^2 + eps)          -- proximity only
  alignment (poly)    a_ij = (xi.xj + b)^2                    -- direction only
  yat (gated)         k_ij = a_ij * h_ij                      -- the biased yat-kernel
  yat-normalized      g_ij = a_ij/((||xi||^2+b)(||xj||^2+b)); k_ij = g_ij * h_ij

If yat / yat-normalized beat BOTH single factors on pair-AUC and precision@k, the target needs the
product and RAY is worth training (it should help). If radial alone already wins (as on HIGGS),
the coupling is absent and Gaussian/IMQ RFF will lead -- skip the big run. This is the real-data
analogue of signal_gate_snr.py and the gate that decides which scaling_suite.py datasets to run.

Works on any (X, y): tabular via scaling_suite.load, or a precomputed embedding .npz with arrays
X (n,d) and y (n,). For vision (the advisor's top coupling bet) extract frozen CLIP/DINOv2
embeddings into an .npz and point --npz at it.

Env: /opt/homebrew/bin/python3 (numpy, sklearn; scaling_suite for tabular loading).
Examples:
  /opt/homebrew/bin/python3 gate_diagnostic.py --datasets susy hepmass miniboone madelon gisette
  /opt/homebrew/bin/python3 gate_diagnostic.py --npz ~/rf_data/cifar10_clip.npz --name cifar10-clip
"""
import argparse, os, json, time
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(__file__)
T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def kernels(X, b, eps):
    """Return dict of n x n kernel matrices for the four scores."""
    G = X @ X.T                                              # alignment Gram
    nrm = np.einsum("ij,ij->i", X, X)                        # ||xi||^2
    sq = nrm[:, None] + nrm[None, :] - 2 * G                 # squared distances
    h = 1.0 / (sq + eps)
    a = (G + b) ** 2
    gate = a / ((nrm[:, None] + b) * (nrm[None, :] + b))     # normalized alignment in [0,1]
    return {"radial": h, "alignment": a, "yat": a * h, "yat_norm": gate * h}


def pair_auc(K, y, rng, n_pairs=200_000):
    """AUC separating same-class from different-class pairs by kernel value."""
    n = len(y)
    i = rng.integers(0, n, n_pairs); j = rng.integers(0, n, n_pairs)
    m = i != j; i, j = i[m], j[m]
    same = (y[i] == y[j]).astype(np.int32)
    score = K[i, j]
    return float(roc_auc_score(same, score)) if same.min() != same.max() else float("nan")


def precision_at_k(K, y, k=10):
    """Mean fraction of the top-k neighbors (excluding self) sharing the query's class."""
    n = len(y)
    Kc = K.copy(); np.fill_diagonal(Kc, -np.inf)
    top = np.argpartition(-Kc, k, axis=1)[:, :k]
    return float((y[top] == y[:, None]).mean())


def diagnose(X, y, name, b, n_sub, seed):
    rng = np.random.default_rng(seed)
    if len(y) > n_sub:
        idx = rng.permutation(len(y))[:n_sub]; X, y = X[idx], y[idx]
    eps = float(np.median(np.sum((X[:1500][:, None] - X[:1500][None]) ** 2, -1)[np.triu_indices(min(1500, len(y)), 1)]))
    Ks = kernels(X.astype(np.float64), b, eps)
    auc = {k: pair_auc(K, y, rng) for k, K in Ks.items()}
    p10 = {k: precision_at_k(K, y, 10) for k, K in Ks.items()}
    best = max(("yat", "yat_norm"), key=lambda k: auc[k])
    coupling = auc[best] > max(auc["radial"], auc["alignment"]) + 1e-3
    log(f"{name:16s} n={len(y)} d={X.shape[1]} eps={eps:.3f} pos={y.mean():.2f}  "
        f"pairAUC[rad={auc['radial']:.3f} al={auc['alignment']:.3f} yat={auc['yat']:.3f} "
        f"yatN={auc['yat_norm']:.3f}]  P@10[rad={p10['radial']:.3f} yat={p10['yat']:.3f}]  "
        f"COUPLING={'YES -> train RAY' if coupling else 'no -> Gaussian likely leads'}")
    return {"name": name, "n": int(len(y)), "d": int(X.shape[1]), "eps": eps,
            "pos_rate": float(y.mean()), "pair_auc": auc, "prec_at_10": p10,
            "coupling": bool(coupling)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=[])
    ap.add_argument("--npz", default=None, help="path to .npz with X,(n,d) and y,(n,)")
    ap.add_argument("--name", default="npz")
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--n-sub", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--raw", action="store_true", help="skip standardization (npz path); default standardizes to match scaling_suite")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "gate_diagnostic.json"))
    args = ap.parse_args()
    res = []
    if args.npz:
        z = np.load(os.path.expanduser(args.npz)); X = z["X"].astype(np.float64)
        if not args.raw:                                     # match scaling_suite._offsphere: per-dim standardize + bound norm
            X = (X - X.mean(0)) / (X.std(0) + 1e-6)
            X = X / (np.percentile(np.linalg.norm(X, axis=1), 99.9) + 1e-9)
        res.append(diagnose(X, z["y"], args.name, args.b, args.n_sub, args.seed))
    if args.datasets:
        from scaling_suite import load
        for name in args.datasets:
            Xtr, ytr, _, _ = load(name, args.n_sub * 4, 1)
            res.append(diagnose(Xtr, ytr, name, args.b, args.n_sub, args.seed))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()

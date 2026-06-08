#!/usr/bin/env python3
"""
Off-sphere bounded-ball validation (R4 #1, #7): does RAY work where no dot-product
reduction exists?

Every other experiment is sphere-normalized, where k_yat,b reduces to a dot-product
kernel. The RAY paper's distinct claim is GENERAL R^d applicability. We test it on a
bounded ball with VARYING norms (||x|| in [0,R], not all =1), so k_yat,b is genuinely
nonstationary and non-dot-product: it is not a function of x.w alone.

Points: direction ~ uniform on S^{d-1}, radius r = R * U^{1/d} (uniform in the ball),
giving a spread of norms. We approximate the exact biased Gram with RAY (flat D'=1) and
uniform Nystrom, and report relative Frobenius error vs D (the O(1/sqrt D) rate) and vs d.

------------------------------------------------------------------------------
REPRODUCIBILITY
    Env : python3>=3.9, numpy, scikit-learn. CPU. Run: ~/.pixi/envs/jax/bin/python3 off_sphere_gram.py
    Out : results/off_sphere_gram.json (+ stdout). Deterministic seeds.
    Reuses k_yat / ray_cross / sqdist / nystrom from krr_downstream.py.
------------------------------------------------------------------------------
"""
import argparse, json, os, time
import numpy as np
import krr_downstream as K

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def sample_ball(N, d, R, rng, r_lo=0.25):
    """Uniform direction, radius uniform in [r_lo, R] -- a genuine d-independent norm
    spread (uniform-in-ball would concentrate at R as d grows, collapsing back to the
    sphere). Norms span [r_lo, R] at every d, so k_yat,b is genuinely non-dot-product."""
    g = rng.normal(size=(N, d))
    dirs = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
    r = rng.uniform(r_lo, R, size=(N, 1))
    return dirs * r


def rel_fro(A, B):
    return float(np.linalg.norm(A - B) / np.linalg.norm(B))


def _nystrom_from_landmarks(X, Z, b, eps):
    Kmm_pinv = np.linalg.pinv(K.k_yat(Z, Z, b, eps), rcond=1e-10)
    Knm = K.k_yat(X, Z, b, eps)
    return Knm @ Kmm_pinv @ Knm.T


def nystrom_gram(X, b, eps, m, rng):
    """Uniform-landmark Nystrom."""
    idx = rng.choice(X.shape[0], size=min(m, X.shape[0]), replace=False)
    return _nystrom_from_landmarks(X, X[idx], b, eps)


def nystrom_gram_kmeans(X, b, eps, m, seed):
    """Stronger baseline: k-means landmarks (cluster centers), not random rows."""
    from sklearn.cluster import KMeans
    Z = KMeans(n_clusters=min(m, X.shape[0]), n_init=4, random_state=seed).fit(X).cluster_centers_
    return _nystrom_from_landmarks(X, Z, b, eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--R", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--ds", type=int, nargs="+", default=[2, 8, 16, 32])
    ap.add_argument("--Ds", type=int, nargs="+", default=[10, 50, 100, 500, 1000])
    ap.add_argument("--m", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "results", "off_sphere_gram.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    log(f"config: {vars(args)}")
    out = {"config": vars(args), "by_d": {}}

    for d in args.ds:
        X = sample_ball(args.N, d, args.R, np.random.default_rng(0))
        norms = np.linalg.norm(X, axis=1)
        sq = K.sqdist(X, X)
        eps = float(np.median(sq[np.triu_indices(args.N, 1)]))
        Kex = K.k_yat(X, X, args.b, eps)
        # confirm genuinely off-sphere / non-dot-product: norms vary
        log(f"=== d={d}: ||x|| in [{norms.min():.2f},{norms.max():.2f}] mean {norms.mean():.2f}, "
            f"eps={eps:.3f} ===")
        rec = {"eps": eps, "norm_min": float(norms.min()), "norm_max": float(norms.max()),
               "ryf": {}, "nystrom": None}
        for D in args.Ds:
            errs = [rel_fro(K.ray_cross(X, X, args.b, eps, D, np.random.default_rng(10 + s)), Kex)
                    for s in range(args.seeds)]
            rec["ryf"][str(D)] = [float(np.mean(errs)), float(np.std(errs))]
        nerrs = [rel_fro(nystrom_gram(X, args.b, eps, args.m, np.random.default_rng(20 + s)), Kex)
                 for s in range(args.seeds)]
        rec["nystrom"] = [float(np.mean(nerrs)), float(np.std(nerrs))]
        kerrs = [rel_fro(nystrom_gram_kmeans(X, args.b, eps, args.m, 30 + s), Kex)
                 for s in range(args.seeds)]
        rec["nystrom_kmeans"] = [float(np.mean(kerrs)), float(np.std(kerrs))]
        out["by_d"][str(d)] = rec
        row = "  ".join(f"D{D}={rec['ryf'][str(D)][0]:.3f}" for D in args.Ds)
        log(f"  RAY: {row}   Nystrom(m={args.m})={rec['nystrom'][0]:.3f}  "
            f"kmeans-Nystrom={rec['nystrom_kmeans'][0]:.3f}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()

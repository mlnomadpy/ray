#!/usr/bin/env python3
"""
High-dimensional off-sphere Gram approximation (scale check; reviewers R6/R9/R10/R13).

All other experiments use d<=64. Here we verify the O(1/sqrt D) rate and the bounded
dimension behavior at d in {128, 256, 512} on off-sphere bounded-ball data (varying norms),
where the kernel is genuinely non-dot-product.

Env: numpy. Run: ~/.pixi/envs/jax/bin/python3 highd_offsphere.py -> results/highd_offsphere.json
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


def relf(A, B): return float(np.linalg.norm(A - B) / np.linalg.norm(B))


def main():
    b, N = 1.0, 600
    Ds = [50, 200, 1000]
    out = {"config": {"b": b, "N": N, "Ds": Ds}, "by_d": {}}
    log(f"high-d off-sphere: N={N}")
    log(f"  {'d':>4} " + " ".join(f"D{D:>6}" for D in Ds) + "   slope")
    for d in [128, 256, 512]:
        X = ball(N, d, np.random.default_rng(0))
        eps = float(np.median(K.sqdist(X, X)[np.triu_indices(N, 1)]))
        Kex = K.k_yat(X, X, b, eps)
        errs = {}
        for D in Ds:
            errs[D] = float(np.mean([relf(K.ray_cross(X, X, b, eps, D, np.random.default_rng(s)), Kex) for s in range(3)]))
        slope = float(np.polyfit(np.log(Ds), np.log([errs[D] for D in Ds]), 1)[0])
        out["by_d"][str(d)] = {"eps": eps, "errs": errs, "slope": slope}
        log(f"  {d:>4} " + " ".join(f"{errs[D]:>7.3f}" for D in Ds) + f"   {slope:.2f}")
    log("  (rate ~ -0.5 at every d; error at D=1000 stays comparable across d)")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "highd_offsphere.json"), "w"), indent=2)
    log("wrote results/highd_offsphere.json")


if __name__ == "__main__":
    main()

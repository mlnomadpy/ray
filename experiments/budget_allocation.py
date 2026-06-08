#!/usr/bin/env python3
"""
Outer/inner budget allocation for Random Yat-Features (#9).

The estimator has two sampling knobs: D outer radial scales (t ~ Exp(eps)) and D'
inner Gaussian RFF per scale. They cost the same: building the Gram is O(N^2 D D')
either way. Question: at a FIXED random-feature budget B = D * D', what split
minimizes error?

Variance decomposition (per pair): Var = poly^2 [ Var_t(g_t)/D + E_t v(t)/(D D') ].
With D D' = B fixed the second term is constant; the first falls as 1/D. So the
prediction is that error is MINIMIZED by making D as large as possible, i.e.
D' = 1 -- flat sampling of independent (scale, frequency) pairs beats the
two-level scheme at equal cost. We test this directly on the Gram matrix.

------------------------------------------------------------------------------
REPRODUCIBILITY
    Env  : python3>=3.9, numpy>=1.24 (CPU). Run: ~/.pixi/envs/jax/bin/python3 budget_allocation.py
    Out  : results/budget_allocation.json (+ stdout). Wall: ~1-2 min. Deterministic seeds.
------------------------------------------------------------------------------
"""
import argparse, json, os, time
import numpy as np

T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def sphere(rng, N, d):
    X = rng.uniform(-1.0, 1.0, size=(N, d))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def exact_gram(X, b, eps):
    G = X @ X.T
    sq = np.sum(X * X, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * G
    return (G + b) ** 2 / (D2 + eps)


def ray_gram(X, b, eps, D, Dp, rng):
    N = X.shape[0]
    P = (X @ X.T + b) ** 2 / eps
    K = np.zeros((N, N))
    ts = rng.exponential(scale=1.0 / eps, size=D)
    for t in ts:
        W = rng.normal(size=(X.shape[1], Dp)) * np.sqrt(2.0 * t)
        beta = rng.uniform(0.0, 2.0 * np.pi, size=Dp)
        Psi = np.sqrt(2.0 / Dp) * np.cos(X @ W + beta)
        K += (Psi @ Psi.T) * P
    return K / D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=400)
    ap.add_argument("--d", type=int, default=10)
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--budgets", type=int, nargs="+", default=[200, 1000])
    ap.add_argument("--Dprimes", type=int, nargs="+", default=[1, 2, 5, 10, 25, 50, 100, 200])
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "results", "budget_allocation.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    log(f"config: {vars(args)}")

    out = {"config": vars(args), "error": {}}
    for B in args.budgets:
        log(f"=== budget B = D*D' = {B} ===")
        for Dp in args.Dprimes:
            if B % Dp != 0 or B // Dp < 1:
                continue
            D = B // Dp
            errs = []
            for s in range(args.seeds):
                rng = np.random.default_rng(5 * s + 13)
                X = sphere(rng, args.N, args.d)
                K = exact_gram(X, args.b, args.eps)
                Kapp = ray_gram(X, args.b, args.eps, D, Dp, np.random.default_rng(600 + s))
                errs.append(float(np.linalg.norm(Kapp - K) / np.linalg.norm(K)))
            out["error"].setdefault(str(B), {})[str(Dp)] = {
                "D": D, "Dprime": Dp, "rel_fro": float(np.mean(errs)), "std": float(np.std(errs))}
            log(f"  B={B}  D={D:4d} D'={Dp:3d}  rel_fro={np.mean(errs):.4f} +/- {np.std(errs):.4f}")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()

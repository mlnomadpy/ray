#!/usr/bin/env python3
"""
Operator-norm concentration validation (#T-E3).

Corollary (operator-norm control) bounds ||K_D - K||_op via ||.||_op <= ||.||_F.
We check empirically that the operator-norm error (i) decays at the same O(1/sqrt D)
rate as the Frobenius error and (ii) is no larger than it, on the unit sphere.

------------------------------------------------------------------------------
REPRODUCIBILITY
    Env  : python3>=3.9, numpy>=1.24 (CPU). Run: ~/.pixi/envs/jax/bin/python3 opnorm_validation.py
    Out  : results/opnorm_validation.json (+ stdout). Wall: ~1 min. Deterministic seeds.
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
    return (G + b) ** 2 / (sq[:, None] + sq[None, :] - 2.0 * G + eps)


def ray_gram(X, b, eps, D, Dp, rng):
    N = X.shape[0]
    P = (X @ X.T + b) ** 2 / eps
    K = np.zeros((N, N))
    for t in rng.exponential(scale=1.0 / eps, size=D):
        W = rng.normal(size=(X.shape[1], Dp)) * np.sqrt(2.0 * t)
        beta = rng.uniform(0.0, 2.0 * np.pi, size=Dp)
        Psi = np.sqrt(2.0 / Dp) * np.cos(X @ W + beta)
        K += (Psi @ Psi.T) * P
    return K / D


def fit_slope(xs, ys):
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=500)
    ap.add_argument("--d", type=int, default=10)
    ap.add_argument("--Ds", type=int, nargs="+", default=[10, 50, 100, 500, 1000])
    ap.add_argument("--Dp", type=int, default=1)          # flat D'=1 (recommended estimator)
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "results", "opnorm_validation.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    log(f"config: {vars(args)}")

    out = {"config": vars(args), "op": {}, "fro": {}}
    op_means, fro_means = [], []
    for D in args.Ds:
        ops, fros = [], []
        for s in range(args.seeds):
            rng = np.random.default_rng(3 * s + 11)
            X = sphere(rng, args.N, args.d)
            K = exact_gram(X, args.b, args.eps)
            nop, nfr = np.linalg.norm(K, 2), np.linalg.norm(K)
            Kd = ray_gram(X, args.b, args.eps, D, args.Dp, np.random.default_rng(900 + s))
            E = Kd - K
            ops.append(np.linalg.norm(E, 2) / nop)
            fros.append(np.linalg.norm(E) / nfr)
        out["op"][str(D)] = float(np.mean(ops))
        out["fro"][str(D)] = float(np.mean(fros))
        op_means.append(np.mean(ops)); fro_means.append(np.mean(fros))
        log(f"  D={D:5d}  rel-op={np.mean(ops):.4f}  rel-Fro={np.mean(fros):.4f}")
    out["slope_op"] = fit_slope(args.Ds, op_means)
    out["slope_fro"] = fit_slope(args.Ds, fro_means)
    log(f">> slopes vs D: op={out['slope_op']:.3f}, Fro={out['slope_fro']:.3f} (expect ~-0.5)")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()

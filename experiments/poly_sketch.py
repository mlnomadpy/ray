#!/usr/bin/env python3
"""
Polynomial sketching of the biased factor (#8).

The exact biased polynomial feature is d_b = d^2+d+1 dimensional, the only part of
RAY that scales poorly in d. The paper claims TensorSketch reduces it to a chosen
D_poly << d^2 with controllable error. We verify this.

Augment x_aug = (x, sqrt(b)) in R^{d+1} so x_aug^T w_aug = x^T w + b; then
TensorSketch of degree 2 gives an unbiased D_poly-dim feature whose inner product
approximates (x_aug^T w_aug)^2 = (x^T w + b)^2. We report (a) the pure polynomial
sketch error ||P_sketch - P_exact||_F / ||P_exact||_F and (b) the full RAY Gram
error with the sketched polynomial, both vs D_poly, against (d+1)^2.

------------------------------------------------------------------------------
REPRODUCIBILITY
    Env  : python3>=3.9, numpy>=1.24 (CPU). Run: ~/.pixi/envs/jax/bin/python3 poly_sketch.py
    Out  : results/poly_sketch.json (+ stdout). Wall: ~1 min. Deterministic seeds.
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


def count_sketch(X, Dpoly, rng):
    din = X.shape[1]
    h = rng.integers(0, Dpoly, size=din)
    s = rng.choice([-1.0, 1.0], size=din)
    out = np.zeros((X.shape[0], Dpoly))
    for i in range(din):
        out[:, h[i]] += s[i] * X[:, i]
    return out


def tensorsketch_deg2(X, Dpoly, rng):
    """Unbiased TensorSketch: <TS(x),TS(w)> ~ (x^T w)^2."""
    C1 = count_sketch(X, Dpoly, rng)
    C2 = count_sketch(X, Dpoly, rng)
    F = np.fft.fft(C1, axis=1) * np.fft.fft(C2, axis=1)
    return np.fft.ifft(F, axis=1).real


def ray_gram_with_P(X, P, eps, D, Dp, rng):
    N = X.shape[0]
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
    ap.add_argument("--Dpolys", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    ap.add_argument("--D", type=int, default=500)
    ap.add_argument("--Dp", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "results", "poly_sketch.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dexact = (args.d + 1) ** 2
    log(f"config: {vars(args)}  exact poly dim (d+1)^2 = {dexact}")

    out = {"config": vars(args), "exact_poly_dim": dexact, "sketch": {}}
    for Dpoly in args.Dpolys:
        perr, kerr = [], []
        for s in range(args.seeds):
            rng = np.random.default_rng(7 * s + 3)
            X = sphere(rng, args.N, args.d)
            Xaug = np.concatenate([X, np.full((args.N, 1), np.sqrt(args.b))], axis=1)
            P_exact = (X @ X.T + args.b) ** 2 / args.eps
            TS = tensorsketch_deg2(Xaug, Dpoly, np.random.default_rng(400 + s))
            P_sketch = (TS @ TS.T) / args.eps
            perr.append(float(np.linalg.norm(P_sketch - P_exact) / np.linalg.norm(P_exact)))
            K = exact_gram(X, args.b, args.eps)
            Kapp = ray_gram_with_P(X, P_sketch, args.eps, args.D, args.Dp,
                                   np.random.default_rng(500 + s))
            kerr.append(float(np.linalg.norm(Kapp - K) / np.linalg.norm(K)))
        out["sketch"][str(Dpoly)] = {"poly_rel_fro": float(np.mean(perr)),
                                     "full_rel_fro": float(np.mean(kerr))}
        log(f"  D_poly={Dpoly:4d}  poly_err={np.mean(perr):.4f}  full_RAY_err={np.mean(kerr):.4f}")
    # reference: exact polynomial (no sketch)
    ref = []
    for s in range(args.seeds):
        rng = np.random.default_rng(7 * s + 3)
        X = sphere(rng, args.N, args.d)
        P_exact = (X @ X.T + args.b) ** 2 / args.eps
        K = exact_gram(X, args.b, args.eps)
        Kapp = ray_gram_with_P(X, P_exact, args.eps, args.D, args.Dp, np.random.default_rng(500 + s))
        ref.append(float(np.linalg.norm(Kapp - K) / np.linalg.norm(K)))
    out["exact_poly_full_rel_fro"] = float(np.mean(ref))
    log(f"  exact polynomial (dim {dexact}): full_RAY_err={np.mean(ref):.4f}")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()

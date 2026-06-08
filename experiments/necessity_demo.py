#!/usr/bin/env python3
"""
When is the yat-kernel the right inductive bias? (R7 #7, the "so what")

The yat-kernel couples ALIGNMENT (x.w) and PROXIMITY (||x-w||). To show this coupling
is necessary -- not just that we can approximate the kernel -- we build three off-sphere
regression targets and compare exact kernels + RAY by KRR test error:

  coupled : y = sum_k a_k (u_k.x)^2 / (||x-v_k||^2 + e0)   (needs BOTH)
  prox    : y = sum_k a_k / (||x-v_k||^2 + e0)             (proximity only)
  align   : y = sum_k a_k (u_k.x)^2                         (alignment only)

Prediction (the necessity argument): the yat-kernel wins on `coupled` and only there;
on `prox` the distance kernels (Gaussian/IMQ) match it, on `align` the polynomial kernel
matches it. So the win is attributable to the coupling, not to the kernel being generically
better. Data is off-sphere (radii in [0.3,1.5], varying norms) so the kernel does not
collapse to a dot-product kernel.

------------------------------------------------------------------------------
REPRODUCIBILITY
    Env : python3>=3.9, numpy, scipy, scikit-learn. CPU.
          Run: ~/.pixi/envs/jax/bin/python3 necessity_demo.py
    Out : results/necessity_demo.json (+ stdout). Deterministic seeds.
    Reuses k_yat/k_imq/k_gauss/ray_cross/sqdist/evaluate from krr_downstream.py.
------------------------------------------------------------------------------
"""
import argparse, json, os, time
import numpy as np
import krr_downstream as K

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def sample_ball(N, d, rng, r_lo=0.3, r_hi=1.5):
    g = rng.normal(size=(N, d))
    dirs = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
    return dirs * rng.uniform(r_lo, r_hi, size=(N, 1))


def k_poly(A, B, b):
    return (A @ B.T + b) ** 2


def make_targets(X, rng, K_atoms=6, e0=0.3):
    """The two single-factor CONTROLS are kernel-natural so they cleanly favor the right
    kernel (alignment = (u.x)^2 favors the polynomial kernel; proximity = 1/(.^2+e0) favors
    the distance kernels). The COUPLED target is deliberately NOT yat-shaped -- it multiplies
    a tanh alignment by a Laplace proximity, matching no candidate kernel's form -- so a yat
    win there cannot be an artifact of yat-generated data."""
    d = X.shape[1]
    U = rng.normal(size=(K_atoms, d)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    V = sample_ball(K_atoms, d, rng)
    a = rng.normal(size=K_atoms)
    align_poly = (X @ U.T) ** 2                            # polynomial-shaped (control)
    prox_rat = 1.0 / (K.sqdist(X, V) + e0)                 # rational-distance (control)
    align_tanh = np.tanh(2.0 * (X @ U.T))                  # non-yat alignment (coupled only)
    prox_lap = np.exp(-np.sqrt(K.sqdist(X, V) + 1e-9))     # non-yat proximity (coupled only)
    y_coupled = (align_tanh * prox_lap) @ a                # needs both, matches no kernel
    y_prox = prox_rat @ a
    y_align = align_poly @ a
    out = {}
    for name, y in [("coupled", y_coupled), ("prox", y_prox), ("align", y_align)]:
        out[name] = (y - y.mean()) / (y.std() + 1e-9)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--D", type=int, default=4000)       # RAY radial draws (enough to converge to exact yat)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "results", "necessity_demo.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    log(f"config: {vars(args)}")

    agg = {}  # (target, method) -> list of RMSE
    for s in range(args.seeds):
        rng = np.random.default_rng(10 + s)
        X = sample_ball(args.N, args.d, rng)
        targets = make_targets(X, np.random.default_rng(777))  # fixed target across seeds
        ntr = args.N * 2 // 3
        perm = rng.permutation(args.N); tr, te = perm[:ntr], perm[ntr:]
        Xtr, Xte = X[tr], X[te]
        sub = rng.choice(args.N, size=min(args.N, 1000), replace=False)
        eps = float(np.median(K.sqdist(X[sub], X[sub])[np.triu_indices(len(sub), 1)]))
        gamma = 1.0 / eps
        # precompute kernel Grams once per seed
        grams = {
            "Gaussian": (K.k_gauss(Xtr, Xtr, gamma), K.k_gauss(Xte, Xtr, gamma)),
            "IMQ":      (K.k_imq(Xtr, Xtr, eps), K.k_imq(Xte, Xtr, eps)),
            "poly2":    (k_poly(Xtr, Xtr, args.b), k_poly(Xte, Xtr, args.b)),
            "yat":      (K.k_yat(Xtr, Xtr, args.b, eps), K.k_yat(Xte, Xtr, args.b, eps)),
            "RYF":      (K.ray_cross(Xtr, Xtr, args.b, eps, args.D, np.random.default_rng(900 + s)),
                         K.ray_cross(Xte, Xtr, args.b, eps, args.D, np.random.default_rng(900 + s))),
        }
        for tname, y in targets.items():
            ytr, yte = y[tr], y[te]
            for mname, (Ktr, Kte) in grams.items():
                rmse = K.evaluate("reg", Ktr, Kte, ytr, yte, args.lam)
                agg.setdefault((tname, mname), []).append(rmse)

    summary = {f"{t}|{m}": [float(np.mean(v)), float(np.std(v))] for (t, m), v in agg.items()}
    methods = ["Gaussian", "IMQ", "poly2", "yat", "RYF"]
    log(f"=== off-sphere KRR test RMSE (d={args.d}, mean over {args.seeds} seeds) ===")
    log(f"  {'target':9s} " + "  ".join(f"{m:>9s}" for m in methods))
    for t in ["coupled", "prox", "align"]:
        row = "  ".join(f"{summary[f'{t}|{m}'][0]:9.3f}" for m in methods)
        best = min(methods, key=lambda m: summary[f'{t}|{m}'][0])
        log(f"  {t:9s} {row}    best={best}")
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "results": summary}, f, indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()

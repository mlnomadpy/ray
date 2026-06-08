#!/usr/bin/env python3
"""
Bias-scaling validation for Random Yat-Features (#1).

Theorem 3.3 bounds the estimator variance by
    (||x||^2 + b)^2 (||w||^2 + b)^2 / (D (2||x-w||^2 + eps))  <=  (R^2+b)^4 / (D eps),
so for UNIT vectors (R=1) the bound scales as (1+b)^4 and its sqrt (the uniform
error of Theorem 3.4) as (1+b)^2.

The *actual* per-pair variance scales with the polynomial factor squared,
(x^T w + b)^4, which is the bound's value only when x^T w = R^2 (aligned pair);
for a non-aligned pair it is smaller, the gap being exactly the Cauchy--Schwarz
slack x^T w <= ||x|| ||w||.

We test both: an aligned pair (rho = x^T w = 1, i.e. x = w) where actual = bound
scaling, and a rho = 0.5 pair. For each we sweep b and fit the log-log slope of
empirical variance vs (rho + b); the prediction is slope ~ 4 (and ~2 for std).

------------------------------------------------------------------------------
REPRODUCIBILITY
    Env  : python3>=3.9, numpy>=1.24 (CPU). Run: ~/.pixi/envs/jax/bin/python3 bias_scaling.py
    Out  : results/bias_scaling.json (+ stdout). Wall: seconds. Deterministic seeds.
------------------------------------------------------------------------------
"""
import argparse, json, os, time
import numpy as np

T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def cos_est(x, w, b, eps, D, Dp, rng):
    """Cosine RAY estimate of k_{E,b}(x,w)."""
    d = x.shape[0]
    poly = (x @ w + b) ** 2 / eps
    ts = rng.exponential(scale=1.0 / eps, size=D)
    acc = 0.0
    for t in ts:
        W = rng.normal(size=(d, Dp)) * np.sqrt(2.0 * t)
        beta = rng.uniform(0.0, 2.0 * np.pi, size=Dp)
        zx = np.sqrt(2.0 / Dp) * np.cos(x @ W + beta)
        zw = np.sqrt(2.0 / Dp) * np.cos(w @ W + beta)
        acc += float(zx @ zw)
    return (acc / D) * poly


def fit_slope(xs, ys):
    """Least-squares slope of log(ys) vs log(xs)."""
    lx, ly = np.log(np.asarray(xs)), np.log(np.asarray(ys))
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, _ = np.linalg.lstsq(A, ly, rcond=None)[0]
    return float(slope)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=float, nargs="+", default=[0.0, 0.1, 0.5, 1.0, 2.0, 5.0])
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--D", type=int, default=200)
    ap.add_argument("--Dp", type=int, default=50)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "results", "bias_scaling.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    log(f"config: {vars(args)}")

    pairs = {  # rho = x^T w for unit vectors in R^2
        "aligned_rho1.0": (np.array([1.0, 0.0]), np.array([1.0, 0.0])),
        "rho0.5": (np.array([1.0, 0.0]), np.array([0.5, np.sqrt(0.75)])),
    }
    out = {"config": vars(args), "R": 1.0, "pairs": {}}
    for name, (x, w) in pairs.items():
        rho = float(x @ w)
        vars_, stds_, shifted = [], [], []
        for b in args.bs:
            ests = [cos_est(x, w, b, args.eps, args.D, args.Dp,
                            np.random.default_rng(101 * r + int(1000 * b)))
                    for r in range(args.reps)]
            v = float(np.var(ests))
            vars_.append(v); stds_.append(float(np.sqrt(v))); shifted.append(rho + b)
            log(f"  {name}  b={b:<4} (rho+b={rho+b:.2f})  var={v:.3e}  "
                f"var/(R^2+b)^4={v/((1.0+b)**4):.3e}")
        # fit exponent over b such that (rho+b)>0 (skip rho+b<=0)
        mask = [s > 1e-6 for s in shifted]
        xs = [s for s, m in zip(shifted, mask) if m]
        vy = [v for v, m in zip(vars_, mask) if m]
        sy = [s for s, m in zip(stds_, mask) if m]
        slope_var = fit_slope(xs, vy)
        slope_std = fit_slope(xs, sy)
        out["pairs"][name] = {"rho": rho, "bs": args.bs, "var": vars_, "std": stds_,
                              "slope_var_vs_(rho+b)": slope_var,
                              "slope_std_vs_(rho+b)": slope_std}
        log(f"  >> {name}: variance exponent={slope_var:.2f} (predict 4), "
            f"std exponent={slope_std:.2f} (predict 2)")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()

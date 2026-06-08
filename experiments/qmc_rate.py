#!/usr/bin/env python3
"""
QMC convergence-rate test (#5).

Section 4 claims quasi-Monte Carlo sampling of the radial scale t improves the
estimator error from the Monte Carlo rate O(1/sqrt(D)) toward O((log D)^s / D).
The Section 5.2 table only showed a constant-factor gain at a few D; here we test
the RATE directly by measuring the estimator RMSE (= sqrt(empirical variance) of
the unbiased kernel estimate) over a wide D grid and fitting log-log slopes:
    MC  : predicted slope -0.5
    QMC : predicted steeper (toward -1 if the rate claim holds)
We report the fitted slopes and the realized values, and let the data decide
whether QMC buys a rate or only a constant.

------------------------------------------------------------------------------
REPRODUCIBILITY
    Env  : python3>=3.9, numpy>=1.24, scipy>=1.10 (scipy.stats.qmc).
           Run: ~/.pixi/envs/jax/bin/python3 qmc_rate.py
    Out  : results/qmc_rate.json (+ stdout). Wall: ~1 min. Deterministic seeds.
------------------------------------------------------------------------------
"""
import argparse, json, os, time
import numpy as np

try:
    from scipy.stats import qmc
    HAVE_QMC = True
except Exception:
    HAVE_QMC = False

T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def exact_k(x, w, b, eps):
    return (x * w + b) ** 2 / ((x - w) ** 2 + eps)


def sample_t(D, eps, rng, sob):
    if sob is None:
        return rng.exponential(scale=1.0 / eps, size=D)
    u = np.clip(sob.random(D).ravel(), 1e-12, 1 - 1e-12)
    return -np.log(u) / eps


def est(method, x, w, b, eps, D, Dp, rng):
    poly = (x * w + b) ** 2 / eps
    sob = qmc.Sobol(d=1, scramble=True, seed=int(rng.integers(1 << 30))) \
        if method == "qmc" else None
    ts = sample_t(D, eps, rng, sob)
    acc = 0.0
    for t in ts:
        wf = rng.normal(size=Dp) * np.sqrt(2.0 * t)
        beta = rng.uniform(0.0, 2.0 * np.pi, size=Dp)
        zx = np.sqrt(2.0 / Dp) * np.cos(wf * x + beta)
        zw = np.sqrt(2.0 / Dp) * np.cos(wf * w + beta)
        acc += float(zx @ zw)
    return (acc / D) * poly


def fit_slope(xs, ys):
    lx, ly = np.log(np.asarray(xs)), np.log(np.asarray(ys))
    A = np.vstack([lx, np.ones_like(lx)]).T
    return float(np.linalg.lstsq(A, ly, rcond=None)[0][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=float, default=0.5)
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--Dp", type=int, default=50)
    ap.add_argument("--Ds", type=int, nargs="+",
                    default=[10, 20, 50, 100, 200, 500, 1000, 2000])
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "results", "qmc_rate.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if not HAVE_QMC:
        log("WARNING: scipy.stats.qmc unavailable -> QMC disabled.")
    kt = exact_k(args.x, args.w, args.b, args.eps)
    log(f"config: {vars(args)}  k_true={kt:.5f}")

    out = {"config": vars(args), "k_true": float(kt), "rmse": {"mc": {}, "qmc": {}}}
    methods = ["mc"] + (["qmc"] if HAVE_QMC else [])
    for m in methods:
        for D in args.Ds:
            ests = [est("basic" if m == "mc" else "qmc",
                        args.x, args.w, args.b, args.eps, D, args.Dp,
                        np.random.default_rng(57 * r + D + (0 if m == "mc" else 99999)))
                    for r in range(args.reps)]
            rmse = float(np.sqrt(np.var(ests)))   # unbiased -> RMSE ~ std
            out["rmse"][m][str(D)] = rmse
            log(f"  {m:4s} D={D:5d}  rmse={rmse:.4e}")
    out["slope_mc"] = fit_slope(args.Ds, [out["rmse"]["mc"][str(D)] for D in args.Ds])
    if HAVE_QMC:
        out["slope_qmc"] = fit_slope(args.Ds, [out["rmse"]["qmc"][str(D)] for D in args.Ds])
        log(f">> MC slope={out['slope_mc']:.3f} (predict -0.5), "
            f"QMC slope={out['slope_qmc']:.3f} (predict steeper if rate holds)")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()

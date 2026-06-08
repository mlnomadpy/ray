#!/usr/bin/env python3
"""
Importance-sampling validation (#7).

Section 4 proposes drawing the radial scale from t ~ Exp(eps+eta) (eta>0) with
weight w_t = (eps/(eps+eta)) e^{eta t}, claiming the estimator stays unbiased and
its variance drops for NEARBY pairs (||x-w||^2 <= eta) by up to (eps/(eps+eta))^2.

We test it: for pairs at several squared distances, estimate k_{E,b} over many
repetitions with eta in {0, 0.1, 0.5, 1.0} and report (a) that the mean stays at
k_true (unbiasedness) and (b) the variance vs eta, expecting a reduction for the
near pairs and a mild increase for the far ones.

------------------------------------------------------------------------------
REPRODUCIBILITY
    Env  : python3>=3.9, numpy>=1.24 (CPU). Run: ~/.pixi/envs/jax/bin/python3 importance_sampling.py
    Out  : results/importance_sampling.json (+ stdout). Wall: seconds. Deterministic seeds.
------------------------------------------------------------------------------
"""
import argparse, json, os, time
import numpy as np

T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def exact_k(dist2, xw, b, eps):
    return (xw + b) ** 2 / (dist2 + eps)


def est_is(dist2, xw, b, eps, eta, D, Dp, rng):
    """IS estimator: t ~ Exp(eps+eta), weight w_t = (eps/(eps+eta)) e^{eta t}.
    1-D pair with given ||x-w||^2 = dist2 and x^T w = xw; we only need scalars."""
    poly = (xw + b) ** 2 / eps
    rate = eps + eta
    ts = rng.exponential(scale=1.0 / rate, size=D)
    wts = (eps / rate) * np.exp(eta * ts)               # importance weights (1 if eta=0)
    # Gaussian factor estimate at scale t via D' cosine RFF, for a pair at distance^2=dist2.
    # E_omega[2 cos(w(x)+b)cos(w(y)+b)] over the 1-D gap g=sqrt(dist2): cos(omega*g).
    g = np.sqrt(dist2)
    acc = 0.0
    for t, wt in zip(ts, wts):
        om = rng.normal(size=Dp) * np.sqrt(2.0 * t)
        beta = rng.uniform(0.0, 2.0 * np.pi, size=Dp)
        ghat = float(np.mean(2.0 * np.cos(om * g + beta) * np.cos(beta)))  # unbiased for e^{-t g^2}
        acc += wt * ghat
    return (acc / D) * poly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist2s", type=float, nargs="+", default=[0.05, 0.25, 1.0, 4.0])
    ap.add_argument("--xw", type=float, default=0.5)      # x^T w, fixed
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--etas", type=float, nargs="+", default=[0.0, 0.1, 0.5, 1.0])
    ap.add_argument("--D", type=int, default=200)
    ap.add_argument("--Dp", type=int, default=50)
    ap.add_argument("--reps", type=int, default=3000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "results", "importance_sampling.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    log(f"config: {vars(args)}")

    out = {"config": vars(args), "results": {}}
    for dist2 in args.dist2s:
        kt = exact_k(dist2, args.xw, args.b, args.eps)
        row = {}
        for eta in args.etas:
            ests = [est_is(dist2, args.xw, args.b, args.eps, eta, args.D, args.Dp,
                           np.random.default_rng(31 * r + int(100 * eta) + int(10 * dist2)))
                    for r in range(args.reps)]
            mean, var = float(np.mean(ests)), float(np.var(ests))
            row[str(eta)] = {"mean": mean, "var": var, "bias": mean - kt}
            log(f"  ||x-w||^2={dist2:<4} eta={eta:<4}  mean={mean:.4f} (k={kt:.4f}, "
                f"bias={mean-kt:+.4f})  var={var:.3e}")
        v0 = row[str(args.etas[0])]["var"]
        for eta in args.etas:
            row[str(eta)]["var_ratio_vs_eta0"] = row[str(eta)]["var"] / v0
        out["results"][str(dist2)] = {"k_true": kt, "by_eta": row}
        best = min(args.etas, key=lambda e: row[str(e)]["var"])
        log(f"  >> ||x-w||^2={dist2}: best eta={best} "
            f"(var ratio {row[str(best)]['var']/v0:.2f} vs eta=0)")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Validate the exact flat-estimator variance (thm:exact_variance).

For a single pair (x,w) the flat RAY estimate is
    khat = (a/eps) * 2 cos(w.x+beta) cos(w.w'+beta),
    a = (x.w+b)^2,  T ~ Exp(eps),  omega|T ~ N(0, 2T I),  beta ~ U[0,2pi].
The theorem predicts the EXACT one-draw variance
    Var(khat) = a^2/eps^2 * [ 1 + (1/2) eps/(eps+4r) - (eps/(eps+r))^2 ],  r=||x-w||^2.
We measure the empirical variance over many independent draws for a grid of pairs spanning
a range of distances r and biases b, and compare to the closed form (ratio should be ~1).

------------------------------------------------------------------------------
REPRODUCIBILITY
    Env : python3>=3.9, numpy. CPU. Run: ~/.pixi/envs/jax/bin/python3 exact_variance_check.py
    Out : results/exact_variance_check.json (+ stdout). Deterministic seeds.
------------------------------------------------------------------------------
"""
import json, os, time
import numpy as np

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
HERE = os.path.dirname(__file__)


def predicted_var(a, r, eps):
    bracket = 1.0 + 0.5 * eps / (eps + 4 * r) - (eps / (eps + r)) ** 2
    return a ** 2 / eps ** 2 * bracket


def empirical_var(x, w, b, eps, M, rng):
    a = (x @ w + b) ** 2
    d = x.shape[0]
    est = np.empty(M)
    ts = rng.exponential(scale=1.0 / eps, size=M)
    for i, t in enumerate(ts):
        om = rng.normal(size=d) * np.sqrt(2.0 * t)
        beta = rng.uniform(0, 2 * np.pi)
        est[i] = (a / eps) * 2.0 * np.cos(om @ x + beta) * np.cos(om @ w + beta)
    return float(np.var(est)), float(np.mean(est)), a


def main():
    eps, d, M = 1.0, 6, 200000
    rng = np.random.default_rng(0)
    rows = []
    log(f"exact-variance check: eps={eps}, d={d}, M={M} draws/pair")
    log(f"  {'b':>4} {'r':>6} {'a':>7} {'emp.Var':>10} {'pred.Var':>10} {'ratio':>7} {'mean/k':>8}")
    for b in [0.0, 1.0, 2.0]:
        for seed in range(6):
            rg = np.random.default_rng(100 + seed)
            x = rg.normal(size=d); w = rg.normal(size=d)
            x *= rg.uniform(0.3, 1.2) / (np.linalg.norm(x) + 1e-9)
            w *= rg.uniform(0.3, 1.2) / (np.linalg.norm(w) + 1e-9)
            r = float(np.sum((x - w) ** 2))
            ev, em, a = empirical_var(x, w, b, eps, M, np.random.default_rng(500 + seed))
            pv = predicted_var(a, r, eps)
            ktrue = a / (r + eps)               # exact kernel value (mean target)
            rows.append({"b": b, "r": r, "a": a, "emp_var": ev, "pred_var": pv,
                         "ratio": ev / pv, "emp_mean": em, "k_true": ktrue})
            log(f"  {b:>4.1f} {r:>6.3f} {a:>7.3f} {ev:>10.4f} {pv:>10.4f} {ev/pv:>7.3f} {em/ktrue:>8.3f}")
    ratios = np.array([row["ratio"] for row in rows])
    log(f"  ratio empirical/predicted: mean {ratios.mean():.3f}, std {ratios.std():.3f} (target 1.000)")
    out = {"config": {"eps": eps, "d": d, "M": M}, "rows": rows,
           "ratio_mean": float(ratios.mean()), "ratio_std": float(ratios.std())}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "exact_variance_check.json"), "w") as f:
        json.dump(out, f, indent=2)
    log("wrote results/exact_variance_check.json")


if __name__ == "__main__":
    main()

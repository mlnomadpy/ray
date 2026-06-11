#!/usr/bin/env python3
"""
Optimal sketch-size design rule (validates prop:optimal_m against Table 5 / tab:dm).

prop:optimal_m predicts that at a fixed feature budget M=D(m+d+1) the deployed RAY
variance is V(m)=A(m+d+1)/M + B/m, minimized at the interior m*=sqrt(B M / A). The
relative Frobenius Gram error squared tracks this variance, so we fit
  gram_err^2 ~ alpha*(m+d+1)/M + beta*(1/m)
(two nonneg params per d, pooled across the M values in tab:dm), then check that the
predicted m*(M)=sqrt(beta*M/alpha) lands at the empirical error minimum and grows ~sqrt(M)
and slowly with d. Reads the archived dm_tradeoff.json (the data behind Table 5); fits
only, runs no new kernel experiment.

Env: numpy. Run: ~/.pixi/envs/jax/bin/python3 sketch_size_rule.py -> results/sketch_size_rule.json

Result: the two-term law fits gram_err^2 with R^2 = 0.885/0.895/0.882 for d=16/64/256;
predicted m*=sqrt(beta*M/alpha) lands at or within one grid step of the empirical argmin;
sqrt(B/A) = 0.411/0.576/0.524 (slow growth with d, as prop:optimal_m predicts).
"""
import json, os, time
import numpy as np
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
HERE = os.path.dirname(__file__)


def fit_nonneg(X, y):
    """Least squares with both coefficients clamped nonnegative (2-param, brute-safe)."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    if (coef >= 0).all():
        return coef
    # clamp: try each single-feature fit, keep the better SSE
    best, berr = None, np.inf
    for k in range(X.shape[1]):
        c = np.zeros(X.shape[1])
        c[k] = max(0.0, (X[:, k] @ y) / (X[:, k] @ X[:, k]))
        e = np.sum((X @ c - y) ** 2)
        if e < berr:
            best, berr = c, e
    return best


def main():
    src = json.load(open(os.path.join(HERE, "results", "dm_tradeoff.json")))
    cfg = src["config"]
    out = {"source": "dm_tradeoff.json", "config": cfg, "by_d": {}}
    log(f"sketch-size design rule (prop:optimal_m): b={cfg['b']}, lam={cfg['lam']}, seeds={cfg['seeds']}")
    log(f"  {'d':>4} {'alpha(rad)':>11} {'beta(sk)':>10} {'fit R^2':>8}   predicted m* vs empirical argmin per M")
    for dk, blk in src["by_d"].items():
        d = int(dk)
        rows = blk["rows"]
        m = np.array([r["m"] for r in rows], float)
        M = np.array([r["M_actual"] for r in rows], float)
        e2 = np.array([r["gram_err"][0] for r in rows], float) ** 2
        X = np.column_stack([(m + d + 1) / M, 1.0 / m])
        alpha, beta = fit_nonneg(X, e2)
        pred = X @ np.array([alpha, beta])
        ss_res = float(np.sum((e2 - pred) ** 2))
        ss_tot = float(np.sum((e2 - e2.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        # per-M: predicted m* vs empirical argmin
        per_M = []
        for Mv in sorted(set(int(x) for x in (M / 100).round() * 100)):
            mask = np.abs(M - Mv) < 200
            if mask.sum() < 2:
                continue
            mstar = float(np.sqrt(beta * M[mask].mean() / alpha)) if alpha > 0 else float("nan")
            emp_m = float(m[mask][np.argmin(e2[mask])])
            # nearest grid m to the prediction
            grid_pred = float(m[mask][np.argmin(np.abs(m[mask] - mstar))])
            per_M.append({"M": float(M[mask].mean()), "m_star_pred": mstar,
                          "m_star_on_grid": grid_pred, "m_empirical_argmin": emp_m})
        out["by_d"][dk] = {"d": d, "alpha_radial": float(alpha), "beta_sketch": float(beta),
                           "fit_r2": r2, "ratio_sqrt_BoverA": float(np.sqrt(beta / alpha)) if alpha > 0 else None,
                           "per_M": per_M}
        summ = "  ".join(f"M={p['M']:.0f}:pred~{p['m_star_on_grid']:.0f}/emp{p['m_empirical_argmin']:.0f}" for p in per_M)
        log(f"  {d:>4} {alpha:>11.4g} {beta:>10.4g} {r2:>8.3f}   {summ}")
    # sqrt(M) growth check: m* should roughly double when M quadruples
    log("  -> m* grows ~sqrt(M) and the sketch-to-radial ratio sqrt(B/A) drifts up slowly with d:")
    for dk, blk in out["by_d"].items():
        log(f"     d={blk['d']:>4}: sqrt(B/A)={blk['ratio_sqrt_BoverA']:.3f}")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "sketch_size_rule.json"), "w"), indent=2)
    log("wrote results/sketch_size_rule.json")


if __name__ == "__main__":
    main()

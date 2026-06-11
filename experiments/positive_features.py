#!/usr/bin/env python3
"""
Positive (FAVOR+) radial features vs the trigonometric estimator (validates prop:positive_dichotomy).

The Gaussian factor admits the positive feature phi+_t(x) = exp(omega.x - 2t||x||^2),
E[phi+(x)phi+(w)] = e^{-t||x-w||^2} (prop:positive). Composing it with the Bernstein
mixing law T~Exp(eps), the per-draw radial estimate ghat+ = phi+_T(x) phi+_T(w) is
unbiased for eps/(||x-w||^2+eps), BUT its second moment is
    E[(ghat+)^2] = E_T[e^{8 T x.w}] = eps/(eps - 8 x.w)   if 8 x.w < eps,  = +inf otherwise,
so the variance is INFINITE for every aligned pair with x.w >= eps/8.  The trigonometric
estimator has E[ghat^2 | T] <= 3/2 uniformly.  Scale truncation T<=Tmax restores a finite
second moment <= e^{8 Tmax x.w} at relative bias <= e^{-eps Tmax}.

We check:
 (A) pointwise: empirical second moment of ghat+ vs reps for a weakly aligned pair
     (8u<eps: converges to eps/(eps-8u)) and an aligned pair (8u>eps: grows without bound);
     trig second moment converges <= 3/2 for both.
 (B) prediction sweep: empirical Var[ghat+] across pairs with 8u/eps in [0.1, 2] matches
     eps/(eps-8u) - (eps/(eps+r))^2 below the threshold and explodes above it.
 (C) Gram: off-sphere ball, eps = {4,1,0.25} x median sqdist; relative Frobenius error of
     exact-modulation estimators with trig / positive-naive / positive-truncated radial
     features at matched draw budgets. Median and worst seed reported (positive-naive is
     heavy-tailed by (A)).

------------------------------------------------------------------------------
REPRODUCIBILITY
    Env : python3>=3.9, numpy. CPU. Run: ~/.pixi/envs/jax/bin/python3 positive_features.py
    Out : results/positive_features.json (+ stdout). Deterministic seeds.
------------------------------------------------------------------------------
"""
import json, os, time
import numpy as np

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
HERE = os.path.dirname(__file__)
SQ2 = float(np.sqrt(2.0))


def sqd(X):
    nn = (X * X).sum(1); return np.maximum(nn[:, None] + nn[None, :] - 2 * X @ X.T, 0.0)


def pair(u, d, ax=0.9, aw=0.9):
    """Two vectors in R^d with x.w=u, ||x||=ax, ||w||=aw; returns (x, w, r=||x-w||^2)."""
    assert abs(u) <= ax * aw + 1e-12, "infeasible pair"
    x = np.zeros(d); x[0] = ax
    w = np.zeros(d); w[0] = u / ax; w[1] = np.sqrt(max(aw ** 2 - (u / ax) ** 2, 0.0))
    return x, w, ax ** 2 + aw ** 2 - 2 * u


def ghat_pos(x, w, eps, n, rng, tmax=None):
    """n draws of phi+_T(x) phi+_T(w), T~Exp(eps) (truncated at tmax if given)."""
    if tmax is None:
        t = rng.exponential(1.0 / eps, size=n)
        scale = 1.0
    else:
        u = rng.uniform(0, 1, size=n)
        t = -np.log(1.0 - u * (1.0 - np.exp(-eps * tmax))) / eps
        scale = 1.0 - np.exp(-eps * tmax)       # unbiased for the truncated integral
    d = x.shape[0]
    om = rng.normal(size=(n, d)) * np.sqrt(2.0 * t)[:, None]
    return scale * np.exp(om @ x - 2 * t * (x @ x)) * np.exp(om @ w - 2 * t * (w @ w))


def ghat_trig(x, w, eps, n, rng):
    t = rng.exponential(1.0 / eps, size=n)
    d = x.shape[0]
    om = rng.normal(size=(n, d)) * np.sqrt(2.0 * t)[:, None]
    beta = rng.uniform(0, 2 * np.pi, size=n)
    return 2.0 * np.cos(om @ x + beta) * np.cos(om @ w + beta)


def gram_radial(X, eps, D, rng, kind, tmax=None):
    """(N,N) radial-factor estimate, exact modulation applied by caller."""
    N, d = X.shape
    acc = np.zeros((N, N))
    if kind == "trig":
        t = rng.exponential(1.0 / eps, size=D)
        Om = (rng.normal(size=(d, D)) * np.sqrt(2 * t)[None, :])
        C = SQ2 * np.cos(X @ Om + rng.uniform(0, 2 * np.pi, D))
        return (C @ C.T) / D
    if kind == "pos":
        t = rng.exponential(1.0 / eps, size=D); scale = 1.0
    elif kind == "pos_trunc":
        u = rng.uniform(0, 1, size=D)
        t = -np.log(1.0 - u * (1.0 - np.exp(-eps * tmax))) / eps
        scale = 1.0 - np.exp(-eps * tmax)
    Om = (rng.normal(size=(d, D)) * np.sqrt(2 * t)[None, :])
    F = np.exp(X @ Om - 2.0 * t[None, :] * (X * X).sum(1)[:, None])   # (N,D) phi+ features
    return scale * (F @ F.T) / D


def main():
    rng = np.random.default_rng(0)
    d, eps = 16, 1.0
    out = {"config": {"d": d}}

    # ---------------- (A) second-moment convergence / divergence
    log("(A) second moment of ghat+ vs reps (eps=1)")
    parts = []
    for name, u in [("weak (8u=0.4<eps)", 0.05), ("aligned (8u=4>eps)", 0.5)]:
        x, w, r = pair(u, d)
        row = {"pair": name, "u": u, "r": r, "pred_2nd": (eps / (eps - 8 * u) if 8 * u < eps else None),
               "pos": {}, "trig": {}}
        for n in [10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6]:
            gp = ghat_pos(x, w, eps, n, np.random.default_rng(1))
            gt = ghat_trig(x, w, eps, n, np.random.default_rng(1))
            row["pos"][n] = float(np.mean(gp ** 2))
            row["trig"][n] = float(np.mean(gt ** 2))
        parts.append(row)
        log(f"  {name}: pos 2nd moment {row['pos']}  (pred {row['pred_2nd']});"
            f" trig {row['trig'][10**6]:.3f} (<=1.5)")
    out["A_second_moment"] = parts

    # ---------------- (B) prediction sweep across the threshold
    log("(B) Var[ghat+] vs prediction across 8u/eps")
    reps = 200000
    sweep = []
    for ratio in [0.1, 0.25, 0.5, 0.75, 0.9, 1.1, 1.5, 2.0]:
        u = ratio * eps / 8.0
        x, w, r = pair(u, d)
        gp = ghat_pos(x, w, eps, reps, np.random.default_rng(2))
        emp = float(np.var(gp))
        mean_emp = float(np.mean(gp))
        pred = (eps / (eps - 8 * u) - (eps / (eps + r)) ** 2) if 8 * u < eps else None
        sweep.append({"ratio_8u_over_eps": ratio, "r": r, "emp_var": emp, "pred_var": pred,
                      "emp_mean": mean_emp, "true_mean": eps / (eps + r)})
        log(f"  8u/eps={ratio:4.2f}: emp var {emp:10.3f}  pred {pred if pred else 'inf'}"
            f"  (mean {mean_emp:.4f} vs {eps/(eps+r):.4f})")
    out["B_variance_sweep"] = sweep

    # ---------------- (C) Gram error off-sphere, three radial estimators
    log("(C) off-sphere Gram error, trig vs positive (naive/truncated)")
    N, b = 400, 1.0
    U = rng.normal(size=(N, d)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    X = U * rng.uniform(0.3, 1.0, size=(N, 1))
    med = float(np.median(sqd(X)[np.triu_indices(N, 1)]))
    Pmat = (X @ X.T + b) ** 2
    SEEDS = 10
    rows = []
    for mult in [4.0, 1.0, 0.25]:
        e = mult * med
        Kex = Pmat / (sqd(X) + e)
        Rex = e / (sqd(X) + e)
        Pn = Pmat / e
        align = float(np.max(8.0 * (X @ X.T) / e))
        tmax = np.log(100.0) / e                # 1% truncation bias
        for D in [128, 512, 2048]:
            errs = {"trig": [], "pos": [], "pos_trunc": []}
            for s in range(SEEDS):
                for kind in errs:
                    Rh = gram_radial(X, e, D, np.random.default_rng(10 * s + 3), kind,
                                     tmax=(tmax if kind == "pos_trunc" else None))
                    KD = Pn * Rh
                    errs[kind].append(float(np.linalg.norm(KD - Kex) / np.linalg.norm(Kex)))
            r = {"eps_mult": mult, "max_8u_over_eps": align, "D": D}
            for kind in errs:
                r[kind + "_med"] = float(np.median(errs[kind]))
                r[kind + "_max"] = float(np.max(errs[kind]))
            rows.append(r)
            log(f"  eps={mult}x med (max 8u/eps={align:5.1f}) D={D}: "
                f"trig {r['trig_med']:.3f}  pos {r['pos_med']:.3f} (worst {r['pos_max']:.3f})  "
                f"pos_trunc {r['pos_trunc_med']:.3f} (worst {r['pos_trunc_max']:.3f})")
    out["C_gram"] = rows

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "positive_features.json"), "w") as f:
        json.dump(out, f, indent=1)
    log("wrote results/positive_features.json")


if __name__ == "__main__":
    main()

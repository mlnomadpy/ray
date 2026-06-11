#!/usr/bin/env python3
"""
A kernel-grammar customer for the Bernstein-Schur class: (x.w+b)^2 x RationalQuadratic.

Compositional kernel search (the Automatic Statistician; Duvenaud et al. 2013) builds GP
kernels as products like LIN^2 x RQ. The rational-quadratic radial
    RQ_alpha(r) = (1 + r/(2 alpha sigma^2))^{-alpha},   r = ||x-w||^2,
is completely monotone in r: with eps' = 2 alpha sigma^2 it is (eps')^alpha (r+eps')^{-alpha},
i.e. a generalized IMQ with Gamma(alpha, rate eps') Bernstein mixing law and mass
m_f = RQ(0) = 1. So the grammar product (x.w+b)^2 RQ_alpha is a Bernstein-Schur kernel
and the class estimator of thm:bernstein_schur linearizes it: keep the degree-2 modulation
exact, draw T ~ Gamma(alpha, eps'), apply RFF -- no new analysis needed (thm:class_bernstein).

Checks on california housing (cached sklearn, d=8, standardized, max-norm scaled so the
ball is off-sphere with varying norms):
 (A) Gram fidelity: relative Frobenius error of the class estimator vs the exact composite
     Gram at N=2000 follows the MC rate in D;
 (B) downstream: exact composite-kernel KRR (N=3000 Gram) vs the class estimator in Gram
     form at D in {200,1000,4000} -- test RMSE converges to the exact kernel's;
 (C) scale: full-dataset (N=19640 train) primal ridge with explicit features
     (exact modulation d_b=45, D=100 -> M=4500), where the exact Gram (N^2=3.9e8 entries)
     is no longer comfortable -- the class estimator runs as an ordinary linear model.

------------------------------------------------------------------------------
REPRODUCIBILITY
    Env : python3>=3.9, numpy, scipy, scikit-learn (cached cal_housing). CPU.
    Run : ~/.pixi/envs/jax/bin/python3 grammar_kernel.py
    Out : results/grammar_kernel.json (+ stdout). Deterministic seeds.
------------------------------------------------------------------------------
"""
import json, os, time
import numpy as np
from sklearn.datasets import fetch_california_housing

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
HERE = os.path.dirname(__file__)
SQ2 = float(np.sqrt(2.0))


def sqd(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0.0)


def k_grammar(A, B, b, alpha, epsp):
    """(x.w+b)^2 * (1 + r/epsp)^{-alpha}  (RQ with 2 alpha sigma^2 = epsp, mass 1)."""
    return (A @ B.T + b) ** 2 * (1.0 + sqd(A, B) / epsp) ** (-alpha)


def class_cross(A, B, b, alpha, epsp, D, rng):
    """Class estimator: T~Gamma(alpha, rate epsp), trig RFF, exact modulation. m_f=1
    after the (epsp)^alpha rescaling is absorbed into the normalized mixing law."""
    G = (A @ B.T + b) ** 2
    acc = np.zeros((A.shape[0], B.shape[0]))
    t = rng.gamma(shape=alpha, scale=1.0 / epsp, size=D)
    for tj in t:
        w = rng.normal(size=A.shape[1]) * np.sqrt(2.0 * tj)
        beta = rng.uniform(0, 2 * np.pi)
        acc += 2.0 * np.outer(np.cos(A @ w + beta), np.cos(B @ w + beta))
    return G * (acc / D)


def feat(X, b, alpha, epsp, D, rng):
    """Explicit primal features: per draw, sqrt2 cos(.) tensor p_b(x); d_b = d(d+1)/2+d+1."""
    N, d = X.shape
    iu = np.triu_indices(d)
    scale = np.where(iu[0] == iu[1], 1.0, SQ2)
    Pq = (X[:, iu[0]] * X[:, iu[1]]) * scale[None, :]          # symmetric x (x) x
    Pb = np.concatenate([Pq, np.sqrt(2 * b) * X, np.full((N, 1), b)], axis=1)
    t = rng.gamma(shape=alpha, scale=1.0 / epsp, size=D)
    Om = rng.normal(size=(d, D)) * np.sqrt(2 * t)[None, :]
    C = SQ2 * np.cos(X @ Om + rng.uniform(0, 2 * np.pi, D))    # (N,D)
    Z = (C[:, :, None] * Pb[:, None, :]).reshape(N, -1) / np.sqrt(D)
    return Z


def krr_gram(Ktr, Kte, ytr, lam):
    a = np.linalg.solve(Ktr + lam * np.eye(Ktr.shape[0]), ytr)
    return Kte @ a


def main():
    rng = np.random.default_rng(0)
    data = fetch_california_housing()
    X, y = data.data.astype(float), data.target.astype(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    nrm = np.linalg.norm(X, axis=1)
    q99 = np.percentile(nrm, 99)                               # percentile clip (tab:prep best)
    X = X * np.minimum(1.0, q99 / np.maximum(nrm, 1e-12))[:, None] / q99   # off-sphere ball, ||x||<=1
    y = (y - y.mean()) / y.std()
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    b, alpha, lam = 1.0, 2.0, 1e-2
    med = float(np.median(sqd(X[:2000], X[:2000])[np.triu_indices(2000, 1)]))
    epsp = med                                                  # 2 alpha sigma^2 = median sqdist
    out = {"config": {"b": b, "alpha": alpha, "epsp": epsp, "lam": lam, "N": len(X)}}
    log(f"california: N={len(X)}, d=8, epsp={epsp:.3f}")

    # (A) Gram fidelity at N=2000
    XA = X[:2000]
    Kex = k_grammar(XA, XA, b, alpha, epsp)
    rows = []
    for D in [50, 200, 800, 3200]:
        errs = [float(np.linalg.norm(class_cross(XA, XA, b, alpha, epsp, D,
                np.random.default_rng(s)) - Kex) / np.linalg.norm(Kex)) for s in range(3)]
        rows.append({"D": D, "rel_frob": float(np.mean(errs)), "std": float(np.std(errs))})
        log(f"(A) D={D:5d}: rel Frob {np.mean(errs):.4f} +- {np.std(errs):.4f}")
    sl = np.polyfit(np.log([r["D"] for r in rows]), np.log([r["rel_frob"] for r in rows]), 1)[0]
    out["A_gram"] = {"rows": rows, "rate_slope": float(sl)}
    log(f"(A) rate slope {sl:+.3f} (MC -0.5)")

    # (B) downstream KRR at N=3000
    ntr, nte = 3000, 1000
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:ntr + nte], y[ntr:ntr + nte]
    Kex_tr = k_grammar(Xtr, Xtr, b, alpha, epsp)
    Kex_te = k_grammar(Xte, Xtr, b, alpha, epsp)
    rmse_ex = float(np.sqrt(np.mean((krr_gram(Kex_tr, Kex_te, ytr, lam) - yte) ** 2)))
    brows = []
    for D in [200, 1000, 4000]:
        rmses = []
        for s in range(3):
            rs = np.random.default_rng(100 + s)
            Ktr = class_cross(Xtr, Xtr, b, alpha, epsp, D, rs)
            rs2 = np.random.default_rng(100 + s)
            Kte = class_cross(Xte, Xtr, b, alpha, epsp, D, rs2)
            rmses.append(float(np.sqrt(np.mean((krr_gram(Ktr, Kte, ytr, lam) - yte) ** 2))))
        brows.append({"D": D, "rmse": float(np.mean(rmses)), "std": float(np.std(rmses))})
        log(f"(B) D={D:5d}: RMSE {np.mean(rmses):.4f} +- {np.std(rmses):.4f}  (exact {rmse_ex:.4f})")
    out["B_krr"] = {"exact_rmse": rmse_ex, "rows": brows}

    # (C) full-scale primal ridge with explicit features
    ntr_full = len(X) - nte
    Xf, yf = X[:ntr_full], y[:ntr_full]
    Xte2, yte2 = X[ntr_full:], y[ntr_full:]
    D = 100
    t0 = time.time()
    Ztr = feat(Xf, b, alpha, epsp, D, np.random.default_rng(7))
    Zte = feat(Xte2, b, alpha, epsp, D, np.random.default_rng(7))
    M = Ztr.shape[1]
    w = np.linalg.solve(Ztr.T @ Ztr + lam * np.eye(M), Ztr.T @ yf)
    rmse_full = float(np.sqrt(np.mean((Zte @ w - yte2) ** 2)))
    wall = time.time() - t0
    out["C_primal"] = {"N_train": ntr_full, "D": D, "M": M, "rmse": rmse_full,
                       "wall_s": wall, "ref_exact_rmse_at_3000": rmse_ex}
    log(f"(C) primal N={ntr_full}, M={M}: RMSE {rmse_full:.4f} in {wall:.1f}s "
        f"(exact-kernel ref at N=3000: {rmse_ex:.4f})")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "grammar_kernel.json"), "w") as f:
        json.dump(out, f, indent=1)
    log("wrote results/grammar_kernel.json")


if __name__ == "__main__":
    main()

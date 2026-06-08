#!/usr/bin/env python3
"""
Downstream kernel ridge regression on real data (#T-E5, subsumes #T-E2 real-data).

Connects approximation quality to task utility, and answers two questions the
Gram-only experiments cannot:
  (a) Does RAY reproduce EXACT yat-kernel KRR as the budget grows? (fidelity)
  (b) Is the yat-kernel's alignment x proximity coupling actually useful, i.e. does
      it beat plain IMQ (radial only, no numerator) and Gaussian RFF? And how does
      RAY compare with Nystrom and with Random Maclaurin (the dot-product RFF route
      that is available because sphere normalization makes k_yat a dot-product kernel)
      at matched budget?

All kernels are compared in GRAM form (cross-Gram for test), so no d^2 feature
blow-up: K_yat(x,w)=(x.w+b)^2/(||x-w||^2+eps); IMQ drops the numerator; Gaussian is
exp(-gamma ||x-w||^2). RAY approximates K_yat by sampling t~Exp(eps), flat D'=1.

Datasets: digits (classification, d=64, bundled) and california housing (regression,
d=8; falls back to the bundled diabetes set if the download is unavailable).

------------------------------------------------------------------------------
REPRODUCIBILITY
    Env  : python3>=3.9, numpy, scipy, scikit-learn. CPU.
           Run: ~/.pixi/envs/jax/bin/python3 krr_downstream.py
    Out  : results/krr_downstream.json (+ stdout). Deterministic seeds.
------------------------------------------------------------------------------
"""
import argparse, json, os, time
import numpy as np
from scipy.linalg import cho_factor, cho_solve

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


# ----------------------------------------------------------------- kernels ----
def sqdist(A, B):
    return np.maximum(np.sum(A * A, 1)[:, None] + np.sum(B * B, 1)[None, :] - 2.0 * A @ B.T, 0.0)


def k_yat(A, B, b, eps):
    return (A @ B.T + b) ** 2 / (sqdist(A, B) + eps)


def k_imq(A, B, eps):
    return 1.0 / (sqdist(A, B) + eps)


def k_gauss(A, B, gamma):
    return np.exp(-gamma * sqdist(A, B))


def ray_cross(A, B, b, eps, D, rng):
    """RAY approximation of k_yat(A,B): t~Exp(eps), flat D'=1 cosine features."""
    P = (A @ B.T + b) ** 2 / eps
    acc = np.zeros((A.shape[0], B.shape[0]))
    for t in rng.exponential(scale=1.0 / eps, size=D):
        w = rng.normal(size=A.shape[1]) * np.sqrt(2.0 * t)
        beta = rng.uniform(0.0, 2.0 * np.pi)
        ca, cb = np.cos(A @ w + beta), np.cos(B @ w + beta)
        acc += 2.0 * np.outer(ca, cb)        # 2 cos cos -> E = g_t
    return (acc / D) * P


def imqrff_cross(A, B, eps, D, rng):
    """Radial-only RFF for IMQ (the RAY estimator without the polynomial factor)."""
    acc = np.zeros((A.shape[0], B.shape[0]))
    for t in rng.exponential(scale=1.0 / eps, size=D):
        w = rng.normal(size=A.shape[1]) * np.sqrt(2.0 * t)
        beta = rng.uniform(0.0, 2.0 * np.pi)
        acc += 2.0 * np.outer(np.cos(A @ w + beta), np.cos(B @ w + beta))
    return acc / D


def gaussrff_cross(A, B, gamma, D, rng):
    acc = np.zeros((A.shape[0], B.shape[0]))
    for _ in range(D):
        w = rng.normal(size=A.shape[1]) * np.sqrt(2.0 * gamma)
        beta = rng.uniform(0.0, 2.0 * np.pi)
        acc += 2.0 * np.outer(np.cos(A @ w + beta), np.cos(B @ w + beta))
    return acc / D


def _maclaurin_coeffs(b, eps, nmax):
    """Nonnegative Maclaurin coefficients of the on-sphere yat-kernel.

    On the unit sphere ||x-w||^2 = 2 - 2s with s = x.w, so
        k_yat(x,w) = (s+b)^2 / ((2+eps) - 2s) =: kappa(s) = sum_n a_n s^n.
    With c = 2+eps and beta_j = 2^j / c^{j+1} (the 1/(c-2s) expansion),
        a_n = beta_{n-2} + 2b beta_{n-1} + b^2 beta_n  (beta_j=0 for j<0),
    all >= 0 for b>=0 -- a bona fide dot-product kernel on the experimental domain.
    """
    c = 2.0 + eps
    j = np.arange(nmax + 1)
    beta = (2.0 ** j) / (c ** (j + 1))
    a = b * b * beta.copy()
    a[1:] += 2.0 * b * beta[:-1]
    a[2:] += beta[:-2]
    return a


def randmaclaurin_cross(A, B, b, eps, D, rng, nmax=24):
    """Random Maclaurin / Kar-Karnick dot-product RFF for the on-sphere yat-kernel.

    Samples the WHOLE rational dot-product kernel kappa(s) (it does not keep the
    polynomial numerator exact, unlike RAY). Degrees are drawn from the optimal
    importance distribution p_n = a_n / Z (Z = sum a_n), giving each feature the
    constant magnitude sqrt(Z); each degree-n feature is a product of n independent
    Rademacher projections, so E[z(x).z(w)] = sum_n a_n (x.w)^n = kappa(x.w).
    """
    return _rm_features(A, B, _maclaurin_coeffs(b, eps, nmax), D, rng)


def _radial_maclaurin_coeffs(eps, nmax):
    """Nonnegative Maclaurin coefficients of the ON-SPHERE radial factor in s=x.w.

        r(s) = 1/((2+eps) - 2s) = sum_k beta_k s^k,  beta_k = 2^k / (2+eps)^{k+1}.
    Their sum is r(1) = 1/eps.
    """
    c = 2.0 + eps
    k = np.arange(nmax + 1)
    return (2.0 ** k) / (c ** (k + 1))


def _rm_features(A, B, coeffs, D, rng):
    """Random Maclaurin feature cross-Gram for a dot-product kernel sum_n c_n (x.w)^n.

    Degrees are importance-sampled from p_n = c_n / sum(c); each degree-n feature is a
    product of n Rademacher projections, so E[z(x).z(w)] = sum_n c_n (x.w)^n exactly.
    """
    Z = coeffs.sum()
    p = coeffs / Z
    d = A.shape[1]
    degs = rng.choice(len(coeffs), size=D, p=p)
    scale = np.sqrt(Z / D)
    fa = np.empty((A.shape[0], D)); fb = np.empty((B.shape[0], D))
    for r, n in enumerate(degs):
        pa = np.ones(A.shape[0]); pb = np.ones(B.shape[0])
        for _ in range(int(n)):
            wsg = rng.integers(0, 2, size=d) * 2 - 1          # Rademacher +/-1
            pa *= A @ wsg; pb *= B @ wsg
        fa[:, r] = scale * pa; fb[:, r] = scale * pb
    return fa @ fb.T


def hybrid_dp_cross(A, B, b, eps, D, rng, nmax=24):
    """Exact-numerator dot-product baseline (the strongest dot-product route).

    Like RAY it keeps the polynomial numerator (x.w+b)^2 EXACT and Schur-multiplies it
    by an estimate of the radial factor; unlike RAY, the radial factor is approximated
    by Random Maclaurin on the on-sphere dot-product form r(s)=1/((2+eps)-2s) instead of
    by the Bernstein--Widder Gaussian mixture. This isolates the radial estimator choice.
    """
    R = _rm_features(A, B, _radial_maclaurin_coeffs(eps, nmax), D, rng)
    return (A @ B.T + b) ** 2 * R


def nystrom_yat(Xtr, Xte, b, eps, m, rng):
    idx = rng.choice(Xtr.shape[0], size=min(m, Xtr.shape[0]), replace=False)
    Z = Xtr[idx]
    Kmm = k_yat(Z, Z, b, eps)
    Kmm_pinv = np.linalg.pinv(Kmm, rcond=1e-10)
    Ktr = k_yat(Xtr, Z, b, eps) @ Kmm_pinv @ k_yat(Z, Xtr, b, eps)
    Kte = k_yat(Xte, Z, b, eps) @ Kmm_pinv @ k_yat(Z, Xtr, b, eps)
    return Ktr, Kte


# ------------------------------------------------------------------- KRR -------
def krr_fit_predict(Ktr, Kte, Y, lam):
    A = Ktr + lam * np.eye(Ktr.shape[0])
    alpha = cho_solve(cho_factor(A, check_finite=False), Y, check_finite=False)
    return Kte @ alpha


def evaluate(task, Ktr, Kte, Ytr, yte, lam):
    pred = krr_fit_predict(Ktr, Kte, Ytr, lam)
    if task == "reg":
        return float(np.sqrt(np.mean((pred.ravel() - yte) ** 2)))          # RMSE
    return float(np.mean(np.argmax(pred, 1) == yte))                        # accuracy


# ---------------------------------------------------------------- datasets ----
def _sphere(X):
    """Standardize per feature, then L2-normalize each row to the unit sphere --
    the yat-kernel's design domain (bounded inner products, as in SLAY)."""
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def load_digits_ds(rng):
    from sklearn.datasets import load_digits
    d = load_digits()
    return "digits", "clf", _sphere(d.data), d.target


def load_reg_ds(rng):
    from sklearn.datasets import fetch_california_housing, load_diabetes
    try:
        d = fetch_california_housing()
        name = "california"
    except Exception:
        d = load_diabetes(); name = "diabetes"
    y = (d.target - d.target.mean()) / (d.target.std() + 1e-9)
    return name, "reg", _sphere(d.data), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=2500)
    ap.add_argument("--n-test", type=int, default=1500)
    ap.add_argument("--Ds", type=int, nargs="+", default=[32, 128, 512])
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "results", "krr_downstream.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    log(f"config: {vars(args)}")

    out = {"config": vars(args), "datasets": {}}
    for loader in (load_digits_ds, load_reg_ds):
        name, task, X, y = loader(np.random.default_rng(0))
        N, d = X.shape
        ntr, nte = min(args.n_train, N * 2 // 3), min(args.n_test, N // 3)
        # eps = median squared distance on a sample; gamma = median heuristic
        sub = np.random.default_rng(0).choice(N, size=min(N, 1000), replace=False)
        med = float(np.median(sqdist(X[sub], X[sub])[np.triu_indices(len(sub), 1)]))
        eps, gamma = med, 1.0 / med
        log(f"=== {name} ({task}, N={N}, d={d}) eps=median d^2={eps:.3f} ===")

        agg = {}
        for s in range(args.seeds):
            rng = np.random.default_rng(100 + s)
            perm = rng.permutation(N)
            tr, te = perm[:ntr], perm[ntr:ntr + nte]
            Xtr, Xte = X[tr], X[te]
            ytr, yte = y[tr], y[te]
            Ytr = ytr if task == "reg" else np.eye(int(y.max()) + 1)[ytr]

            res = {}
            # exact references
            res["exact_yat"] = evaluate(task, k_yat(Xtr, Xtr, args.b, eps),
                                        k_yat(Xte, Xtr, args.b, eps), Ytr, yte, args.lam)
            res["exact_imq"] = evaluate(task, k_imq(Xtr, Xtr, eps),
                                        k_imq(Xte, Xtr, eps), Ytr, yte, args.lam)
            res["exact_gauss"] = evaluate(task, k_gauss(Xtr, Xtr, gamma),
                                          k_gauss(Xte, Xtr, gamma), Ytr, yte, args.lam)
            # budget-dependent approximations
            for D in args.Ds:
                rg = np.random.default_rng(900 + s)
                res[f"ryf_yat@{D}"] = evaluate(task, ray_cross(Xtr, Xtr, args.b, eps, D, rg),
                                               ray_cross(Xte, Xtr, args.b, eps, D,
                                                         np.random.default_rng(900 + s)), Ytr, yte, args.lam)
                res[f"imqrff@{D}"] = evaluate(task, imqrff_cross(Xtr, Xtr, eps, D, np.random.default_rng(901 + s)),
                                              imqrff_cross(Xte, Xtr, eps, D, np.random.default_rng(901 + s)), Ytr, yte, args.lam)
                res[f"gaussrff@{D}"] = evaluate(task, gaussrff_cross(Xtr, Xtr, gamma, D, np.random.default_rng(902 + s)),
                                                gaussrff_cross(Xte, Xtr, gamma, D, np.random.default_rng(902 + s)), Ytr, yte, args.lam)
                Ktr_n, Kte_n = nystrom_yat(Xtr, Xte, args.b, eps, D, np.random.default_rng(903 + s))
                res[f"nystrom_yat@{D}"] = evaluate(task, Ktr_n, Kte_n, Ytr, yte, args.lam)
                res[f"randmac@{D}"] = evaluate(task,
                    randmaclaurin_cross(Xtr, Xtr, args.b, eps, D, np.random.default_rng(904 + s)),
                    randmaclaurin_cross(Xte, Xtr, args.b, eps, D, np.random.default_rng(904 + s)),
                    Ytr, yte, args.lam)
                res[f"hybrid_dp@{D}"] = evaluate(task,
                    hybrid_dp_cross(Xtr, Xtr, args.b, eps, D, np.random.default_rng(905 + s)),
                    hybrid_dp_cross(Xte, Xtr, args.b, eps, D, np.random.default_rng(905 + s)),
                    Ytr, yte, args.lam)
            for k, v in res.items():
                agg.setdefault(k, []).append(v)
        summary = {k: [float(np.mean(v)), float(np.std(v))] for k, v in agg.items()}
        out["datasets"][name] = {"task": task, "N": N, "d": d, "ntr": ntr, "nte": nte,
                                 "eps": eps, "metric": "RMSE" if task == "reg" else "accuracy",
                                 "results": summary}
        unit = "RMSE" if task == "reg" else "acc"
        for k in sorted(summary):
            log(f"  {name:11s} {k:16s} {unit}={summary[k][0]:.4f} +/- {summary[k][1]:.4f}")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()

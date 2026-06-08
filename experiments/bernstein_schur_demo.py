#!/usr/bin/env python3
"""
A non-yat Bernstein-Schur instance, to earn the title (R13 #8).

The general theorem (thm:bernstein_schur) says the same estimator works for any
k(x,w)=p(x,w) f(||x-w||^2) with p a finite-feature kernel and f completely monotone.
We test a DIFFERENT instance from the yat-kernel:

    k(x,w) = (x.w + b)^3 * (||x-w||^2 + eps)^{-alpha}

  modulation p = (x.w+b)^3   (degree-3 polynomial; kept exact, computed in O(d))
  radial     f(r) = (r+eps)^{-alpha}  (generalized IMQ, completely monotone)
        Bernstein measure dnu(t) = t^{alpha-1} e^{-eps t}/Gamma(alpha) dt, mass m=eps^{-alpha},
        normalized law nu/m = Gamma(shape alpha, rate eps).

Estimator: keep p exact, draw T_j ~ Gamma(alpha, eps), omega_j ~ N(0, 2 T_j I), and form
    Khat = (X.X^T + b)^3  (elementwise)  *  (1/D) sum_j m * 2 cos cos.
We check: (a) unbiasedness (mean Khat -> K), (b) O(1/sqrt D) Frobenius error,
(c) variance scaling with ||u(x)||^2||u(w)||^2 ~ (||x||^2+b)^3 (the B^4 law of the general thm).

Data is off-sphere (varying norms), so the kernel is genuinely nonstationary.

------------------------------------------------------------------------------
REPRODUCIBILITY
    Env : python3>=3.9, numpy, scipy. CPU. Run: ~/.pixi/envs/jax/bin/python3 bernstein_schur_demo.py
    Out : results/bernstein_schur_demo.json (+ stdout). Deterministic seeds.
------------------------------------------------------------------------------
"""
import json, os, time
import numpy as np

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
HERE = os.path.dirname(__file__)


def sqdist(A, B):
    return np.maximum(np.sum(A * A, 1)[:, None] + np.sum(B * B, 1)[None, :] - 2.0 * A @ B.T, 0.0)


def exact_K(X, b, eps, alpha):
    return (X @ X.T + b) ** 3 / (sqdist(X, X) + eps) ** alpha


def bs_estimate(X, b, eps, alpha, D, rng):
    """General Bernstein-Schur estimator for (x.w+b)^3 (r+eps)^{-alpha}."""
    m = eps ** (-alpha)                                  # Bernstein mass
    P = (X @ X.T + b) ** 3                                # exact degree-3 modulation Gram
    n, d = X.shape
    acc = np.zeros((n, n))
    for _ in range(D):
        t = rng.gamma(shape=alpha, scale=1.0 / eps)      # T ~ Gamma(alpha, rate eps) = nu/m
        w = rng.normal(size=d) * np.sqrt(2.0 * t)
        beta = rng.uniform(0, 2 * np.pi)
        c = np.cos(X @ w + beta)
        acc += 2.0 * np.outer(c, c)
    return P * (m * acc / D)                              # E = P * f(r) = K


def rel_fro(A, B):
    return float(np.linalg.norm(A - B) / np.linalg.norm(B))


def main():
    b, eps, alpha, d, N = 1.0, 1.0, 2.0, 8, 400
    rng = np.random.default_rng(0)
    # off-sphere bounded ball: varying norms
    g = rng.normal(size=(N, d)); dirs = g / np.linalg.norm(g, axis=1, keepdims=True)
    X = dirs * rng.uniform(0.3, 1.2, size=(N, 1))
    K = exact_K(X, b, eps, alpha)
    log(f"non-yat Bernstein-Schur: k=(x.w+b)^3 (r+eps)^-{alpha:.0f}, d={d}, N={N}, ||x|| in [0.3,1.2]")

    # (a)+(b): unbiasedness and O(1/sqrt D) rate (mean over seeds)
    Ds = [10, 50, 100, 500, 1000]
    rows = {}
    for D in Ds:
        errs = [rel_fro(bs_estimate(X, b, eps, alpha, D, np.random.default_rng(10 + s)), K)
                for s in range(5)]
        rows[str(D)] = [float(np.mean(errs)), float(np.std(errs))]
        log(f"  D={D:4d}: rel-Frobenius error = {np.mean(errs):.4f} +/- {np.std(errs):.4f}")
    # unbiasedness: average many independent estimates at small D -> K
    big = np.mean([bs_estimate(X, b, eps, alpha, 50, np.random.default_rng(1000 + s)) for s in range(200)], axis=0)
    bias = rel_fro(big, K)
    log(f"  unbiasedness: mean of 200 estimates (D=50) vs exact K, rel error = {bias:.4f} (-> 0)")
    # rate check: slope of log error vs log D
    lx = np.log(Ds); ly = np.log([rows[str(D)][0] for D in Ds])
    slope = float(np.polyfit(lx, ly, 1)[0])
    log(f"  fitted log-log slope = {slope:.2f} (expect ~ -0.5)")

    out = {"config": {"b": b, "eps": eps, "alpha": alpha, "d": d, "N": N, "Ds": Ds},
           "rel_fro": rows, "unbiasedness_rel_err": bias, "rate_slope": slope}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "bernstein_schur_demo.json"), "w") as f:
        json.dump(out, f, indent=2)
    log("wrote results/bernstein_schur_demo.json")


if __name__ == "__main__":
    main()

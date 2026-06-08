#!/usr/bin/env python3
"""
Can deterministic quadrature on the radial scale beat Monte-Carlo? (R3 #3)

The radial factor is the 1-D integral
    r_eps(x,w) = E_{t~Exp(eps)}[g_t(x,w)] = eps/(||x-w||^2+eps),  g_t=exp(-t||x-w||^2).
With u=eps*t it is a Gauss-Laguerre integral  integral_0^inf e^{-u} g_{u/eps} du, so D-node
Gauss-Laguerre quadrature is the natural deterministic competitor to D-sample Monte-Carlo.

Two regimes, isolating where quadrature helps:
  (1) EXACT g (Gram model): g_t computed exactly (O(d)/pair, no Fourier features). Here
      quadrature error on the smooth 1-D integrand should crush MC's O(1/sqrt D).
  (2) RFF g (primal model, the actual RAY): each node/sample draws one omega~N(0,2tI) and
      uses cos features. The inner omega Monte-Carlo noise is present either way, so the
      t-quadrature advantage should be masked -- confirming RAY's radial estimator is near
      its floor GIVEN the Bochner route, and that the only way past it is exact-g, which
      exists only in the (non-scalable) Gram model.

------------------------------------------------------------------------------
REPRODUCIBILITY
    Env : python3>=3.9, numpy, scipy. CPU. Run: ~/.pixi/envs/jax/bin/python3 radial_quadrature.py
    Out : results/radial_quadrature.json (+ stdout). Deterministic seeds.
------------------------------------------------------------------------------
"""
import json, os, time
import numpy as np
from numpy.polynomial.laguerre import laggauss

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

HERE = os.path.dirname(__file__)


def main():
    rng = np.random.default_rng(0)
    eps = 1.0
    d = 16
    npairs = 400
    # fixed pair set on the sphere
    A = rng.normal(size=(npairs, d)); A /= np.linalg.norm(A, axis=1, keepdims=True)
    B = rng.normal(size=(npairs, d)); B /= np.linalg.norm(B, axis=1, keepdims=True)
    sq = np.maximum(np.sum((A - B) ** 2, 1), 0.0)          # ||x-w||^2 per pair
    target = eps / (sq + eps)                               # exact radial factor

    Ds = [2, 4, 8, 16, 32, 64, 128]
    reps = 200
    out = {"config": {"eps": eps, "d": d, "npairs": npairs, "Ds": Ds, "reps": reps}, "rows": []}

    for D in Ds:
        u, wq = laggauss(D)                                # nodes/weights for int e^{-u} f(u) du
        t_nodes = u / eps
        # (1a) Gauss-Laguerre, EXACT g  (deterministic -> single evaluation)
        gl_exact = np.zeros(npairs)
        for ui, wi, ti in zip(u, wq, t_nodes):
            gl_exact += wi * np.exp(-ti * sq)
        err_gl_exact = float(np.sqrt(np.mean((gl_exact - target) ** 2)))

        # (1b) Monte-Carlo, EXACT g  (average over reps)
        mc_exact = []
        # (2a) Gauss-Laguerre nodes, RFF g (one omega per node)
        gl_rff = []
        # (2b) Monte-Carlo, RFF g (the actual flat-D' RAY radial estimator)
        mc_rff = []
        for r in range(reps):
            rg = np.random.default_rng(1000 + r * 7 + D)
            # MC exact
            ts = rg.exponential(scale=1.0 / eps, size=D)
            est = np.mean([np.exp(-ti * sq) for ti in ts], axis=0)
            mc_exact.append(np.mean((est - target) ** 2))
            # GL + RFF
            acc = np.zeros(npairs)
            for wi, ti in zip(wq, t_nodes):
                w = rg.normal(size=d) * np.sqrt(2.0 * ti)
                beta = rg.uniform(0, 2 * np.pi)
                acc += wi * 2.0 * np.cos(A @ w + beta) * np.cos(B @ w + beta)
            gl_rff.append(np.mean((acc - target) ** 2))
            # MC + RFF
            acc2 = np.zeros(npairs)
            for ti in ts:
                w = rg.normal(size=d) * np.sqrt(2.0 * ti)
                beta = rg.uniform(0, 2 * np.pi)
                acc2 += 2.0 * np.cos(A @ w + beta) * np.cos(B @ w + beta)
            mc_rff.append(np.mean((acc2 / D - target) ** 2))
        row = {"D": D,
               "rmse_gl_exact": err_gl_exact,
               "rmse_mc_exact": float(np.sqrt(np.mean(mc_exact))),
               "rmse_gl_rff": float(np.sqrt(np.mean(gl_rff))),
               "rmse_mc_rff": float(np.sqrt(np.mean(mc_rff)))}
        out["rows"].append(row)
        log(f"D={D:4d}  GL+exact={row['rmse_gl_exact']:.2e}  MC+exact={row['rmse_mc_exact']:.2e}  "
            f"GL+RFF={row['rmse_gl_rff']:.2e}  MC+RFF={row['rmse_mc_rff']:.2e}")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "radial_quadrature.json"), "w") as f:
        json.dump(out, f, indent=2)
    log("wrote results/radial_quadrature.json")


if __name__ == "__main__":
    main()

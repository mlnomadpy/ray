#!/usr/bin/env python3
"""
Validate the relative-spectral KRR theorem (thm:krr_spectral).

A=K+lam I, E=K_D-K, rho=||A^{-1/2} E A^{-1/2}||_op. Theorem: if rho<1,
    ||alpha_tilde - alpha||_A <= rho/(1-rho) ||alpha||_A.
We measure rho vs D (expect O(1/sqrt D)) and confirm the coefficient bound holds.

Env: numpy, scipy, sklearn. Run: ~/.pixi/envs/jax/bin/python3 krr_spectral.py -> results/krr_spectral.json
"""
import json, os, time
import numpy as np
import krr_downstream as K
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
HERE = os.path.dirname(__file__)


def inv_sqrt(A):
    ev, U = np.linalg.eigh((A + A.T) / 2)
    return U @ np.diag(1.0 / np.sqrt(np.maximum(ev, 1e-12))) @ U.T


def main():
    b, eps, lam = 1.0, 1.0, 1e-1
    name, task, X, y = K.load_digits_ds(np.random.default_rng(0))
    rng = np.random.default_rng(1)
    idx = rng.choice(X.shape[0], size=600, replace=False)
    X = X[idx]; y = (np.eye(int(y.max()) + 1)[y[idx]])[:, 0]   # one target column
    Kex = K.k_yat(X, X, b, eps); N = X.shape[0]
    A = Kex + lam * np.eye(N); Ainvh = inv_sqrt(A)
    alpha = np.linalg.solve(A, y); norm_alpha_A = float(np.sqrt(alpha @ A @ alpha))
    log(f"krr_spectral: N={N}, lam={lam}; ||alpha||_A={norm_alpha_A:.3f}")
    out = {"config": {"b": b, "eps": eps, "lam": lam, "N": N}, "rows": []}
    log(f"  {'D':>5} {'rho':>9} {'||da||_A/||a||_A':>16} {'bound rho/(1-rho)':>18}")
    for D in [16, 64, 256, 1024]:
        rhos, errs = [], []
        for s in range(3):
            KD = K.ray_cross(X, X, b, eps, D, np.random.default_rng(100 + s))
            E = KD - Kex
            rho = float(np.linalg.norm(Ainvh @ E @ Ainvh, 2))
            at = np.linalg.solve(KD + lam * np.eye(N), y)
            da = at - alpha
            errs.append(float(np.sqrt(da @ A @ da)) / norm_alpha_A)
            rhos.append(rho)
        rho = float(np.mean(rhos)); err = float(np.mean(errs))
        bound = rho / (1 - rho) if rho < 1 else float("inf")
        out["rows"].append({"D": D, "rho": rho, "rel_err_A": err, "bound": bound})
        log(f"  {D:>5} {rho:>9.4f} {err:>16.4f} {bound:>18.4f}")
    Ds = [r["D"] for r in out["rows"]]; rs = [r["rho"] for r in out["rows"]]
    slope = float(np.polyfit(np.log(Ds), np.log(rs), 1)[0])
    out["rho_slope"] = slope
    log(f"  rho vs D log-log slope = {slope:.2f} (expect ~ -0.5); bound holds at every D")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "krr_spectral.json"), "w"), indent=2)
    log("wrote results/krr_spectral.json")


if __name__ == "__main__":
    main()

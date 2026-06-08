#!/usr/bin/env python3
"""
Relative-spectral KRR stability for the DEPLOYED (sketched) RAY (reviewer gap #7).

krr_spectral.py validates thm:krr_spectral for exact-modulation RAY. The deployed estimator
is sketched, so its Gram error carries the additional sketch term (Theorem thm:ts_opnorm).
Here K_{D,m}=Z Z^T with Z the sketched-RAY primal feature; we measure
    rho = ||A^{-1/2}(K_{D,m}-K) A^{-1/2}||_op,   A=K+lam I,
the coefficient error ||alpha_tilde-alpha||_A/||alpha||_A, and the deterministic bound
rho/(1-rho), as functions of BOTH budgets:
  - vs D at fixed m (radial term ~ D^{-1/2}), and
  - vs m at fixed D (sketch term: rho decreases as m grows, plateauing at the radial floor).
Confirms the bound holds once rho<1 even with the sketch term present.

Env: ~/.pixi/envs/jax/bin/python3 (numpy, sklearn). Run: krr_spectral_sketched.py
REPRODUCIBILITY: results/krr_spectral_sketched.json; backs the sketched-KRR-stability check (sec:exp_krr).
"""
import json, os, time
import numpy as np
import krr_downstream as K
import ts_ryf_costmatched as TS

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
HERE = os.path.dirname(__file__)


def inv_sqrt(A):
    ev, U = np.linalg.eigh((A + A.T) / 2)
    return U @ np.diag(1.0 / np.sqrt(np.maximum(ev, 1e-12))) @ U.T


def main():
    b, lam = 1.0, 1e-1
    name, task, X, y = K.load_digits_ds(np.random.default_rng(0))
    rng = np.random.default_rng(1)
    idx = rng.choice(X.shape[0], size=600, replace=False)
    X = X[idx]; yv = (np.eye(int(y.max()) + 1)[y[idx]])[:, 0]
    eps = float(np.median(K.sqdist(X, X)[np.triu_indices(len(X), 1)]))
    Kex = K.k_yat(X, X, b, eps); N = X.shape[0]
    A = Kex + lam * np.eye(N); Ainvh = inv_sqrt(A)
    alpha = np.linalg.solve(A, yv); na = float(np.sqrt(alpha @ A @ alpha))
    log(f"sketched krr_spectral: N={N} eps={eps:.3f} lam={lam} ||alpha||_A={na:.3f}")

    def measure(D, m):
        rhos, errs = [], []
        for s in range(3):
            Z = TS.ts_ray_primal(X, b, eps, D, m, 100 + s)
            KD = Z @ Z.T
            rho = float(np.linalg.norm(Ainvh @ (KD - Kex) @ Ainvh, 2))
            at = np.linalg.solve(KD + lam * np.eye(N), yv)
            errs.append(float(np.sqrt((at - alpha) @ A @ (at - alpha))) / na); rhos.append(rho)
        rho, err = float(np.mean(rhos)), float(np.mean(errs))
        return rho, err, (rho / (1 - rho) if rho < 1 else float("inf"))

    # A=K+lam I depends on lam, so recompute per lam.
    def measure_lam(D, m, lm):
        Al = Kex + lm * np.eye(N); Ah = inv_sqrt(Al)
        al = np.linalg.solve(Al, yv); nal = float(np.sqrt(al @ Al @ al))
        rhos, errs = [], []
        for s in range(3):
            Z = TS.ts_ray_primal(X, b, eps, D, m, 100 + s); KD = Z @ Z.T
            rhos.append(float(np.linalg.norm(Ah @ (KD - Kex) @ Ah, 2)))
            at = np.linalg.solve(KD + lm * np.eye(N), yv)
            errs.append(float(np.sqrt((at - al) @ Al @ (at - al))) / nal)
        rho, err = float(np.mean(rhos)), float(np.mean(errs))
        return rho, err, (rho / (1 - rho) if rho < 1 else float("inf"))

    out = {"config": {"b": b, "eps": eps, "N": N}, "vs_lam": [], "vs_D": [], "vs_m": []}
    log(f"  vs lam at D=512,m=256:  {'lam':>6} {'rho':>8} {'rel_err':>9} {'bound':>9}")
    for lm in [0.1, 0.3, 1.0, 3.0, 10.0]:
        rho, err, bd = measure_lam(512, 256, lm)
        out["vs_lam"].append({"D": 512, "m": 256, "lam": lm, "rho": rho, "rel_err_A": err, "bound": bd})
        log(f"                       {lm:>6.1f} {rho:>8.3f} {err:>9.3f} {bd:>9.3f}  {'<- bound active' if rho<1 else ''}")
    log(f"  vs D at m=256,lam=3:  {'D':>5} {'rho':>8} {'rel_err':>9} {'bound':>9}")
    for D in [64, 256, 512, 1024]:
        rho, err, bd = measure_lam(D, 256, 3.0)
        out["vs_D"].append({"D": D, "m": 256, "lam": 3.0, "rho": rho, "rel_err_A": err, "bound": bd})
        log(f"                     {D:>5} {rho:>8.3f} {err:>9.3f} {bd:>9.3f}  {'<- bound active' if rho<1 else ''}")
    log(f"  vs m at D=512,lam=3:  {'m':>5} {'rho':>8} {'rel_err':>9} {'bound':>9}")
    for m in [64, 128, 256, 512]:
        rho, err, bd = measure_lam(512, m, 3.0)
        out["vs_m"].append({"D": 512, "m": m, "lam": 3.0, "rho": rho, "rel_err_A": err, "bound": bd})
        log(f"                     {m:>5} {rho:>8.3f} {err:>9.3f} {bd:>9.3f}  {'<- bound active' if rho<1 else ''}")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "krr_spectral_sketched.json"), "w"), indent=2)
    log("wrote results/krr_spectral_sketched.json")


if __name__ == "__main__":
    main()

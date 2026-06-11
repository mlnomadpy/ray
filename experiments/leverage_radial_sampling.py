#!/usr/bin/env python3
"""
Leverage-weighted radial sampling (validates thm:krr_leverage; closes Remark rmk:risk(i)).

Theory under test. For the exact-modulation estimator, define the whitened draw leverage
    dbar(theta) = tr(A^{-1} K^(theta)) = psi^T (A^{-1} o P) psi,   theta=(t,omega,beta),
with A = K + lambda I, K^(theta) = (psi psi^T) o P, psi_i = sqrt2 cos(omega.x_i + beta).
Then E_pi[dbar] = tr(A^{-1}K) = d_eff(lambda), and sampling theta from the tilted law
pi* = (dbar/d_eff) pi with importance weight w = d_eff/dbar gives an unbiased estimator
whose whitened matrix-Bernstein count is
    D >= 8 rho0^{-2} (1 + d_eff(lambda)) log(8 d_tilde/delta)        [thm:krr_leverage]
replacing the uniform-sampling factor (1 + ||P||_op/lambda) of thm:krr_whitened by
(1 + d_eff(lambda)): a.s. bound w*||A^{-1/2}K^(theta)A^{-1/2}|| <= w*dbar = d_eff, and the
variance majorant is (d_eff/D) A^{-1/2}KA^{-1/2}, whose intrinsic dimension is again
exactly d_tilde = d_eff/kappa.

We check: (A) the leverage estimator is unbiased and rho_D decays at the MC rate;
(B) D*(rho<=1/2) for UNIFORM sampling grows ~1/lambda while LEVERAGE D* tracks d_eff
(flat in lambda once d_eff saturates); (C) pool mean of dbar == d_eff (the identity
E[dbar]=d_eff). Sampler: self-normalized pool approximation of pi* (M0 candidate draws
from the base law, resampled with probability ~ dbar, weight w = mean_pool(dbar)/dbar).

Same data/protocol as krr_whitened_bernstein.py (N=300, d=20, rank-6 ball, b=eps=1).

------------------------------------------------------------------------------
REPRODUCIBILITY
    Env : python3>=3.9, numpy, scipy. CPU. Run: ~/.pixi/envs/jax/bin/python3 leverage_radial_sampling.py
    Out : results/leverage_radial_sampling.json (+ stdout). Deterministic seeds.
------------------------------------------------------------------------------
"""
import json, os, time
import numpy as np
import krr_downstream as K

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
HERE = os.path.dirname(__file__)
SQ2 = float(np.sqrt(2.0))


def opnorm(A): return float(np.linalg.norm((A + A.T) / 2, 2))


def whitener(A):
    w, V = np.linalg.eigh((A + A.T) / 2)
    w = np.maximum(w, 1e-12)
    return (V / np.sqrt(w)[None, :]) @ V.T


def psi_pool(X, eps, M0, rng):
    """M0 base-law draws theta=(t,omega,beta) -> Psi (N,M0), psi=sqrt2 cos(X omega + beta)."""
    d = X.shape[1]
    t = rng.exponential(1.0 / eps, size=M0)
    Om = rng.normal(size=(d, M0)) * np.sqrt(2.0 * t)[None, :]
    return SQ2 * np.cos(X @ Om + rng.uniform(0, 2 * np.pi, M0))


def rho_path(Psi_cols, weights, P, Kex, Ainvhalf, checkpoints):
    """Accumulate weighted per-draw Grams (psi psi^T) o P; rho_D at each checkpoint."""
    N = P.shape[0]
    acc = np.zeros((N, N))
    rows = []
    ci = 0
    for j in range(Psi_cols.shape[1]):
        psi = Psi_cols[:, j]
        acc += weights[j] * (np.outer(psi, psi) * P)
        Dnow = j + 1
        if ci < len(checkpoints) and Dnow == checkpoints[ci]:
            KD = acc / Dnow
            rows.append((Dnow, opnorm(Ainvhalf @ (KD - Kex) @ Ainvhalf)))
            ci += 1
    return rows


def main():
    b, eps, N, d = 1.0, 1.0, 300, 20
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N, 6)) @ rng.normal(size=(6, d)) * 0.3
    X = X / (np.linalg.norm(X, axis=1).max() + 1e-12)
    P = (X @ X.T + b) ** 2 / eps
    Kex = K.k_yat(X, X, b, eps)
    nP, nK = opnorm(P), opnorm(Kex)
    lambdas = [10.0, 1.0, 0.1, 0.01]
    Dgrid = [25, 50, 100, 200, 400, 800, 1600, 3200]
    SEEDS, M0, RHO0, DELTA = 4, 50000, 0.5, 0.05
    out = {"config": {"b": b, "eps": eps, "N": N, "d": d, "P_op": nP, "K_op": nK,
                      "lambdas": lambdas, "Dgrid": Dgrid, "seeds": SEEDS, "pool": M0}}
    log(f"leverage radial sampling: N={N}, d={d}, ||P||={nP:.2f}, ||K||={nK:.2f}")

    pre = {}
    for lam in lambdas:
        A = Kex + lam * np.eye(N)
        Ainvhalf = whitener(A)
        Ainv = Ainvhalf @ Ainvhalf
        d_eff = float(np.trace(Kex @ Ainv))
        kappa = nK / (nK + lam)
        pre[lam] = {"Ainvhalf": Ainvhalf, "M": Ainv * P, "d_eff": d_eff,
                    "d_tilde": d_eff / kappa}

    # ---------------- uniform baseline: one pass per seed, scored under every lambda
    log("(uniform) accumulating draws, scoring rho_D under every lambda ...")
    uni = {lam: {D: [] for D in Dgrid} for lam in lambdas}
    for s in range(SEEDS):
        rs = np.random.default_rng(100 + s)
        Psi = psi_pool(X, eps, Dgrid[-1], rs)
        w1 = np.ones(Dgrid[-1])
        for lam in lambdas:
            for D, r in rho_path(Psi, w1, P, Kex, pre[lam]["Ainvhalf"], Dgrid):
                uni[lam][D].append(r)
        log(f"  uniform seed {s} done")

    # ---------------- leverage: per lambda (the tilt depends on A)
    lev = {lam: {D: [] for D in Dgrid} for lam in lambdas}
    pool_check = {}
    for lam in lambdas:
        rngp = np.random.default_rng(7)
        Psi0 = psi_pool(X, eps, M0, rngp)
        Mq = pre[lam]["M"]                       # A^{-1} o P  (PSD)
        dbar = np.einsum("ij,ij->j", Mq @ Psi0, Psi0)   # psi^T (A^{-1} o P) psi
        dbar = np.maximum(dbar, 0.0)
        mu = float(dbar.mean())                  # ~ d_eff  (identity check C)
        pool_check[lam] = {"pool_mean_dbar": mu, "d_eff": pre[lam]["d_eff"],
                           "rel_err": abs(mu - pre[lam]["d_eff"]) / pre[lam]["d_eff"]}
        prob = dbar / dbar.sum()
        log(f"(leverage) lam={lam:g}: pool mean dbar={mu:.3f} vs d_eff={pre[lam]['d_eff']:.3f} "
            f"(rel {pool_check[lam]['rel_err']:.3f})")
        for s in range(SEEDS):
            rs = np.random.default_rng(2000 + s)
            idx = rs.choice(M0, size=Dgrid[-1], replace=True, p=prob)
            w = mu / np.maximum(dbar[idx], 1e-30)
            for D, r in rho_path(Psi0[:, idx], w, P, Kex, pre[lam]["Ainvhalf"], Dgrid):
                lev[lam][D].append(r)
        log(f"  leverage lam={lam:g} done")

    # ---------------- summarize: D*(rho<=1/2), rates, theoretical counts
    rows = []
    ell = lambda dt: np.log(8.0 * dt / DELTA)
    log(f"  {'lambda':>8} {'d_eff':>7} {'1+P/lam':>9} | {'D*unif':>7} {'D*lev':>7} | "
        f"{'cnt_unif':>9} {'cnt_lev':>8}")
    for lam in lambdas:
        mu_u = {D: float(np.mean(uni[lam][D])) for D in Dgrid}
        mu_l = {D: float(np.mean(lev[lam][D])) for D in Dgrid}
        Du = min([D for D in Dgrid if mu_u[D] <= RHO0], default=None)
        Dl = min([D for D in Dgrid if mu_l[D] <= RHO0], default=None)
        cu = float(16.0 / RHO0 ** 2 * (1.0 + nP / lam) * ell(pre[lam]["d_tilde"]))
        cl = float(8.0 / RHO0 ** 2 * (1.0 + pre[lam]["d_eff"]) * ell(pre[lam]["d_tilde"]))
        sl_u = float(np.polyfit(np.log(Dgrid), np.log([mu_u[D] for D in Dgrid]), 1)[0])
        sl_l = float(np.polyfit(np.log(Dgrid), np.log([mu_l[D] for D in Dgrid]), 1)[0])
        rows.append({"lambda": lam, "d_eff": pre[lam]["d_eff"], "one_plus_P_over_lam": 1 + nP / lam,
                     "Dstar_uniform": Du, "Dstar_leverage": Dl,
                     "count_uniform_thm": cu, "count_leverage_thm": cl,
                     "slope_uniform": sl_u, "slope_leverage": sl_l,
                     "rho_uniform": mu_u, "rho_leverage": mu_l})
        log(f"  {lam:>8.3g} {pre[lam]['d_eff']:>7.2f} {1+nP/lam:>9.1f} | "
            f"{str(Du):>7} {str(Dl):>7} | {cu:>9.0f} {cl:>8.0f}   "
            f"slopes {sl_u:+.2f}/{sl_l:+.2f}")
    out["pool_identity"] = pool_check
    out["rows"] = rows
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "leverage_radial_sampling.json"), "w") as f:
        json.dump(out, f, indent=1)
    log("wrote results/leverage_radial_sampling.json")


if __name__ == "__main__":
    main()

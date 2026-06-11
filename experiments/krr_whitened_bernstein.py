#!/usr/bin/env python3
"""
Whitened matrix-Bernstein KRR condition (validates thm:krr_whitened, cor:krr_highprob, thm:class_bernstein).

Theory under test, with A=K+lambda I and rho_D = ||A^{-1/2}(K_D-K)A^{-1/2}||_op:
  (A) eq:whitened_tail -- rho_D decays at the O(D^{-1/2}) Monte-Carlo rate, for every lambda;
  (B) eq:whitened_count -- the draw count D*(rho0=1/2) to reach rho_D<=1/2 scales with (1+||P||_op/lambda);
  (C) the whitened variance majorant A^{-1/2}KA^{-1/2} has intrinsic dimension EXACTLY
      d_tilde_lambda = d_eff(lambda)/kappa_lambda, d_eff=tr(K A^{-1}), kappa=||K||/(||K||+lambda)
      -- the effective dimension is the intrinsic dimension of the whitened variance, not imported;
  (D) cor:krr_highprob -- the ridge objective value lambda y^T(K_D+lambda I)^{-1}y lands in
      [1/(1+rho_D), 1/(1-rho_D)] times the exact lambda y^T(K+lambda I)^{-1}y;
  (E) thm:class_bernstein on the polynomially modulated Matern-1/2 kernel
      (x.w+b)^q e^{-||x-w||/sigma}: the Levy sampler T=1/(2 sigma^2 Z^2), Z~N(0,1) (mass m_f=1)
      is unbiased, and the same whitened bound holds with P_u = m_f * modulation Gram.

Env: numpy. Run: ~/.pixi/envs/jax/bin/python3 krr_whitened_bernstein.py -> results/krr_whitened_bernstein.json

Result (N=300, d=20, ||P||=306.4): (C) intdim(A^{-1/2}KA^{-1/2})==d_eff/kappa to machine
precision, rel.err <= 4e-15 across lambda in {10,1,0.1,0.01}; (A) whitened rho_D rate
slopes -0.50..-0.70 (O(D^{-1/2}) upper bound, steeper in the small-rho regime); (B)
D*(rho<=1/2)=50 at lambda=10 vs 800 at lambda=1, tracking the (1+||P||/lambda) scaling
(the two smaller lambda need D beyond the 800 grid cap, as predicted); (D) objective
sandwich holds on every seed with rho_D<1; (E) Matern-1/2 Levy sampler unbiased
(rel.err 0.026 at 8 reps -> 0), whitened rate slope -0.76, ||P_u||=306.4 (m_f=1).
"""
import json, os, time
import numpy as np
import krr_downstream as K
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
HERE = os.path.dirname(__file__)


def opnorm(A): return float(np.linalg.norm((A + A.T) / 2, 2))


def whitener(A):
    """Return A^{-1/2} (symmetric) for SPD A via eigendecomposition."""
    w, V = np.linalg.eigh((A + A.T) / 2)
    w = np.maximum(w, 1e-12)
    return (V / np.sqrt(w)[None, :]) @ V.T


def rho_whitened(Ainvhalf, KD, Kex):
    return opnorm(Ainvhalf @ (KD - Kex) @ Ainvhalf)


# ---- Matern-1/2 modulated estimator (thm:class_bernstein instance) --------------------
def k_matern12(A, B, b, sigma, q=2):
    """Exact (x.w+b)^q * exp(-||x-w||/sigma); f(r)=exp(-sqrt(r)/sigma) is CM in r=||x-w||^2, m_f=1."""
    return (A @ B.T + b) ** q * np.exp(-np.sqrt(K.sqdist(A, B)) / sigma)


def matern12_cross(A, B, b, sigma, D, rng, q=2):
    """Class estimator: radial scale by the Levy law T=1/(2 sigma^2 Z^2), modulation kept exact."""
    G = (A @ B.T + b) ** q                       # P_u = m_f * G_u = 1 * G  (m_f=1)
    acc = np.zeros((A.shape[0], B.shape[0]))
    for _ in range(D):
        Z = rng.normal()
        t = 1.0 / (2.0 * sigma ** 2 * Z ** 2)    # Levy(0, c) with c=1/(2 sigma^2): law of c/Z^2
        w = rng.normal(size=A.shape[1]) * np.sqrt(2.0 * t)
        beta = rng.uniform(0.0, 2.0 * np.pi)
        ca, cb = np.cos(A @ w + beta), np.cos(B @ w + beta)
        acc += 2.0 * np.outer(ca, cb)            # E[2 cos cos] = e^{-t||x-w||^2}
    return (acc / D) * G


def main():
    b, eps, N, d = 1.0, 1.0, 300, 20
    rng = np.random.default_rng(0)
    # moderate-spectrum bounded-ball dataset (rank ~6 -> nontrivial d_eff, ||P||,||K|| << N)
    X = rng.normal(size=(N, 6)) @ rng.normal(size=(6, d)) * 0.3
    X = X / (np.linalg.norm(X, axis=1).max() + 1e-12)
    P = (X @ X.T + b) ** 2 / eps
    Kex = K.k_yat(X, X, b, eps)
    nP, nK = opnorm(P), opnorm(Kex)
    y = rng.normal(size=N)
    lambdas = [10.0, 1.0, 0.1, 0.01]
    Dgrid = [25, 50, 100, 200, 400, 800]
    SEEDS = 6
    out = {"config": {"b": b, "eps": eps, "N": N, "d": d, "P_op": nP, "K_op": nK,
                      "lambdas": lambdas, "Dgrid": Dgrid, "seeds": SEEDS}}
    log(f"whitened matrix-Bernstein KRR: N={N}, d={d}, ||P||={nP:.2f}, ||K||={nK:.2f}")

    # precompute per-lambda whiteners + intrinsic-dimension identities (Part C)
    pre = {}
    log(f"  (C) intrinsic dim of whitened variance == d_eff/kappa exactly?")
    log(f"  {'lambda':>8} {'d_eff':>8} {'kappa':>7} {'d_tilde':>8} {'intdim(M~)':>11} {'rel.err':>9}")
    cpart = []
    for lam in lambdas:
        A = Kex + lam * np.eye(N)
        Ainvhalf = whitener(A)
        Ainv = Ainvhalf @ Ainvhalf
        d_eff = float(np.trace(Kex @ Ainv))
        kappa = nK / (nK + lam)
        d_tilde = d_eff / kappa
        Mt = Ainvhalf @ Kex @ Ainvhalf            # whitened variance core A^{-1/2} K A^{-1/2}
        intdim = float(np.trace(Mt) / opnorm(Mt))
        rel = abs(intdim - d_tilde) / d_tilde
        pre[lam] = {"A": A, "Ainvhalf": Ainvhalf, "d_eff": d_eff, "kappa": kappa, "d_tilde": d_tilde}
        cpart.append({"lambda": lam, "d_eff": d_eff, "kappa": kappa,
                      "d_tilde": d_tilde, "intdim_whitened": intdim, "rel_err": rel})
        log(f"  {lam:>8.3g} {d_eff:>8.2f} {kappa:>7.3f} {d_tilde:>8.2f} {intdim:>11.2f} {rel:>9.2e}")
    out["C_intrinsic_dimension"] = cpart

    # (A) rho_D vs D rate, all lambda; (B) draw count to reach rho<=1/2; (D) objective sandwich
    log(f"  (A) rho_D vs D (O(D^-1/2) rate) and (B) D*(rho0=1/2) ~ (1+||P||/lambda)")
    apart, bpart, dpart = [], [], []
    yexact = {lam: float(lam * y @ np.linalg.solve(pre[lam]["A"], y)) for lam in lambdas}
    for lam in lambdas:
        Ainvhalf = pre[lam]["Ainvhalf"]
        rows = []
        for D in Dgrid:
            rhos, ratios = [], []
            for s in range(SEEDS):
                KD = K.ray_cross(X, X, b, eps, D, np.random.default_rng(1000 * s + D))
                r = rho_whitened(Ainvhalf, KD, Kex)
                rhos.append(r)
                if r < 1.0:                       # objective sandwich only meaningful for rho<1
                    approx = float(lam * y @ np.linalg.solve(KD + lam * np.eye(N), y))
                    ratios.append(approx / yexact[lam])
            mean_rho = float(np.mean(rhos))
            rows.append({"D": D, "rho": mean_rho, "rho_std": float(np.std(rhos)),
                         "obj_ratio": (float(np.mean(ratios)) if ratios else None)})
        # rate slope on log-log (expect ~ -0.5)
        Ds = np.array([r["D"] for r in rows], float)
        Rs = np.array([r["rho"] for r in rows], float)
        slope = float(np.polyfit(np.log(Ds), np.log(Rs), 1)[0])
        # smallest D in grid achieving mean rho <= 1/2 (None if not reached)
        reached = [r["D"] for r in rows if r["rho"] <= 0.5]
        Dstar = (min(reached) if reached else None)
        apart.append({"lambda": lam, "rate_slope": slope, "rows": rows})
        bpart.append({"lambda": lam, "1+P/lam": 1.0 + nP / lam, "D_star_rho_half": Dstar,
                      "pred_count_rho_half": float((16.0 / 0.25) * (1.0 + nP / lam) *
                                                   np.log(8.0 * pre[lam]["d_tilde"] / 0.05))})
        log(f"  lambda={lam:>7.3g}: rate slope={slope:+.3f}  D*(rho<=1/2)={Dstar}  1+||P||/lam={1.0+nP/lam:.1f}")
    out["A_rate"] = apart
    out["B_draw_count"] = bpart

    # (D) objective-value sandwich at a fixed D where rho is moderate, realized per-seed
    log(f"  (D) objective sandwich: lambda y^T(K_D+lI)^-1 y in [1/(1+rho),1/(1-rho)] * exact")
    Dfix = 200
    for lam in lambdas:
        Ainvhalf = pre[lam]["Ainvhalf"]
        ok, n = 0, 0
        rec = []
        for s in range(SEEDS):
            KD = K.ray_cross(X, X, b, eps, Dfix, np.random.default_rng(7000 + 13 * s))
            r = rho_whitened(Ainvhalf, KD, Kex)
            if r >= 1.0:
                continue
            approx = float(lam * y @ np.linalg.solve(KD + lam * np.eye(N), y))
            ratio = approx / yexact[lam]
            lo, hi = 1.0 / (1.0 + r), 1.0 / (1.0 - r)
            inside = bool(lo - 1e-9 <= ratio <= hi + 1e-9)
            ok += inside; n += 1
            rec.append({"rho": r, "ratio": ratio, "lo": lo, "hi": hi, "inside": inside})
        dpart.append({"lambda": lam, "D": Dfix, "inside_frac": (ok / n if n else None), "records": rec})
        log(f"  lambda={lam:>7.3g}: sandwich holds {ok}/{n} seeds at D={Dfix}")
    out["D_objective_sandwich"] = dpart

    # (E) Matern-1/2 class instance: unbiasedness + whitened bound with P_u
    log(f"  (E) Matern-1/2 (x.w+b)^2 e^{{-||x-w||/sigma}}: Levy sampler unbiased + whitened bound")
    sigma = 1.0
    Kmat = k_matern12(X, X, b, sigma, q=2)
    Gu = (X @ X.T + b) ** 2                        # P_u = m_f * G_u, m_f=1
    nPu = opnorm(Gu)
    # unbiasedness: average many independent draws -> exact kernel
    Dbig, reps = 400, 8
    avg = np.zeros((N, N))
    for s in range(reps):
        avg += matern12_cross(X, X, b, sigma, Dbig, np.random.default_rng(20000 + s), q=2)
    avg /= reps
    unbias_rel = float(np.linalg.norm(avg - Kmat) / np.linalg.norm(Kmat))
    lam = 0.1
    Amat = Kmat + lam * np.eye(N)
    Ainvhalf_m = whitener(Amat)
    erows = []
    for D in [50, 100, 200, 400, 800]:
        rs = [rho_whitened(Ainvhalf_m,
                           matern12_cross(X, X, b, sigma, D, np.random.default_rng(30000 + 7 * s + D), q=2),
                           Kmat) for s in range(SEEDS)]
        erows.append({"D": D, "rho": float(np.mean(rs))})
    eslope = float(np.polyfit(np.log([r["D"] for r in erows]),
                              np.log([r["rho"] for r in erows]), 1)[0])
    out["E_matern12"] = {"sigma": sigma, "Pu_op": nPu, "m_f": 1.0,
                         "unbias_rel_err": unbias_rel, "lambda": lam,
                         "rate_slope": eslope, "rows": erows}
    log(f"  Matern-1/2: unbiased rel.err={unbias_rel:.4f} (->0), whitened rate slope={eslope:+.3f}, ||P_u||={nPu:.2f}")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "krr_whitened_bernstein.json"), "w"), indent=2)
    log("wrote results/krr_whitened_bernstein.json")


if __name__ == "__main__":
    main()

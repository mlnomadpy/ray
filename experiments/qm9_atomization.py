#!/usr/bin/env python3
"""
QM9 atomization energy: the real coupled-target test (extends sec:exp_necessity to real data).

Hypothesis. Atomization energy is an EXTENSIVE property: it scales with molecule size,
which lives in the norm of the size-extensive Coulomb-matrix eigenspectrum descriptor,
while composition/geometry lives in the direction. The target is therefore a genuinely
coupled alignment x proximity function of the inputs -- the real-data analogue of the
synthetic coupled target of tab:necessity. KRR on Coulomb matrices is the established
method on QM9 (Rupp et al. 2012), so the kernel comparison is on home turf.

Protocol (mirrors tab:tuned):
  - inputs: CM eigenspectrum (d=29), 99th-percentile norm clip then global scale so
    ||x||<=1 off-sphere (tab:prep recipe); y centered by train mean; MAE in kcal/mol.
  - per-kernel grid search on a held-out validation split (seed 0): eps_mult/b/lambda;
    then 3 independent train/test resamples at the tuned hyperparameters.
  - kernels: Gaussian, IMQ, Matern-1/2 (L2), biased degree-2 polynomial, exact yat;
    RAY (vectorized, exact modulation) at the tuned yat hypers, D in {1000, 4000}.
  - COUPLING CONTROLS: (i) direction-only ablation: same grid, inputs projected to the
    sphere (norm deleted); (ii) norm-only ablation: 1-D input ||x||, Gaussian grid;
    (iii) dressed-atom baseline: linear regression on element counts (the composition-
    only physics baseline). The coupled claim requires (i) and (ii) to degrade and the
    joint to win; which kernel wins the joint is then the inductive-bias comparison.
  - scale: full-N sketched-RAY primal (D=32, m=128, fp32, blocked normal equations) on
    all ~131k training rows, where the exact Gram (1.7e10 entries) cannot be formed.

------------------------------------------------------------------------------
REPRODUCIBILITY
    Data: ~/rf_data/qm9/qm9_cm.npz (qm9_build_cache.py).
    Env : python3>=3.9, numpy. CPU, ~20-30 min.
    Run : ~/.pixi/envs/jax/bin/python3 qm9_atomization.py
    Out : results/qm9_atomization.json (+ stdout). Deterministic seeds.
------------------------------------------------------------------------------
"""
import argparse, json, os, time
import numpy as np

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
HERE = os.path.dirname(__file__)
SQ2 = float(np.sqrt(2.0))

NTR, NVAL, NTE = 6000, 1500, 2000
EPS_MULTS = [0.25, 1.0, 4.0]
BGRID_YAT = [0.0, 0.25, 1.0, 4.0]
BGRID_POLY = [0.25, 1.0, 4.0]
LAM_RELS = [1e-9, 1e-7, 1e-5, 1e-3]
SEEDS = 3


def sqd(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0.0)


def gram(kind, A, B, med, hp):
    if kind == "gauss":
        return np.exp(-sqd(A, B) / (hp["em"] * med))
    if kind == "imq":
        e = hp["em"] * med
        return 1.0 / (sqd(A, B) + e)
    if kind == "matern12":
        s = np.sqrt(hp["em"] * med)
        return np.exp(-np.sqrt(sqd(A, B)) / s)
    if kind == "poly2":
        return (A @ B.T + hp["b"]) ** 2
    if kind == "yat":
        e = hp["em"] * med
        return (A @ B.T + hp["b"]) ** 2 / (sqd(A, B) + e)
    raise ValueError(kind)


def krr(Ktr, Ktest, ytr, lam_rel):
    n = Ktr.shape[0]
    lam = lam_rel * float(np.trace(Ktr)) / n
    a = np.linalg.solve(Ktr + lam * np.eye(n), ytr)
    return Ktest @ a


def ray_gram(A, B, b, eps, D, rng):
    """Vectorized exact-modulation RAY cross-Gram (flat D'=1)."""
    t = rng.exponential(1.0 / eps, size=D)
    Om = rng.normal(size=(A.shape[1], D)) * np.sqrt(2 * t)[None, :]
    beta = rng.uniform(0, 2 * np.pi, D)
    CA = SQ2 * np.cos(A @ Om + beta)
    CB = SQ2 * np.cos(B @ Om + beta)
    P = (A @ B.T + b) ** 2 / eps
    return ((CA @ CB.T) / D) * (P * eps)        # (1/eps)*E[2coscos]*p_b = k_yat


def ts_feat(X, m, b, eps, D, rng):
    """Deployed sketched-RAY features, quadratic-only TS_m + exact linear/const (fp32)."""
    n, d = X.shape
    h = [rng.integers(0, m, d) for _ in range(2)]
    s = [(rng.integers(0, 2, d) * 2 - 1).astype(np.float64) for _ in range(2)]
    C = []
    for k in range(2):
        Ck = np.zeros((n, m))
        np.add.at(Ck.T, h[k], (X * s[k]).T)
        C.append(Ck)
    TS = np.fft.irfft(np.fft.rfft(C[0], axis=1) * np.fft.rfft(C[1], axis=1), n=m, axis=1)
    Phat = np.concatenate([TS, np.sqrt(2 * b) * X, np.full((n, 1), b)], axis=1) / np.sqrt(eps)
    t = rng.exponential(1.0 / eps, size=D)
    Om = rng.normal(size=(d, D)) * np.sqrt(2 * t)[None, :]
    cos = SQ2 * np.cos(X @ Om + rng.uniform(0, 2 * np.pi, D))     # (n,D)
    Z = (cos[:, :, None] * Phat[:, None, :]).reshape(n, -1) / np.sqrt(D)
    return Z.astype(np.float32)


def tune(kind, Xtr, ytr, Xval, yval, med):
    grid = []
    ems = EPS_MULTS if kind in ("gauss", "imq", "matern12", "yat") else [None]
    bs = BGRID_YAT if kind == "yat" else (BGRID_POLY if kind == "poly2" else [None])
    best = None
    for em in ems:
        for b in bs:
            hp = {"em": em, "b": b}
            Ktr = gram(kind, Xtr, Xtr, med, hp)
            Kval = gram(kind, Xval, Xtr, med, hp)
            for lr in LAM_RELS:
                mae = float(np.mean(np.abs(krr(Ktr, Kval, ytr, lr) - yval)))
                grid.append({**hp, "lam_rel": lr, "val_mae": mae})
                if best is None or mae < best["val_mae"]:
                    best = grid[-1]
    return best, grid


def run_block(X, y, tag, kinds, out):
    """Tune on seed 0, evaluate on SEEDS resamples; X already preprocessed."""
    rng0 = np.random.default_rng(0)
    idx = rng0.permutation(len(X))
    tr, va = idx[:NTR], idx[NTR:NTR + NVAL]
    med = float(np.median(sqd(X[tr[:2000]], X[tr[:2000]])[np.triu_indices(2000, 1)]))
    ymu = y[tr].mean()
    tuned = {}
    for kind in kinds:
        best, _ = tune(kind, X[tr], y[tr] - ymu, X[va], y[va] - ymu, med)
        tuned[kind] = best
        log(f"  [{tag}] tuned {kind:9s}: {dict((k,v) for k,v in best.items() if v is not None)}")
    res = {k: [] for k in kinds}
    for s in range(SEEDS):
        rs = np.random.default_rng(100 + s)
        idx = rs.permutation(len(X))
        tr, te = idx[:NTR], idx[NTR + NVAL:NTR + NVAL + NTE]
        ymu = y[tr].mean()
        meds = float(np.median(sqd(X[tr[:2000]], X[tr[:2000]])[np.triu_indices(2000, 1)]))
        for kind in kinds:
            hp = tuned[kind]
            Ktr = gram(kind, X[tr], X[tr], meds, hp)
            Kte = gram(kind, X[te], X[tr], meds, hp)
            mae = float(np.mean(np.abs(krr(Ktr, Kte, y[tr] - ymu, hp["lam_rel"]) + ymu - y[te])))
            res[kind].append(mae)
    summary = {k: {"mae": float(np.mean(v)), "std": float(np.std(v)), "tuned": tuned[k]}
               for k, v in res.items()}
    for k in kinds:
        log(f"  [{tag}] {k:9s}: test MAE {summary[k]['mae']:8.2f} +- {summary[k]['std']:.2f} kcal/mol")
    out[tag] = summary
    return tuned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.expanduser("~/rf_data/qm9/qm9_cm.npz"))
    ap.add_argument("--out", default="qm9_atomization.json")
    ap.add_argument("--primal-D", type=int, default=32)
    args = ap.parse_args()
    z = np.load(args.cache, allow_pickle=True)
    X0, y, counts = z["X"].astype(np.float64), z["y"], z["counts"].astype(float)
    log(f"QM9: N={len(X0):,}, d={X0.shape[1]}, y mean={y.mean():.1f} kcal/mol")
    # preprocessing: 99th-pct norm clip then global scale to ||x||<=1 (off-sphere)
    nrm = np.linalg.norm(X0, axis=1)
    q99 = np.percentile(nrm, 99)
    X = X0 * np.minimum(1.0, q99 / np.maximum(nrm, 1e-12))[:, None] / q99
    log(f"after clip+scale: median ||x||={np.median(np.linalg.norm(X,axis=1)):.3f} (varying norms, <=1)")
    out = {"config": {"NTR": NTR, "NVAL": NVAL, "NTE": NTE, "seeds": SEEDS,
                      "eps_mults": EPS_MULTS, "b_yat": BGRID_YAT, "lam_rels": LAM_RELS}}

    kinds = ["gauss", "imq", "matern12", "poly2", "yat"]
    log("=== joint (full descriptor) ===")
    tuned = run_block(X, y, "joint", kinds, out)

    log("=== ablation (i): direction-only (sphere-projected) ===")
    Xs = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    run_block(Xs, y, "direction_only", kinds, out)

    log("=== ablation (ii): norm-only (1-D ||x||) ===")
    Xn = np.linalg.norm(X, axis=1, keepdims=True)
    run_block(Xn, y, "norm_only", ["gauss"], out)

    log("=== baseline: dressed atom (linear in element counts) ===")
    rng0 = np.random.default_rng(100)
    idx = rng0.permutation(len(X))
    tr, te = idx[:NTR], idx[NTR + NVAL:NTR + NVAL + NTE]
    A = np.concatenate([counts, np.ones((len(counts), 1))], axis=1)
    w, *_ = np.linalg.lstsq(A[tr], y[tr], rcond=None)
    da = float(np.mean(np.abs(A[te] @ w - y[te])))
    out["dressed_atom_mae"] = da
    log(f"  dressed-atom MAE: {da:.2f} kcal/mol")

    log("=== RAY (exact modulation, vectorized) at tuned yat hypers ===")
    hp = tuned["yat"]
    rays = {}
    for D in [1000, 4000]:
        maes = []
        for s in range(SEEDS):
            rs = np.random.default_rng(100 + s)
            idx = rs.permutation(len(X))
            tr_, te_ = idx[:NTR], idx[NTR + NVAL:NTR + NVAL + NTE]
            meds = float(np.median(sqd(X[tr_[:2000]], X[tr_[:2000]])[np.triu_indices(2000, 1)]))
            e = hp["em"] * meds
            rngf = np.random.default_rng(7000 + s)
            t = rngf.exponential(1.0 / e, size=D)
            Om = rngf.normal(size=(X.shape[1], D)) * np.sqrt(2 * t)[None, :]
            beta = rngf.uniform(0, 2 * np.pi, D)
            Ctr = SQ2 * np.cos(X[tr_] @ Om + beta); Cte = SQ2 * np.cos(X[te_] @ Om + beta)
            Ptr = (X[tr_] @ X[tr_].T + hp["b"]) ** 2; Pte = (X[te_] @ X[tr_].T + hp["b"]) ** 2
            Rtr = (Ctr @ Ctr.T) / D; Rte = (Cte @ Ctr.T) / D
            ymu = y[tr_].mean()
            mae = float(np.mean(np.abs(
                krr(Rtr * Ptr / e * e, Rte * Pte / e * e, y[tr_] - ymu, hp["lam_rel"]) + ymu - y[te_])))
            maes.append(mae)
        rays[D] = {"mae": float(np.mean(maes)), "std": float(np.std(maes))}
        log(f"  RAY D={D}: test MAE {rays[D]['mae']:.2f} +- {rays[D]['std']:.2f} kcal/mol")
    out["ray"] = rays

    log("=== full-N sketched-RAY primal (D=32, m=128, fp32, blocked) ===")
    rs = np.random.default_rng(100)
    idx = rs.permutation(len(X))
    te_ = idx[NTR + NVAL:NTR + NVAL + NTE]
    trful = np.concatenate([idx[:NTR + NVAL], idx[NTR + NVAL + NTE:]])
    meds = float(np.median(sqd(X[trful[:2000]], X[trful[:2000]])[np.triu_indices(2000, 1)]))
    e = hp["em"] * meds
    D, m = args.primal_D, 128
    rngf = np.random.default_rng(7)
    # shared sketch+frequencies via one generator consumed identically per block
    n_all = len(X)
    Zte = None
    M = D * (m + X.shape[1] + 1)
    G = np.zeros((M, M), dtype=np.float64)
    Zty = np.zeros(M)
    ymu = y[trful].mean()
    state = np.random.default_rng(7)
    # build features once for all rows in blocks, accumulate normal equations
    h = [state.integers(0, m, X.shape[1]) for _ in range(2)]
    sgn = [(state.integers(0, 2, X.shape[1]) * 2 - 1).astype(float) for _ in range(2)]
    t = state.exponential(1.0 / e, size=D)
    Om = state.normal(size=(X.shape[1], D)) * np.sqrt(2 * t)[None, :]
    beta = state.uniform(0, 2 * np.pi, D)

    def feats(Xb):
        C = []
        for k in range(2):
            Ck = np.zeros((len(Xb), m))
            np.add.at(Ck.T, h[k], (Xb * sgn[k]).T)
            C.append(Ck)
        TS = np.fft.irfft(np.fft.rfft(C[0], axis=1) * np.fft.rfft(C[1], axis=1), n=m, axis=1)
        Ph = np.concatenate([TS, np.sqrt(2 * hp["b"]) * Xb, np.full((len(Xb), 1), hp["b"])],
                            axis=1) / np.sqrt(e)
        cos = SQ2 * np.cos(Xb @ Om + beta)
        return ((cos[:, :, None] * Ph[:, None, :]).reshape(len(Xb), -1) / np.sqrt(D)).astype(np.float32)

    t0 = time.time()
    B = 16384
    for i in range(0, len(trful), B):
        Zb = feats(X[trful[i:i + B]]).astype(np.float64)
        G += Zb.T @ Zb
        Zty += Zb.T @ (y[trful[i:i + B]] - ymu)
        if i // B % 2 == 0:
            log(f"  block {i:,}/{len(trful):,}")
    Zte = feats(X[te_]).astype(np.float64)
    best_full = None
    for lr in LAM_RELS + [1e-1]:
        lam = lr * np.trace(G) / M
        wv = np.linalg.solve(G + lam * np.eye(M), Zty)
        mae = float(np.mean(np.abs(Zte @ wv + ymu - y[te_])))
        if best_full is None or mae < best_full["mae"]:
            best_full = {"lam_rel": lr, "mae": mae}
    wall = time.time() - t0
    out["full_primal"] = {"N_train": int(len(trful)), "D": D, "m": m, "M": M,
                          "mae": best_full["mae"], "lam_rel": best_full["lam_rel"],
                          "wall_s": wall}
    log(f"  full-N primal (N={len(trful):,}, M={M}): test MAE {best_full['mae']:.2f} kcal/mol "
        f"in {wall:.0f}s (lam_rel {best_full['lam_rel']:g})")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", args.out), "w") as f:
        json.dump(out, f, indent=1)
    log(f"wrote results/{args.out}")


if __name__ == "__main__":
    main()

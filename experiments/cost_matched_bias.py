#!/usr/bin/env python3
"""
Cost-matched among TRUE k_yat,b-approximators, with a bias sweep (R3 #2 follow-up).

Fixes two things from cost_matched.py:
  - Drops the Gaussian/IMQ baselines, which approximate DIFFERENT kernels. Here every
    method approximates the SAME biased yat-kernel k_yat,b: RAY and the exact-numerator
    hybrid (both keep the three-atom polynomial exact), whole-kernel Random Maclaurin
    (dimension-efficient, gives up exactness), and the optimal rank-M oracle (ceiling).
  - Sweeps the bias b. The exact feature is the THREE ATOMS
        p_b(x) = ( vec(x (x) x) ,  sqrt(2b) x ,  b ),
    with on-sphere squared norms  1 : 2b : b^2  (summing to (1+b)^2). So b retunes how
    much of the kernel lives in the quadratic atom (the only one costing O(d^2)) versus
    the cheap linear/const atoms. b->0 is quad-only (unbiased); large b is const-dominated.

Question: does any bias make the exact d^2 numerator dimension-competitive, or is the
dimension-efficient dot-product route (Random Maclaurin) better at every b?

Evaluation is in Gram form (K=Phi Phi^T) so matched dimension M only drives feature-build
cost, not an M^3 solve -- identical KRR result, far faster across the sweep.

------------------------------------------------------------------------------
REPRODUCIBILITY
    Env : python3>=3.9, numpy, scipy, scikit-learn. CPU.
          Run: ~/.pixi/envs/jax/bin/python3 cost_matched_bias.py
    Out : results/cost_matched_bias.json (+ stdout). Deterministic seeds.
    Reuses loaders/kernels/coeffs from krr_downstream.py.
------------------------------------------------------------------------------
"""
import argparse, json, os, time
import numpy as np
import krr_downstream as K

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


# --------------------------------- three-atom exact polynomial feature p_b(x) ------
def poly2_features(X, b, eps):
    """Three atoms: quadratic vec(x⊗x) [d(d+1)/2], linear sqrt(2b)x [d], constant b [1].

    Drops the linear+const atoms when b=0 (they are identically zero), so the unbiased
    kernel is charged only its true d(d+1)/2 quadratic dimension.
    """
    n, d = X.shape
    iu = np.triu_indices(d, 1)
    quad = np.concatenate([X * X, np.sqrt(2.0) * X[:, iu[0]] * X[:, iu[1]]], axis=1)  # (x.w)^2
    if b == 0.0:
        P = quad
    else:
        P = np.concatenate([quad, np.sqrt(2.0 * b) * X, np.full((n, 1), b)], axis=1)
    return P / np.sqrt(eps)


def d_b(d, b):
    return d * (d + 1) // 2 + (d + 1 if b != 0.0 else 0)


def atom_norm_fractions(b):
    tot = (1.0 + b) ** 2
    return {"quad": 1.0 / tot, "linear": 2.0 * b / tot, "const": b * b / tot}


# --------------------------------------------------------- primal feature builders ----
def ray_primal(X, b, eps, D, rng):
    P = poly2_features(X, b, eps); n, d = X.shape
    blocks = []
    for t in rng.exponential(scale=1.0 / eps, size=D):
        w = rng.normal(size=d) * np.sqrt(2.0 * t)
        beta = rng.uniform(0.0, 2.0 * np.pi)
        blocks.append((np.sqrt(2.0) * np.cos(X @ w + beta)[:, None]) * P)
    return np.concatenate(blocks, axis=1) / np.sqrt(D)


def hybrid_primal(X, b, eps, Mr, rng, nmax=24):
    """Exact three-atom numerator (x) Random-Maclaurin features of the radial factor."""
    P = poly2_features(X, b, eps); n, d = X.shape
    coeffs = K._radial_maclaurin_coeffs(eps, nmax); Z = coeffs.sum(); p = coeffs / Z
    degs = rng.choice(len(coeffs), size=Mr, p=p)
    blocks = []
    for deg in degs:
        col = np.ones(n)
        for _ in range(int(deg)):
            col *= X @ (rng.integers(0, 2, size=d) * 2 - 1)
        blocks.append((np.sqrt(Z) * col)[:, None] * P)
    return np.concatenate(blocks, axis=1) / np.sqrt(Mr)


def randmac_primal(X, b, eps, M, rng, nmax=24):
    a = K._maclaurin_coeffs(b, eps, nmax); Z = a.sum(); p = a / Z
    n, d = X.shape
    degs = rng.choice(nmax + 1, size=M, p=p)
    F = np.empty((n, M))
    for r, deg in enumerate(degs):
        col = np.ones(n)
        for _ in range(int(deg)):
            col *= X @ (rng.integers(0, 2, size=d) * 2 - 1)
        F[:, r] = col
    return np.sqrt(Z / M) * F


def gram_eval(task, Ptr, Pte, Ytr, yte, lam):
    return K.evaluate(task, Ptr @ Ptr.T, Pte @ Ptr.T, Ytr, yte, lam)


def oracle_eval(task, Xtr, Xte, b, eps, M, Ytr, yte, lam):
    G = K.k_yat(Xtr, Xtr, b, eps)
    ev, U = np.linalg.eigh((G + G.T) / 2.0)
    idx = np.argsort(ev)[::-1][:M]; Uu = U[:, idx]; lm = np.maximum(ev[idx], 0.0)
    return K.evaluate(task, (Uu * lm) @ Uu.T, (K.k_yat(Xte, Xtr, b, eps) @ Uu) @ Uu.T, Ytr, yte, lam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--bs", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--draws", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "results", "cost_matched_bias.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    log(f"config: {vars(args)}")

    # three-atom sanity: poly Gram reproduces (x.w+b)^2/eps exactly
    Xs = K._sphere(np.random.default_rng(1).normal(size=(16, 5)))
    for bb in (0.0, 1.0):
        P = poly2_features(Xs, bb, 1.0)
        err = np.max(np.abs(P @ P.T - (Xs @ Xs.T + bb) ** 2))
        log(f"  three-atom check b={bb}: max|Phi Phi^T - (x.w+b)^2| = {err:.2e}  (d_b={P.shape[1]})")

    out = {"config": vars(args), "datasets": {}}
    for loader in (K.load_digits_ds, K.load_reg_ds):
        name, task, X, y = loader(np.random.default_rng(0))
        N, d = X.shape
        ntr, nte = min(args.n_train, N * 2 // 3), min(args.n_test, N // 3)
        sub = np.random.default_rng(0).choice(N, size=min(N, 1000), replace=False)
        eps = float(np.median(K.sqdist(X[sub], X[sub])[np.triu_indices(len(sub), 1)]))
        unit = "RMSE" if task == "reg" else "acc"
        log(f"=== {name} ({task}, N={N}, d={d}) eps={eps:.3f} ntr={ntr} ===")
        ds = {"task": task, "d": d, "eps": eps, "by_b": {}}
        for b in args.bs:
            dbv = d_b(d, b); Ms = [dr * dbv for dr in args.draws]
            agg = {}
            for s in range(args.seeds):
                rng = np.random.default_rng(100 + s)
                perm = rng.permutation(N); tr, te = perm[:ntr], perm[ntr:ntr + nte]
                Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
                Ytr = ytr if task == "reg" else np.eye(int(y.max()) + 1)[ytr]
                agg.setdefault("exact", []).append(
                    K.evaluate(task, K.k_yat(Xtr, Xtr, b, eps), K.k_yat(Xte, Xtr, b, eps), Ytr, yte, args.lam))
                for dr, M in zip(args.draws, Ms):
                    agg.setdefault(f"ryf@{M}", []).append(gram_eval(task,
                        ray_primal(Xtr, b, eps, dr, np.random.default_rng(900 + s)),
                        ray_primal(Xte, b, eps, dr, np.random.default_rng(900 + s)), Ytr, yte, args.lam))
                    agg.setdefault(f"hybrid@{M}", []).append(gram_eval(task,
                        hybrid_primal(Xtr, b, eps, dr, np.random.default_rng(906 + s)),
                        hybrid_primal(Xte, b, eps, dr, np.random.default_rng(906 + s)), Ytr, yte, args.lam))
                    agg.setdefault(f"randmac@{M}", []).append(gram_eval(task,
                        randmac_primal(Xtr, b, eps, M, np.random.default_rng(904 + s)),
                        randmac_primal(Xte, b, eps, M, np.random.default_rng(904 + s)), Ytr, yte, args.lam))
                    agg.setdefault(f"oracle@{M}", []).append(
                        oracle_eval(task, Xtr, Xte, b, eps, min(M, ntr), Ytr, yte, args.lam))
            summary = {k: [float(np.mean(v)), float(np.std(v))] for k, v in agg.items()}
            ds["by_b"][f"{b}"] = {"d_b": dbv, "Ms": Ms,
                                  "atom_fractions": atom_norm_fractions(b), "results": summary}
            frac = atom_norm_fractions(b)
            log(f"  b={b:<4} d_b={dbv:5d}  atoms(quad/lin/const)="
                f"{frac['quad']:.2f}/{frac['linear']:.2f}/{frac['const']:.2f}  "
                f"exact={summary['exact'][0]:.4f}")
            for M in Ms:
                log(f"    M={M:6d}  RAY={summary[f'ryf@{M}'][0]:.4f}  "
                    f"hybrid={summary[f'hybrid@{M}'][0]:.4f}  "
                    f"randMac={summary[f'randmac@{M}'][0]:.4f}  "
                    f"oracle={summary[f'oracle@{M}'][0]:.4f}  [{unit}]")
        ds["metric"] = "RMSE" if task == "reg" else "accuracy"
        out["datasets"][name] = ds
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()

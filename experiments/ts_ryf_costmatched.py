#!/usr/bin/env python3
"""
TensorSketch-compressed RAY at matched feature dimension (R5/R6 #1).

RAY keeps the degree-2 polynomial factor exact, costing d_b=O(d^2) coordinates per
radial draw, so at a matched explicit feature dimension M it can afford few draws. We
compress the polynomial factor with TensorSketch (Pham & Pagh 2013): a degree-2 sketch
TS_2(x) of dimension m with E[TS_2(x).TS_2(w)] = (x.w)^2, giving a biased feature
  TS_b(x) = ( TS_2(x) , sqrt(2b) x , b ) / sqrt(eps),   dim m + d + 1,
with E[TS_b(x).TS_b(w)] = (x.w+b)^2/eps. Tensored with the radial RFF this yields
TensorSketch-RAY of dimension D(m+d+1) -- so at fixed M it affords many more radial
draws than exact modulation, trading polynomial exactness for radial resolution.

We compare at matched M (digits d=64 where the d^2 floor bites, california d=8 where it
does not) against exact modulation, whole-kernel Random Maclaurin, and the optimal rank-M oracle.
Question: does compressing the polynomial let RAY recover the dimension efficiency it
loses by keeping the numerator exact?

------------------------------------------------------------------------------
REPRODUCIBILITY
    Env : python3>=3.9, numpy, scikit-learn. CPU. Run: ~/.pixi/envs/jax/bin/python3 ts_ryf_costmatched.py
    Out : results/ts_ryf_costmatched.json (+ stdout). Deterministic seeds.
    Reuses kernels/loaders from krr_downstream.py and primal helpers from cost_matched_bias.py.
------------------------------------------------------------------------------
"""
import argparse, json, os, time
import numpy as np
import krr_downstream as K
import cost_matched_bias as CMB

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


# ----------------------------------------------------------------- TensorSketch ----
def _count_sketch(X, m, h, s):
    n, d = X.shape
    C = np.zeros((n, m))
    Xs = X * s[None, :]
    for i in range(d):
        C[:, h[i]] += Xs[:, i]
    return C


def _tensorsketch2(X, m, rng):
    """Degree-2 TensorSketch: E[TS(x).TS(w)] = (x.w)^2, dimension m."""
    d = X.shape[1]
    h1 = rng.integers(0, m, d); s1 = rng.integers(0, 2, d) * 2 - 1
    h2 = rng.integers(0, m, d); s2 = rng.integers(0, 2, d) * 2 - 1
    C1 = _count_sketch(X, m, h1, s1)
    C2 = _count_sketch(X, m, h2, s2)
    return np.fft.irfft(np.fft.rfft(C1, axis=1) * np.fft.rfft(C2, axis=1), n=m, axis=1)


def _ts_poly_b(X, m, b, eps, rng):
    TS = _tensorsketch2(X, m, rng)
    P = np.concatenate([TS, np.sqrt(2.0 * b) * X, np.full((X.shape[0], 1), b)], axis=1)
    return P / np.sqrt(eps)


def ts_ray_primal(X, b, eps, D, m, seed):
    """TensorSketch-RAY primal feature, dim D(m+d+1). seed fixes sketch+radial draws."""
    rng = np.random.default_rng(seed)
    P = _ts_poly_b(X, m, b, eps, rng)
    blocks = []
    for t in rng.exponential(scale=1.0 / eps, size=D):
        w = rng.normal(size=X.shape[1]) * np.sqrt(2.0 * t)
        beta = rng.uniform(0.0, 2.0 * np.pi)
        blocks.append((np.sqrt(2.0) * np.cos(X @ w + beta))[:, None] * P)
    return np.concatenate(blocks, axis=1) / np.sqrt(D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--draws", type=int, nargs="+", default=[1, 2, 4])  # exact-modulation draws -> M grid
    ap.add_argument("--sketch", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "results", "ts_ryf_costmatched.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    log(f"config: {vars(args)}")

    # sanity: E[TS(x).TS(w)] -> (x.w)^2
    Xs = K._sphere(np.random.default_rng(1).normal(size=(8, 12)))
    acc = np.zeros((8, 8))
    for r in range(400):
        TS = _tensorsketch2(Xs, 64, np.random.default_rng(r))
        acc += TS @ TS.T
    acc /= 400
    log(f"  TensorSketch sanity: max|E[TS.TS]-(x.w)^2| = {np.max(np.abs(acc - (Xs@Xs.T)**2)):.3f} (mean of 400 sketches)")

    out = {"config": vars(args), "datasets": {}}
    for loader in (K.load_digits_ds, K.load_reg_ds):
        name, task, X, y = loader(np.random.default_rng(0))
        N, d = X.shape
        ntr, nte = min(args.n_train, N * 2 // 3), min(args.n_test, N // 3)
        sub = np.random.default_rng(0).choice(N, size=min(N, 1000), replace=False)
        eps = float(np.median(K.sqdist(X[sub], X[sub])[np.triu_indices(len(sub), 1)]))
        dbv = CMB.d_b(d, args.b); Ms = [dr * dbv for dr in args.draws]
        unit = "RMSE" if task == "reg" else "acc"
        log(f"=== {name} ({task}, d={d}) d_b={dbv} eps={eps:.3f} M grid={Ms} ===")
        agg = {}
        for s in range(args.seeds):
            rng = np.random.default_rng(100 + s)
            perm = rng.permutation(N); tr, te = perm[:ntr], perm[ntr:ntr + nte]
            Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
            Ytr = ytr if task == "reg" else np.eye(int(y.max()) + 1)[ytr]
            agg.setdefault("exact_kernel", []).append(
                K.evaluate(task, K.k_yat(Xtr, Xtr, args.b, eps), K.k_yat(Xte, Xtr, args.b, eps), Ytr, yte, args.lam))
            for dr, M in zip(args.draws, Ms):
                agg.setdefault(f"ryf@{M}", []).append(CMB.gram_eval(task,
                    CMB.ray_primal(Xtr, args.b, eps, dr, np.random.default_rng(900 + s)),
                    CMB.ray_primal(Xte, args.b, eps, dr, np.random.default_rng(900 + s)), Ytr, yte, args.lam))
                agg.setdefault(f"randmac@{M}", []).append(CMB.gram_eval(task,
                    CMB.randmac_primal(Xtr, args.b, eps, M, np.random.default_rng(904 + s)),
                    CMB.randmac_primal(Xte, args.b, eps, M, np.random.default_rng(904 + s)), Ytr, yte, args.lam))
                agg.setdefault(f"oracle@{M}", []).append(
                    CMB.oracle_eval(task, Xtr, Xte, args.b, eps, min(M, ntr), Ytr, yte, args.lam))
                for m in args.sketch:
                    Dts = max(1, round(M / (m + d + 1)))
                    agg.setdefault(f"ts{m}@{M}", []).append(CMB.gram_eval(task,
                        ts_ray_primal(Xtr, args.b, eps, Dts, m, 1000 + s),
                        ts_ray_primal(Xte, args.b, eps, Dts, m, 1000 + s), Ytr, yte, args.lam))
        summary = {k: [float(np.mean(v)), float(np.std(v))] for k, v in agg.items()}
        out["datasets"][name] = {"task": task, "d": d, "d_b": dbv, "eps": eps, "Ms": Ms,
                                 "sketch": args.sketch, "metric": "RMSE" if task == "reg" else "accuracy",
                                 "results": summary}
        log(f"  exact kernel: {summary['exact_kernel'][0]:.4f}")
        for dr, M in zip(args.draws, Ms):
            ts = "  ".join(f"ts{m}={summary[f'ts{m}@{M}'][0]:.4f}(D{max(1,round(M/(m+d+1)))})" for m in args.sketch)
            log(f"  M={M:6d} (exact-modulation D={dr}): RAY={summary[f'ryf@{M}'][0]:.4f}  {ts}  "
                f"randMac={summary[f'randmac@{M}'][0]:.4f}  oracle={summary[f'oracle@{M}'][0]:.4f} [{unit}]")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Validation-tuned downstream comparison (reviewer gap #3).

The main downstream tables fix b=1, eps=median sqdist, lambda fixed. This asks the fairer
question: when every kernel is tuned for prediction on a validation split, is the yat-kernel
(and its RAY approximation) still competitive? We grid-search (b, eps-multiplier, lambda) on
a held-out validation set and report TEST metric, for:
  Gaussian RFF-kernel, IMQ, degree-2 polynomial, exact yat, deployed (sketched) RAY, Nystrom-yat.

Datasets (off-sphere bounded ball, the paper's niche):
  - coupled  : synthetic tanh-alignment x Laplace-proximity target (d=16, regression)
  - digits   : standardized + max-norm scaled to ||x||<=1 (d=64, classification)

Env: ~/.pixi/envs/jax/bin/python3 (numpy, sklearn). Run: tuned_downstream.py
REPRODUCIBILITY: results/tuned_downstream.json; backs the tuned-comparison table (sec:exp_necessity).
"""
import json, os, time, itertools
import numpy as np
import krr_downstream as K
import ts_ryf_costmatched as TS

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
HERE = os.path.dirname(__file__)


def ball(N, d, rng, lo=0.3, hi=1.5):
    U = rng.normal(size=(N, d)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    return U * rng.uniform(lo, hi, size=(N, 1))


def coupled_target(X, rng, k=6):
    d = X.shape[1]
    U = rng.normal(size=(k, d)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    V = ball(k, d, rng); a = rng.normal(size=k)
    y = sum(a[j] * np.tanh(2 * X @ U[j]) * np.exp(-np.linalg.norm(X - V[j], axis=1)) for j in range(k))
    return y


def k_poly(A, B, b): return (A @ B.T + b) ** 2


def load_coupled(seed):
    rng = np.random.default_rng(seed)
    X = ball(1400, 16, rng)
    y = coupled_target(X, np.random.default_rng(777))
    return "coupled", "reg", X, y


def load_digits_offsphere(seed):
    name, task, X, y = K.load_digits_ds(np.random.default_rng(seed))  # returns sphere-normalized
    # undo to off-sphere: K.load_digits_ds spheres; we want bounded ball -> rescale by max row norm
    # load raw via sklearn instead:
    from sklearn.datasets import load_digits
    D = load_digits(); X = D.data.astype(float); y = D.target
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    X = X / np.linalg.norm(X, axis=1).max()
    return "digits-offsphere", "cls", X, y


def split3(X, y, rng, ntr, nval, nte):
    perm = rng.permutation(len(y))
    return perm[:ntr], perm[ntr:ntr + nval], perm[ntr + nval:ntr + nval + nte]


def main():
    seeds = 3
    bs = [0.0, 0.5, 1.0, 2.0]
    emul = [0.25, 1.0, 4.0]
    lams = [1e-3, 1e-2, 1e-1]
    out = {"config": {"seeds": seeds, "bs": bs, "emul": emul, "lams": lams}, "datasets": {}}

    for loader in (load_coupled, load_digits_offsphere):
        name, task, X, y = loader(0)
        N, d = X.shape
        ntr, nval, nte = min(800, N // 2), min(300, N // 5), min(400, N // 4)
        Ytrans = (lambda yy: yy) if task == "reg" else (lambda yy: np.eye(int(y.max()) + 1)[yy])
        log(f"=== {name} ({task}, N={N}, d={d}) ntr={ntr} nval={nval} nte={nte} ===")
        res = {}
        for s in range(seeds):
            rng = np.random.default_rng(100 + s)
            tr, va, te = split3(X, y, rng, ntr, nval, nte)
            Xtr, Xva, Xte = X[tr], X[va], X[te]
            ytr, yva, yte = y[tr], y[va], y[te]
            Ytr = Ytrans(ytr)
            med = float(np.median(K.sqdist(Xtr, Xtr)[np.triu_indices(len(tr), 1)]))

            def tune(build_gram, grid):
                best, best_v = None, (-1e18 if task == "cls" else 1e18)
                for cfg in grid:
                    Ktr, Kva, Kte = build_gram(cfg)
                    v = K.evaluate(task, Ktr, Kva, Ytr, yva, cfg["lam"])
                    better = (v > best_v) if task == "cls" else (v < best_v)
                    if better: best_v, best = v, (cfg, Ktr, Kte)
                cfg, Ktr, Kte = best
                return K.evaluate(task, Ktr, Kte, Ytr, yte, cfg["lam"]), cfg

            grids = {
                "gaussian": [{"em": e, "lam": l} for e in emul for l in lams],
                "imq":      [{"em": e, "lam": l} for e in emul for l in lams],
                "poly":     [{"b": b, "lam": l} for b in bs for l in lams],
                "yat":      [{"b": b, "em": e, "lam": l} for b in bs for e in emul for l in lams],
                "ray":      [{"b": b, "em": e, "lam": l} for b in bs for e in emul for l in lams],
                "nystrom":  [{"b": b, "em": e, "lam": l} for b in bs for e in emul for l in lams],
            }

            def builders(cfg):
                return cfg

            def gram_gaussian(cfg):
                g = 1.0 / (cfg["em"] * med)
                return K.k_gauss(Xtr, Xtr, g), K.k_gauss(Xva, Xtr, g), K.k_gauss(Xte, Xtr, g)

            def gram_imq(cfg):
                e = cfg["em"] * med
                return K.k_imq(Xtr, Xtr, e), K.k_imq(Xva, Xtr, e), K.k_imq(Xte, Xtr, e)

            def gram_poly(cfg):
                b = cfg["b"]
                return k_poly(Xtr, Xtr, b), k_poly(Xva, Xtr, b), k_poly(Xte, Xtr, b)

            def gram_yat(cfg):
                b, e = cfg["b"], cfg["em"] * med
                return K.k_yat(Xtr, Xtr, b, e), K.k_yat(Xva, Xtr, b, e), K.k_yat(Xte, Xtr, b, e)

            def gram_ray(cfg):
                b, e = cfg["b"], cfg["em"] * med
                D, m = 24, 128
                Ztr = TS.ts_ray_primal(Xtr, b, e, D, m, 1000 + s)
                Zva = TS.ts_ray_primal(Xva, b, e, D, m, 1000 + s)
                Zte = TS.ts_ray_primal(Xte, b, e, D, m, 1000 + s)
                return Ztr @ Ztr.T, Zva @ Ztr.T, Zte @ Ztr.T

            def gram_nys(cfg):
                b, e = cfg["b"], cfg["em"] * med
                m = min(400, len(tr))
                rngn = np.random.default_rng(30 + s)
                Z = Xtr[rngn.choice(len(tr), size=m, replace=False)]
                Kmm_pinv = np.linalg.pinv(K.k_yat(Z, Z, b, e), rcond=1e-10)
                ktr, kva, kte = K.k_yat(Xtr, Z, b, e), K.k_yat(Xva, Z, b, e), K.k_yat(Xte, Z, b, e)
                kZtr = K.k_yat(Z, Xtr, b, e)
                return ktr @ Kmm_pinv @ kZtr, kva @ Kmm_pinv @ kZtr, kte @ Kmm_pinv @ kZtr

            gram_fns = {"gaussian": gram_gaussian, "imq": gram_imq, "poly": gram_poly,
                        "yat": gram_yat, "ray": gram_ray, "nystrom": gram_nys}
            for kn, gfn in gram_fns.items():
                m_test, cfg = tune(gfn, grids[kn])
                res.setdefault(kn, {"test": [], "cfg": []})
                res[kn]["test"].append(m_test); res[kn]["cfg"].append(cfg)
            log(f"  seed {s}: " + "  ".join(f"{kn}={np.mean(res[kn]['test']):.3f}" for kn in gram_fns))
        unit = "RMSE" if task == "reg" else "acc"
        summary = {kn: [float(np.mean(v["test"])), float(np.std(v["test"]))] for kn, v in res.items()}
        out["datasets"][name] = {"task": task, "d": d, "metric": unit, "results": summary,
                                 "best_cfgs": {kn: res[kn]["cfg"] for kn in res}}
        log(f"  {name} [{unit}]: " + "  ".join(f"{kn}={summary[kn][0]:.3f}+-{summary[kn][1]:.3f}" for kn in summary))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "tuned_downstream.json"), "w"), indent=2)
    log("wrote results/tuned_downstream.json")


if __name__ == "__main__":
    main()

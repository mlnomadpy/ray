#!/usr/bin/env python3
"""
Large-scale streaming primal training on HIGGS with MLX (Apple-Silicon GPU).

Proves the paper's practical claim: TensorSketch-RAY trains at million-scale WITHOUT ever
materializing the N x N Gram (or the N x M feature matrix). Features are built per mini-batch
on the GPU and discarded; only the M-dim weights persist. Compares, at matched explicit
feature dimension M, Gaussian RFF / exact RAY / TensorSketch-RAY (+ logistic-regression
baseline) by test AUC, log loss, wall-clock, and peak memory.

Data: HIGGS (UCI), 11M rows, col 0 = label, cols 1..28 = features. Off-sphere preprocessing
(standardize, then clip to the unit ball -> ||x||<=1 with varying norms).

  matched M:  Gaussian RFF -> D=M ;  exact RAY -> D=M/d_b ;  TS-RAY -> D=M/(m+d+1)

Env: /opt/homebrew/bin/python3 (mlx, numpy, pandas, sklearn). Data: ~/higgs_data/HIGGS.csv.gz.
Run: /opt/homebrew/bin/python3 higgs_scaling.py --n-train 200000 --n-test 50000 --M 4096

REPRODUCIBILITY (results integrated into main.tex Table tab:higgs, sec:exp_higgs):
  /opt/homebrew/bin/python3 higgs_scaling.py --n-train 10500000 --n-test 500000 \
      --Ms 512 1024 2048 4096 8192 --epochs 2
  -> results/higgs_scaling_full.json ; 287s total on M5 Pro GPU, peak 8.5GB flat.
  eps(median sq dist)=0.127, d_b=435. Test AUC (gauss / ray(D) / tsray(D)):
    M=512  0.719 / 0.659(1)  / 0.699(3)
    M=1024 0.737 / 0.698(2)  / 0.719(6)
    M=2048 0.749 / 0.732(4)  / 0.734(13)
    M=4096 0.755 / 0.741(9)  / 0.743(26)
    M=8192 0.760 / 0.736(18) / 0.749(52)
  linear baseline AUC 0.682. Headline: memory-flat million-scale streaming primal;
  TS-RAY > exact RAY at matched M (radial starvation); Gaussian leads (HIGGS lacks coupling).
"""
import argparse, time, os, json, resource
import numpy as np
import mlx.core as mx

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
HERE = os.path.dirname(__file__)
PATH = os.path.expanduser("~/higgs_data/HIGGS.csv.gz")
SQ2 = float(np.sqrt(2.0))


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9   # bytes on macOS


# ---------------------------------------------------------------- data ----------
def load(n_train, n_test):
    import pandas as pd
    n = n_train + n_test
    log(f"reading {n:,} rows from HIGGS.csv.gz ...")
    df = pd.read_csv(PATH, header=None, nrows=n, dtype=np.float32)
    y = df.iloc[:, 0].to_numpy(np.float32)
    X = df.iloc[:, 1:29].to_numpy(np.float32)
    mu, sd = X[:n_train].mean(0), X[:n_train].std(0) + 1e-6
    X = (X - mu) / sd
    nrm_tr = np.linalg.norm(X[:n_train], axis=1)
    scale = np.percentile(nrm_tr, 99.9)                # robust to outliers; ||x|| mostly in (0,1], VARYING
    X = X / (scale + 1e-9)
    log(f"loaded; ||x|| in [{np.linalg.norm(X,axis=1).min():.2f},{np.linalg.norm(X,axis=1).max():.2f}] (off-sphere)")
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


# --------------------------------------------------- feature builders (MLX) -----
def make_params(method, d, M, eps, gamma, b, m_sketch, seed):
    rng = np.random.default_rng(seed)
    d_b = d * (d + 1) // 2 + d + 1
    p = {"method": method, "d": d, "eps": eps, "b": b, "d_b": d_b}
    if method == "ray":                                  # off-diagonal index only needed by the exact polynomial
        iu = np.triu_indices(d, 1)
        p["iu0"] = mx.array(iu[0].astype(np.int32)); p["iu1"] = mx.array(iu[1].astype(np.int32))
    if method == "linear":
        p["M"] = d
    elif method == "gauss":
        p["W"] = mx.array((rng.normal(size=(d, M)) * np.sqrt(2 * gamma)).astype(np.float32))
        p["beta"] = mx.array(rng.uniform(0, 2 * np.pi, M).astype(np.float32))
        p["M"] = M
    elif method in ("ray", "tsray"):
        if method == "ray":
            dim = d_b; D = max(1, M // d_b)
        else:
            dim = m_sketch + d + 1; D = max(1, M // dim)
            # two count-sketch one-hot selection matrices (d x m) + signs
            for k in (1, 2):
                h = rng.integers(0, m_sketch, d); S = np.zeros((d, m_sketch), np.float32)
                S[np.arange(d), h] = 1.0
                p[f"S{k}"] = mx.array(S); p[f"s{k}"] = mx.array((rng.integers(0, 2, d) * 2 - 1).astype(np.float32))
            p["m_sketch"] = m_sketch
        t = rng.exponential(1.0 / eps, size=D)
        p["Om"] = mx.array((rng.normal(size=(d, D)) * np.sqrt(2 * t)[None, :]).astype(np.float32))
        p["beta"] = mx.array(rng.uniform(0, 2 * np.pi, D).astype(np.float32))
        p["D"] = D; p["dim"] = dim; p["M"] = D * dim
    return p


def poly_exact(Xb, p):
    b = p["b"]
    diag = Xb * Xb
    off = SQ2 * (mx.take(Xb, p["iu0"], axis=1) * mx.take(Xb, p["iu1"], axis=1))
    lin = float(np.sqrt(2 * b)) * Xb
    const = mx.full((Xb.shape[0], 1), b, dtype=mx.float32)
    return mx.concatenate([diag, off, lin, const], axis=1) * float(1.0 / np.sqrt(p["eps"]))


def poly_sketch(Xb, p):
    m = p["m_sketch"]; b = p["b"]
    C1 = (Xb * p["s1"]) @ p["S1"]; C2 = (Xb * p["s2"]) @ p["S2"]
    TS2 = mx.fft.irfft(mx.fft.rfft(C1, axis=1) * mx.fft.rfft(C2, axis=1), n=m, axis=1)
    lin = float(np.sqrt(2 * b)) * Xb
    const = mx.full((Xb.shape[0], 1), b, dtype=mx.float32)
    return mx.concatenate([TS2, lin, const], axis=1) * float(1.0 / np.sqrt(p["eps"]))


def build(Xnp, p):
    Xb = mx.array(Xnp)
    if p["method"] == "linear":
        return Xb
    if p["method"] == "gauss":
        return float(np.sqrt(2.0 / p["M"])) * mx.cos(Xb @ p["W"] + p["beta"])
    P = poly_exact(Xb, p) if p["method"] == "ray" else poly_sketch(Xb, p)   # (n, dim)
    cos = SQ2 * mx.cos(Xb @ p["Om"] + p["beta"])                            # (n, D)
    Z = (cos[:, :, None] * P[:, None, :]).reshape(Xb.shape[0], p["D"] * p["dim"]) * float(1.0 / np.sqrt(p["D"]))
    return Z


# ------------------------------------------------- streaming logistic SGD -------
def train(Xtr, ytr, Xte, yte, p, epochs, bs, lr, lam):
    M = p["M"]
    w = mx.zeros((M,)); bias = mx.zeros((1,))
    mw = mx.zeros((M,)); vw = mx.zeros((M,)); mb = mx.zeros((1,)); vb = mx.zeros((1,))
    b1, b2, adeps = 0.9, 0.999, 1e-8
    step = 0
    t_build = 0.0
    n = Xtr.shape[0]
    for ep in range(epochs):
        order = np.random.default_rng(ep).permutation(n)
        for i in range(0, n, bs):
            idx = order[i:i + bs]
            tb = time.time()
            Z = build(Xtr[idx], p); mx.eval(Z); t_build += time.time() - tb
            yb = mx.array(ytr[idx])
            pr = mx.sigmoid(Z @ w + bias)
            resid = pr - yb
            gw = Z.T @ resid / len(idx) + lam * w
            gb = resid.mean()[None]
            step += 1
            mw = b1 * mw + (1 - b1) * gw; vw = b2 * vw + (1 - b2) * gw * gw
            mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb * gb
            bc1, bc2 = 1 - b1 ** step, 1 - b2 ** step
            w = w - lr * (mw / bc1) / (mx.sqrt(vw / bc2) + adeps)
            bias = bias - lr * (mb / bc1) / (mx.sqrt(vb / bc2) + adeps)
            mx.eval(w, bias)
    # evaluate
    logits = []
    for i in range(0, Xte.shape[0], 20000):
        Z = build(Xte[i:i + 20000], p)
        logits.append(np.array(Z @ w + bias))
    logits = np.concatenate(logits)
    from sklearn.metrics import roc_auc_score, log_loss
    prob = 1.0 / (1.0 + np.exp(-logits))
    auc = float(roc_auc_score(yte, prob)); ll = float(log_loss(yte, np.clip(prob, 1e-7, 1 - 1e-7)))
    return auc, ll, t_build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=200000)
    ap.add_argument("--n-test", type=int, default=50000)
    ap.add_argument("--M", type=int, default=4096)
    ap.add_argument("--Ms", type=int, nargs="+", default=None)   # budget sweep; overrides --M
    ap.add_argument("--m-sketch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--lam", type=float, default=1e-5)
    ap.add_argument("--methods", nargs="+", default=["linear", "gauss", "ray", "tsray"])
    ap.add_argument("--out", default=os.path.join(HERE, "results", "higgs_scaling.json"))
    args = ap.parse_args()
    log(f"config: {vars(args)} | device {mx.default_device()}")
    Xtr, ytr, Xte, yte = load(args.n_train, args.n_test)
    d = Xtr.shape[1]; eps = 1.0; gamma = 1.0 / eps; b = 1.0
    eps = float(np.median(np.sum((Xtr[:2000][:, None] - Xtr[:2000][None]) ** 2, -1)[np.triu_indices(2000, 1)]))
    gamma = 1.0 / eps
    log(f"eps (median sq dist) = {eps:.3f}; d_b={d*(d+1)//2+d+1}")
    out = {"config": vars(args), "eps": eps, "rows": []}
    Ms = args.Ms if args.Ms else [args.M]
    # (method, target M) jobs: linear once at M=d; feature methods at each budget
    jobs = []
    if "linear" in args.methods:
        jobs.append(("linear", d))
    for M in Ms:
        for method in [m for m in args.methods if m != "linear"]:
            jobs.append((method, M))
    for method, Mtarget in jobs:
        p = make_params(method, d, Mtarget, eps, gamma, b, args.m_sketch, 0)
        t0 = time.time()
        auc, ll, tb = train(Xtr, ytr, Xte, yte, p, args.epochs, args.bs, args.lr, args.lam)
        wall = time.time() - t0
        row = {"method": method, "M_target": Mtarget, "M_actual": p["M"], "D": p.get("D"),
               "auc": auc, "log_loss": ll, "train_wall_s": wall, "build_wall_s": tb,
               "mem_repr_mb": float(Xtr.shape[0] * p["M"] * 4 / 1e6), "peak_gb": peak_gb()}
        out["rows"].append(row)
        log(f"  {method:6s} M={p['M']:6d} D={str(p.get('D')):>4}  AUC={auc:.4f}  logloss={ll:.4f}  "
            f"wall={wall:.1f}s (build {tb:.1f}s)  peak={peak_gb():.1f}GB")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()

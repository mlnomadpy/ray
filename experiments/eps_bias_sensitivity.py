#!/usr/bin/env python3
"""
Hyperparameter sensitivity of RAY to the bias b and the radial scale eps (reviewer R20 #7).

For a grid b in {0, 0.1, 1, 10} x eps in {0.25, 1, 4} x (median squared distance), we measure the
relative Frobenius Gram-approximation error of the flat (D'=1) estimator at fixed D, for three
variants: exact RAY, normalized RAY, and TensorSketch-RAY. Each variant is compared to ITS OWN
exact kernel (normalized RAY targets the rescaled kernel q_b * h, not k_yat,b). Off-sphere
synthetic data (varying norms), numpy, mean over seeds.

Kernel: k(x,w) = (x.w+b)^2 / (||x-w||^2 + eps). Radial RFF: t~Exp(rate eps), omega~N(0,2t),
phi=sqrt(2/D)cos(omega x+beta); E[<phi(x),phi(w)>] = eps*h, so scale the polynomial by 1/sqrt(eps).

Env: /opt/homebrew/bin/python3 (numpy). Run: eps_bias_sensitivity.py
REPRODUCIBILITY (n=300, d=16, D=200, m=128, 5 seeds; median sq dist 1.713; table in appendix).
  rel. Frobenius Gram error (exact / norm / TS), per (b, eps/median):
    b=0    0.25x 0.095/0.114/0.256   1x 0.081/0.083/0.384   4x 0.063/0.062/0.547
    b=0.1  0.25x 0.092/0.107/0.231   1x 0.077/0.073/0.337   4x 0.058/0.057/0.481
    b=1    0.25x 0.190/0.165/0.201   1x 0.102/0.086/0.123   4x 0.065/0.061/0.106
    b=10   0.25x 0.251/0.235/0.251   1x 0.111/0.107/0.111   4x 0.069/0.067/0.069
  Per-cell std over 5 seeds: exact/normalized <=0.023, TS <=0.046.
  Headline: exact and normalized RAY stay in a tight 0.06-0.25 band across 2 orders of b and
  16x in eps (insensitive); TS-RAY adds sketch error, largest at small b/eps where the quadratic
  term dominates. eps=median (1x) is a sound default.
"""
import argparse, os, json, time
import numpy as np

T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)
HERE = os.path.dirname(__file__)


def data(n, d, seed):
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n, d)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    r = rng.uniform(0.3, 1.5, size=(n, 1))
    return U * r


def exact_gram(X, b, eps, normalized):
    S = X @ X.T; nn = (X * X).sum(1)
    D2 = np.maximum(nn[:, None] + nn[None, :] - 2 * S, 0.0)
    P = (S + b) ** 2
    if normalized:                                        # q_b inner product: divide by (||xi||^2+b)(||xj||^2+b)
        P = P / ((nn[:, None] + b) * (nn[None, :] + b))
    return P / (D2 + eps)


def poly_part(X, b, m, variant, seed):
    """Exact polynomial feature, or its degree-2 TensorSketch (sketch RAW x, append linear+const)."""
    n, d = X.shape
    if variant == "ts":
        rng = np.random.default_rng(seed); C = []
        for k in range(2):                                # two count-sketches of RAW x -> convolution ~ (x.w)^2
            h = rng.integers(0, m, d); s = rng.integers(0, 2, d) * 2 - 1
            Ck = np.zeros((n, m)); np.add.at(Ck.T, h, (X * s).T); C.append(Ck)
        TS2 = np.fft.irfft(np.fft.rfft(C[0], axis=1) * np.fft.rfft(C[1], axis=1), n=m, axis=1)
        quad = TS2
    else:
        iu = np.triu_indices(d, 1)
        quad = np.concatenate([X * X, np.sqrt(2.0) * (X[:, iu[0]] * X[:, iu[1]])], axis=1)
    lin = np.sqrt(2 * b) * X if b > 0 else np.zeros((n, 0))
    const = np.full((n, 1), b) if b > 0 else np.zeros((n, 0))
    P = np.concatenate([quad, lin, const], axis=1)
    if variant == "norm":                                 # q_b = p_b / (||x||^2+b), unit-norm feature
        P = P / ((X * X).sum(1, keepdims=True) + b)
    return P


def ray_gram(X, b, eps, D, variant, seed, m=128):
    rng = np.random.default_rng(seed + 7); n, d = X.shape
    P = poly_part(X, b, m, variant, seed) / np.sqrt(eps)
    t = rng.exponential(1.0 / eps, size=D)
    Om = rng.normal(size=(d, D)) * np.sqrt(2 * t)[None, :]
    beta = rng.uniform(0, 2 * np.pi, D)
    cos = np.sqrt(2.0) * np.cos(X @ Om + beta)            # (n,D)
    Z = (cos[:, :, None] * P[:, None, :]).reshape(n, D * P.shape[1]) / np.sqrt(D)
    return Z @ Z.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--D", type=int, default=200); ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "eps_bias_sensitivity.json"))
    args = ap.parse_args()
    bs = [0.0, 0.1, 1.0, 10.0]; eps_facs = [0.25, 1.0, 4.0]
    X0 = data(args.n, args.d, 0)
    S = X0 @ X0.T; nn = (X0 * X0).sum(1); D2 = nn[:, None] + nn[None, :] - 2 * S
    med = float(np.median(D2[np.triu_indices(args.n, 1)]))
    log(f"median sq dist = {med:.3f}; grid b={bs} eps_fac={eps_facs}")
    rows = []
    for b in bs:
        for ef in eps_facs:
            eps = ef * med
            errs = {v: [] for v in ("exact", "norm", "ts")}
            for sd in range(args.seeds):
                X = data(args.n, args.d, sd)
                for v in ("exact", "norm", "ts"):
                    K = exact_gram(X, b, eps, normalized=(v == "norm"))
                    Kh = ray_gram(X, b, eps, args.D, v, sd, args.m)
                    errs[v].append(np.linalg.norm(Kh - K) / np.linalg.norm(K))
            row = {"b": b, "eps_fac": ef, "eps": eps,
                   **{v: float(np.mean(errs[v])) for v in errs},
                   **{v + "_std": float(np.std(errs[v])) for v in errs}}
            rows.append(row)
            log(f"  b={b:<4} eps={ef}x  exact={row['exact']:.3f}+-{row['exact_std']:.3f}  "
                f"norm={row['norm']:.3f}+-{row['norm_std']:.3f}  ts={row['ts']:.3f}+-{row['ts_std']:.3f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"config": vars(args), "median": med, "rows": rows}, open(args.out, "w"), indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()

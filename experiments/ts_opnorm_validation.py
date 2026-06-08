#!/usr/bin/env python3
"""
Validate Theorem thm:ts_opnorm: the operator-norm error of DOUBLY-randomized RAY
(sketched modulation P_hat_m  o  radial RFF R_hat_D) splits, by conditioning on the
sketch, into a radial term that decays as O(D^-1/2) and a sketch term ~ eta||P||_op that
is independent of D and shrinks with m; m->inf (exact modulation) recovers the exact-RAY
operator-norm error of Theorem thm:bernstein.

Conventions match the paper: P = [(x_i.x_j+b)^2/eps], R = unit-diagonal radial Gram
[eps/(eps+||x_i-x_j||^2)], K = P o R (Schur/elementwise). The radial RFF gives
R_hat_D with E[R_hat_D]=R; the degree-2 TensorSketch of the augmented feature gives
P_hat_m with E[P_hat_m]=P. K_S = P_hat_m o R is the conditional mean given the sketch.

Checks: (a) ||K_hat - K_S||_op ~ D^-1/2 (radial);  (b) ||K_S - K||_op = ||E_P o R||_op
independent of D, <= ||E_P||_op (R unit-diagonal), decaying with m;  (c) the OSE
||E_P||_op <= eta||P||_op holds;  (d) m->inf recovers exact-RAY ||K_hat - K||_op.

Env: /opt/homebrew/bin/python3 (numpy). Run: ts_opnorm_validation.py
REPRODUCIBILITY: results/ts_opnorm_validation.json; backs thm:ts_opnorm (sec:exp_ts).
  off-sphere d=16, n=300, eps=1.71, ||P||_op=186, ||K||_op=102, 5 seeds:
  sketch error vs m: eta=||E_P||/||P|| 0.23/0.17/0.09/0.10 ; ||E_P o R|| 19.5/14.1/7.4/7.8 (m=64/128/256/512)
  vs D at m=128: radial 40.5/18.4/10.3/7.1/3.6 (D=10/30/100/300/1000) ~ D^-1/2; sketch bias 14.06 (const in D);
    total 45.9/23.1/17.4/16.1/14.4 (<= radial+sketch, -> sketch floor as D grows).
  m->inf (exact modulation): sketch=0, total=radial 39.3->3.5 -> recovers exact-RAY (thm:bernstein).
  Confirms: radial O(D^-1/2), sketch ~eta||P|| independent of D and decaying with m, ||E_P o R||<=||E_P||.
"""
import argparse, os, json, time
import numpy as np

T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)
HERE = os.path.dirname(__file__)
SQ2 = float(np.sqrt(2.0))
def opn(A): return float(np.linalg.norm(A, 2))


def data(n, d, seed):
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n, d)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    return U * rng.uniform(0.3, 1.5, size=(n, 1))


def sqd(X):
    nn = (X * X).sum(1); return np.maximum(nn[:, None] + nn[None, :] - 2 * X @ X.T, 0.0)


def exact_P(X, b, eps): return (X @ X.T + b) ** 2 / eps
def exact_R(X, eps):    return eps / (eps + sqd(X))


def sketch_feat(X, b, m, seed):                      # degree-2 TensorSketch of augmented (x,sqrt b), /sqrt eps applied by caller
    rng = np.random.default_rng(seed); n, d = X.shape
    Xa = np.concatenate([X, np.full((n, 1), np.sqrt(b))], axis=1); da = d + 1
    C = []
    for _ in range(2):
        h = rng.integers(0, m, da); s = rng.integers(0, 2, da) * 2 - 1
        Ck = np.zeros((n, m)); np.add.at(Ck.T, h, (Xa * s).T); C.append(Ck)
    return np.fft.irfft(np.fft.rfft(C[0], axis=1) * np.fft.rfft(C[1], axis=1), n=m, axis=1)


def radial_Rhat(X, eps, D, seed):
    rng = np.random.default_rng(seed); d = X.shape[1]
    t = rng.exponential(1.0 / eps, size=D)
    Om = rng.normal(size=(d, D)) * np.sqrt(2 * t)[None, :]
    cos = SQ2 * np.cos(X @ Om + rng.uniform(0, 2 * np.pi, D))     # (n,D)
    return (cos @ cos.T) / D                                       # E = R (unit diagonal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--b", type=float, default=1.0); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--Ds", type=int, nargs="+", default=[10, 30, 100, 300, 1000])
    ap.add_argument("--ms", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--out", default=os.path.join(HERE, "results", "ts_opnorm_validation.json"))
    args = ap.parse_args()
    X = data(args.n, args.d, 0)
    eps = float(np.median(sqd(X)[np.triu_indices(args.n, 1)]))
    P, R = exact_P(X, args.b, eps), exact_R(X, eps)
    K = P * R; Pop = opn(P)
    log(f"n={args.n} d={args.d} eps={eps:.3f}  ||P||_op={Pop:.2f}  ||K||_op={opn(K):.2f}")
    rows = []
    # (b)/(c): sketch error vs m (independent of D)
    for m in args.ms:
        eP, sb = [], []
        for s in range(args.seeds):
            Phi = sketch_feat(X, args.b, m, 10 + s) / np.sqrt(eps)
            Ph = Phi @ Phi.T; EP = Ph - P
            eP.append(opn(EP) / Pop); sb.append(opn(EP * R))
        log(f"  m={m:4d}: eta_emp(||E_P||/||P||)={np.mean(eP):.3f}  sketch_bias||E_P o R||={np.mean(sb):.3f}")
        rows.append({"kind": "sketch", "m": m, "eta_emp": float(np.mean(eP)), "sketch_bias": float(np.mean(sb))})
    # (a)/(d): total / radial / sketch vs D, at a fixed m and at m=inf (exact modulation)
    for m in [128, None]:
        for D in args.Ds:
            tot, rad, skb = [], [], []
            for s in range(args.seeds):
                Ph = P if m is None else (sketch_feat(X, args.b, m, 10 + s) / np.sqrt(eps)) @ (sketch_feat(X, args.b, m, 10 + s) / np.sqrt(eps)).T
                Rh = radial_Rhat(X, eps, D, 700 + s)
                Kh = Ph * Rh; KS = Ph * R
                tot.append(opn(Kh - K)); rad.append(opn(Kh - KS)); skb.append(opn(KS - K))
            tag = "exactmod" if m is None else f"m={m}"
            log(f"  {tag:8s} D={D:5d}: total={np.mean(tot):.3f} radial={np.mean(rad):.3f} sketch={np.mean(skb):.3f}")
            rows.append({"kind": "vsD", "m": m or "inf", "D": D, "total": float(np.mean(tot)),
                         "radial": float(np.mean(rad)), "sketch": float(np.mean(skb))})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"eps": eps, "Pop": Pop, "rows": rows}, open(args.out, "w"), indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()

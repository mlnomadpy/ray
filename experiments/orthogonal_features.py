#!/usr/bin/env python3
"""
Orthogonal inner features in d>1 (#T-E4).

Section 4 claims orthogonal random features (ORF) reduce the inner Gaussian RFF
variance. In d=1 (Section 5.2) the antithetic version did nothing -- there is no
direction to orthogonalize. Here we test the genuine multi-dimensional case: for a
single scale t, estimate the Gaussian factor g_t(x,w) with D'=d inner frequencies,
drawn either i.i.d. N(0,2tI) or as a scaled Haar-orthogonal block (Yu et al. 2016:
omega_i = sqrt(2t) * s_i * q_i, q_i orthonormal rows of a Haar matrix, s_i ~ chi_d,
which preserves the N(0,2tI) marginal). We report Var_ORF / Var_iid at d in {5,20}.

------------------------------------------------------------------------------
REPRODUCIBILITY
    Env  : python3>=3.9, numpy>=1.24 (CPU). Run: ~/.pixi/envs/jax/bin/python3 orthogonal_features.py
    Out  : results/orthogonal_features.json (+ stdout). Wall: ~1 min. Deterministic seeds.
------------------------------------------------------------------------------
"""
import argparse, json, os, time
import numpy as np

T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def gauss_iid(x, w, t, Dp, rng):
    d = x.shape[0]
    W = rng.normal(size=(d, Dp)) * np.sqrt(2.0 * t)
    beta = rng.uniform(0.0, 2.0 * np.pi, size=Dp)
    zx = np.sqrt(2.0 / Dp) * np.cos(x @ W + beta)
    zw = np.sqrt(2.0 / Dp) * np.cos(w @ W + beta)
    return float(zx @ zw)


def gauss_orf(x, w, t, Dp, rng):
    """Scaled Haar-orthogonal frequencies (Dp <= d). Preserves N(0,2tI) marginal."""
    d = x.shape[0]
    G = rng.normal(size=(d, d))
    Q, _ = np.linalg.qr(G)                     # Haar-orthonormal columns
    s = np.linalg.norm(rng.normal(size=(d, d)), axis=0)   # chi_d scalings
    Omega = (Q * s) * np.sqrt(2.0 * t)         # columns: omega_i = sqrt(2t) s_i q_i
    W = Omega[:, :Dp]
    beta = rng.uniform(0.0, 2.0 * np.pi, size=Dp)
    zx = np.sqrt(2.0 / Dp) * np.cos(x @ W + beta)
    zw = np.sqrt(2.0 / Dp) * np.cos(w @ W + beta)
    return float(zx @ zw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", type=int, nargs="+", default=[5, 20])
    ap.add_argument("--rho", type=float, default=0.3)     # x^T w for unit vectors
    ap.add_argument("--t", type=float, default=1.0)       # fixed scale
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "results", "orthogonal_features.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    log(f"config: {vars(args)}")

    out = {"config": vars(args), "results": {}}
    for d in args.ds:
        Dp = d                                            # one orthogonal block
        x = np.zeros(d); x[0] = 1.0
        w = np.zeros(d); w[0] = args.rho; w[1] = np.sqrt(1.0 - args.rho ** 2)
        g_true = np.exp(-args.t * float(np.sum((x - w) ** 2)))
        iid = [gauss_iid(x, w, args.t, Dp, np.random.default_rng(13 * r + d)) for r in range(args.reps)]
        orf = [gauss_orf(x, w, args.t, Dp, np.random.default_rng(13 * r + d + 7)) for r in range(args.reps)]
        v_iid, v_orf = float(np.var(iid)), float(np.var(orf))
        out["results"][str(d)] = {"Dp": Dp, "g_true": g_true,
                                  "var_iid": v_iid, "var_orf": v_orf,
                                  "ratio_orf_over_iid": v_orf / v_iid,
                                  "bias_iid": float(np.mean(iid)) - g_true,
                                  "bias_orf": float(np.mean(orf)) - g_true}
        log(f"  d={d:3d} D'={Dp}  var_iid={v_iid:.3e}  var_orf={v_orf:.3e}  "
            f"ratio={v_orf/v_iid:.3f}  (bias iid={np.mean(iid)-g_true:+.4f}, orf={np.mean(orf)-g_true:+.4f})")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()

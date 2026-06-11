#!/usr/bin/env python3
"""
Complex vs real degree-2 TensorSketch for the modulation (tests rmk:complex / Wacker et al. 2024).

The modulation randomizer sketches the quadratic term (x.w)^2 (the deployed quadratic-only
variant keeps 2b x.w + b^2 exact). Wacker, Kanagawa & Filippone show complex-valued
polynomial sketches have strictly smaller variance than real Rademacher sketches because
the unit-modulus fourth moment is smaller. We substitute complex signs s in {1,i,-1,-i}
for the Rademacher signs in the degree-2 TensorSketch and use Re(<TS(x), conj TS(w)>).

Checks, on the fig_ts_opnorm setup (off-sphere, d=16, N=300, b=1, eps=1):
 (0) unbiasedness of the complex sketch (mean over sketches -> (x.w)^2);
 (1) eta = ||Phat - P||_op / ||P||_op, real vs complex, m in {64,128,256,512};
 (2) the sketch term ||(Phat-P) o R||_op of thm:ts_opnorm, real vs complex;
 (3) pointwise variance ratio complex/real of the quadratic-term estimate.

------------------------------------------------------------------------------
REPRODUCIBILITY
    Env : python3>=3.9, numpy. CPU. Run: ~/.pixi/envs/jax/bin/python3 complex_sketch.py
    Out : results/complex_sketch.json (+ stdout). Deterministic seeds.
------------------------------------------------------------------------------
"""
import json, os, time
import numpy as np

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
HERE = os.path.dirname(__file__)


def sqd(X):
    nn = (X * X).sum(1); return np.maximum(nn[:, None] + nn[None, :] - 2 * X @ X.T, 0.0)


def ts_quadratic(X, m, rng, complex_signs=False):
    """Degree-2 TensorSketch features of X for (x.w)^2. Returns (N,m) real or complex."""
    n, d = X.shape
    C = []
    for _ in range(2):
        h = rng.integers(0, m, d)
        if complex_signs:
            s = np.exp(1j * (np.pi / 2.0) * rng.integers(0, 4, d))   # {1,i,-1,-i}
        else:
            s = (rng.integers(0, 2, d) * 2 - 1).astype(float)
        Ck = np.zeros((n, m), dtype=(complex if complex_signs else float))
        np.add.at(Ck.T, h, (X * s).T)
        C.append(Ck)
    return np.fft.ifft(np.fft.fft(C[0], axis=1) * np.fft.fft(C[1], axis=1), axis=1) \
        if complex_signs else \
        np.fft.irfft(np.fft.rfft(C[0], axis=1) * np.fft.rfft(C[1], axis=1), n=m, axis=1)


def phat_quad(X, m, rng, complex_signs):
    Z = ts_quadratic(X, m, rng, complex_signs)
    G = Z @ np.conj(Z).T
    return np.real(G)


def opn(A): return float(np.linalg.norm((A + A.T) / 2, 2))


def main():
    rng = np.random.default_rng(0)
    N, d, b, eps = 300, 16, 1.0, 1.0
    U = rng.normal(size=(N, d)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    X = U * rng.uniform(0.3, 1.5, size=(N, 1))
    Q = (X @ X.T) ** 2                              # quadratic part, the sketched object
    P = ((X @ X.T + b) ** 2) / eps                  # full modulation Gram
    R = eps / (eps + sqd(X))
    lin = (2 * b * (X @ X.T) + b ** 2) / eps        # kept exact in the deployed variant
    out = {"config": {"N": N, "d": d, "b": b, "eps": eps}}

    # (0) unbiasedness of the complex sketch on a few pairs
    pairs = [(0, 1), (2, 3), (10, 200)]
    reps = 400
    bias = []
    for (i, j) in pairs:
        est = np.mean([phat_quad(X[[i, j]], 64, np.random.default_rng(s), True)[0, 1]
                       for s in range(reps)])
        bias.append({"pair": [i, j], "mean_est": float(est), "true": float(Q[i, j]),
                     "rel_err": float(abs(est - Q[i, j]) / (abs(Q[i, j]) + 1e-12))})
    out["unbiasedness_complex"] = bias
    log("(0) complex-sketch unbiasedness rel errs: "
        + ", ".join(f"{r['rel_err']:.3f}" for r in bias))

    # (1)+(2) eta and sketch term vs m, real vs complex
    SEEDS = 10
    rows = []
    log(f"  {'m':>5} | {'eta_real':>9} {'eta_cplx':>9} | {'term_real':>10} {'term_cplx':>10} | {'var ratio':>9}")
    for m in [64, 128, 256, 512]:
        res = {k: [] for k in ["eta_r", "eta_c", "term_r", "term_c"]}
        var_pt = {"r": [], "c": []}
        for s in range(SEEDS):
            for tag, cs in [("r", False), ("c", True)]:
                Ph = (phat_quad(X, m, np.random.default_rng(50 * s + (1 if cs else 0)), cs)) / eps + lin
                E = Ph - P
                res["eta_" + tag].append(opn(E) / opn(P))
                res["term_" + tag].append(opn(E * R))
        # (3) pointwise variance of the quadratic estimate on one pair, many sketches
        i, j = 5, 17
        for tag, cs in [("r", False), ("c", True)]:
            ests = [phat_quad(X[[i, j]], m, np.random.default_rng(1000 + s), cs)[0, 1]
                    for s in range(300)]
            var_pt[tag] = float(np.var(ests))
        row = {"m": m,
               "eta_real": float(np.mean(res["eta_r"])), "eta_complex": float(np.mean(res["eta_c"])),
               "term_real": float(np.mean(res["term_r"])), "term_complex": float(np.mean(res["term_c"])),
               "var_real_pt": var_pt["r"], "var_complex_pt": var_pt["c"],
               "var_ratio_cplx_over_real": var_pt["c"] / max(var_pt["r"], 1e-30)}
        rows.append(row)
        log(f"  {m:>5} | {row['eta_real']:>9.4f} {row['eta_complex']:>9.4f} | "
            f"{row['term_real']:>10.3f} {row['term_complex']:>10.3f} | "
            f"{row['var_ratio_cplx_over_real']:>9.3f}")
    out["rows"] = rows

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "complex_sketch.json"), "w") as f:
        json.dump(out, f, indent=1)
    log("wrote results/complex_sketch.json")


if __name__ == "__main__":
    main()

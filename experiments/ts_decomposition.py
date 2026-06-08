#!/usr/bin/env python3
"""
Validate the TensorSketch-RAY two-error decomposition (prop:ts_variance).

khat_{D,m} = phat_m * hhat_D  (independent), with
  phat_m = TS2(x).TS2(w) + 2b x.w + b^2   (unbiased for p=(x.w+b)^2)
  hhat_D = (1/(eps D)) sum 2cos cos, t~Exp(eps)   (unbiased for h=1/(r+eps))
Theorem:  Var[khat] = p^2 Var[hhat_D] + h^2 Var[phat_m] + Var[phat_m] Var[hhat_D].
We show the two error sources separate: vary D at fixed m (radial term ~1/D dominates),
vary m at fixed D (sketch term ~1/m dominates), and check the empirical Var matches the
3-term formula.

Env: python3, numpy. Run: ~/.pixi/envs/jax/bin/python3 ts_decomposition.py  -> results/ts_decomposition.json
"""
import json, os, time
import numpy as np
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
HERE = os.path.dirname(__file__)


def count_sketch(x, m, h, s):
    c = np.zeros(m)
    for i in range(len(x)):
        c[h[i]] += s[i] * x[i]
    return c


def ts2(x, m, rng):
    d = len(x)
    h1 = rng.integers(0, m, d); s1 = rng.integers(0, 2, d) * 2 - 1
    h2 = rng.integers(0, m, d); s2 = rng.integers(0, 2, d) * 2 - 1
    c1 = count_sketch(x, m, h1, s1); c2 = count_sketch(x, m, h2, s2)
    return np.fft.irfft(np.fft.rfft(c1) * np.fft.rfft(c2), n=m)


def phat(x, w, b, m, rng):
    return ts2(x, m, rng) @ ts2(w, m, rng) + 2 * b * (x @ w) + b ** 2


def hhat(x, w, eps, D, rng):
    d = len(x); acc = 0.0
    for t in rng.exponential(1.0 / eps, size=D):
        om = rng.normal(size=d) * np.sqrt(2 * t); beta = rng.uniform(0, 2 * np.pi)
        acc += 2 * np.cos(om @ x + beta) * np.cos(om @ w + beta)
    return acc / (eps * D)


def var_p(x, w, b, m, reps, rng):
    return float(np.var([phat(x, w, b, m, rng) for _ in range(reps)]))


def var_h(x, w, eps, D, reps, rng):
    return float(np.var([hhat(x, w, eps, D, rng) for _ in range(reps)]))


def var_k(x, w, b, eps, D, m, reps, rng):
    return float(np.var([phat(x, w, b, m, rng) * hhat(x, w, eps, D, rng) for _ in range(reps)]))


def main():
    eps, b, d = 1.0, 1.0, 16
    rng = np.random.default_rng(3)
    x = rng.normal(size=d); x *= 0.8 / np.linalg.norm(x)
    w = rng.normal(size=d); w *= 0.9 / np.linalg.norm(w)
    p = (x @ w + b) ** 2; r = float(np.sum((x - w) ** 2)); h = 1.0 / (r + eps)
    reps = 4000
    out = {"config": {"eps": eps, "b": b, "d": d, "p": p, "h": h, "r": r, "reps": reps},
           "vary_D": [], "vary_m": []}
    log(f"TS decomposition: p={p:.3f}, h={h:.3f}, r={r:.3f}")
    log("=== vary D at m=256 (radial source) ===")
    m0 = 256
    vp0 = var_p(x, w, b, m0, reps, np.random.default_rng(10))
    for D in [10, 50, 200, 1000]:
        vh = var_h(x, w, eps, D, reps, np.random.default_rng(20 + D))
        vk = var_k(x, w, b, eps, D, m0, reps, np.random.default_rng(30 + D))
        pred = p ** 2 * vh + h ** 2 * vp0 + vp0 * vh
        out["vary_D"].append({"D": D, "var_h": vh, "var_p": vp0, "var_k": vk, "pred": pred})
        log(f"  D={D:4d}: Var_h={vh:.2e} Var_p={vp0:.2e}  Var_k={vk:.2e} pred={pred:.2e} ratio={vk/pred:.2f}")
    log("=== vary m at D=1000 (sketch source) ===")
    D0 = 1000
    vh0 = var_h(x, w, eps, D0, reps, np.random.default_rng(99))
    for m in [64, 128, 512, 2048]:
        vp = var_p(x, w, b, m, reps, np.random.default_rng(40 + m))
        vk = var_k(x, w, b, eps, D0, m, reps, np.random.default_rng(50 + m))
        pred = p ** 2 * vh0 + h ** 2 * vp + vp * vh0
        out["vary_m"].append({"m": m, "var_p": vp, "var_h": vh0, "var_k": vk, "pred": pred})
        log(f"  m={m:4d}: Var_p={vp:.2e} Var_h={vh0:.2e}  Var_k={vk:.2e} pred={pred:.2e} ratio={vk/pred:.2f}")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "ts_decomposition.json"), "w"), indent=2)
    log("wrote results/ts_decomposition.json")


if __name__ == "__main__":
    main()

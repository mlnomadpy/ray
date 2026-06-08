#!/usr/bin/env python3
"""
Feature-construction runtime and memory vs dimension d (reviewer gap #9).

The core computational claim is that sketching the modulation removes the O(d^2)
exact-modulation floor. The HIGGS/scalability results scale N; this scales d directly.
At a fixed radial-draw count D, we measure, for exact modulation vs deployed (sketched) RAY:
  - explicit feature dimension per point (D*d_b vs D*(m+d+1)),
  - feature-build wall-clock,
  - representation memory (N * dim * 8 bytes).
Exact modulation's d_b = d(d+1)/2+d+1 grows as O(d^2) and becomes impossible to build at
large d; the sketch stays linear in d. We cap the exact feature dimension and report N/A
beyond it, exactly as one must in practice.

Env: ~/.pixi/envs/jax/bin/python3 (numpy). Run: runtime_vs_d.py
REPRODUCIBILITY: results/runtime_vs_d.json; backs the runtime-vs-d table (sec:exp_scaling / Limitation v).
"""
import json, os, time
import numpy as np
import cost_matched_bias as CMB        # ray_primal(X,b,eps,D,rng), d_b(d,b)
import ts_ryf_costmatched as TS        # ts_ray_primal(X,b,eps,D,m,seed)
import krr_downstream as K

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
HERE = os.path.dirname(__file__)


def ball(N, d, rng, lo=0.3, hi=1.5):
    U = rng.normal(size=(N, d)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    return U * rng.uniform(lo, hi, size=(N, 1))


def main():
    b, D, m = 1.0, 8, 128
    N = 1000
    ds = [8, 16, 32, 64, 128, 256, 512, 1024]
    dim_cap = 2_000_000          # max explicit feature dim per point we allow exact modulation to build
    out = {"config": {"b": b, "D": D, "m": m, "N": N, "ds": ds, "dim_cap": dim_cap}, "rows": []}
    log(f"D={D} m={m} N={N}  (exact-mod dim=D*d_b grows O(d^2); sketch dim=D*(m+d+1))")
    for d in ds:
        rng = np.random.default_rng(0)
        X = ball(N, d, rng)
        eps = float(np.median(K.sqdist(X[:300], X[:300])[np.triu_indices(300, 1)]))
        d_b = CMB.d_b(d, b)
        exact_dim = D * d_b
        sk_dim = D * (m + d + 1)
        row = {"d": d, "d_b": d_b, "exact_dim": exact_dim, "sketch_dim": sk_dim,
               "exact_mem_mb": N * exact_dim * 8 / 1e6, "sketch_mem_mb": N * sk_dim * 8 / 1e6}
        # sketched RAY build time (always feasible)
        t0 = time.perf_counter()
        Zs = TS.ts_ray_primal(X, b, eps, D, m, 7)
        row["sketch_build_s"] = time.perf_counter() - t0
        # exact-modulation build time (only if under the cap)
        if exact_dim <= dim_cap:
            t0 = time.perf_counter()
            Ze = CMB.ray_primal(X, b, eps, D, np.random.default_rng(7))
            row["exact_build_s"] = time.perf_counter() - t0
            row["exact_feasible"] = True
        else:
            row["exact_build_s"] = None
            row["exact_feasible"] = False
        out["rows"].append(row)
        ex = f"{row['exact_build_s']:.3f}s" if row["exact_feasible"] else "N/A (too large)"
        log(f"  d={d:5d}: d_b={d_b:9d}  exact dim={exact_dim:10d} ({row['exact_mem_mb']:8.1f}MB) build={ex}"
            f"   sketch dim={sk_dim:7d} ({row['sketch_mem_mb']:6.1f}MB) build={row['sketch_build_s']:.3f}s")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "runtime_vs_d.json"), "w"), indent=2)
    log("wrote results/runtime_vs_d.json")


if __name__ == "__main__":
    main()

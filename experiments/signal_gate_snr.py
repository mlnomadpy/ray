#!/usr/bin/env python3
"""
Signal-gate / false-positive suppression (checks the R16 "modulation gates radial noise" idea).

The yat-kernel is alignment-gate x proximity. We test whether the gate suppresses radial
false positives, and whether it survives the RAY (and TensorSketch-RAY) approximation.

Off-sphere PAIR types (varying norms make these geometrically realizable):
  true        : both moderate norm, close AND aligned (high x.w, small ||x-w||)
  radial_dist : both small norm near origin -> close but LOW alignment (small x.w)
  align_dist  : large/small norm same direction -> aligned (moderate x.w) but FAR

We score pairs with: IMQ (radial only), poly2 (alignment only), exact yat, exact RAY,
TensorSketch-RAY; and report AUC for separating `true` from each distractor. A radial-only
kernel should false-positive on radial_dist (low AUC); the yat product should suppress it.

Env: numpy, sklearn. Run: ~/.pixi/envs/jax/bin/python3 signal_gate_snr.py -> results/signal_gate_snr.json
"""
import json, os, time
import numpy as np
from sklearn.metrics import roc_auc_score
import krr_downstream as K
import ts_ryf_costmatched as TS
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
HERE = os.path.dirname(__file__)


def unit(v): return v / (np.linalg.norm(v) + 1e-12)


def gen(M, d, rng):
    A, B, lab = [], [], []
    for _ in range(M):
        u = unit(rng.normal(size=d))
        # true: close + aligned, moderate norm
        x = 0.7 * u; w = 0.7 * unit(u + 0.12 * rng.normal(size=d))
        A.append(x); B.append(w); lab.append("true")
        # radial distractor: near origin -> close but low inner product
        c = unit(rng.normal(size=d))
        x = 0.18 * unit(rng.normal(size=d)); w = 0.18 * unit(rng.normal(size=d))
        A.append(x); B.append(w); lab.append("radial")
        # alignment distractor: same direction, different magnitudes -> aligned but far
        v = unit(rng.normal(size=d))
        A.append(1.6 * v); B.append(0.5 * v); lab.append("align")
    return np.array(A), np.array(B), np.array(lab)


def diag_score(fn):
    return lambda A, B: np.diag(fn(A, B))


def main():
    b, eps, d, M = 1.0, 1.0, 16, 400
    rng = np.random.default_rng(0)
    A, B, lab = gen(M, d, rng)
    scorers = {
        "IMQ": diag_score(lambda A, B: K.k_imq(A, B, eps)),
        "poly2": diag_score(lambda A, B: (A @ B.T + b) ** 2),
        "yat": diag_score(lambda A, B: K.k_yat(A, B, b, eps)),
        "RAY": diag_score(lambda A, B: K.ray_cross(A, B, b, eps, 1000, np.random.default_rng(7))),
        "TS-RAY": diag_score(lambda A, B: TS.ts_ray_primal(A, b, eps, 30, 128, 7) @ TS.ts_ray_primal(B, b, eps, 30, 128, 7).T),
    }
    out = {"config": {"b": b, "eps": eps, "d": d, "M": M}, "auc": {}, "mean_score": {}}
    log(f"signal-gate SNR: d={d}, {M} pairs/type")
    log(f"  {'method':8} {'AUC true vs radial':>18} {'AUC true vs align':>18}")
    def auc_vs(s, distractor):
        mask = (lab == "true") | (lab == distractor)
        return float(roc_auc_score((lab[mask] == "true").astype(int), s[mask]))
    for name, sc in scorers.items():
        s = sc(A, B)
        auc_r = auc_vs(s, "radial"); auc_a = auc_vs(s, "align")
        out["auc"][name] = {"true_vs_radial": auc_r, "true_vs_align": auc_a}
        out["mean_score"][name] = {t: float(np.mean(s[lab == t])) for t in ["true", "radial", "align"]}
        log(f"  {name:8} {auc_r:>18.3f} {auc_a:>18.3f}   means(t/r/a)={out['mean_score'][name]['true']:.3f}/{out['mean_score'][name]['radial']:.3f}/{out['mean_score'][name]['align']:.3f}")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "signal_gate_snr.json"), "w"), indent=2)
    log("wrote results/signal_gate_snr.json")


if __name__ == "__main__":
    main()

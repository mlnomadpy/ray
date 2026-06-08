#!/usr/bin/env python3
"""
RAY as a linear-time, streaming yat-attention primitive: approximation-quality + scaling.

Attention with the yat-kernel is attn(q_i)=sum_j k_yat(q_i,k_j) v_j / sum_j k_yat(q_i,k_j),
exact at O(N^2). RAY's feature map phi (E[phi(q).phi(k)]=k_yat) factorizes it to O(NM):
    out_i = phi(q_i)^T ( sum_j phi(k_j) v_j^T ) / ( phi(q_i)^T sum_j phi(k_j) ),
one map per token, with an EXACT causal recurrence S_t=S_{t-1}+phi(k_t)v_t^T (constant memory).

In-scope checks for the theory paper (NOT language quality):
  (a) output fidelity vs feature dimension M, and the attention-weight-matrix error;
  (b) fidelity vs sequence length N (does the linearization hold at long context?);
  (c) fidelity vs attention sharpness eps (peaked attention is harder for RFF);
  (d) scaling: exact O(N^2) vs RAY O(NM) memory/time, plus the constant O(M dv) decode state;
  (e) the causal streaming recurrence is exact.

Env: ~/.pixi/envs/jax/bin/python3 (numpy, sklearn). Run: yat_attention.py
REPRODUCIBILITY: results/yat_attention.json; backs the linear-attention figure (sec:exp_attention).
"""
import json, os, time
import numpy as np
import krr_downstream as K
import ts_ryf_costmatched as TS

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
HERE = os.path.dirname(__file__)


def ball(N, d, rng, lo=0.5, hi=1.0):
    U = rng.normal(size=(N, d)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    return U * rng.uniform(lo, hi, size=(N, 1))


def med_rel(a, b):
    return float(np.median(np.linalg.norm(a - b, axis=1) / (np.linalg.norm(b, axis=1) + 1e-12)))


def exact_attn(Q, Kk, V, b, eps):
    A = K.k_yat(Q, Kk, b, eps)
    return (A @ V) / (A.sum(1, keepdims=True) + 1e-12), A


def ray_phi(X, b, eps, D, m, seed): return TS.ts_ray_primal(X, b, eps, D, m, seed)


def ray_attn_from_phi(PhiQ, PhiK, V):
    KV = PhiK.T @ V; Z = PhiK.sum(0)
    num = PhiQ @ KV; den = PhiQ @ Z
    return num / (den[:, None] + np.where(np.abs(den)[:, None] < 1e-9, 1e-9, 0.0))


def main():
    b, dval = 1.0, 16
    out = {"config": {"b": b, "dval": dval}, "fidelity": [], "fid_vs_N": [], "fid_vs_eps": [],
           "scaling": [], "causal": {}, "decode": {}}

    # ---------- (a) fidelity + attention-weight-matrix error vs M (N=512, d=32) ----------
    N, d, seeds = 512, 32, 3
    rng = np.random.default_rng(0)
    Q, Kk, V = ball(N, d, rng), ball(N, d, rng), rng.normal(size=(N, dval))
    eps = float(np.median(K.sqdist(Kk, Kk)[np.triu_indices(N, 1)]))
    oe, A = exact_attn(Q, Kk, V, b, eps)
    Anorm = A / (A.sum(1, keepdims=True) + 1e-12)
    log(f"(a) fidelity vs M: N={N} d={d} eps={eps:.3f}")
    for Dd in [4, 8, 16, 32, 64, 128]:
        m = 64; oerr, werr = [], []
        for s in range(seeds):
            PhiQ, PhiK = ray_phi(Q, b, eps, Dd, m, 1000 + s), ray_phi(Kk, b, eps, Dd, m, 1000 + s)
            oerr.append(med_rel(ray_attn_from_phi(PhiQ, PhiK, V), oe))
            Ahat = PhiQ @ PhiK.T; Ahat = Ahat / (Ahat.sum(1, keepdims=True) + 1e-12)
            werr.append(float(np.linalg.norm(Ahat - Anorm) / np.linalg.norm(Anorm)))
        M = Dd * (m + d + 1)
        out["fidelity"].append({"D": Dd, "m": m, "M": M, "out_err": [float(np.mean(oerr)), float(np.std(oerr))],
                                "weight_err": float(np.mean(werr))})
        log(f"  M={M:6d}: out-err={np.mean(oerr):.4f}  weight-matrix-err={np.mean(werr):.4f}")

    # ---------- (b) fidelity vs sequence length N (fixed M) ----------
    d, Dd, m = 32, 32, 64; Mfix = Dd * (m + d + 1)
    log(f"(b) fidelity vs N at fixed M={Mfix} (D={Dd},m={m}):")
    for N in [256, 512, 1024, 2048, 4096, 8192]:
        rng = np.random.default_rng(10)
        Q, Kk, V = ball(N, d, rng), ball(N, d, rng), rng.normal(size=(N, dval))
        sN = min(N, 400); eps = float(np.median(K.sqdist(Kk[:sN], Kk[:sN])[np.triu_indices(sN, 1)]))
        oe, _ = exact_attn(Q, Kk, V, b, eps)
        errs = [med_rel(ray_attn_from_phi(ray_phi(Q, b, eps, Dd, m, 100 + s), ray_phi(Kk, b, eps, Dd, m, 100 + s), V), oe)
                for s in range(3)]
        out["fid_vs_N"].append({"N": N, "M": Mfix, "out_err": [float(np.mean(errs)), float(np.std(errs))]})
        log(f"  N={N:5d}: out-err={np.mean(errs):.4f}")

    # ---------- (c) fidelity vs attention sharpness eps ----------
    N, d = 512, 32; rng = np.random.default_rng(20)
    Q, Kk, V = ball(N, d, rng), ball(N, d, rng), rng.normal(size=(N, dval))
    med = float(np.median(K.sqdist(Kk, Kk)[np.triu_indices(N, 1)]))
    log("(c) fidelity vs sharpness eps (small eps = peaked attention):")
    for mult in [0.25, 0.5, 1.0, 2.0, 4.0]:
        eps = mult * med; oe, _ = exact_attn(Q, Kk, V, b, eps)
        errs = [med_rel(ray_attn_from_phi(ray_phi(Q, b, eps, 32, 64, 100 + s), ray_phi(Kk, b, eps, 32, 64, 100 + s), V), oe)
                for s in range(3)]
        out["fid_vs_eps"].append({"eps_mult": mult, "out_err": [float(np.mean(errs)), float(np.std(errs))]})
        log(f"  eps={mult:.2f}x median: out-err={np.mean(errs):.4f}")

    # ---------- (d) scaling: exact O(N^2) vs RAY O(NM); constant O(M dv) decode state ----------
    d, Dd, m = 32, 16, 64; Mfix = Dd * (m + d + 1); exact_cap = 8192
    decode_state_gb = Mfix * dval * 8 / 1e9
    log(f"(d) scaling: d={d} RAY M={Mfix}; decode state = M*dv = {decode_state_gb*1e6:.3f} MB (constant in N)")
    for N in [256, 1024, 4096, 8192, 16384, 32768, 65536, 131072]:
        rng = np.random.default_rng(1)
        Q, Kk, V = ball(N, d, rng), ball(N, d, rng), rng.normal(size=(N, dval))
        sN = min(N, 400); eps = float(np.median(K.sqdist(Kk[:sN], Kk[:sN])[np.triu_indices(sN, 1)]))
        row = {"N": N, "M": Mfix, "exact_mem_gb": N * N * 8 / 1e9, "ray_prefill_mem_gb": N * Mfix * 8 / 1e9,
               "decode_state_mem_gb": decode_state_gb, "kv_cache_gb": N * (d + dval) * 8 / 1e9}
        if N <= exact_cap:
            t0 = time.perf_counter(); exact_attn(Q, Kk, V, b, eps); row["exact_s"] = time.perf_counter() - t0
        else:
            row["exact_s"] = None
        t0 = time.perf_counter()
        ray_attn_from_phi(ray_phi(Q, b, eps, Dd, m, 7), ray_phi(Kk, b, eps, Dd, m, 7), V)
        row["ray_s"] = time.perf_counter() - t0
        out["scaling"].append(row)
        es = f"{row['exact_s']:.3f}s" if row["exact_s"] else "N/A"
        log(f"  N={N:6d}: exact {es:>8s} ({row['exact_mem_gb']:7.2f}GB)  RAY {row['ray_s']:.3f}s "
            f"prefill {row['ray_prefill_mem_gb']:.3f}GB  decode-state {decode_state_gb*1e3:.2f}MB")

    # ---------- (e) causal streaming recurrence is exact ----------
    N, d = 512, 32; rng = np.random.default_rng(2)
    Q, Kk, V = ball(N, d, rng), ball(N, d, rng), rng.normal(size=(N, dval))
    eps = float(np.median(K.sqdist(Kk, Kk)[np.triu_indices(N, 1)]))
    A = K.k_yat(Q, Kk, b, eps); Am = A * np.tril(np.ones((N, N)))
    oe_causal = (Am @ V) / (Am.sum(1, keepdims=True) + 1e-12)
    PhiQ, PhiK = ray_phi(Q, b, eps, 64, 64, 7), ray_phi(Kk, b, eps, 64, 64, 7)
    KVcum = np.cumsum(PhiK[:, :, None] * V[:, None, :], axis=0); Zcum = np.cumsum(PhiK, axis=0)
    num = np.einsum('im,imv->iv', PhiQ, KVcum); den = np.einsum('im,im->i', PhiQ, Zcum)
    or_causal = num / (den[:, None] + np.where(np.abs(den)[:, None] < 1e-9, 1e-9, 0.0))
    out["causal"] = {"N": N, "D": 64, "m": 64, "rel_err_vs_exact_causal": med_rel(or_causal, oe_causal)}
    out["decode"] = {"state_mem_mb": decode_state_gb * 1e3, "note": "O(M*dv) constant in N; KV cache is O(N*d) and grows"}
    log(f"(e) causal recurrence vs exact masked attention: rel-err={out['causal']['rel_err_vs_exact_causal']:.4f}")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "yat_attention.json"), "w"), indent=2)
    log("wrote results/yat_attention.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
fig_attention.pdf for papers/01_theory/biased_random_features/main.tex, from
results/yat_attention.json (no recompute).

  (a) linear yat-attention fidelity vs feature dimension M (output + weight-matrix error);
  (b) fidelity vs attention sharpness eps -- diffuse is easy, peaked is hard (RFF limit);
  (c) the O(N^2) wall: exact attention vs RAY's O(NM) prefill and O(M dv) constant decode state.

Env: ~/.pixi/envs/jax/bin/python3 (matplotlib, numpy). Run: make_attention_fig.py
REPRODUCIBILITY: reads results/yat_attention.json only.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
C = {"ray": "#1b9e77", "exact": "#444444", "cap": "#e7298a", "wt": "#d95f02", "dec": "#7570b3"}


def main():
    d = json.load(open(os.path.join(HERE, "results", "yat_attention.json")))
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13.5, 3.5))

    # (a) fidelity vs M (output + weight matrix), stable-M regime
    fid = [r for r in d["fidelity"] if r["M"] >= 1552]
    M = np.array([r["M"] for r in fid])
    a1.loglog(M, [r["out_err"][0] for r in fid], "o-", color=C["ray"], lw=1.9, ms=6, label="output")
    a1.loglog(M, [r["weight_err"] for r in fid], "s--", color=C["wt"], lw=1.6, ms=5, label="attn. weights")
    a1.set_xlabel("feature dimension $M$"); a1.set_ylabel("median rel. error")
    a1.set_title("(a) faithful, falls with $M$", fontsize=10, pad=8)
    a1.legend(frameon=False, fontsize=9); a1.grid(True, which="both", ls=":", alpha=0.4)

    # (b) fidelity vs sharpness eps
    fe = d["fid_vs_eps"]; em = np.array([r["eps_mult"] for r in fe]); ee = np.array([r["out_err"][0] for r in fe])
    a2.semilogx(em, ee, "o-", color=C["ray"], lw=1.9, ms=6, base=2)
    a2.set_xlabel(r"radial scale $\varepsilon$ ($\times$ median dist.)"); a2.set_ylabel("median per-token error")
    a2.set_title(r"(b) diffuse easy, peaked hard", fontsize=10, pad=8)
    a2.set_xticks(em); a2.set_xticklabels([f"{v:g}" for v in em])
    a2.annotate("peaked", (em[0], ee[0]), textcoords="offset points", xytext=(6, -2), fontsize=8.5)
    a2.annotate("diffuse", (em[-1], ee[-1]), textcoords="offset points", xytext=(-30, 6), fontsize=8.5)
    a2.grid(True, which="both", ls=":", alpha=0.4)

    # (c) memory wall + constant decode state
    sc = d["scaling"]; N = np.array([r["N"] for r in sc])
    exmem = np.array([r["exact_mem_gb"] for r in sc]); pre = np.array([r["ray_prefill_mem_gb"] for r in sc])
    dec = np.array([r["decode_state_mem_gb"] for r in sc])
    feas = np.array([r["exact_s"] is not None for r in sc])
    a3.loglog(N, exmem, "s--", color=C["exact"], lw=1.6, ms=5, label=r"exact ($N^2$)")
    a3.loglog(N[~feas], exmem[~feas], "x", color=C["cap"], ms=8, mew=2, label="exact: cannot store")
    a3.loglog(N, pre, "o-", color=C["ray"], lw=1.9, ms=6, label=r"RAY prefill ($NM$)")
    a3.loglog(N, dec, "-", color=C["dec"], lw=1.8, label=r"RAY decode state ($Mdv$, const.)")
    a3.set_xlabel("sequence length $N$"); a3.set_ylabel("memory (GB)")
    a3.set_title(r"(c) clears the $O(N^2)$ wall", fontsize=10, pad=8)
    a3.legend(frameon=False, fontsize=8.0, loc="upper left"); a3.grid(True, which="both", ls=":", alpha=0.4)

    fig.tight_layout(w_pad=1.8)
    out = os.path.join(HERE, "..", "fig_attention.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", os.path.abspath(out),
          f"(causal rel-err {d['causal']['rel_err_vs_exact_causal']:.3f}; decode state {d['decode']['state_mem_mb']:.2f} MB)")


if __name__ == "__main__":
    main()

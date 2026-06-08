#!/usr/bin/env python3
"""
Figure 1 for papers/01_theory/biased_random_features/main.tex.

(a) Relative Frobenius error vs D (log-log) for RAY at d in {2,10,20}, with an
    O(1/sqrt D) reference slope -> shows the rate and near-dimension-independence.
(b) Samples for a 0.05-approximation vs dimension: RAY radial count D* (flat) vs
    Nystrom landmark count m* (blows up) -> the dimension-free sample complexity.

Reads results/gram_approx.json and results/dimension_free.json. Writes ../fig1.pdf.
Run: ~/.pixi/envs/jax/bin/python3 make_fig1.py
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")

gram = json.load(open(os.path.join(RES, "gram_approx.json")))
dimfree = json.load(open(os.path.join(RES, "dimension_free.json")))

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4))

# ---- panel (a): error vs D, log-log ----
colors = {"2": "#1b9e77", "10": "#d95f02", "20": "#7570b3"}
Ds = [int(d) for d in ["10", "50", "100", "500", "1000"]]
for d in ["2", "10", "20"]:
    errs = [gram["cosine"][d][str(D)][0] for D in Ds]
    ax1.loglog(Ds, errs, "o-", color=colors[d], label=f"$d={d}$", lw=1.8, ms=5)
# O(1/sqrt D) reference anchored at the d=10, D=10 point
e0 = gram["cosine"]["10"]["10"][0]
ax1.loglog(Ds, [e0 * np.sqrt(Ds[0] / D) for D in Ds], "k--", lw=1.2, label=r"$O(1/\sqrt{D})$")
ax1.set_xlabel("radial samples $D$")
ax1.set_ylabel(r"rel. Frobenius error $\|K_D-K\|_F/\|K\|_F$")
ax1.set_title("(a) Convergence at the $O(1/\\sqrt{D})$ rate")
ax1.legend(frameon=False, fontsize=9)
ax1.grid(True, which="both", ls=":", alpha=0.4)

# ---- panel (b): D*(d) and m*(d) ----
dims = sorted(int(d) for d in dimfree["ryf_Dstar"])
Dstar = [dimfree["ryf_Dstar"][str(d)] for d in dims]
mstar = [dimfree["nystrom_mstar"][str(d)] for d in dims]
cap = 350  # grid ceiling; None means "did not reach within the grid"
ax2.plot(dims, Dstar, "o-", color="#1b9e77", lw=1.8, ms=6, label="RAY radial $D^\\star$")
reach = [(d, m) for d, m in zip(dims, mstar) if m is not None]
if reach:
    ax2.plot([d for d, _ in reach], [m for _, m in reach], "s-", color="#d95f02",
             lw=1.8, ms=6, label="Nyström $m^\\star$")
unreach = [d for d, m in zip(dims, mstar) if m is None]
for d in unreach:
    ax2.annotate("", xy=(d, cap + 130), xytext=(d, cap),
                 arrowprops=dict(arrowstyle="-|>", color="#d95f02", lw=1.6))
if unreach:
    ax2.text(unreach[0], cap + 150, "Nyström $> 350$ (target unreached)",
             color="#d95f02", fontsize=8.5, va="bottom")
ax2.axhline(cap, color="grey", ls=":", lw=1)
ax2.set_xscale("log")
ax2.set_xticks(dims); ax2.set_xticklabels(dims)
ax2.set_xlabel("dimension $d$")
ax2.set_ylabel(f"radial samples for $\\leq {dimfree['config']['delta']:g}$ error")
ax2.set_title("(b) Dimension-free sample complexity")
ax2.legend(frameon=False, fontsize=9, loc="center left")
ax2.grid(True, which="both", ls=":", alpha=0.4)
ax2.set_ylim(0, max([d for d in Dstar if d] + [cap]) + 150)

fig.tight_layout()
out = os.path.join(HERE, "..", "fig1.pdf")
fig.savefig(out, bbox_inches="tight")
print("wrote", os.path.abspath(out))

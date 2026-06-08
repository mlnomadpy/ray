#!/usr/bin/env python3
"""
Figures 2-4 for papers/01_theory/biased_random_features/main.tex, from archived JSONs.

  fig_scaling.pdf : wall-clock and memory vs N (exact ~N^2 wall vs RAY/Nystrom linear)
  fig_bias.pdf    : variance vs (x.w + b) log-log, exponent-4 (R^2+b)^4 law
  fig_krr.pdf     : downstream test metric vs D -- RAY reaches the exact kernel with
                    far fewer random features than Gaussian/IMQ-RFF/Nystrom

Run: ~/.pixi/envs/jax/bin/python3 make_figures.py
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
C = {"exact": "#444444", "ray": "#1b9e77", "imq": "#d95f02", "gauss": "#7570b3",
     "nys": "#e7298a", "rm": "#66a61e", "hyb": "#1f78b4"}


def load(name): return json.load(open(os.path.join(RES, name)))


# ----------------------------------------------------------- fig: scaling -----
def fig_scaling():
    d = load("timing_scaling.json"); rows = d["rows"]
    N = [r["N"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.4))
    # time
    for key, lab, c in [("t_exact", "exact", C["exact"]), ("t_ryf", "RAY (ours)", C["ray"]),
                        ("t_nys", "Nyström", C["nys"])]:
        xs = [r["N"] for r in rows if r.get(key)]; ys = [r[key] for r in rows if r.get(key)]
        a1.loglog(xs, ys, "o-", color=c, lw=1.8, ms=5, label=lab)
    a1.set_xlabel("$N$"); a1.set_ylabel("ridge-fit wall-clock (s)")
    a1.set_title("(a) Time: exact $\\sim N^2$, others linear")
    a1.legend(frameon=False, fontsize=9); a1.grid(True, which="both", ls=":", alpha=0.4)
    # memory
    a2.loglog(N, [r["mem_exact_gb"] for r in rows], "o-", color=C["exact"], lw=1.8, ms=5, label="exact ($N^2$)")
    a2.loglog(N, [r["mem_ryf_gb"] for r in rows], "o-", color=C["ray"], lw=1.8, ms=5, label="RAY ($NM$)")
    a2.loglog(N, [r["mem_nys_gb"] for r in rows], "o-", color=C["nys"], lw=1.8, ms=5, label="Nyström ($Nm$)")
    a2.axhline(d["config"]["mem_cap_gb"], color="grey", ls="--", lw=1)
    a2.text(N[0], d["config"]["mem_cap_gb"] * 1.15, "exact skipped above", fontsize=8.5, color="grey")
    a2.set_xlabel("$N$"); a2.set_ylabel("representation memory (GB)")
    a2.set_title("(b) Memory: the $O(N^2)$ wall")
    a2.legend(frameon=False, fontsize=9, loc="upper left"); a2.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout(); out = os.path.join(HERE, "..", "fig_scaling.pdf")
    fig.savefig(out, bbox_inches="tight"); print("wrote", os.path.abspath(out))


# -------------------------------------------------------------- fig: bias -----
def fig_bias():
    d = load("bias_scaling.json"); fig, ax = plt.subplots(figsize=(4.7, 3.6))
    for name, c, mk in [("aligned_rho1.0", C["ray"], "o"), ("rho0.5", C["gauss"], "s")]:
        p = d["pairs"][name]; rho = p["rho"]
        xs = [rho + b for b in p["bs"]]; ys = p["var"]
        sl = p["slope_var_vs_(rho+b)"]
        ax.loglog(xs, ys, mk + "-", color=c, lw=1.8, ms=5,
                  label=f"$x^\\top w={rho:g}$ (slope {sl:.2f})")
    # slope-4 guide anchored at the first aligned point
    p0 = d["pairs"]["aligned_rho1.0"]; x0, y0 = 1.0 + p0["bs"][0], p0["var"][0]
    xg = np.array([x0, 1.0 + p0["bs"][-1]])
    ax.loglog(xg, y0 * (xg / x0) ** 4, "k--", lw=1.2, label="slope $4$")
    ax.set_xlabel("$x^\\top w + b$"); ax.set_ylabel("estimator variance")
    ax.set_title("Variance follows $(x^\\top w+b)^4$")
    ax.legend(frameon=False, fontsize=9); ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout(); out = os.path.join(HERE, "..", "fig_bias.pdf")
    fig.savefig(out, bbox_inches="tight"); print("wrote", os.path.abspath(out))


# --------------------------------------------------------------- fig: krr -----
def fig_krr():
    d = load("krr_downstream.json"); Ds = d["config"]["Ds"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    panel = {"digits": (axes[0], "accuracy", "(a) digits (accuracy $\\uparrow$)"),
             "california": (axes[1], "RMSE", "(b) california (RMSE $\\downarrow$)")}
    series = [("ryf_yat", "RAY (ours)", C["ray"], "o"),
              ("imqrff", "IMQ-RFF", C["imq"], "s"),
              ("gaussrff", "Gaussian-RFF", C["gauss"], "^"),
              ("nystrom_yat", "Nyström-Yat", C["nys"], "d")]
    for name, (ax, ylab, title) in panel.items():
        if name not in d["datasets"]:
            continue
        R = d["datasets"][name]["results"]
        for key, lab, c, mk in series:
            ys = np.array([R[f"{key}@{D}"][0] for D in Ds])
            es = np.array([R[f"{key}@{D}"][1] for D in Ds])
            ax.semilogx(Ds, ys, mk + "-", color=c, lw=1.7, ms=5, label=lab)
            ax.fill_between(Ds, ys - es, ys + es, color=c, alpha=0.15, lw=0)
        ax.axhline(R["exact_yat"][0], color=C["exact"], ls="--", lw=1.3, label="exact Yat-KRR")
        ax.set_xlabel("random features $D$"); ax.set_ylabel(ylab); ax.set_title(title)
        ax.grid(True, which="both", ls=":", alpha=0.4)
    axes[0].legend(frameon=False, fontsize=8.3, loc="lower right")
    fig.tight_layout(); out = os.path.join(HERE, "..", "fig_krr.pdf")
    fig.savefig(out, bbox_inches="tight"); print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    import sys
    # Use a font that has the Tifinagh glyph; fall back silently if labels lack it.
    for fn in (fig_scaling, fig_bias, fig_krr):
        try:
            fn()
        except Exception as e:
            print("SKIP", fn.__name__, e, file=sys.stderr)

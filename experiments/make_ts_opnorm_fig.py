#!/usr/bin/env python3
"""
fig_ts_opnorm.pdf for papers/01_theory/biased_random_features/main.tex, from the
archived results/ts_opnorm_validation.json (no recompute).

Visualizes Theorem thm:ts_opnorm, the operator-norm error of doubly-randomized RAY:
  ||K_hat_{D,m} - K||_op  <=  [radial term, O(D^-1/2)]  +  [sketch term, eta*||P||].

  (a) error vs D at fixed m=128: the radial term falls as O(D^-1/2) while the sketch
      term eta||P|| is a D-independent floor; the total decays to that floor. The
      m->inf (exact-modulation) curve is the limit where the floor is zero.
  (b) sketch term vs m: eta||P|| (and eta=||E_P||/||P||) shrink with the sketch size m,
      so m->inf recovers exact-modulation RAY (Corollary cor:bernstein_tail).

Env: /opt/homebrew/bin/python3 or ~/.pixi/envs/jax/bin/python3 (matplotlib, numpy).
Run: make_ts_opnorm_fig.py   ->   ../fig_ts_opnorm.pdf
REPRODUCIBILITY: reads results/ts_opnorm_validation.json only; deterministic.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
C = {"total": "#444444", "radial": "#1b9e77", "sketch": "#e7298a", "exact": "#7570b3"}


def main():
    d = json.load(open(os.path.join(RES, "ts_opnorm_validation.json")))
    rows = d["rows"]
    Pop = d["Pop"]

    vsD_m = sorted([r for r in rows if r["kind"] == "vsD" and r["m"] == 128], key=lambda r: r["D"])
    vsD_inf = sorted([r for r in rows if r["kind"] == "vsD" and r["m"] == "inf"], key=lambda r: r["D"])
    sk = sorted([r for r in rows if r["kind"] == "sketch"], key=lambda r: r["m"])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.5))

    # ---- panel (a): error vs D at m=128 ----
    D = np.array([r["D"] for r in vsD_m], float)
    total = np.array([r["total"] for r in vsD_m])
    radial = np.array([r["radial"] for r in vsD_m])
    sketch_floor = vsD_m[0]["sketch"]            # D-independent
    inf_total = np.array([r["total"] for r in vsD_inf])

    a1.loglog(D, total, "o-", color=C["total"], lw=1.9, ms=6, label=r"total $\|\widehat K_{D,m}-K\|_{\mathrm{op}}$")
    a1.loglog(D, radial, "s-", color=C["radial"], lw=1.8, ms=5, label="radial term")
    a1.loglog(D, radial[0] * np.sqrt(D[0] / D), "--", color=C["radial"], lw=1.1, label=r"$O(1/\sqrt{D})$")
    a1.axhline(sketch_floor, color=C["sketch"], ls="-.", lw=1.6, label=r"sketch term $\eta\|P\|_{\mathrm{op}}$")
    a1.loglog(D, inf_total, "^:", color=C["exact"], lw=1.5, ms=5, label=r"exact mod. ($m\to\infty$)")
    a1.set_xlabel("radial draws $D$"); a1.set_ylabel("operator-norm error")
    a1.set_title("(a) error vs $D$ at $m{=}128$")
    a1.legend(frameon=False, fontsize=8.0, loc="lower left")
    a1.grid(True, which="both", ls=":", alpha=0.4)

    # ---- panel (b): sketch term and eta vs m ----
    m = np.array([r["m"] for r in sk], float)
    sketch_bias = np.array([r["sketch_bias"] for r in sk])
    eta = np.array([r["eta_emp"] for r in sk])

    a2.plot(m, sketch_bias, "o-", color=C["sketch"], lw=1.9, ms=6, label=r"sketch term $\|E_P\circ R\|_{\mathrm{op}}$")
    a2.plot(m, sketch_bias[0] * np.sqrt(m[0] / m), "--", color=C["sketch"], lw=1.1, label=r"$O(1/\sqrt{m})$")
    a2.set_xscale("log", base=2)
    a2.set_xlabel("sketch dimension $m$"); a2.set_ylabel(r"sketch term $\|E_P\circ R\|_{\mathrm{op}}$")
    a2.set_title(r"(b) sketch term vanishes as $m\to\infty$")
    a2.set_xticks(m); a2.set_xticklabels([f"{int(v)}" for v in m])
    a2.grid(True, which="both", ls=":", alpha=0.4)
    # eta on a twin axis
    a2b = a2.twinx()
    a2b.spines["top"].set_visible(False)
    a2b.plot(m, eta, "d:", color=C["radial"], lw=1.5, ms=5, label=r"$\eta=\|E_P\|/\|P\|$")
    a2b.set_ylabel(r"relative sketch error $\eta$", color=C["radial"])
    a2b.tick_params(axis="y", colors=C["radial"])
    a2b.set_ylim(0, max(eta) * 1.25)
    h1, l1 = a2.get_legend_handles_labels(); h2, l2 = a2b.get_legend_handles_labels()
    a2.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.0, loc="upper right")

    fig.tight_layout()
    out = os.path.join(HERE, "..", "fig_ts_opnorm.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", os.path.abspath(out), f"(eps={d['eps']:.3f}, ||P||_op={Pop:.1f})")


if __name__ == "__main__":
    main()

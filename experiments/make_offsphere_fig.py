#!/usr/bin/env python3
"""
fig_offsphere.pdf -- the off-sphere result promoted to a figure (R13).
(a) RAY relative Frobenius error vs D for several d, with the O(1/sqrt D) guide.
(b) error at D=1000 vs d: RAY stays bounded, uniform/k-means Nystrom worsen with d.
Reads results/off_sphere_gram.json. Run: ~/.pixi/envs/jax/bin/python3 make_offsphere_fig.py
"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
d = json.load(open(os.path.join(HERE, "results", "off_sphere_gram.json")))
by_d = d["by_d"]; Ds = d["config"]["Ds"]
dims = sorted(int(k) for k in by_d)
colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(dims)))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 3.6))
for c, dim in zip(colors, dims):
    ys = [by_d[str(dim)]["ryf"][str(D)][0] for D in Ds]
    a1.loglog(Ds, ys, "o-", color=c, lw=1.7, ms=5, label=f"$d={dim}$")
g = np.array([Ds[0], Ds[-1]]); y0 = by_d[str(dims[0])]["ryf"][str(Ds[0])][0]
a1.loglog(g, y0 * np.sqrt(g[0] / g), "k--", lw=1.1, label=r"$O(1/\sqrt{D})$")
a1.set_xlabel("radial samples $D$"); a1.set_ylabel("relative Frobenius error")
a1.set_title("(a) RAY: Monte-Carlo rate at every $d$", fontsize=10, pad=8)
a1.legend(frameon=False, fontsize=8.5, ncol=2); a1.grid(True, which="both", ls=":", alpha=0.4)

ray = [by_d[str(dim)]["ryf"]["1000"][0] for dim in dims]
uni = [by_d[str(dim)]["nystrom"][0] for dim in dims]
km = [by_d[str(dim)]["nystrom_kmeans"][0] for dim in dims]
a2.plot(dims, ray, "o-", color="#1b9e77", lw=1.9, ms=6, label="RAY ($D{=}1000$)")
a2.plot(dims, uni, "s--", color="#e7298a", lw=1.6, ms=5, label="uniform Nyström ($m{=}100$)")
a2.plot(dims, km, "d--", color="#7570b3", lw=1.6, ms=5, label="k-means Nyström ($m{=}100$)")
a2.set_xlabel("dimension $d$"); a2.set_ylabel("relative Frobenius error")
a2.set_title("(b) matched count: RAY bounded, Nyström worsens", fontsize=10, pad=8)
a2.set_xticks(dims); a2.legend(frameon=False, fontsize=8.5); a2.grid(True, ls=":", alpha=0.4)
fig.tight_layout(w_pad=2.5); out = os.path.join(HERE, "..", "fig_offsphere.pdf")
fig.savefig(out, bbox_inches="tight"); print("wrote", os.path.abspath(out))

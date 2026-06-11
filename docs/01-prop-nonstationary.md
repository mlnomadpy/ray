# Proposition: Off-Sphere the Kernel Is Neither Stationary Nor Dot-Product
**Label:** `prop:nonstationary` | **Location:** main.tex line 123

## What it says
For every $b \ge 0$ and $\varepsilon > 0$, the biased ⵟ-kernel
$$k_{ⵟ,b}(w,x) = \frac{(w^\top x + b)^2}{\|w-x\|^2 + \varepsilon}$$
falls outside both classical random-feature templates:

1. **Not shift-invariant** on any domain containing two points of different norm. There is no function $\phi$ with $k_{ⵟ,b}(x,w) = \phi(x-w)$.
2. **Not a dot-product kernel** on any domain containing two pairs of equal inner product $s = x^\top w \ne -b$ but different distance. There is no function $\kappa$ with $k_{ⵟ,b}(x,w) = \kappa(x^\top w)$. (The value $s = -b$ is excluded because the numerator vanishes for both pairs there, so such pairs cannot witness the failure.)

On the unit sphere, where distance and inner product are in bijection ($\|x-w\|^2 = 2 - 2x^\top w$), the kernel **does** reduce to a dot-product kernel — the failure is specifically an off-sphere fact.

## Why it matters
This proposition is the structural justification for the entire paper. Random Fourier features require shift-invariance (Bochner's theorem); polynomial sketches require dot-product form. $k_{ⵟ,b}$ has neither, so no existing random-feature template applies to the full kernel directly, and exact methods revert to the $O(N^2)$ Gram matrix. The failure is not a matter of degree but a structural fact: it forces the Schur-factorization route (eq:schur) that randomizes each factor by its native tool. It also dictates the paper's key empirical regime: the off-sphere bounded-ball experiment (Section sec:exp_offsphere, Figure fig:offsphere) is the test where the kernel is genuinely non-dot-product and no on-sphere reduction is available.

## Proof idea
Both witnesses are immediate.

**Shift-invariance fails:** the diagonal $k_{ⵟ,b}(x,x) = (\|x\|^2 + b)^2/\varepsilon$ varies with $\|x\|$, whereas any shift-invariant kernel is constant on the diagonal. Two points of different norm suffice.

**Dot-product form fails:** take $x = w = e_1$ versus $x' = \sqrt{2}\,e_1$, $w' = e_1/\sqrt{2}$. Both pairs have inner product $x^\top w = x'^\top w' = 1$, but $\|x - w\|^2 = 0 \ne \tfrac{1}{2} = \|x' - w'\|^2$, so the kernel values differ (the denominators are $\varepsilon$ versus $\tfrac{1}{2}+\varepsilon$ with equal numerators $(1+b)^2$, provided $1 \ne -b$, which holds since $b \ge 0$). Hence $k_{ⵟ,b}$ is not a function of $x^\top w$ alone.

## Connections
**Depends on:** the definition of $k_{ⵟ,b}$ (eq:yat_biased); nothing else.
**Used by:** the motivation for the Schur factorization (eq:schur) and the five-step construction of Section sec:ryf; the framing of the off-sphere experiment (sec:exp_offsphere, Figure fig:offsphere) as the key test; contribution (i) cites it as showing the flagship instance is genuinely neither stationary nor dot-product.
**Validated by:** `off_sphere_gram.py`, `highd_offsphere.py`, `make_offsphere_fig.py` (the off-sphere regime where Nyström degrades with $d$ while RAY holds the $O(1/\sqrt{D})$ rate).

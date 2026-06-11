# Proposition: Explicit (D, D′) Variance and Optimal Allocation
**Label:** `prop:budget` | **Location:** main.tex line 755

## What it says
Let $v_t(x,w) = \mathrm{Var}_\omega[\,2\cos(\omega^\top x+\beta)\cos(\omega^\top w+\beta)\,]$ be the variance of one trigonometric feature pair at scale $t$, and set

$$V_{\mathrm{out}} = \mathrm{Var}_{T\sim\mathrm{Exp}(\varepsilon)}[g_T(x,w)], \qquad V_{\mathrm{in}} = \mathbb{E}_{T\sim\mathrm{Exp}(\varepsilon)}[v_T(x,w)].$$

Then the two-level estimator with $D$ outer scales and $D'$ inner frequencies per scale satisfies the **exact identity**

$$\mathrm{Var}[z(x)^\top z(w)] = \frac{(x^\top w+b)^4}{\varepsilon^2}\left(\frac{V_{\mathrm{out}}}{D} + \frac{V_{\mathrm{in}}}{D\,D'}\right).$$

At a fixed feature budget $B = DD'$, the right side is minimized by $D' = 1$ (hence $D = B$) whenever $V_{\mathrm{out}} > 0$: the inner term $V_{\mathrm{in}}/B$ is fixed by the budget, while the outer term $V_{\mathrm{out}}/D$ decreases as $1/D$. If $V_{\mathrm{out}} = 0$ (for example at $x = w$), every allocation with the same $DD'$ ties.

## Why it matters
This is the result that collapses the two-level hierarchical estimator to the flat one the paper deploys everywhere. The construction naturally produces two sampling levels (a Bernstein scale $t$, then Gaussian frequencies at that scale), and this identity proves the inner level is a budget artifact: spending the budget on more independent $(t_j, \omega_j)$ pairs strictly beats sharing frequencies within a scale, except in the degenerate zero-outer-variance case. It justifies the recommendation $D'=1$ (eq:flat), which Section sec:step-rff then identifies with standard IMQ random Fourier features tensored with the exact polynomial feature — the conceptual simplification that the inner level "is redundant" (line 183). All main Gram, dimension-free, and ridge-regression experiments run at flat $D'=1$ on the strength of this proposition. Its $V_{\mathrm{out}}$ term is also the $D'\to\infty$ limit appearing in `thm:exact_variance`, and the deterministic radial quadrature (line 460) is framed as replacing exactly this across-scale variance by a controllable bias.

## Proof idea
The polynomial factor is deterministic, so $\mathrm{Var}[Y_1] = (p(x)^\top p(w))^2\,\mathrm{Var}[\psi_T(x)^\top \psi_T(w)]$ with $(p(x)^\top p(w))^2 = (x^\top w+b)^4/\varepsilon^2$. The law of total variance, conditioning on the scale $T$ and then on the $D'$ inner frequencies, gives

$$\mathrm{Var}[\psi_T(x)^\top\psi_T(w)] = \mathbb{E}_T\bigl[\mathrm{Var}_\omega(\psi_T(x)^\top\psi_T(w) \mid T)\bigr] + \mathrm{Var}_T\bigl(\mathbb{E}_\omega[\psi_T(x)^\top\psi_T(w) \mid T]\bigr).$$

The conditional mean is $g_T(x,w)$, giving the outer term $\mathrm{Var}_T(g_T) = V_{\mathrm{out}}$. The conditional object is an average of $D'$ i.i.d. feature pairs, so its conditional variance is $v_T/D'$, giving the inner term $\mathbb{E}_T[v_T]/D' = V_{\mathrm{in}}/D'$. Dividing by $D$ (the $D$ blocks are i.i.d.) yields the identity. The allocation argument is then immediate: with $DD' = B$ fixed, $V_{\mathrm{in}}/(DD') = V_{\mathrm{in}}/B$ is constant in the split and $V_{\mathrm{out}}/D$ is decreasing in $D$, so $D' = 1$ is optimal.

## Connections
**Depends on:** the two-level estimator of Definition def:ryf (Steps 3–4 of the construction); `prop:biased_feature` (deterministic modulation factor); the law of total variance.
**Used by:** the flat-estimator recommendation eq:flat and the "inner level is redundant" discussion (line 183); `thm:exact_variance` (its $D'\to\infty$ comparison is the $V_{\mathrm{out}}$ term); `thm:variance` (the envelope's $3/(2D')$ term is the bounded inner term); the deterministic radial quadrature paragraph (line 460, replacing $V_{\mathrm{out}}$ by bias); all main experiments, which adopt flat $D'=1$.
**Validated by:** `budget_allocation.py` (Table tab:budget, Appendix sec:exp_budget): at fixed budget $B = DD' = 1000$ ($d=10$, $N=400$, 4 seeds), relative Frobenius error rises monotonically with $D'$ — $0.059$ at $D'=1$ vs. $0.175$ at $D'=50$, a $3\times$ gap — exactly the predicted $V_{\mathrm{out}}/D$ mechanism.

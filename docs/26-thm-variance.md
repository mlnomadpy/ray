# Theorem: Variance Envelope
**Label:** `thm:variance` | **Location:** main.tex line 733

## What it says
Let $V_D = \mathrm{Var}[z(x)^\top z(w)]$ for the two-level estimator with $D$ outer radial scales and $D'$ inner frequencies per scale. Accounting for both sampling levels,

$$V_D \le \frac{(\|x\|^2+b)^2(\|w\|^2+b)^2}{D\,\varepsilon^2}\left(\frac{\varepsilon}{2\|x-w\|^2+\varepsilon} + \frac{3}{2D'}\right) \le \frac{(R^2+b)^4}{D\,\varepsilon^2}\left(1 + \frac{3}{2D'}\right),$$

valid for **all** $D, D' \ge 1$ on data with $\|x\| \le R$, $b \ge 0$. The $3/(2D')$ term is the inner random-Fourier-feature variance; it does not vanish at the recommended $D'=1$ (where the constant is $5/2$), only as $D'\to\infty$.

## Why it matters
This is the coarse all-$(D,D')$ envelope that establishes the headline order $O((R^2+b)^4/(D\varepsilon^2))$: the variance of estimating k_ⵟ,b is governed by the fourth power of the bias-enlarged effective radius $R^2+b$ and the inverse square of the radial scale $\varepsilon$. It is the bound the bias-cost discussion reads across $b$ (setting $b=0$ recovers $R^8$-type unbiased constants; $b>0$ inflates by powers of $1+b/R^2$), it is the prefactor whose unavoidability `prop:variance_sharp` certifies, and its removal is exactly what `prop:normalized` buys by changing the kernel. Where `thm:exact_variance` gives the exact law for the recommended flat instance, this envelope covers the whole two-level hierarchy and is the form quoted in the comparison Table tab:rf_comparison's sample-complexity constant. The Cauchy–Schwarz step that makes it an envelope rather than an identity is also why it is empirically loose off the aligned configuration (at $D=100$ in the d=1 probe: bound $0.16$ vs. measured variance $0.002$).

## Proof idea
Write $z(x)^\top z(w) = \tfrac1D \sum_j Y_j$ with i.i.d. per-scale terms $Y_j = (\psi_t(x)^\top \psi_t(w))(p(x)^\top p(w))$, so $V_D = \tfrac1D \mathrm{Var}[Y_1] \le \tfrac1D \mathbb{E}[Y_1^2]$. The polynomial factor is deterministic and obeys $(p(x)^\top p(w))^2 = (x^\top w+b)^4/\varepsilon^2 \le (\|x\|^2+b)^2(\|w\|^2+b)^2/\varepsilon^2$ by Cauchy–Schwarz and $b \ge 0$ (the $1/\varepsilon^2$ comes from the normalization $p = \varepsilon^{-1/2}p_b$). For the Gaussian factor, the law of total variance over $\omega \mid T$ then $T$ splits

$$\mathbb{E}[(\psi_T(x)^\top \psi_T(w))^2] = \mathbb{E}_T[g_T(x,w)^2] + \mathbb{E}_T[v_T]/D',$$

where $v_t = \mathrm{Var}_\omega[2\cos(\omega^\top x+\beta)\cos(\omega^\top w+\beta)]$. The outer term is the Laplace transform $\mathbb{E}_{T\sim\mathrm{Exp}(\varepsilon)}[e^{-2T\|x-w\|^2}] = \varepsilon/(2\|x-w\|^2+\varepsilon)$. For the inner term, $\mathbb{E}_{\omega,\beta}[(2\cos(\omega^\top x+\beta)\cos(\omega^\top w+\beta))^2] = 1 + \tfrac12 e^{-4t\|x-w\|^2} \le \tfrac32$, so $v_t \le \tfrac32$ and $\mathbb{E}_T[v_T] \le \tfrac32$. Substituting and dividing by $D$ gives the first form; bounding $\varepsilon/(2\|x-w\|^2+\varepsilon) \le 1$ gives the second.

## Connections
**Depends on:** the per-scale term structure $Y = (\psi_t(x)^\top \psi_t(w))(p(x)^\top p(w))$ with exact polynomial factor (app:proofs preamble, line 731); `prop:biased_feature` (the exact modulation feature); the Bernstein scale mixture (the $\mathrm{Exp}(\varepsilon)$ Laplace transform); Cauchy–Schwarz.
**Used by:** `thm:exact_variance` (whose exact constants recover this order, line 265); `prop:variance_sharp` (sharpness of the prefactor); the bias-cost discussion (line 392, Section sec:exp_bias) and Table tab:rf_comparison (the $(R^2+b)^4\varepsilon^{-2}$ constant); `prop:normalized` (removal of the blow-up); the variance-reduction discussion (line 1164, looseness of the Cauchy–Schwarz step).
**Validated by:** `bias_scaling.py` (Figure fig:bias — fitted log-log slopes $4.01$ and $3.99$ against the slope-4 guide; for the aligned pair the ratio $\mathrm{Var}/(R^2+b)^4$ is constant $\approx 5\times 10^{-5}$, so the envelope is tight there, while the $x^\top w = 0.5$ pair sits below it by exactly the Cauchy–Schwarz gap); `variance_validation.py` (Table tab:variance_reduction — the bound holds but is loose pointwise).

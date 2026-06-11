# Theorem: Exact Variance of the Flat Estimator
**Label:** `thm:exact_variance` | **Location:** main.tex line 258

## What it says
Write $r = \|x-w\|^2$ and $a = (x^\top w + b)^2$. The **flat estimator** — *one* inner frequency per scale, $D'=1$ (not the $D'\to\infty$ inner-averaged estimator) —

$$\widehat{k}_D = \frac{a}{\varepsilon}\,\frac{1}{D}\sum_{j=1}^D 2\cos(\omega_j^\top x + \beta_j)\cos(\omega_j^\top w + \beta_j),$$

with $T_j \sim \mathrm{Exp}(\varepsilon)$ and $\omega_j \mid T_j \sim \mathcal{N}(0, 2T_j I_d)$, has the **closed-form variance**

$$\mathrm{Var}[\widehat{k}_D(x,w)] = \frac{a^2}{D\varepsilon^2}\left[\,1 + \frac{1}{2}\,\frac{\varepsilon}{\varepsilon + 4r} - \left(\frac{\varepsilon}{\varepsilon + r}\right)^2\,\right].$$

This is an identity, not a bound. At $r=0$ the bracket equals $\tfrac12$ (the one-frequency variance does not vanish on the diagonal); the bracket tends to $1$ as $r\to\infty$ and is uniformly bounded by $\tfrac32$. By contrast, the $D'\to\infty$ inner-averaged estimator has outer-only variance $\tfrac{a^2}{D\varepsilon^2}\bigl[\tfrac{\varepsilon}{\varepsilon+2r} - (\tfrac{\varepsilon}{\varepsilon+r})^2\bigr]$, which is exactly the $V_{\mathrm{out}}$ term of `prop:budget`.

## Why it matters
This is the variance law of the estimator the paper actually recommends ($D'=1$, the flat draw of $D$ independent $(t_j,\omega_j)$ pairs). It locates the variance precisely: the scale is set by the fourth power of the bias-shifted alignment, $a^2 = (x^\top w + b)^4$, times $\varepsilon^{-2}$, while the radial bracket only modulates between $\tfrac12$ and $1$. So the alignment numerator, not the distance, sets the variance — this recovers the $O((R^2+b)^4/(D\varepsilon^2))$ order of the envelope `thm:variance` with exact constants, quantifies exactly what the bias $b$ costs (the $(R^2+b)^4$ inflation studied in Section sec:exp_bias), and is the instance that `prop:variance_sharp` exhibits to show the prefactor is attained with equality. It also feeds directly into `prop:normalized`: substituting the unit-norm modulation $q_b$ (so $a \le 1$) into this identity gives the radius- and bias-free bound $3/(2D\varepsilon^2)$.

## Proof idea
Set $Y = 2\cos(\omega^\top x + \beta)\cos(\omega^\top w + \beta) = \cos(\omega^\top(x-w)) + \cos(\omega^\top(x+w) + 2\beta)$ by product-to-sum. The second term has zero mean over $\beta \sim \mathrm{Unif}[0, 2\pi]$, so with the Gaussian characteristic function $\mathbb{E}_\omega[\cos(\omega^\top u)] = e^{-T\|u\|^2}$ for $\omega \mid T \sim \mathcal{N}(0, 2TI_d)$:

- **Mean:** $\mathbb{E}[Y \mid T] = e^{-Tr}$, hence $\mathbb{E}[Y] = \mathbb{E}_{T\sim\mathrm{Exp}(\varepsilon)}[e^{-Tr}] = \varepsilon/(\varepsilon + r)$ (the Laplace transform of the exponential law — unbiasedness for the rescaled radial factor).
- **Second moment:** $\mathbb{E}_\beta[Y^2 \mid \omega] = \cos^2(\omega^\top(x-w)) + \tfrac12$, and $\mathbb{E}_\omega[\cos^2(\omega^\top(x-w)) \mid T] = \tfrac12(1 + e^{-4Tr})$ via $\cos^2\theta = \tfrac12(1+\cos 2\theta)$ and $\mathbb{E}[\cos(2\omega^\top(x-w)) \mid T] = e^{-4Tr}$. Hence $\mathbb{E}[Y^2 \mid T] = 1 + \tfrac12 e^{-4Tr}$, and integrating over $T \sim \mathrm{Exp}(\varepsilon)$ gives $\mathbb{E}[Y^2] = 1 + \tfrac12\,\varepsilon/(\varepsilon + 4r)$.

Thus $\mathrm{Var}(Y) = 1 + \tfrac12\,\varepsilon/(\varepsilon+4r) - (\varepsilon/(\varepsilon+r))^2$. Since $\widehat{k}_D$ averages $D$ i.i.d. copies scaled by the deterministic factor $a/\varepsilon$, $\mathrm{Var}[\widehat{k}_D] = (a/\varepsilon)^2\,\mathrm{Var}(Y)/D$. (Proof at main.tex lines 813–815.)

## Connections
**Depends on:** the flat estimator definition (eq:flat, line 184); unbiasedness `thm:unbiased`; the Gaussian characteristic function and the Laplace transform of $\mathrm{Exp}(\varepsilon)$ (the Bernstein scale mixture of Section sec:step-bernstein).
**Used by:** `prop:variance_sharp` (this is the equality instance of the factored family); `prop:budget` (the $V_{\mathrm{out}}$ comparison at $D'\to\infty$); `prop:normalized` (the same identity with $a \le 1$ gives the $3/(2D\varepsilon^2)$ bound); `thm:bernstein_schur` (the class-level proof reuses the trigonometric bound $\mathbb{E}[Y^2 \mid T] \le \tfrac32$); `prop:pos_dichotomy` (cites the trigonometric second-moment computation); the prose after `thm:variance` (exact-constants recovery of the envelope order).
**Validated by:** `exact_variance_check.py` — across $b \in \{0,1,2\}$ and 18 off-sphere pairs with $r \in [0.4, 2.2]$, the ratio of empirical (over $2\times 10^5$ draws) to predicted variance is $1.001 \pm 0.002$, with bias within $0.5\%$ (Section sec:exp_bias); `variance_validation.py` (Table tab:variance_reduction, the $O(1/D)$ decay and QMC constant).

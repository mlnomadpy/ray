# Proposition: Positive-Weight Radial Quadrature
**Label:** `prop:quadrature` | **Location:** main.tex line 462

## What it says

The outer scale integral of the radial factor can be discretized deterministically instead of by Monte Carlo, replacing across-scale variance by a uniform, controllable bias — answering the kernel-quadrature question of the Discussion in the affirmative for the radial factor.

The radial factor is the Laplace transform

$$h_\varepsilon(r) = \int_0^\infty e^{-\varepsilon t}\,e^{-t r}\,dt = \frac{1}{\varepsilon + r}.$$

For every tolerance $\tau > 0$ there exist, by exponential-sum approximation of $1/x$ on a bounded interval (Beylkin–Monzón 2010),

$$D = O\bigl(\log(1 + 4R^2/\varepsilon)\,\log(1/\tau)\bigr)$$

**positive** nodes $t_1, \dots, t_D > 0$ and **positive** weights $w_1, \dots, w_D > 0$ with

$$\sup_{r \in [0,\,4R^2]} \Bigl|\sum_{j=1}^D w_j\,e^{-t_j r} - h_\varepsilon(r)\Bigr| \le \tau.$$

Replacing the random scales $t_j \sim \mathrm{Exp}(\varepsilon)$ by these nodes and the average by $\sum_j w_j(\cdot)$ leaves a surrogate radial Gram that is a nonnegative combination of Gaussian kernels, so the surrogate $\widehat k = \bigl(\sum_j w_j e^{-t_j\|\cdot-\cdot\|^2}\bigr)\cdot p$ is **positive definite**, and the entrywise error is the deterministic bias

$$\sup_{i,j}|\widehat k_{ij} - k_{ij}| \le \tau\,(R^2+b)^2,$$

free of the $O(D^{-1/2})$ radial fluctuation of `thm:uniform`.

## Why it matters

Three roles. (i) It shows the radial node count grows only **logarithmically** in the dynamic range $4R^2/\varepsilon$ and in $1/\tau$ — exponentially better than the $O(\tau^{-2})$ Monte-Carlo count — when the Gaussian factor can be computed exactly (small-Gram / exact-surrogate settings). (ii) Positivity of the weights is structural, not cosmetic: it keeps the surrogate a nonnegative Gaussian combination, hence PSD after the Schur product with the modulation, so the surrogate is a genuine kernel. (iii) The bounded node set $\{t_j\} \subset (0, t_{\max}]$ is exactly the bounded scale set that positive features require to escape the variance dichotomy (`prop:pos_dichotomy` caps the second moment through the largest node). The price is the loss of exact unbiasedness, traded for the uniform bias above; the unbiased sampler remains the variant for the streaming and analysis claims.

**Empirical scoping is sharp** (Appendix app:exp_details, `radial_quadrature`): with the Gaussian factor computed *exactly* (the Gram model), a $D$-node Gauss–Laguerre rule reaches machine precision by $D = 32$ where Monte Carlo is still at $5.4 \times 10^{-2}$ RMSE. But in the deployed estimator the inner Fourier noise dominates, and a naive equal-allocation node rule paired with RFF is in fact **worse** than i.i.d. sampling at matched draws ($0.38$ vs. $0.17$ at $D = 32$), because the nodes spend budget on scales the random law visits in proportion to their weight. The nodes are the right tool for exact-Gaussian surrogates and small-Gram settings, not a drop-in replacement for the trigonometric sampler.

## Proof idea

The quadrature bound is the cited exponential-sum result applied to $1/x$ on the interval $[\varepsilon, \varepsilon + 4R^2]$ after the shift $x = \varepsilon + r$: $1/x$ admits an approximation by $D = O(\log(x_{\max}/x_{\min})\log(1/\tau))$ exponentials with positive coefficients on a bounded interval, and $r \in [0, 4R^2]$ (the maximal squared distance for $\|x\| \le R$) maps to exactly that interval. Positive weights keep each term $w_j e^{-t_j\|x-w\|^2}$ a (scaled) Gaussian kernel; a nonnegative combination of PSD kernels is PSD, and the Schur product with the PSD modulation $p$ stays PSD by the Schur product theorem. Multiplying the radial bound by the modulation, which is bounded by $(R^2+b)^2$ on the ball, gives the entrywise kernel bias.

## Connections

**Depends on:** The Laplace/Bernstein representation $h_\varepsilon(r) = \int_0^\infty e^{-\varepsilon t}e^{-tr}dt$ (the same integral underlying `thm:unbiased`), exponential-sum approximation of $1/x$ (Beylkin–Monzón 2010), `lem:schur` (PSD-ness of the surrogate).
**Used by:** `prop:pos_dichotomy` (the bounded node set is how positive features get a finite second moment, through the largest node $t_{\max}$), the variance-reduction menu of Section sec:variance_reduction, the Discussion's kernel-quadrature question.
**Validated by:** `radial_quadrature.py` (exact-Gaussian Gauss–Laguerre at machine precision by $D=32$ vs. MC at $5.4\times10^{-2}$; RFF-paired equal-allocation nodes worse than i.i.d., $0.38$ vs. $0.17$ at $D=32$).

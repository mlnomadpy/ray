# Theorem: Unbiasedness
**Label:** `thm:unbiased` | **Location:** main.tex line 189

## What it says
For the exact-modulation ⵟ-feature map $z$ of def:ryf (any $D, D'$, $\varepsilon > 0$, $b \ge 0$),
$$\mathbb{E}\bigl[z(x)^\top z(w)\bigr] = k_{ⵟ,b}(x, w) = \frac{(x^\top w + b)^2}{\|x - w\|^2 + \varepsilon}$$
for all $x, w \in \mathbb{R}^d$. The random-feature inner product is an exactly unbiased estimator of the biased ⵟ-kernel.

## Why it matters
Unbiasedness is the entry ticket for everything that follows: it makes $\widehat{K} = ZZ^\top$ a sum of independent mean-$K$ terms, so the exact variance (thm:exact_variance), Hoeffding/union uniform bounds (thm:uniform), matrix-Bernstein operator-norm bounds (thm:bernstein), and the whitened KRR guarantees all apply with no bias correction. It also underwrites rmk:learnable: gradients in $(\varepsilon, b)$ flow through an unbiased kernel estimate. The proof's structure — exact polynomial inner product times the radial expectation — is precisely what generalizes to the whole Bernstein–Schur class in thm:bernstein_schur, of which this theorem is the flagship instance.

## Proof idea
Three exact identities chain together. Write
$$z(x)^\top z(w) = \frac{1}{D}\sum_{j=1}^{D}\bigl(\psi_{t_j}(x)^\top \psi_{t_j}(w)\bigr)\bigl(p(x)^\top p(w)\bigr).$$
First, the polynomial inner product is exact, not estimated: $p(x)^\top p(w) = (x^\top w + b)^2/\varepsilon$ (prop:biased_feature). Second, the standard RFF identity gives, conditionally on the scale, $\mathbb{E}_\omega[\psi_{t_j}(x)^\top \psi_{t_j}(w) \mid t_j] = e^{-t_j\|x-w\|^2}$. Third, taking expectation over $t_j \sim \operatorname{Exp}(\varepsilon)$ and using eq:expectation,
$$\mathbb{E}[z(x)^\top z(w)] = \mathbb{E}_{T \sim \operatorname{Exp}(\varepsilon)}\bigl[e^{-T\|x-w\|^2}\bigr] \cdot \frac{(x^\top w + b)^2}{\varepsilon} = \frac{\varepsilon}{\|x-w\|^2 + \varepsilon} \cdot \frac{(x^\top w + b)^2}{\varepsilon} = k_{ⵟ,b}(x, w).$$
Nothing in the argument is specific to $k_{ⵟ,b}$ beyond its structure as a finite-feature kernel times a completely monotone shift-invariant kernel — the observation that launches the class.

## Connections
**Depends on:** def:ryf, prop:biased_feature (exact modulation inner product), eq:expectation (Bernstein–Widder expectation identity), the Rahimi–Recht RFF identity.
**Used by:** thm:exact_variance, thm:uniform, thm:bernstein, cor:bernstein_tail, thm:krr_spectral, thm:krr_whitened, thm:krr_leverage, cor:sample_complexity (all concentration is around this mean); rmk:learnable (gradients through an unbiased estimate); thm:ts_opnorm (the sketched estimator's conditional mean); generalized by thm:bernstein_schur.
**Validated by:** `gram_approx.py`, `make_fig1.py` (relative Frobenius error follows the $O(1/\sqrt{D})$ Monte-Carlo rate), `variance_validation.py`.

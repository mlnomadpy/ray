# Definition: Exact-Modulation ⵟ-Feature
**Label:** `def:ryf` | **Location:** main.tex line 172

## What it says
Given $D, D' \in \mathbb{N}$, $\varepsilon > 0$, $b \ge 0$, the **exact-modulation ⵟ-feature map** $z: \mathbb{R}^d \to \mathbb{R}^{M_b}$ with $M_b = D\,D'\,d_b$ (where $d_b = d(d+1)/2 + d + 1$ is the symmetrized polynomial feature dimension) is built as:

1. Draw radial scales $t_1, \dots, t_D \overset{\text{iid}}{\sim} \operatorname{Exp}(\varepsilon)$.
2. For each $j$, draw frequencies $\omega_{j,1}, \dots, \omega_{j,D'} \overset{\text{iid}}{\sim} \mathcal{N}(0, 2t_j I_d)$ and phases $\beta_{j,\ell} \overset{\text{iid}}{\sim} \operatorname{Unif}([0, 2\pi])$ (the factor 2 in the covariance is the standard RFF convention so that $\mathbb{E}_{\omega,\beta}[2\cos(\omega^\top x + \beta)\cos(\omega^\top w + \beta)] = e^{-t_j\|x-w\|^2}$).
3. Gaussian RFF at scale $t_j$: $\psi_{t_j}(x) = \sqrt{2/D'}\,\bigl(\cos(\omega_{j,\ell}^\top x + \beta_{j,\ell})\bigr)_{\ell=1}^{D'} \in \mathbb{R}^{D'}$.
4. Exact biased polynomial feature: $p(x) = \varepsilon^{-1/2} p_b(x) \in \mathbb{R}^{d_b}$.
5. Block: $z_j(x) = D^{-1/2}\,\psi_{t_j}(x) \otimes p(x)$; concatenate $z(x) = (z_1(x)^\top, \dots, z_D(x)^\top)^\top$.

**Spectral reading.** The hierarchical draw $t \sim \operatorname{Exp}(\varepsilon)$, $\omega \sim \mathcal{N}(0, 2tI_d)$ produces $\omega$ with marginal density $\int_0^\infty \mathcal{N}(\omega; 0, 2tI_d)\,\varepsilon e^{-\varepsilon t}\,dt$, which by the Gaussian scale mixture (eq:bernstein) is exactly the normalized Bochner spectral distribution of the rescaled radial factor $\varepsilon h_\varepsilon$; the missing total mass $1/\varepsilon$ is carried by the feature scaling. The scheme is therefore standard random Fourier features for the IMQ factor $h_\varepsilon$ tensored with the exact polynomial feature — the radial-scale mixture is a tractable sampler for the IMQ spectral distribution (expressible via modified Bessel functions), not a competitor to it.

**Flat form.** The inner level is redundant: at equal cost $DD'$, the $D'$ frequencies of a block share one scale $t_j$ and are correlated, whereas $D'$ independent spectral draws are not, so variance is minimized at $D' = 1$ except in the degenerate zero-outer-variance case. The recommended estimator draws $D$ independent pairs $(t_j, \omega_j)$ and collapses to (eq:flat):
$$z(x) = D^{-1/2}\Bigl(\sqrt{2}\,\cos(\omega_j^\top x + \beta_j)\,p(x)\Bigr)_{j=1}^{D}, \qquad \omega_j \sim \mathcal{N}(0, 2t_j I_d),\ t_j \sim \operatorname{Exp}(\varepsilon).$$

## Why it matters
This is the analyzable limit of the paper's estimator: with the modulation exact, the only randomness is radial, so unbiasedness (thm:unbiased), the exact variance (thm:exact_variance), the uniform Gram bound (thm:uniform), and all matrix-Bernstein and KRR guarantees of Section sec:guarantees are stated cleanly for this object. The deployed RAY map (Step 5, eq:doubly) is this definition with $p_b$ replaced by a TensorSketch — i.e., def:ryf is the $m \to \infty$ limit of RAY, and thm:ts_opnorm transfers its guarantees to finite $m$ at the cost of one additive sketch term. The flat $D'=1$ recommendation, proved optimal in prop:budget, fixes the deployed estimator's shape.

## Proof idea
Not a theorem, but the construction is forced step by step by the pipeline: Schur factorization (eq:schur) isolates a finite-feature modulation and a completely monotone radial factor; Bernstein–Widder (eq:bernstein) writes $h_\varepsilon$ as a Gaussian scale mixture whose normalized mixing law is $\operatorname{Exp}(\varepsilon)$ (eq:expectation); Monte-Carlo sampling of the one-dimensional scale (eq:mc) discretizes the mixture; Bochner applies to each Gaussian $g_{t_j}$ even though it does not apply to $k_{ⵟ,b}$, giving the RFF block; and tensoring with the exact $p_b$ multiplies the kernels back together.

## Connections
**Depends on:** eq:schur (Schur factorization), prop:biased_feature (the exact $p_b$), eq:bernstein / eq:expectation (Bernstein–Widder mixture and $\operatorname{Exp}(\varepsilon)$ sampling), the standard RFF identity of Rahimi–Recht.
**Used by:** thm:unbiased, thm:exact_variance, prop:budget (proves $D'=1$ optimal), prop:normalized, thm:uniform, thm:bernstein, cor:bernstein_tail, thm:krr_spectral, thm:krr_whitened, thm:krr_leverage; Step 5 / eq:doubly and thm:ts_opnorm (RAY sketches its modulation block); ex:ray; thm:bernstein_schur generalizes it to the class.
**Validated by:** `gram_approx.py`, `make_fig1.py` ($O(1/\sqrt{D})$ rate), `budget_allocation.py` (flat $D'=1$ optimality), `exact_variance_check.py`, `variance_validation.py`.

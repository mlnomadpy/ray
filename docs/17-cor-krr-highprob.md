# Corollary: High-Probability KRR Guarantee
**Label:** `cor:krr_highprob` | **Location:** main.tex line 336 (proof at lines 848–850)

## What it says

On the event $\rho_D \le \rho_0 < 1$ — so, under the draw count of `thm:krr_whitened`, with probability at least $1 - \delta$ — Theorem `thm:krr_spectral` applies:

$$(1-\rho_0)A \preceq K_D + \lambda I \preceq (1+\rho_0)A, \qquad \|\tilde\alpha - \hat\alpha\|_A \le \frac{\rho_0}{1-\rho_0}\|\hat\alpha\|_A.$$

Moreover the **optimal value of the kernel-ridge objective**,

$$\min_f \sum_i (f(x_i) - y_i)^2 + \lambda\|f\|_{\mathcal H}^2 = \lambda\, y^\top (K + \lambda I)^{-1} y,$$

is preserved to relative constants:

$$\lambda\, y^\top (K_D + \lambda I)^{-1} y \;\in\; \Bigl[\tfrac{1}{1+\rho_0},\ \tfrac{1}{1-\rho_0}\Bigr] \cdot \lambda\, y^\top (K + \lambda I)^{-1} y.$$

At $\rho_0 = \tfrac12$: the coefficient error is at most $\|\hat\alpha\|_A$ and the objective value is preserved within $[\tfrac23, 2]$.

## Why it matters

This is the end-user statement of the uniform-sampling KRR chain: it assembles the probabilistic condition (`thm:krr_whitened`) and the deterministic conversion (`thm:krr_spectral`) into one event-level guarantee, and adds the objective-value preservation that neither piece states alone. The objective-value sandwich is what the experiments check directly (it requires no access to $\hat\alpha$), and it is the statement that transfers unchanged to the leverage-tilted estimator $K_D^*$ (`rmk:risk` notes the coefficient and objective-value guarantees hold for $K_D^*$ as they stand), to the deployed sketched estimator (`cor:krr_deployed` concludes with this corollary verbatim), and to the whole Bernstein–Schur class (`thm:class_bernstein`).

## Proof idea

The sandwich and the coefficient bound are `thm:krr_spectral` evaluated on the event $\rho_D \le \rho_0$. For the objective value: the representer computation gives the optimal value $\lambda\, y^\top (K + \lambda I)^{-1} y$ for the exact problem — substitute $\hat\alpha = (K + \lambda I)^{-1} y$ into the objective:

$$\|K\hat\alpha - y\|^2 + \lambda\,\hat\alpha^\top K \hat\alpha = \lambda^2 y^\top A^{-2} y + \lambda\, y^\top A^{-1} K A^{-1} y = \lambda\, y^\top A^{-1} y,$$

and likewise $\lambda\, y^\top (K_D + \lambda I)^{-1} y$ for the feature-space ridge problem with Gram $K_D$. Operator anti-monotonicity of the inverse applied to $(1-\rho_0)A \preceq K_D + \lambda I \preceq (1+\rho_0)A$ gives

$$(1+\rho_0)^{-1} A^{-1} \preceq (K_D + \lambda I)^{-1} \preceq (1-\rho_0)^{-1} A^{-1};$$

evaluating the quadratic form at $y$ finishes.

## Connections

**Depends on:** `thm:krr_spectral` (deterministic sandwich + coefficient bound), `thm:krr_whitened` (probability of the event via the draw count), operator anti-monotonicity of the matrix inverse.
**Used by:** `thm:krr_leverage` / `rmk:risk` (the guarantees hold for the tilted $K_D^*$ unchanged), `cor:krr_deployed` (holds verbatim for the deployed Gram $\widehat K_{D,m}$ on the combined sketch+radial event), `thm:class_bernstein` (verbatim with $P \mapsto P_u$).
**Validated by:** `krr_whitened_bernstein.py` (objective value stays in the sandwich on every seed with $\rho_D < 1$), `krr_downstream.py`.

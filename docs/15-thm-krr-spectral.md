# Theorem: KRR Stability under Relative Spectral Approximation
**Label:** `thm:krr_spectral` | **Location:** main.tex line 314 (proof at lines 817–819)

## What it says

For a target $y \in \mathbb{R}^N$ and ridge $\lambda > 0$, let $\hat\alpha = (K + \lambda I)^{-1} y$ be the exact KRR coefficients, $\tilde\alpha = (K_D + \lambda I)^{-1} y$ the approximate ones, $A = K + \lambda I$, and $E = K_D - K$. If the **whitened** error obeys

$$\|A^{-1/2} E A^{-1/2}\|_{\mathrm{op}} \le \rho < 1,$$

then

$$(1-\rho) A \preceq K_D + \lambda I \preceq (1+\rho) A, \qquad \|\tilde\alpha - \hat\alpha\|_A \le \frac{\rho}{1-\rho}\,\|\hat\alpha\|_A,$$

where $\|v\|_A^2 = v^\top A v$. The theorem is **deterministic**: it converts any whitened operator-norm bound into a spectral sandwich and a relative coefficient guarantee in the natural $A$-norm. A cruder coefficient bound $\|\hat\alpha - \tilde\alpha\|_2 \le \lambda^{-2}\|K_D - K\|_{\mathrm{op}}\|y\|_2$ also holds, since $K_D = ZZ^\top \succeq 0$ makes $K_D + \lambda I$ invertible with inverse norm $\le 1/\lambda$.

## What governs the predictor

This is the spectral-approximation form standard in random-feature KRR analysis (Avron et al. 2017): the **relative** error $\rho$, not the raw $\|K_D - K\|_{\mathrm{op}}$, controls the downstream estimator. The whitening by $A^{-1/2}$ measures the Gram error against the ridge-protected spectrum — error in directions where $K + \lambda I$ is large is discounted.

## Why it matters

It is the deterministic interface between the concentration theory and kernel ridge regression. Everything probabilistic in the chain ( `thm:krr_whitened`, `thm:krr_leverage`, `cor:krr_deployed`) exists to supply the single hypothesis $\rho < 1$ with high probability; once supplied, this theorem and `cor:krr_highprob` deliver the sandwich, the coefficient bound, and the objective-value preservation. Separating the deterministic conversion from the probabilistic condition is what lets the same interface serve the uniform sampler, the leverage-tilted sampler, the sketched deployed estimator, and the entire Bernstein–Schur class without re-proving anything downstream.

## Proof idea

Set $B = A^{-1/2} E A^{-1/2}$. From $\|B\|_{\mathrm{op}} \le \rho$, $-\rho I \preceq B \preceq \rho I$; conjugating by $A^{1/2}$ gives $-\rho A \preceq E \preceq \rho A$, and since $K_D + \lambda I = A + E$ this is the sandwich. For the coefficients, the resolvent identity gives

$$\tilde\alpha - \hat\alpha = (A+E)^{-1} y - A^{-1} y = -(A+E)^{-1} E \hat\alpha,$$

so $A^{1/2}(\tilde\alpha - \hat\alpha) = -(I+B)^{-1} B\, A^{1/2}\hat\alpha$. With $\|B\|_{\mathrm{op}} \le \rho < 1$, the Neumann bound $\|(I+B)^{-1}\|_{\mathrm{op}} \le (1-\rho)^{-1}$ yields $\|\tilde\alpha - \hat\alpha\|_A \le \frac{\rho}{1-\rho}\|\hat\alpha\|_A$.

## Connections

**Depends on:** nothing probabilistic — pure linear algebra (Loewner order conjugation, resolvent identity, Neumann series).
**Used by:** `thm:krr_whitened` (supplies its hypothesis with high probability), `cor:krr_highprob` (applies it on the event $\rho_D \le \rho_0$ and adds objective-value preservation), `thm:krr_leverage` (same interface, tilted sampler), `cor:krr_deployed` (sketched estimator), `rmk:risk` (its limits: a sandwich alone does not bound risk), `thm:class_bernstein` (transfers with the class modulation Gram).
**Validated by:** `krr_downstream.py` ($1/\sqrt D$ convergence of the downstream predictor, Section sec:exp_krr; Table tab:krr_sketched), `krr_spectral.py`.

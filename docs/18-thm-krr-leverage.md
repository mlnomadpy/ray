# Theorem: Leverage-Weighted Radial Sampling Achieves the Effective-Dimension Count
**Label:** `thm:krr_leverage` | **Location:** main.tex line 346 (proof at lines 836–846)

## What it says

Let $\theta = (t, \omega, \beta)$ denote one radial draw with base law $\pi$ ($t \sim \mathrm{Exp}(\varepsilon)$, $\omega \mid t \sim \mathcal{N}(0, 2tI_d)$, $\beta \sim \mathrm{Unif}[0, 2\pi]$), per-draw Gram $K^{(\theta)} = (\psi_\theta \psi_\theta^\top) \circ P$ with $\psi_\theta[i] = \sqrt2\cos(\omega^\top x_i + \beta)$, and fix $\lambda > 0$, $A = K + \lambda I$. Define the **whitened draw leverage**

$$\bar d_\lambda(\theta) := \operatorname{tr}\bigl(A^{-1} K^{(\theta)}\bigr) = \psi_\theta^\top \bigl(A^{-1} \circ P\bigr)\psi_\theta \;\ge\; 0, \qquad \mathbb{E}_\pi[\bar d_\lambda] = d_{\mathrm{eff}}(\lambda)$$

— the mean over the base law is **exactly** the ridge effective dimension. Draw $\theta_1, \dots, \theta_D$ i.i.d. from the tilted law $d\pi^*_\lambda = (\bar d_\lambda / d_{\mathrm{eff}})\, d\pi$ and set $K^*_D = \frac1D \sum_j \frac{d_{\mathrm{eff}}}{\bar d_\lambda(\theta_j)} K^{(\theta_j)}$. Then $K^*_D$ is unbiased, and with probability at least $1 - \delta$, with $\ell = \log(8\tilde d_\lambda/\delta)$,

$$\rho_D = \bigl\|A^{-1/2}(K^*_D - K)A^{-1/2}\bigr\|_{\mathrm{op}} \;\le\; \sqrt{\frac{2\, d_{\mathrm{eff}}(\lambda)\,\kappa_\lambda\,\ell}{D}} \;+\; \frac{2}{3}\,\frac{(1 + d_{\mathrm{eff}}(\lambda))\,\ell}{D}.$$

In particular, for any $\rho_0 \in (0,1]$,

$$D \;\ge\; \frac{8}{\rho_0^2}\bigl(1 + d_{\mathrm{eff}}(\lambda)\bigr)\log\frac{8\tilde d_\lambda}{\delta} \qquad\Longrightarrow\qquad \mathbb{P}\{\rho_D \le \rho_0\} \ge 1 - \delta,$$

replacing the factor $1 + \|P\|_{\mathrm{op}}/\lambda$ of `thm:krr_whitened` by $1 + d_{\mathrm{eff}}(\lambda)$. Since $d_{\mathrm{eff}}(\lambda) \le N$ for every $\lambda$, the leverage count never exceeds $O(N\log(\tilde d_\lambda/\delta))$, whereas the uniform count grows without bound as $\lambda \to 0$.

## Why it matters

This is the leverage-score result of the random-feature KRR literature (Avron et al. 2017, Li et al. 2021), obtained on the **Gram side** with the same matrix-Bernstein proof — and it tilts the **joint** radial draw $(t, \omega, \beta)$, not the scale alone. The uniform count of `thm:krr_whitened` pays $\|P\|_{\mathrm{op}}/\lambda$ because plain $\mathrm{Exp}(\varepsilon)$ sampling carries the worst-case per-draw norm; the right tilt is computable in closed form as the trace of the whitened per-draw Gram, a quadratic form in $\psi_\theta$ with the fixed PSD matrix $A^{-1} \circ P$. Interpretation: draws whose cosine pattern injects energy where $A^{-1}$ is large — the small-eigenvalue directions of $K$ that the ridge does not protect — are over-sampled and down-weighted. Exact tilting needs $A^{-1} \circ P$ (the usual leverage chicken-and-egg); the standard answers (pilot uniform estimate, Nyström or sketched approximations of $A^{-1}$, iterative reweighting) apply verbatim since the proof only uses the importance identity and the trace normalization.

## Proof idea

**Leverage identity.** With $D_\psi = \operatorname{diag}(\psi_\theta)$, $K^{(\theta)} = D_\psi P D_\psi$, so $\operatorname{tr}(A^{-1}K^{(\theta)}) = \psi_\theta^\top(A^{-1} \circ P)\psi_\theta$. The matrix $A^{-1} \circ P$ is a Schur product of PSD matrices, hence PSD, so $\bar d_\lambda \ge 0$; and $\mathbb{E}_\pi[\psi_\theta\psi_\theta^\top] = R$ (the unit-diagonal radial Gram) gives $\mathbb{E}_\pi[\bar d_\lambda] = \operatorname{tr}(A^{-1}(R \circ P)) = \operatorname{tr}(A^{-1}K) = d_{\mathrm{eff}}(\lambda)$, so $\pi^*_\lambda$ is a probability law. On $\{\bar d_\lambda = 0\}$ the PSD matrix $A^{-1/2}K^{(\theta)}A^{-1/2}$ has zero trace, hence is zero, so discarding those draws loses nothing; unbiasedness follows from the importance identity.

**Whitened ingredients.** Set $M_\theta = \frac{d_{\mathrm{eff}}}{\bar d_\lambda(\theta)} A^{-1/2} K^{(\theta)} A^{-1/2} \succeq 0$.
- **(a) A.s. bound.** On the support of $\pi^*_\lambda$, $\|M_\theta\|_{\mathrm{op}} \le \operatorname{tr}(M_\theta) = \frac{d_{\mathrm{eff}}}{\bar d_\lambda}\operatorname{tr}(A^{-1}K^{(\theta)}) = d_{\mathrm{eff}}$ — a PSD matrix's norm is at most its trace, and **the tilt normalizes the trace exactly**. Hence $L^* = (1 + d_{\mathrm{eff}})/D$.
- **(b) Variance majorant.** $M_\theta^2 \preceq d_{\mathrm{eff}} M_\theta$ and $\mathbb{E}_{\pi^*_\lambda}[M_\theta] = A^{-1/2}KA^{-1/2}$, giving $\widetilde V^* = \frac{d_{\mathrm{eff}}}{D} A^{-1/2}KA^{-1/2}$.
- **(c) Intrinsic dimension.** $\widetilde V^*$ is the **same core matrix** $A^{-1/2}KA^{-1/2}$ as in `thm:krr_whitened`, rescaled, so $\operatorname{intdim}(\widetilde V^*) = \tilde d_\lambda$ exactly and $\|\widetilde V^*\|_{\mathrm{op}} = d_{\mathrm{eff}}\kappa_\lambda/D$.

Intrinsic matrix Bernstein on $\pm\sum_j \widetilde Y_j$, inverted as in `cor:bernstein_tail`, gives the tail; the count follows with $\kappa_\lambda \le 1$, $\rho_0 \le 1$.

## Numerical confirmation

On the protocol of `thm:krr_whitened` ($N = 300$, $\|P\|_{\mathrm{op}} = 306$): the uniform draw count $D^*(\rho_D \le \tfrac12)$ climbs $50 \to 800 \to {>}3200$ as $\lambda$ drops $10 \to 1 \to 0.1$, while the leverage-tilted sampler reaches it at $D^* = 50/200/400/1600$ across $\lambda \in \{10, 1, 0.1, 0.01\}$, tracking $d_{\mathrm{eff}} = 12.3/38.6/90.8/163.3$. The identity $\mathbb{E}_\pi[\bar d_\lambda] = d_{\mathrm{eff}}$ holds on a $5 \times 10^4$-draw pool to within 2%, and both samplers keep the $O(D^{-1/2})$ whitened rate (fitted slopes $-0.47$ to $-0.63$).

## Connections

**Depends on:** `thm:krr_whitened` (same core variance matrix, same intrinsic-dimension identity), `lem:schur` (PSD-ness of $A^{-1} \circ P$), `cor:bernstein_tail` (inversion), the importance identity $\mathbb{E}_\pi[\psi\psi^\top] = R$, intrinsic matrix Bernstein.
**Used by:** `rmk:risk` (reading of the counts; the remaining open computational question of approximating the tilt), `cor:krr_highprob` (its guarantees hold for $K^*_D$ as they stand), `thm:class_bernstein` (transfers verbatim with $P \mapsto P_u$ since the proof only consumes facts (i)–(iii) plus $\mathbb{E}_\pi[\psi\psi^\top] = R_u$).
**Validated by:** `experiments/leverage_radial_sampling.py` (Appendix app:exp_details, results in `experiments/results/leverage_radial_sampling.json`).

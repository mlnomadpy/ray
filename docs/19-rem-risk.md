# Remark: From Spectral Approximation to Risk, and What Leverage Buys
**Label:** `rmk:risk` | **Location:** main.tex line 365

## What it says

Three reads of `thm:krr_whitened` and `thm:krr_leverage`.

**(i) The counts.** $D = O\bigl((1 + \|P\|_{\mathrm{op}}/\lambda)\log d_{\mathrm{eff}}\bigr)$ is the plain-i.i.d.-sampling rate — the Gram-side analogue of the $\Omega(1/\lambda)$ feature counts for **uniformly** sampled random Fourier features in KRR (Avron et al. 2017, Rudi–Rosasco 2017, Bach 2017). `thm:krr_leverage` improves the leading factor to $d_{\mathrm{eff}}(\lambda)$ itself — the ridge-leverage count of Avron et al. (2017) and Li et al. (2021) — by tilting the **joint** radial draw $(t, \omega, \beta)$ rather than the scale alone. What remains open on this axis is purely computational: a deployment-grade approximation of the tilt $\bar d_\lambda$ (pilot estimates, Nyström approximations of $A^{-1}$) with the approximation error folded into the bound.

**(ii) Rates.** The condition $\rho \le \tfrac12$ at the regularization schedule $\lambda = \lambda_N$ of the source/capacity conditions is the interface assumption of the random-feature KRR literature, and both counts stay polynomial in $N$ along any such schedule — the leverage count uniformly so, since $d_{\mathrm{eff}} \le N$.

**(iii) An honest caveat.** A spectral sandwich alone does **not** bound the fixed-design risk. Counterexample: $K = 0$ with $K' = \rho\lambda I$ satisfies the $\rho$-sandwich

$$(1-\rho)(K + \lambda I) \preceq K' + \lambda I \preceq (1+\rho)(K + \lambda I),$$

yet inflates the in-sample variance term from $0$ to $\sigma^2\rho^2/(1+\rho)^2$. A risk theorem therefore needs, in addition, control of the **approximate** kernel's effective dimension $d_{\mathrm{eff}}(K^*_D)$, which a plain sandwich does not supply. With the count side now closed by `thm:krr_leverage`, this is the **one missing ingredient** between that theorem and a minimax-optimal RAY-KRR statement. The coefficient and objective-value guarantees of `cor:krr_highprob` hold for $K^*_D$ as they stand.

## Why it matters

This remark is the paper's calibration of its own KRR claims. It places the two draw counts precisely in the random-feature KRR literature (Gram-side analogues of the uniform and ridge-leverage feature counts), states exactly what is achieved (spectral approximation, coefficient stability, objective-value preservation, polynomial counts along source/capacity schedules) and exactly what is not (a fixed-design risk bound), and exhibits the $K = 0$ vs $K' = \rho\lambda I$ counterexample showing the gap is real rather than a proof artifact. Without it, the chain `thm:krr_whitened` → `thm:krr_leverage` → `cor:krr_highprob` could be misread as a generalization-error guarantee. It also names the two open items on this axis: a deployable approximation of the leverage tilt with its error folded into the bound, and control of $d_{\mathrm{eff}}(K^*_D)$ for a minimax statement.

## Proof idea

Not a theorem; the one mathematical claim is the counterexample. Take $K = 0$, $K' = \rho\lambda I$. Then $K + \lambda I = \lambda I$ and $K' + \lambda I = (1+\rho)\lambda I$, so the sandwich holds with constant $\rho$. The exact KRR predictor for $K = 0$ is identically zero — in-sample variance $0$ — while ridge regression with Gram $K'$ has hat matrix $K'(K' + \lambda I)^{-1} = \frac{\rho}{1+\rho} I$, contributing in-sample variance $\sigma^2\rho^2/(1+\rho)^2 > 0$. The fixed-design risk thus moves by an amount the sandwich cannot see; what the sandwich misses is precisely $d_{\mathrm{eff}}$ of the approximate kernel ($d_{\mathrm{eff}}(K') = N\rho/(1+\rho)$ vs $d_{\mathrm{eff}}(K) = 0$).

## Connections

**Depends on:** `thm:krr_whitened` (uniform count), `thm:krr_leverage` (leverage count), `cor:krr_highprob` (which guarantees do survive), the random-feature KRR literature for the comparison points.
**Used by:** frames the paper's open problems on the KRR axis (deployable tilt approximation; risk theorem via $d_{\mathrm{eff}}(K^*_D)$ control); the scoping of claims in the introduction and conclusion.
**Validated by:** none (positioning remark; the counterexample is exact algebra).

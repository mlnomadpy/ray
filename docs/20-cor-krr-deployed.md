# Corollary: Ridge-Relative KRR Condition for the Deployed Estimator
**Label:** `cor:krr_deployed` | **Location:** main.tex line 371

## What it says

The whitened high-probability KRR condition (`thm:krr_whitened`), proved for the exact-modulation estimator, carries to the deployed doubly-randomized RAY estimate by the same conditioning device that carried `cor:bernstein_tail` to `thm:ts_opnorm`.

**Setup.** Let $\widehat K_{D,m}$ be the RAY estimate (eq:doubly) and fix a target $\rho_0 \in (0,1]$. Suppose the sketch is a $(1\pm\eta)$ ridge embedding of $P$ at scale $\lambda$ (`rmk:ose`) with

$$\eta \le \rho_0/4.$$

**Conclusion.** On the sketch event, with probability at least $1-\delta$ over the radial draws,

$$D \;\ge\; \frac{64}{\rho_0^{2}}\Bigl(1 + \frac{(1+\eta)\|P\|_{\mathrm{op}}}{\lambda}\Bigr)\log\frac{8\tilde d_{\lambda,S}}{\delta} \qquad\Longrightarrow\qquad \bigl\|A^{-1/2}(\widehat K_{D,m} - K)A^{-1/2}\bigr\|_{\mathrm{op}} \le \rho_0,$$

with $A = K + \lambda I$ and $\tilde d_{\lambda,S}$ the intrinsic dimension of the sketch-conditioned whitened variance. Consequently `cor:krr_highprob` (coefficient and objective-value guarantees) holds for the deployed Gram $\widehat K_{D,m}$ verbatim.

**The scale-free punchline.** The sketch requirement is $\eta \le \rho_0/4$ — a constant, rather than the $\eta \lesssim \lambda/\|P\|_{\mathrm{op}}$ that the absolute-event route would demand — and the sketch size is polynomial in the statistical dimension $s_\lambda(P)$ rather than $\operatorname{rank}(P)$. The radial count carries $\log \tilde d_{\lambda,S}$ rather than worst-case $\log N$: the ridge embedding gives $\tilde d_{\lambda,S} \le (\tilde d_\lambda + \eta N \kappa_\lambda^{-1})/(1-\eta)$, so the count is the genuine effective-dimension count $\log \tilde d_\lambda$ once $\eta \lesssim \tilde d_\lambda/N$, degrading to $\log N$ only as the sketch coarsens. The honest content of the transfer: the sketch-*accuracy* requirement drops to a constant unconditionally, while the logarithmic dimension is intrinsic only when the sketch is fine enough.

## Why it matters

This corollary is where the paper's KRR theory meets the object one actually runs. `thm:krr_whitened` gives the effective-dimension radial count, but only for the exact-modulation estimator with its $O(d^2)$ feature floor; `cor:krr_deployed` certifies that the $Dm$-dimensional deployed map preserves kernel-ridge coefficients and objective value under the same kind of count, at the price of a constant-accuracy sketch. Without it, deploying the sketch would either void the KRR guarantee or force the scale-dependent sketch accuracy $\eta \lesssim \lambda/\|P\|_{\mathrm{op}}$, which blows up as $\lambda \to 0$ exactly where KRR operates. It is the terminal node of the conditioning pipeline: `prop:ridge_sketch` handles the sketch bias, `thm:krr_whitened` handles the radial fluctuation, and this corollary splices them at the deployed estimator.

## Proof idea

Split the whitened error at the sketch-conditioned target $K_S = \widehat P_m \circ R$ and budget $\rho_0$ as (sketch part) + (radial part).

**Sketch part.** `prop:ridge_sketch` gives, deterministically on the sketch event, $\rho_{\mathrm{sk}} := \|A^{-1/2}(K_S - K)A^{-1/2}\|_{\mathrm{op}} \le \eta$.

**Radial part.** Conditioned on the sketch, $\widehat K_{D,m}$ is an exact-modulation estimate of $K_S$, so `thm:krr_whitened` applies to the pair $(\widehat P_m, K_S)$ with $\|\widehat P_m\|_{\mathrm{op}} \le (1+\eta)\|P\|_{\mathrm{op}}$, target $\rho_0/2$, whitened by $A_S = K_S + \lambda I$: under the stated count, $\|A_S^{-1/2}(\widehat K_{D,m} - K_S)A_S^{-1/2}\|_{\mathrm{op}} \le \rho_0/2$.

**Splice.** Since $A_S \preceq (1+\eta)A$, the $A$-whitened norm is at most $(1+\eta)$ times the $A_S$-whitened one. The triangle inequality then needs $(1+\eta)\frac{\rho_0}{2} + \eta \le \rho_0$, which holds precisely under $\eta \le \rho_0/4$ — the origin of the scale-free constant.

(The appendix proof now follows the same conditioning route as the statement: apply `thm:krr_whitened` to the sketch-conditioned pair $(\widehat P_m,K_S)$ and keep the intrinsic log $\log(8\tilde d_{\lambda,S}/\delta)$ rather than falling back to $\log N$.)

## Connections

**Depends on:** `prop:ridge_sketch` (deterministic sketch part), `thm:krr_whitened` (radial part, applied to the conditioned pair $(\widehat P_m, K_S)$), `rmk:ose` (the ridge-embedding event and its $s_\lambda(P)$ sketch size), `lem:schur`, eq:doubly, the conditioning device of `thm:ts_opnorm`.
**Used by:** `cor:krr_highprob` is extended to the deployed Gram verbatim; the Discussion's deployment narrative (effective-dimension counts for the map one actually runs); `thm:class_bernstein` extends the whole guarantee set to every Bernstein–Schur kernel.
**Validated by:** — (the constituent pieces are validated separately: `krr_whitened_bernstein.py` for `thm:krr_whitened`, `ts_opnorm_validation.py` for the conditioning split, `krr_spectral_sketched.py` for sketched-KRR behavior).

# Theorem: Whitened Matrix Bernstein — the High-Probability KRR Condition
**Label:** `thm:krr_whitened` | **Location:** main.tex line 323 (proof at lines 821–834)

## What it says

Let $A = K + \lambda I$ with $\lambda > 0$, let $K_D$ be the exact-modulation estimate from $D$ i.i.d. radial draws, and set $\rho_D = \|A^{-1/2}(K_D - K)A^{-1/2}\|_{\mathrm{op}}$. With the ridge effective dimension $d_{\mathrm{eff}}(\lambda) = \operatorname{tr}(K A^{-1})$, $\kappa_\lambda = \|K\|_{\mathrm{op}}/(\|K\|_{\mathrm{op}} + \lambda) < 1$, and $\tilde d_\lambda = d_{\mathrm{eff}}(\lambda)/\kappa_\lambda$, with probability at least $1 - \delta$,

$$\rho_D \;\le\; 2\sqrt{\frac{\kappa_\lambda\,\|P\|_{\mathrm{op}}\,\log(8\tilde d_\lambda/\delta)}{\lambda\, D}} \;+\; \frac{2}{3}\Bigl(1 + \frac{2\|P\|_{\mathrm{op}}}{\lambda}\Bigr)\frac{\log(8\tilde d_\lambda/\delta)}{D}.$$

In particular, for any target $\rho_0 \in (0,1]$,

$$D \;\ge\; \frac{16}{\rho_0^2}\Bigl(1 + \frac{\|P\|_{\mathrm{op}}}{\lambda}\Bigr)\log\frac{8\tilde d_\lambda}{\delta} \qquad\Longrightarrow\qquad \mathbb{P}\{\rho_D \le \rho_0\} \ge 1 - \delta.$$

The decisive structural point: the effective dimension $d_{\mathrm{eff}}(\lambda)$ is **not an assumption imported from the KRR literature** — it is the **exact** intrinsic dimension of the whitened matrix variance. The variance majorant of the whitened summands is $\frac{2\|P\|_{\mathrm{op}}}{\lambda D} A^{-1/2} K A^{-1/2}$, and $\operatorname{intdim}(A^{-1/2} K A^{-1/2}) = d_{\mathrm{eff}}(\lambda)/\kappa_\lambda = \tilde d_\lambda$ identically.

## Why it matters

`thm:krr_spectral` is deterministic and leaves its hypothesis $\rho < 1$ open; this theorem closes it. It is the bridge from the Gram-side matrix Bernstein machinery to a usable KRR condition: the dimension appearing in the log is the ridge effective dimension, which is the quantity the random-feature KRR literature postulates — here it falls out of the proof as an identity. The draw count carries the factor $\|P\|_{\mathrm{op}}/\lambda$, the price of plain $\mathrm{Exp}(\varepsilon)$ sampling paying the worst-case per-draw norm; that factor is exactly what `thm:krr_leverage` removes by tilting. The theorem also transfers verbatim to the deployed sketched estimator (`cor:krr_deployed`, by conditioning) and to every Bernstein–Schur kernel (`thm:class_bernstein`).

## Proof idea

Write $A^{-1/2}(K_D - K)A^{-1/2} = \sum_j \widetilde Y_j$ with $\widetilde Y_j = D^{-1} A^{-1/2}(K^{(j)} - K)A^{-1/2}$, i.i.d. mean zero. Whiten the three ingredients of the proof of `thm:bernstein`:

- **(a) A.s. bound.** $K^{(j)} \succeq 0$ with $\|K^{(j)}\|_{\mathrm{op}} \le 2\|P\|_{\mathrm{op}}$ gives $0 \preceq A^{-1/2} K^{(j)} A^{-1/2} \preceq (2\|P\|_{\mathrm{op}}/\lambda) I$; and $K \preceq A$ gives $0 \preceq A^{-1/2} K A^{-1/2} \preceq I$. Hence $\|\widetilde Y_j\|_{\mathrm{op}} \le L_\lambda := (1 + 2\|P\|_{\mathrm{op}}/\lambda)/D$. Per-draw positivity survives conjugation.
- **(b) Variance majorant.** For PSD $M$, $M A^{-1} M = M^{1/2}(M^{1/2} A^{-1} M^{1/2})M^{1/2} \preceq \lambda^{-1}\|M\|_{\mathrm{op}} M$; with $M = K^{(j)}$, $\mathbb{E}[K^{(j)} A^{-1} K^{(j)}] \preceq (2\|P\|_{\mathrm{op}}/\lambda) K$, so
$$\sum_j \mathbb{E}[\widetilde Y_j^2] \preceq \widetilde V := \frac{2\|P\|_{\mathrm{op}}}{\lambda D}\, A^{-1/2} K A^{-1/2}.$$
- **(c) Intrinsic dimension.** $\|\widetilde V\|_{\mathrm{op}} = \frac{2\|P\|_{\mathrm{op}}}{\lambda D}\kappa_\lambda$ and $\operatorname{tr}(\widetilde V) = \frac{2\|P\|_{\mathrm{op}}}{\lambda D} d_{\mathrm{eff}}(\lambda)$, so $\operatorname{intdim}(\widetilde V) = d_{\mathrm{eff}}(\lambda)/\kappa_\lambda = \tilde d_\lambda$ **exactly**.

The intrinsic-dimension matrix Bernstein inequality accepts the semidefinite majorant $\widetilde V$; applied to $\pm\sum_j \widetilde Y_j$ (the two-tail union gives the prefactor 8) and inverted by the quadratic-root computation of `cor:bernstein_tail`, it yields the tail bound. The count follows by splitting the target $\rho_0$ between the two terms and using $\kappa_\lambda \le 1$, $1 + 2s \le 2(1+s)$, $\rho_0 \le 1$.

## Numerical confirmation

Across $\lambda \in \{10, 1, 0.1, 0.01\}$ the intrinsic dimension of $A^{-1/2} K A^{-1/2}$ matches $\tilde d_\lambda$ to machine precision (relative error $\le 4 \times 10^{-15}$); the whitened $\rho_D$ decays at the $O(D^{-1/2})$ rate; the objective value stays in the `cor:krr_highprob` sandwich on every seed where $\rho_D < 1$; and the class instance of `thm:class_bernstein` — the polynomially modulated Matérn-½ kernel with its Lévy sampler — reproduces all three.

## Connections

**Depends on:** `thm:bernstein` (the three ingredients, whitened), `lem:schur`, `cor:bernstein_tail` (quadratic-root inversion), intrinsic matrix Bernstein (Tropp 2015, Thm. 7.3.1).
**Used by:** `cor:krr_highprob` (supplies the event), `thm:krr_leverage` (same core variance matrix $A^{-1/2} K A^{-1/2}$; replaces $\|P\|_{\mathrm{op}}/\lambda$ by $d_{\mathrm{eff}}$), `cor:krr_deployed` (applied to $(\widehat P_m, K_S)$ conditioned on the sketch), `rmk:risk`, `thm:class_bernstein` (verbatim with $P \mapsto P_u$).
**Validated by:** `krr_whitened_bernstein.py` (intrinsic-dimension identity, whitened rate, sandwich, Matérn-½ instance; Appendix app:exp_details).

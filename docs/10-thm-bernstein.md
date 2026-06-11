# Theorem: Expected Matrix-Bernstein Operator-Norm Bound
**Label:** `thm:bernstein` | **Location:** main.tex line 269 (proof at lines 798–811)

## What it says

Let $K = [k_{ⵟ,b}(x_i, x_j)]$ be the kernel Gram matrix and $P = [(x_i^\top x_j + b)^2/\varepsilon]$ the polynomial (modulation) Gram matrix — both PSD — and let $K_D$ be the exact-modulation estimate from $D$ i.i.d. radial draws, $K_D = \frac1D \sum_j K^{(j)}$ with $K^{(j)} = (\Psi_j\Psi_j^\top) \circ P$. Let $V = \sum_j \mathbb{E}[(D^{-1}(K^{(j)} - K))^2]$ be the matrix variance and

$$d_{\mathrm{int}} = \frac{\operatorname{tr}(V)}{\|V\|_{\mathrm{op}}} \le N$$

its intrinsic dimension (with the convention $d_{\mathrm{int}} = 1$ when $V = 0$, in which case the estimate is exact). Then

$$\mathbb{E}\,\|K_D - K\|_{\mathrm{op}} \;\le\; 3\sqrt{\frac{2\|P\|_{\mathrm{op}}\,\|K\|_{\mathrm{op}}\,\log(8d_{\mathrm{int}})}{D}} \;+\; \frac{6\|P\|_{\mathrm{op}}\,\log(8d_{\mathrm{int}})}{D}.$$

The leading term scales with the **top eigenvalues** $\|P\|_{\mathrm{op}}, \|K\|_{\mathrm{op}}$ and the **effective rank** $d_{\mathrm{int}}$ — all $\ll N$ for spectrally concentrated data. Since $\|P\|_{\mathrm{op}} \le \operatorname{tr}(P) \le N(R^2+b)^2/\varepsilon$ and $d_{\mathrm{int}} \le N$, the bound never exceeds the order of the elementary route (Corollary `thm:gram_concentration`, $N\max_{ij}|\cdot|$) and is data-adaptively tighter.

## Why it matters

This is the upgrade from entrywise to spectral control, and the entry point of the whole guarantee chain. The Hoeffding-plus-union route to operator norm costs a factor $N$ ($\|A\|_{\mathrm{op}} \le N\max_{ij}|A_{ij}|$), which throws away the structure of the per-draw error matrices. The theorem exploits the structure: each radial draw contributes a **PSD** matrix $K^{(j)} = (\Psi_j\Psi_j^\top) \circ P$ (a Schur product of PSD matrices), so matrix Bernstein with an intrinsic dimension applies and the price drops from $N$ to top eigenvalues times $\log d_{\mathrm{int}}$. Everything downstream — the tail bound (`cor:bernstein_tail`), the whitened KRR condition (`thm:krr_whitened`), the leverage-tilted improvement (`thm:krr_leverage`), the deployed-estimator bound (`thm:ts_opnorm`), and the class-level transfer (`thm:class_bernstein`) — is this argument re-run with different conjugations, weights, or modulation Grams. Without it the paper has only entrywise Gram control.

## Proof idea

Write $K_D - K = \sum_{j=1}^D Y_j$ with $Y_j = D^{-1}(K^{(j)} - K)$, i.i.d. symmetric, mean zero by unbiasedness (`thm:unbiased`). Two structural facts from `lem:schur`: $K^{(j)} \succeq 0$ (Schur product of PSD factors), and $\|K^{(j)}\|_{\mathrm{op}} \le 2\|P\|_{\mathrm{op}}$ since $(\Psi_j\Psi_j^\top)_{ii} = \|\psi_{t_j}(x_i)\|^2 \le 2$; likewise $\|K\|_{\mathrm{op}} = \|R \circ P\|_{\mathrm{op}} \le \|P\|_{\mathrm{op}}$ with $R$ the unit-diagonal radial Gram.

- **A.s. bound.** $\|Y_j\|_{\mathrm{op}} \le D^{-1}(\|K^{(j)}\|_{\mathrm{op}} + \|K\|_{\mathrm{op}}) \le 3\|P\|_{\mathrm{op}}/D =: L$.
- **Matrix variance.** For PSD $M$, $M^2 \preceq \|M\|_{\mathrm{op}} M$; with $\|K^{(j)}\|_{\mathrm{op}} \le 2\|P\|_{\mathrm{op}}$ this gives $(K^{(j)})^2 \preceq 2\|P\|_{\mathrm{op}} K^{(j)}$, hence $\mathbb{E}[(K^{(j)} - K)^2] = \mathbb{E}[(K^{(j)})^2] - K^2 \preceq 2\|P\|_{\mathrm{op}} K$ and
$$V \preceq \frac{2\|P\|_{\mathrm{op}}}{D} K, \qquad \|V\|_{\mathrm{op}} \le \frac{2\|P\|_{\mathrm{op}}\|K\|_{\mathrm{op}}}{D} =: v.$$
- **Conclusion.** Integrating the two-sided intrinsic tail with $\ell_0=\log(8d_{\mathrm{int}})$ gives $\mathbb{E}\|\sum_j Y_j\|_{\mathrm{op}} \le 3\sqrt{v\ell_0}+2L\ell_0$; substituting $v$ and $L$ yields the stated bound. This avoids treating Tropp's ambient-dimension expectation theorem as if it were directly intrinsic-dimension-valued.

The pivotal step is $(K^{(j)})^2 \preceq 2\|P\|_{\mathrm{op}} K^{(j)}$ — per-draw positivity converts the second moment into a multiple of the **mean**, so the variance is controlled by $K$ itself rather than a worst-case envelope.

## Connections

**Depends on:** `lem:schur` (parts a–b for per-draw and kernel norms), `thm:unbiased` (mean-zero summands), the intrinsic-dimension matrix Bernstein inequality of Tropp (2012, 2015).
**Used by:** `cor:bernstein_tail` (same $L, v$, tail form), `thm:ts_opnorm` (applied verbatim conditioned on the sketch, $P \mapsto \widehat P_m$), `thm:krr_whitened` (the same three ingredients whitened by $A^{-1/2}$), `thm:class_bernstein` (verbatim with $P \mapsto P_u = m_f G_u$); discussed against `thm:gram_concentration` as the data-adaptive replacement of the factor-$N$ route.
**Validated by:** `bernstein_intrinsic.py` (data-adaptive constant across datasets of varying spectral spread, Section sec:exp_gram), `opnorm_validation.py`.

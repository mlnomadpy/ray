# Theorem: Uniform Gram Approximation
**Label:** `thm:uniform` | **Location:** main.tex line 770

## What it says
For a dataset $X = \{x_1, \dots, x_N\}$ with $\|x_i\| \le R$, the exact-modulation estimator with $D$ outer radial samples satisfies, with probability at least $1-\delta$,

$$\sup_{i,j \in [N]} \bigl| z(x_i)^\top z(x_j) - k_{ⵟ,b}(x_i, x_j) \bigr| \le \frac{(R^2+b)^2}{\varepsilon}\sqrt{\frac{8\log(2N^2/\delta)}{D}}.$$

The constant uses the almost-sure bound $|\psi_t(x)^\top \psi_t(w)| \le 2$ on the trigonometric factor, not the expectation-level value $1$.

## Why it matters
This is the entrywise concentration backbone of the dataset-level guarantee chain: it converts the per-pair variance picture into a simultaneous bound over all $N^2$ Gram entries with only a logarithmic cost in $N$, and it carries no explicit dependence on the input dimension $d$. From it follow immediately the sample complexity `cor:sample_complexity` and the crude operator-norm bound `thm:gram_concentration` (the corollary whose wasteful factor $N$ motivates the matrix-Bernstein development of `thm:bernstein`). It also calibrates the deterministic radial quadrature comparison (line 467: the quadrature bias $\tau(R^2+b)^2$ is "free of the $O(D^{-1/2})$ radial fluctuation of Theorem thm:uniform") and anchors the honest comparison of Table tab:rf_comparison, where the paper is explicit that dimension-freeness at this dataset level is a property of the Hoeffding-plus-union analysis, shared by standard RFF, not a special feature of the construction.

## Proof idea
Fix a pair $(x_i, x_j)$. The estimator averages $D$ i.i.d. terms $Y_k \in [-c, c]$ with $c = 2(R^2+b)^2/\varepsilon$: the polynomial factor is deterministic and bounded by $(R^2+b)^2/\varepsilon$ (Cauchy–Schwarz), and the trigonometric factor obeys the a.s. bound $|\psi_t(x)^\top \psi_t(w)| \le \|\psi_t(x)\|\|\psi_t(w)\| \le 2$ (each $\|\psi_t\|^2 = \tfrac{2}{D'}\sum_\ell \cos^2 \le 2$). Hoeffding's inequality gives

$$\mathbb{P}\bigl(|z(x_i)^\top z(x_j) - k_{ⵟ,b}| > s\bigr) \le 2\exp\!\left(-\frac{Ds^2}{2c^2}\right) = 2\exp\!\left(-\frac{Ds^2\varepsilon^2}{8(R^2+b)^4}\right).$$

A union bound over the $\le N^2$ pairs and solving $2N^2\exp(\cdot) \le \delta$ for $s$ yields the stated bound.

## Connections
**Depends on:** `thm:unbiased` (centering); the a.s. bound $|\psi_t(x)^\top\psi_t(w)| \le 2$ and the deterministic modulation bound (app:proofs preamble, line 731); Hoeffding's inequality plus a union bound.
**Used by:** `cor:sample_complexity` (inverting the bound for $D$); `thm:gram_concentration` (multiplying by $N$ for operator norm); the radial-quadrature comparison (line 467); Table tab:rf_comparison and the dimension-freeness discussion of Section sec:comparison; `thm:bernstein_schur` reuses the same Hoeffding argument at the class level with range $4m_f B^2$.
**Validated by:** `gram_approx.py` and `dimension_free.py` (the dataset-level Gram error follows the $O(D^{-1/2})$ rate, Table tab:dimfree showing the radial count $D^\star$ plateaus with $d$).

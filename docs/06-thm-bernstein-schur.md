# Theorem: Bernstein–Schur Random Features
**Label:** `thm:bernstein_schur` | **Location:** main.tex line 205 (proof in Appendix, main.tex lines 928–931)

## What it says
Let $k(x,w) = p(x,w)\,f(\|x-w\|^2)$ be a **Bernstein–Schur kernel**: $p(x,w) = u(x)^\top u(w)$ for a finite feature $u: \mathbb{R}^d \to \mathbb{R}^{d_p}$, and $f$ completely monotone with Bernstein mixture $f(r) = \int_0^\infty e^{-tr}\,d\nu(t)$ of **finite** mass $m_f := \nu(\mathbb{R}_{\ge 0}) = f(0) < \infty$ (this $m_f$ is distinct from the sketch dimension $m$ of Step 5). Draw
$$T_j \sim \nu/m_f, \qquad \omega_j \mid T_j \sim \mathcal{N}(0, 2T_j I_d), \qquad \beta_j \sim \operatorname{Unif}[0, 2\pi],$$
and set $z(x) = \bigl(\sqrt{m_f/D}\,\sqrt{2}\cos(\omega_j^\top x + \beta_j)\,u(x)\bigr)_{j=1}^{D}$. Then:

1. **Unbiasedness:** $\mathbb{E}[z(x)^\top z(w)] = k(x,w)$.
2. **Variance:** if $\|u(x)\| \le B$ on $X$, then $\operatorname{Var}[z(x)^\top z(w)] \le \dfrac{3\,m_f^2 B^4}{2D}$.
3. **Uniform entrywise bound:** with probability $\ge 1 - \delta$,
$$\sup_{i,j}\bigl|z(x_i)^\top z(x_j) - k(x_i, x_j)\bigr| \le m_f B^2 \sqrt{\frac{8\log(2N^2/\delta)}{D}}.$$

The biased ⵟ-kernel is the case $u = p_b$, $f(r) = (r+\varepsilon)^{-1}$ with $d\nu(t) = e^{-\varepsilon t}\,dt$, so $m_f = 1/\varepsilon$, the law $\nu/m_f$ is $\operatorname{Exp}(\varepsilon)$, and with $B = \max_x \|p_b(x)\| = R^2 + b$ the variance scale $m_f^2 B^4 = (R^2+b)^4/\varepsilon^2$ recovers the bounds of Section sec:guarantees.

**Class members (Table tab:bernstein_schur).** Only $u$ and $\nu$ change between instances; the estimator is the same:
- $(x^\top w + b)^2 \times$ IMQ $(r+\varepsilon)^{-1}$, mixing $\operatorname{Exp}(\varepsilon)$, $d_p = d(d+1)/2 + d + 1$ — alignment $\times$ proximity (the biased ⵟ-kernel).
- $(x^\top w + b)^q \times$ generalized IMQ $(r+\varepsilon)^{-\alpha}$, mixing $\Gamma(\alpha, \varepsilon)$ (mass $m_f = \varepsilon^{-\alpha}$), $d_p = \binom{d+q}{q}$ — $q$-way interactions $\times$ locality.
- $(x^\top w + b)^q \times$ Matérn-$\tfrac{1}{2}$ radial $e^{-\sqrt{r}/\sigma}$ (completely monotone in $r = \|x-w\|^2$), Lévy/inverse-Gaussian mixing with mass $m_f = 1$ and the exact two-line sampler $T = 1/(2\sigma^2 Z^2)$, $Z \sim \mathcal{N}(0,1)$ — $q$-way interactions $\times$ exponential locality.
- $(x^\top w + b) \times$ any finite-mass completely monotone $f$, $d_p = d + 1$ — signed local-linear trends.
- $u(x)^\top u(w) \times$ IMQ / Gaussian mixtures, $d_p = \dim u$ — data-driven modulation $\times$ proximity.

The rational-quadratic radial $\mathrm{RQ}_\alpha(r) = (1 + r/(2\alpha\sigma^2))^{-\alpha}$ **is** the generalized IMQ after rescaling ($\varepsilon' = 2\alpha\sigma^2$, $m_f = 1$, mixing $\Gamma(\alpha, \varepsilon')$), so every compositional-kernel-search product such as $\mathrm{LIN}^2 \times \mathrm{RQ}$ is a Bernstein–Schur kernel and inherits the estimator and the full guarantee set with no new analysis.

## Why it matters
This is the theorem that turns the paper from "random features for one kernel" into "random features for a class." It establishes unbiasedness, the variance, and the uniform entrywise bound for every Bernstein–Schur kernel in one stroke, generalizing thm:unbiased and thm:uniform; the matrix-Bernstein and kernel-ridge statements, proved for $k_{ⵟ,b}$, are extended to every member by thm:class_bernstein with the modulation Gram $P_u = m_f G_u$ in place of $P$. The class packages kernels that are often neither stationary nor pure dot-product — precisely the gap between the two random-feature templates — and it has standing customers: Automatic-Statistician grammar products like $\mathrm{LIN}^2 \times \mathrm{RQ}$ previously had no scaling story past the Gram matrix and now inherit one for free.

## Proof idea
For one draw set $Y = 2\cos(\omega^\top x + \beta)\cos(\omega^\top w + \beta)$.

**Unbiasedness:** conditioned on $T$, the standard RFF identity gives $\mathbb{E}[Y \mid T] = e^{-T\|x-w\|^2}$; averaging over $T \sim \nu/m_f$ and restoring the mass, $\mathbb{E}[m_f Y] = \int_0^\infty e^{-t\|x-w\|^2}\,d\nu(t) = f(\|x-w\|^2)$. Multiplying by the exact $u(x)^\top u(w) = p(x,w)$ gives $\mathbb{E}[z(x)^\top z(w)] = p\,f = k$.

**Variance:** $|m_f Y\,u(x)^\top u(w)| \le 2m_f B^2$ and $\mathbb{E}[Y^2 \mid T] \le \tfrac{3}{2}$ (as in the proof of thm:exact_variance), so the one-draw variance is at most $\tfrac{3}{2} m_f^2 B^4$; averaging $D$ i.i.d. copies divides by $D$.

**Uniform bound:** each summand lies in an interval of length $4m_f B^2$, so Hoeffding gives $\mathbb{P}(|z(x_i)^\top z(x_j) - k(x_i,x_j)| \ge s) \le 2\exp(-Ds^2/(8m_f^2 B^4))$; a union bound over $\le N^2$ pairs yields the claim.

## Connections
**Depends on:** the Bernstein–Widder theorem (complete monotonicity $\Rightarrow$ Gaussian scale mixture, the finite-mass hypothesis $f(0) < \infty$), the Rahimi–Recht RFF identity, Hoeffding plus union bound; structurally it abstracts eq:schur, eq:bernstein, eq:expectation, def:ryf, thm:unbiased.
**Used by:** thm:class_bernstein (extends thm:bernstein, cor:bernstein_tail, thm:krr_whitened, cor:krr_highprob, thm:krr_leverage to the class with $P \mapsto P_u$); Table tab:bernstein_schur; the grammar-kernel application (California housing, $(x^\top w + b)^2\,\mathrm{RQ}_\alpha$: Monte-Carlo slope $-0.47$, matches exact composite-kernel ridge at $D = 200$, fits $N = 19{,}640$ in 1.2 s); contribution (i).
**Validated by:** `bernstein_schur_demo.py` (degree-3 modulation $\times$ generalized-IMQ with $\Gamma(2,\varepsilon)$ mixing: unbiased, Monte-Carlo rate), `grammar_kernel.py` (the $\mathrm{LIN}^2 \times \mathrm{RQ}$ California-housing run).

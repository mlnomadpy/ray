# Theorem: Class-Level Matrix Concentration and KRR Condition
**Label:** `thm:class_bernstein` | **Location:** main.tex line 387 (proof at lines 933–948, Appendix app:spectral)

## What it says

Let $k = p \cdot f$ be a Bernstein–Schur kernel in the setting of `thm:bernstein_schur`: $p(x,w) = u(x)^\top u(w)$ for a finite modulation feature $u: \mathbb{R}^d \to \mathbb{R}^{d_p}$, and $f$ completely monotone with Bernstein mixing measure $\nu$ of **finite** mass $m_f = \nu(\mathbb{R}_{\ge0}) = f(0) < \infty$. Let $G_u = [u(x_i)^\top u(x_j)]$ be the modulation Gram and set

$$P_u := m_f\, G_u.$$

Then `thm:bernstein`, `cor:bernstein_tail`, `thm:krr_whitened`, `thm:krr_leverage`, and `cor:krr_highprob` hold **verbatim** for the class estimator of `thm:bernstein_schur` with $P$ replaced by $P_u$. For k_ⵟ,b, $P_u = P$ — the ⵟ-kernel bounds are recovered as the flagship instance ($u = p_b$, $m_f = 1/\varepsilon$).

**The Matérn-½ instance.** The polynomially modulated Matérn-½ kernel $(x^\top w + b)^q\, e^{-\|x-w\|/\sigma}$ is a working member: its radial factor $f(r) = e^{-\sqrt r/\sigma}$ is completely monotone in $r = \|x-w\|^2$ with the Lévy/inverse-Gaussian Bernstein representation

$$e^{-\sqrt r/\sigma} = \int_0^\infty \frac{1}{2\sigma\sqrt{\pi}}\, t^{-3/2} e^{-1/(4\sigma^2 t)}\, e^{-tr}\, dt,$$

mass $m_f = f(0) = 1$, and the **exact two-line sampler** $T = 1/(2\sigma^2 Z^2)$, $Z \sim \mathcal{N}(0,1)$ (the Lévy law of parameter $c = 1/(2\sigma^2)$ is the law of $c/Z^2$). Its modulation Gram is $G_u = [(x_i^\top x_j + b)^q]$ with $\|u(x)\|^2 = (\|x\|^2 + b)^q \le (R^2 + b)^q$, so $P_u = G_u$ and every bound applies with $\|P_u\|_{\mathrm{op}} \le N(R^2+b)^q$ in the worst case and the data-adaptive $\|P_u\|_{\mathrm{op}}$ in general.

## Why it matters

None of the matrix-level analysis is special to the ⵟ-kernel, and this theorem is the proof. It upgrades the Bernstein–Schur class from "unbiased with entrywise control" (`thm:bernstein_schur`) to the **full operator-norm and kernel-ridge guarantee set** — expected and tail matrix-Bernstein bounds, the whitened high-probability KRR condition with the exact effective-dimension identity, the leverage-tilted effective-dimension count, and the objective-value sandwich — for every kernel of the form (finite-feature kernel) × (completely monotone radial kernel). The proofs consume exactly **three structural facts** (a PSD per-draw Gram that is a Schur product with a bounded-diagonal rank-one factor, a PSD modulation Gram, and a unit-diagonal PSD radial Gram), and every Bernstein–Schur kernel supplies all three. One theorem, one substitution $P \mapsto P_u$, an entire class covered.

## Proof idea

The class estimator has per-draw Gram $K^{(j)} = (\psi_j\psi_j^\top) \circ P_u$ with $\psi_j[i] = \sqrt2\cos(\omega_j^\top x_i + \beta_j)$ and $P_u = m_f G_u$ (the mass $m_f$ entering through the $\sqrt{m_f/D}$ scaling of the feature). Verify the three structural facts the cited proofs consume:

**(i) PSD per-draw Gram with bounded norm.** $\psi_j\psi_j^\top \succeq 0$ is rank one with diagonal $2\cos^2(\cdot) \le 2$, and $G_u \succeq 0$ is a Gram matrix, so $K^{(j)} \succeq 0$ by the Schur product theorem and $\|K^{(j)}\|_{\mathrm{op}} \le 2\|P_u\|_{\mathrm{op}}$ by `lem:schur`(a) with $a = 2$.

**(ii) Unbiasedness through a unit-diagonal PSD radial Gram.** Conditional on $T_j$, the RFF identity gives $\mathbb{E}[(\psi_j\psi_j^\top)_{ik} \mid T_j] = e^{-T_j\|x_i - x_k\|^2}$; averaging over $T_j \sim \nu/m_f$ yields $\mathbb{E}[K^{(j)}] = R_u \circ P_u = K$ with $(R_u)_{ik} = f(\|x_i - x_k\|^2)/m_f$. The radial Gram $R_u$ is PSD (a nonnegative mixture of Gaussian Grams under $\nu/m_f$) with unit diagonal ($f(0) = m_f$), so `lem:schur`(a) also gives $K \succeq 0$ and $\|K\|_{\mathrm{op}} \le \|P_u\|_{\mathrm{op}}$.

**(iii) The variance step.** For PSD $M$, $M^2 \preceq \|M\|_{\mathrm{op}} M$ and $M A^{-1} M \preceq \lambda^{-1}\|M\|_{\mathrm{op}} M$; with (i) these give $\mathbb{E}[(K^{(j)} - K)^2] \preceq 2\|P_u\|_{\mathrm{op}} K$ and $\mathbb{E}[(K^{(j)} - K)A^{-1}(K^{(j)} - K)] \preceq (2\|P_u\|_{\mathrm{op}}/\lambda) K$ — the two variance majorants.

Substituting (i)–(iii) into the proofs of `thm:bernstein` / `cor:bernstein_tail` ($L = 3\|P_u\|_{\mathrm{op}}/D$, $v = 2\|P_u\|_{\mathrm{op}}\|K\|_{\mathrm{op}}/D$) and of `thm:krr_whitened` / `cor:krr_highprob` ($L_\lambda = (1 + 2\|P_u\|_{\mathrm{op}}/\lambda)/D$, majorant $\frac{2\|P_u\|_{\mathrm{op}}}{\lambda D} A^{-1/2}KA^{-1/2}$, whose intrinsic dimension is again **exactly** $\tilde d_\lambda$) reproduces every statement verbatim. The leverage theorem transfers identically: beyond (i)–(iii) its proof consumes only $\mathbb{E}_\pi[\psi\psi^\top] = R_u$, so $\bar d_\lambda(\theta) = \psi_\theta^\top(A^{-1} \circ P_u)\psi_\theta$ has mean $d_{\mathrm{eff}}(\lambda)$ and `thm:krr_leverage` holds with $P \mapsto P_u$. For k_ⵟ,b, $u = p_b$ and $m_f = 1/\varepsilon$ give $P_u = [(x_i^\top x_j + b)^2/\varepsilon] = P$.

## Connections

**Depends on:** `thm:bernstein_schur` (the class, its estimator, $m_f$, $\nu$), `lem:schur`, the proofs of `thm:bernstein`, `cor:bernstein_tail`, `thm:krr_whitened`, `thm:krr_leverage`, `cor:krr_highprob` (consumed as templates), the RFF conditional identity, Bernstein–Widder.
**Used by:** the paper's class-level claim (contribution 1: one estimator and one guarantee set for the whole Bernstein–Schur class); the Matérn-½ instance with its Lévy sampler (Table tab:bernstein_schur); discussed in the bias-cost reading ($b = 0$ recovers unbiased bounds; $b > 0$ inflates by powers of $1 + b/R^2$).
**Validated by:** `krr_whitened_bernstein.py` (the Matérn-½ class instance reproduces the intrinsic-dimension identity, the whitened rate, and the sandwich), `bernstein_schur_demo.py`.

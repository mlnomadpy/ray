# Proposition: Variance Dichotomy — the Mixing Law Caps Positive Features
**Label:** `prop:pos_dichotomy` | **Location:** main.tex line 667

## What it says

Positivity (`prop:positive`) is not free: composing the FAVOR$^+$ feature with the Bernstein mixing law has a sharp failure mode the signed estimator does not, and it determines exactly where each feature belongs.

Let $\widehat g^+ = \phi^+_T(x)\,\phi^+_T(w)$ with $T \sim \mathrm{Exp}(\varepsilon)$ and $\omega \mid T \sim \mathcal{N}(0, 2TI_d)$ — the positive-feature estimate of the rescaled radial factor. Then $\mathbb{E}[\widehat g^+] = \varepsilon/(\|x-w\|^2 + \varepsilon)$ (unbiased), but

$$\mathbb{E}\bigl[(\widehat g^+)^2\bigr] = \mathbb{E}_T\bigl[e^{8T x^\top w}\bigr] = \begin{cases} \dfrac{\varepsilon}{\varepsilon - 8x^\top w}, & 8x^\top w < \varepsilon,\\[4pt] +\infty, & 8x^\top w \ge \varepsilon. \end{cases}$$

So the positive-feature radial estimator has **infinite variance on every pair with $x^\top w \ge \varepsilon/8$**.

Truncating the scale at $T \le T_{\max}$ (relative bias at most $e^{-\varepsilon T_{\max}}$ on the radial factor) caps the second moment at $e^{8T_{\max} x^\top w}$ — which **remains exponential** in $x^\top w/\varepsilon$. The trigonometric estimator instead satisfies

$$\mathbb{E}[\widehat g^2 \mid T] = 1 + \tfrac12 e^{-4T\|x-w\|^2} \le \tfrac32$$

uniformly in the scale and the pair.

## Why it matters

This is the sharp negative answer to whether the positivity fix of `prop:positive` also cures the peaked-attention regime: it cannot. The dichotomy scopes the FAVOR$^+$ alternative for ⵟ-attention precisely. In the **diffuse** regime ($8x^\top w \ll \varepsilon$ for all pairs) the positive pair matches the trigonometric estimator and adds nonnegative weights for free; in the **aligned or peaked** regime ($8x^\top w \ge \varepsilon$ for some pairs) the signed trigonometric estimator is the only finite-variance choice between the two — and truncation rescues nothing, since the capped moment is still exponential in the alignment. The peaked regime itself remains open (open problem (3), Discussion). The mechanism is structural: FAVOR$^+$'s variance control for the Performer relies on a *fixed* Gaussian scale, while RAY must *integrate over* scales $T \sim \mathrm{Exp}(\varepsilon)$, and the exponential second moment $e^{8t x^\top w}$ meets the exponential mixing density head-on, diverging exactly when $8x^\top w \ge \varepsilon$.

## Proof idea

**Conditional second moment.** Given $T = t$, $(\widehat g^+)^2 = e^{2\omega^\top(x+w)}e^{-4t(\|x\|^2 + \|w\|^2)}$. The Gaussian MGF $\mathbb{E}_\omega[e^{a^\top\omega}] = e^{t\|a\|^2}$ for $\omega \sim \mathcal{N}(0, 2tI_d)$ (with $a = 2(x+w)$) gives

$$\mathbb{E}_\omega[(\widehat g^+)^2 \mid t] = e^{4t\|x+w\|^2 - 4t\|x\|^2 - 4t\|w\|^2} = e^{8t x^\top w}.$$

**Integration against the mixing law.** Integrating $e^{8tx^\top w}$ against $\varepsilon e^{-\varepsilon t}\,dt$ gives $\varepsilon/(\varepsilon - 8x^\top w)$ when $8x^\top w < \varepsilon$ and diverges otherwise — the dichotomy is exactly the convergence condition of an exponential integral.

**Truncation.** On $\{T \le T_{\max}\}$, the conditional bound $e^{8tx^\top w} \le e^{8T_{\max}x^\top w}$ (for $x^\top w \ge 0$; at most $1$ for $x^\top w < 0$) caps the second moment, and the truncated integral $\int_0^{T_{\max}} \varepsilon e^{-\varepsilon t}e^{-tr}\,dt$ misses at most $e^{-(\varepsilon + r)T_{\max}} \le e^{-\varepsilon T_{\max}}$ of the total in relative terms.

**Trigonometric contrast.** The uniform bound $\mathbb{E}[\widehat g^2 \mid T] \le 3/2$ is established in the proof of `thm:exact_variance`.

## Connections

**Depends on:** `prop:positive` (the estimator being analyzed), the Gaussian MGF identity, the $\mathrm{Exp}(\varepsilon)$ Bernstein mixing law of the radial factor, `thm:exact_variance` (the trigonometric bracket bound $\le 3/2$).
**Used by:** Section sec:exp_attention (the scoped recommendation: positive features only in the diffuse regime, trigonometric is the only finite-variance choice when $8x^\top w \ge \varepsilon$), `prop:quadrature` (bounded node sets are the scale truncation positive features require, through the largest node), the Discussion's open problem (3) on the peaked regime.
**Validated by:** `experiments/positive_features.py` — pointwise, the empirical second moment grows without plateau above the threshold ($4.5 \to 15.2$ over $10^3 \to 10^6$ repetitions at $8x^\top w/\varepsilon = 4$, vs. $1.14$ trigonometric), and the finite-side prediction $\varepsilon/(\varepsilon - 8x^\top w)$ is approached from below; at the Gram level the three regimes separate exactly as stated (diffuse: $0.084$ vs. $0.080$ at $D{=}128$; $\varepsilon = 1\times$: $2$–$3\times$ worse with heavy-tailed seeds, worst $1.43$; peaked: median $0.42$–$0.77$, worst $9$–$17$, not improving at the MC rate, vs. trigonometric $0.08$–$0.31$; truncation rescues nothing).

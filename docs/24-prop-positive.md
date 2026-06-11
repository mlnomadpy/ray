# Proposition: Positive Features Give Nonnegative Attention
**Label:** `prop:positive` | **Location:** main.tex line 655

## What it says

The trigonometric radial feature of RAY is signed, so the estimated ⵟ-attention weights $\phi(q)^\top\phi(k)$ can be negative. This proposition supplies the principled fix: the Gaussian radial factor admits a **positive** random feature. For $t > 0$ and $\omega \sim \mathcal{N}(0, 2tI_d)$,

$$e^{-t\|x-w\|^2} = \mathbb{E}_\omega\bigl[\phi^+_t(x)\,\phi^+_t(w)\bigr], \qquad \phi^+_t(x) = \exp\bigl(\omega^\top x - 2t\|x\|^2\bigr) > 0,$$

the FAVOR$^+$ construction of Choromanski et al. (2021), with controlled relative error where the kernel is small.

Combined with a **nonnegative modulation feature** — anchor squares $\phi_a(x) = (a^\top x)^2 \ge 0$, exact on the sphere — the product feature $\Phi(x) = \phi^+_t(x)\,\phi_a(x)$ is entrywise positive, so the estimated ⵟ-attention weights

$$\Phi(q)^\top\Phi(k) \ge 0$$

are **nonnegative by construction**, which the signed trigonometric/TensorSketch estimator cannot guarantee.

## Why it matters

In the linear-time streaming ⵟ-attention application (Section sec:exp_attention), attention weights are normalized kernel evaluations; a signed estimator can emit negative weights and even negative normalizers, which is qualitatively wrong for a smoother. This proposition shows sign indefiniteness is not intrinsic to RAY: the radial factor has a FAVOR$^+$-style positive randomization that composes with a nonnegative modulation feature into a fully positive ⵟ-attention feature map. It sets up the sharp scoping question that `prop:pos_dichotomy` answers negatively: positivity does **not** also cure the peaked-attention regime — composing $\phi^+$ with the Bernstein mixing law has an infinite-variance failure mode the signed estimator does not. Together the two propositions partition the design space: positive pair in the diffuse regime (where it adds the nonnegativity guarantee for free), trigonometric everywhere else.

## Proof idea

A Gaussian moment-generating-function computation. For $\omega \sim \mathcal{N}(0, 2tI_d)$, $\mathbb{E}_\omega[e^{\omega^\top a}] = e^{t\|a\|^2}$. With $a = x + w$:

$$\mathbb{E}_\omega[\phi^+_t(x)\phi^+_t(w)] = e^{-2t\|x\|^2 - 2t\|w\|^2}\,\mathbb{E}_\omega[e^{\omega^\top(x+w)}] = e^{-2t\|x\|^2 - 2t\|w\|^2}e^{t\|x+w\|^2} = e^{-t\|x-w\|^2},$$

since $\|x+w\|^2 - 2\|x\|^2 - 2\|w\|^2 = -\|x-w\|^2$. Positivity of $\phi^+_t$ (an exponential) and of $\phi_a$ (a square) is immediate, and a product of nonnegative features has a nonnegative inner product.

## Connections

**Depends on:** The Gaussian MGF identity, the FAVOR$^+$ construction (Choromanski et al. 2021), anchor-square modulation features (exact on the sphere, biased off it — Section sec:discussion), the Bernstein scale-mixture structure of $h_\varepsilon$.
**Used by:** `prop:pos_dichotomy` (the variance analysis of exactly this estimator under the $\mathrm{Exp}(\varepsilon)$ mixing law), Section sec:exp_attention (the nonnegative-attention design option in the diffuse regime), the Discussion's open problem on the peaked regime.
**Validated by:** `positive_features.py` (the diffuse-regime parity with the trigonometric estimator, $0.084$ vs. $0.080$ at $D{=}128$, where positivity comes free).

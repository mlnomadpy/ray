# Example: One RAY Coordinate, End to End
**Label:** `ex:ray` | **Location:** main.tex line 240

## What it says
A complete walkthrough of a single radial draw of the deployed doubly-randomized RAY map (eq:doubly). Fix $b = 1$, $\varepsilon = 1$, $d = 3$.

1. **Draw a scale** $t \sim \operatorname{Exp}(1)$, say $t = 0.4$.
2. **Draw a frequency and phase** $\omega \sim \mathcal{N}(0, 2t I_3)$, $\beta \sim \operatorname{Unif}[0, 2\pi)$, giving the radial Fourier feature $\sqrt{2}\cos(\omega^\top x + \beta)$ — unbiased for the Gaussian $e^{-t\|x-w\|^2}$ at that scale.
3. **Form the modulation sketch** $\widehat{p}_m(x) = \mathrm{TS}_m(x, \sqrt{b})$, a degree-2 TensorSketch of the augmented input — unbiased for $(x^\top w + 1)^2$.
4. **Tensor them:** $z_1(x) = \sqrt{2}\cos(\omega^\top x + \beta)\,\widehat{p}_m(x) \in \mathbb{R}^m$.

Then
$$\mathbb{E}\bigl[z_1(x)^\top z_1(w)\bigr] = e^{-t\|x-w\|^2}\,(x^\top w + 1)^2,$$
and averaging the scale over $t \sim \operatorname{Exp}(1)$ recovers $h_\varepsilon \cdot p_b = k_{ⵟ,1}$. Stacking $D$ such coordinates and dividing by $\sqrt{D}$ gives the full $\mathbb{R}^{Dm}$ feature map. Setting $b = 0$ and replacing $\widehat{p}_m$ by the exact $\operatorname{vec}(x \otimes x)$ recovers the unbiased exact-modulation feature of def:ryf.

## Why it matters
The example is the paper's five-step construction collapsed into one concrete coordinate, making the two sources of randomness — the Bernstein radial scale and the modulation sketch — visible in a single line of algebra. It shows exactly where each pipeline stage acts: the Schur factorization supplies the two factors being multiplied, the $\operatorname{Exp}(\varepsilon)$ draw discretizes the Bernstein–Widder mixture, the cosine is the Bochner feature of the Gaussian at the drawn scale, and the TensorSketch removes the $O(d^2)$ modulation floor. It also makes the limiting relationships explicit: exact modulation ($m \to \infty$) recovers def:ryf, and $b = 0$ recovers the unbiased ⵟ-feature. Anyone implementing RAY can check their code against this coordinate.

## Proof idea
Direct composition of two conditional unbiasedness facts. Conditional on $t$, the RFF identity gives $\mathbb{E}_{\omega,\beta}[2\cos(\omega^\top x+\beta)\cos(\omega^\top w+\beta)] = e^{-t\|x-w\|^2}$; independently, the TensorSketch satisfies $\mathbb{E}[\mathrm{TS}_m(x,\sqrt{b})^\top \mathrm{TS}_m(w,\sqrt{b})] = (x^\top w + b)^2$ over the sketch randomness. Independence of the two randomness sources factorizes the expectation of the product. Averaging over $t \sim \operatorname{Exp}(\varepsilon)$ and applying eq:expectation reassembles $k_{ⵟ,b}$.

## Connections
**Depends on:** eq:doubly (the deployed RAY map), def:ryf (the exact-modulation limit), eq:expectation ($\operatorname{Exp}(\varepsilon)$ scale law), prop:biased_feature (the exact feature it specializes to at $b=0$), the TensorSketch unbiasedness of Step 5.
**Used by:** serves as the concrete reference implementation for Section sec:step-sketch and for thm:ts_opnorm's setting; rmk:complex modifies exactly the $\widehat{p}_m$ factor of this coordinate.
**Validated by:** the implementation path exercised by `ts_opnorm_validation.py`, `ts_decomposition.py`, and `ts_ryf_costmatched.py`.

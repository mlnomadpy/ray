# Theorem: Operator-Norm Error of RAY
**Label:** `thm:ts_opnorm` | **Location:** main.tex line 287

## What it says

This is the matrix-level guarantee for the estimator one actually deploys — the doubly-randomized RAY map of Step 5 (eq:doubly), which randomizes both factors of $k_{ⵟ,b} = p \cdot h_\varepsilon$: the radial scale by the Bernstein mixture, the modulation by a sketch of dimension $m$ free of the $O(d^2)$ floor.

**Setup.** Let $\widehat K_{D,m} = \widehat P_m \circ \widehat R_D$ be the doubly-randomized Gram estimate:

- A single degree-2 TensorSketch of dimension $m$ (drawn once, shared across all radial draws) gives the modulation Gram $\widehat P_m = [\mathrm{TS}_m(x_i)^\top \mathrm{TS}_m(x_j)] \succeq 0$ with $\mathbb{E}\,\widehat P_m = P = [(x_i^\top x_j + b)^2/\varepsilon]$.
- $D$ radial draws give the radial Gram $\widehat R_D$ with $\mathbb{E}\,\widehat R_D = R$, the unit-diagonal Gram $R_{ij} = \varepsilon/(\varepsilon + \|x_i - x_j\|^2)$, independent of the sketch. So $K = R \circ P$ is the exact kernel Gram of $k_{ⵟ,b}$.

**Hypothesis.** The sketch satisfies the spectral event $\|\widehat P_m - P\|_{\mathrm{op}} \le \eta\,\|P\|_{\mathrm{op}}$ (when the degree-2 TensorSketch achieves this is the subject of `rmk:ose`).

**Conclusion.** On that event, with probability at least $1-\delta$ over the radial draws,

$$\|\widehat K_{D,m} - K\|_{\mathrm{op}} \le \underbrace{2\sqrt{\frac{(1+\eta)\|P\|_{\mathrm{op}}\|K_S\|_{\mathrm{op}}\log(8 d_{\mathrm{int},S}/\delta)}{D}} + \frac{2(1+\eta)\|P\|_{\mathrm{op}}\log(8 d_{\mathrm{int},S}/\delta)}{D}}_{\text{radial},\ O(D^{-1/2})} \;+\; \underbrace{\eta\,\|P\|_{\mathrm{op}}}_{\text{sketch}},$$

where $K_S = \widehat P_m \circ R$ is the sketch-conditioned target, $\|K_S\|_{\mathrm{op}} \le (1+\eta)\|P\|_{\mathrm{op}}$, and $d_{\mathrm{int},S} = \operatorname{tr}(V_S)/\|V_S\|_{\mathrm{op}} \le N$ is the intrinsic dimension of the sketch-conditioned matrix variance $V_S$ — the variance of `thm:bernstein` for the pair $(\widehat P_m, K_S)$ rather than $(P, K)$, reducing to $d_{\mathrm{int}}$ as $m \to \infty$.

At $m \to \infty$ ($\eta \to 0$) the sketch term vanishes and the bound is exactly `cor:bernstein_tail`: randomizing the modulation costs a **single additive term** $\eta\|P\|_{\mathrm{op}}$, set by $m$ independently of $D$. The theorem covers both the fully sketched form (eq:doubly) and the quadratic-only variant of dimension $D(m+d+1)$, since the analysis touches the modulation only through $\widehat P_m$.

## Why it matters

The clean concentration identities of Section sec:guarantees (`thm:bernstein`, `cor:bernstein_tail`) are stated for the exact-modulation estimator, whose feature size $D\,d_b = O(D d^2)$ is exactly the floor the deployed estimator exists to remove. Without this theorem the paper's headline object — the $Dm$-dimensional map free of $d^2$ — would carry no operator-norm guarantee at all, and the whole "clean identities at the limit, then the doubly-randomized bound" architecture of Steps 4–5 would collapse. The theorem also fixes the design interface of the method: the modulation randomizer is modular (TensorSketch, complex sketches per `rmk:complex`, random-Maclaurin, anchor features) because the bound consumes only PSD-ness and an $\eta$-spectral approximation of $P$. The radial/sketch split it predicts is what `prop:optimal_m` later optimizes and what Section sec:exp_ts validates.

## Proof idea

**Conditioning is the whole argument.** Given the sketch, $\widehat P_m$ is a fixed PSD modulation Gram, and $\widehat K_{D,m} = \widehat P_m \circ \widehat R_D$ is an *exact-modulation* estimator of $K_S = \widehat P_m \circ R$. So `thm:bernstein` and `cor:bernstein_tail` apply verbatim with $P \mapsto \widehat P_m$ — using $\|\widehat P_m\|_{\mathrm{op}} \le (1+\eta)\|P\|_{\mathrm{op}}$ from the spectral event, and $\|K_S\|_{\mathrm{op}} \le \|\widehat P_m\|_{\mathrm{op}}$ by the Schur-multiplier bound (Lemma `lem:schur`, $R$ unit-diagonal). This yields the radial term.

The remaining bias is $K_S - K = (\widehat P_m - P) \circ R = R \circ E_P$ with $E_P := \widehat P_m - P$ symmetric but **not** PSD; here the symmetric form of the Schur-multiplier bound, Lemma `lem:schur`(b) with $A = R \succeq 0$ and $\max_i R_{ii} = 1$, gives $\|R \circ E_P\|_{\mathrm{op}} \le \|E_P\|_{\mathrm{op}} \le \eta\|P\|_{\mathrm{op}}$.

The triangle inequality combines the two pieces. The conservative entrywise Frobenius decomposition (`prop:ts_variance`) needs no subspace-embedding hypothesis and remains available as a fallback.

## Connections

**Depends on:** `thm:bernstein`, `cor:bernstein_tail` (applied verbatim to the conditioned pair $(\widehat P_m, K_S)$), `lem:schur` (parts (a) and (b)), eq:doubly (the deployed RAY map, Step 5), the spectral event discussed in `rmk:ose`.
**Used by:** `rmk:ose` (when the hypothesis holds), `prop:ridge_sketch` (sharpens the additive sketch term), `cor:krr_deployed` (same conditioning device for the KRR condition), `rmk:complex` (complex sketches lower the additive term), `prop:optimal_m` (operationalizes the radial-vs-sketch split), Section sec:exp_ts and Table tab:dm.
**Validated by:** `ts_opnorm_validation.py` (Figure fig:ts_opnorm: at $m{=}128$ the radial term falls as $D^{-1/2}$, $40.5 \to 3.6$ over $D = 10 \to 1000$, while the sketch term is a $D$-independent floor decaying with $m$, $19.5 \to 7.4$ as $m = 64 \to 256$).

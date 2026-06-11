# Remark: Complex Modulation Sketches
**Label:** `rmk:complex` | **Location:** main.tex line 244

## What it says
The modulation randomizer is the only part of RAY that need not be real. Following Wacker et al. (2024), complex-valued polynomial sketches have strictly smaller variance than real Rademacher sketches for dot-product kernels, because the unit-modulus fourth moment is smaller. Substituting unit-phase signs $s \in \{1, i, -1, -i\}$ in the quadratic-only $\mathrm{TS}_m$ and reading off $\operatorname{Re}\langle \mathrm{TS}_m(x), \overline{\mathrm{TS}_m(w)} \rangle$ leaves the estimator unbiased while lowering the additive sketch term in thm:ts_opnorm. The radial factor is unaffected.

Measured on the protocol of Figure fig:ts_opnorm, the drop-in is real at every sketch size:
- the spectral error $\eta$ falls by $1.5$–$1.7\times$ (at $m = 128$: $0.042 \to 0.027$);
- the sketch term $\|E_P \circ R\|_{\mathrm{op}}$ falls by the same factor ($3.84 \to 2.51$);
- the per-entry sketch variance is halved (ratios $0.43$–$0.52$ across $m \in \{64, \dots, 512\}$),

consistent with the fourth-moment computation of Wacker et al.

## Why it matters
Step 5 introduced the single new error source of the deployed estimator — the additive modulation-sketch term in thm:ts_opnorm. This remark shows that term is not fixed: it is reducible by a free, structural substitution that costs nothing in the analysis (unbiasedness is preserved, the radial guarantees are untouched) and roughly halves the sketch variance in practice. It also demonstrates the modularity claim of Section sec:step-sketch concretely: the modulation randomizer is a plug-in slot, and the complex sketch is the first nontrivial upgrade plugged into it.

## Proof idea
The TensorSketch hash signs are the only randomness in $\widehat{p}_m$; replacing Rademacher signs $s \in \{\pm 1\}$ by uniform fourth-roots-of-unity phases $s \in \{1, i, -1, -i\}$ keeps $\mathbb{E}[s\bar{s}'] = \delta_{ss'}$-type second-moment structure, so $\mathbb{E}\operatorname{Re}\langle \mathrm{TS}_m(x), \overline{\mathrm{TS}_m(w)} \rangle = (x^\top w)^2$ unchanged — unbiasedness survives. The variance of a polynomial sketch is governed by the fourth moment of the sign variable, and $\mathbb{E}[|s|^4]$-type cross terms are strictly smaller for unit-phase complex signs than for real Rademacher signs (the Wacker et al. computation). Since thm:ts_opnorm touches the modulation only through the sketch error matrix $E_P$, a smaller per-entry sketch variance lowers the additive term $\|E_P \circ R\|_{\mathrm{op}}$ directly; the empirical $1.5$–$1.7\times$ spectral-error reduction matches this prediction.

## Connections
**Depends on:** Step 5 / eq:doubly (the quadratic-only sketched estimator), thm:ts_opnorm (the additive sketch term being lowered), the complex-sketch variance analysis of Wacker et al. (2024).
**Used by:** the deployed-estimator recommendations of Section sec:step-sketch; the experimental record in Appendix app:exp_details (`complex_sketch` row).
**Validated by:** `complex_sketch.py` (results in `experiments/results/complex_sketch.json`): spectral error $\eta$ $0.042 \to 0.027$ at $m = 128$, sketch operator-norm term $3.84 \to 2.51$, per-entry variance ratios $0.43$–$0.52$ for $m \in \{64, \dots, 512\}$.

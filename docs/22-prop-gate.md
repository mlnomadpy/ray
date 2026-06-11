# Proposition: Modulation–Radial Error Decomposition
**Label:** `prop:gate` | **Location:** main.tex line 398

## What it says
Write the kernel Gram as the Schur product $K = P \circ H$ with $P_{ij} = p_b(x_i, x_j)$ (modulation) and $H_{ij} = h_\varepsilon(x_i, x_j)$ (radial). With $\widehat{P}_m$ *any* modulation approximation and $\widehat{H}_D$ the radial estimate, the deployed estimator $\widehat{K}_{D,m} = \widehat{P}_m \circ \widehat{H}_D$ satisfies the exact three-term decomposition

$$\widehat{K}_{D,m} - K = \underbrace{P \circ (\widehat{H}_D - H)}_{\text{radial error, modulated by the finite feature}} + \underbrace{(\widehat{P}_m - P) \circ H}_{\text{modulation error, localized by proximity}} + \underbrace{(\widehat{P}_m - P) \circ (\widehat{H}_D - H)}_{\text{interaction}}.$$

For exact modulation ($\widehat{P}_m = P$) this collapses to $\widehat{K}_D - K = P \circ (\widehat{H}_D - H)$, so entrywise

$$|\widehat{K}_{ij} - K_{ij}| = p_b(x_i, x_j)\,|\widehat{H}_{ij} - H_{ij}|.$$

## Why it matters
This decomposition explains why the alignment×proximity product is worth keeping even when the modulation is itself compressed — the "signal gate" reading of the kernel. The radial Monte-Carlo noise enters each Gram entry Schur-scaled by the modulation: under bounded norms, and *literally* for the normalized variant of `prop:normalized` where $G_{ij} = (x_i^\top x_j + b)^2/((\|x_i\|^2+b)(\|x_j\|^2+b)) \in [0,1]$, the modulation acts as an alignment gate that suppresses noise exactly on the weakly aligned pairs ($p_b$ in raw form also carries norm and bias scale, which is why the gate statement is literal only after normalization). Symmetrically, modulation error is localized by radial proximity through $H$, and only the interaction term mixes the two sources. This is the mechanism behind the coupled-target kernel preference (Section sec:exp_necessity): the product sharpens *between*-pair discrimination while leaving any single pair's relative error unchanged, and — the operative point for deployment — this gating survives sketching the polynomial factor (contribution list, line 79). It is also the structural sibling of the operator-norm split in `thm:ts_opnorm`, which is the same modulation/radial separation executed in spectral norm.

## Proof idea
One line, stated inline in the tex (line 405): expand $(P + E_P) \circ (H + E_H)$ with $E_P = \widehat{P}_m - P$, $E_H = \widehat{H}_D - H$, using bilinearity of the Schur product, and subtract $K = P \circ H$. The three cross terms are exactly the displayed ones. The exact-modulation collapse sets $E_P = 0$; the entrywise identity is then immediate because the Schur product acts entrywise and $p_b \ge 0$.

## Connections
**Depends on:** the Schur factorization $K = P \circ H$ (eq:schur, the paper's core structural identity); bilinearity of the Schur product; nothing probabilistic.
**Used by:** Section sec:exp_gate (the alignment numerator as signal modulation — explains the coupled-target preference of Section sec:exp_necessity); `prop:normalized` (the variant for which the $[0,1]$ gate is literal); the contributions list (line 79: alignment×proximity suppresses similarity false positives even with sketched modulation); companion in spirit to `prop:ts_variance` (the same decomposition at the level of scalar variances) and `thm:ts_opnorm` (the same split in operator norm).
**Validated by:** `signal_gate_snr.py` and `gate_diagnostic.py` (Section sec:exp_gate): on off-sphere data ($d=16$), scoring true pairs against radial distractors and alignment distractors by AUC (400 pairs/type), radial-only IMQ false-positives on radial distractors (AUC $0.22$), alignment-only degree-2 polynomial false-positives on alignment distractors (AUC $0.00$), while the ⵟ product suppresses both (AUC $1.00$/$1.00$); the exact-modulation estimator inherits this almost exactly, and deployed RAY with sketched modulation partially survives on alignment distractors (AUC $0.73$).

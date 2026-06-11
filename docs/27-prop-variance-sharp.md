# Proposition: Sharpness of the Variance Prefactor
**Label:** `prop:variance_sharp` | **Location:** main.tex line 744

## What it says
Consider any unbiased estimator of $k_{ⵟ,b}(x,w)$ of the **factored form** $\widehat{k} = (p(x)^\top p(w))\,\widehat{h}$, where the modulation inner product is kept exact (deterministic) and $\widehat{h}$ is any unbiased estimator of the radial factor $h_\varepsilon(x,w)$. Then

$$\mathrm{Var}[\widehat{k}] = \frac{(x^\top w+b)^4}{\varepsilon^2}\,\mathrm{Var}[\widehat{h}].$$

The prefactor $(x^\top w+b)^4/\varepsilon^2 \le (R^2+b)^4/\varepsilon^2$ is therefore common to the **entire** exact-modulation factored family and is attained with *equality*, not as a slack upper bound. `thm:exact_variance` is the instance with $\widehat{h}$ the flat radial estimator. No sharper analysis can remove the prefactor while the modulation is kept exact; lowering it requires leaving the family, either by normalizing the modulation (`prop:normalized`, which estimates a different, cosine-rescaled kernel) or by randomizing it (`prop:ts_variance`, which only adds variance).

## Why it matters
This proposition closes the door on hoping the $(R^2+b)^4$ variance constant is an artifact of the proof of `thm:variance`. It is structural: any estimator that keeps the polynomial alignment factor exact — the entire design space of "better radial estimators" (QMC, orthogonal frequencies, importance sampling per `prop:importance`, deterministic quadrature) — carries the identical prefactor with equality. This is the honest price of the alignment numerator, stated as such in the limitations (Section sec:discussion, line 709: "it is sharp, not a loose bound") and in the caption of Table tab:rf_comparison. It also sharpens the open problem (open problem 2, line 711): whether *any* unbiased data-independent feature map of k_ⵟ,b can beat the prefactor — for instance by correlating the modulation and radial randomness so their errors cancel — is open, and the paper conjectures the prefactor is optimal. Without this proposition, the variance story would be an upper bound that a reviewer could reasonably hope to improve; with it, improvement provably requires changing the estimator family.

## Proof idea
Two lines. The modulation factor $p(x)^\top p(w) = (x^\top w+b)^2/\varepsilon$ is a deterministic scalar, so it passes through the variance as its square: $\mathrm{Var}[(p(x)^\top p(w))\,\widehat{h}] = (p(x)^\top p(w))^2\,\mathrm{Var}[\widehat{h}] = \tfrac{(x^\top w+b)^4}{\varepsilon^2}\mathrm{Var}[\widehat{h}]$, an identity. The uniform bound on the prefactor is Cauchy–Schwarz, $(x^\top w+b)^2 \le (\|x\|^2+b)(\|w\|^2+b) \le (R^2+b)^2$, using $b \ge 0$.

## Connections
**Depends on:** `prop:biased_feature` (the exact modulation feature gives the deterministic factor $p(x)^\top p(w) = (x^\top w+b)^2/\varepsilon$); Cauchy–Schwarz.
**Used by:** `thm:variance` and `thm:exact_variance` (interprets their common prefactor as unimprovable); `prop:normalized` and `prop:ts_variance` (positioned as the only two exits from the family); the limitations paragraph (line 709) and open problem 2 (line 711, the conjecture that the prefactor is optimal among all unbiased data-independent feature maps).
**Validated by:** `bias_scaling.py` (Figure fig:bias — the empirical fourth-power law with tightness for aligned pairs is the prefactor in action); `exact_variance_check.py` (the equality instance confirmed pointwise to ratio $1.001 \pm 0.002$).

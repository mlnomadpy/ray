# Proposition: Normalized Estimator
**Label:** `prop:normalized` | **Location:** main.tex line 856

## What it says
Let $q_b(x) = p_b(x)/(\|x\|^2 + b)$, so $\|q_b(x)\| = 1$ (for $b > 0$ this is defined on all of $\mathbb{R}^d$; for $b = 0$ the domain excludes the origin). Then

$$q_b(x)^\top q_b(w) = \frac{(x^\top w + b)^2}{(\|x\|^2+b)(\|w\|^2+b)},$$

and the **normalized kernel**

$$\bar{k}_{ⵟ,b}(x,w) = q_b(x)^\top q_b(w)\,\frac{1}{\|x-w\|^2 + \varepsilon}$$

is positive definite. Its flat estimator (replace $p_b$ by $q_b$ in the flat construction) is unbiased for $\bar{k}_{ⵟ,b}$ with, by `thm:exact_variance` and $\|q_b\| = 1$,

$$\mathrm{Var}[\widehat{\bar{k}}_D(x,w)] \le \frac{1}{D\varepsilon^2}\left(1 + \frac{1}{2}\,\frac{\varepsilon}{\varepsilon + 4\|x-w\|^2}\right) \le \frac{3}{2D\varepsilon^2},$$

a bound **free of the data radius $R$ and the bias $b$**.

## Why it matters
This is the designated exit from the $(R^2+b)^4$ variance blow-up that `prop:variance_sharp` proves is unavoidable within the exact-modulation factored family. Normalizing the modulation to unit norm removes the radius and bias from the variance entirely — but the paper is explicit about what is and is not bought: the normalized estimator is *not* an approximation of k_ⵟ,b; it is an unbiased estimator of a *different*, cosine-rescaled kernel, so the gain is stability, not a free fix (line 394). It is the stable choice for large-radius or large-bias data, and it is the variant for which the alignment gate of `prop:gate` holds literally, since the normalized modulation Gram has entries $G_{ij} \in [0,1]$. It also appears in the caption of Table tab:rf_comparison as the only way to remove the $(R^2+b)^4$ constant from the sample-complexity count.

## Proof idea
By `prop:biased_feature`, $\|p_b(x)\|^2 = (\|x\|^2+b)^2$, so $q_b = p_b/(\|x\|^2+b)$ has unit norm and the inner product follows by direct division: $q_b(x)^\top q_b(w) = p_b(x)^\top p_b(w)/((\|x\|^2+b)(\|w\|^2+b)) = (x^\top w+b)^2/((\|x\|^2+b)(\|w\|^2+b))$. The normalized kernel is the Schur (pointwise) product of this PSD cosine-type kernel and the IMQ radial factor $h_\varepsilon$, hence PSD by the Schur product theorem. The variance bound is `thm:exact_variance` applied with modulation value $a = q_b(x)^\top q_b(w) \le 1$: the bracket $1 + \tfrac12\varepsilon/(\varepsilon+4r) - (\varepsilon/(\varepsilon+r))^2 \le \tfrac32$ and $a^2 \le 1$ give $3/(2D\varepsilon^2)$.

## Connections
**Depends on:** `prop:biased_feature` ($\|p_b(x)\|^2 = (\|x\|^2+b)^2$); `thm:exact_variance` (the exact flat-estimator variance, applied at $a \le 1$); the Schur product theorem (PSD of the normalized kernel).
**Used by:** `prop:variance_sharp` (cited as the kernel-changing exit from the factored family); `prop:gate` (the normalized modulation Gram $G_{ij} \in [0,1]$ makes the alignment gate literal); Table tab:rf_comparison (caption: the only removal of $(R^2+b)^4$); the bias-cost discussion at line 394; open problem 2 ("normalizing changes the kernel").
**Validated by:** `normalized_ray.py` (Table tab:normalized, Section sec:exp_normalized): sweeping $b$ at $R=1$ and $R$ at $b=1$ with $2\times 10^5$ draws, the measured variance ratio exact/normalized matches $(R^2+b)^4$ to three figures; exact-modulation variance grows $0.02 \to 449$ across the sweep while the normalized variant stays in $[0.045, 0.53] \le \tfrac32$, exactly as the proposition predicts.

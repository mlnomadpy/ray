# Remark: Differentiability and Learnable $(\varepsilon, b)$
**Label:** `rmk:learnable` | **Location:** main.tex line 201

## What it says
The estimator is differentiable in its own kernel parameters with the base randomness held fixed. Reparameterize the radial draw as
$$t_j = -\varepsilon^{-1}\log u_j, \quad u_j \sim \operatorname{Unif}[0,1], \qquad \omega_j = \sqrt{2t_j}\,g_j, \quad g_j \sim \mathcal{N}(0, I_d),$$
so that $\varepsilon$ enters $z(x)$ smoothly through $t_j$ and $\omega_j$, while $b$ enters through the exact polynomial feature $p_b$. At every fixed $(\varepsilon, b)$ the map is unbiased (thm:unbiased), so a gradient of any downstream objective in $(\varepsilon, b)$ flows through an unbiased kernel estimate. Per-head learnable $(\varepsilon, b)$ in ⵟ-attention is therefore a reparameterization away, with $\varepsilon$ free to adapt to the local attention sharpness — the hard regime of the attention experiment (sec:exp_attention).

## Why it matters
This remark converts RAY from a fixed-kernel approximation device into a trainable layer. The two kernel hyperparameters are exactly the ones a practitioner wants to learn: $b$ controls the alignment bias and $\varepsilon$ the proximity bandwidth (and hence the sharpness of the attention smoother). Because the reparameterization isolates all parameter dependence in smooth deterministic maps of fixed base noise $(u_j, g_j)$ — the standard pathwise/reparameterization-trick structure — backpropagation through $z(x)$ is well-defined, and unbiasedness at every fixed $(\varepsilon, b)$ means the training signal is a gradient through an unbiased estimate of the true kernel rather than of a biased surrogate. This is the bridge from the theory to random-feature linear ⵟ-attention with per-head adaptive kernels.

## Proof idea
Two observations. (1) **Pathwise smoothness:** $\operatorname{Exp}(\varepsilon)$ is sampled by inverse-CDF, $t_j = -\varepsilon^{-1}\log u_j$, which is smooth in $\varepsilon$ for fixed $u_j$; the conditional Gaussian $\mathcal{N}(0, 2t_jI_d)$ is sampled by scaling fixed standard noise, $\omega_j = \sqrt{2t_j}\,g_j$, smooth in $t_j$. The cosine features and the entries of $p_b$ (polynomial in $x$ with coefficients $\sqrt{2b}$, $b$) are smooth in their arguments, so $z(x)$ is differentiable in $(\varepsilon, b)$ for $b > 0$ with the base randomness $(u_j, g_j, \beta_j)$ frozen. (2) **Unbiasedness pointwise in the parameters:** thm:unbiased holds at every fixed $(\varepsilon, b)$, so the estimator surface over parameter space sits, in expectation, exactly on the kernel surface.

## Connections
**Depends on:** def:ryf (the estimator being reparameterized), thm:unbiased (pointwise-in-parameters unbiasedness), prop:biased_feature (smooth $b$-dependence of $p_b$).
**Used by:** the ⵟ-attention experiment (sec:exp_attention), where adaptive $\varepsilon$ targets the sharp-attention hard regime; the linear-attention application framing in the introduction.
**Validated by:** `yat_attention.py`, `make_attention_fig.py` (the attention experiment exercising the regime the remark targets).

# Proposition: Biased Polynomial Feature
**Label:** `prop:biased_feature` | **Location:** main.tex line 128

## What it says
For $b \ge 0$ define the finite feature map
$$p_b(x) = \bigl(\operatorname{vec}(x \otimes x),\; \sqrt{2b}\,x,\; b\bigr)^\top \in \mathbb{R}^{d^2 + d + 1}.$$
Then
$$p_b(x)^\top p_b(w) = (x^\top w)^2 + 2b\,(x^\top w) + b^2 = (x^\top w + b)^2,$$
and the feature norm is exactly
$$\|p_b(x)\|^2 = (\|x\|^2 + b)^2.$$
So the biased degree-2 polynomial modulation $p_b(x,w) = (x^\top w + b)^2$ is a finite-feature kernel with an explicit, exact map. The feature is real exactly on the kernel's domain $b \ge 0$ (the entry $\sqrt{2b}$ requires it). By symmetry of $x \otimes x$ the dimension reduces losslessly from $d^2 + d + 1$ to $d_b = d(d+1)/2 + d + 1$ using the upper triangle with $\sqrt{2}$ scaling on off-diagonals; this symmetrized $p_b$ is the default from this point in the paper. Setting $b = 0$ recovers the unbiased feature $p_0(x) = \operatorname{vec}(x \otimes x)$.

## Why it matters
This is one of the two pillars of the Schur factorization (eq:schur): it proves the polynomial factor is positive definite by exhibiting its feature map, which combined with the Bernstein–Widder PSD-ness of the radial factor gives PSD-ness of $k_{ⵟ,b}$ by the Schur product theorem. It is also the object every later step manipulates: the exact-modulation estimator (def:ryf) tensors radial Fourier features with $p_b$; the deployed RAY estimator (Step 5) sketches exactly this feature to escape its $O(d^2)$ size; and the norm identity $\|p_b(x)\|^2 = (\|x\|^2 + b)^2$ supplies the bound $B = R^2 + b$ that sets the scale of every variance and concentration result in the paper — the exact $(R^2+b)^4$ bias law confirmed empirically is this identity propagated through the variance. Without it there is no factorization, no estimator, and no constants.

## Proof idea
Direct computation. The Kronecker-square identity $\operatorname{vec}(x \otimes x)^\top \operatorname{vec}(w \otimes w) = (x^\top w)^2$ gives the quadratic term; the $\sqrt{2b}\,x$ blocks contribute $2b\,x^\top w$; the constant coordinates contribute $b \cdot b = b^2$. The three terms assemble the binomial expansion of $(x^\top w + b)^2$. Setting $w = x$ gives $\|p_b(x)\|^2 = \|x\|^4 + 2b\|x\|^2 + b^2 = (\|x\|^2 + b)^2$.

## Connections
**Depends on:** the Kronecker-square vectorization identity; the domain restriction $b \ge 0$.
**Used by:** eq:schur (PSD-ness of the polynomial factor); def:ryf step 4 (the exact modulation block $p(x) = \varepsilon^{-1/2} p_b(x)$); thm:unbiased (the exact inner product $p(x)^\top p(w) = (x^\top w + b)^2/\varepsilon$); thm:bernstein_schur (the ⵟ-kernel is the instance $u = p_b$ with $B = R^2 + b$); Step 5 / thm:ts_opnorm (TensorSketch of this feature); the variance scale $(R^2+b)^4/\varepsilon^2$ throughout Section sec:guarantees; prop:imq_findiff (the quadratic-in-$b$ structure it encodes).
**Validated by:** `bias_scaling.py` (the exact $(R^2+b)^4$ bias law), `exact_variance_check.py`.

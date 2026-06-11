# Corollary: Sample Complexity
**Label:** `cor:sample_complexity` | **Location:** main.tex line 777

## What it says
$$D = O\!\bigl((R^2+b)^4\,\varepsilon^{-2}\,\tau^{-2}\,\log(N/\eta)\bigr)$$

outer radial samples suffice for the exact-modulation estimator to achieve entrywise Gram error at most $\tau$ over all $N^2$ pairs simultaneously, with probability $1-\eta$, **independent of the input dimension $d$**.

## Why it matters
This is the headline sample-complexity count of the dataset-level analysis, and the entry the paper puts in the comparison Table tab:rf_comparison: RAY's count is exactly the radial IMQ–Laplace $\varepsilon^{-2}$ multiplied by the unbounded polynomial modulation range $(R^2+b)^4$ — the honest price of the alignment numerator, removable only by normalizing the modulation (`prop:normalized`). The corollary also carries the paper's carefully scoped dimension-freeness claim (Section sec:comparison, line 410): at this dataset level (union bound over $\le N^2$ pairs), the count is free of explicit $d$, but so is standard RFF for Gaussian and IMQ kernels — dimension-freeness is a property of the dataset-level analysis, not a special feature of the construction. The genuine contrast with RFF is qualitative applicability: RFF needs shift-invariance, polynomial sketches need dot-product form, and k_ⵟ,b off-sphere has neither (`prop:nonstationary`). The corollary is what makes the radial half of RAY a fixed, dataset-size-driven budget while the modulation half carries the $d$-dependence (the $O(d^2)$ exact feature, sketched away in `thm:ts_opnorm`).

## Proof idea
Invert `thm:uniform`. Setting the right-hand side $\frac{(R^2+b)^2}{\varepsilon}\sqrt{8\log(2N^2/\delta)/D}$ equal to the target $\tau$ and solving for $D$ gives $D = 8(R^2+b)^4\varepsilon^{-2}\tau^{-2}\log(2N^2/\eta)$, and $\log(2N^2/\eta) = O(\log(N/\eta))$. No step introduces $d$: the Hoeffding range $c = 2(R^2+b)^2/\varepsilon$ depends only on the data radius, bias, and radial scale, and the union bound depends only on $N$.

## Connections
**Depends on:** `thm:uniform` (the entrywise Hoeffding + union bound), hence the a.s. trigonometric bound and the deterministic modulation bound behind it.
**Used by:** Table tab:rf_comparison and the overclaiming-guard discussion of Section sec:comparison; the construction narrative (line 165: the fixed-dataset count "carries no explicit $d$-dependence"); `thm:gram_concentration` (same chain, pushed to operator norm); the contributions list (item on sharp variance and Gram concentration).
**Validated by:** `dimension_free.py` (Table tab:dimfree, Figure fig:overview b): the empirical radial count $D^\star$ for relative Frobenius error $\le 0.10$ stays bounded and plateaus as $d$ runs from $2$ to $100$, while Nyström's landmark count exceeds the tested range ($>350$) for $d \ge 20$.

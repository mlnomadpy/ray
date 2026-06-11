# Corollary: High-Probability Operator-Norm Bound
**Label:** `cor:bernstein_tail` | **Location:** main.tex line 277

## What it says

With the summands $Y_j = D^{-1}(K^{(j)} - K)$, the almost-sure bound

$$\|Y_j\|_{\mathrm{op}} \le L := \frac{3\|P\|_{\mathrm{op}}}{D},$$

and the matrix variance

$$\Bigl\|\sum_j \mathbb{E}[Y_j^2]\Bigr\|_{\mathrm{op}} \le v := \frac{2\|P\|_{\mathrm{op}}\|K\|_{\mathrm{op}}}{D},$$

the tail form of the intrinsic matrix-Bernstein inequality (Tropp 2015, Thm. 7.3.1) gives, with probability at least $1 - \delta$,

$$\|K_D - K\|_{\mathrm{op}} \le 2\sqrt{\frac{\|P\|_{\mathrm{op}}\|K\|_{\mathrm{op}}\log(8d_{\mathrm{int}}/\delta)}{D}} + \frac{2\|P\|_{\mathrm{op}}\log(8d_{\mathrm{int}}/\delta)}{D}.$$

This is the high-probability counterpart to `thm:bernstein`: the same $O(D^{-1/2})$ leading term governed by top eigenvalues and intrinsic dimension, now as a tail event.

## Why it matters

The expectation bound of `thm:bernstein` cannot be conditioned on; downstream guarantees need an **event**. This corollary is the form that actually propagates: `thm:ts_opnorm` applies it verbatim conditioned on the sketch (with $P \mapsto \widehat P_m$, $K \mapsto K_S$) to bound the deployed doubly-randomized estimator, and its quadratic-root inversion is reused as a subroutine inside the proofs of `thm:krr_whitened` and `thm:krr_leverage`. Without the tail form, the operator-norm analysis would stop at an average-case statement and the deployed-estimator and KRR chains could not be assembled.

## Proof idea

The constants $L$ and $v$ are established exactly as in the proof of `thm:bernstein`: per-draw positivity $K^{(j)} \succeq 0$ plus the Schur-multiplier bound (`lem:schur`) give $\|K^{(j)}\|_{\mathrm{op}} \le 2\|P\|_{\mathrm{op}}$ and $\|K\|_{\mathrm{op}} \le \|P\|_{\mathrm{op}}$, whence $L = 3\|P\|_{\mathrm{op}}/D$ and $(K^{(j)})^2 \preceq 2\|P\|_{\mathrm{op}} K^{(j)}$ gives $v = 2\|P\|_{\mathrm{op}}\|K\|_{\mathrm{op}}/D$. The intrinsic tail inequality states

$$\mathbb{P}\Bigl\{\bigl\|\textstyle\sum_j Y_j\bigr\|_{\mathrm{op}} \ge s\Bigr\} \le 8 d_{\mathrm{int}} \exp\Bigl(-\frac{s^2/2}{v + Ls/3}\Bigr) \quad \text{for } s \ge \sqrt v + L/3,$$

where the factor $8$ comes from applying the one-sided intrinsic Bernstein tail to both signs. Set the right side to $\delta$, write $\ell = \log(8d_{\mathrm{int}}/\delta)$, and invert: the quadratic $s^2 - \tfrac23 L\ell\, s - 2v\ell = 0$ has positive root $s = \tfrac13 L\ell + \sqrt{(\tfrac13 L\ell)^2 + 2v\ell} \le \sqrt{2v\ell} + \tfrac23 L\ell$. Since $\ell\ge\log 8>1$, the root satisfies the required range $s\ge\sqrt v+L/3$. Substituting $v$ and $L$ gives the displayed bound (the linear coefficient is $\tfrac23 L = 2\|P\|_{\mathrm{op}}/D$).

## Connections

**Depends on:** `thm:bernstein` (same $L$, $v$, and structural facts), `lem:schur`, the tail form of intrinsic matrix Bernstein (Tropp 2015, Thm. 7.3.1).
**Used by:** `thm:ts_opnorm` (conditioned on the sketch, this is the radial term of the deployed bound; at $m \to \infty$ the deployed bound reduces exactly to this corollary), `thm:krr_whitened` and `thm:krr_leverage` (the quadratic-root inversion is reused), `thm:class_bernstein` (holds verbatim with $P \mapsto P_u$).
**Validated by:** `opnorm_validation.py`.

# Lemma: Schur Multiplier Bound
**Label:** `lem:schur` | **Location:** main.tex line 787

## What it says

Let $A \succeq 0$ with bounded diagonal, $\max_i A_{ii} \le a$. Then for the Schur (entrywise) product $\circ$:

**(a)** If $P \succeq 0$, then

$$0 \preceq A \circ P \preceq a\,\|P\|_{\mathrm{op}}\, I.$$

In particular $A \circ P$ is PSD with $\|A \circ P\|_{\mathrm{op}} \le a\,\|P\|_{\mathrm{op}}$.

**(b)** If $E = E^\top$ is merely symmetric — **not** necessarily PSD — then

$$\|A \circ E\|_{\mathrm{op}} \le a\,\|E\|_{\mathrm{op}}.$$

So a PSD matrix with diagonal at most $a$ acts as a Schur multiplier of norm at most $a$ on all symmetric matrices, not just on the PSD cone.

## Why it matters

This single lemma is the structural engine of the entire matrix-level guarantee set. Every per-draw Gram in the paper has the form $K^{(j)} = (\Psi_j\Psi_j^\top) \circ P$ with $(\Psi_j\Psi_j^\top)_{ii} \le 2$; part (a) is what makes $K^{(j)} \succeq 0$ with $\|K^{(j)}\|_{\mathrm{op}} \le 2\|P\|_{\mathrm{op}}$, and likewise $\|K\|_{\mathrm{op}} = \|R \circ P\|_{\mathrm{op}} \le \|P\|_{\mathrm{op}}$ (the radial Gram $R$ is PSD with unit diagonal). These two facts feed directly into the a.s. bound and variance majorant of `thm:bernstein` and all its descendants. Part (b) is indispensable for the sketched estimator: the sketch error $E_P = \widehat P_m - P$ is symmetric but **not** PSD, and `thm:ts_opnorm` and `prop:ridge_sketch` need $\|R \circ E_P\|_{\mathrm{op}} \le \|E_P\|_{\mathrm{op}}$, which only part (b) supplies. Without (b), the radial modulation could not be shown to leave the sketch error uninflated, and the additive sketch term $\eta\|P\|_{\mathrm{op}}$ in the deployed bound would break.

## Proof idea

Factor $A = UU^\top = \sum_r u_r u_r^\top$ with $u_r$ the columns of $U$. The key identity is that Schur multiplication by a rank-one symmetric matrix is diagonal conjugation: $(u_r u_r^\top) \circ E = D_r E D_r$ with $D_r = \operatorname{diag}(u_r)$, so $A \circ E = \sum_r D_r E D_r$. The diagonal matrix $\sum_r D_r^2$ has entries $\sum_r (u_r)_i^2 = A_{ii} \le a$, hence $\sum_r D_r^2 \preceq aI$. For a unit vector $x$ and symmetric $E$,

$$\bigl|x^\top (A \circ E)\, x\bigr| = \Bigl|\sum_r (D_r x)^\top E (D_r x)\Bigr| \le \|E\|_{\mathrm{op}} \sum_r \|D_r x\|^2 = \|E\|_{\mathrm{op}}\, x^\top\Bigl(\sum_r D_r^2\Bigr)x \le a\,\|E\|_{\mathrm{op}},$$

which is (b). For (a), take $E = P \succeq 0$: each summand $(D_r x)^\top P (D_r x)$ is nonnegative, so the absolute value is free, giving $0 \le x^\top (A \circ P)x \le a\|P\|_{\mathrm{op}}$; positivity of $A \circ P$ is the Schur product theorem.

## Connections

**Depends on:** the Schur product theorem; the Gram factorization $A = UU^\top$; the rank-one identity $(uu^\top) \circ E = \operatorname{diag}(u)\, E\, \operatorname{diag}(u)$.
**Used by:** `thm:bernstein` (per-draw norm $\|K^{(j)}\|_{\mathrm{op}} \le 2\|P\|_{\mathrm{op}}$ and $\|K\|_{\mathrm{op}} \le \|P\|_{\mathrm{op}}$), `cor:bernstein_tail`, `thm:ts_opnorm` (part (b) for the sketch bias $R \circ E_P$), `prop:ridge_sketch` (part (a) for the ridge sandwich transfer), `thm:krr_whitened`, `thm:krr_leverage` (PSD-ness of $A^{-1} \circ P$), `cor:krr_deployed`, `thm:class_bernstein` (structural fact (i) with $P_u$ in place of $P$).
**Validated by:** indirectly through every operator-norm experiment; no dedicated script (the lemma is exact algebra).

# Proposition: Ridge-Relative Sketch Transfer
**Label:** `prop:ridge_sketch` | **Location:** main.tex line 300

## What it says

Suppose the modulation sketch is a $(1 \pm \eta)$ ridge subspace embedding of the modulation Gram $P$ at scale $\lambda$ (the event of `rmk:ose`):

$$(1-\eta)(P + \lambda I) \preceq \widehat P_m + \lambda I \preceq (1+\eta)(P + \lambda I).$$

Then the exact-radial deployed Gram $K_S = \widehat P_m \circ R$ satisfies the **same sandwich at the same ridge scale** with respect to the kernel Gram $K = P \circ R$:

$$(1-\eta)(K + \lambda I) \preceq K_S + \lambda I \preceq (1+\eta)(K + \lambda I),$$

equivalently, with $A = K + \lambda I$,

$$\bigl\|A^{-1/2}(K_S - K)A^{-1/2}\bigr\|_{\mathrm{op}} \le \eta.$$

Consequently $\|K_S - K\|_{\mathrm{op}} \le \eta(\|K\|_{\mathrm{op}} + \lambda) \le \eta(\|P\|_{\mathrm{op}} + \lambda)$, which sharpens the additive sketch term $\eta\|P\|_{\mathrm{op}}$ of `thm:ts_opnorm` whenever $\lambda \ll \|P\|_{\mathrm{op}}$.

## Why it matters

This is the lossless bridge from sketch guarantees on the modulation factor to sketch guarantees on the full ⵟ-kernel Gram. The sketch literature proves ridge embeddings of $P$ with size polynomial in the statistical dimension $s_\lambda(P)$; what KRR needs is a whitened bound on $K_S - K$ relative to $A = K + \lambda I$. This proposition shows the Schur product with the radial factor transports the entire ridge sandwich — same $\eta$, same $\lambda$, no degradation — so the deployed estimator gets a ridge-relative sketch guarantee for free. It is the load-bearing step in `cor:krr_deployed` (it supplies the deterministic sketch part $\rho_{\mathrm{sk}} \le \eta$) and it closes Remark 4.5's gap between the absolute spectral event (rank-of-$P$ sketch sizes) and the deployable $s_\lambda(P)$-sized sketch. Without it, the KRR condition for the deployed estimator would require $\eta \lesssim \lambda/\|P\|_{\mathrm{op}}$ — a scale-dependent, potentially brutal sketch accuracy — instead of the scale-free constant $\eta \le \rho_0/4$.

## Proof idea

The whole mechanism is that $R$ is PSD with **unit diagonal**, so the Schur product with $R$ fixes the identity: $R \circ I = I$, hence $R \circ (M + \lambda I) = R \circ M + \lambda I$ for every symmetric $M$. In particular

$$R \circ (P + \lambda I) = K + \lambda I, \qquad R \circ (\widehat P_m + \lambda I) = K_S + \lambda I.$$

Both gaps in the hypothesis sandwich, $(1+\eta)(P+\lambda I) - (\widehat P_m + \lambda I)$ and $(\widehat P_m + \lambda I) - (1-\eta)(P + \lambda I)$, are PSD; Schur-multiplying each by $R \succeq 0$ keeps it PSD (Schur product theorem, Lemma `lem:schur`(a)). Substituting the two identities gives the transferred sandwich. Conjugating by $A^{-1/2}$ gives the whitened form. For the operator-norm consequence: the sandwich gives $-\eta(K + \lambda I) \preceq K_S - K \preceq \eta(K + \lambda I)$, so $\|K_S - K\|_{\mathrm{op}} \le \eta(\|K\|_{\mathrm{op}} + \lambda)$, and $\|K\|_{\mathrm{op}} \le \|P\|_{\mathrm{op}}$ by Lemma `lem:schur` ($R$ unit-diagonal).

## Connections

**Depends on:** The ridge-subspace-embedding event of `rmk:ose`, Lemma `lem:schur`(a) (Schur product of PSD matrices is PSD) and the unit-diagonal Schur-multiplier bound, the unit-diagonal structure of the radial Gram $R_{ij} = \varepsilon/(\varepsilon + \|x_i - x_j\|^2)$.
**Used by:** `cor:krr_deployed` (supplies the deterministic sketch part of the decomposition), `thm:ts_opnorm` (sharpens its additive sketch term when $\lambda \ll \|P\|_{\mathrm{op}}$), `rmk:ose` (realizes the transfer it announces).
**Validated by:** — (deterministic matrix inequality; exercised end-to-end through the deployed-KRR experiments and `ts_opnorm_validation.py`).

# Corollary: Operator-Norm Control (the Crude N-Factor Route)
**Label:** `thm:gram_concentration` | **Location:** main.tex line 780

## What it says
Since $\|A\|_{\mathrm{op}} \le \|A\|_F \le N \max_{ij} |A_{ij}|$, Theorem `thm:uniform` immediately gives, with probability $1-\delta$,

$$\|K_D - K\|_{\mathrm{op}} \le N\,\frac{(R^2+b)^2}{\varepsilon}\sqrt{\frac{8\log(2N^2/\delta)}{D}}.$$

The factor $N$ is the price of this elementary route. Theorem `thm:bernstein` removes it via matrix concentration, replacing $N\max_{ij}|\cdot|$ with the top eigenvalues $\|P\|_{\mathrm{op}}, \|K\|_{\mathrm{op}}$ of the modulation and kernel Gram matrices and an intrinsic dimension $d_{\mathrm{int}}$ (which can still scale with $N$ in the worst case).

## Why it matters
This corollary exists to be beaten, and saying so precisely is its job. It is the baseline operator-norm guarantee obtainable from entrywise concentration alone, and its explicit factor $N$ is what motivates the paper's matrix-level program: the per-draw error matrices are not unstructured — each radial draw contributes $K^{(j)} = (\Psi_j\Psi_j^\top) \circ P$, a Schur product of PSD matrices and hence itself PSD — and matrix Bernstein exploits exactly that structure (line 267). The comparison is kept honest in both directions: `thm:bernstein` never exceeds this corollary's order (since $\|P\|_{\mathrm{op}} \le \mathrm{tr}(P) \le N(R^2+b)^2/\varepsilon$ and $d_{\mathrm{int}} \le N$) and is data-adaptively tighter for spectrally concentrated data (line 275). Removing this $N$ factor is the first step of the chain that ends in the whitened KRR guarantees (`thm:krr_whitened`, `thm:krr_leverage`) where the effective dimension replaces $N$ outright.

## Proof idea
No separate proof is needed (line 731): it is the norm chain $\|A\|_{\mathrm{op}} \le \|A\|_F \le N\max_{ij}|A_{ij}|$ applied to the entrywise event of `thm:uniform`. The Frobenius step costs $\sqrt{N^2} = N$ over the max entry; the operator-norm step is free. Every bit of structure in the error matrix — PSD per-draw Grams, the Schur factorization — is discarded, which is precisely the slack `thm:bernstein` recovers.

## Connections
**Depends on:** `thm:uniform` (the entrywise high-probability bound); the elementary norm inequalities $\|A\|_{\mathrm{op}} \le \|A\|_F \le N\max_{ij}|A_{ij}|$.
**Used by:** `thm:bernstein` (stated and proved as the bound this corollary's $N$ factor motivates; the worst-case comparison $\|P\|_{\mathrm{op}} \le N(R^2+b)^2/\varepsilon$, $d_{\mathrm{int}} \le N$ shows Bernstein never does worse); the Hoeffding-chain narrative of Section sec:guarantees (line 267).
**Validated by:** `opnorm_validation.py` and `bernstein_intrinsic.py` (Section sec:exp_gram): the measured operator-norm error tracks the data-adaptive Bernstein constant across datasets of varying spectral spread, confirming the $N$-factor route is the loose one.

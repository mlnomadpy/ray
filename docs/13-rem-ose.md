# Remark: Achieving the Spectral Event
**Label:** `rmk:ose` | **Location:** main.tex line 296

## What it says

This remark settles when the hypothesis of `thm:ts_opnorm` — the spectral event $\|\widehat P_m - P\|_{\mathrm{op}} \le \eta\|P\|_{\mathrm{op}}$ — actually holds, and identifies the more useful guarantee.

The degree-2 TensorSketch is an **oblivious subspace embedding** (Pham–Pagh 2013, Avron–Nguyen–Woodruff 2014): with sketch dimension $m$ polynomial in the statistical dimension

$$s_\lambda = \operatorname{tr}\bigl(P(P + \lambda I)^{-1}\bigr)$$

and in $\eta^{-1}$ (the exact dependence is sketch- and degree-specific — up to degree-dependent factors, and the original degree-2 TensorSketch bound of Avron et al. is superlinear in $s_\lambda$), it delivers with probability $1 - \delta_s$ a $(1 \pm \eta)$ **ridge subspace embedding**:

$$(1-\eta)(P + \lambda I) \preceq \widehat P_m + \lambda I \preceq (1+\eta)(P + \lambda I).$$

This is a *regularized* guarantee. The absolute event $\|\widehat P_m - P\|_{\mathrm{op}} \le \eta\|P\|_{\mathrm{op}}$ assumed in `thm:ts_opnorm` is its $\lambda \to 0$ idealization: there $s_\lambda \to \operatorname{rank}(P)$ and the required sketch count becomes $\Omega(\eta^{-2}\operatorname{rank}(P))$.

**The ridge-embedding event is the more useful one**, because it transfers through the Schur product without loss: `prop:ridge_sketch` carries it from the modulation Gram $P$ to the kernel Gram $K$ at the *same* ridge scale, so the deployed estimator inherits a ridge-relative sketch guarantee with sketch size polynomial in $s_\lambda(P)$ rather than $\operatorname{rank}(P)$. The entrywise decomposition of `prop:ts_variance` needs no embedding hypothesis at all.

## Why it matters

`thm:ts_opnorm` is stated conditionally on the spectral event; without this remark the hypothesis would be an unverified assumption and the deployed guarantee vacuous in practice. The remark does two jobs. First, it grounds the absolute event in the OSE literature and exposes its cost honestly: rank-of-$P$ sketch sizes in the $\lambda \to 0$ idealization. Second, it redirects the analysis toward the ridge-relative route — the regularized sandwich that the sketch literature actually proves — which is exactly the path `prop:ridge_sketch` and `cor:krr_deployed` then take, replacing $\operatorname{rank}(P)$ by the statistical dimension $s_\lambda(P)$ and yielding the scale-free sketch requirement $\eta \le \rho_0/4$ downstream. The remark is the hinge between the absolute-event operator-norm bound and the ridge-relative KRR pipeline.

## Proof idea

Not a proof but a citation plus a structural observation. The OSE property of TensorSketch for degree-2 polynomial kernels is the cited result of Pham–Pagh and Avron–Nguyen–Woodruff: the sketch preserves the ridge-regularized quadratic form $(1\pm\eta)$-multiplicatively once $m$ is polynomial in $s_\lambda$ and $\eta^{-1}$. The relation between the two events is elementary: as $\lambda \to 0$ the ridge sandwich degenerates to the absolute multiplicative event, while $s_\lambda$ climbs to $\operatorname{rank}(P)$, which prices the idealization at $\Omega(\eta^{-2}\operatorname{rank}(P))$ sketch rows. The transfer-without-loss claim is delegated to `prop:ridge_sketch`, whose one-line mechanism is that $R$ has unit diagonal, so Schur-multiplying by $R$ fixes the ridge term $\lambda I$ exactly.

## Connections

**Depends on:** Oblivious-subspace-embedding results for degree-2 TensorSketch (Pham–Pagh 2013; Avron–Nguyen–Woodruff 2014), the spectral-event hypothesis of `thm:ts_opnorm`.
**Used by:** `thm:ts_opnorm` (justifies its hypothesis), `prop:ridge_sketch` (executes the ridge-relative transfer this remark advocates), `cor:krr_deployed` (inherits the $s_\lambda(P)$-sized sketch and the scale-free $\eta \le \rho_0/4$ requirement).
**Validated by:** — (definitional/literature remark; the downstream pipeline is validated through fig:ts_opnorm and the KRR experiments).

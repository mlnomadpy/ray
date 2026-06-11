# Proposition: RAY Error Decomposition
**Label:** `prop:ts_variance` | **Location:** main.tex line 863

## What it says

The variance of the deployed (doubly-randomized) RAY estimate of a single kernel entry splits exactly into three terms: radial Monte Carlo, polynomial sketch, and their interaction.

**Setup (quadratic-only sketch).** Sketch only the quadratic term:

$$\widehat p_m(x,w) = \mathrm{TS}_2(x)^\top \mathrm{TS}_2(w) + 2b\,x^\top w + b^2, \qquad \mathbb{E}[\mathrm{TS}_2(x)^\top\mathrm{TS}_2(w)] = (x^\top w)^2,$$

so $\mathbb{E}[\widehat p_m] = p = (x^\top w + b)^2$ — the linear and constant terms are kept exact and add no sketch variance. Let $\widehat p_m$ be independent of the unbiased radial estimate $\widehat h_D$ of $h(x,w) = (\|x-w\|^2 + \varepsilon)^{-1}$.

**Conclusion.** The product $\widehat k_{D,m} = \widehat p_m\,\widehat h_D$ is unbiased for $k_{ⵟ,b}$, and

$$\mathrm{Var}[\widehat k_{D,m}] = \underbrace{p^2\,\mathrm{Var}[\widehat h_D]}_{\text{radial Monte Carlo}} + \underbrace{h^2\,\mathrm{Var}[\widehat p_m]}_{\text{polynomial sketch}} + \underbrace{\mathrm{Var}[\widehat p_m]\,\mathrm{Var}[\widehat h_D]}_{\text{interaction}}.$$

The degree-2 TensorSketch bound $\mathrm{Var}[\widehat p_m] \le C\,\|x\|^4\|w\|^4/m$ (the bias terms being exact) makes the sketch terms vanish as $m \to \infty$, recovering the exact-modulation estimator.

## Why it matters

This is the entrywise accounting identity behind the entire deployed-estimator design. It says the radial term scales as $D^{-1}$ in variance, the sketch term as $m^{-1}$, and the cross term as $(Dm)^{-1}$ — so the two randomizations are separately tunable and their errors do not conspire. It is the no-hypothesis fallback to `thm:ts_opnorm` (it needs no subspace-embedding event whatsoever, only independence and unbiasedness), it is the input to the optimal-allocation rule `prop:optimal_m` (whose two-term budget law is exactly this decomposition with $D = M/(m+d+1)$ substituted), and it explains the experimental structure of Section sec:exp_ts: varying $D$ at fixed $m$ moves only the radial term, varying $m$ at fixed $D$ only the sketch term. It also justifies the quadratic-only design choice: keeping $2b\,x^\top w + b^2$ exact costs only an additive $O(d)$ per draw and removes those terms from the sketch variance entirely.

## Proof idea

Pure independence algebra. By independence of $\widehat p_m$ and $\widehat h_D$,

$$\mathbb{E}[\widehat p_m \widehat h_D] = p\,h = k,$$

so the product is unbiased, and

$$\mathbb{E}[\widehat p_m^2 \widehat h_D^2] = \mathbb{E}[\widehat p_m^2]\,\mathbb{E}[\widehat h_D^2] = (p^2 + \mathrm{Var}\,\widehat p_m)(h^2 + \mathrm{Var}\,\widehat h_D).$$

Expanding and subtracting $k^2 = p^2 h^2$ leaves exactly the three terms: $p^2\mathrm{Var}\,\widehat h_D + h^2\mathrm{Var}\,\widehat p_m + \mathrm{Var}\,\widehat p_m\,\mathrm{Var}\,\widehat h_D$. The TensorSketch variance bound is the standard degree-2 result, applied only to the quadratic piece since the linear and constant pieces are deterministic.

## Connections

**Depends on:** Independence of the sketch and radial randomness, unbiasedness of both factors (`thm:unbiased` for the radial estimate, the TensorSketch unbiasedness for the quadratic term), the degree-2 TensorSketch variance bound (Pham–Pagh 2013, Avron et al. 2014).
**Used by:** `prop:optimal_m` (substitutes the budget constraint into this decomposition), `thm:ts_opnorm` and `rmk:ose` (cite it as the embedding-hypothesis-free fallback), Section sec:exp_ts (the validated split), Table tab:ts.
**Validated by:** `ts_decomposition.py` (fixed pair, $4000$ repetitions: varying $D$ at $m{=}256$ drives only the radial term, $0.09 \to 8.5\times10^{-4}$; varying $m$ at $D{=}1000$ only the sketch term, $4.2\times10^{-3} \to 1.4\times10^{-4}$; empirical variance matches the three-term formula at ratio $0.94$–$1.06$ throughout), `ts_ryf_costmatched.py` (Table tab:ts: at matched dimension $M = d_b$ on $d{=}64$ digits, RAY $m{=}128$ reaches $0.977$ where exact modulation starved to one radial draw scores $0.928$).

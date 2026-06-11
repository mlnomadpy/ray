# Proposition: Exponential Proposal for the Radial Factor
**Label:** `prop:importance` | **Location:** main.tex line 1106

## What it says
Importance-sample the radial scale with the tilted exponential proposal $T \sim \mathrm{Exp}(\varepsilon + \eta)$, $\eta \ge 0$, and the reweighted estimator

$$\widehat{h}_\eta(r) = \frac{1}{\varepsilon+\eta}\,e^{-(r-\eta)T}.$$

Then for every squared distance $r$:

1. **Unbiasedness:** $\mathbb{E}[\widehat{h}_\eta(r)] = (r + \varepsilon)^{-1}$, for every $\eta \ge 0$.
2. **Finite-variance window:** $\mathbb{E}[\widehat{h}_\eta(r)^2] = \bigl((\varepsilon+\eta)(\varepsilon + 2r - \eta)\bigr)^{-1}$, which is finite **iff** $\eta < \varepsilon + 2r$.

A proposal that is safe for all pairs simultaneously, including coincident points $r = 0$, therefore requires $\eta < \varepsilon$.

## Why it matters
This is the variance-reduction toolbox entry for the outer Bernstein scale: tilting the radial proposal toward larger scales reduces variance for pairs at moderate distance, and the proposition delimits exactly how far the tilt can go before the second moment blows up. The window $\eta < \varepsilon + 2r$ is pair-dependent, and the safe-for-all-pairs condition $\eta < \varepsilon$ is the deployable constraint for a single data-independent proposal. The empirical follow-up (line 1113) is honest about the payoff: the reduction is real but tuning-dependent — variance ratios down to $0.42\times$ at $r = 1$ (with $\eta = 0.5$) and $0.55\times$ at $r = 0.25$ (with $\eta = 0.1$), but *no* improvement for the nearest pair $r = 0.05$, where $\eta = 0$ is already best. Since the optimal $\eta$ is data-dependent, a single fixed proposal is a compromise; this limitation is restated in the discussion ("the data-independent importance proposal is likewise suboptimal when distances spread widely", line 709). The same tilt-the-radial-law idea, executed with the principled whitened-leverage density instead of a fixed exponential, is what `thm:krr_leverage` develops at the matrix level.

## Proof idea
Both moments are elementary Laplace-transform integrals against the proposal density $(\varepsilon+\eta)e^{-(\varepsilon+\eta)t}$.

**Mean:** $\mathbb{E}[\widehat{h}_\eta(r)] = \int_0^\infty \frac{1}{\varepsilon+\eta}e^{-(r-\eta)t}\,(\varepsilon+\eta)e^{-(\varepsilon+\eta)t}\,dt = \int_0^\infty e^{-(r+\varepsilon)t}\,dt = (r+\varepsilon)^{-1}$ — the tilt $\eta$ cancels between the weight and the proposal, leaving exactly the Bernstein integral for the IMQ radial factor.

**Second moment:** $\mathbb{E}[\widehat{h}_\eta(r)^2] = \frac{1}{\varepsilon+\eta}\int_0^\infty e^{-(\varepsilon+2r-\eta)t}\,dt$, which converges iff the exponent $\varepsilon + 2r - \eta > 0$ and then equals $((\varepsilon+\eta)(\varepsilon+2r-\eta))^{-1}$. The worst pair is $r = 0$, giving the global condition $\eta < \varepsilon$.

## Connections
**Depends on:** the Bernstein–Widder representation of the radial factor $h_\varepsilon(r) = \int_0^\infty e^{-tr}\,\varepsilon e^{-\varepsilon t}\,dt$ (Section sec:step-bernstein); elementary exponential integrals.
**Used by:** the variance-reduction menu of Section sec:variance_reduction; the limitations discussion (line 709, data-independent proposal suboptimal under spread distances); conceptual precursor to the whitened-leverage tilt of `thm:krr_leverage`, which replaces the fixed exponential tilt by the closed-form leverage density.
**Validated by:** `importance_sampling.py` (Appendix app:exp_details, importance-sampling paragraph at line 1113): on pairs at several squared distances ($x^\top w = 0.5$, $b = 1$, $\varepsilon = 1$, $\eta < \varepsilon$, $D = 200$, 3000 repetitions), bias $\le 10^{-3}$ throughout, variance ratios $0.42\times$ ($r=1$, $\eta=0.5$) and $0.55\times$ ($r=0.25$, $\eta=0.1$), no improvement at $r=0.05$.

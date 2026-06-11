# RAY / Bernstein–Schur kernels — theory documentation

One file per result in `main.tex` (TMLR preprint, *Bernstein–Schur Kernels: Random
Features by Sketched Modulation and Radial Randomization*). Same template as the
monograph docs at the repo root: What it says / Why it matters / Proof idea / Connections.
The full audit and gap inventory is [AUDIT.md](AUDIT.md).

**The kernel.** k_ⵟ,b(w,x) = (wᵀx+b)²/(‖w−x‖²+ε) — neither shift-invariant nor
dot-product off the sphere. **The class.** Bernstein–Schur kernels = (finite-feature
modulation u(x)ᵀu(w)) × (completely monotone radial f(‖x−w‖²) of finite mass m_f=f(0)).
**The estimator (RAY).** Sample the radial Bernstein–Widder scale t~Exp(ε), apply
Gaussian RFF at that scale, tensor with the modulation feature; sketch the modulation
(degree-2 TensorSketch, width m) so the deployed feature dimension is Dm, free of O(d²).

## Reading order (the spine)

1. **Construction:** [01](01-prop-nonstationary.md) why no template applies →
   [02](02-prop-biased-feature.md) exact modulation feature →
   [03](03-def-ryf.md) the exact-modulation map →
   [04](04-thm-unbiased.md) unbiasedness →
   [06](06-thm-bernstein-schur.md) the class theorem →
   [07](07-ex-ray-coordinate.md) one coordinate end to end.
2. **Variance:** [09](09-thm-exact-variance.md) exact flat variance →
   [26](26-thm-variance.md) envelope → [27](27-prop-variance-sharp.md) prefactor
   sharpness → [28](28-prop-budget.md) flat D′=1 optimal →
   [33](33-prop-normalized.md) the bounded-variance rescaled kernel.
3. **Concentration → KRR (the guarantee chain):**
   [32](32-lem-schur.md) Schur multiplier →
   [29](29-thm-uniform.md)/[30](30-cor-sample-complexity.md)/[31](31-cor-gram-concentration.md)
   entrywise route → [10](10-thm-bernstein.md) matrix Bernstein →
   [11](11-cor-bernstein-tail.md) tail →
   [15](15-thm-krr-spectral.md) deterministic interface →
   [16](16-thm-krr-whitened.md) d_eff is the exact whitened intrinsic dimension →
   [17](17-cor-krr-highprob.md) high-probability KRR →
   [18](18-thm-krr-leverage.md) leverage tilt achieves the d_eff count →
   [19](19-rem-risk.md) what still separates this from a risk theorem.
4. **The deployed (sketched) estimator:** [12](12-thm-ts-opnorm.md) one additive sketch
   term → [13](13-rem-ose.md)/[14](14-prop-ridge-sketch.md) ridge-embedding transfer →
   [20](20-cor-krr-deployed.md) deployed KRR condition →
   [34](34-prop-ts-variance.md)/[35](35-prop-optimal-m.md) sketch-size allocation.
5. **Class-level transfer:** [21](21-thm-class-bernstein.md) everything above for every
   Bernstein–Schur kernel (P ↦ m_f·G_u).
6. **Attention & alternatives:** [22](22-prop-gate.md) modulation gates radial noise →
   [24](24-prop-positive.md) FAVOR⁺ positivity →
   [25](25-prop-pos-dichotomy.md) infinite variance for 8xᵀw≥ε →
   [23](23-prop-quadrature.md) deterministic radial nodes.
7. **Side results:** [05](05-rem-learnable.md) differentiable (ε,b) ·
   [08](08-rem-complex-sketch.md) complex sketches ·
   [36](36-prop-imq-findiff.md) IMQ by finite differences ·
   [37](37-prop-importance.md) importance-sampling window.

## Result index

| Doc | Label | Result |
|---|---|---|
| [01](01-prop-nonstationary.md) | `prop:nonstationary` | Off-sphere: neither stationary nor dot-product |
| [02](02-prop-biased-feature.md) | `prop:biased_feature` | Exact biased degree-2 feature, ‖p_b(x)‖²=(‖x‖²+b)² |
| [03](03-def-ryf.md) | `def:ryf` | Exact-modulation ⵟ-feature map (analyzable limit) |
| [04](04-thm-unbiased.md) | `thm:unbiased` | E[z(x)ᵀz(w)] = k_ⵟ,b(x,w) |
| [05](05-rem-learnable.md) | `rmk:learnable` | Reparameterized differentiability in (ε,b) |
| [06](06-thm-bernstein-schur.md) | `thm:bernstein_schur` | Class estimator: unbiased + variance + uniform bound |
| [07](07-ex-ray-coordinate.md) | `ex:ray` | One RAY coordinate end to end |
| [08](08-rem-complex-sketch.md) | `rmk:complex` | Complex unit-phase sketch halves sketch variance |
| [09](09-thm-exact-variance.md) | `thm:exact_variance` | Exact flat-estimator variance (bracket ≤ 3/2) |
| [10](10-thm-bernstein.md) | `thm:bernstein` | Expected matrix-Bernstein op-norm bound |
| [11](11-cor-bernstein-tail.md) | `cor:bernstein_tail` | High-probability op-norm tail |
| [12](12-thm-ts-opnorm.md) | `thm:ts_opnorm` | Deployed RAY: radial bound + additive sketch term |
| [13](13-rem-ose.md) | `rmk:ose` | Achieving the sketch spectral event (ridge OSE) |
| [14](14-prop-ridge-sketch.md) | `prop:ridge_sketch` | Ridge sandwich transfers P → K through Schur product |
| [15](15-thm-krr-spectral.md) | `thm:krr_spectral` | Deterministic KRR stability under ρ-sandwich |
| [16](16-thm-krr-whitened.md) | `thm:krr_whitened` | Whitened Bernstein; d_eff = exact intrinsic dim |
| [17](17-cor-krr-highprob.md) | `cor:krr_highprob` | High-prob KRR coefficients + objective sandwich |
| [18](18-thm-krr-leverage.md) | `thm:krr_leverage` | Leverage tilt: count (1+‖P‖/λ) → (1+d_eff) |
| [19](19-rem-risk.md) | `rmk:risk` | Spectral ≠ risk; the two missing ingredients |
| [20](20-cor-krr-deployed.md) | `cor:krr_deployed` | Deployed KRR condition, scale-free η ≤ ρ₀/4 |
| [21](21-thm-class-bernstein.md) | `thm:class_bernstein` | Full matrix-level set for the whole class |
| [22](22-prop-gate.md) | `prop:gate` | Modulation–radial error decomposition (gating) |
| [23](23-prop-quadrature.md) | `prop:quadrature` | Positive-weight radial nodes, log dynamic range |
| [24](24-prop-positive.md) | `prop:positive` | FAVOR⁺ product feature ⇒ nonnegative attention |
| [25](25-prop-pos-dichotomy.md) | `prop:pos_dichotomy` | Infinite variance for 8xᵀw ≥ ε; trig ≤ 3/2 |
| [26](26-thm-variance.md) | `thm:variance` | (R²+b)⁴/(Dε²) envelope, all (D,D′) |
| [27](27-prop-variance-sharp.md) | `prop:variance_sharp` | Prefactor attained with equality in the family |
| [28](28-prop-budget.md) | `prop:budget` | Two-level identity; D′=1 optimal |
| [29](29-thm-uniform.md) | `thm:uniform` | Entrywise Hoeffding + union bound |
| [30](30-cor-sample-complexity.md) | `cor:sample_complexity` | D = O((R²+b)⁴ε⁻²τ⁻² log N), d-free |
| [31](31-cor-gram-concentration.md) | `thm:gram_concentration` | Crude N·max op-norm route (what Bernstein removes) |
| [32](32-lem-schur.md) | `lem:schur` | Schur multiplier bound, PSD and symmetric cases |
| [33](33-prop-normalized.md) | `prop:normalized` | Unit-norm modulation: variance ≤ 3/(2Dε²) |
| [34](34-prop-ts-variance.md) | `prop:ts_variance` | Three-term radial/sketch/interaction split |
| [35](35-prop-optimal-m.md) | `prop:optimal_m` | m\* = √(BM/A) at fixed feature budget |
| [36](36-prop-imq-findiff.md) | `prop:imq_findiff` | IMQ = forward second difference in b, exact |
| [37](37-prop-importance.md) | `prop:importance` | Exp(ε+η) proposal finite-variance iff η < ε+2r |

Experiment-to-result mapping lives in the paper's [README](../README.md); per-table
protocol in `main.tex` Appendix `app:exp_details`.

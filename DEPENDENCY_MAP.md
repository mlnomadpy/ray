# Dependency map — Bernstein–Schur Kernels (the RAY paper)

Updated 2026-06-04 (post R3–R11). Theorem-level dependency graph for the math currently
in `main.tex`. `[ext]` = imported from Bouhsine (2026a) `bouhsine2026action`, proved there.
Method is **RAY** (random ⵟ-feature). On-sphere dot-product material lives in the
**separate** paper `spherical_yat_features/` and is NOT in this graph.

## 1. Result dependency graph

Arrows = "is used by / depends on".

```
k_{ⵟ,b}=(wᵀx+b)²/(‖w-x‖²+ε)   ── the object (b≥0, ε>0)
   │
   ├─► prop:nonstationary  (off-sphere: not shift-inv, not dot-product)   [elementary witnesses]
   │        └─► motivates the off-sphere experiment (sec:exp_offsphere) + necessity demo
   │
   ├─► eq:schur  (Schur factorization k_{ⵟ,b}=p_b·h_ε)                     [elementary]
   │       │
   │       ├─► prop:biased_feature (exact poly feature p_b, dim d_b=d(d+1)/2+d+1)  [b≥0 ⇒ PSD]
   │       │        └─► def RAY map, thm:unbiased, App Kronecker, prop:normalized
   │       │
   │       └─► eq:bernstein (Bernstein–Widder Laplace rep of h_ε)          [ext: complete monotonicity]
   │                └─► radial-scale mixture = IMQ spectral sampler (sec:step-rff, the flat D'=1 identity)
   │                         └─► def RAY map  ◄── RFF identity [ext: Rahimi–Recht]
   │
   ▼
thm:unbiased  (needs prop:biased_feature + eq:bernstein + RFF identity)
   │
   ├─────────────────────────► thm:bernstein_schur  (GENERAL: any u(x)ᵀu(w)·f(‖x-w‖²),
   │                              f completely monotone; unbiasedness + variance 3m²B⁴/2D
   │                              + uniform mB²√(8log(2N²/δ)/D)). ⵟ-kernel = flagship (m=1, B²=R²+b).
   │                              ↑ subsumes thm:variance, thm:uniform for the whole class
   │
   ├──── shared per-scale term Y_j (a.s. |Y_j|≤2(R²+b)²/ε ; 2nd moment ≤3/2) ────┐
   │   used by ↓                ↓                    ↓                            │
 thm:variance            thm:uniform          thm:bernstein  ◄── lem:schur        │
 (envelope, C–S +        (Hoeffding+union)    (matrix Bernstein, intrinsic-dim    │
  Gauss 2nd moment)           │                ‖P‖,‖K‖ not N·max; Tropp 2015)     │
   │                          ▼                     │                             │
   ▼                  cor:sample_complexity         ▼                             │
prop:budget  ─────►  thm:exact_variance        thm:krr_spectral                  │
(two-level identity,  (EXACT flat var:          (relative spectral: ‖A^{-½}EA^{-½}‖≤ρ │
 D'=1 optimal)         a²/Dε²·[1+½ε/(ε+4r)        ⇒ ‖α̃-α‖_A ≤ ρ/(1-ρ)‖α‖_A;        │
   │                   -(ε/(ε+r))²])              ρ=O(D^{-½}) via whitened Bernstein) │
   │                        │                                                      │
   │                        ├─► prop:normalized (q_b=p_b/(‖x‖²+b); var ≤ 3/(2Dε²), no R,b) │
   │                        └─► bias_scaling validates (R²+b)⁴ law                  │
   └──────────────────────────────────────────────────────────────────────────────┘

TensorSketch-RAY (sec:exp_ts):  prop:ts_variance  (Var = p²Var[ĥ_D] + h²Var[p̂_m] + product;
   radial MC ~D^{-½}  ⊕  poly sketch ~m^{-½})  ◄── independence of radial & sketch randomness
   (separate estimator: unbiased only over sketch randomness, NOT the exact-modulation theorems)

variance reduction (App D): QMC[Koksma–Hlawka], ORF[Yu], importance sampling ◄── eq:bernstein
```

## 2. Theorem → experiment validation map

| Experiment (script)                       | Validates                                            |
|-------------------------------------------|------------------------------------------------------|
| gram_approx.py (§4.1, Fig 1)              | thm:uniform, O(1/√D) rate; thm:bernstein (op-norm)   |
| variance_validation.py (§4.2)             | thm:variance + QMC/ORF reduction                     |
| bias_scaling.py (§4.3)                    | thm:variance / thm:exact_variance — (R²+b)⁴ (exp 4.01)|
| dimension_free.py (§4.4, Fig 1b)          | cor:sample_complexity                                |
| budget_allocation.py (§4.5)               | prop:budget (flat D'=1 optimal)                      |
| timing_scaling.py (§4.6)                  | scalability (linear-in-N representation)            |
| krr_downstream.py (§4.7)                  | thm:krr_spectral (downstream KRR)                   |
| **off_sphere_gram.py (§4.8)**             | **prop:nonstationary** — general-R^d validity; vs uniform+k-means Nyström |
| ts_ryf_costmatched.py (§4.9)              | prop:ts_variance — TensorSketch-RAY at matched dim   |
| fair_cost.py (§4.10)                      | matched memory/wall-clock                            |
| necessity_demo.py (§4.11)                 | prop:nonstationary motivation — coupling necessity (non-circular target) |
| **exact_variance_check.py (§4.3)**        | **thm:exact_variance** — empirical/predicted ratio 1.001 |
| **normalized_ray.py (§4.3b)**             | **prop:normalized** — Var ratio = (R²+b)⁴ exact, normalized bounded |
| **ts_decomposition.py (§4.9)**            | **prop:ts_variance** — 3-term Var, radial 1/D + sketch 1/m separate |
| **krr_spectral.py (§4.7)**                | **thm:krr_spectral** — ρ→0, bound holds at ρ<1 |
| **signal_gate_snr.py (§4.12)**            | **prop:gate** — product gates both distractor types (AUC) |
| **leverage_nystrom.py (§4.8)**            | stronger Nyström baseline (leverage≈uniform, RAY wins) |
| **highd_offsphere.py (§4.8)**             | scale — O(1/√D) at d=128/256/512 |

Coverage: every main-text theorem/prop has a backing experiment. The general
thm:bernstein_schur is validated through its ⵟ-kernel instance (all of the above).

## 3. New since R1 (what changed in the math)

- **prop:nonstationary** (R11 #6): formalizes "off-sphere neither stationary nor dot-product".
- **thm:bernstein_schur** (R11 #2): general class promoted Remark→**main Theorem** w/ guarantees → earns the title.
- **thm:exact_variance** (R11 #1): EXACT flat-estimator variance, supersedes the loose envelope.
- **thm:bernstein** + **lem:schur** (R7/R8): matrix-Bernstein op-norm (intrinsic-dim), with a proved Schur-multiplier lemma.
- **thm:krr_spectral** (R11 #5): relative-spectral KRR, supersedes the weak λ⁻² coefficient corollary.
- **prop:normalized** (R11 #7): bounded-variance normalized RAY (kills (R²+b)⁴ blow-up).
- **prop:ts_variance** (R11 #3 / R10 g4): TensorSketch-RAY error decomposition.
- TensorSketch §4.9, off-sphere §4.8, fair-cost §4.10, necessity §4.11 added; method renamed RYF→RAY; title → "Bernstein–Schur Kernels".

## 4. Still open / flagged (not yet in the math)

- Matrix-level TensorSketch op-norm prop (R11 #4) — could add to App.
- Importance-sampling 2nd-moment condition η<ε+2r (R11 #8) — fixes App D claim; not yet added.
- Fair-cost m*≈√M optimization remark (R11 #10) — approach paragraph; not yet added.
- Full excess-risk (Rudi–Rosasco) bound — left as future work (thm:krr_spectral is the partial step).
- Real large-scale off-sphere dataset; stronger leverage-score Nyström — experiments, not math.
- A non-ⵟ Bernstein–Schur experiment (e.g. (xᵀw+b)·Matérn) to exercise thm:bernstein_schur empirically.

## 5. Update 2026-06-10 (leverage + dichotomy batch)

New math:
- **thm:krr_leverage** — whitened-leverage tilted radial sampling. Leverage
  d̄_λ(θ)=ψᵀ(A⁻¹∘P)ψ with E_π[d̄]=d_eff(λ) **exactly**; tilt π* ∝ d̄·π, weight d_eff/d̄.
  Count (1+‖P‖/λ) → (1+d_eff(λ)); same Tropp machinery, the variance majorant core
  A^{-1/2}KA^{-1/2} is unchanged so intdim = d̃_λ exactly. Closes rmk:risk(i).
  Depends on: thm:krr_whitened ingredients + lem:schur + E_π[ψψᵀ]=R. Class-wide via
  thm:class_bernstein (only needs R_u unit-diagonal). Validated: leverage_radial_sampling.py
  (uniform D* 50→800→>3200 as λ 10→1→0.1; leverage 50/200/400/1600 tracking d_eff).
- **prop:pos_dichotomy** — FAVOR⁺ ∘ Exp(ε) second moment = ε/(ε−8xᵀw), **infinite** for
  8xᵀw ≥ ε; truncation caps at e^{8T_max xᵀw} (still exponential). Trig bracket ≤ 3/2
  uniformly. Scopes prop:positive to the diffuse regime; corrects the intro "no reduction"
  claim (per-scale positive factorization exists off-sphere but is not variance-viable when
  aligned). Validated: positive_features.py.

Empirical upgrades:
- rmk:complex now measured (complex_sketch.py): η and sketch term ~1.6× lower, variance halved.
- Kernel-grammar customer: LIN²×RQ_α IS Bernstein–Schur (Γ(α,2ασ²) mixing, m_f=1);
  california end-to-end (grammar_kernel.py) — cite duvenaud2013structure.
- prop:quadrature scoped by radial_quadrature.py (exact-g: machine precision by D=32;
  RFF-paired: inner noise dominates, naive nodes worse than MC).
- tab:higgs upgraded to 3 seeds (higgs_scaling.py --seed).
- m→m_f notation fixed in the thm:bernstein_schur appendix proof.

Now open (discussion §6 renumbered):
1. deployable approximation of the leverage tilt + d_eff(K*_D) control ⇒ minimax risk;
   matching Ω(‖P‖/λ) lower bound for uniform sampling.
2. is the (R²+b)⁴ prefactor optimal over ALL unbiased data-independent feature maps
   (prop:variance_sharp only pins the factored family)? conjectured yes.
3. finite-variance estimator for the peaked regime (trig: sharpness blow-up; positive:
   infinite variance — prop:pos_dichotomy).
4. thin-shell/manifold variance tightening (d_int empirically tiny).

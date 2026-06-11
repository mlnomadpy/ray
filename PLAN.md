# Project Plan — Random Features for the Biased ⵟ-Kernel

Living task board for `papers/01_theory/biased_random_features/`.
Last updated 2026-06-04. Companion to `DEPENDENCY_MAP.md` (theorem-level deps).

Legend: `[x]` done · `[~]` in progress · `[ ]` todo · **P0** must-have · **P1** strong · **P2** nice.

---

## EXPERIMENT BOARD (2026-06-04, post-audit — "validate the new theorem package")

Driver: the theory outran validation. The R11/R12 theorem package was added faster than
experiments. This board closes every theorem→experiment gap + the recurring scale/baseline
asks. **All 9 experiments below are RUN (results archived); remaining work is INTEGRATION
into the papers (tracked as harness tasks #1–#6).**

### RAY paper — experiments (all RUN ✅, integration pending)
- [x] `exact_variance_check.py` — validates **thm:exact_variance**. Empirical/predicted ratio **1.001±0.002** across b∈{0,1,2}, r∈[0.4,2.2]; unbiased.
- [x] `normalized_ray.py` — validates **prop:normalized**. Var(exact)/Var(norm) = **(R²+b)⁴ exactly** (1,5.06,16,81,625); normalized variance bounded in [0.045,0.53]≤3/2. *(highest-value: untested claim + stability story)*
- [x] `ts_decomposition.py` — validates **prop:ts_variance**. 3-term Var formula ratio **~1.0**; radial source ~1/D (Var_h 0.09→0.0008), sketch source ~1/m (Var_p 0.004→0.0001) separate cleanly.
- [x] `krr_spectral.py` — validates **thm:krr_spectral**. ρ→0 (slope −0.83); bound ρ/(1−ρ) holds once ρ<1 (D=1024: rel-err 0.18 ≤ 2.18). Honest: ρ>1 at small D for λ=0.1.
- [x] `leverage_nystrom.py` — stronger baseline (R3/R5/R10/R13). leverage≈uniform, k-means best Nyström, **RAY@1000 still wins by d=16** (0.051 vs 0.079).
- [x] `highd_offsphere.py` — scale (R6/R9/R10/R13). **O(1/√D) holds at d=128/256/512** (slopes −0.41/−0.51/−0.52); D=1000 error 0.05–0.064 across d.
- [x] `signal_gate_snr.py` — R16 idea check (retrieval/AUC). **yat separates true from BOTH distractors (AUC 1.0/1.0)**; IMQ fails radial (0.22), poly fails align (0.0); RAY inherits, TS-RAY mostly (0.73 on align). Validates the gate story in cross-pair discrimination (honest: absolute not relative).
- [x] `bernstein_schur_demo.py` — **thm:bernstein_schur** on a NON-yat instance (degree-3 × gen-IMQ): unbiased (0.023→0), O(1/√D) (slope −0.48). *(already integrated, App D)*

### Spherical paper — experiments
- [x] `random_maclaurin_runtime.py` — R9 #2. **E[N] grows 2→28 as ε:2→0.05**, p99 7→78; the "M free of d²" claim trades a polynomial dim for a degree-sampling cost O(M·E[N]·d), heavy-tailed as ε→0.

### Integration tasks (harness #1–#6)
- [ ] #1 exact-variance + normalized-RAY → §4 + tables
- [ ] #2 ts-decomposition + krr-spectral → §4.9 / guarantees tables
- [ ] #3 leverage-Nyström + high-d → off-sphere/fair-cost/dimension tables
- [ ] #4 R16 error-decomposition prop + gate remark + signal_gate_snr (NO anchor, NO full reframe)
- [ ] #5 Random-Maclaurin runtime → spherical paper
- [ ] #6 reproducibility headers + build both + DEPENDENCY_MAP + commit

### R16 (signal gate) verdict
ADD: matrix-level 3-term error-decomposition prop (generalizes prop:ts_variance). ADD as
remark: gate suppresses ABSOLUTE radial noise (P∘E_H), WITH caveat that relative per-pair
error is unchanged (P cancels). SKIP: Anchor-RAY (user), whole-paper reframe (overclaim).

### Still NOT done (lower priority)
- One larger real off-sphere dataset (KRR+necessity at scale).
- Off-sphere promoted to main Figure 1 (presentation, R13).
- Matrix-Bernstein intrinsic-dim *scaling* demo (opnorm_validation has the rate only).

---

## DATASET SCALING SUITE (2026-06-04, R19 + "add all datasets")

HIGGS done (✅ §4.13 `sec:exp_higgs`, tab:higgs): 11M streaming primal, memory-flat 8.5GB, TS-RAY > exact RAY
at matched M, Gaussian leads (no coupling). HIGGS is only d=28 (d_b=435) so it does NOT stress the d_b=O(d²)
floor — exact RAY runs fine there, so TS-RAY only looks optional. The suite below fixes that: it adds datasets
across two axes so the paper's two large-scale claims each get the case that actually forces them.

Two reviewer angles, reconciled into two tracks:
- **Floor/scalability axis (mine):** high-d forces d_b=O(d²) → exact RAY impossible → TS-RAY is *necessary*;
  large-N forces the memory-flat streaming primal. Datasets prove TS-RAY *enables* what exact RAY can't.
- **Coupling axis (R20 advisor):** pick datasets whose target = local proximity × directional alignment, where
  RAY should actually *win* (HIGGS lacked it, Gaussian led). Physics signal/background + frozen vision embeddings.

**Harnesses:**
- `experiments/scaling_suite.py` — reuses HIGGS feature builders + streaming-Adam trainer verbatim; dataset
  registry + LIBSVM/CSV/UCI loaders + auto-download + a `--ray-cap` guard marking exact RAY N/A when d_b too large.
  HIGGS `make_params` patched to build the triu index lazily (ray-only) so high-d sets don't choke.
- `experiments/gate_diagnostic.py` — **the cheap pre-flight (run FIRST).** On a 50k subsample, scores pairs by
  radial / alignment / yat / yat-normalized and reports pair-AUC + precision@10. If yat beats BOTH single factors →
  coupling present → train RAY. If radial alone wins (HIGGS-like) → Gaussian will lead → skip the big run. Real-data
  analogue of `signal_gate_snr.py`; also accepts an embeddings `.npz` (X,y) for the vision track.

| dataset | N_train | N_test | d | d_b | axis | exact RAY |
|---|---|---|---|---|---|---|
| higgs (done) | 10.5M | 500k | 28 | 435 | large N, no coupling | yes |
| **susy** | 4.5M | 500k | 18 | 190 | large-N + coupling? (physics) | yes |
| **hepmass** | 7.0M | 3.5M | 27 | 406 | largest-N + coupling? (physics) | yes |
| **miniboone** | 104k | 26k | 50 | 1,326 | fast physics coupling probe | yes |
| **covtype.binary** | ~522k | ~58k | 54 | 1,540 | mid-N / mid-d | yes |
| **a9a (adult)** | 32.5k | 16.3k | 123 | 7,626 | standard tabular sanity | yes (heavy) |
| **madelon** | 2.0k | 1.8k | 500 | 125,751 | nonlinear XOR / coupling proxy | **N/A → TS-RAY only** |
| **epsilon** | 400k | 100k | 2000 | 2,003,000 | high-d **and** large-N | **N/A → TS-RAY only** |
| **gisette** | 6.0k | 1.0k | 5000 | 12,512,500 | extreme high-d, pixel-products | **N/A → TS-RAY only** |

**Coupling-win track (R20, separate harness — the most-likely-to-win experiment):** frozen vision embeddings, NOT
raw pixels. CIFAR-10/100 (60k) then ImageNet (1.2M) via CLIP / DINOv2 / ResNet penultimate. Class identity is
angular (alignment) while local neighborhoods are proximity → exactly alignment-gate × local-proximity. Plan:
extract embeddings → `.npz` → `gate_diagnostic.py --npz` (cheap check) → if coupling, `scaling_suite`-style
AUC/acc vs M with a close-but-wrong-class false-positive analysis. NOT built yet (needs CLIP/DINOv2 extraction;
confirm before pulling models). This is where the signal-gate story has the best shot on real data.

**Why each earns its place (no table-padding; R19 warned against it):**
- **gisette / epsilon** are load-bearing: at d=2000–5000 the exact polynomial feature (d_b≈2M–12.5M floats *per
  point*) cannot be built, so ONLY TS-RAY runs — converts the floor (Limitation v) from "expensive" to "exact RAY
  impossible; compression is the enabling step." HIGGS (d=28) cannot show this.
- **susy / hepmass / miniboone** are the coupling bets: physics signal/background may need kinematic proximity ×
  alignment with derived-physics directions. Gate-diagnostic first; train only those that pass.
- **gisette / madelon** also carry real degree-2 interaction structure (pixel products; hypercube-XOR, 96% probes)
  → real-data proxies for the coupling/signal-gate story (Sec 4.11–4.12), otherwise only synthetic d=16.
- **covtype / a9a** = large-N companion + tabular sanity; a9a's d_b=7626 is the borderline where exact RAY is
  buildable but already starved at matched M — cleanest single-dataset picture of the floor biting.

**Run order (gate the spend):** (0) `gate_diagnostic.py` on susy/hepmass/miniboone/madelon/gisette — minutes, no big
download for the small ones. (1) P0 gisette+madelon (floor + coupling). (2) P1 whichever physics sets pass the gate
(susy/hepmass/miniboone). (3) P1 epsilon (high-d at scale; ~12GB download). (4) coupling-win: CIFAR embeddings if a
physics/embedding gate-check is positive. Paper table LEADS with gisette/epsilon (exact RAY = "—, impossible"),
keeps susy/covtype/a9a as a compact breadth row, and reports any coupling win prominently.

### RESULTS (2026-06-04, ran the pre-flight + floor sets)

**Gate diagnostic (pair-AUC, 50k subsample) — NO real coupling anywhere:**
| dataset | radial | alignment | yat | verdict |
|---|---|---|---|---|
| madelon | 0.505 | 0.506 | 0.508 | chance (96% probes, no raw structure) |
| covtype | 0.508 | 0.510 | 0.509 | chance |
| susy | 0.558 | **0.580** | 0.562 | alignment leads |
| miniboone | 0.588 | **0.642** | 0.602 | alignment leads (−999 sentinel imputed) |
| hepmass | 0.607 | **0.656** | 0.633 | alignment leads |
| gisette | 0.528 | **0.622** | 0.538 | alignment leads (pixel-product structure) |

Wherever real tabular/physics data has structure it is **pure alignment**, never alignment×proximity coupling.
`yat` never beats its own alignment factor. CONSEQUENCE: the ⵟ-coupling claim stays SYNTHETIC (Sec 4.11);
do not chase a coupling win on tabular data; Gaussian/poly will lead. CIFAR/ImageNet embeddings remain the only
untested regime with plausible coupling → the coupling-win track is the highest-value remaining experiment.

**Floor-breaker training (gisette/madelon) — HONEST NEGATIVES, not paper-worthy:**
- gisette d=5000: linear=**0.997** (linearly separable!) → kernels are the wrong tool, Gaussian AND TS-RAY both
  ~chance (0.5–0.6). TS-RAY *runs* where exact RAY can't, but at chance accuracy that is not a "useful" result.
  Degenerate probe features also blow up under standardization (logit saturation).
- madelon d=500: all methods weak (~0.55–0.61), TS-RAY ≈ Gaussian. No story.
- Surfaced + REVERTED a tempting "augmented-sketch" TS-RAY (sketch x'=(x,√b) → dim m instead of m+d+1): it lowers
  the floor m+d→m but SKETCHES the bias terms, breaking prop:ts_variance's clean "bias terms exact" decomposition.
  Its only benefit shows at d≫m (gisette), which is a negative anyway. Kept the paper's single m+d+1 definition.

**Net for the paper:** HIGGS large-scale section (committed 7f6f5a2) stands. NO new dataset earns inclusion — gate
says no coupling, floor datasets are negatives. The floor story stays as the existing representation-cost argument
(§4.9 O(d²) floor + Limitation v). Real limitation worth one honest sentence: at extreme d even TS-RAY's m+d+1 floor
starves the radial draws, and the high-d regime is better served by the on-sphere dot-product route (spherical paper).

- [x] #11 gate_diagnostic (6 datasets) → results/gate_diagnostic_{small,physics}.json. Finding: no real coupling.
- [x] #12 gisette+madelon floor → results/scaling_suite_floor.json. Negatives (linearly separable / structureless).
- [~] #13 physics sets — gate says SKIP for coupling (Gaussian leads); optional susy/hepmass as large-N breadth only.
- [ ] #14 epsilon (d=2000, dense pre-normalized, N=400k) — the ONE high-d set where kernels can help + exact RAY truly
      impossible. 12GB download, uncertain payoff. The only floor dataset that might yield a real win.
- [x] #15 CIFAR frozen-embedding coupling track — DONE, NEGATIVE. `cifar_embed.py` (CLIP ViT-B/32, MPS, projected
      512-d) → `gate_diagnostic.py --npz`. Standardized pair-AUC:
        cifar10  rad=0.862 **al=0.919** yat=0.917 yatN=0.885
        cifar100 rad=0.843 **al=0.904** yat=0.898 yatN=0.862
      Alignment dominates (CLIP's cosine training); yat ≈ alignment, NEVER beats it. Raw (uncentered) embeddings
      flip to radial-dominated (rad 0.861) — either way one factor dominates, no coupling. Advisor's top bet fails.
- [ ] #16 integrate ONLY if a clean result appears; none has → HIGGS stands alone as the large-scale section.

### COUPLING VERDICT (2026-06-05, definitive)
Tested the alignment×proximity coupling on EVERY accessible real regime: tabular (madelon/covtype/a9a), physics
(susy/hepmass/miniboone), and frozen vision embeddings (CIFAR-10/100 CLIP). **It is absent everywhere.** Wherever
real data has pair structure, a SINGLE factor carries it — alignment after centering (susy 0.58, hepmass 0.66,
gisette 0.62, cifar 0.90+), radial when uncentered/proximity-driven — and the ⵟ product `yat` tracks that factor
without beating it. The coupling regime where the product is *necessary* is, empirically, an ENGINEERED one
(the synthetic necessity target tanh(u·x)·exp(−‖x−v‖), Sec 4.11). This is a clean, honest scoping result:
it says precisely when ⵟ helps (jointly-coded targets) and when it does not (real data dominated by one geometry).
RECOMMENDATION: stop the coupling hunt; keep the synthetic necessity demo as the coupling evidence; optionally add
one diagnostic-backed sentence to Sec 4.11 scoping the regime (preempts the "why only synthetic?" reviewer question).

### RESOLUTION (2026-06-05): exact kernel is NO-REGRET; RF needs more draws at high d
Pushed past the gate diagnostic into actual classification on CIFAR CLIP embeddings (cifar_classify.py,
cifar_kernel_krr.py, cifar_rf_fair.py). Two-level finding, both now reflected honestly in the paper:
- **EXACT kernel is no-regret** (tab:cifar added to sec:exp_necessity): exact KRR on CLIP embeddings under
  centered (alignment-dominated) AND bounded-ball (proximity-dominated) preprocessing — yat ties the best
  baseline within 0.5% in BOTH regimes, never below Gaussian (C10 0.940 both; C100 0.759/0.763). Confirms the
  user's point: ONE yat-kernel tracks whichever geometry carries the labels; Gaussian trails on alignment data.
- **RF approximation yat-RAY LOSES at matched M on alignment-dominated CIFAR** (d=512), even with fair tuning
  (per-feature znorm + per-method λ-sweep + converged closed-form): C10@4096 tsray 0.933 < gauss/poly 0.945.
  NOT a head-training deficiency (closed-form is converged) — STRUCTURAL: each radial draw costs O(m+d)≈d at
  d=512 so matched-M starves draws (D=1 at M≤1024), and the radial spend is wasted when alignment alone suffices.
  yat-RAY climbs toward its no-regret kernel as M grows (0.869→0.933, still rising; exact ceiling 0.9405) — it
  needs MORE draws than a single-factor RFF. This is Limitation (v) (O(d^2) floor) witnessed on real high-d data;
  added as a one-clause caveat to the tab:cifar paragraph. NO RF-win claim on CIFAR (there isn't one).
- High-M closed-form confirmation (M=32768) OOM-crashed (stores 3×45k×32768 float32 feature matrices); not needed,
  trend already clear. If ever wanted, stream the Gram (O(M^2) not O(N·M)) or cap M≤16384.
Paper builds 24pp. Scripts archived. The paper is mostly THEORETICAL — kept this to one paragraph + tab:cifar.

---

## Review batch R8 (2026-06-04, hardening pass — "promising and close, submit after one more pass")
Positive verdict; raised the bar by inspecting the new theorem + TensorSketch.
DONE: (1) matrix-Bernstein OVERCLAIM fixed — abstract/contribution no longer say "rather than N" (magnitude carries ‖P‖,‖K‖ which can scale w/ N); now "controlled by top eigenvalues + intrinsic dim, replacing the crude N·max entrywise route"; added + PROVED the Schur-multiplier Lemma lem:schur (A⪰0,P⪰0,max A_ii≤a ⟹ A∘P⪯a‖P‖I; proof via A=UU^T, A∘P=ΣD_rPD_r, ΣD_r²=diag(A)⪯aI). (2) "finite-rank"→"finite-feature" everywhere (+abstract parenthetical "explicit finite-dim feature map"). (3) TensorSketch CLARIFIED: exact RYF unbiased conditional on deterministic poly map (the exact-modulation thms); TS-RYF unbiased only over sketch randomness (one shared sketch), 2nd approx error, matrix-Bernstein governs radial not sketch — "practical variant, not main theorem". (4) "recovers dimension efficiency"→"recovers MUCH of the lost". (5) Bernstein–Schur table MOVED to main text + enriched cols (modulation/radial/mixing ν/feat-dim/bias). (6) necessity demo DE-CIRCULARIZED (was yat-shaped coupled target → "of course yat wins"): coupled now tanh(2u·x)·exp(−‖x−v‖) matching NO kernel; controls kept kernel-natural. Rerun: coupled yat 0.087 best (vs Gaussian 0.098/IMQ 0.099/poly 0.350); prox IMQ 0.141; align poly 0.001 — clean 3-way AND non-circular. (7) flat-sampling sharper abstract line. Build 18pp clean.
REMAINING (flagged, bigger lifts / stylistic): abstract still dense (trim); leverage-score/recursive Nyström baseline; a REAL (not synthetic) off-sphere task; title alternatives (user chose Bernstein–Schur title, keep).

## CURRENT STATE (2026-06-04, post R3–R7)

**Venue:** TMLR (style adopted: `tmlr.sty`/`.bst`, `[preprint]`, `\name/\email/\addr`; free `\AND` before `algorithmic`). Paper builds 16 pp clean (XeLaTeX).
**Committed** on `mlnomadpy/monterrey-v2`: bd99cfd (R3), 39a7219 (TMLR+off-sphere+R4/R5), cff55ce (standalone spherical paper), d2e09bf (R6 copy-edit), 3068624 (TensorSketch), 1ef75fa (drop RandMac from RYF).

**PAPER SPLIT (decisive):** the on-sphere dot-product story (Random Maclaurin, exact-numerator hybrid, cost-matched/regime map, Funk–Hecke, Gauss–Laguerre) lives in a **separate standalone paper** `papers/01_theory/spherical_yat_features/`. The RYF paper does NOT cite it (on-sphere route = "future work"); Random Maclaurin must NOT reappear in the RYF paper.

**R3–R6 done (in RYF paper):** numerator-not-denominator nonstationarity; QMC-unbiasedness caveat; stale D'=50 removed; **off-sphere bounded-ball experiment §4.8** (off_sphere_gram.py, the load-bearing general-R^d evidence: RYF O(1/√D) at all d, uniform Nyström degrades); conclusion reworded (no "joins families"); Bernstein–Schur promoted to main-text Remark; abstract softened ("uniform-landmark Nyström", "Bochner does not apply to the full kernel", bounded-norm not sphere); **TensorSketch-RYF §4.9 + tab:ts** (closes the hidden-O(d²) attack: digits d=64 M=d_b, exact RYF 0.928→TS-RYF m=128 0.977≈oracle 0.986).

**Reframed north-star (R7):** "Some useful nonstationary kernels are products of a finite-rank alignment factor and a completely monotone radial factor — they fall between the Bochner and dot-product templates. For this **Bernstein–Schur** class, exact modulation + radial randomization gives unbiased, analyzable, low-draw approximations; the biased ⵟ-kernel is the first useful instance." (NOT "RYF beats RFF".)

---

## Review batch R7 (2026-06-04, positioning/depth — strongest review)  (T-R7*)  ✅ ALL DONE
No bugs; all positioning + depth. Core: the construction risks reading as "tensor two known feature maps." Fix = reframe + add depth.
TMLR-sufficient = R7-1,2,3,4,10,11. NeurIPS-grade = R7-5,6,7 (+8).
DONE: R7-1 reframe (abstract+intro: "factorization unlocks standard tools", contribution = identify/analyze Bernstein–Schur class), R7-2 examples table (tab:bernstein_schur), R7-3 family-over-b (abstract+intro: {k_{ⵟ,b}} spans IMQ, not fixed b), R7-4 dimension-free soften, R7-5 k-means Nyström (off_sphere_gram.py + fair_cost.py; stronger than uniform but still degrades w/ d, RYF beats by d=8), R7-6 matrix-Bernstein op-norm thm:bernstein (E‖K_D−K‖≤2√(‖P‖‖K‖log(2d_int)/D)+‖P‖log/D, intrinsic-dim not N; full proof via PSD-square domination + Schur-Hadamard + Tropp2015), R7-7 necessity_demo.py (yat wins coupled target ONLY: 0.163 vs Gaussian 0.178/IMQ 0.252/poly 0.337; IMQ wins prox, poly wins align → coupling attributable), R7-8 fair_cost.py (matched-mem digits: RYF-exact 0.928/20.6MB/0.02s, TS-RYF 0.977/20.3MB/0.009s fastest, kmeans-Nys 0.986/11.5MB/1.18s), R7-9 "three facts taken from bouhsine2026action" stmt, R7-10 already-correct (sampler not no-spectral-route, line 157), R7-11 symmetric d_b=d(d+1)/2+d+1 default. Build 18pp clean.
- [ ] **P0 · T-R7-1 REFRAME** (the central fix). Reposition: "factorization unlocks standard tools" not "standard tools can't touch this." Contribution = identifying/analyzing the Bernstein–Schur class, biased ⵟ as motivating instance. Soften intro "neither Bochner nor polynomial sketching applies" → "...do not apply to the full kernel directly; the Schur factorization exposes a polynomial piece (exact feature) and a completely monotone radial piece (mixture-RFF)." Touch abstract + intro + §3.2.
- [ ] **P0 · T-R7-3 family-over-b** (real precision bug). IMQ comes from finite differences in b → it's the FAMILY {k_{ⵟ,b}} that spans IMQ, not a single fixed b. Reword abstract+intro: "the family {k_{ⵟ,b}: b≥0} spans the IMQ factor by finite differences in b \citep{bouhsine2026action}; we fix b and study scalable random features for that member." Avoid implying a single fixed b contains IMQ.
- [ ] **P1 · T-R7-2 Bernstein–Schur depth.** Add a table of nontrivial instances (degree-q poly × Matérn/Laplace mixture; linear × CM radial; finite NTK block × IMQ; degree-2 biased poly × IMQ = ⵟ_b) to App C / the Remark. Converts the closure property into a substantive contribution.
- [ ] **P1 · T-R7-4 dimension-free abstract soften.** → "For fixed finite datasets the Monte-Carlo draw count obeys the same no-explicit-d dataset-level behavior as RFF for stationary kernels, while the total representation still scales through the polynomial feature dimension D·d_b." Make Nyström contrast empirical, not theoretical-war.
- [ ] **P1 · T-R7-5 stronger Nyström** (recurring R3/R5/R6/R7). Add k-means Nyström (min) + leverage-score/recursive if feasible to gram_approx + KRR; report. The "RYF degrades less with d than Nyström" contrast needs a non-weak baseline.
- [ ] **P1 · T-R7-6 matrix Bernstein** (highest-value theory upgrade → "TMLR theory"). Replace the crude entrywise-Hoeffding+union (N·max) op-norm corollary with a matrix-Bernstein / effective-dimension bound for ‖K_D−K‖_op; connect to KRR excess risk (supersedes the loose cor:krr / open T-T3).
- [ ] **P1 · T-R7-7 kernel-necessity demo** (the "so what"). SYNTHETIC off-sphere task where alignment-only fails AND distance-only fails BUT alignment×proximity wins; compare Gaussian / IMQ / polynomial / ⵟ kernels + RYF. SCOPE NOTE: kernel merit is partly the theory paper's burden; a necessity demo (not SOTA) suffices here. Do NOT just sphere-normalize a standard dataset (collapses to dot-product).
- [ ] **P2 · T-R7-8 fair-cost table.** Matched-memory + matched-wall-clock (not just matched-draws) table: RYF / TS-RYF / Gaussian-RFF / IMQ-RFF / k-means-Nyström. Partially covered by §4.9 (matched dim) + §4.6 (timing); extend to one consolidated table.
- [ ] **P2 · T-R7-9 "what we assume from bouhsine2026action" statement.** One short paragraph: PSD/Mercer positivity, universality (family-over-b), exact infinite feature map — taken as given; this paper's contribution starts after.
- [ ] **P2 · T-R7-10 "convenient sampler" wording.** Position the Bernstein mixture as a convenient sampler of the IMQ spectral law, NOT "no spectral route exists" (IMQ spectral densities are known).
- [ ] **P2 · T-R7-11 symmetric dimension default.** Make p_b dimension statements use d(d+1)/2+d+1 (symmetric) as the default, not d²+d+1 (already used in experiments/TensorSketch).

---

## Review batch R2 (2026-06-04, second reviewer — rigorous)  (T-V*)  ✅ ALL DONE
Verified: bugs are in EXPOSITION, not code (experiments use exact (x.w+b)^2 + correct √2), so NO results changed.
DONE: V1 (√b→b + proof typo), V2 (√2 in eq:flat + timing code), V3 (variance bound now has (1+3/(2D')) inner term,
valid at D'=1, proof via total variance), V4 (dimension-free softened: abstract/Fig1/contributions/§3.2/§4.4),
V5 (features→draws everywhere; matched-D≈matched-compute note + 64× dwarfs it), V6 (RERAN Gram+dimfree at D'=1;
dimfree at δ=0.10 → D*={200,350,500,750,750,750} plateaus, Nyström fails d≥20; Fig1 regenerated),
V7 (uniform Nyström stated, leverage-score future work), V8 (KRR fig ±1sd bands + λ=1e-2 fixed-not-tuned protocol),
V9 (cor:krr KRR-stability corollary), V10 (prop:bernstein_schur formalized: general k=p·f, unbiasedness+feature dim),
V11 (dot-product RFF cite kar2012random + reworded "sample only radial scale"), V12 (b<0 example removed — kernel is b≥0 by def).
- [x] **P0 · T-V1** Prop 2.1 bug: last coord of p_b is `b` not `√b` (squared-augmentation constant = √b·√b = b, self-product b²).
      Fix eq:biased_poly + proof typo "+b = +b²". Also b<0 note: the imaginary entry is `√(2b)` (linear), not √b.
- [ ] **P0 · T-V2** eq:flat missing √2: ψ at D'=1 is √2 cos, so z(x)=D^{-1/2}(√2 cos(·) p(x))_j. Fix display + timing code.
- [ ] **P0 · T-V3** Variance theorem omits inner O(1/D') term → invalid at recommended D'=1. Make Prop 3.2 (two-level
      identity) the PRIMARY variance result; derive thm:variance as a corollary (D'→∞ / explicit (1+c/D') factor).
- [ ] **P1 · T-V4** Soften dimension-free wording in abstract / Fig 1 caption / contributions: "radial sample count
      has no explicit d for a fixed dataset; total feature dim still scales with the polynomial dim."
- [ ] **P1 · T-V5** KRR framing: stop saying "D=32 features" → "D=32 random draws/scales"; report wall-clock; note
      matched-D ≈ matched Gram-assembly compute and the 64× gap dwarfs the ~3× compute diff. Standardize D/D'/M terms.
- [ ] **P1 · T-V6** Rerun Gram + dimension-free at recommended D'=1; report D_radial and M_explicit=D·d_b.
- [ ] **P1 · T-V7** Nyström: state "uniform Nyström; adaptive/leverage-score left to future work."
- [ ] **P2 · T-V8** More datasets + error bars + hyperparameter protocol (ε=median, b, λ fixed not tuned); sphere-norm first-class.
- [ ] **P2 · T-V9** KRR-stability corollary under ‖K−K̃‖_op (even if loose, N-factor).
- [ ] **P2 · T-V10** Bernstein–Schur: formalize (general k=p·f, unbiasedness/variance/feature dim) OR demote to discussion.
- [ ] **P1 · T-V11** Sharpen novelty (preserve poly exact, randomize only radial; flat beats two-level) + add product-kernel /
      dot-product-RFF related work. Reword "sample only the radial scale" (also samples ω; it's the IMQ spectral sampler).
- [ ] **P2 · T-V12** b<0 example: relabel as polynomial Gram, or give full-kernel eigenvalues (ε=1: 0.70, −0.20).

## 0. Goals

- **G1.** Give the random-feature scheme for the biased ⵟ-kernel `k_{\yat,b}` (sample only the radial
  scale, keep the polynomial factor exact) and prove it unbiased. → **DONE**
- **G2.** Bias-aware approximation theory: variance, uniform Gram error, op-norm concentration,
  dimension-free radial sample complexity. → **DONE (1 rigor gap open, T-T1)**
- **G3.** Empirically validate every theorem + show the method *scales* and is *useful*.
  → **intrinsic validation DONE; scalability + usefulness OPEN (T-E1..E5)**
- **G4.** Decide venue and fit the format. → **OPEN (T-W1)**

**North-star claim:** "The biased ⵟ-kernel admits a dimension-free random-feature approximation that lets
its kernel methods scale past the O(N²) Gram bottleneck, and the approximation is accurate, cheap, and
useful on real tasks."  ← the *useful on real tasks* clause is the unproven half.

---

## 1. Task board

### Review response (R1, borderline-accept, 2026-06-04)  (T-R*)
- [x] **P0 · T-R1** Reframe the dimension-free claim & Table 1 → §3.3 "Where the dimension enters, and what to
      compare against". Table now dataset-level for all 3 (all d-free; differ in constant + APPLICABILITY since
      RFF needs shift-invariance/dot-product, k_{\yat,b} has neither). Prose: domain-level d hits RYF's inner
      features too; the genuine d-contrast is **vs Nyström** (O(m^{-2s/d})). DONE.
- [x] **P0 · T-R2** Equivalence stated as linking commentary in Step 4: D'=1 estimator = exact-poly ⊗ RFF-on-IMQ
      (the scale mixture IS the IMQ spectral sampler, eq:bernstein); D'>1 = scale-clustered/correlated → worse →
      why D'=1 optimal (eq:flat). Repositioned the contribution (Schur exact-poly + IMQ spectral sampler), not a
      win over IMQ-RFF. Answers the reviewer's baseline by equivalence. DONE.
- [x] **STYLE** All `\begin{remark}` boxes converted to flowing context commentary that links sections
      (rem:normalization→eq:expectation bridging Step 2→3; rem:flat→Step-4 commentary threading Steps 1/2/§3.3/§5.5;
      rem:reading-bias→links §3↔intro↔§5.3). Per user request.
- [x] **P2 · T-R3** DONE (subsumed by T-E5): IMQ-RFF is implemented as the radial-only RYF; §5.7 confirms it.
- [x] **P1 · T-R4** DONE: intro now positions the work re mixture-based RFF (cite wilson2013gpkernels) + exact⊗random
      feature composition; the scheme "sits at the intersection of mixture-based RFF and exact feature maps."
- [x] **P2 · T-R5** DONE: ε/b + sphere-normalization sensitivity now a Limitation (iii) and an open question
      (learn ε,b unbiasedly); the §5.7 caveat documents the raw-vs-sphere instability.
- Reviewer points already tracked elsewhere: constants 1-vs-2 → T-T1; explicit D,D' bound → T-T2 (also makes
  "D'=1 optimal" a theorem, dovetails T-R2); downstream task → T-E5; stronger/leverage Nyström → T-E2;
  notation garbling kΣ/kΞ/kE = the ⵟ font bug, FIXED via \yat.

### Theory  (T-T*)
- [x] Construction in 4 steps (Schur → Bernstein–Widder → sampling → RFF) + estimator
- [x] Unbiasedness (Thm)
- [x] Variance bound, carrying bias as (R²+b)⁴ (Thm)
- [x] Uniform Gram approximation (Thm) + sample-complexity corollary
- [x] Operator-norm concentration (Thm)
- [x] Per-scale term factored as the single shared object behind all 3 concentration results
- [x] Bernstein–Schur generalization corollary
- [x] **P0 · T-T1** Constant audit DONE. Found+fixed THREE errors: (1) variance dropped a 1/ε → now
      V_D ≤ (R²+b)⁴/(Dε²) [abstract/conclusion updated]; (2) uniform used |ψᵀψ|≤1 (should be ≤2 a.s.) AND /√ε
      (Hoeffding gives /ε) → now (R²+b)²/ε·√(8log(2N²/δ)/D), and Cor sample-complexity ε⁻¹→ε⁻²; (3) op-norm
      "concentration" hid an N factor (L=(R²+b)²/ε wrong) → demoted to honest Corollary via ‖·‖_op≤‖·‖_F≤N·max,
      giving N(R²+b)²/ε·√(...). Comparison table RYF row → ε⁻². Appendix proofs rewritten explicitly.
- [x] **P2 · T-T2** DONE → Proposition (explicit D,D' variance, prop:budget): Var = poly²·(V_out/D + V_in/(DD')),
      so D'=1 optimal at fixed budget — now a first-principles statement, not a heuristic. Proof in App A. §5.5 cites it.
- [~] **P1 · T-T3** Excess-risk bound DEFERRED. The downstream experiment (T-E5) demonstrates RYF→exact KRR
      empirically; a formal bound via the loose op-norm corollary would be N-weak. Listed as an open question
      (feature-covariance / Rudi–Rosasco route). Not blocking.

### Experiments  (T-E*)
- [x] Gram-matrix error vs Nyström, O(1/√D) (§5.1, Fig 1a, `gram_approx.py`)
- [x] Variance + QMC/ORF reduction (§5.2, `variance_validation.py`)
- [x] QMC rate honesty check — constant not rate (§5.2, `qmc_rate.py`)
- [x] Bias scaling, exponent 4.01 (§5.3, `bias_scaling.py`)
- [x] Dimension-free sample complexity (§5.4, Fig 1b, `dimension_free.py`)
- [x] Outer/inner budget → flat sampling optimal (§5.5, `budget_allocation.py`)
- [x] Importance sampling (App, `importance_sampling.py`)
- [x] Polynomial sketching (App, `poly_sketch.py`)
- [x] **P0 · T-E1** Timing+memory scaling DONE (`timing_scaling.py`, §5.6, tab:scaling). Exact ridge time-slope
      2.1 (≈N²), Gram=33GB@N=64k (OOM); RYF 1.5s/1.5GB, Nyström 0.02s/30MB @64k. Surfaced honest caveat:
      RYF representation M=D·d_b' carries the d² polynomial → bigger constant than Nyström's m (2880 vs 64);
      symmetric reduction/sketching shrink it; the N-scaling (not constant) clears the wall.
- [x] **P1 · T-E2** Real data DONE (subsumed by T-E5): digits + california, sphere-normalized.
- [x] **P1 · T-E3** Op-norm validation DONE (`opnorm_validation.py`, §5.1 sentence). rel-op decays slope −0.40
      (Fro −0.45), stays below Fro (0.43→0.059 vs 0.59→0.071 over D=10→1000). Confirms the op-norm corollary.
- [x] **P2 · T-E4** ORF in d>1 DONE (`orthogonal_features.py`, App C). Variance ratio 0.72 (d=5) / 0.76 (d=20),
      unbiased → §4 ORF claim STANDS (backed), not downgraded. d=1 still shows nothing (no direction).
- [x] **P1 · T-E5** Downstream KRR DONE (`krr_downstream.py`, §5.7, tab:krr). **Headline result:** at matched
      random-feature budget the EXACT polynomial makes RYF crush feature-sampling baselines at low D — RYF@32
      = 0.980 digits acc / 0.547 cali RMSE vs Gaussian-RFF/IMQ-RFF/Nyström 0.81–0.89 / 0.60–0.62; RYF→exact
      ⵟ-KRR as D grows. Backs the comparison table empirically (resolves review point). ⵟ competitive as a kernel
      (ties digits, trails Gaussian 0.03 on cali). Honest caveat: needs sphere normalization (raw → exact 0.84, RYF diverges).
- [-] **P2 · T-E6** ε-scaling SKIPPED (low value; corrected variance bound already has ε², and §5.7 surfaced the
      practically relevant ε/normalization sensitivity).

### Writing / structure  (T-W*)
- [x] SLAY-style intro with related work blended (no separate Related Work section)
- [x] Step-by-step method section
- [x] Figure 1 (convergence + dimension-free)
- [x] Appendix moved after references; full proofs → appendix
- [x] ⵟ glyph via `\yat` macro (XeLaTeX/ebrima); em dashes removed
- [x] Cite `bouhsine2026action` as a normal reference (no "parent paper" disclaimer)
- [ ] **P0 · T-W1** Decide venue (JMLR = no length limit, theory-fit · vs NeurIPS/ICML = 9-pg main).
      Blocks the trim decision.
- [ ] **P1 · T-W2** Length pass: main is **11 pp**. If NeurIPS/ICML, trim to 9 (fold the 5 experiment
      subsections tighter, push more to appendix). *(depends on T-W1)*
- [ ] **P2 · T-W3** Second figure (timing plot from T-E1, or the bias-scaling log-log).
- [ ] **P2 · T-W4** Final proofread: text↔table number consistency, captions self-contained, abstract numbers.

### Ops  (T-O*)
- [x] All scripts archived with reproducibility headers + deterministic seeds
- [x] `DEPENDENCY_MAP.md`, `notes.md`, `PLAN.md`
- [ ] **P1 · T-O1** Commit paper + experiments (nothing committed yet).
- [ ] **P2 · T-O2** README in `experiments/` listing each script → which table/figure it backs.

---

## 2. Task dependency map

```
                 ┌─────────────────────────────────────────────┐
                 │  DONE: construction, theory(stmts), proofs,  │
                 │  intrinsic experiments, Fig 1, restructure,  │
                 │  font, em-dash, citation style               │
                 └───────────────────────┬─────────────────────┘
                                         │
   independent (do anytime)              │ unlock the remaining contribution
   ────────────────────────              ▼
   T-T1 (op-norm fix) ─────────────► [internally complete & honest paper]
   T-E1 (timing) ──────► T-W3 (timing figure)
   T-E3 (op-norm exp) ◄── validates T-T (concentration thm)
   T-E4 (ORF d>1) ────► fixes §4 ORF claim
   T-E6 (eps scaling)

   the "useful" half (fork):
   T-E2 (real-data Gram) ──┐
                           ├──► T-E5 (downstream KRR/GP task) ──► empirical backing for comparison table
                           │                                  └─► T-T3 (excess-risk bound)
   venue:
   T-W1 (venue decision) ──► T-W2 (length trim 11→9)   [only if NeurIPS/ICML]

   closeout:
   all content tasks ──► T-W4 (proofread) ──► T-O1 (commit)
```

**Critical paths**
- *Minimal complete (tight pure-approximation paper):* T-T1 + T-E1 + T-E3 (+ T-E2) → T-W4 → T-O1. ~1.5 days.
- *Full contribution:* the above **+** T-E2 → T-E5 → T-T3. +~2 days. Reopens downstream scope.
- *Venue/format:* T-W1 gates T-W2; independent of the science.

---

## 3. Milestones

- **M1 — Internally complete & honest** ✅ **REACHED 2026-06-04**: T-R1, T-R2, T-T1, T-E1, T-E3, T-E4 all done.
  Op-norm now rigorous (honest Frobenius corollary), all constants audited+fixed, scalability shown (§5.6),
  every theorem/corollary has a backing experiment, dimension-free claim honest, IMQ-RFF equivalence stated.
- **M2 — Real-data credible:** + T-E2.
- **M3 — Full contribution:** + T-E5 + T-T3 (downstream usefulness + comparison empirically backed).
- **M4 — Submission-ready:** + T-W1/W2 (venue + length), T-W4 (proofread), T-O1 (commit).

**Suggested order:** T-E1 → T-E3 → T-T1 → T-E2 → (decide on T-E5 fork) → T-W1/W2 → T-W4 → T-O1.

---

## 2026-06-10 BATCH — leverage theorem, positive-feature dichotomy, grammar + QM9 (all DONE, integrated)

Driver: a research-strengthening pass ("make the research stronger, don't care about acceptance").
Six items shipped to `main.tex` + `references.bib` (compiles clean, 44pp). Scripts archived + reproducibility headers.

- [x] **thm:krr_leverage** (was the rmk:risk open problem) — whitened-leverage tilted radial sampling
  d̄_λ(θ)=ψᵀ(A⁻¹∘P)ψ, E_π[d̄]=d_eff(λ) exactly → count (1+‖P‖/λ) → (1+d_eff(λ)), same Tropp machinery,
  same variance core ⇒ intdim = d̃_λ. Class-wide via thm:class_bernstein. `leverage_radial_sampling.py`:
  uniform D* 50→800→>3200 as λ 10→1→0.1; leverage 50/200/400/1600 tracking d_eff=12/39/91/163.
- [x] **prop:pos_dichotomy** — FAVOR⁺∘Exp(ε) 2nd moment = ε/(ε−8x·w), INFINITE for 8x·w≥ε. Scopes
  prop:positive to the diffuse regime; corrects the intro "no reduction" claim. `positive_features.py`.
- [x] **rmk:complex measured** — complex degree-2 TS: η + sketch term ~1.6× lower, variance halved. `complex_sketch.py`.
- [x] **kernel-grammar customer** — LIN²×RQ_α IS Bernstein–Schur (Γ(α,2ασ²), m_f=1); california end-to-end.
  `grammar_kernel.py`; cite `duvenaud2013structure`.
- [x] **prop:quadrature scoped** — `radial_quadrature.py`: exact-g machine precision by D=32; RFF-paired inner
  noise dominates (naive nodes WORSE than MC). Nodes = right tool for exact-g / positive-feature bounded scales only.
- [x] **tab:higgs → 3 seeds** (`higgs_scaling.py --seed`); exact-mod D=1 instability (±.027) now visible.
- [x] **m→m_f** notation fix in thm:bernstein_schur appendix proof.
- [x] Discussion open-problems rewritten: (1) deployable leverage tilt + d_eff(K*_D) ⇒ risk + Ω(‖P‖/λ) lower bound;
  (2) is (R²+b)⁴ optimal over ALL unbiased maps (conj. yes); (3) peaked-regime finite-variance estimator; (4) manifold.

### QM9 atomization (tab:qm9) — the real coupled-target test (DONE, honest)
Replaces the long-standing "one larger real off-sphere dataset" / M2 gap. QM9 (133,885 mols),
Coulomb-matrix eigenspectrum (d=29) + sorted full CM (d=435), tuned KRR + 2 coupling ablations + dressed-atom.
`qm9_build_cache.py`, `qm9_atomization.py`.
- **Coupling: real but modest.** Direction-only ablation costs 2–11% (extensivity ⇒ small genuine norm signal);
  norm-only ≈ useless (161–173 vs ~190 mean). Direction-dominated, not strongly coupled.
- **Kernel preference does NOT transfer:** ⵟ ties IMQ/best on full CM (15.9), behind Gaussian on eig (35.7 vs 31.6).
  Consistent with the 2026-06-05 COUPLING VERDICT (real data ⇒ one geometry dominates).
- **Two construction-positive findings:** (a) RAY beats its OWN exact kernel on eig (32.6 vs 35.7) — RF low-rank
  truncation = regularizer on a near-noiseless target. (b) full-N sketched primal DOMINATES: at d=435 exact
  modulation (d_b≈95k/draw) + N×N Gram (1.7e10) both impossible, yet M=9024 primal on all 131,885 rows → 14.5
  kcal/mol in 35s, BEATS the best 6k-Gram kernel (15.9). "More data through compressed features > better kernel on a subsample."

---

## ROADMAP — Gaussian-process problems for the Bernstein–Schur class (NEXT, not yet built)

Why GP-shaped: the class = (finite modulation) × (completely monotone radial). GP kernel-search grammars
(duvenaud2013structure) emit exactly these products (LIN^p × {RQ, Matérn, Gaussian, IMQ}); RAY scales them
where the Gram can't form. Two GP-specific extras the paper does NOT yet claim but RAY supports:
  (a) **predictive variance** σ²·z(x)ᵀ(ZᵀZ+σ²I)⁻¹z(x) — full GP (mean + uncertainty) from the same M×M primal solve;
  (b) **ML-II hyperparameter learning** — rmk:learnable already gives differentiability in (ε,b) with base
      randomness fixed ⇒ marginal-likelihood optimization through unbiased features. Currently unused.

STRUCTURAL FILTER (state before picking): IN-class = LIN/poly modulation × CM radial (RQ, Matérn, Gauss, IMQ, exp).
OUT = anything PERIODIC (PER not CM) and pure stationary RBF/Matérn alone (that's just RFF). So the classic
SE×PER×LIN time-series grammar is only PARTLY ours.

**Tier 1 — "exact GP on a million points" suite (strongest scaling story; Wang et al. 2019 needed 8 GPUs + CG):**
- [ ] **3DRoad** (434k, d=3) — BEST FIRST: small-d (no O(d²) floor), spatial ⇒ Matérn-natural, published RMSEs,
      lets us show mean AND predictive variance from one primal solve. Kernel: LIN²×Matérn-½ (two-line Lévy sampler ready).
- [ ] **HouseElectric** (2M, d=11) — the million-point headline.
- [ ] Song/YearMSD (515k, d=90), Buzz (583k, d=77) — breadth.
  CAVEAT: these benchmarks default to plain RBF; our claim is "a product kernel, fit where exact can't," vs SGPR/SVGP/SKI.
  CONFIRM download (UCI) before committing — figshare WAF-blocked QM9; deepchem S3 mirror saved it.
- [~] **QM9** — DONE (tab:qm9); the molecular-GP member of this suite.

**Tier 2 — geostatistics / universal kriging (best PHYSICAL motivation):**
- [ ] Universal kriging = polynomial drift × stationary (Matérn) covariance — the multiplicative non-stationary
      version IS k_ⵟ,b. Spatial datasets (precipitation, climate) make the modulation physically meaningful, large N.

**Tier 3 — SVGP home turf:**
- [ ] **Airline delays** (5.9M) — Hensman et al. 2013 stochastic-variational benchmark; RAY streaming primal =
      direct data-independent competitor to inducing-point SVGP. Hardest to source (US flights).

**Recommended next:** 3DRoad with LIN²×Matérn-½ + predictive variance (clean grammar_kernel.py extension), gate
download first. Then HouseElectric for the million-point headline. The predictive-variance + ML-II angle is the
genuinely NEW GP content beyond the current KRR results.

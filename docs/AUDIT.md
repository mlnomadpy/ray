# RAY paper — full audit (2026-06-10)

Scope: every stated result and proof in `main.tex` (37 results), the experiment-to-claim
mapping, and the open-problem inventory. Verdict first, then findings ordered by severity.

**Verdict.** The theory chain is sound. Every proof was checked by hand; the load-bearing
arguments (Schur-multiplier lemma, the three matrix-Bernstein ingredients and their
whitening, the leverage-tilt trace normalization, the positive-feature MGF computation,
the ridge-sketch transfer through the unit-diagonal Schur product) are correct, including
the constants in `thm:exact_variance`, `thm:uniform`, `thm:bernstein_schur`,
`cor:bernstein_tail`, `thm:krr_whitened`, and `thm:krr_leverage`, which I re-derived. The
findings below are: four proof-hygiene items (none breaks a result; all are
constant/provenance/wording-level), two consistency nits, and then the genuine research
gaps — theoretical, empirical, and positioning.

---

## A. Proof-hygiene findings (resolved in `main.tex`)

### A1. Provenance of the *expectation-form* intrinsic-dimension Bernstein (`thm:bernstein`)
**Status:** fixed. The proof now derives the expectation bound by integrating the two-sided intrinsic tail, yielding the displayed constants in `thm:bernstein` instead of citing an expectation-form intrinsic Bernstein statement.

Original finding: the proof cited Tropp for $\mathbb{E}\|\sum_j Y_j\| \le \sqrt{2v\log(2d_{\mathrm{int}})} + \tfrac13 L\log(2d_{\mathrm{int}})$.
Tropp's intrinsic-dimension result (2015, Thm 7.3.1) is a **tail** bound, valid for
$s \ge \sqrt v + L/3$; the expectation bound in this form is the **ambient-dimension**
statement (Tropp 2012, Thm 6.1) with $d_{\mathrm{int}}$ substituted for $N$. That
substitution is true up to constants (integrate the 7.3.1 tail), but it is not literally
in the cited source. Fix: one appendix sentence deriving the expectation form by tail
integration, or restate with the constants that integration actually gives.

### A2. Tail prefactor 4 vs 8 (`cor:bernstein_tail` vs the whitened theorems)
**Status:** fixed. `cor:bernstein_tail` and the sketched radial term now use the two-sided prefactor, with logs `\log(8d_{\mathrm{int}}/\delta)` and `\log(8d_{\mathrm{int},S}/\delta)`.

Original finding: Thm 7.3.1 bounds $\lambda_{\max}$ only. The whitened proofs correctly apply it to
$\pm\sum_j\widetilde Y_j$ and carry prefactor $8\tilde d_\lambda$. `cor:bernstein_tail`
states $\mathbb{P}\{\|\sum_j Y_j\|\ge s\}\le 4d_{\mathrm{int}}e^{-\cdot}$ — the operator
norm needs the same two-sided union, i.e. $8d_{\mathrm{int}}$ (the summands are not
symmetric in distribution: $K^{(j)}\succeq 0$ is one-sided). Cosmetic — only the constant
inside the log — but it is an internal inconsistency a careful referee will spot.

### A3. Unstated validity restriction in the tail inversions
**Status:** fixed. The proof of `cor:bernstein_tail` now records the range check: with $\ell=\log(8d_{\mathrm{int}}/\delta)\ge\log8>1$, the inverted root is at least $\sqrt v+L/3$; the whitened proofs inherit the same check with their own `\ell`.

Original finding: all three tail inversions (`cor:bernstein_tail`, `thm:krr_whitened`,
`thm:krr_leverage`) ignore 7.3.1's range restriction $s \ge \sqrt v + L/3$. It is
automatically satisfied — $\ell = \log(4d_{\mathrm{int}}/\delta) \ge \log 4 > 1$ gives
$\sqrt{2v\ell} \ge \sqrt v$ and $\tfrac23 L\ell \ge L/3$ — but the one-line check should
appear once (e.g. in the proof of `cor:bernstein_tail`, inherited by the others).

### A4. $\ell$ mismatch in the `cor:krr_deployed` proof
**Status:** fixed. The appendix proof now applies `thm:krr_whitened` to the sketch-conditioned pair and keeps the statement's $\log(8\tilde d_{\lambda,S}/\delta)$ throughout.

Original finding: the statement's count uses $\log(8\tilde d_{\lambda,S}/\delta)$; the proof set
$\ell=\log(8N/\delta)$ via "intrinsic dimension at most $N$". The statement is the
correct (stronger) form — the true intrinsic dimension of the majorant *is*
$\tilde d_{\lambda,S}$ by definition — so the proof's $N$ is unnecessary slack and the
two should use the same $\ell$. (The arithmetic $\eta\le\rho_0/4 \Rightarrow
(1+\eta)\rho_0/2+\eta\le\rho_0$ checks out: the exact threshold is
$\eta\le\rho_0/(\rho_0+2)$, and $\rho_0/4 < \rho_0/3 \le \rho_0/(\rho_0+2)$ for $\rho_0\le1$.)

### A5. Dangling "spherical companion" reference (`rmk:complex`)
**Status:** fixed. The dangling spherical-companion sentence was removed from `rmk:complex` and its per-result doc.

Original finding: `rmk:complex` said "the spherical companion derives the on-sphere version of the complex
fourth moment explicitly" with no citation, while the intro says the on-sphere route is
"left to future work." Now that `papers/01_theory/spherical_yat_features/` exists, either
cite it in both places or keep both as future work — currently the paper points at a
document it never names.

### A6. Finite-mass hypothesis reach (`tab:bernstein_schur`)
**Status:** fixed. The table caption now states the finite-mass hypothesis $m_f=f(0)<\infty$, and the linear-modulation row says "finite-mass completely monotone $f$."

Original finding: the class table's linear-modulation row said "any completely monotone $f$" — it silently
inherits `thm:bernstein_schur`'s hypothesis $m_f=f(0)<\infty$, which excludes radial
factors singular at $0$ (e.g. $r^{-\alpha}$ unshifted). One phrase in the caption closes it.

---

## B. Theory gaps (the real open problems, ranked by value)

### B1. The excess-risk theorem (the missing headline)
`rmk:risk` is honest: a $\rho$-sandwich does not bound fixed-design risk (the
$K=0$ vs $K'=\rho\lambda I$ counterexample). With the count side closed by
`thm:krr_leverage`, a minimax RAY-KRR statement needs exactly two ingredients:
(i) a **deployment-grade tilt** — pilot uniform estimate or Nyström/sketched $A^{-1}$ —
with its approximation error folded into the bound; (ii) **control of
$d_{\mathrm{eff}}(K^*_D)$**, the approximate kernel's effective dimension (the
Rudi–Rosasco feature-covariance step). Both are standard-shaped; this is the highest-value
single theorem the paper could still add.

### B2. Lower bound: is the tilt *necessary*?
A matching $\Omega(\|P\|_{\mathrm{op}}/\lambda)$ draw-count lower bound for plain
$\Exp(\varepsilon)$ sampling on worst-case data would certify that leverage tilting is
necessary, not merely sufficient. The validation data already points there (uniform
$D^*$ blows past 3200 at $\lambda=0.1$ while leverage sits at 400). Likely provable by a
two-point construction concentrating $K$'s small eigenvalues where the base law rarely puts energy.

### B3. Prefactor optimality (declared conjecture)
`prop:variance_sharp` pins $(R^2+b)^4/\varepsilon^2$ with equality *within* the factored
family. Whether any unbiased data-independent feature map of k_ⵟ,b beats it must couple
the modulation and radial randomness so errors cancel. Either construct one (negative
result for the conjecture) or prove a family-free lower bound via an unbiasedness
constraint on the second moment. Currently neither direction has an attack written down.

### B4. The peaked regime (blocks the attention application)
Small $\varepsilon$ + aligned data defeats both estimators: trig error grows with
sharpness, and `prop:pos_dichotomy` proves positive features have infinite variance
exactly there ($8x^\top w\ge\varepsilon$). Any ⵟ-attention deployment at realistic
attention temperatures runs into this. Candidate directions: data-dependent normalization
(FAVOR-type), the bounded-scale node sets of `prop:quadrature` (the paper already notes
nodes are "the bounded scale sets positive features require" — that hybrid is unbuilt),
or a control-variate around the dominant scale.

### B5. Sketch-side leverage
`thm:krr_leverage` tilts only the radial draw; the modulation sketch still pays the
generic OSE rate (`rmk:ose`, superlinear in $s_\lambda$ for degree-2 TensorSketch). Two
easy strengthenings: (a) swap in Ahle et al. 2020 oblivious sketches for a near-linear
$s_\lambda$ count — no new analysis, just a citation and a constant; (b) the genuinely
open one: a *joint* radial-tilt × leverage-sketch count.

### B6. Geometry of $d_{\mathrm{int}}$ (open problem 4)
The data-adaptive constant is empirically tiny ($d_{\mathrm{int}}\in[1.4,3.5]$ on the
tested balls) but the paper has no statement linking manifold/thin-shell structure to
$d_{\mathrm{int}}$. A bound $d_{\mathrm{int}} \lesssim$ intrinsic manifold dimension
under a reach condition would turn the empirical observation into a theorem.

### B7. Class breadth: infinite-dimensional modulations and CM×CM products
`thm:bernstein_schur` requires a **finite** modulation feature. Two natural extensions:
(i) modulation with an infinite but RFF-able feature (Gaussian×IMQ — i.e. SE×RQ, which
the Automatic Statistician grammar emits constantly and the current class does *not*
cover: sample both mixing measures); (ii) products of two completely monotone radials.
Both are one-page extensions of the unbiasedness proof; the matrix-level transfer needs
the modulation Gram bound replaced by a high-probability one.

### B8. GP functionals (roadmap, no theorem yet)
PLAN.md lists the GP program. The cheap theorem available now: the sandwich already
controls $y^\top(K+\lambda I)^{-1}y$-type functionals, but **pointwise predictive
variance** $k(x,x)-k_x^\top(K+\lambda I)^{-1}k_x$ at a test point needs the whitened
bound extended to bordered Grams. Worth writing before the 3DRoad experiment so the
experiment validates a stated guarantee rather than just behavior.

---

## C. Empirical gaps

1. **Leverage tilt is validated only synthetically** ($N=300$, pool resampling). No
   pilot-estimate (deployable) tilt has been run, and no real mid-size dataset (e.g.
   california $N=3000$) demonstrates that the $d_{\mathrm{eff}}$ count predicts $D^*$.
   This is the experiment that would make B1(i) concrete.
2. **No real-data win for the coupled bias.** CIFAR ties, QM9 ties-or-trails, HIGGS goes
   to Gaussian. The synthetic preference (`tab:necessity`) is the only place ⵟ wins *as a
   kernel*. Either find a genuinely coupled real domain or promote the scoping (the
   construction/guarantees are the contribution; the kernel is the flagship instance)
   from the discussion into the framing.
3. **QM9 "RAY beats its own exact kernel" is unexplained-but-explained.** The
   truncation-as-regularizer story (32.6 vs 35.7) is plausible and untested: an exact
   kernel + rank-capped Gram control would isolate it in one run.
4. **Full-N primal rows are single-seed** (QM9 26.8/14.5; grammar primal 0.480). Cheap to
   re-run at 3 seeds; the paper's own standard (multi-seed default) asks for it.
5. **The d=5000 floor-breaker is archived but unused** (`scaling_suite.py`, gisette):
   the cleanest "exact feature impossible, only RAY runs" demonstration. Candidate
   high-$d$ table; README already flags it.
6. **Attention is fidelity-only** (deliberately scoped) — the trained ⵟ-attention LM
   belongs to the architecture line, but the peaked-regime gap (B4) should be re-tested
   the moment any candidate estimator exists.

## D. Positioning

1. The abstract/intro still cast k_ⵟ,b as protagonist while the results' center of
   gravity is class + construction + complete guarantee chain (three honest negatives on
   real-data kernel preference now in print). The discussion already says this; the intro
   doesn't yet. One re-pitched paragraph would align them.
2. `tab:rf_comparison` and the flat-estimator identity ("RAY = IMQ-RFF × exact
   modulation", line ~187) are the right honesty; keep them prominent — they pre-empt
   the "isn't this just RFF with extra steps" referee question by answering it.

---

*Method note: all proofs re-derived by hand against the tex; constants in
`thm:uniform` ($\sqrt8$), `thm:exact_variance` (bracket $=\tfrac12$ at $r=0$, $\le\tfrac32$),
`thm:bernstein_schur` ($\tfrac32 m_f^2B^4/D$, Hoeffding range $4m_fB^2$),
`cor:krr_deployed` ($\eta\le\rho_0/4$ threshold), and the leverage identity
$\operatorname{tr}(A^{-1}K^{(\theta)})=\psi_\theta^\top(A^{-1}\circ P)\psi_\theta$ all verified.*

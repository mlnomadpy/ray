# Bernstein–Schur Kernels (RAY) — build & reproducibility

## Theory documentation

`docs/` holds one markdown file per result (37 results, monograph template:
What it says / Why it matters / Proof idea / Connections), indexed by
[`docs/00-overview.md`](docs/00-overview.md). The full proof audit and the
ranked gap inventory (proof hygiene, open theory, missing experiments) is
[`docs/AUDIT.md`](docs/AUDIT.md).

## Building the PDF

The paper compiles with **XeLaTeX** (not pdfLaTeX — it uses `fontspec` + a Tifinagh
glyph for the ⵟ symbol).

```bash
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex   # second pass for refs
```

**Requirements**
- XeLaTeX (TeX Live 2022+), `fontspec`, `tmlr.sty`/`tmlr.bst` (TMLR style, included).
- Font: `ebrima.ttf` for the ⵟ (U+2D5F) glyph, declared via
  `\DeclareRobustCommand{\yat}{\text{\normalfont\tifinaghfont ⵟ}}`. If the font is
  unavailable, replace the `\yat` macro with a text fallback (e.g. `\mathrm{Y}`) — no
  other change is needed.
- Figures (PDF, in this directory): `fig_offsphere.pdf`, `fig1.pdf`, `fig_bias.pdf`,
  `fig_scaling.pdf`, `fig_krr.pdf`.

`[preprint]` mode is set in `tmlr`; the author block is de-anonymized for the preprint.
For anonymous submission, switch to the anonymous TMLR option and remove `\name/\email/\addr`.

## Experiments → tables/figures

All scripts are in `experiments/`, seed-deterministic, each with a reproducibility header.
Run with `~/.pixi/envs/jax/bin/python3` (numpy/sklearn) or `/opt/homebrew/bin/python3`
(the MLX/torch ones). Results are archived under `experiments/results/*.json`.

| Script | Backs |
|---|---|
| `off_sphere_gram.py` | Fig. `fig:offsphere`, Table `tab:offsphere` (lead experiment) |
| `variance_validation.py`, `exact_variance_check.py` | `tab:variance_reduction`, `thm:exact_variance` |
| `budget_allocation.py` | `tab:budget` (flat sampling optimal) |
| `ts_ryf_costmatched.py`, `ts_decomposition.py` | `tab:ts`, `prop:ts_variance` |
| `ts_opnorm_validation.py`, `make_ts_opnorm_fig.py` | `thm:ts_opnorm`, Fig. `fig:ts_opnorm` (doubly-randomized RAY operator-norm) |
| `dm_tradeoff.py` | `tab:dm` (deployed-RAY D-m allocation at fixed M, d=16/64/256) |
| `tuned_downstream.py` | `tab:tuned` (validation-tuned b,eps,lambda for all kernels) |
| `runtime_vs_d.py` | `tab:runtime_d` (exact-mod vs sketched feature-build runtime/memory vs d) |
| `preprocessing_ablation.py` | `tab:prep` (bounded-input preprocessing ablation) |
| `krr_spectral_sketched.py` | `tab:krr_sketched` (KRR-spectral stability for sketched RAY, vs D/m/lambda) |
| `coupled_matched.py` | `tab:coupled_matched` (matched-cost RF inductive-bias test; honest negative) |
| `yat_attention.py`, `make_attention_fig.py` | Fig. `fig:attention` (linear-time streaming yat-attention: fidelity + O(N^2) wall + exact causal recurrence) |
| `fair_cost.py`, `off_sphere_faircost.py` | now also report `rls-Nystrom` (ridge-leverage) |
| `fair_cost.py` | `tab:faircost` (sphere) |
| `off_sphere_faircost.py` | `tab:offsphere_faircost` (off-sphere) |
| `necessity_demo.py` | `tab:necessity` (when the coupling matters) |
| `signal_gate_snr.py` | `tab:gate` |
| `higgs_scaling.py` | `tab:higgs` (MLX GPU; HIGGS.csv.gz) |
| `bernstein_intrinsic.py`, `opnorm_validation.py` | `thm:bernstein` checks |
| `gram_approx.py`, `dimension_free.py`, `bias_scaling.py`, `krr_downstream.py` | Appendix Further Validations |
| `cifar_embed.py`, `cifar_kernel_krr.py` | `tab:cifar` (CLIP embeddings, appendix) |
| `eps_bias_sensitivity.py` | `tab:sensitivity` (appendix) |
| `bernstein_schur_demo.py` | non-ⵟ Bernstein–Schur instance (appendix) |
| `leverage_radial_sampling.py` | `thm:krr_leverage` (whitened-leverage tilt: D* tracks d_eff, not ‖P‖/λ) |
| `positive_features.py` | `prop:pos_dichotomy` (FAVOR+ infinite-variance threshold 8x·w≥ε; off-sphere trig vs positive Gram) |
| `complex_sketch.py` | `rmk:complex` (complex vs real degree-2 TensorSketch: η and sketch term ~1.6× lower, variance halved) |
| `grammar_kernel.py` | kernel-grammar customer (LIN²×RQ on california; class estimator end-to-end, appendix) |
| `radial_quadrature.py` | `prop:quadrature` empirical scoping (exact-g: machine precision by D=32; RFF-paired: inner noise dominates) |
| `qm9_build_cache.py`, `qm9_atomization.py` | `tab:qm9` (real coupled-target test: QM9 atomization; CM eigenspectrum d=29 + full sorted CM d=435; ablations; full-N sketched primal) |

Large datasets (HIGGS, CIFAR, CLIP weights) auto-download to `~/rf_data/` / `~/higgs_data/`
on first run; see each script's header for the exact command and expected numbers.

### Exploratory scripts not in the paper

These ran but are not cited by any table/figure in `main.tex`; they are kept as archived,
reproducible artifacts.

| Script / result | Status |
|---|---|
| `scaling_suite.py` → `results/scaling_suite_floor.json` | Multi-dataset floor-breaker (gisette `d=5000`, madelon). Superseded as the in-paper systems result by HIGGS (`tab:higgs`) and the appendix scalability figure (`fig:scaling`). Still the cleanest demonstration that at `d=5000` the exact-modulation feature (`d_b≈1.25e7`) is impossible to build and only sketched RAY runs — a direct proof of Limitation (v). Not yet integrated; candidate for a future high-`d` table. |
| `gate_diagnostic.py` → `results/gate_diagnostic_*.json` | Pre-flight gate diagnostics; the in-paper gate result uses `signal_gate_snr.py` (`tab:gate`). |

# GenFin-Hybrid: leakage-aware hybrid modelling of corporate capital structure

Python code for a leakage-aware modelling framework for corporate capital structure, with
validation on real firm data. It contains the experiment scripts, the figure scripts, the result
tables and the figures they produce.

## What the code does

- generates a reference synthetic firm panel (296 firms x 15 years = 4,440 firm-years) from a
  documented data-generating process (`code/common.py::generate_panel`, seed 42);
- audits 40 candidate features for target leakage and builds a leakage-free (strict) feature set;
- benchmarks machine-learning models under random, firm-grouped cross-validation, out-of-time and
  unseen-firm protocols, including deep tabular models (E1-E3);
- compares constrained data generation with CTGAN, TVAE and noise baselines (E4);
- runs a trade-off optimisation with Monte Carlo sensitivity analysis (E5) and panel econometrics (E6);
- validates on two real samples: 46 S&P-100 firms from Yahoo Finance (E7) and an SEC XBRL 10-K
  panel (E8).

## 1. Quick start

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/smoke_test.py         # about 30 s; no third-party data needed
```

The smoke test byte-compiles all scripts, runs `code/test_common.py` with outputs sent to a
temporary folder, and checks that the regenerated leakage audit, data-generating-process parameters
and variance decomposition equal the reference copies in `results/reference_tables/`. It prints
`SMOKE TEST PASSED` and exits with code 0 on success.

Full regeneration (needs the third-party data in section 5):

```bash
bash scripts/run_all.sh                                        # Linux/macOS/Git Bash
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1   # Windows
```

Options: `PYTHON=...` / `-Python ...` (interpreter), `DOWNLOAD_YAHOO=1` / `-DownloadYahoo` (fetch
Yahoo Finance statements first), `SKIP_SEC=1` / `-SkipSec` (skip E8 when the SEC file is not
available). Logs go to `logs/`.

## 2. Configuration

No path is hard-coded. `code/config.py` resolves every location relative to the repository folder,
and each can be overridden with an environment variable:

| Variable | Default | Used for |
|---|---|---|
| `GENFIN_ROOT` | repository folder | base of all relative paths |
| `GENFIN_OUT` | `output/` | all experiment outputs |
| `GENFIN_FIG_DIR` | `figures/` | figure set written by the figure scripts |
| `GENFIN_YAHOO_DIR` | `data/recent_data/` | Yahoo Finance statement CSVs |
| `GENFIN_SEC_FILE` | `data/sec_datavidence/datavidence-financials-sample.csv.gz` | SEC XBRL sample |

`python code/config.py` prints the resolved paths and whether the data files were found.

## 3. Scripts

Run in this order (the run scripts do this). Times are wall-clock times on a 32-core x86-64 CPU
without GPU (see `ENVIRONMENT.md`).

| Step | Script | What it does | Output folder | Time |
|---|---|---|---|---|
| 0 | `code/test_common.py` | tests of `common.py`; panel reproduction; leakage audit by perturbation; split sizes; real-panel construction | `foundation/` | about 30 s |
| 1 | `code/e1_benchmark.py` | 12 models x 2 feature sets x 4 protocols; corrected resampled t-tests; panel seeds; DGP sensitivity; secondary targets; learning curve (resumable, `--fresh` refits) | `e1_benchmark/` | about 13 min |
| 2 | `code/e2_deep_tabular.py` | FT-Transformer, TabNet, tuned MLP and tuned LightGBM, 3 seeds (`--workers 5`, `--pilot`, `--figures-only`) | `e2_deep_tabular/` | about 53 min |
| 3 | `code/e3_shap.py` | TreeSHAP for feature specifications S1-S4, out-of-fold, rank stability, surrogate model | `e3_shap/` | about 1 min |
| 4 | `code/e4_generative.py` | noise baselines, CTGAN and TVAE (3 seeds), post-hoc projection; KS, Wasserstein, JSD, correlation error, constraint violations, discriminator AUC; sensitivity grids (fits cached in `deep_cache/`) | `e4_generative/` | about 12 min with cached fits |
| 5 | `code/e5_optimization.py` | trade-off model (Hamada relevering, Merton-linked spread, tax shield, distress cost); Pareto sets; Latin-hypercube Monte Carlo; tornado | `e5_optimization/` | about 5 min |
| 6 | `code/e6_econometrics.py` | OLS, firm-clustered and Driscoll-Kraay SEs; fixed and random effects, Hausman and Mundlak tests; real-panel regressions | `e6_econometrics/` | about 20 s |
| 7 | `code/e7_real_data.py` | 46-firm Yahoo Finance sample: descriptives, grouped CV, leave-one-firm-out, out-of-time, firm bootstrap, transfer from the synthetic panel, SHAP | `e7_real_data/` | about 12 min |
| 8 | `code_sec/e8_sec_validation.py` | SEC XBRL panel (built by `code_sec/sec_data.py`): regressions with wild cluster bootstrap, prediction protocols, transfer, SHAP | `e8_sec/` | about 10 min |
| 9 | `code/build_master.py` | assembles the main and supplementary tables (no model fitting) | `MAIN_TABLES/`, `SUPP_TABLES/` | seconds |
| 10 | `code/final_figures_main.py`, `code/final_figures_si_a.py`, `code/final_figures_si_b.py` | draw the figures from the saved CSVs (style in `code/figstyle.py`) | `figures/` | under 1 min |

`code/common.py` is the shared module (data-generating process, feature sets, leakage audit, split
functions, corrected resampled t-test, Yahoo panel loader). `code/config.py` holds the paths.
`outputs_original/` and `generative/baselines/` are inputs read by E1, E3 and E4 for consistency
checks against an earlier version of the analysis.

## 4. Results and figures

| Folder | Contents |
|---|---|
| `results/figures/` | the figures as PDF: `fig1`-`fig5` (main) and `figS01`-`figS30` (supplementary) |
| `results/reference_tables/` | the result tables (`MAIN_TABLES/`, `SUPP_TABLES/` with `INDEX.csv`) and `figure_sources.csv` (source data of each figure) |
| `results/figure_data/` | the CSV files the figure scripts read (same layout as `output/`) |
| `results/reference_synthetic_panel_seed42.csv.gz` | the reference synthetic panel (4,440 rows) |

Main figures and the experiment that produces them:

| Figure | File | Produced by |
|---|---|---|
| Workflow | `fig1_workflow.pdf` | `final_figures_main.py` (schematic) |
| Deep tabular models | `fig2_deep_tabular_r2.pdf` | E2 |
| SHAP and leakage ablation | `fig3_shap_leakage_ablation.pdf` | E3 |
| Optimisation Monte Carlo | `fig4_optimisation_mc.pdf` | E5 |
| Real-sample validation | `fig5_real_validation_r2.pdf` | E7 |

Supplementary figures: S01-S06 E1, S07 E2, S08-S12 E3, S13-S18 E4, S19-S22 E5, S23 E6, S24-S28 E7,
S29-S30 E8.

To redraw the figures from the included CSV files, without re-running the experiments:

```bash
GENFIN_OUT=results/figure_data python code/final_figures_main.py
GENFIN_OUT=results/figure_data python code/final_figures_si_a.py
GENFIN_OUT=results/figure_data python code/final_figures_si_b.py
```

(On Windows PowerShell: `$env:GENFIN_OUT = "results/figure_data"` first.) This redraws Figs. 1-5,
S1-S23, S27 and S29-S30. Figs. S24-S26 and S28 plot firm-level Yahoo Finance values, which are not
redistributed; they are drawn after E7 has been run on downloaded statements.

### Key results

Values below are copied from `results/reference_tables/MAIN_TABLES/` (R^2 of the debt ratio; full
tables in the CSV files).

Removing leaky features and validating on unseen firms lowers the apparent accuracy sharply
(`table1_benchmark_r2.csv`):

| Model | Original features, random 80/20 | Strict features, firm-grouped 5x5 CV, mean (s.d.) | Strict, temporal |
|---|---|---|---|
| LightGBM | 0.895 | 0.082 (0.162) | 0.312 |
| XGBoost | 0.892 | 0.070 (0.156) | 0.318 |
| CatBoost | 0.871 | 0.125 (0.146) | 0.326 |
| Random forest | 0.863 | 0.045 (0.182) | 0.325 |
| Ridge | 0.695 | 0.180 (0.108) | 0.235 |

Generative models compared on the synthetic panel (`table2_generative.csv`; mean ± s.d. over seeds;
CVR = share of rows violating at least one accounting constraint):

| Generator | KS mean | CVR (any) | Logistic discriminator AUC |
|---|---|---|---|
| B1 noise baseline (original mechanism) | 0.136 ± 0.002 | 0.656 ± 0.011 | 0.563 ± 0.006 |
| CTGAN, 300 epochs | 0.210 ± 0.003 | 1.000 ± 0.000 | 0.929 ± 0.016 |
| CTGAN + projection, 300 epochs | 0.198 ± 0.011 | 0.608 ± 0.116 | 0.912 ± 0.006 |
| TVAE, 300 epochs | 0.092 ± 0.008 | 0.999 ± 0.000 | 0.650 ± 0.002 |
| TVAE + projection, 300 epochs | 0.098 ± 0.009 | 0.347 ± 0.048 | 0.694 ± 0.014 |

ROA coefficient on the debt ratio (`table3_roa_coefficient.csv`):

| Sample and specification | ROA coef. | SE | 95% CI |
|---|---|---|---|
| Synthetic panel, pooled OLS, original regressors, firm cluster | 0.1388 | 0.0141 | [0.111, 0.167] |
| Synthetic panel, firm FE, strict regressors, firm cluster | 0.0019 | 0.0030 | [-0.004, 0.008] |
| Real 46-firm sample, firm + year FE, spec A | -0.3124 | 0.0942 | [-0.499, -0.126] |

## 5. Data

| Data | In this repository | How to obtain |
|---|---|---|
| Reference synthetic panel | `results/reference_synthetic_panel_seed42.csv.gz`; also regenerated exactly by `common.generate_panel(seed=42)` | runs automatically |
| Yahoo Finance statements, 46 S&P-100 firms | not included (Yahoo's terms restrict redistribution) | `python data/download_recent_data.py`, or set `GENFIN_YAHOO_DIR` |
| SEC XBRL 10-K sample (Datavidence) | not included (vendor file) | place the file in `data/sec_datavidence/` or set `GENFIN_SEC_FILE`; see `data/sec_datavidence/README.md` |

Yahoo Finance returns only the latest four to five annual statements, so a new download covers
later fiscal years and the real-sample numbers will differ from the included results. The
synthetic-panel experiments do not depend on the download.

## 6. Repository layout

```
GenFin-Hybrid/
  README.md  LICENSE  CITATION.cff  ENVIRONMENT.md  SECURITY.md  .gitignore
  requirements.txt            pinned direct dependencies
  requirements-freeze.txt     full pip freeze of the tested environment
  code/                       config.py, common.py, test_common.py, e1-e7, build_master.py,
                              figstyle.py, final_figures_main.py, final_figures_si_a.py, final_figures_si_b.py
  code_sec/                   sec_data.py, e8_sec_validation.py
  scripts/                    run_all.sh, run_all.ps1, smoke_test.py
  data/                       download_recent_data.py; recent_data/ and sec_datavidence/ (empty, READMEs)
  generative/baselines/       earlier comparison scripts (read by E4)
  outputs_original/           earlier result files (read by E1, E3, E4)
  results/                    figures/, reference_tables/, figure_data/, reference synthetic panel
```

## 7. Licence and citation

The code is released under the MIT licence (`LICENSE`). Third-party data keep their own terms.
See `CITATION.cff` for how to cite this software.

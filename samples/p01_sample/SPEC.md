---
model_type: LME
target: units
grouping: superbrand
period: weekly
builds_upon: p00
---

# p01 — Panel per Brand, Sales

## What Is It

A panel LME model predicting weekly sales volume (units) for 3 brands — brand_1, brand_2, brand_3 — over 2019–2030 (~620 weeks × 3 brands = ~1,860 observations). Brands share global media parameters (adstock decay, saturation) but have brand-specific intercepts via random effects.

## What Is Special

- First **panel model** — introduces `superbrand` as the grouping variable and LME estimation
- **Price volatility transformation** (`adhoc_unparam_fin`): computes absolute and relative price deviations from market average
- Media variables use distribution-index interactions (`:` patsy syntax) — e.g., `brnd_tv_ots_clip_softmax_lag_adstockp:di_soph_ssn_median_sm_softmax`
- True underlying coefficients are defined in `overwrite_target.py` with per-brand intercepts (1200, 800, 600)
- Full pipeline includes **optimization** (COBYLA) to tune outer contour parameters

Builds upon p00 by adding panel structure, mixed effects, and parameterized transformations.

## File Structure

```
p01_sample/
├── code_py/
│   ├── adhoc/
│   │   ├── model.py              # adhoc_unparam_fin (price volatility)
│   │   ├── reporting.py          # reporting helpers
│   │   └── translators.py        # variable name translators
│   ├── model_data/
│   │   ├── model_conf.xlsx       # model configuration
│   │   └── options/              # budget options, seasonality
│   ├── scripts/
│   │   ├── run_test.py           # full pipeline test (transform → optimize → fit → metrics)
│   │   └── reporting.py          # PPTX report generation
│   └── _generate_scripts/
│       ├── generate_data.py       # synthetic data: 3 brands, weekly, media + seasonality + holidays
│       ├── overwrite_target.py    # overwrite units with known coefficients + noise
│       └── create_model_conf.py   # create model_conf.xlsx programmatically
├── data/
│   └── data_processed/
│       └── data_for_sample_p01.xlsx
└── output/                        # generated reports
```

## How To Work With It

### Pipeline Test (`run_test.py`)

```bash
cd samples/p01_sample
python code_py/scripts/run_test.py
```

Runs the full 6-step pipeline: load data → create model → transform (set_start) → optimize (COBYLA, 200 iters) → transform (final) → fit → evaluate. Prints R², MAPE, and per-brand metrics.

### Interactive Exploration (`model.ipynb`)

The notebook walks through the pipeline in stages, with **feature guards** (booleans) controlling which sections execute:

| Guard | Default | Controls |
|-------|---------|----------|
| `if_opt` | `False` | Optimization step (set `True` to tune params) |
| `cur_algo` | `'l_cma'` | Optimizer choice: `'l_cobyla'`, `'l_cma'`, `'g_de'` |
| `if_visual` | `True` | Interactive Plotly charts (fit, decomposition, waterfall) |
| `if_save` | `False` | Save trained model to disk |

Notebook section flow (each is a markdown + code cell pair):

| # | Section | What It Does |
|---|---------|-------------|
| 1 | Set environment | Imports, `metrics_max/min` trackers, guards |
| 2 | How to set model | Reference instructions for model_conf.xlsx and data |
| 3 | Load data + transform | `pd.read_excel` → `fm.Model` → `fm.transform(set_start=True)` |
| 4 | Optimising (optional) | Guarded by `if_opt`: COBYLA / CMA-ES / DE tuning of outer contour params |
| 5 | Get results | `fm.transform(set_start=False)` → `fm.fit_predict()` → `fm.predict_ext()` → `fmr.ModelRep()` |
| 6 | Statistics and metrics | RMSE, scoring, R², MAPE, model summary, `fmr.metrics_group_table()`, VIF |
| 7 | Observed vs predicted | Guarded by `if_visual`: `fmr.fit_mult_plot_vol()` with `group_var=['superbrand']` |
| 8 | Decomposition + Waterfall | Guarded by `if_visual`: `fmr.extract_effect()` → `decompose_sub_plot()` + `waterfall_plot()` |
| 9 | Save or export | Guarded by `if_save`: `model.save()` |

```bash
cd samples/p01_sample/code_py/scripts
jupyter notebook model.ipynb
```

### Report Generation (`reporting.py`)

The reporting script uses a **phase-gated architecture**: boolean flags at the top control which phases execute. Set a flag to `True` to activate that phase.

| Flag | Typical | Phase | What It Does |
|------|---------|-------|-------------|
| — | — | **Load** | Loads saved model, budget options (`current` + `main`), `ModelRep` |
| — | — | **Option run** | Runs budget scenarios for 2025 and 2026 via `fmr.option_report()` |
| `if_visual` | `False` | **Visualizations** | R²/MAPE by brand, `fit_mult_plot_vol()`, decomposition, waterfall |
| `if_curves` | `False` | **Response curves** | Per-media response curves via `option_report_multi_post()` (multi-core) |
| `if_export` | `True` | **PPTX export** | Retro detail, curves (`export_curves`), option comparison table (`export_option_table`) |
| `if_save` | `False` | **Save** | Saves model with reporting metadata for dashboard |
| `if_debug` | `True` | **Debug** | Drops into `breakpoint()` at script end |

```bash
cd samples/p01_sample
python code_py/scripts/reporting.py
```

## How To Regenerate

```bash
cd samples/p01_sample
python code_py/_generate_scripts/generate_data.py
python code_py/_generate_scripts/overwrite_target.py
python code_py/_generate_scripts/create_model_conf.py
```

## Expected Results

| Metric | Expected Range |
|--------|---------------|
| R² train | > 0.80 |
| MAPE train | < 10% |
| MAPE test | < 15% |
| R² per brand | > 0.75 each |

## Prerequisites

- Python 3.10+
- `fermatrica`
- `fermatrica_rep`
- `fermatrica_utils`
- `openpyxl`, `pandas`, `numpy`, `statsmodels`

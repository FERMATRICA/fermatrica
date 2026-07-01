---
model_type: LME
target: units
grouping: bs_key
period: weekly
builds_upon: p01
---

# p02 — Panel per SKU, Sales

## What Is It

A panel LME model predicting weekly sales volume (units) at the **SKU level**. Three brands decompose into master names and form variants, producing 13 unique SKUs (bs_key). Data spans 2019–2030 weekly (~620 weeks × 13 SKUs = ~8,060 observations).

## What Is Special

- **SKU decomposition**: each brand has multiple masters, each master has multiple forms — the model captures SKU-level variation
- **Distribution index decomposition** (`adhoc_create_di2`): clones a distribution variable to support SKU-level allocation
- **All params are fixed** — no optimization step; the model uses pre-set parameter values from `model_conf.xlsx`
- Demonstrates the transform → fit pipeline when outer contour parameters are known a priori
- Uses `bs_key` (numeric SKU identifier) as the panel grouping variable

Builds upon p01 by adding multi-level grouping (brand → master → form → SKU) and fixed-parameter workflow.

## File Structure

```
p02_sample/
├── code_py/
│   ├── adhoc/
│   │   ├── model.py              # adhoc_unparam_fin, adhoc_create_di2
│   │   ├── reporting.py          # reporting helpers
│   │   └── translators.py        # variable name translators
│   ├── model_data/
│   │   ├── model_conf.xlsx       # model configuration (all params fixed=1)
│   │   └── options/              # budget options
│   ├── scripts/
│   │   ├── run_test.py           # pipeline test (no optimization — all params fixed)
│   │   └── reporting.py          # PPTX report generation
│   └── _generate_scripts/
│       ├── generate_data.py       # synthetic data: 3 brands × masters × forms = 13 SKUs
│       ├── overwrite_target.py    # overwrite units with known coefficients + noise
│       └── create_model_conf.py   # create model_conf.xlsx programmatically
├── data/
│   └── data_processed/
│       └── data_for_sample_p02.xlsx
└── output/
```

## How To Work With It

### Pipeline Test (`run_test.py`)

```bash
cd samples/p02_sample
python code_py/scripts/run_test.py
```

Runs transform → fit → evaluate with **all params fixed** (no optimization). Prints R², MAPE, and per-brand metrics.

### Interactive Exploration (`model.ipynb`)

Same structure as p01. The notebook walks through the pipeline in stages with feature guards:

| Guard | Default | Controls |
|-------|---------|----------|
| `if_opt` | `False` | Optimization (all params fixed — would do nothing) |
| `cur_algo` | `'l_cma'` | Optimizer choice (not used when `if_opt=False`) |
| `if_visual` | `True` | Interactive Plotly charts |
| `if_save` | `False` | Save trained model |

Notebook section flow (same as p01):

| # | Section | What It Does |
|---|---------|-------------|
| 1 | Set environment | Imports, guards |
| 2 | How to set model | Reference instructions |
| 3 | Load data + transform | `pd.read_excel` → `fm.Model` → `fm.transform(set_start=True)` |
| 4 | Optimising (optional) | Skipped (params fixed) |
| 5 | Get results | `fm.transform(set_start=False)` → `fm.fit_predict()` → `fm.predict_ext()` |
| 6 | Statistics and metrics | RMSE, scoring, R², MAPE, model summary, VIF |
| 7 | Observed vs predicted | Guarded by `if_visual`: fit plots per SKU |
| 8 | Decomposition + Waterfall | Guarded by `if_visual`: SKU-level decomposition |
| 9 | Save or export | Guarded by `if_save` |

```bash
cd samples/p02_sample/code_py/scripts
jupyter notebook model.ipynb
```

### Report Generation (`reporting.py`)

Phase-gated architecture (same as p01):

| Flag | Typical | Phase | What It Does |
|------|---------|-------|-------------|
| — | — | **Load** | Loads saved model, budget options, `ModelRep` |
| — | — | **Option run** | Runs budget scenarios for 2025 and 2026 |
| `if_visual` | `False` | **Visualizations** | R²/MAPE by brand, fit plots with `group_var=['superbrand']` |
| `if_curves` | `False` | **Response curves** | Per-media response curves |
| `if_export` | `True` | **PPTX export** | Retro detail, curves, option table |
| `if_save` | `False` | **Save** | Save for dashboard |
| `if_debug` | `True` | **Debug** | `breakpoint()` |

```bash
cd samples/p02_sample
python code_py/scripts/reporting.py
```

## How To Regenerate

```bash
cd samples/p02_sample
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

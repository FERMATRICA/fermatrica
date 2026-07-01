---
model_type: LME (chained)
target: aba → units
grouping: superbrand → bs_key
period: monthly → weekly
builds_upon: p03, p02
---

# p05 — Two-Level Panel: BHT → Sales

## What Is It

A **chained two-model system**: a BHT (Brand Health Tracking) model predicts monthly ABA per brand, then a Sales model predicts weekly units per SKU using the predicted ABA as an input variable. Three brands (brand_1, brand_2, brand_3) with 13 SKUs. Two separate datasets: monthly for BHT (~144 months × 3 brands), weekly for Sales (~620 weeks × 13 SKUs).

## What Is Special

- **Only sample with chained models** — the output of model 1 feeds into model 2
- **Two separate model configurations**: `model_data_bht/model_conf.xlsx` and `model_data_sales/model_conf.xlsx`
- **Two separate data files**: `data_for_sample_p05_bht.xlsx` (monthly) and `data_for_sample_p05_sales.xlsx` (weekly)
- **`pred_merge_bht()`** chaining function in `code_py/adhoc/reporting.py`: merges predicted ABA from the brand-month model into the SKU-week dataset by matching year+month+superbrand
- **Two Jupyter notebooks**: `model_bht.ipynb` and `model_sales.ipynb` for interactive exploration
- **Two reporting scripts**: `reporting_bht.py` and `reporting_sales.py`
- All three adhoc functions: `adhoc_create_di2`, `adhoc_bht_mult`, `adhoc_unparam_fin`

Builds upon p03 (BHT model with ABA target + BHT multiplier) and p02 (SKU decomposition + fixed params).

### Model 1: BHT (Brand Health Tracking)

| Property | Value |
|----------|-------|
| Model type | LME |
| Target | aba (percentage points) |
| Grouping | superbrand |
| Period | monthly |
| Config | `code_py/model_data_bht/model_conf.xlsx` |
| Data | `data/data_processed/data_for_sample_p05_bht.xlsx` |

### Model 2: Sales

| Property | Value |
|----------|-------|
| Model type | LME |
| Target | units |
| Grouping | bs_key |
| Period | weekly |
| Config | `code_py/model_data_sales/model_conf.xlsx` |
| Data | `data/data_processed/data_for_sample_p05_sales.xlsx` |
| Special input | `aba_predicted` — merged from model 1 predictions |

### Chaining Flow

```
Model 1 (BHT): brand × month
    │
    ▼
  predicted ABA per brand-month
    │
    ▼
  pred_merge_bht() — match by year+month+superbrand
    │
    ▼
  aba_predicted column in SKU-week dataset
    │
    ▼
Model 2 (Sales): SKU × week (with aba_predicted as RHS variable)
```

## File Structure

```
p05_sample/
├── code_py/
│   ├── adhoc/
│   │   ├── model.py              # adhoc_create_di2, adhoc_bht_mult, adhoc_unparam_fin
│   │   └── reporting.py          # pred_merge_bht() chaining function
│   ├── model_data_bht/
│   │   └── model_conf.xlsx       # BHT model configuration
│   ├── model_data_sales/
│   │   └── model_conf.xlsx       # Sales model configuration
│   ├── model_data/
│   │   └── options/              # shared budget options, seasonality
│   ├── scripts/
│   │   ├── run_test.py           # full two-model pipeline test
│   │   ├── reporting_bht.py      # BHT model PPTX report
│   │   ├── reporting_sales.py    # Sales model PPTX report
│   │   ├── model_bht.ipynb       # interactive BHT model notebook
│   │   └── model_sales.ipynb     # interactive Sales model notebook
│   └── _generate_scripts/
│       ├── generate_data.py       # generates BOTH datasets (monthly + weekly)
│       ├── overwrite_target.py    # overwrite both targets with known coefficients
│       ├── create_model_conf_bht.py   # create BHT model_conf.xlsx
│       ├── create_model_conf_sales.py # create Sales model_conf.xlsx
│       ├── create_options.py      # create shared options
│       └── create_notebook.py     # create Jupyter notebooks
├── data/
│   └── data_processed/
│       ├── data_for_sample_p05_bht.xlsx   # monthly BHT data
│       └── data_for_sample_p05_sales.xlsx # weekly Sales data
└── output/
```

## How To Work With It

### Pipeline Test (`run_test.py`)

```bash
cd samples/p05_sample
python code_py/scripts/run_test.py
```

Runs a **3-part sequential pipeline**:
1. **BHT model**: load monthly data → transform → optimize (COBYLA, 200 iters) → fit → evaluate → predict future aba → save
2. **Merge**: `pred_merge_bht()` — joins predicted ABA into weekly Sales data by year+month+superbrand
3. **Sales model**: load weekly data (with `aba_predicted`) → transform → optimize (COBYLA, 200 iters) → fit → evaluate → save

Prints R², MAPE per brand for both models.

### Interactive Exploration (`model_bht.ipynb` + `model_sales.ipynb`)

Two separate notebooks. Each follows the standard 9-section structure.

**`model_bht.ipynb`** — Monthly ABA per brand:

| Guard | Default | Controls |
|-------|---------|----------|
| `if_opt` | `False` | BHT optimization |
| `cur_algo` | `'l_cma'` | Optimizer choice |
| `if_visual` | `True` | Interactive charts |
| `if_save` | `True` | Save BHT model (needed by Sales model) |

```bash
cd samples/p05_sample/code_py/scripts
jupyter notebook model_bht.ipynb
```

**`model_sales.ipynb`** — Weekly units per SKU with aba_predicted:

```bash
cd samples/p05_sample/code_py/scripts
jupyter notebook model_sales.ipynb
```

### Report Generation (`reporting_bht.py` + `reporting_sales.py`)

Two separate reporting scripts, each with the standard phase-gated architecture.

**`reporting_bht.py`** — BHT model report:

| Flag | Typical | Phase | What It Does |
|------|---------|-------|-------------|
| — | — | **Load** | Loads BHT model (auto-detects latest in `model_data_bht/`), budget options, `ModelRep` |
| — | — | **Option run** | Budget scenarios via `fmr.option_report()` |
| `if_visual` | `False` | **Visualizations** | R²/MAPE by brand, fit plots |
| `if_curves` | `False` | **Response curves** | Per-media response curves |
| `if_export` | `True` | **PPTX export** | BHT report slides |
| `if_save` | `False` | **Save** | Save for dashboard |
| `if_debug` | `True` | **Debug** | `breakpoint()` |

```bash
cd samples/p05_sample
python code_py/scripts/reporting_bht.py
```

**`reporting_sales.py`** — Sales model report with BHT chaining:

| Flag | Typical | Phase | What It Does |
|------|---------|-------|-------------|
| — | — | **Load model 1** | Loads saved BHT model, generates `dt_bht_pred` |
| — | — | **Load model 2** | Loads saved Sales model |
| — | — | **Merge** | `pred_merge_bht()` — joins BHT predictions into Sales dataset |
| — | — | **Option run** | Runs budget scenarios using both models |
| `if_visual` | `False` | **Visualizations** | Per-brand metrics and charts |
| `if_curves` | `False` | **Response curves** | Response curves through both models |
| `if_export` | `True` | **PPTX export** | Sales report slides with BHT context |
| `if_save` | `False` | **Save** | Save for dashboard |
| `if_debug` | `True` | **Debug** | `breakpoint()` |

```bash
cd samples/p05_sample
python code_py/scripts/reporting_sales.py
```

## How To Regenerate

```bash
cd samples/p05_sample
python code_py/_generate_scripts/generate_data.py
python code_py/_generate_scripts/overwrite_target.py
python code_py/_generate_scripts/create_model_conf_bht.py
python code_py/_generate_scripts/create_model_conf_sales.py
```

## Expected Results

### Model 1: BHT

| Metric | Expected Range |
|--------|---------------|
| R² train | > 0.70 |
| MAPE train | < 15% |
| MAPE test | < 20% |
| R² per brand | > 0.65 each |

### Model 2: Sales

| Metric | Expected Range |
|--------|---------------|
| R² train | > 0.80 |
| MAPE train | < 10% |
| MAPE test | < 15% |
| R² per brand | > 0.75 each |

Note: The Sales model's performance depends on BHT model accuracy. If BHT predictions are poor, Sales R² may degrade.

## Prerequisites

- Python 3.10+
- `fermatrica`
- `fermatrica_rep`
- `fermatrica_utils`
- `openpyxl`, `pandas`, `numpy`, `statsmodels`
- `jupyter` (for notebooks only)

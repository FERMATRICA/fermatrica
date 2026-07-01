---
model_type: LME
target: aba
grouping: superbrand
period: monthly
builds_upon: p01
---

# p03 — Panel per Brand, Brand Awareness (ABA)

## What Is It

A panel LME model predicting **Aided Brand Awareness** (ABA) measured in percentage points. Three brands (brand_1, brand_2, brand_3) at **monthly** frequency over 2019–2030 (~144 months × 3 brands = ~432 observations). Base ABA levels: 65pp, 45pp, 35pp respectively.

## What Is Special

- First sample with a **non-sales target** — ABA is a brand health metric (percentage points), not volume
- **Monthly frequency** — unlike p00–p02 which are weekly; demonstrates different period handling
- **BHT multiplier** (`adhoc_bht_mult`): media effects diminish as ABA approaches its ceiling — `media_effect × ((1 − aba_lag) × (1 − offset) + offset)`. This captures diminishing returns as brand awareness saturates
- `adhoc_unparam_fin` for price volatility (same as p01/p02)
- Reports metrics in percentage points (RMSE in pp, not units)

Builds upon p01's panel-per-brand structure but uses ABA target and monthly cadence.

## File Structure

```
p03_sample/
├── code_py/
│   ├── adhoc/
│   │   ├── model.py              # adhoc_bht_mult, adhoc_unparam_fin
│   │   ├── reporting.py          # reporting helpers
│   │   └── translators.py        # variable name translators
│   ├── model_data/
│   │   ├── model_conf.xlsx       # model configuration
│   │   └── options/              # budget options, seasonality
│   ├── scripts/
│   │   ├── run_test.py           # full pipeline test (monthly ABA model)
│   │   └── reporting.py          # PPTX report generation
│   └── _generate_scripts/
│       ├── generate_data.py       # synthetic data: 3 brands, monthly, ABA base + media
│       ├── overwrite_target.py    # overwrite aba with known coefficients + noise
│       └── create_model_conf.py   # create model_conf.xlsx programmatically
├── data/
│   └── data_processed/
│       └── data_for_sample_p03.xlsx
└── output/
```

## How To Work With It

### Pipeline Test (`run_test.py`)

```bash
cd samples/p03_sample
python code_py/scripts/run_test.py
```

Runs the full pipeline for the **monthly ABA model**: transform → optimize (COBYLA, 200 iters) → transform → fit → evaluate. Prints R², MAPE, RMSE (in percentage points), and per-brand metrics.

### Interactive Exploration (`model.ipynb`)

Same structure as p01, adapted for monthly ABA data:

| Guard | Default | Controls |
|-------|---------|----------|
| `if_opt` | `False` | Optimization (set `True` to tune BHT multiplier params) |
| `cur_algo` | `'l_cma'` | Optimizer choice |
| `if_visual` | `True` | Interactive Plotly charts |
| `if_save` | `False` | Save trained model |

Notebook section flow (same 9 sections as p01):
1. Set environment → 2. How to set model → 3. Load data + transform → 4. Optimising (optional) → 5. Get results → 6. Statistics and metrics (target is `aba`, metrics in pp) → 7. Observed vs predicted → 8. Decomposition + Waterfall → 9. Save or export

```bash
cd samples/p03_sample/code_py/scripts
jupyter notebook model.ipynb
```

### Report Generation (`reporting.py`)

Phase-gated architecture (same as p01), with `period='month'` in visualizations to match monthly data:

| Flag | Typical | Phase | What It Does |
|------|---------|-------|-------------|
| — | — | **Load** | Loads saved model, budget options, `ModelRep` |
| — | — | **Option run** | Runs budget scenarios |
| `if_visual` | `False` | **Visualizations** | R²/MAPE by brand, fit plots (monthly period), decomposition, waterfall |
| `if_curves` | `False` | **Response curves** | Per-media response curves |
| `if_export` | `True` | **PPTX export** | Retro detail, curves, option table |
| `if_save` | `False` | **Save** | Save for dashboard |
| `if_debug` | `True` | **Debug** | `breakpoint()` |

```bash
cd samples/p03_sample
python code_py/scripts/reporting.py
```

## How To Regenerate

```bash
cd samples/p03_sample
python code_py/_generate_scripts/generate_data.py
python code_py/_generate_scripts/overwrite_target.py
python code_py/_generate_scripts/create_model_conf.py
```

## Expected Results

| Metric | Expected Range |
|--------|---------------|
| R² train | > 0.70 |
| MAPE train | < 20% |
| MAPE test | < 25% |
| RMSE all | < 5 pp |
| R² per brand | > 0.65 each |

Note: ABA models typically have lower R² and higher MAPE than sales models due to noisier brand health data.

## Prerequisites

- Python 3.10+
- `fermatrica`
- `fermatrica_rep`
- `fermatrica_utils`
- `openpyxl`, `pandas`, `numpy`, `statsmodels`

---
model_type: OLS
target: units
grouping: null
period: weekly
builds_upon: null
---

# p00 — Time Series Model

## What Is It

The simplest possible FERMATRICA model: a single time series with OLS estimation and no panel effects. One brand ("brand_x"), one target variable (units), weekly frequency, ~250 observations spanning 2013–2017 for training.

Used as the **integration test reference sample** — `tests/test_integration/` fixtures load this sample's data and config to validate the full pipeline.

## What Is Special

- Only sample using **pure OLS** (no mixed effects, no panel grouping)
- **Pre-generated data** — no `generate_data.py` script; data comes as-is
- No custom adhoc transformations (`code_py/adhoc/model.py` is empty)
- Serves as the **canonical minimal example** — the starting point for understanding FERMATRICA basics
- Saved trained models exist in `code_py/model_data/model_*/` for quick loading

## File Structure

```
p00_sample/
├── code_py/
│   ├── adhoc/
│   │   ├── model.py              # empty (no custom transformations)
│   │   ├── reporting.py          # reporting helper functions
│   │   └── translators.py        # variable name translators for reports
│   ├── model_data/
│   │   ├── model_conf.xlsx       # model configuration (7 sheets)
│   │   ├── options/              # budget options
│   │   └── model_*/              # saved trained models (timestamped dirs)
│   └── scripts/
│       └── reporting.py          # PPTX report generation
├── data/
│   └── data_processed/
│       └── data_for_sample_p00_w.xlsx  # input data (weekly)
└── output/                       # generated reports (PPTX)
```

## How To Work With It

### Pipeline Test (via integration tests)

p00 is the integration test reference sample — the full pipeline is exercised through pytest:

```bash
pytest tests/test_integration/test_p00_sample_integration.py -v
```

This validates: data loading, model creation, transform, fit, predict, metrics, save, and load.

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
| 4 | Optimising (optional) | Guarded by `if_opt`: COBYLA / CMA-ES / DE tuning |
| 5 | Get results | `fm.transform(set_start=False)` → `fm.fit_predict()` → `fm.predict_ext()` |
| 6 | Statistics and metrics | RMSE, scoring, R², MAPE, model summary, VIF table |
| 7 | Observed vs predicted | Guarded by `if_visual`: interactive `fit_mult_plot_vol()` chart |
| 8 | Decomposition + Waterfall | Guarded by `if_visual`: `extract_effect()` → contribution + waterfall charts |
| 9 | Save or export | Guarded by `if_save`: `model.save()` |

```bash
cd samples/p00_sample/code_py/scripts
jupyter notebook model.ipynb
```

### Report Generation (`reporting.py`)

The reporting script uses a **phase-gated architecture**: boolean flags at the top control which phases execute.

| Flag | Typical | Phase | What It Does |
|------|---------|-------|-------------|
| — | — | **Load** | Loads saved model, budget options, `ModelRep` reporting environment |
| — | — | **Option run** | Runs budget scenarios via `fmr.option_report()` |
| `if_visual` | `False` | **Visualizations** | R²/MAPE by group, fit plots, decomposition, waterfall |
| `if_curves` | `False` | **Response curves** | Media response curves (expensive, multi-core) |
| `if_export` | `False` | **PPTX export** | Generates slides: retro detail, curves, option tables |
| `if_save` | `True` | **Save** | Saves model with reporting metadata for dashboard |
| `if_debug` | `True` | **Debug** | Drops into `breakpoint()` at script end |

```bash
cd samples/p00_sample
python code_py/scripts/reporting.py
```

Load a saved model:

```python
import fermatrica as fm
model, ds, errors = fm.model_load_ext("code_py/model_data/<model_dir>/")
```

## How To Regenerate

This sample has **no generation scripts**. The data is pre-generated and not regenerable from scripts.

To reproduce the saved model state:
1. Load the pre-generated data: `data/data_processed/data_for_sample_p00_w.xlsx`
2. Load model config: `code_py/model_data/model_conf.xlsx`
3. Run the full pipeline (transform → fit → save)

## Expected Results

| Metric | Expected Range |
|--------|---------------|
| R² train | > 0.70 |
| MAPE train | < 15% |
| MAPE test | < 20% |
| RMSE | > 0 (finite) |
| Overall score | finite value |

## Prerequisites

- Python 3.10+
- `fermatrica`
- `fermatrica_rep` (for reporting only)
- `fermatrica_utils`
- `openpyxl`, `pandas`, `numpy`, `statsmodels`

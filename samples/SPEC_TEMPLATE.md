---
model_type: ""       # OLS | LME | LMEA | FE
target: ""           # column name of Y variable
grouping: ""         # column name for panel grouping, or null
period: ""           # weekly | monthly | daily
builds_upon: ""      # sample name this builds on, or null
---

# Sample XX — Short Title

## What Is It

[Model type, target variable, grouping dimension, data frequency, number of groups, time range]

## What Is Special

[Unique features vs other samples: transformation types, adhoc functions, target type, grouping logic]

Builds upon: [prior sample and which patterns/concepts are reused]

## File Structure

```
pXX_sample/
├── code_py/
│   ├── adhoc/
│   │   └── model.py          # [custom transformations, list functions]
│   ├── model_data/
│   │   ├── model_conf.xlsx   # [model configuration]
│   │   └── options/          # [budget options, seasonality]
│   ├── scripts/
│   │   ├── run_test.py       # [full pipeline test]
│   │   └── reporting.py      # [PPTX report generation]
│   └── _generate_scripts/
│       ├── generate_data.py       # [synthetic data generator]
│       ├── overwrite_target.py    # [overwrite target with known coefficients]
│       └── create_model_conf.py   # [programmatically create model_conf.xlsx]
├── data/
│   └── data_processed/
│       └── data_for_sample_pXX.xlsx  # [input data]
└── output/                  # [generated reports]
```

## How To Work With It

### Pipeline Test (`run_test.py`)

```bash
cd samples/pXX_sample
python code_py/scripts/run_test.py
```

Runs the full 6-step pipeline: load data → create model → transform (set_start) → optimize → transform (final) → fit → evaluate. Prints R², MAPE, and per-group metrics.

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
cd samples/pXX_sample/code_py/scripts
jupyter notebook model.ipynb
```

### Report Generation (`reporting.py`)

The reporting script has a **phase-gated architecture**: each major section is controlled by a boolean flag at the top of the script. Set a flag to `True` to activate that phase.

| Flag | Typical | Phase | What It Does |
|------|---------|-------|-------------|
| — | — | **Load** | Loads saved model, budget options, `ModelRep` reporting environment |
| — | — | **Option run** | Runs budget scenarios for target years via `fmr.option_report()` |
| `if_visual` | `False` | **Visualizations** | R²/MAPE by group, interactive fit plots, decomposition, waterfall |
| `if_curves` | `False` | **Response curves** | Media response curves via `option_report_multi_post()` (expensive, multi-core) |
| `if_export` | `True` | **PPTX export** | Generates slides: retro detail, curves, option comparison tables |
| `if_save` | `False` | **Save** | Saves model with reporting metadata for dashboard |
| `if_debug` | `True` | **Debug** | Drops into `breakpoint()` at script end for interactive inspection |

```bash
cd samples/pXX_sample
python code_py/scripts/reporting.py
```

## How To Regenerate

Ordered regeneration scripts:

```bash
cd samples/pXX_sample
python code_py/_generate_scripts/generate_data.py
python code_py/_generate_scripts/overwrite_target.py
python code_py/_generate_scripts/create_model_conf.py
```

## Expected Results

| Metric | Expected Range |
|--------|---------------|
| R² train | [TBD] |
| MAPE train | [TBD]% |
| MAPE test | [TBD]% |

## Prerequisites

- Python 3.10+
- `fermatrica` (core framework)
- `fermatrica_utils` (utility functions)
- `fermatrica_rep` (reporting — if used)
- `fermatrica_data` (data preprocessing — if used)
- `openpyxl`, `pandas`, `numpy`, `statsmodels`

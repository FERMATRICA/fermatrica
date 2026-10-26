---
name: fermatrica
description: FERMATRICA core framework expertise — model definition, transformations, optimization, evaluation, model_conf.xlsx configuration, code conventions, canonical workflow, testing procedures, and common pitfalls. Use when writing or reviewing code in the fermatrica package, building MMM models, or debugging the core pipeline.
---

# FERMATRICA Core Framework

FERMATRICA is an econometrics framework for building time series and panel models, with a focus on Marketing Mix Modeling (MMM). The framework separates model specification into **two contours**:

1. **Outer contour** — parameterized non-linear transformations applied to X and Y variables before/after the linear model: adstock decay, saturation curves (softmax, logistic, weibull), lags, price adjustments, multiplicative LHS decompositions.

2. **Inner contour** — a linear model (OLS or panel LME) fitted on the transformed variables via statsmodels.

## Multi-Repo Architecture

FERMATRICA is developed across three repositories. This repo (`fermatrica`) is the **core**.

| Repo | Purpose | Python package |
|------|---------|---------------|
| `fermatrica` (this repo) | Core: model definition, transformation, optimization, evaluation | `fermatrica` |
| `fermatrica_utils` | Utility functions (`StableClass`, `DotDict`, `groupby_eff`, `select_eff`) | `fermatrica_utils` |
| `fermatrica_rep` | Reporting: decomposition, waterfall charts, PPTX export | `fermatrica_rep` |

## Code Organization (This Repo)

```
fermatrica/
├── basics/          # Foundation: FermatricaError, params_to_dict, logger
│   └── basics.py
├── model/           # Core modeling (depends on basics)
│   ├── model.py           # Model class — links conf + obj, save/load
│   ├── model_conf.py      # Config from XLSX (params, RHS, LHS, setup, scoring sheets)
│   ├── model_obj.py       # Callable object container, pickling
│   ├── transform.py       # High-level transformation dispatch
│   ├── transform_fun.py   # Low-level vector operations (no Model dependency)
│   ├── lhs_fun.py         # LHS decomposition functions
│   └── predict.py         # Inner layer: OLS/LME fit + predict
├── evaluation/      # Metrics and scoring (depends on model)
│   ├── metrics.py          # 30+ statistical metrics (R², RMSE, MAPE, VIF, DW, poolability...)
│   └── scoring.py          # Composite weighted scoring for optimization
├── optim/           # Optimization algorithms (depends on model + evaluation)
│   ├── objective.py        # ObjectiveFunction class (scoring loop)
│   ├── locals.py           # COBYLA, BOBYQA, Subplex, RBF, CMA-ES
│   ├── globals.py          # PyGad GA, Optuna TPE
│   └── globals_deap.py     # DEAP differential evolution
├── tests/           # Unit + integration tests
├── docs/            # API reference (pdoc)
└── build/           # Build artifacts
```

**Dependency chain**: `basics` → `model` → `evaluation` → `optim`

## Local Configuration

Machine-specific settings are stored in `config.toml` (repo root, gitignored, created by each developer):

1. Copy `config_example.toml` to `config.toml`
2. Optionally set `[paths].samples` to a directory containing sample model projects used by integration tests
3. Set `[paths].python` to the Python executable path in your Conda environment
4. Set `[python].env_name` to your Conda environment name

The `[paths].python` field points to the Python interpreter the AI should invoke. When running Python commands, always prefix with this path:

```
C:\Users\...\anaconda3\envs\<env_name>\python.exe -m pytest tests/
```

The `[python].env_name` field is informational — the environment name corresponding to the interpreter.

Integration tests resolve the samples path from (highest to lowest):
- `--samples-path` pytest CLI flag
- `config.toml` `[paths].samples`

## Model Pipeline (Canonical Workflow)

```python
import fermatrica as fm

# 1. Load data
ds = pd.read_excel("data.xlsx", parse_dates=["date"])

# 2. Create model
model = fm.Model(path="model_conf.xlsx", adhoc_code=[code_py.adhoc.model], ds=ds)

# 3. Transform with start values
model = fm.transform(ds=ds, model=model, set_start=True, if_by_ref=True)

# 4. Optimize outer contour
model = fm.optimize_local_cobyla(ds, model, revert_score_sign=True, epochs=3, iters_epoch=300, error_score=1e12)

# 5. Transform + Fit
model = fm.transform(ds=ds, model=model, set_start=False, if_by_ref=True)
pred, pred_raw, model = fm.fit_predict(ds, model, if_fit=True, return_full=True)

# 6. Evaluate
r2 = fm.metrics.r_squared(ds["units"], pred)
mape = fm.metrics.mapef(ds["units"], pred)
```

## model_conf.xlsx — Configuration Reference

The Excel workbook has 7 sheets. All column names and sheet names must match exactly.

### Sheet: `setup` — Global model settings

| Key | Description |
|-----|-------------|
| `Y_var` | Target variable name (e.g., `"units"`, `"aba"`) |
| `model_type` | `"OLS"`, `"LME"`, `"LMEA"`, or `"FE"` |
| `period_var` | Period column name (e.g., `"week"`) |
| `price_var` | Price variable name (e.g., `"price_ret"`) |
| `conversion_var` | Conversion factor variable (or NaN) |
| `summary_type` | `"full"` or `"short"` |
| `bs_key` | Grouping variable(s) for LME (semicolon-separated: `"superbrand;market"`) |
| `fixed_effect_var` | Random effects variable (e.g., `"bs_key"`) |
| `lme_method` | LME optimization methods: `"powell"`, `"lbfgs"` (comma-separated) |
| `target_superbrand` | Primary brand for metrics focus |
| `custom_scoring` | Custom scoring formula |
| `subfolder_orig` | Output subfolder name |

### Sheet: `params` — Transformation parameters to optimize

Columns: `variable`, `fun`, `arg`, `lower`, `upper`, `value`, `type`, `fixed`, `if_active`, `index_vars`, `index_aggr`, `set_start`, `formula`

- `fun` — transformation function: `lag`, `adstock`, `adstockp`, `adstockpd`, `softmax`, `softmaxfull`, `logistic`, `dwbl`, `dwblp`, `mar`, `expm1`, `expm1scaled`, `scale`, `share`, `infl`, `age`, `price`, `coef`, `adbudg`, `gaussian`, `gompertz`, `power`, `arl`, `arlp`, `arld`, `adhoc_*` (custom), `fe_model_*` (auxiliary models), `follow_me` (parameter linkage)
- `fixed` — 1 = not optimized, kept at `value`
- `if_active` — 1 = active, 0 = skipped
- `index_vars` — for panel models, group-specific params (e.g., `"superbrand"`)

### Sheet: `RHS` — Right-hand side variables for linear model

Columns: `token`, `fixed`, `if_active`, `sign`, `sign_weight`, `signif_weight`, `signif_threshold`, `display_var`

- `token` — formula token. Use `:` for interactions: `"media_softmax:seasonality"`
- `sign` — expected sign (`1` or `-1`) for scoring penalties

### Sheet: `LHS` — Left-hand side multiplicative decomposition

Columns: `type`, `name`, `token`, `if_active`, `display_var`

### Sheet: `scoring` — Composite objective function

Columns: `metrics`, `if_invert`, `width`, `weight`, `if_active`

- `metrics` — key: `r_squared`, `r_squared_adjusted`, `rmse`, `rmse_bs_key`, `sign_simple`, `sign_t_value`, `sign_t_value_s`, `sign_t_value_st`, `t_value`, `p_value`, `durbin_watson_score`, `vif`
- `if_invert` — 1 = lower is better (RMSE), 0 = higher is better (R²)

### Sheet: `trans_path_df` — Variable-to-cost mapping

Columns: `variable`, `variable_fin`, `price`, `display_var`

## Key Concepts

### Transformation Chains

Media variables typically undergo a chain: `raw → softmax (saturation) → lag → adstockp (decay memory)`. Output columns follow `{variable}_{fun}` naming (e.g., `tv_ots → tv_ots_softmax → tv_ots_softmax_lag → tv_ots_softmax_lag_adstockp`).

### LHS vs RHS

- **RHS** transformations produce additive terms in the linear model formula.
- **LHS** transformations produce multiplicative factors applied to Y. This allows mixing additive (RHS) and multiplicative (LHS) factors in the same model without log-linear forms.

### Optimization

- The objective function runs the full loop: set params → transform → fit → score. Each iteration is expensive (2-30 seconds).
- `error_score` (default `1e12` or `-1e12`) is returned when model fitting fails for a given parameter set.
- `follow_me` params copy values from a leader parameter.

### Model Save/Load

```python
model.save()        # Creates timestamped dir: model_YYYY-MM-DD_HHMMSS/
model_loaded, ds, errors = fm.model_load_ext("path/to/model_dir/")
```

## Data Conventions

- `listed` column: `2` = train, `3` = test, `4` = future (NaN target)
- `superbrand` = brand identifier string
- `bs_key` = numeric panel group identifier
- `date` = datetime column
- Panel models: `model_type = "LME"`, `fixed_effect_var = "bs_key"`
- RHS interactions: `:` separator

## Code Conventions

### Imports

```python
# Standard library
import copy, logging, os, sys

# Third-party
import numpy as np
import pandas as pd
from line_profiler import profile

# fermatrica_utils
from fermatrica_utils import StableClass, DotDict

# fermatrica internal
from fermatrica.basics.basics import fermatrica_error, FermatricaError
import fermatrica.model.transform_fun as ftr
```

### Error handling

```python
fermatrica_error("Descriptive message about what went wrong")
```

Non-fatal issues: use the `fermatrica` logger at WARNING level.

### Type annotations

Use Python 3.10+ union syntax: `str | None`, `pd.DataFrame | None`. Not `Optional[str]`.

### StableClass

`Model` extends `StableClass` — no new attributes after `__init__`. Set all attributes during construction.

### In-place transforms

Use `if_by_ref=True` for performance on large datasets.

## Testing Conventions

### Running tests

```bash
pytest tests/
pytest tests/test_model/test_transform.py
pytest tests/test_integration/ -v
pytest tests/ --if-debug
```

### Test patterns

- **YAML-driven parametrization**: Test cases in `test_cases/*.yaml` with custom constructors
- **Error testing**: `exp_err` field — `pytest.raises(FermatricaError)` or `nullcontext()`
- **Integration tests**: Session-scoped fixtures load p00 sample data

### What to test when changing code

| Change location | What tests to run |
|----------------|-------------------|
| `fermatrica/model/transform_fun.py` | `test_transform_fun.py` |
| `fermatrica/model/transform.py` | `test_transform.py` |
| `fermatrica/model/lhs_fun.py` | `test_lhs_fun.py` |
| `fermatrica/model/model_conf.py` | `test_model_conf.py` |
| `fermatrica/evaluation/` | `test_metrics.py`, `test_scoring.py` |
| Any change affecting pipeline | `test_integration/` |

## Common Pitfalls

1. **Reordering pipeline**: Always transform(set_start=True) → optimize → transform(set_start=False) → fit.
2. **Modifying params directly**: Changes require full re-transform.
3. **Breaking variable_fin naming**: Output columns must follow `{variable}_{fun}` pattern.
4. **Forgetting if_by_ref=True**: Doubles memory for large datasets without it.
5. **LME bs_key**: Must uniquely identify each panel unit.

# FERMATRICA — AI Instructions

> **⚠️ Maintenance reminder**: If you add new modules, rename directories, change conventions, or modify the project structure, update this file. The AI does not auto-detect staleness. Run `/fermatrica-sync-instructions` to have the AI review and update this file against the current codebase.

## Framework Overview

FERMATRICA is an econometrics framework for building time series and panel models, with a focus on Marketing Mix Modeling (MMM). The framework separates model specification into **two contours**:

1. **Outer contour** — parameterized non-linear transformations applied to X and Y variables before/after the linear model. This is effectively parameterized feature engineering: adstock decay, saturation curves (softmax, logistic, weibull), lags, price adjustments, multiplicative LHS decompositions.

2. **Inner contour** — a linear model (OLS or panel LME) fitted on the transformed variables. Since non-linearity is handled by the outer contour, the inner contour remains simple, fast, and statistically interpretable via statsmodels.

The outer contour parameters (adstock decay rates, saturation midpoints, etc.) are tuned by optimization algorithms (COBYLA, BOBYQA, DE, GA, TPE) that minimize a composite scoring function combining R², RMSE, sign correctness, t-values, VIF, and Durbin-Watson.

## Multi-Repo Architecture

FERMATRICA is developed across three repositories. This repo (`fermatrica`) is the **core**. When working on code that depends on or affects the other repos, check their READMEs for details. Cross-repo changes require coordination.

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

The `[paths].python` field points to the Python interpreter the AI should invoke. When running Python commands (`python ...`, `pytest ...`), always prefix with this path:

```
C:\Users\...\anaconda3\envs\<env_name>\python.exe -m pytest tests/
```

The `[python].env_name` field is informational — the environment name corresponding to the interpreter.

Integration tests resolve the samples path from (highest to lowest):
- `--samples-path` pytest CLI flag
- `config.toml` `[paths].samples`

## Model Pipeline (Canonical Workflow)

Every model follows this 6-step pipeline. This is the invariant pattern — do not reorder or skip steps:

```python
import fermatrica as fm

# 1. LOAD DATA — pandas DataFrame with date column, target var, media vars, control vars
ds = pd.read_excel("data.xlsx", parse_dates=["date"])

# 2. CREATE MODEL — from model_conf.xlsx with optional adhoc code
model = fm.Model(path="code_py/model_data/model_conf.xlsx",
                 adhoc_code=[code_py.adhoc.model],
                 ds=ds)

# 3. TRANSFORM (with start values) — compute initial parameter values
model = fm.transform(ds=ds, model=model, set_start=True, if_by_ref=True)

# 4. OPTIMIZE — tune outer contour parameters (skip if all params are fixed)
model = fm.optimize_local_cobyla(ds, model, revert_score_sign=True,
                                  epochs=3, iters_epoch=300, error_score=1e12)

# 5. TRANSFORM (with tuned params) + FIT — run inner linear model
model = fm.transform(ds=ds, model=model, set_start=False, if_by_ref=True)
pred, pred_raw, model = fm.fit_predict(ds, model, if_fit=True, return_full=True)

# 6. EVALUATE — compute metrics, scoring
r2 = fm.metrics.r_squared(ds["units"], pred)
mape = fm.metrics.mapef(ds["units"], pred)
```

### Data conventions

- `listed` column splits data: `2` = train, `3` = test, `4` = future (NaN target)
- `superbrand` = brand identifier string (e.g., `"brand_1"`)
- `bs_key` = numeric group identifier for panel models
- `date` = datetime column
- For panel models: `model_type = "LME"`, `fixed_effect_var = "bs_key"`
- RHS interactions use `:` separator: `"media_var_softmax_lag:seasonality_share"`

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
| `custom_scoring` | Custom scoring formula (e.g., `"rmse_bs_key * ((sign_t_value_st + 1) ** 10)"`) |
| `subfolder_orig` | Output subfolder name (usually date-based) |

### Sheet: `params` — Transformation parameters to optimize

Columns: `variable`, `fun`, `arg`, `lower`, `upper`, `value`, `type`, `fixed`, `if_active`, `index_vars`, `index_aggr`, `set_start`, `formula`

- `variable` — input column name
- `fun` — transformation function: `lag`, `adstock`, `adstockp`, `adstockpd`, `softmax`, `softmaxfull`, `logistic`, `dwbl`, `dwblp`, `mar`, `expm1`, `expm1scaled`, `scale`, `share`, `infl`, `age`, `price`, `coef`, `adbudg`, `gaussian`, `gompertz`, `power`, `arl`, `arlp`, `arld`, `adhoc_*` (custom), `fe_model_*` (auxiliary models), `follow_me` (parameter linkage)
- `arg` — function argument name being optimized (e.g., `"decay"`, `"lambda"`, `"n"`)
- `lower` / `upper` — optimization bounds
- `value` — starting value
- `fixed` — 1 = not optimized, kept at `value`
- `if_active` — 1 = active, 0 = skipped
- `index_vars` — for panel models, group-specific params (e.g., `"superbrand"`)
- `set_start` / `formula` — for computing `value` from training data at first transform

### Sheet: `RHS` — Right-hand side variables for linear model

Columns: `token`, `fixed`, `if_active`, `sign`, `sign_weight`, `signif_weight`, `signif_threshold`, `display_var`

- `token` — formula token. Can use `:` for interactions: `"media_softmax:seasonality"`
- `sign` — expected sign (`1` or `-1`). Used in scoring — penalizes wrong-sign coefficients.
- `sign_weight` — penalty weight for sign violations
- `if_active` — 1 = include in model
- `display_var` — friendly name for reporting

### Sheet: `LHS` — Left-hand side multiplicative decomposition

Columns: `type`, `name`, `token`, `if_active`, `display_var`

- `token` — formula string evaluated against dataset to produce a multiplicative factor
- `Y = Y_transformed * LHS_token_1 * LHS_token_2 * ...` (back-transformation after prediction)

### Sheet: `scoring` — Composite objective function

Columns: `metrics`, `if_invert`, `width`, `weight`, `if_active`

- `metrics` — metric key: `r_squared`, `r_squared_adjusted`, `rmse`, `rmse_bs_key`, `sign_simple`, `sign_t_value`, `sign_t_value_s`, `sign_t_value_st`, `t_value`, `p_value`, `durbin_watson_score`, `vif`
- `if_invert` — 1 = lower is better (RMSE), 0 = higher is better (R²)
- `width` — bell curve width for score normalization
- `weight` — metric weight in composite score

### Sheet: `trans_path_df` — Variable-to-cost mapping

Columns: `variable`, `variable_fin`, `price`, `display_var`

Maps original variables through transformation chains to final model variables. Auto-generated if not provided.

## Key Concepts

### Transformation Chains

Media variables typically undergo a chain: `raw → softmax (saturation) → lag → adstockp (decay memory)`. Each step has its own params row. The framework processes them in dependency order based on `variable` → `variable_fin` naming: intermediate columns get named `{variable}_{fun}` (e.g., `tv_ots → tv_ots_softmax → tv_ots_softmax_lag → tv_ots_softmax_lag_adstockp`).

### LHS vs RHS

- **RHS** transformations produce additive terms in the linear model formula. After transformation, RHS variables enter the regression as independent variables.
- **LHS** transformations produce multiplicative factors applied to Y. This allows mixing additive (RHS) and multiplicative (LHS) factors in the same model without resorting to log-linear forms.

### Optimization

- The objective function runs the full loop: set params → transform → fit → score. Each iteration is expensive (2-30 seconds).
- All optimizers accept `revert_score_sign=True` (higher score = better model).
- `error_score` (default `1e12` or `-1e12`) is returned when model fitting fails for a given parameter set.
- `follow_me` params copy values from a leader parameter — used when the same value should apply to multiple transformations.

### Model Save/Load

```python
model.save()        # Creates timestamped dir: model_YYYY-MM-DD_HHMMSS/
                    # Contains: model_conf.xlsx, model_objects.pkl.lzma, dt_p.pkl.zstd

model_loaded, ds, errors = fm.model_load_ext("path/to/model_dir/")
```

## Sample Structure

A complete model project is self-contained:

```
p01_sample/
├── data/data_processed/data_for_sample_p01.xlsx   # Input data
├── code_py/
│   ├── adhoc/
│   │   └── model.py         # Custom transformation functions (adhoc_*)
│   ├── model_data/
│   │   ├── model_conf.xlsx   # Model definition (7 sheets)
│   │   └── options/          # Budget options, seasonality data
│   ├── scripts/
│   │   ├── run_test.py       # Full pipeline test (canonical usage example)
│   │   ├── reporting.py      # PPTX generation script
│   │   └── model.ipynb       # Interactive Jupyter notebook
│   └── _generate_scripts/
│       ├── generate_data.py       # Synthetic data generator
│       ├── overwrite_target.py    # Overwrite target with known coefficients + noise
│       └── create_model_conf.py   # Programmatically create model_conf.xlsx
└── output/                   # Generated reports (PPTX)
```

## Testing Conventions

### Running tests

```bash
# All tests
pytest tests/

# Specific module
pytest tests/test_model/test_transform.py

# Integration tests
pytest tests/test_integration/ -v

# With debug mode (breaks at assertion failures)
pytest tests/ --if-debug
```

### Test patterns

- **YAML-driven parametrization**: Test cases in `test_cases/*.yaml` with custom constructors (`!pd.DataFrame`, `!pd.Series`, `!np.array`). Use `parametrize_fun()` and `parametrize_class()` helpers from `tests/basics/basics.py`.
- **Error testing**: `exp_err` field — `pytest.raises(FermatricaError)` or `nullcontext()`
- **Integration tests**: Session-scoped fixtures in `tests/test_integration/conftest.py` load p00 sample data, create model, train, and compute metrics. Each test method verifies a specific invariant.
- **`if_debug` fixture**: When `True`, drops into `breakpoint()` on assertion failure. Set via `pytest --if-debug`.

### What to test when changing code

| Change location | What tests to run |
|----------------|-------------------|
| `fermatrica/model/transform_fun.py` | `test_transform_fun.py` |
| `fermatrica/model/transform.py` | `test_transform.py` |
| `fermatrica/model/lhs_fun.py` | `test_lhs_fun.py` |
| `fermatrica/model/model_conf.py` | `test_model_conf.py` |
| `fermatrica/evaluation/` | `test_metrics.py`, `test_scoring.py` |
| Any change affecting pipeline | `test_integration/` (full pipeline tests) |

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

Use the framework's error function, never raw `raise`:

```python
fermatrica_error("Descriptive message about what went wrong")
```

Non-fatal issues: use the `fermatrica` logger at WARNING level.

### Type annotations

Use Python 3.10+ union syntax: `str | None`, `pd.DataFrame | None`. Not `Optional[str]`.

### Performance profiling

Most functions are decorated with `@profile` from `line_profiler`. Leave these in place. They are no-ops when profiling is not active.

### Statsmodels dependency

The inner model uses `statsmodels.formula.api` (`smf.ols`, `smf.mixedlm`). All patsy formula syntax applies. RHS tokens with `:` become interaction terms (both main effects + interaction are auto-included by patsy unless prevented).

## Common Mistakes to Avoid

1. **Modifying model attributes after init**: `Model` extends `StableClass` — no new attributes can be added after `__init__`. Set all attributes during construction.

2. **Reordering pipeline steps**: Always transform(set_start=True) → optimize → transform(set_start=False) → fit. Do not skip `set_start` or run optimize before transform.

3. **Modifying params directly**: params DataFrame lives in `model.conf.params`. Changes to it require re-running the full pipeline from transform.

4. **Breaking the `variable_fin` naming convention**: Transformation output column names follow the pattern `{variable}_{fun}`. The framework auto-derives dependency order from this naming. Do not use arbitrary output names.

5. **Forgetting `if_by_ref=True`**: Transformations modify the dataset in-place for performance. Without this flag, a copy is made, doubling memory usage for large datasets.

6. **Not handling panel data groups correctly**: For LME models, `bs_key` must uniquely identify each panel unit. The `index_vars` column in params controls which transformations are computed per-group vs globally.

## Samples

Sample models are not shipped with this repository. Integration tests resolve an external samples directory via `--samples-path` or `[paths].samples` in `config.toml`.

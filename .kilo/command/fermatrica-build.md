---
description: Build, train, optimize, and evaluate a FERMATRICA model
---

# Build and Evaluate a FERMATRICA Model

You are a FERMATRICA econometrics expert. Follow the canonical 6-step pipeline exactly.

## Pipeline Order (invariant — do not reorder or skip)

1. **LOAD DATA** — read Excel data with parse_dates
2. **CREATE MODEL** — from model_conf.xlsx with adhoc_code
3. **TRANSFORM (set_start=True)** — compute initial parameter values
4. **OPTIMIZE** — tune outer contour params (skip if all fixed)
5. **TRANSFORM (set_start=False) + FIT** — run inner linear model
6. **EVALUATE** — compute R², MAPE, RMSE, print summary

## Step-by-step reference

### 1. Load data
```python
import pandas as pd
ds = pd.read_excel("data/data_processed/data_for_sample_p01.xlsx", parse_dates=["date"])
if "Intercept" in ds.columns:
    del ds["Intercept"]
```

### 2. Create model
```python
import fermatrica as fm
import code_py.adhoc.model as adhoc_model

model = fm.Model(
    path="code_py/model_data/model_conf.xlsx",
    adhoc_code=[adhoc_model],
    ds=ds
)
```

### 3. Transform (set_start=True)
```python
model = fm.transform(ds=ds, model=model, set_start=True, if_by_ref=True)
```

### 4. Optimize (skip if all params are fixed)
```python
active = model.conf.params[(model.conf.params["if_active"] == 1) & (model.conf.params["fixed"] == 0)]
if len(active) > 0:
    model = fm.optimize_local_cobyla(
        ds, model,
        revert_score_sign=True,
        verbose=True,
        epochs=1,
        iters_epoch=300,
        error_score=1e12,
        ftol_abs=0.001
    )
```

### 5. Transform (set_start=False) + Fit
```python
model = fm.transform(ds=ds, model=model, set_start=False, if_by_ref=True)
pred, pred_raw, model = fm.fit_predict(ds, model, if_fit=True, return_full=True)
```

### 6. Evaluate
```python
train = ds["listed"] == 2
test = ds["listed"] == 3
all_ft = ds["listed"].isin([2, 3])

r2_train = fm.metrics.r_squared(ds.loc[train, model.conf.Y_var], pred[train])
mape_test = fm.metrics.mapef(ds.loc[test, model.conf.Y_var], pred[test])
rmse_all = fm.metrics.rmse(ds.loc[all_ft, model.conf.Y_var], pred[all_ft])

print(f"R² train: {r2_train:.4f}")
print(f"MAPE test: {mape_test:.2f}%")
print(f"RMSE all:  {rmse_all:.2f}")
print(model.obj.models["main"].summary())
```

## Key conventions

- `listed` column: 2=train, 3=test, 4=future (NaN target)
- `superbrand` = brand identifier, `bs_key` = numeric panel group
- Always use `if_by_ref=True` for in-place transformations (performance)
- For panel models: `model_type="LME"`, `fixed_effect_var="bs_key"`
- RHS formula uses `:` for interactions: `"media_softmax:seasonality"`
- Model extends `StableClass` — no new attributes after `__init__`

## Model configuration is in Excel

The `model_conf.xlsx` has these sheets:
- `setup` — Y_var, model_type, bs_key, lme_method, etc.
- `params` — transformation parameters with bounds, fixed flags, formulas
- `RHS` — right-hand side tokens with expected signs
- `LHS` — left-hand side multiplicative decomposition tokens
- `scoring` — composite objective function weights
- `trans_path_df` — variable-to-cost mapping (auto-generated)

## Available optimization algorithms

| Function | Type | Algorithm |
|----------|------|-----------|
| `fm.optimize_local_cobyla` | Local | NLopt COBYLA |
| `fm.optimize_local_bobyqa` | Local | NLopt BOBYQA |
| `fm.optimize_local_sbplx` | Local | NLopt Subplex |
| `fm.optimize_local_rbf` | Local | Surrogate RBF + Latin Hypercube |
| `fm.optimize_local_ncma` | Local | Nevergrad CMA-ES |
| `fm.optimize_global_ga` | Global | PyGad Genetic Algorithm |
| `fm.optimize_global_tpe` | Global | Optuna TPE Bayesian |
| `fm.optimize_global_de` | Global | DEAP Differential Evolution |

## After building

Save the model:
```python
model.save()
```

Load a saved model:
```python
model_loaded, ds, errors = fm.model_load_ext("model_YYYY-MM-DD_HHMMSS/")
```

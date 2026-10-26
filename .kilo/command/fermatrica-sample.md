---
description: Create a new FERMATRICA sample model (data, config, scripts)
---

# Create a New FERMATRICA Sample or Model

You are a FERMATRICA framework expert. Create complete, self-contained sample models following the established patterns.

## Reference implementation

The canonical sample is `p01_sample` (each sample is a self-contained project). Every new sample should mirror this structure:

```
pXX_sample/
├── __init__.py
├── data/data_processed/
│   └── data_for_sample_pXX.xlsx
├── code_py/
│   ├── __init__.py
│   ├── adhoc/
│   │   ├── __init__.py
│   │   ├── model.py          # Custom transform functions (adhoc_*)
│   │   ├── reporting.py      # Custom reporting functions
│   │   └── translators.py    # Custom translators
│   ├── model_data/
│   │   ├── model_conf.xlsx   # Model definition (7 sheets: setup, params, RHS, LHS, scoring, trans_path_df)
│   │   └── options/
│   │       ├── options.xlsx
│   │       └── seasonality.xlsx
│   ├── scripts/
│   │   ├── run_test.py       # Full pipeline test
│   │   ├── reporting.py      # PPTX generation
│   │   └── model.ipynb       # Interactive notebook
│   └── _generate_scripts/
│       ├── generate_data.py       # Synthetic data generator
│       ├── overwrite_target.py    # Overwrite target with known coefficients + noise
│       └── create_model_conf.py   # Programmatically create model_conf.xlsx
└── output/
```

## Step 1: Create directory structure

Mirror the structure above exactly. Copy `__init__.py` files with proper sub-package imports.

## Step 2: Generate synthetic data (`generate_data.py`)

Generate weekly time series with:
- Multiple brands (e.g., brand_1, brand_2, brand_3) using `superbrand` column
- **Target variable**: `units` (sales volume) or `aba` (brand awareness [0-100]%)
- **Media channels**: TV OTS (brnd_tv_ots_clip), OOH (brnd_ots_ooh), Digital (brnd_cl_dgtind_perform, display, video, special, indoor)
- **Other variables**: e_com, pos, pr_costs_sm, price_ret, distribution indices, seasonality, holidays
- **Column naming**: follow the canonical sample's naming exactly
- **Data columns**: date, superbrand, bs_key, week, listed (2=train, 3=test, 4=future), plus all variables
- **Effect pattern**: base volume × exp(∑media_contributions) / lhs_factors × scale + noise
- **Listed split**: e.g., train through 2023, test 2024, future 2025+

## Step 3: Overwrite target with known coefficients (`overwrite_target.py`)

This is critical for validation — it ensures the model has a known "ground truth":

```python
# 1. Run fm.transform() to get all transformed variables
# 2. Apply known true coefficients to RHS transformed variables
# 3. Divide by LHS decomposition factors (seasonality, holidays, price, volatility)
# 4. Add multiplicative noise: exp(np.random.randn(N) * noise_std)
# 5. Set future listed==4 target to NaN
```

## Step 4: Create model_conf.xlsx programmatically (`create_model_conf.py`)

Use `openpyxl.Workbook` to create the 7-sheet Excel file:

### setup sheet
```python
keys = ["Y_var", "model_type", "period_var", "price_var", "conversion_var",
        "summary_type", "bs_key", "fixed_effect_var", "lme_method",
        "target_superbrand", "custom_scoring", "subfolder_orig"]
values = ["units", "LME", "week", "price_ret", None,
          "short", "superbrand", "bs_key", "powell, lbfgs",
          "brand_1", "rmse_bs_key * ((sign_t_value_st + 1) ** 10)", "20260617"]
```

### params sheet
Define transformation chains for each media variable. Typical pattern:
- **Softmax saturation**: `lambda` (steepness), `avg` (midpoint from train data), `std` (from train data)
- **Lag**: `n=0` (no lag, adjustable during optimization)
- **Adstock decay**: `decay` (memory retention, bounded 0-1)
- **Seasonality share**: `share` (weight between regions/channels)
- **Price MAR**: `decay` (moving average retention)
- **LHS coefficients**: `coef` type with fixed/maybe-optimized flags

Key column values:
- `fixed`: 1 for computed values (avg/std from training data), 0 for optimizable
- `if_active`: 1 to include, 0 to skip
- `index_vars`: `"superbrand"` for per-brand parameters
- `set_start`: 1 for values computed via formula from training data
- `formula`: Python expression evaluated against the dataset

### RHS sheet
List all transformed variables that enter the linear model as additive terms:
```
token = "brnd_tv_ots_clip_softmax_lag_adstockp:di_soph_ssn_median_sm_softmax"
sign = 1 (positive expected effect)
```

Use `:` to create interaction terms with seasonality or other modifiers.

### LHS sheet
Multiplicative factors applied to Y:
```python
{"name": "ssn_mrk", "token": "(coef.tt_pk_tr_ssn + 1) / (coef.tt_pk_tr_ssn * ds.tt_pk_tr_ssn + 1)"}
{"name": "hdays",    "token": "1 + ds.hdays * coef.hdays"}
{"name": "price",    "token": "1 + coef.price * ds.price_ret_rel_market_mar_softmax"}
```

### scoring sheet
```python
{"metrics": "r_squared",          "if_invert": 0, "width": 1.0, "weight": 1.0, "if_active": 1}
{"metrics": "rmse_bs_key",        "if_invert": 1, "width": 1.0, "weight": 1.0, "if_active": 1}
{"metrics": "sign_t_value_st",    "if_invert": 0, "width": 1.0, "weight": 1.0, "if_active": 1}
```

### trans_path_df sheet
Auto-generated if omitted — maps `variable` through `variable_fin` chain.

## Step 5: Create adhoc model functions (`code_py/adhoc/model.py`)

For custom transformations referenced as `adhoc_*` in params:
```python
def adhoc_function_name(ds, params_subset, index_vars=None):
    params_dict = fermatrica.basics.basics.params_to_dict(params_subset)
    # Apply custom transformation
    ds["new_column"] = ds["input_column"].array * params_dict["param_name"]
```

## Step 6: Create run_test.py

Follow the canonical `run_test.py` exactly (load → model → transform → optimize → fit → metrics). The pipeline is invariant.

## Step 7: Create model.ipynb

Copy the canonical sample notebook and update paths. The notebook should allow interactive modeling with all consumer-facing cells (data inspection, transformation, optimization, fit, decomposition plots, waterfall charts).

## Key design principles

1. **Transformation chain naming**: `{variable}_{fun}` — e.g., `tv_ots → tv_ots_softmax → tv_ots_softmax_lag → tv_ots_softmax_lag_adstockp`
2. **All params for a transformation must share `if_active` status** — either all active or all inactive
3. **`set_start=1` implies `fixed=1`** — start values are not optimized
4. **`follow_me` params auto-fixed** — they copy from a leader parameter
5. **Bounds must contain start values** — validation enforced by ModelConf
6. **Data split convention**: `listed` column: 2=train, 3=test, 4=future (NaN target)

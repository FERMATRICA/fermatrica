---
model_type: LME
target: units
grouping: master
period: weekly
builds_upon: p01
---

# p04 — Panel per Region, Sales

## What Is It

A panel LME model predicting weekly sales volume (units) for 4 **geographic regions**: Moscow, Paris, Warsaw, Cairo. Data spans 2019–2030 weekly (~620 weeks × 4 regions = ~2,480 observations). Each region has its own base price and media multiplier.

## What Is Special

- Grouping by **geographic region** (`master` column) — not brand like p01/p03, not SKU like p02
- **Regional media prices** — each region has different cost-per-media-unit in `options/media_prices`
- **Option/budget scenario calculations** — the `run_test.py` includes a complete budget optimization test using `fermatrica_rep.option_report()`
- Simplest adhoc: only `adhoc_unparam_fin` (price volatility) — no other custom transformations needed
- Demonstrates how the same framework works for geographic panels vs brand panels

Builds upon p01's panel-per-group structure but applies it to regions with regional pricing.

## File Structure

```
p04_sample/
├── code_py/
│   ├── adhoc/
│   │   ├── model.py              # adhoc_unparam_fin (price volatility)
│   │   ├── reporting.py          # reporting helpers
│   │   └── translators.py        # variable name translators
│   ├── model_data/
│   │   ├── model_conf.xlsx       # model configuration
│   │   └── options/
│   │       ├── options.xlsx      # budget options + media_prices + translation
│   │       └── seasonality.xlsx  # seasonal patterns per region
│   ├── scripts/
│   │   ├── run_test.py           # pipeline test + option/budget scenario test
│   │   └── reporting.py          # PPTX report generation
│   └── _generate_scripts/
│       ├── generate_data.py       # synthetic data: 4 regions, weekly, campaign spikes
│       ├── overwrite_target.py    # overwrite units with known coefficients + noise
│       └── create_model_conf.py   # create model_conf.xlsx programmatically
├── data/
│   └── data_processed/
│       └── data_for_sample_p04.xlsx
└── output/
```

## How To Work With It

### Pipeline Test (`run_test.py`)

```bash
cd samples/p04_sample
python code_py/scripts/run_test.py
```

Runs full pipeline + **option/budget scenario test**: transform → optimize (COBYLA, 300 iters) → transform → fit → evaluate → option calculation per region. Prints R², MAPE, per-region metrics, and option test PASS/FAIL.

### Interactive Exploration (`model.ipynb`)

Same structure as p01, adapted for regional (`master`) grouping:

| Guard | Default | Controls |
|-------|---------|----------|
| `if_opt` | `False` | Optimization |
| `cur_algo` | `'l_cma'` | Optimizer choice |
| `if_visual` | `True` | Interactive Plotly charts |
| `if_save` | `False` | Save trained model |

Notebook section flow (same 9 sections as p01):
1. Set environment → 2. How to set model → 3. Load data + transform → 4. Optimising (optional) → 5. Get results → 6. Statistics and metrics (per `master`) → 7. Observed vs predicted → 8. Decomposition + Waterfall → 9. Save or export

```bash
cd samples/p04_sample/code_py/scripts
jupyter notebook model.ipynb
```

### Report Generation (`reporting.py`)

Phase-gated architecture with a **per-region loop** in the option run phase. Each region (Moscow, Paris, Warsaw, Cairo) has its own budget options and media prices.

| Flag | Typical | Phase | What It Does |
|------|---------|-------|-------------|
| — | — | **Load** | Loads saved model, per-master `options_m` dict, per-master `media_prices` |
| — | — | **Option run** | Loops over `for ms in masters:` — runs `fmr.option_report()` per region with per-master `trans_path_df` |
| `if_visual` | `False` | **Visualizations** | R²/MAPE by region, fit plots with `group_var=['superbrand', 'master']` |
| `if_curves` | `False` | **Response curves** | Per-media response curves |
| `if_export` | `False` | **PPTX export** | Retro detail, curves, option table |
| `if_save` | `False` | **Save** | Save for dashboard |
| `if_debug` | `True` | **Debug** | `breakpoint()` |

```bash
cd samples/p04_sample
python code_py/scripts/reporting.py
```

## How To Regenerate

```bash
cd samples/p04_sample
python code_py/_generate_scripts/generate_data.py
python code_py/_generate_scripts/overwrite_target.py
python code_py/_generate_scripts/create_model_conf.py
```

## Expected Results

| Metric | Expected Range |
|--------|---------------|
| R² train | > 0.75 |
| MAPE train | < 15% |
| MAPE test | < 20% |
| R² per region | > 0.70 each |
| Option test | PASS (total_pred > 0) |

## Prerequisites

- Python 3.10+
- `fermatrica`
- `fermatrica_rep`
- `fermatrica_utils`
- `openpyxl`, `pandas`, `numpy`, `statsmodels`

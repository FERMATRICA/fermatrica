"""
Step 4: Overwrite target variables for both models in a two-level pipeline.

1. Load BHT dataset → create model_1 → transform → generate ABA from transformed RHS
2. Merge generated ABA into sales dataset as aba_predicted
3. Load sales dataset → create model_2 → transform → generate units from transformed RHS
"""
import os
import shutil
import sys
import tempfile
import warnings
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
p05_root = os.path.join(_SCRIPT_DIR, '..', '..')
os.chdir(p05_root)
sys.path.insert(0, p05_root)

import fermatrica as fm
import code_py.adhoc.model

warnings.simplefilter(action='ignore')
np.random.seed(42)

# ============================================================
# STEP 1: BHT model — generate ABA
# ============================================================

dt_bht = pd.read_excel('data/data_processed/data_for_sample_p05_bht.xlsx', parse_dates=['date'])
raw_cols_bht = dt_bht.columns.tolist()
dt_bht_orig = dt_bht.copy()

pth_bht = os.path.join('code_py', 'model_data_bht', 'model_conf.xlsx')
model_1 = fm.Model(path=pth_bht, adhoc_code=[code_py.adhoc.model], ds=dt_bht)
model_1 = fm.transform(ds=dt_bht, model=model_1, set_start=True, if_by_ref=True)

# Per-superbrand intercepts
superbrand_intercepts = {
    'brand_1': 15.0,
    'brand_2': 20.0,
    'brand_3': 30.0,
}

true_coefs_bht = {
    'di_soph_ssn_median_sm_softmax': 12000.0,
    'brnd_tv_ots_clip_softmax_bht_lag_adstockp': 15000.0,
    'brnd_ots_ooh_softmax_bht_lag_adstockp': 2000.0,
    'brnd_cl_dgtind_display_lag_softmax_adstockp': 1500.0,
    'brnd_cl_dgtind_video_lag_softmax_adstockp': 1500.0,
    'brnd_cl_dgtind_special_lag_softmax_adstockp': 1200.0,
    'brnd_cl_dgtind_indoor_lag_softmax_adstockp': 1000.0,
    'pr_costs_sm_infl_lag_softmax_adstockp': 8000.0,
    'pos_infl_lag_softmax_adstockp': 5000.0,
}

true_lhs_bht = {
    'tt_pk_tr_ssn': 0.05,
    'hdays': -0.001,
    'price': 0.0,
    'price_volat_rel': 0.0,
}

y_linear = np.array([superbrand_intercepts.get(sb, 55.0) for sb in dt_bht['superbrand']])

model_rhs_1 = model_1.conf.model_rhs
N_bht = len(dt_bht)
for _, row in model_rhs_1.iterrows():
    token = row['token']
    coef = true_coefs_bht.get(token, 0.0)
    if token in dt_bht.columns:
        x = dt_bht[token].values.copy()
    elif ':' in token:
        parts = token.split(':')
        x = np.ones(N_bht)
        for p in parts:
            p = p.strip()
            if p in dt_bht.columns:
                x = x * dt_bht[p].values
    else:
        x = np.zeros(N_bht)
    y_linear += coef * x

ssn_factor = (true_lhs_bht['tt_pk_tr_ssn'] + 1) / (true_lhs_bht['tt_pk_tr_ssn'] * dt_bht['tt_pk_tr_ssn'].values + 1)
hdays_factor = 1 + dt_bht['hdays'].values * true_lhs_bht['hdays']
price_factor = 1 + true_lhs_bht['price'] * dt_bht['price_ret_rel_market_mar_softmax'].values
volat_factor = 1 + true_lhs_bht['price_volat_rel'] * dt_bht['price_ret_volat_rel_softmax'].values

aba_new = y_linear / ssn_factor / hdays_factor / price_factor / volat_factor
aba_new = aba_new + np.random.randn(N_bht) * 1.1
aba_new = aba_new / aba_new.max() * 70

dt_bht_orig['aba'] = np.round(aba_new, 2)
dt_bht_orig['aba_lag'] = dt_bht_orig.groupby('superbrand')['aba'].shift(1)
dt_bht_orig['aba_lag'] = dt_bht_orig.groupby('superbrand')['aba_lag'].bfill()

print(f'BHT ABA: min={dt_bht_orig["aba"].min():.2f}, max={dt_bht_orig["aba"].max():.2f}, mean={dt_bht_orig["aba"].mean():.2f}')

# Save BHT data
keep_cols_bht = [c for c in raw_cols_bht if c in dt_bht_orig.columns]
out_cols_bht = keep_cols_bht + ['aba', 'aba_lag']
out_cols_bht = [c for c in out_cols_bht if c in dt_bht_orig.columns]
dt_bht_out = dt_bht_orig[out_cols_bht].copy()

mask_bht_future = dt_bht_out['listed'] == 4
dt_bht_out.loc[mask_bht_future, 'aba'] = np.nan

tmp_bht = os.path.join(tempfile.gettempdir(), 'tmp_data_p05_bht.xlsx')
out_path_bht = os.path.join(p05_root, 'data', 'data_processed', 'data_for_sample_p05_bht.xlsx')
dt_bht_out.to_excel(tmp_bht, index=False)
shutil.copy2(tmp_bht, out_path_bht)
os.remove(tmp_bht)

# Save model_1 for later training
model_1.save(ds=dt_bht, path=os.path.join(os.getcwd(), 'code_py', 'model_data_bht'))

print(f'Saved BHT: {len(dt_bht_out)} rows')

# ============================================================
# STEP 2: Merge generated ABA into sales dataset
# ============================================================

dt_sales = pd.read_excel('data/data_processed/data_for_sample_p05_sales.xlsx', parse_dates=['date'])
raw_cols_sales = dt_sales.columns.tolist()
dt_sales_orig = dt_sales.copy()

# Extract ABA from the ORIGINAL generated BHT data (including future periods)
# Key: use dt_bht_orig which has full generated ABA, NOT dt_bht_out which has NaN for future
aba_source = dt_bht_orig[['superbrand', 'date', 'aba']].copy()
aba_source['year'] = aba_source['date'].dt.year
aba_source['month'] = aba_source['date'].dt.month

dt_sales['year'] = dt_sales['date'].dt.year
dt_sales['month'] = dt_sales['date'].dt.month

# Merge ABA (generated actual, not predicted) as aba_predicted
dt_sales = pd.merge(dt_sales, aba_source[['superbrand', 'year', 'month', 'aba']],
                    on=['superbrand', 'year', 'month'], how='left', suffixes=('', '_bht'))

# Overwrite aba_predicted with actual ABA from BHT
dt_sales['aba_predicted'] = dt_sales['aba'].fillna(dt_sales['aba_predicted'])

del dt_sales['year']
del dt_sales['month']
del dt_sales['aba']

# Update the original copy
dt_sales_orig['aba_predicted'] = dt_sales['aba_predicted'].values

print(f'Sales aba_predicted: min={dt_sales_orig["aba_predicted"].min():.2f}, max={dt_sales_orig["aba_predicted"].max():.2f}, mean={dt_sales_orig["aba_predicted"].mean():.2f}')

# ============================================================
# STEP 3: Sales model — generate units
# ============================================================

dt_sales_t = dt_sales.copy()
pth_sales = os.path.join('code_py', 'model_data_sales', 'model_conf.xlsx')
model_2 = fm.Model(path=pth_sales, adhoc_code=[code_py.adhoc.model], ds=dt_sales_t)
model_2 = fm.transform(ds=dt_sales_t, model=model_2, set_start=True, if_by_ref=True)

# Per-SKU intercepts
sku_intercepts = {
    1: 1000.0, 2: 5000.0, 3: 700.0, 4: 800.0, 5: 1200.0, 6: 300.0,
    7: 600.0,  8: 1600.0,  9: 800.0,  10: 900.0, 11: 750.0, 12: 600.0,
    13: 300.0, 14: 200.0, 15: 900.0,
    16: 1500.0, 17: 600.0, 18: 300.0, 19: 700.0, 20: 600.0, 21: 400.0,
}

true_coefs_sales = {
    'di_soph_ssn_median_sm_softmax': 8e3,
    'aba_predicted_softmax': 8e3,
    'brnd_tv_ots_clip_softmax_lag_adstockp:di_soph_ssn_median_sm_softmax:di_soph_ssn_median_sm_2_share_softmax': 3e4,
    'brnd_ots_ooh_softmax_lag_adstockp:di_soph_ssn_median_sm_softmax:di_soph_ssn_median_sm_2_share_softmax': 2e4,
    'brnd_cl_dgtind_perform_lag_softmax_adstockp:di_soph_ssn_median_sm_2_share_softmax': 8e3,
    'brnd_cl_dgtind_display_lag_softmax_adstockp:di_soph_ssn_median_sm_softmax:di_soph_ssn_median_sm_2_share_softmax': 1.2e4,
    'brnd_cl_dgtind_video_lag_softmax_adstockp:di_soph_ssn_median_sm_softmax:di_soph_ssn_median_sm_2_share_softmax': 1.2e4,
    'brnd_cl_dgtind_special_lag_softmax_adstockp:di_soph_ssn_median_sm_softmax:di_soph_ssn_median_sm_2_share_softmax': 1e4,
    'e_com_infl_lag_softmax_adstockp:di_soph_ssn_median_sm_2_share_softmax': 5e2,
    'pr_costs_sm_infl_lag_softmax_adstockp:di_soph_ssn_median_sm_softmax:di_soph_ssn_median_sm_2_share_softmax': 6e3,
    'pos_infl_lag_softmax_adstockp:di_soph_ssn_median_sm_softmax:di_soph_ssn_median_sm_2_share_softmax': 5e3,
}

true_lhs_sales = {
    'tt_pk_tr_ssn': 1.5,
    'hdays': -0.04,
    'price': -0.2,
    'price_volat_rel': -0.11,
}

model_rhs_2 = model_2.conf.model_rhs
N_sls = len(dt_sales_t)
y_linear = np.array([sku_intercepts.get(bs_key_val, 200.0) for bs_key_val in dt_sales_t['bs_key']])

for _, row in model_rhs_2.iterrows():
    token = row['token']
    coef = true_coefs_sales.get(token, 0.0)
    if token in dt_sales_t.columns:
        x = dt_sales_t[token].values.copy()
    elif ':' in token:
        parts = token.split(':')
        x = np.ones(N_sls)
        for p in parts:
            p = p.strip()
            if p in dt_sales_t.columns:
                x = x * dt_sales_t[p].values
    else:
        x = np.zeros(N_sls)
    y_linear += coef * x

# Apply LHS decomposition
ssn_factor_s = (true_lhs_sales['tt_pk_tr_ssn'] + 1) / (true_lhs_sales['tt_pk_tr_ssn'] * dt_sales_t['tt_pk_tr_ssn'].values + 1)
hdays_factor_s = 1 + dt_sales_t['hdays'].values * true_lhs_sales['hdays']
price_factor_s = 1 + true_lhs_sales['price'] * dt_sales_t['price_ret_rel_market_mar_softmax'].values
volat_factor_s = 1 + true_lhs_sales['price_volat_rel'] * dt_sales_t['price_ret_volat_rel_softmax'].values

units_new = y_linear / ssn_factor_s / hdays_factor_s / price_factor_s / volat_factor_s
units_new = units_new * np.exp(np.random.randn(N_sls) * 0.08)
units_new = np.round(units_new).astype(int)
units_new = np.clip(units_new, 50, None)

dt_sales_orig['units'] = units_new * 100.0

print(f'Sales units: min={dt_sales_orig["units"].min()}, max={dt_sales_orig["units"].max()}, mean={dt_sales_orig["units"].mean():.2f}')

# Save Sales data
keep_cols_sales = [c for c in raw_cols_sales if c in dt_sales_orig.columns]
out_cols_sales = keep_cols_sales + ['aba_predicted', 'units']
out_cols_sales = [c for c in out_cols_sales if c in dt_sales_orig.columns]
dt_sales_out = dt_sales_orig[out_cols_sales].copy()

mask_sales_future = dt_sales_out['listed'] == 4
dt_sales_out.loc[mask_sales_future, 'units'] = np.nan

tmp_sales = os.path.join(tempfile.gettempdir(), 'tmp_data_p05_sales.xlsx')
out_path_sales = os.path.join(p05_root, 'data', 'data_processed', 'data_for_sample_p05_sales.xlsx')
dt_sales_out.to_excel(tmp_sales, index=False)
shutil.copy2(tmp_sales, out_path_sales)
os.remove(tmp_sales)

# Save model_2 for later training
model_2.save(ds=dt_sales_t, path=os.path.join(os.getcwd(), 'code_py', 'model_data_sales'))

print(f'Saved Sales: {len(dt_sales_out)} rows x {len(dt_sales_out.columns)} cols')
print('\n===== OVERWRITE COMPLETE =====')

"""
Step 4: Overwrite target variable to be consistent with model structure.
Generate units from transformed RHS variables with known "true" coefficients + noise.
"""
import os
import shutil
import sys
import tempfile
import warnings
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
p01_root = os.path.join(_SCRIPT_DIR, '..', '..')
os.chdir(p01_root)
sys.path.insert(0, p01_root)

import fermatrica as fm
import code_py.adhoc.model

warnings.simplefilter(action='ignore')
np.random.seed(42)

dt_p = pd.read_excel('data/data_processed/data_for_sample_p01.xlsx', parse_dates=['date'])
if 'Intercept' in dt_p.columns:
    del dt_p['Intercept']

raw_cols = dt_p.columns.tolist()
dt_p_orig = dt_p.copy()

pth = os.path.join('code_py', 'model_data', 'model_conf.xlsx')
model = fm.Model(path=pth, adhoc_code=[code_py.adhoc.model], ds=dt_p)
model = fm.transform(ds=dt_p, model=model, set_start=True, if_by_ref=True)

# Per-superbrand intercepts for panel structure (base volume contribution)
superbrand_intercepts = {
    'brand_1': 1200.0,
    'brand_2': 800.0,
    'brand_3': 600.0,
}

true_coefs = {
    'di_soph_ssn_median_sm_softmax': 2000.0,
    'di_soph_ssn_median_sm_share_softmax': 1000.0,
    'brnd_tv_ots_clip_softmax_lag_adstockp:di_soph_ssn_median_sm_softmax': 3000.0,
    'brnd_ots_ooh_softmax_lag_adstockp:di_soph_ssn_median_sm_softmax': 2000.0,
    'brnd_cl_dgtind_perform_lag_softmax_adstockp': 800.0,
    'brnd_cl_dgtind_display_lag_softmax_adstockp:di_soph_ssn_median_sm_softmax': 1200.0,
    'brnd_cl_dgtind_video_lag_softmax_adstockp:di_soph_ssn_median_sm_softmax': 1200.0,
    'brnd_cl_dgtind_special_lag_softmax_adstockp:di_soph_ssn_median_sm_softmax': 1000.0,
    'e_com_infl_lag_softmax_adstockp:di_soph_ssn_median_sm_softmax': 500.0,
    'pr_costs_sm_infl_lag_softmax_adstockp:di_soph_ssn_median_sm_softmax': 600.0,
    'pos_infl_lag_softmax_adstockp:di_soph_ssn_median_sm_softmax': 500.0,
}

true_lhs = {
    'tt_pk_tr_ssn': 1.5,
    'hdays': -0.04,
    'price': -0.2,
    'price_volat_rel': -0.11,
}

# Build RHS using interaction terms
model_rhs = model.conf.model_rhs
N = len(dt_p)
y_linear = np.array([superbrand_intercepts.get(sb, 1000.0) for sb in dt_p['superbrand']])

for _, row in model_rhs.iterrows():
    token = row['token']
    coef = true_coefs.get(token, 0.0)
    if token in dt_p.columns:
        x = dt_p[token].values.copy()
    elif ':' in token:
        parts = token.split(':')
        x = np.ones(N)
        for p in parts:
            p = p.strip()
            if p in dt_p.columns:
                x = x * dt_p[p].values
    else:
        x = np.zeros(N)
    y_linear += coef * x

# Apply LHS decomposition
ssn_factor = (true_lhs['tt_pk_tr_ssn'] + 1) / (true_lhs['tt_pk_tr_ssn'] * dt_p['tt_pk_tr_ssn'].values + 1)
hdays_factor = 1 + dt_p['hdays'].values * true_lhs['hdays']
price_factor = 1 + true_lhs['price'] * dt_p['price_ret_rel_market_mar_softmax'].values
volat_factor = 1 + true_lhs['price_volat_rel'] * dt_p['price_ret_volat_rel_softmax'].values

units_new = y_linear / ssn_factor / hdays_factor / price_factor / volat_factor
units_new = units_new * np.exp(np.random.randn(N) * 0.08)
units_new = np.round(units_new).astype(int)
units_new = np.clip(units_new, 50, None)

dt_p_orig['units'] = units_new * 100.0

print(f'Overwritten units: min={dt_p_orig["units"].min()}, max={dt_p_orig["units"].max()}, mean={dt_p_orig["units"].mean():.2f}, std={dt_p_orig["units"].std():.2f}')

# Save - only original raw columns plus units
keep_cols = [c for c in raw_cols + ['units'] if c in dt_p_orig.columns]
dt_p_out = dt_p_orig[keep_cols]

mask = dt_p_out['listed'] == 4
dt_p_out.loc[mask, 'units'] = np.nan

tmp_path = os.path.join(tempfile.gettempdir(), 'tmp_data_p01.xlsx')
out_path = os.path.join(_SCRIPT_DIR, '..', '..', 'data', 'data_processed', 'data_for_sample_p01.xlsx')
dt_p_out.to_excel(tmp_path, index=False)
shutil.copy2(tmp_path, out_path)
os.remove(tmp_path)
print(f'Saved {len(dt_p_out)} rows x {len(dt_p_out.columns)} cols')

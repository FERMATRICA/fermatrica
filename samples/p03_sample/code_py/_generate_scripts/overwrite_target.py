"""
Step 4: Overwrite target variable to be consistent with model structure.
Generate ABA from transformed RHS variables with known "true" coefficients + noise.
"""
import os
import shutil
import sys
import tempfile
import warnings
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
p03_root = os.path.join(_SCRIPT_DIR, '..', '..')
os.chdir(p03_root)
sys.path.insert(0, p03_root)

import fermatrica as fm
import code_py.adhoc.model

warnings.simplefilter(action='ignore')
np.random.seed(42)

dt_p = pd.read_excel('data/data_processed/data_for_sample_p03.xlsx', parse_dates=['date'])
if 'Intercept' in dt_p.columns:
    del dt_p['Intercept']

raw_cols = dt_p.columns.tolist()
dt_p_orig = dt_p.copy()

pth = os.path.join('code_py', 'model_data', 'model_conf.xlsx')
model = fm.Model(path=pth, adhoc_code=[code_py.adhoc.model], ds=dt_p)
model = fm.transform(ds=dt_p, model=model, set_start=True, if_by_ref=True)

# Per-superbrand intercepts for panel structure
superbrand_intercepts = {
    'brand_1': 15.0,
    'brand_2': 20.0,
    'brand_3': 30.0,
}

true_coefs = {
    'di_soph_ssn_median_sm_softmax': 30.0,
    # 'di_soph_ssn_median_sm_share_softmax': 10.0,
    'brnd_tv_ots_clip_softmax_bht_lag_adstockp:di_soph_ssn_median_sm_softmax': 35000.0,
    'brnd_ots_ooh_softmax_bht_lag_adstockp:di_soph_ssn_median_sm_softmax': 50000.0,
    # 'brnd_cl_dgtind_perform_lag_softmax_adstockp': 20.0,
    'brnd_cl_dgtind_display_lag_softmax_adstockp:di_soph_ssn_median_sm_softmax': 15000.0,
    'brnd_cl_dgtind_video_lag_softmax_adstockp:di_soph_ssn_median_sm_softmax': 15000.0,
    'brnd_cl_dgtind_special_lag_softmax_adstockp:di_soph_ssn_median_sm_softmax': 12000.0,
    'brnd_cl_dgtind_indoor_lag_softmax_adstockp:di_soph_ssn_median_sm_softmax': 10000.0,
    # 'e_com_infl_lag_softmax_adstockp:di_soph_ssn_median_sm_softmax': 6.0,
    'pr_costs_sm_infl_lag_softmax_adstockp:di_soph_ssn_median_sm_softmax': 8000.0,
    'pos_infl_lag_softmax_adstockp:di_soph_ssn_median_sm_softmax': 5000.0,
}

true_lhs = {
    'tt_pk_tr_ssn': 0.05,
    'hdays': -0.001,
    'price': 0.0,
    'price_volat_rel': 0.0,
}

# Build RHS using interaction terms
model_rhs = model.conf.model_rhs
y_linear = np.array([superbrand_intercepts.get(sb, 55.0) for sb in dt_p['superbrand']])

for _, row in model_rhs.iterrows():
    token = row['token']
    coef = true_coefs.get(token, 0.0)
    if token in dt_p.columns:
        x = dt_p[token].values.copy()
    elif ':' in token:
        parts = token.split(':')
        x = np.ones(len(dt_p))
        for p in parts:
            p = p.strip()
            if p in dt_p.columns:
                x = x * dt_p[p].values
    else:
        x = np.zeros(len(dt_p))
    y_linear += coef * x

# Apply LHS decomposition
ssn_factor = (true_lhs['tt_pk_tr_ssn'] + 1) / (true_lhs['tt_pk_tr_ssn'] * dt_p['tt_pk_tr_ssn'].values + 1)
hdays_factor = 1 + dt_p['hdays'].values * true_lhs['hdays']
price_factor = 1 + true_lhs['price'] * dt_p['price_ret_rel_market_mar_softmax'].values
volat_factor = 1 + true_lhs['price_volat_rel'] * dt_p['price_ret_volat_rel_softmax'].values

aba_new = y_linear / ssn_factor / hdays_factor / price_factor / volat_factor
aba_new = aba_new + np.random.randn(len(dt_p)) * 1.1
aba_new = aba_new / aba_new.max() * 70

dt_p_orig['aba'] = np.round(aba_new, 2)
dt_p_orig['aba_lag'] = dt_p_orig.groupby('superbrand')['aba'].shift(1)
dt_p_orig['aba_lag'] = dt_p_orig.groupby('superbrand')['aba_lag'].bfill()

print(f'Overwritten ABA: min={dt_p_orig["aba"].min():.2f}, max={dt_p_orig["aba"].max():.2f}, mean={dt_p_orig["aba"].mean():.2f}, std={dt_p_orig["aba"].std():.2f}')

# Save - only original raw columns plus aba and aba_lag
keep_cols = [c for c in raw_cols if c in dt_p_orig.columns]
extra_cols = ['aba_lag'] if 'other' not in keep_cols else []
out_cols = keep_cols + [c for c in ['aba', 'aba_lag'] if c not in keep_cols]
dt_p_out = dt_p_orig[out_cols]

mask = dt_p_out['listed'] == 4
dt_p_out.loc[mask, 'aba'] = np.nan

tmp_path = os.path.join(tempfile.gettempdir(), 'tmp_data_p03.xlsx')
out_path = os.path.join(_SCRIPT_DIR, '..', '..', 'data', 'data_processed', 'data_for_sample_p03.xlsx')
dt_p_out.to_excel(tmp_path, index=False)
shutil.copy2(tmp_path, out_path)
os.remove(tmp_path)
print(f'Saved {len(dt_p_out)} rows x {len(dt_p_out.columns)} cols')

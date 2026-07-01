"""Full pipeline test for p04_sample"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
p04_root = os.path.join(_SCRIPT_DIR, '..', '..')
os.chdir(p04_root)
sys.path.insert(0, p04_root)

import fermatrica as fm
import code_py.adhoc.model

warnings.simplefilter(action='ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)

dt_p = pd.read_excel('data/data_processed/data_for_sample_p04.xlsx', parse_dates=['date'])
if 'Intercept' in dt_p.columns:
    del dt_p['Intercept']

print(f'Data: {dt_p.shape}, regions={dt_p["master"].nunique()}, weeks={dt_p["date"].nunique()}')

pth = os.path.join('code_py', 'model_data', 'model_conf.xlsx')
model = fm.Model(path=pth, adhoc_code=[code_py.adhoc.model], ds=dt_p)

t0 = time.time()
model = fm.transform(ds=dt_p, model=model, set_start=True, if_by_ref=True)
print(f'Transform (set_start): {time.time()-t0:.2f}s')

active = model.conf.params[(model.conf.params['if_active'] == 1) & (model.conf.params['fixed'] == 0)]
print(f'Active non-fixed params: {len(active)}')

if len(active) > 0:
    t0 = time.time()
    model = fm.optimize_local_cobyla(dt_p, model, revert_score_sign=False, verbose=True,
                                      epochs=1, iters_epoch=300, error_score=1e+12, ftol_abs=.001)
    print(f'Optimization: {time.time()-t0:.2f}s')
else:
    print('Skipping optimization (all params fixed)')

model = fm.transform(ds=dt_p, model=model, set_start=False, if_by_ref=True)

t0 = time.time()
pred, pred_raw, model = fm.fit_predict(dt_p, model, if_fit=True, return_full=True)
print(f'Fit time: {time.time()-t0:.2f}s')

if model is None or model.obj is None:
    print('FIT FAILED - model is None')
    sys.exit(1)

print(f'\nModel type: {type(model.obj.models["main"]).__name__}')
print(f'Model converged: {model.obj.models["main"].converged}')

t0 = time.time()
pred = fm.fit_predict(dt_p, model, if_fit=False, return_full=False)
print(f'Predict time: {time.time()-t0:.2f}s')

train = dt_p['listed'] == 2
test = dt_p['listed'] == 3
all_ft = dt_p['listed'].isin([2, 3])

r2_train = fm.metrics.r_squared(dt_p.loc[train, model.conf.Y_var], pred[train])
r2_all = fm.metrics.r_squared(dt_p.loc[all_ft, model.conf.Y_var], pred[all_ft])
mape_train = fm.metrics.mapef(dt_p.loc[train, model.conf.Y_var], pred[train])
mape_test = fm.metrics.mapef(dt_p.loc[test, model.conf.Y_var], pred[test])

print(f'\n===== METRICS =====')
print(f'R2 train: {r2_train:.4f}')
print(f'R2 all:   {r2_all:.4f}')
print(f'MAPE train: {mape_train:.2f}%')
print(f'MAPE test:  {mape_test:.2f}%')

print(f'\n===== BY REGION =====')
for r in dt_p['master'].unique():
    mask = (dt_p['master'] == r) & all_ft
    r2_r = fm.metrics.r_squared(dt_p.loc[mask, model.conf.Y_var], pred[mask])
    print(f'{r}: R2={r2_r:.4f}, n={mask.sum()}')

print(f'\n===== COEFFICIENTS =====')
print(model.obj.models['main'].summary())

# ═══════════════════════════════════════════
# Option calculation test
# ═══════════════════════════════════════════

print('\n===== OPTION TEST =====')
import fermatrica_rep as fmr

options_c_raw = pd.read_excel('code_py/model_data/options/options.xlsx', sheet_name='current')
options_c = {}
for ms in options_c_raw['text_flags'].unique():
    if pd.isna(ms):
        continue
    options_c[ms] = fmr.budget_dict_create(dct=options_c_raw[options_c_raw['text_flags'] == ms])

media_prices = pd.read_excel('code_py/model_data/options/options.xlsx', sheet_name='media_prices')
media_prices = pd.melt(media_prices, id_vars='master', var_name='variable', value_name='price')
masters = media_prices['master'].unique().tolist()

trans_dict = fmr.trans_dict_create(path='code_py/model_data/options/options.xlsx', sheet='translation')
model_rep = fmr.ModelRep(dt_p, trans_dict=trans_dict, language='english')

dt_p_opt = fmr.media_ssn_apply(dt_p, path='code_py/model_data/options/seasonality.xlsx', sheet='data')

total_pred = 0
for ms in masters:
    print(f'Option test: {ms}')
    model.conf.trans_path_df = media_prices[media_prices['master'] == ms]
    opt_set = fmr.OptionSettings(target=[['brand_1'], [ms]]
                                 , date_start=pd.Timestamp('2026-01-01').date()
                                 , date_end=pd.Timestamp('2026-12-31').date()
                                 , ref_date_start=pd.Timestamp('2025-01-01').date()
                                 , ref_date_end=pd.Timestamp('2025-12-31').date()
                                 , apply_vars=["superbrand", "master"]
                                 , plan_period='exact')
    dt_p_opt, dt_pred, opt_sum = fmr.option_report(
        model, dt_p_opt, model_rep, options_c[ms]['predefined'], opt_set, if_exact=True)
    ms_pred = dt_pred.loc[dt_pred['listed'] == 4, 'predicted'].sum()
    total_pred += ms_pred
    print(f'  {ms}: predicted = {ms_pred:.0f}')

print(f'\nTotal predicted volume across all masters: {total_pred:.0f}')

if total_pred > 0:
    print('Option test PASSED')
else:
    print('Option test FAILED')

print('\n===== SUCCESS =====')

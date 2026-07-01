"""Full pipeline test for p05_sample — two-level panel model"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
p05_root = os.path.join(_SCRIPT_DIR, '..', '..')
os.chdir(p05_root)
sys.path.insert(0, p05_root)

import fermatrica as fm
import fermatrica_rep as fmr
import code_py.adhoc.model
from code_py.adhoc.reporting import pred_merge_bht

warnings.simplefilter(action='ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)

print('=' * 80)
print('TWO-LEVEL PANEL MODEL TEST (p05)')
print('=' * 80)

# ============================================================
# PART 1: BHT Model (brand-month-ABA)
# ============================================================
print('\n' + '=' * 60)
print('PART 1: BHT Model (brand-month-ABA)')
print('=' * 60)

dt_bht = pd.read_excel('data/data_processed/data_for_sample_p05_bht.xlsx', parse_dates=['date'])
if 'Intercept' in dt_bht.columns:
    del dt_bht['Intercept']

print(f'BHT Data: {dt_bht.shape}, brands={dt_bht["superbrand"].nunique()}, months={dt_bht["date"].nunique()}')
print(f'ABA range: {dt_bht["aba"].min():.2f} - {dt_bht["aba"].max():.2f}, mean={dt_bht["aba"].mean():.2f}')

pth_bht = os.path.join('code_py', 'model_data_bht', 'model_conf.xlsx')
model_1 = fm.Model(path=pth_bht, adhoc_code=[code_py.adhoc.model], ds=dt_bht)

t0 = time.time()
model_1 = fm.transform(ds=dt_bht, model=model_1, set_start=True, if_by_ref=True)
print(f'Transform (set_start): {time.time()-t0:.2f}s')

active_1 = model_1.conf.params[(model_1.conf.params['if_active'] == 1) & (model_1.conf.params['fixed'] == 0)]
print(f'Active non-fixed params: {len(active_1)}')

if len(active_1) > 0:
    t0 = time.time()
    model_1 = fm.optimize_local_cobyla(dt_bht, model_1, revert_score_sign=False, verbose=True,
                                        epochs=1, iters_epoch=200, error_score=1e+12, ftol_abs=.001)
    print(f'Optimization: {time.time()-t0:.2f}s')
else:
    print('Skipping optimization (all params fixed)')

model_1 = fm.transform(ds=dt_bht, model=model_1, set_start=False, if_by_ref=True)

t0 = time.time()
pred_1, pred_raw_1, model_1 = fm.fit_predict(dt_bht, model_1, if_fit=True, return_full=True)
print(f'Fit time: {time.time()-t0:.2f}s')

if model_1 is None or model_1.obj is None:
    print('BHT FIT FAILED')
    sys.exit(1)

print(f'BHT Model converged: {model_1.obj.models["main"].converged}')

t0 = time.time()
pred_1 = fm.fit_predict(dt_bht, model_1, if_fit=False, return_full=False)
print(f'Predict time: {time.time()-t0:.2f}s')

train_1 = dt_bht['listed'] == 2
test_1 = dt_bht['listed'] == 3
all_1 = dt_bht['listed'].isin([2, 3])

r2_train_1 = fm.metrics.r_squared(dt_bht.loc[train_1, 'aba'], pred_1[train_1])
r2_all_1 = fm.metrics.r_squared(dt_bht.loc[all_1, 'aba'], pred_1[all_1])
mape_train_1 = fm.metrics.mapef(dt_bht.loc[train_1, 'aba'], pred_1[train_1])
mape_test_1 = fm.metrics.mapef(dt_bht.loc[test_1, 'aba'], pred_1[test_1])

print(f'\n===== BHT METRICS =====')
print(f'R2 train: {r2_train_1:.4f}')
print(f'R2 all:   {r2_all_1:.4f}')
print(f'MAPE train: {mape_train_1:.2f}%')
print(f'MAPE test:  {mape_test_1:.2f}%')

print(f'\n===== BHT BY BRAND =====')
for b in dt_bht['superbrand'].unique():
    mask = (dt_bht['superbrand'] == b) & all_1
    r2_b = fm.metrics.r_squared(dt_bht.loc[mask, 'aba'], pred_1[mask])
    print(f'{b}: R2={r2_b:.4f}, n={mask.sum()}')

# Make predictions for future (listed=4) to use in model_2
dt_pred_1 = fm.predict_ext(model_1, dt_bht)

# Save model_1
model_1.save(ds=dt_bht, path=os.path.join(os.getcwd(), 'code_py', 'model_data_bht'))

# ============================================================
# PART 2: Merge BHT predictions into Sales data
# ============================================================
print('\n' + '=' * 60)
print('PART 2: Merge BHT predictions into Sales data')
print('=' * 60)

dt_sales = pd.read_excel('data/data_processed/data_for_sample_p05_sales.xlsx', parse_dates=['date'])
if 'Intercept' in dt_sales.columns:
    del dt_sales['Intercept']

print(f'Sales Data (raw): {dt_sales.shape}, SKUs={dt_sales["bs_key"].nunique()}, weeks={dt_sales["date"].nunique()}')

# Merge predicted ABA from model_1
dt_sales = pred_merge_bht(dt_sales, dt_pred_1)

print(f'aba_predicted after merge: min={dt_sales["aba_predicted"].min():.2f}, max={dt_sales["aba_predicted"].max():.2f}')
print(f'aba_predicted NaN count: {dt_sales["aba_predicted"].isna().sum()}')

# ============================================================
# PART 3: Sales Model (SKU-week-units + ABA)
# ============================================================
print('\n' + '=' * 60)
print('PART 3: Sales Model (SKU-week-units with ABA)')
print('=' * 60)

pth_sales = os.path.join('code_py', 'model_data_sales', 'model_conf.xlsx')
model_2 = fm.Model(path=pth_sales, adhoc_code=[code_py.adhoc.model], ds=dt_sales)

t0 = time.time()
model_2 = fm.transform(ds=dt_sales, model=model_2, set_start=True, if_by_ref=True)
print(f'Transform (set_start): {time.time()-t0:.2f}s')

active_2 = model_2.conf.params[(model_2.conf.params['if_active'] == 1) & (model_2.conf.params['fixed'] == 0)]
print(f'Active non-fixed params: {len(active_2)}')

if len(active_2) > 0:
    t0 = time.time()
    model_2 = fm.optimize_local_cobyla(dt_sales, model_2, revert_score_sign=False, verbose=True,
                                        epochs=1, iters_epoch=200, error_score=1e+12, ftol_abs=.001)
    print(f'Optimization: {time.time()-t0:.2f}s')
else:
    print('Skipping optimization (all params fixed)')

model_2 = fm.transform(ds=dt_sales, model=model_2, set_start=False, if_by_ref=True)

t0 = time.time()
pred_2, pred_raw_2, model_2 = fm.fit_predict(dt_sales, model_2, if_fit=True, return_full=True)
print(f'Fit time: {time.time()-t0:.2f}s')

if model_2 is None or model_2.obj is None:
    print('SALES FIT FAILED')
    sys.exit(1)

print(f'Sales Model converged: {model_2.obj.models["main"].converged}')

t0 = time.time()
pred_2 = fm.fit_predict(dt_sales, model_2, if_fit=False, return_full=False)
print(f'Predict time: {time.time()-t0:.2f}s')

train_2 = dt_sales['listed'] == 2
test_2 = dt_sales['listed'] == 3
all_2 = dt_sales['listed'].isin([2, 3])

r2_train_2 = fm.metrics.r_squared(dt_sales.loc[train_2, 'units'], pred_2[train_2])
r2_all_2 = fm.metrics.r_squared(dt_sales.loc[all_2, 'units'], pred_2[all_2])
mape_train_2 = fm.metrics.mapef(dt_sales.loc[train_2, 'units'], pred_2[train_2])
mape_test_2 = fm.metrics.mapef(dt_sales.loc[test_2, 'units'], pred_2[test_2])

print(f'\n===== SALES METRICS =====')
print(f'R2 train: {r2_train_2:.4f}')
print(f'R2 all:   {r2_all_2:.4f}')
print(f'MAPE train: {mape_train_2:.2f}%')
print(f'MAPE test:  {mape_test_2:.2f}%')

print(f'\n===== SALES BY BRAND =====')
for b in dt_sales['superbrand'].unique():
    mask = (dt_sales['superbrand'] == b) & all_2
    r2_b = fm.metrics.r_squared(dt_sales.loc[mask, 'units'], pred_2[mask])
    print(f'{b}: R2={r2_b:.4f}, n={mask.sum()}')

# Save model_2
dt_pred_2 = fm.predict_ext(model_2, dt_sales)
model_2.save(ds=dt_sales, path=os.path.join(os.getcwd(), 'code_py', 'model_data_sales'))

print(f'\n===== TWO-LEVEL TEST COMPLETE =====')
print(f'BHT R2 (train): {r2_train_1:.4f}')
print(f'Sales R2 (train): {r2_train_2:.4f}')

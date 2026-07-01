"""Full pipeline test for p01_sample"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
p01_root = os.path.join(_SCRIPT_DIR, '..', '..')
os.chdir(p01_root)
sys.path.insert(0, p01_root)

import fermatrica as fm
import fermatrica_rep as fmr
import code_py.adhoc.model

warnings.simplefilter(action='ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)

dt_p = pd.read_excel('data/data_processed/data_for_sample_p01.xlsx', parse_dates=['date'])
if 'Intercept' in dt_p.columns:
    del dt_p['Intercept']

print(f'Data: {dt_p.shape}, brands={dt_p["superbrand"].nunique()}, weeks={dt_p["date"].nunique()}')

pth = os.path.join('code_py', 'model_data', 'model_conf.xlsx')
model = fm.Model(path=pth, adhoc_code=[code_py.adhoc.model], ds=dt_p)

# Transform
t0 = time.time()
model = fm.transform(ds=dt_p, model=model, set_start=True, if_by_ref=True)
print(f'Transform (set_start): {time.time()-t0:.2f}s')

# Optimization
active = model.conf.params[(model.conf.params['if_active'] == 1) & (model.conf.params['fixed'] == 0)]
print(f'Active non-fixed params: {len(active)}')

if len(active) > 0:
    t0 = time.time()
    model = fm.optimize_local_cobyla(dt_p, model, revert_score_sign=False, verbose=True,
                                      epochs=1, iters_epoch=300, error_score=1e+12, ftol_abs=.001)
    print(f'Optimization: {time.time()-t0:.2f}s')
else:
    print('Skipping optimization (all params fixed)')

# Transform again
model = fm.transform(ds=dt_p, model=model, set_start=False, if_by_ref=True)

# Fit
t0 = time.time()
pred, pred_raw, model = fm.fit_predict(dt_p, model, if_fit=True, return_full=True)
print(f'Fit time: {time.time()-t0:.2f}s')

if model is None or model.obj is None:
    print('FIT FAILED - model is None')
    sys.exit(1)

print(f'\nModel type: {type(model.obj.models["main"]).__name__}')
print(f'Model converged: {model.obj.models["main"].converged}')

# Predict
t0 = time.time()
pred = fm.fit_predict(dt_p, model, if_fit=False, return_full=False)
print(f'Predict time: {time.time()-t0:.2f}s')

# Metrics
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

print(f'\n===== BY BRAND =====')
for b in dt_p['superbrand'].unique():
    mask = (dt_p['superbrand'] == b) & all_ft
    r2_b = fm.metrics.r_squared(dt_p.loc[mask, model.conf.Y_var], pred[mask])
    print(f'{b}: R2={r2_b:.4f}, n={mask.sum()}')

print(f'\n===== COEFFICIENTS =====')
print(model.obj.models['main'].summary())

print('\n===== SUCCESS =====')

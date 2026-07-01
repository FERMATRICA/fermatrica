import numpy as np
import pandas as pd


def pred_merge_bht(ds_cur: pd.DataFrame
                   , dt_pred_prev: pd.DataFrame):
    """
    Merge predicted ABA from brand-month model into SKU-week dataset.
    """
    tmp = dt_pred_prev.loc[:, ['date', 'superbrand', 'predicted']]
    tmp['year'] = tmp['date'].dt.year
    tmp['month'] = tmp['date'].dt.month
    del tmp['date']

    ds_cur['year'] = ds_cur['date'].dt.year
    ds_cur['month'] = ds_cur['date'].dt.month

    ds_cur = pd.merge(ds_cur, tmp, how='left', on=['superbrand', 'year', 'month'], sort=False)

    mask = (ds_cur['listed'] == 4) & (ds_cur['predicted'].notna())
    ds_cur.loc[mask, 'aba_predicted'] = ds_cur.loc[mask, 'predicted']

    del ds_cur['year']
    del ds_cur['month']
    del ds_cur['predicted']

    return ds_cur

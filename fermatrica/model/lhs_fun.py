"""
Functions to be used in LHS part of the model (i.e. Y transformations). Basic transformation
is multiplication, all other transformations to be declared here.
"""


import numpy as np
import pandas as pd

from line_profiler_pycharm import profile

from fermatrica_utils import DotDict, select_eff


@profile
def market_resize(ds: pd.DataFrame
                  , y: pd.Series | np.ndarray) -> pd.Series:
    """
    Resize main model by additional category model. Useful for simulating competition

    1. Calculate category prediction
    2. Calculate main (per SKU, brand, region etc.) prediction
    3. If sum of the (2) predictions by period diverges from (1) prediction, proportionally
        weight (2) predictions to make sum of them equal to (1)

    :param ds: dataset
    :param y: predicted Y (target variable)
    :return: predicted Y (target variable)
    """

    ds = select_eff(ds, ['bs_key', 'date', 'listed', 'market', 'pred_mrk'])
    ds['y'] = y

    tmp = ds.loc[ds['listed'] >= 1].groupby(['date', 'market', 'pred_mrk'], as_index=False)['y'].sum()
    tmp = tmp.set_index(['date', 'market'])
    tmp['resize'] = tmp['y'] / tmp['pred_mrk']

    ds = ds.join(select_eff(tmp, ['resize']), on=['date', 'market'], how='left', sort=False)
    y_series = ds['y'] / ds['resize']

    return y_series


@profile
def market_resize_flex(ds: pd.DataFrame
                       , y: pd.Series | np.ndarray
                       , wght: int | float) -> pd.Series:
    """
    Resize main model by additional category model with some discount.
    Use it to simulate competition. More flexible than `market_resize`

    1. Calculate category prediction
    2. Calculate main (per SKU, brand, region etc.) prediction
    3. If sum of the (2) predictions by period diverges from (1) prediction, proportionally
        weight (2) predictions to make sum of them equal to (1) with discount (1 - wght)

    :param ds: dataset
    :param y: predicted Y (target variable)
    :param wght: weight for correction by category prediction
    :return: predicted Y (target variable)
    """

    ds = select_eff(ds, ['bs_key', 'date', 'listed', 'market', 'pred_mrk'])
    ds['y'] = y

    tmp = ds.loc[ds['listed'] >= 1].groupby(['date', 'market', 'pred_mrk'], as_index=False)['y'].sum()
    tmp = tmp.set_index(['date', 'market'])
    tmp['resize'] = tmp['y'] / tmp['pred_mrk']

    ds = ds.join(select_eff(tmp, ['resize']), on=['date', 'market'], how='left', sort=False)
    y_series = ds['y'] * ((1 - wght) + wght / ds['resize'])

    return y_series


@profile
def log_exp(ds: pd.DataFrame
            , y: pd.Series | np.ndarray
            , to_original: bool = False) -> pd.Series:
    """
    Log & exp Y (LHS), e.g. for log-log model

    :param ds: dataset
    :param y: predicted Y (target variable)
    :param to_original: from inner model to outer or vice versa
    :return: predicted Y (target variable), inner or outer
    """

    if isinstance(y, pd.Series):
        y = y.array

    if to_original:
        y_series = np.expm1(y)
    else:
        y_series = np.log1p(y)

    return pd.Series(y_series, index=ds.index)

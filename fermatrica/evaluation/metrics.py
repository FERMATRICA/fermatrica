"""
Metrics to be used in model evaluations. Do not confuse with `scoring` containing functions
to compose complex scoring from basic metrics defined here. Scoring is aimed more to facilitate algorithmic
optimisation, metrics could be interpreted by experts and event clients themselves.
"""


import logging
import numpy as np
import pandas as pd

import statistics
from statsmodels.tools import eval_measures as em
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from fermatrica_utils import rm_1_item_groups
from fermatrica.basics.basics import fermatrica_error


"""
Basic (atomic) error metrics
"""


def r_squared(obs: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
              , pred: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list) -> float:
    """
    Classic (non-adjusted) R^2:
    R^2 = MSE(obs, pred) / Var(obs)

    Independent of model type, just observed and predicted vectors are required.
    Unit-free metrics for regression tasks.

    :param obs: observed values (Y)
    :param pred: predicted values (Y-hat)
    :return:
    """

    if len(obs) != len(pred):
        fermatrica_error("Error in calculation of R^2: length of observed vector is not equal to length of predicted")

    rtrn = 1 - em.mse(obs, pred) / statistics.variance(obs)

    return rtrn


def rmse(obs: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
         , pred: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list) -> float:
    """
    Root mean squared error (RMSE) between observed `obs` and predicted `pred` vectors:
    RMSE = ( SUM((obs - pred)^2) / LEN(obs)) ^ .5

    Most used metrics for regression tasks.

    :param obs: observed values (Y)
    :param pred: predicted values (Y-hat)
    :return:
    """

    if len(obs) != len(pred):
        fermatrica_error("Error in calculation of RMSE: length of observed vector is not equal to length of predicted")

    rtrn = em.rmse(obs, pred)

    return rtrn


def mse(obs: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
        , pred: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list) -> float:
    """
    Mean squared error (MSE) between observed `obs` and predicted `pred` vectors:
    MSE = SUM((obs - pred)^2) / LEN(obs)

    More basic and a bit faster version of RMSE, but calculated in squared units and
    hence less interpretable and popular in the industry (but not in the scientific research)

    :param obs: observed values (Y)
    :param pred: predicted values (Y-hat)
    :return:
    """

    if len(obs) != len(pred):
        fermatrica_error("Error in calculation of MSE: length of observed vector is not equal to length of predicted")

    rtrn = em.mse(obs, pred)

    return rtrn


def mape(obs: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
         , pred: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list) -> float:
    """
    Mean absolute percentage error (MAPE) between observed `obs` and predicted `pred` vectors:
    MAPE = SUM( ABS((obs - pred) / obs) ) / LEN(obs) * 100

    More robust metrics for regression tasks, also most intuitive for non-specialists.
    Some major issues of MAPE are 0-division if observed value is 0 and extreme sensitivity for
    even small absolute errors when observed values are much closer to 0 than the whole vector.

    To avoid it some popular workarounds / alternatives are implemented: `mapef`, `mape_adj`, `smape`

    :param obs: observed values (Y)
    :param pred: predicted values (Y-hat)
    :return:
    """

    if len(obs) != len(pred):
        fermatrica_error("Error in calculation of MAPE: length of observed vector is not equal to length of predicted")

    if not isinstance(obs, np.ndarray):
        obs = np.array(obs)
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)

    rtrn = np.mean(np.abs((obs - pred) / obs)) * 100
    
    return rtrn


def mapef(obs: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
          , pred: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list) -> float:
    """
    Fixed ("f") version of mean absolute percentage error (MAPE) between observed `obs` and predicted `pred` vectors:
    MAPE = SUM( ABS((obs - pred) / obs) ) / LEN(obs) * 100

    More robust metrics for regression tasks, also most intuitive for non-specialists.
    Points with zero observations are removed in this implementation to avoid 0-division.

    :param obs: observed values (Y)
    :param pred: predicted values (Y-hat)
    :return:
    """

    if len(obs) != len(pred):
        fermatrica_error("Error in calculation of MAPE (mapef): length of observed vector is not equal to length of predicted")

    if not isinstance(obs, np.ndarray):
        obs = np.array(obs)
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)

    obs, pred = obs[obs > 0], pred[obs > 0]
    rtrn = np.mean(np.abs((obs - pred) / obs)) * 100

    return rtrn


def mape_adj(obs: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
             , pred: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
             , adj_val: int | float) -> float:
    """
    Adjusted ("adj") version of mean absolute percentage error (MAPE) between observed `obs` and predicted `pred` vectors:
    MAPE = SUM( ABS((obs - pred) / obs) ) / LEN(obs) * 100

    More robust metrics for regression tasks, also most intuitive for non-specialists.
    Points with observations lesser than `adj_val` threshold are removed in this implementation to avoid 0-division
    and / or exclude points with hardly predictable small observed values.

    :param obs: observed values (Y)
    :param pred: predicted values (Y-hat)
    :param adj_val: threshold to exclude small values / bottom outliers
    :return:
    """

    if len(obs) != len(pred):
        fermatrica_error("Error in calculation of MAPE (mape_adj): length of observed vector is not equal to length of predicted")

    if not isinstance(obs, np.ndarray):
        obs = np.array(obs)
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)

    obs = np.where(obs <= adj_val, np.mean(obs), obs)
    rtrn = np.mean(np.abs((obs - pred) / obs)) * 100

    return rtrn


def smape(obs: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
          , pred: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list) -> float:
    """
    Symmetric mean absolute percentage error (SMAPE) between observed `obs` and predicted `pred` vectors:
    SMAPE = SUM( ABS((obs - pred) / (ABS(obs) + ABS(pred))) ) / LEN(obs) * 100 * 2

    More robust metrics for regression tasks, also most intuitive for non-specialists.
    Symmetric version avoids 0-division by dividing on sum of absolute values of `obs` and `pred` instead of
    just `obs`.

    :param obs: observed values (Y)
    :param pred: predicted values (Y-hat)
    :return:
    """

    if len(obs) != len(pred):
        fermatrica_error("Error in calculation of SMAPE: length of observed vector is not equal to length of predicted")

    if not isinstance(obs, np.ndarray):
        obs = np.array(obs)
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)

    rtrn = np.mean(np.abs(obs - pred) / (np.abs(obs) + np.abs(pred))) * 100 * 2

    return rtrn


"""
Grouped error metrics
"""


def r_squared_group(obs: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
                    , pred: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
                    , group: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
                    , reduce: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list | None = None) -> float:
    """
    Calculate R^2 per group (e.g. per superbrand or per region)

    :param obs: vector with observed values (Y)
    :param pred: vector with predicted values (Y-hat)
    :param group: vector with grouping values, e.g. 'superbrand'
    :param reduce: if dataset's entity differs from `group`, additional vector to group by before metrics calculation,
        e.g. 'date'
    :return:
    """

    if len(obs) != len(pred):
        fermatrica_error("Error in calculation of R^2 by group: length of observed vector is not equal to length of predicted")
    if len(obs) != len(group):
        fermatrica_error("Error in calculation of R^2 by group: length of observed vector is not equal to length of grouping vector")

    if reduce is None:
        tmp = pd.DataFrame(data={'obs': obs
                                 , 'pred': pred
                                 , 'group': group})

    else:
        if len(obs) != len(reduce):
            fermatrica_error(
                "Error in calculation of R^2 by group: length of observed vector is not equal to length of reduce vector")

        tmp = pd.DataFrame(data={'obs': obs
                                 , 'pred': pred
                                 , 'group': group
                                 , 'reduce': reduce})

        tmp = tmp.groupby(['group', 'reduce'], as_index=False)[['obs', 'pred']].sum()

    tmp = rm_1_item_groups(tmp, 'group')
    rtrn = tmp.groupby('group').apply(lambda x: r_squared(x['obs'], x['pred'])).mean()

    return rtrn


def rmse_group(obs: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
               , pred: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
               , group: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
               , reduce: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list | None = None) -> float:
    """
    Calculate RMSE per group (e.g. per superbrand or per region)

    :param obs: vector with observed values (Y)
    :param pred: vector with predicted values (Y-hat)
    :param group: vector with grouping values, e.g. 'superbrand'
    :param reduce: if dataset's entity differs from `group`, additional vector to group by before metrics calculation,
        e.g. 'date'
    :return:
    """

    if len(obs) != len(pred):
        fermatrica_error("Error in calculation of RMSE by group: length of observed vector is not equal to length of predicted")
    if len(obs) != len(group):
        fermatrica_error("Error in calculation of RMSE by group: length of observed vector is not equal to length of grouping vector")

    if reduce is None:
        tmp = pd.DataFrame(data={'obs': obs
                                 , 'pred': pred
                                 , 'group': group})

    else:
        if len(obs) != len(reduce):
            fermatrica_error(
                "Error in calculation of RMSE by group: length of observed vector is not equal to length of reduce vector")

        tmp = pd.DataFrame(data={'obs': obs
                                 , 'pred': pred
                                 , 'group': group
                                 , 'reduce': reduce})

        tmp = tmp.groupby(['group', 'reduce'], as_index=False)[['obs', 'pred']].sum()

    tmp = rm_1_item_groups(tmp, 'group')
    rtrn = tmp.groupby('group').apply(lambda x: rmse(x['obs'], x['pred'])).mean()

    return rtrn


def mapef_group(obs: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
                , pred: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
                , group: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
                , reduce: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list | None = None) -> float:
    """
    Calculate MAPE-f (with 0-observations removed) per group (e.g. per superbrand or per region)

    :param obs: vector with observed values (Y)
    :param pred: vector with predicted values (Y-hat)
    :param group: vector with grouping values, e.g. 'superbrand'
    :param reduce: if dataset's entity differs from `group`, additional vector to group by before metrics calculation,
        e.g. 'date'
    :return:
    """

    if len(obs) != len(pred):
        fermatrica_error("Error in calculation of MAPE-f by group: length of observed vector is not equal to length of predicted")
    if len(obs) != len(group):
        fermatrica_error("Error in calculation of MAPE-f by group: length of observed vector is not equal to length of grouping vector")

    if reduce is None:
        tmp = pd.DataFrame(data={'obs': obs
                                 , 'pred': pred
                                 , 'group': group})

    else:
        if len(obs) != len(reduce):
            fermatrica_error(
                "Error in calculation of MAPE-f by group: length of observed vector is not equal to length of reduce vector")

        tmp = pd.DataFrame(data={'obs': obs
                                 , 'pred': pred
                                 , 'group': group
                                 , 'reduce': reduce})

        tmp = tmp.groupby(['group', 'reduce'], as_index=False)[['obs', 'pred']].sum()

    tmp = rm_1_item_groups(tmp, 'group')
    rtrn = tmp.groupby('group').apply(lambda x: mapef(x['obs'], x['pred'])).mean()

    return rtrn


"""
Non-error metrics
"""


def vif_worker(ds: pd.DataFrame
               , col: str):
    """
    Working function to calculate VIF. However, in some cases could be used directly, hence function is not protected

    :param ds: dataset
    :param col: column name
    :return: VIF value
    """

    md = OLS(ds[col], ds.loc[:, ds.columns != col])
    md = md.fit()

    rtrn = 1. / (1. - md.rsquared)

    return rtrn


def vif(model_cur) -> pd.Series:
    """
    Calculate VIF with OLS or similar model. `statsmodels` library is assumed to be used as engine for `model_cur`

    :param model_cur: `statsmodels` OLS model object
    :return: pandas series with VIFs as values and regressor names as series index
    """

    if (not hasattr(model_cur, 'model')) or (not hasattr(model_cur.model, 'data')) or (not hasattr(model_cur.model.data, 'orig_exog')):
        logging.error('VIF cannot be calculated. Supported model types: OLS, LME, LMEA. ' +
                      'Please skip this metrics or change model type to OLS, LME or LMEA')
        return 0

    ds = model_cur.model.data.orig_exog.copy()
    ds = add_constant(ds)

    cols = ds.columns

    rtrn = [vif_worker(ds, col) for col in cols]
    rtrn = pd.Series(rtrn, index=cols, name='vif')

    return rtrn




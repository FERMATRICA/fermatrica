"""
Metrics to be used in model evaluations. Do not confuse with `scoring` containing functions
to compose complex scoring from basic metrics defined here. Scoring is aimed more to facilitate algorithmic
optimisation, metrics could be interpreted by experts and event clients themselves.
"""


import logging
import numpy as np
import pandas as pd
from itertools import combinations

import statistics
from scipy.stats import norm
from statsmodels.tools import eval_measures as em
from statsmodels.regression.linear_model import OLS, OLSResults, RegressionResultsWrapper
from statsmodels.regression.mixed_linear_model import MixedLM, MixedLMResults, MixedLMResultsWrapper
from statsmodels.tools.tools import add_constant
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.weightstats import ztest as st_ztest
from statsmodels.tsa.stattools import adfuller as st_adfuller

from fermatrica_utils import rm_1_item_groups

from fermatrica.model.model import Model
from fermatrica.basics.basics import fermatrica_error


"""
Basic (atomic) error metrics and residuals model-free metrics
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


def eom(obs: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
        , pred: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list) -> float:
    """
    Equality of means (two samples t-test) between observed `obs` and predicted `pred` vectors:
    EOM = (mean(pred) - mean(obs)) / mean(obs)

    If abs(EOM) <= 10%, the result is good
    If 10% < abs(EOM) <= 20%, the result is not good enough
    If abs(EOM) > 20%, the result is bad

    :param obs: observed values (Y)
    :param pred: predicted values (Y-hat)
    :return:
    """

    if len(obs) != len(pred):
        fermatrica_error("Error in calculation of EOM (Equiality of means): " +
                         "length of observed vector is not equal to length of predicted")

    if not isinstance(obs, np.ndarray):
        obs = np.array(obs)
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)

    obs_mean = np.mean(obs)

    if obs_mean == 0:
        fermatrica_error("Error in calculation of EOM (Equiality of means): " +
                         "mean of observed vector is zero")

    rtrn = (np.mean(pred) - obs_mean) / obs_mean

    return rtrn


def ztest(obs: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
          , pred: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
          ) -> tuple | list:
    """
    Z-test for a systematic bias in prediction, used for test period mostly

    H0: the mean of residuals is NOT statistically different from 0 (no systematic overforecast or underforecast)
    H1: the mean of residuals is statistically different from 0 (there is a systematic overforecast or underforecast)
    So, P-value >= 5% is good, P-value < 5% is bad

    :param obs: observed values (Y)
    :param pred: predicted values (Y-hat)
    :return:
    """

    if len(obs) != len(pred):
        fermatrica_error("Error in calculation of z-test: " +
                         "length of observed vector is not equal to length of predicted")

    if not isinstance(obs, np.ndarray):
        obs = np.array(obs)
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)

    rtrn = st_ztest(obs - pred, x2=None, value=0, alternative='two-sided')

    return rtrn


"""
Grouped error and residuals model-free metrics
"""


def r_squared_group(obs: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
                    , pred: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
                    , group: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
                    , reduce: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list | None = None
                    , if_mean: bool = True
                    ) -> float | pd.Series:
    """
    Calculate R^2 per group (e.g. per superbrand or per region)

    :param obs: vector with observed values (Y)
    :param pred: vector with predicted values (Y-hat)
    :param group: vector with grouping values, e.g. 'superbrand'
    :param reduce: if dataset's entity differs from `group`, additional vector to group by before metrics calculation,
        e.g. 'date'
    :param if_mean: return mean result or per group
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
    rtrn = tmp.groupby('group').apply(lambda x: r_squared(x['obs'], x['pred']))

    if if_mean:
        rtrn = rtrn.mean()

    return rtrn


def rmse_group(obs: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
               , pred: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
               , group: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
               , reduce: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list | None = None
               , if_mean: bool = True) -> float | pd.Series:
    """
    Calculate RMSE per group (e.g. per superbrand or per region)

    :param obs: vector with observed values (Y)
    :param pred: vector with predicted values (Y-hat)
    :param group: vector with grouping values, e.g. 'superbrand'
    :param reduce: if dataset's entity differs from `group`, additional vector to group by before metrics calculation,
        e.g. 'date'
    :param if_mean: return mean result or per group
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

        tmp = tmp.groupby(['group', 'reduce'], as_index=False)[['obs', 'pred']].sum(skipna=False)

    tmp = rm_1_item_groups(tmp, 'group')
    rtrn = tmp.groupby('group').apply(lambda x: rmse(x['obs'], x['pred']))

    if if_mean:
        rtrn = rtrn.mean(skipna=False)

    return rtrn


def mapef_group(obs: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
                , pred: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
                , group: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
                , reduce: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list | None = None
                , if_mean: bool = True) -> float | pd.Series:
    """
    Calculate MAPE-f (with 0-observations removed) per group (e.g. per superbrand or per region)

    :param obs: vector with observed values (Y)
    :param pred: vector with predicted values (Y-hat)
    :param group: vector with grouping values, e.g. 'superbrand'
    :param reduce: if dataset's entity differs from `group`, additional vector to group by before metrics calculation,
        e.g. 'date'
    :param if_mean: return mean result or per group
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
    rtrn = tmp.groupby('group').apply(lambda x: mapef(x['obs'], x['pred']))

    if if_mean:
        rtrn = rtrn.mean()

    return rtrn


def mape_adj_group(obs: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
                   , pred: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
                   , group: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
                   , reduce: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list | None = None
                   , adj_val: int | float = .01
                   , if_mean: bool = True) -> float | pd.Series:
    """
    Calculate MAPE-adj (with 0-observations removed) per group (e.g. per superbrand or per region).

    Points with observations lesser than `adj_val` threshold are removed in this implementation to avoid 0-division
    and / or exclude points with hardly predictable small observed values

    :param obs: vector with observed values (Y)
    :param pred: vector with predicted values (Y-hat)
    :param adj_val: threshold to exclude small values / bottom outliers
    :param group: vector with grouping values, e.g. 'superbrand'
    :param reduce: if dataset's entity differs from `group`, additional vector to group by before metrics calculation,
        e.g. 'date'
    :param if_mean: return mean result or per group
    :return:
    """

    if len(obs) != len(pred):
        fermatrica_error("Error in calculation of MAPE-adj by group: length of observed vector is not equal to length of predicted")
    if len(obs) != len(group):
        fermatrica_error("Error in calculation of MAPE-adj by group: length of observed vector is not equal to length of grouping vector")

    if reduce is None:
        tmp = pd.DataFrame(data={'obs': obs
                                 , 'pred': pred
                                 , 'group': group})

    else:
        if len(obs) != len(reduce):
            fermatrica_error(
                "Error in calculation of MAPE-adj by group: length of observed vector is not equal to length of reduce vector")

        tmp = pd.DataFrame(data={'obs': obs
                                 , 'pred': pred
                                 , 'group': group
                                 , 'reduce': reduce})

        tmp = tmp.groupby(['group', 'reduce'], as_index=False)[['obs', 'pred']].sum()

    tmp = rm_1_item_groups(tmp, 'group')
    rtrn = tmp.groupby('group').apply(lambda x: mape_adj(x['obs'], x['pred'], adj_val=adj_val))

    if if_mean:
        rtrn = rtrn.mean()

    return rtrn


def eom_group(obs: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
              , pred: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
              , group: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
              , reduce: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list | None = None
              , if_mean: bool = True) -> float | pd.Series:
    """
    Calculate Equality of means (two samples t-test) per group (e.g. per superbrand or per region).
    EOM = (mean(pred) - mean(obs)) / mean(obs)

    If abs(EOM) <= 10%, the result is good
    If 10% < abs(EOM) <= 20%, the result is not good enough
    If abs(EOM) > 20%, the result is bad

    :param obs: vector with observed values (Y)
    :param pred: vector with predicted values (Y-hat)
    :param group: vector with grouping values, e.g. 'superbrand'
    :param reduce: if dataset's entity differs from `group`, additional vector to group by before metrics calculation,
        e.g. 'date'
    :param if_mean: return mean result or per group
    :return:
    """

    if len(obs) != len(pred):
        fermatrica_error("Error in calculation of EOM by group: length of observed vector is not equal to length of predicted")
    if len(obs) != len(group):
        fermatrica_error("Error in calculation of EOM by group: length of observed vector is not equal to length of grouping vector")

    if reduce is None:
        tmp = pd.DataFrame(data={'obs': obs
                                 , 'pred': pred
                                 , 'group': group})

    else:
        if len(obs) != len(reduce):
            fermatrica_error(
                "Error in calculation of EOM by group: length of observed vector is not equal to length of reduce vector")

        tmp = pd.DataFrame(data={'obs': obs
                                 , 'pred': pred
                                 , 'group': group
                                 , 'reduce': reduce})

        tmp = tmp.groupby(['group', 'reduce'], as_index=False)[['obs', 'pred']].sum()

    tmp = rm_1_item_groups(tmp, 'group')
    rtrn = tmp.groupby('group').apply(lambda x: eom(x['obs'], x['pred']))

    if if_mean:
        rtrn = rtrn.mean()

    return rtrn


def ztest_group(obs: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
                , pred: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
                , group: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
                , reduce: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list | None = None
                , if_mean: bool = True) -> float | pd.Series:
    """
    Calculate Z-test for a systematic bias in prediction per group (e.g. per superbrand or per region).

    H0: the mean of residuals is NOT statistically different from 0 (no systematic overforecast or underforecast)
    H1: the mean of residuals is statistically different from 0 (there is a systematic overforecast or underforecast)
    So, P-value >= 5% is good, P-value < 5% is bad

    :param obs: vector with observed values (Y)
    :param pred: vector with predicted values (Y-hat)
    :param group: vector with grouping values, e.g. 'superbrand'
    :param reduce: if dataset's entity differs from `group`, additional vector to group by before metrics calculation,
        e.g. 'date'
    :param if_mean: return mean result or per group
    :return:
    """

    if len(obs) != len(pred):
        fermatrica_error("Error in calculation of Z-test by group: length of observed vector is not equal to length of predicted")
    if len(obs) != len(group):
        fermatrica_error("Error in calculation of Z-test by group: length of observed vector is not equal to length of grouping vector")

    if reduce is None:
        tmp = pd.DataFrame(data={'obs': obs
                                 , 'pred': pred
                                 , 'group': group})

    else:
        if len(obs) != len(reduce):
            fermatrica_error(
                "Error in calculation of Z-test by group: length of observed vector is not equal to length of reduce vector")

        tmp = pd.DataFrame(data={'obs': obs
                                 , 'pred': pred
                                 , 'group': group
                                 , 'reduce': reduce})

        tmp = tmp.groupby(['group', 'reduce'], as_index=False)[['obs', 'pred']].sum()

    tmp = rm_1_item_groups(tmp, 'group')
    rtrn = tmp.groupby('group').apply(lambda x: ztest(x['obs'], x['pred']))

    if if_mean:
        rtrn = rtrn.apply(lambda x: x[0]).mean()

    return rtrn


"""
Non-error or model specific metrics
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


def durbin_watson(model: "Model") -> int | float:
    """
    Durbin-Watson test for autocorrelation. Assumes OLS linear model as `model.obj.models['main']`

    DW-stat can be from 0 to 4
    If DW-stat is close to 2, there is no autocorrelation in residuals
    If DW-stat is close to 0, there is positive autocorrelation in residuals
    If DW-stat is close to 4, there is negative autocorrelation in residuals

    H0: no first order autocorrelation in residuals
    H1: there is first order autocorrelation in residuals
    So, P-value >= 5% is good, P-value < 5% is bad

    :param model: FERMATRICA Model object
    :return:
    """

    if not isinstance(model.obj.models['main'].model, OLS):
        logging.error('Durbin-Watson cannot be calculated. Supported model types: OLS. Supported data type: time series. ' +
                      'Please change the model type to OLS and be sure data is a time series (not a panel nor slices).')
        return 0

    residuals = model.obj.models['main'].resid
    resid_df = pd.DataFrame([residuals, residuals.shift(1)]).T
    coef = resid_df.corr()[0][1]

    rtrn = 2 * (1 - coef)

    return rtrn


def ljungbox(obs_resid: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list = None
             , pred: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list | None = None
             , model: "Model | None" = None
             , group: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list | None = None
             , reduce: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list | None = None):
    """
    Ljung-Box test for autocorrelation. Assumes OLS or LME linear model as `model.obj.models['main']`

    H0: no first order autocorrelation in residuals
    H1: there is first order autocorrelation in residuals
    So, P-value >= 5% is good, P-value < 5% is bad

    :param obs_resid: vector with observed values (Y) or residulas vector if `pred` is `None`
    :param pred: vector with predicted values (Y-hat) or `None`
    :param model: FERMATRICA Model object or `None`. If `obs` is provided, `model` argument is ignored
    :param group: vector with grouping values, e.g. 'superbrand'
    :param reduce: if dataset's entity differs from `group`, additional vector to group by before metrics calculation,
        e.g. 'date'
    :return:
    """

    # calculate residuals if not provided directly

    if model is not None and obs_resid is None:

        if not isinstance(model.obj.models['main'].model, OLS | OLSResults | MixedLM | MixedLMResults | MixedLMResultsWrapper):
            logging.error(
                'Ljung-Box cannot be calculated. Supported model types: OLS, LME. ' +
                'Please change the model type to OLS or LME.')
            return 0

        obs_resid = model.obj.models['main'].model.fit().resid
        pred = None

    if pred is not None and len(obs_resid) != len(pred):
        fermatrica_error("Error in calculation of Ljung-Box: length of observed vector is not equal to length of predicted")
    if group is not None and len(obs_resid) != len(group):
        fermatrica_error("Error in calculation of Ljung-Box by group: " +
                         "length of observed vector is not equal to length of grouping vector")

    if pred is not None:
        obs_resid = obs_resid - pred

    if group is None:
        group = np.zeros(len(obs_resid))

    # combine

    if reduce is None:
        tmp = pd.DataFrame(data={'obs_resid': obs_resid
                                 , 'group': group})

    else:
        if len(obs_resid) != len(reduce):
            fermatrica_error(
                "Error in calculation of Ljung-Box by group: length of observed vector is not equal to length of reduce vector")

        tmp = pd.DataFrame(data={'obs_resid': obs_resid
                                 , 'group': group
                                 , 'reduce': reduce})

        tmp = tmp.groupby(['group', 'reduce'], as_index=False)[['obs_resid']].sum()

    tmp = rm_1_item_groups(tmp, 'group')
    rtrn = tmp.groupby('group').apply(lambda x: acorr_ljungbox(x['obs_resid'], lags=[1], return_df=True))

    return rtrn


def adfuller(obs_resid: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list = None
             , pred: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list | None = None
             , model: "Model | None" = None
             , group: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list | None = None
             , reduce: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list | None = None):
    """
    Augmented Dickey-Fuller unit root test. Assumes OLS or LME linear model as `model.obj.models['main']`

    H0: there is a unit root in residuals (non-stationarity)
    H1: residuals is a stationary process
    So, P-value <= 1% is good, P-value > 1% is bad

    :param obs_resid: vector with observed values (Y) or residulas vector if `pred` is `None`
    :param pred: vector with predicted values (Y-hat) or `None`
    :param model: FERMATRICA Model object or `None`. If `obs` is provided, `model` argument is ignored
    :param group: vector with grouping values, e.g. 'superbrand'
    :param reduce: if dataset's entity differs from `group`, additional vector to group by before metrics calculation,
        e.g. 'date'
    :return:
    """

    # calculate residuals if not provided directly

    if model is not None and obs_resid is None:

        if not isinstance(model.obj.models['main'].model, OLS | OLSResults | MixedLM | MixedLMResults | MixedLMResultsWrapper):
            logging.error(
                'Augmented Dickey-Fuller cannot be calculated. Supported model types: OLS, LME. ' +
                'Please change the model type to OLS or LME.')
            return 0

        obs_resid = model.obj.models['main'].model.fit().resid
        pred = None

    if pred is not None and len(obs_resid) != len(pred):
        fermatrica_error("Error in calculation of Augmented Dickey-Fuller: length of observed vector is not equal to length of predicted")
    if group is not None and len(obs_resid) != len(group):
        fermatrica_error("Error in calculation of Augmented Dickey-Fuller by group: " +
                         "length of observed vector is not equal to length of grouping vector")

    if pred is not None:
        obs_resid = obs_resid - pred

    if group is None:
        group = np.zeros(len(obs_resid))

    # combine

    if reduce is None:
        tmp = pd.DataFrame(data={'obs_resid': obs_resid
                                 , 'group': group})

    else:
        if len(obs_resid) != len(reduce):
            fermatrica_error(
                "Error in calculation of Augmented Dickey-Fuller by group: length of observed vector is not equal to length of reduce vector")

        tmp = pd.DataFrame(data={'obs_resid': obs_resid
                                 , 'group': group
                                 , 'reduce': reduce})

        tmp = tmp.groupby(['group', 'reduce'], as_index=False)[['obs_resid']].sum()

    tmp = rm_1_item_groups(tmp, 'group')

    # calculate

    rtrn = tmp.groupby('group').apply(lambda x: st_adfuller(x['obs_resid'], autolag="t-stat", regression="ct")).reset_index()

    rtrn[['adf', 'p_value', 'usedlag', 'nobs', 'critical values', 'icbest']] = rtrn[0].tolist()
    rtrn.drop(columns=[0], inplace=True)

    return rtrn


def pesaran_cd(obs_resid: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list = None
               , pred: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list | None = None
               , model: "Model | None" = None
               , group: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list | None = None
               , date: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list | None = None):
    """
    Pesaran cross-sectional dependence test. Assumes OLS or LME linear model as `model.obj.models['main']`

    H0: no cross-section correlation in residuals (no correlation in resuduals among entities)
    H1: there is a cross-section correlation in residuals (correlation in resuduals among entities)
    So, P-value >= 5% is good, P-value < 5% is bad

    :param obs_resid: vector with observed values (Y) or residulas vector if `pred` is `None`
    :param pred: vector with predicted values (Y-hat) or `None`
    :param model: FERMATRICA Model object or `None`. If `obs` is provided, `model` argument is ignored
    :param group: vector with grouping values, e.g. 'superbrand'. Could be `None` only if `model` is provided
    :param date: date vector, obligatory if `model` is not provided
    :return:
    """

    # calculate residuals if not provided directly

    if model is not None and obs_resid is None:

        if not isinstance(model.obj.models['main'].model, MixedLM | MixedLMResults | MixedLMResultsWrapper | OLS | OLSResults | RegressionResultsWrapper):
                logging.error(
                    'Pesaran cross-sectional dependence test cannot be calculated. Supported model types: LME, OLS for panels. ' +
                    'Please change the model type to LME or OLS with groups.')
                return 0


        obs_resid = model.obj.models['main'].model.fit().resid
        pred = None

    if group is None:
        if model is None:
            fermatrica_error("Error in calculation of Pesaran cross-sectional dependence test: " +
                             "Grouping vector nor model object is not provided")
        elif isinstance(model.obj.models['main'].model, OLS | OLSResults | RegressionResultsWrapper):
            logging.error(
                'Pesaran cross-sectional dependence test cannot be calculated for OLS not for panels (without groups). ')
        group = model.obj.models['main'].model.groups

    if pred is not None and len(obs_resid) != len(pred):
        fermatrica_error("Error in calculation of Pesaran cross-sectional dependence test: length of observed vector is not equal to length of predicted")
    if len(obs_resid) != len(group):
        fermatrica_error("Error in calculation of Pesaran cross-sectional dependence test: " +
                         "length of observed vector is not equal to length of grouping vector")

    if pred is not None:
        obs_resid = obs_resid - pred

    if date is None:
        date = model.obj.models['main'].model.data.frame['date']

    if len(obs_resid) != len(date):
        fermatrica_error(
            "Error in calculation of Pesaran cross-sectional dependence test: length of observed vector is not equal to length of reduce vector")

    # combine

    tmp = pd.DataFrame(data={'obs_resid': obs_resid
                             , 'group': group
                             , 'date': date})

    tmp = tmp.groupby(['group', 'date'], as_index=False)[['obs_resid']].sum()

    tmp = rm_1_item_groups(tmp, 'group')

    # calculate

    group_sizes = tmp.groupby('group').size()

    tmp = pd.pivot_table(tmp, index='date', values='obs_resid', columns='group')

    if group_sizes.max() == group_sizes.min():
        # balanced panel

        t = tmp.shape[0]
        n = tmp.shape[1]

        # correlations

        corr_matrix = np.corrcoef(tmp, rowvar=False)

        triu_indices = np.triu_indices(n, k=1)
        correlations = corr_matrix[triu_indices]
        sum_corr = np.sum(correlations)

        # main calculation

        cd_value = sum_corr * np.sqrt(2 * t / (n * (n - 1)))

    else:
        # unbalanced panel

        ent_pairs = list(combinations(tmp.columns, 2))

        corr_w_list = []

        for ep in ent_pairs:
            tmp_cur = tmp.loc[(tmp[ep[0]].notna()) & (tmp[ep[1]].notna()), ep]
            t = tmp_cur.shape[0]

            if t > 1:
                corr_cur = np.corrcoef(tmp_cur, rowvar=False)[0, 1]
                if not np.isnan(corr_cur):
                    # nans could be produced by vectors w/o variability etc.
                    corr_w_list.append(corr_cur * np.sqrt(t))

        if len(corr_w_list) == 0:
            fermatrica_error("Error in calculation of Pesaran cross-sectional dependence test: " +
                             "No intersections between series of the unbalanced panel")

        sum_corr = np.nansum(corr_w_list)
        n = len(corr_w_list)

        cd_value = sum_corr * np.sqrt(2 / (n * (n - 1)))

    p_value = 2 * (1 - norm.cdf(np.abs(cd_value)))

    return {'cd_value': cd_value, 'p_value': p_value}


def poolability(model: "Model"
                , type: str = "entity"
                , entity_var: str | None = None
                , lsdv_vars: list | None = None):
    """
    Panel specification test. Assumes OLS or LME linear model as `model.obj.models['main']`

    model_ols - pool regression with no individual effects
    model_intercept FE regression with individual effects on the intercept
    model_slope regression with individual effects on the intercept and slopes (identical to separate regressions)

    intercept_ols - F-test that model_ols is equal to model_intercept
    slope_intercept - F-test that model_intercept is equal to model_slope
    slope_ols - F-test that model_ols is equal to model_slope

    For each test
    H0: models are statistically equal, we choose the simpliest model (restricted)
    H1: models are statistically different, we choose the unrestricated model
    So, P-value >= 5% means that models are equal, P-value < 5% means that models are not equal

    If at least 2 of 3 tests has P-value >= 5% than data are poolable (we can analyse them as a panel)

    :param model: FERMATRICA Model object
    :param type: type of panel specification test. Can be "entity" or "time" / "date"
    :param entity_var: variable containing entities (required for OLS only, otherwise real groups are used)
    :param lsdv_vars: list of dummy variables from RHS used in panel LSDV model to account for fixed individual effects (entities or time) (required for LSDV model only)
    :return:
    """

    if not isinstance(model.obj.models['main'].model, OLS | OLSResults | MixedLM | MixedLMResults | MixedLMResultsWrapper):
        logging.error(
            'Poolability test cannot be calculated. Supported model types: OLS, LME. ' +
            'Please change the model type to OLS, LME.')
        return 0

    from statsmodels.formula.api import ols

    data_wrk = model.obj.models['main'].model.data.frame.copy()
    data_wrk['date'] = data_wrk['date'].astype(str)

    # create models to compare

    model_base_spec = model.obj.models['main'].model.formula
    model_base_y = model.obj.models['main'].model.endog_names
    model_base_x = model.obj.models['main'].model.exog_names

    model_base_x = list(set(model_base_x) - set(lsdv_vars))

    lsdv_part_spec = ' - '.join(lsdv_vars)
    model_base_spec = model_base_spec + ' - ' + lsdv_part_spec

    model_ols = ols(model_base_spec, data=data_wrk)

    if type == 'entity':

        if entity_var is None and isinstance(model.obj.models['main'].model, MixedLM | MixedLMResults | MixedLMResultsWrapper):
            entity_var = 'entity__'
            data_wrk[entity_var] = model.obj.models['main'].model.groups
        elif entity_var is None:
            logging.error(
                'Poolability test cannot be calculated. Provide LME model or entity_var name for OLS model.')
            return 0

        data_wrk[entity_var] = data_wrk[entity_var].astype(str)

        model_intercept_spec = model_base_spec + ' + ' + entity_var
        model_intercept = ols(model_intercept_spec, data=data_wrk)

        model_slope_spec = model_base_y + ' ~ ' + entity_var + ' + ' + ' + '.join(
            [x + ':' + entity_var for x in model_base_x if x != 'Intercept']
        )
        model_slope = ols(model_slope_spec, data=data_wrk)

    else:
        model_intercept_spec = model_base_spec + ' + date'
        model_intercept = ols(model_intercept_spec, data=data_wrk)

        model_slope_spec = model_base_y + ' ~ date + ' + ' + '.join(
            [x + ':date' for x in model_base_x if x != 'Intercept']
        )
        model_slope = ols(model_slope_spec, data=data_wrk)

    # run models to compare

    model_ols = model_ols.fit()
    model_intercept = model_intercept.fit()
    model_slope = model_slope.fit()

    # tests

    intercept_ols = model_intercept.compare_f_test(model_ols)
    slope_intercept = model_slope.compare_f_test(model_intercept)
    slope_ols = model_slope.compare_f_test(model_ols)

    rtrn = {'intercept_ols': intercept_ols, 'slope_intercept': slope_intercept, 'slope_ols': slope_ols}
    rtrn = pd.DataFrame.from_dict(rtrn, orient='index', columns=['f_test', 'p_value', 'diff'])

    return rtrn



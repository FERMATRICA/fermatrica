"""
Complex scoring system is aimed more to facilitate algorithmic optimisation. Do not confuse with `metrics`.
Metrics are more basic things used both as parts of the scoring and as independent indicators of model
quality intuitive enough for humans
"""


import copy
import inspect
import logging
import numpy as np
import pandas as pd
import re

from statsmodels.regression.linear_model import OLS

from fermatrica.basics.basics import fermatrica_error
from fermatrica.model.model import Model
import fermatrica.evaluation.metrics as mtr


"""
Combined scoring
"""


def scoring(ds: pd.DataFrame
            , pred: pd.Series | np.ndarray
            , model: "Model"):
    """
    Main function to create complex scoring from the set of individual scorings / metrics

    :param ds: dataset containing Y (observed) variable
    :param pred: vector with predicted values
    :param model: FERMATRICA Model object
    :return:
    """

    obs = ds[model.conf.Y_var]
    scoring_dict = copy.deepcopy(model.conf.scoring_dict)

    # iterate over keys in scoring_dict
    for key in scoring_dict:

        fn = globals()[key]

        if ('ds' in inspect.signature(fn).parameters.keys()) and ('model' in inspect.signature(fn).parameters.keys()):
            score = fn(obs, pred, ds, model)
        elif (len(inspect.signature(fn).parameters.keys()) == 1) and ('model' in inspect.signature(fn).parameters.keys()):
            score = fn(model)
        elif 'ds' in inspect.signature(fn).parameters.keys():
            score = fn(obs, pred, ds)
        else:
            score = fn(obs, pred)

        if (not hasattr(model.conf, 'custom_scoring')) or (model.conf.custom_scoring == ''):

            # Calculate mapping of the score
            score = bell_curve(score, scoring_dict[key]['if_invert'], scoring_dict[key]['width'])

            if (isinstance(score, list)) or (isinstance(score, np.ndarray)):
                score = np.average(score)

        scoring_dict[key]['value'] = score

    # calculate final value

    if (not hasattr(model.conf, 'custom_scoring')) or (model.conf.custom_scoring == ''):
        rtrn = sum(scoring_dict[k]['value'] * scoring_dict[k]['weight'] for k in scoring_dict)

    else:
        try:
            rtrn = eval(model.conf.custom_scoring_compiled)
        except (RuntimeError, TypeError, NameError):
            fermatrica_error('Custom scoring function failed. Check if all variables are set ' +
                         '`if_active` = 1 in the `scoring` sheet of the config / model definition file.')

    return rtrn


def bell_curve(x: list | np.ndarray
               , if_invert: int = 0
               , width: float = 1):
    """
    Function to normalise raw individual scores / metrics to 0...1 range using bell curve

    :param x:
    :param if_invert: Should we invert parameter? I.e. RMSE - the smaller, the better, but combined score is maximising
        1 = yes
        0 = no
    :param width:
    :return:
    """

    if isinstance(x, list):
        x = np.array(x)

    # function requires inversion of if_invert (ha-ha), so 0 -> 1 and all other values -> 0
    if_invert = np.where(np.abs(if_invert) > 0, 0, 1)

    rtrn = np.exp(-(x - if_invert) ** 2 / 2 * (1 / width) ** 2)

    return rtrn


"""
Model-free error metrics
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
    return mtr.r_squared(obs, pred)


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

    return mtr.rmse(obs, pred)


def rmse_bs_key(obs: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
                , pred: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
                , ds: pd.DataFrame) -> float:
    """
    Calculate RMSE per `bs_key` (mostly SKU or region)

    :param obs: vector with observed values (Y)
    :param pred: vector with predicted values (Y-hat)
    :param ds: dataset containing `bs_key` column
    :return:
    """

    return mtr.rmse_group(obs, pred, ds['bs_key'])


def r_squared_bs_key(obs: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
                     , pred: pd.Series | pd.api.extensions.ExtensionArray | np.ndarray | list
                     , ds: pd.DataFrame) -> float:
    """
    Calculate R^2 per `bs_key` (mostly SKU or region)

    :param obs: vector with observed values (Y)
    :param pred: vector with predicted values (Y-hat)
    :param ds: dataset containing `bs_key` column
    :return:
    """

    return mtr.r_squared_group(obs, pred, ds['bs_key'])


"""
Model-specific error metrics
"""


def r_squared_adjusted(model: "Model") -> float | int:
    """
    Adjusted R^2 (discounted according to number of the independent variables in linear model).
    As for now supported only for OLS (ordinary least squares linear model).

    :param model: FERMATRICA Model object
    :return:
    """

    if hasattr(model.obj.models['main'], 'rsquared_adj'):
        rtrn = model.obj.models['main'].rsquared_adj
    else:
        rtrn = 0
        logging.error('Adjusted R^2 cannot be calculated. Supported model types: OLS. ' +
                      'Please change metrics according to your model type or change model type to OLS')

    return rtrn


"""
Non-error metrics
"""


def sign_simple(model: "Model") -> int | float:
    """
    Check if all regression coefficient signs are correct (correspond to RHS definition in Model).
    Wrong signs are penalised with 1 * sign_weight for every occurrence, result is weighted by the
    number of regressors.

    This function doesn't take into account how big "error" is, for more elaborate approach try
    `sign_t_value` and `sign_t_value_s`.

    :param model: FERMATRICA Model object
    :return:
    """

    model_rhs = model.conf.model_rhs[['token', 'sign', 'sign_weight']].copy()
    model_rhs['token'] = model_rhs['token'].apply(lambda x: re.sub(r' : ', ':', x))
    # maybe squeeze all spaces - check later

    rtrn = pd.merge(model.obj.models['main'].tvalues.to_frame('t_value').reset_index().rename(columns={'index': 'token'})
                    , model_rhs
                    , how='inner', on=['token'], sort=False)

    rtrn = np.sign(rtrn['t_value']) * rtrn['sign'] * rtrn['sign_weight']
    rtrn = np.abs(rtrn[rtrn < 0].sum()) / model.obj.models['main'].tvalues.shape[0]

    return rtrn


def sign_t_value(model: "Model") -> float | int:
    """
    Check if all regression coefficient signs are correct (correspond to RHS definition in Model).
    Wrong signs are penalised with T-value * sign_weight for every occurrence, result is weighted by the
    number of regressors.

    T-value allows to measure how big the "error" is, so facilitates algorithmic optimisation greatly.
    T-value is preferred over p-value due to the already existing sign and range [-Inf, Inf] rather
    than [0, Inf].

    For simpler approach see `sign_simple`, for even more sophisticated - `sign_t_value_s`

    :param model: FERMATRICA Model object
    :return:
    """

    model_rhs = model.conf.model_rhs[['token', 'sign', 'sign_weight']].copy()
    model_rhs['token'] = model_rhs['token'].apply(lambda x: re.sub(r' : ', ':', x))
    # maybe squeeze all spaces - check later

    rtrn = pd.merge(model.obj.models['main'].tvalues.to_frame('t_value').reset_index().rename(columns={'index': 'token'})
                    , model_rhs
                    , how='inner', on=['token'], sort=False)

    rtrn = rtrn['t_value'] * rtrn['sign'] * rtrn['sign_weight']
    rtrn = np.abs(rtrn[rtrn < 0].sum()) / model.obj.models['main'].tvalues.shape[0]

    return rtrn


def sign_t_value_s(model: "Model") -> float | int:
    """
    Check if all regression coefficient signs are correct (correspond to RHS definition in Model) and
    estimations are significant. Wrong signs and not enough significance are penalised with
    (T-value - 2) * sign_weight for every occurrence, result is weighted by the number of regressors.

    T-value allows to measure how big the "error" is, so facilitates algorithmic optimisation greatly.
    T-value is preferred over p-value due to the already existing sign and range [-Inf, Inf] rather
    than [0, Inf].

    For simpler approaches see `sign_simple` and `sign_t_value_s`, for more flexible - `sign_t_value_st`

    :param model: FERMATRICA Model object
    :return:
    """

    model_rhs = model.conf.model_rhs[['token', 'sign', 'sign_weight']].copy()
    model_rhs['token'] = model_rhs['token'].apply(lambda x: re.sub(r' : ', ':', x))
    # maybe squeeze all spaces - check later

    rtrn = pd.merge(model.obj.models['main'].tvalues.to_frame('t_value').reset_index().rename(columns={'index': 'token'})
                    , model_rhs
                    , how='inner', on=['token'], sort=False)

    rtrn = (rtrn['t_value'] * rtrn['sign'] - 2.0) * rtrn['sign_weight']
    rtrn = np.abs(rtrn[rtrn < 0].sum()) / model.obj.models['main'].tvalues.shape[0]

    return rtrn


def sign_t_value_st(model: "Model") -> float | int:
    """
    Check if all regression coefficient signs are correct (correspond to RHS definition in Model) and
    estimations are significant. Wrong signs and not enough significance are penalised with
    (T-value - threshold) * sign_weight for every occurrence, result is weighted by the number of regressors.

    T-value allows to measure how big the "error" is, so facilitates algorithmic optimisation greatly.
    T-value is preferred over p-value due to the already existing sign and range [-Inf, Inf] rather
    than [0, Inf].

    Thresholds should be defined in Model.model_rhs DataFrame in `signif_threshold` column.

    For less flexible approach (threshold is allways equal to 2) see `sign_t_value_s`

    :param model: FERMATRICA Model object
    :return:
    """

    model_rhs = model.conf.model_rhs[['token', 'sign', 'sign_weight', 'signif_threshold']].copy()
    model_rhs['token'] = model_rhs['token'].apply(lambda x: re.sub(r' : ', ':', x))
    # maybe squeeze all spaces - check later

    rtrn = pd.merge(model.obj.models['main'].tvalues.to_frame('t_value').reset_index().rename(columns={'index': 'token'})
                    , model_rhs
                    , how='inner', on=['token'], sort=False)

    rtrn = (rtrn['t_value'] * rtrn['sign'] - rtrn['signif_threshold']) * rtrn['sign_weight']
    rtrn = np.abs(rtrn[rtrn < 0].sum()) / model.obj.models['main'].tvalues.shape[0]

    return rtrn


def t_value(model: "Model") -> float | int:
    """
    Check if all regression coefficient estimations are significant by t-values.
    Not enough significance is penalised with ABS(T-value * signif_weight) for every occurrence,
    result is weighted by the number of regressors.

    T-value allows to measure how big the "error" is, so facilitates algorithmic optimisation greatly.
    T-value is preferred over p-value due to the already existing sign and range [-Inf, Inf] rather
    than [0, Inf].

    :param model: FERMATRICA Model object
    :return:
    """

    model_rhs = model.conf.model_rhs[['token', 'signif_weight']].copy()
    model_rhs['token'] = model_rhs['token'].apply(lambda x: re.sub(r' : ', ':', x))
    # maybe squeeze all spaces - check later

    rtrn = pd.merge(model.obj.models['main'].tvalues.to_frame('t_value').reset_index().rename(columns={'index': 'token'})
                    , model_rhs
                    , how='inner', on=['token'], sort=False)

    rtrn = rtrn['t_value'] * rtrn['signif_weight']
    rtrn = rtrn.abs().sum() / model.obj.models['main'].tvalues.shape[0]

    return rtrn


def p_value(model: "Model") -> float | int:
    """
    Check if all regression coefficient estimations are significant by p-values.
    Not enough significance to be calculated as (p-value * signif_weight) for every occurrence,
    result is weighted by the number of regressors.

    IMPORTANT. In this function the smaller is the result, the better.

    :param model: FERMATRICA Model object
    :return:
    """

    model_rhs = model.conf.model_rhs[['token', 'signif_weight']].copy()
    model_rhs['token'] = model_rhs['token'].apply(lambda x: re.sub(r' : ', ':', x))
    # maybe squeeze all spaces - check later

    if not hasattr(model.obj.models['main'], 'pvalues'):
        logging.error('P-values cannot be calculated. Supported model types: OLS, LME, LMEA. ' +
                      'Please use t-values instead or change model type to OLS, LME or LMEA')
        return 0

    rtrn = pd.merge(model.obj.models['main'].pvalues.to_frame('p_value').reset_index().rename(columns={'index': 'token'})
                    , model_rhs
                    , how='inner', on=['token'], sort=False)

    rtrn = rtrn['p_value'] * rtrn['signif_weight']
    rtrn = rtrn.abs().sum() / model.obj.models['main'].tvalues.shape[0]

    return rtrn


def durbin_watson(model: "Model") -> int | float:
    """
    Durbin-Watson test for autocorrelation. Assumes OLS linear model as `model.obj.models['main']`

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


def durbin_watson_score(model: "Model"):
    """
    Durbin-Watson test for autocorrelation. Assumes OLS linear model as `model.obj.models['main']`.
    Specific version used mostly for combined scoring

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

    rtrn = abs(2 * (1 - coef) - 2)

    return rtrn


def vif(model: "Model"):
    """
    Calculate VIF with OLS or similar model. `statsmodels` library is assumed to be used as engine for `model_cur`

    :param model: FERMATRICA Model object
    :return:
    """

    return mtr.vif(model.obj.models['main']).mean()


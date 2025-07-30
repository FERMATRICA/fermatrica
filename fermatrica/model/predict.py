"""
Build inner layer of FERMATRICA Model and run LHA part of the outer layer and total inner layer to get prediction.

Architecture could seem a bit unusual: fit and predict operations are combined in the sole function
`fit_predict()` with boolean argument `if_fit`. It is intentional to boost performance: due to
existence of the LHS transformations separate fit would require extra run of the LHS.

Another part of the outer layer (X transformations) is separated into `transform` module and runs separately
to boost performance as well (it is by far the most "expensive" part of the model run). So to build the model
someone needs to run `transform()` and then `fit_predict()` with `if_fit=True` argument.

Useful links:

    statsmodels: https://www.statsmodels.org/stable/example_formulas.html
        https://www.statsmodels.org/dev/examples/notebooks/generated/ols.html
        https://www.statsmodels.org/dev/examples/notebooks/generated/mixed_lm_example.html
    linearmodels: https://bashtage.github.io/linearmodels/panel/panel/linearmodels.panel.model.PanelOLS.html
        main issue : Pandas indices used to define fixed effects

"""


import copy
import inspect
import numpy as np
import pandas as pd
import warnings

from line_profiler_pycharm import profile

import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from fermatrica_utils import DotDict, groupby_eff, select_eff

import fermatrica.basics.basics
from fermatrica.basics.basics import fermatrica_error
from fermatrica.model.model import Model
from fermatrica.model.model_obj import fun_generate, fun_find
from fermatrica.model.transform import transform

warnings.filterwarnings(action='error', category=ConvergenceWarning
                        , message=".*essian matrix at the estimated parameter values is not positive definite.*")
warnings.filterwarnings(action='error', category=UserWarning
                        , message=".*andom effects covariance (matrix )*is singular.*")


@profile
def predict_lme(ds: pd.DataFrame
                , model: "Model"):
    """
    Specific prediction for LME inner model.

    Statsmodels LME predicts only fixed effects (i.e. shared slopes), so random effects (group specific)
    to be added manually.

    :param ds: dataset
    :param model: Model object
    :return: predicted series
    """

    # extract from Model

    model_obj = model.obj.models['main']
    fixed_effect_var = model.conf.fixed_effect_var

    # get prediction

    pred_random = pd.DataFrame.from_dict(model_obj.random_effects, orient='index').reset_index().rename(
        columns={'index': fixed_effect_var})
    pred_fixed = model_obj.predict(ds)

    # combine random and fixed parts together

    pred = pd.merge(ds[fixed_effect_var].reset_index(), pred_random, how='left', on=fixed_effect_var
                    , sort=False).set_index('index')
    pred.loc[pred['Group'].isna(), 'Group'] = 0

    pred = pred['Group'] + pred_fixed

    return pred


@profile
def _predict_worker(ds: pd.DataFrame
                    , model: "Model"
                    , if_fit: bool = True
                    , if_verbose: bool = False
                    , return_full: bool = False):
    """
    Fit and predict main worker function

    :param ds: dataset
    :param model: Model object
    :param if_fit: fit (create / re-create model) or predict using existing model
    :param if_verbose: print / log diagnostic info. Suppress for optimisation
    :param return_full: return model, raw (inner) prediction, final prediction or just final prediction
    :return: predicted series or tuple of predicted series, raw (inner) predicted series and Model object
    """

    if model.obj is None:
        model.obj = {}

    # preliminary model(s)

    custom_predict_fn = model.obj.custom_predict_fn

    # use for multiprocess cases
    if custom_predict_fn is None and model.obj.custom_predict_nm is not None:
        custom_predict_fn = fun_find(model.obj.custom_predict_nm, model.obj.adhoc_code_src)

    if custom_predict_fn is not None:

        try:
            model.obj.models = custom_predict_fn(ds, model, if_fit)
        except Exception as e:
            if if_verbose:
                fermatrica_error('Custom predict function failed. It could be category model or something. ' +
                                 'Error might be inside function itself')
            if return_full:
                return None, None, None
            else:
                return None

        if model.obj.models is None:
            if if_verbose:
                fermatrica_error('Custom predict function failed. It could be category model or something')
            if return_full:
                return None, None, None
            else:
                return None

        elif not isinstance(model.obj.models, dict):
            if if_verbose:
                fermatrica_error('Custom predict function returns unexpected type. ' +
                                 'Please be sure if it returns model.obj.models and not model')
            if return_full:
                return None, None, None
            else:
                return None

    # main model : LHS transformation

    params = model.conf.params
    coef = params[((params['variable'] == '') | (params['variable'].isna())) &
                  (params['fun'] == 'coef')][['arg', 'value']]
    coef_dict = fermatrica.basics.basics.params_to_dict(coef)

    transform_lhs_fn = model.obj.transform_lhs_fn
    transform_lhs_src = model.obj.transform_lhs_src

    # use for multiprocess cases
    if transform_lhs_fn is None and isinstance(transform_lhs_src, list):
        transform_lhs_fn = fun_generate(transform_lhs_src, 'transform_lhs_fn')

    y_var = model.conf.Y_var

    if transform_lhs_fn is None and 'y__' not in ds.columns:
        ds['y__'] = ds.loc[:, y_var]
    elif transform_lhs_fn is None and 'y__' in ds.columns:
        ds.loc[:, 'y__'] = ds.loc[:, y_var]
    elif transform_lhs_fn is not None and 'y__' in ds.columns:
        ds.loc[:, 'y__'] = transform_lhs_fn(ds, ds.loc[:, y_var], coef_dict, to_original=False)
    else:
        ds['y__'] = transform_lhs_fn(ds, ds.loc[:, y_var], coef_dict, to_original=False)

    # main model : RHS & fit

    if if_fit:

        frm = 'y__ ~ ' + model.conf.model_rhs[model.conf.model_rhs['if_active'] == 1]['token'].str.cat(sep=' + ')
        ds_train = ds[ds['listed'] == 2]

        match model.conf.model_type:

            case 'OLS':

                model.obj.models['main'] = smf.ols(frm, ds_train)
                try:
                    model.obj.models['main'] = model.obj.models['main'].fit()
                except (RuntimeError, TypeError, NameError):
                    if return_full:
                        return None, None, None
                    else:
                        return None

            case 'LME' | 'LMEA':

                try:
                    model.obj.models['main'] = smf.mixedlm(frm, ds_train, groups=ds_train[model.conf.fixed_effect_var])
                    if hasattr(model.conf, 'lme_method') and model.conf.lme_method is not None:
                        model.obj.models['main'] = model.obj.models['main'].fit(method=model.conf.lme_method)
                    else:
                        model.obj.models['main'] = model.obj.models['main'].fit(method=["powell", "lbfgs"])

                except (RuntimeError, TypeError, NameError, IndexError, ValueError, UserWarning,
                        ConvergenceWarning, np.linalg.LinAlgError) as e:
                    if if_verbose:
                        raise
                    else:
                        if return_full:
                            return None, None, None
                        else:
                            return None

            case 'FE':

                fermatrica_error('FE model type for panel models is not yet implemented. Use LME instead')

            case _:
                fermatrica_error(model.conf.model_type + ' is not known model type. Please use OLS, LME or LMEA')

    # main model : predict

    match model.conf.model_type:

        case 'OLS':
            pred_raw = model.obj.models['main'].predict(ds)

        case 'LME':
            pred_raw = predict_lme(ds, model)

        case 'LMEA':
            pred_raw = predict_lme(ds, model)

        case 'FE':
            fermatrica_error('FE model type for panel models is not yet implemented. Use LME instead')

        case _:
            fermatrica_error(model.conf.model_type + ' is not known model type. Please use OLS, LME or LMEA')

    # main model : inverse LHS transformation

    if transform_lhs_fn is None:
        pred = pred_raw
    else:
        pred = transform_lhs_fn(ds, pred_raw, coef_dict, to_original=True)

    if if_fit:
        pred = pred.loc[ds['listed'] == 2]
        pred_raw = pred_raw.loc[ds['listed'] == 2]

    # reduce data transfer during optimisation

    if return_full:
        return pred, pred_raw, model
    else:
        return pred


@profile
def fit_predict(ds: pd.DataFrame
                , model: "Model"
                , if_fit: bool = True
                , if_verbose: bool = False
                , return_full: bool = False):
    """
    Fit and predict wrapper function

    :param ds: main dataset
    :param model: Model object
    :param if_fit: fit or predict using existing model
    :param if_verbose: print diagnostic / progress info
    :param return_full: set False when optimising, True when predicting
    :return: predicted series or tuple of predicted series, raw (inner) predicted series and Model object
    """

    if model.conf.model_type in ['OLS', 'LME', 'FE'] or (
            model.conf.model_type in ['LMEA'] and if_fit):
        rtrn = _predict_worker(ds, model, if_fit, if_verbose, return_full)

    elif model.conf.model_type == 'LMEA':

        # get unique dates of future periods

        dates = ds.loc[ds['listed'] == 4, 'date'].unique()
        dates = np.sort(dates)

        # iterate over dates

        for cur_date in dates[1:]:

            rtrn = _predict_worker(ds, model, if_fit, if_verbose, return_full)

            if 'y_for_lag___' not in ds.columns.to_list():
                ds['y_for_lag___'] = rtrn
            else:
                ds.loc[:, 'y_for_lag___'] = rtrn

            ds['y_for_lag___'] = groupby_eff(ds, ['bs_key'], ['y_for_lag___']).shift().bfill()
            mask = (ds['date'] == cur_date)
            ds.loc[mask, model.conf.Y_var + '_lag'] = ds.loc[mask, 'y_for_lag___']

            model = transform(ds=ds
                              , model=model
                              , set_start=False
                              , if_by_ref=True)

        del ds['y_for_lag___']

    else:
        fermatrica_error(model.conf.model_type + ' is not known model type. Please use OLS, LME or LMEA')

    return rtrn


def predict_ext(model: "Model"
                , ds: pd.DataFrame | None = None):
    """
    Create datasets of 'extended' prediction: observed, predicted, unit of analysis details etc.

    :param model: Model object
    :param ds: main dataset
    :return: prediction data
    """

    if_conversion_fun = isinstance(model.obj, dict) and hasattr(model.conf, "conversion_fun") and \
                        isinstance(model.conf.conversion_fun, (list, tuple)) and len(model.conf.conversion_fun) > 0

    # assuming transformation is already done

    pred = fit_predict(ds=ds
                       , model=model
                       , if_fit=False
                       , return_full=False)

    if ds is None:
        ds = model.obj['main'].model.data.frame

    cols = ['date', 'bs_key', 'listed', model.conf.Y_var, model.conf.price_var]

    if hasattr(model.conf, "conversion_var") and model.conf.conversion_var is not None:
        cols.append(model.conf.conversion_var)

    if hasattr(model.conf, "bs_key"):
        cols.extend(copy.deepcopy(model.conf.bs_key))

    dt_pred = pd.concat([ds[cols], pred.to_frame('predicted')], axis=1)
    dt_pred.rename({model.conf.Y_var: 'observed'}, axis=1, inplace=True)

    if if_conversion_fun:

        y_preconv = 'predicted'

        for ind, fun_name in enumerate(model.conf.conversion_fun):
            y_postconv = 'predicted_' + str(ind + 1)
            dt_pred[y_postconv] = eval(fun_name)(model, dt_pred[y_preconv], ds, to_original=False)

            y_preconv = y_postconv

    return dt_pred

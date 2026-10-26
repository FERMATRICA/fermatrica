"""
Non-linear (mostly) transformations of X variables. These are part of the model outer layer and are vital
to model architecture.

Because of fundamental issues with fitting non-linear models, MM model is split into two layers:
1. Non-linear parametric transformations of the independent variables
2. Linear model based of the transformed in (1) variables

FERMATRICA greatly expands (1) layer from just transformations of the independent variables to more complex
outer layer. However, "classic" transformations are still of great importance. These transformations
(with some extras concerning group-by) are defined here.

This module contains high- and medium-level wrappers for the whole transformation pipeline and single
transformation functions as well. Low-level function working with vectors directly are defined in
`fermatrica.model.transform_fun`.

IMPORTANT. Transformation functions are not designed to be used directly. Call them via `model.conf.params`
data frame / `params` sheet in `model_conf.xlsx` file.
"""


import ast
import copy
import inspect
import numpy as np
import pandas as pd
import re
import warnings

from line_profiler import profile

from fermatrica_utils import groupby_eff, select_eff, weighted_mean_group, mar_eff, exec_execute, import_module_from_string

from fermatrica.basics.basics import fermatrica_error
import fermatrica.basics.basics
from fermatrica.model.model_obj import ModelObj
from fermatrica.model.model import Model
import fermatrica.model.transform_fun as ftr

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


"""
Higher-level functions & wrappers
"""


@profile
def transform(ds: pd.DataFrame
              , model: "Model"
              , set_start: bool = False
              , if_by_ref: bool = True):
    """
    Higher-level function to run all transformation defined in `model.conf.params` with `ds` dataset

    :param ds: dataset
    :param model: Model object
    :param set_start: calculate starting values according to 'formula' in params table
    :param if_by_ref: run transformations by reference or not (concerns ds only)
    :return:
    """

    params_run = model.conf.params.copy()
    model_obj = copy.deepcopy(model.obj)

    # make a copy of the dataset or not

    if not if_by_ref:
        ds = ds.copy()

    # prepare params

    params_run = params_run[(params_run['variable'] != '') & (params_run['variable'].notna()) &
                            ((params_run['if_active'] == 1) | ((params_run['fun'] == 'follow_me'))) ].copy()

    # run transformations

    if set_start:
        model_obj, params_run = _transform_all(ds, params_run, model_obj, set_start)
        model.conf.params = pd.concat([params_run
                                               , model.conf.params[(model.conf.params['variable'] == '') | (model.conf.params['variable'].isna())]
                                               , model.conf.params[
                                           (model.conf.params['variable'] != '') & (model.conf.params['variable'].notna()) &
                                           (model.conf.params['if_active'] != 1) & (model.conf.params['fun'] != 'follow_me')]]
                                      , ignore_index=True)
    else:
        model_obj = _transform_all(ds, params_run, model_obj, set_start)

    # assign (may be excessive)

    model.obj = model_obj

    # to enforce numpy / pandas data consolidation
    ds.iloc[0, 0] = ds.iloc[0, 0]

    # return

    if if_by_ref:
        out = model
    else:
        out = model, ds

    return out


@profile
def _params_set_start(ds: pd.DataFrame
                      , params: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate starting values according to 'formula' in params table

    :param ds: dataset
    :param params: transformation params table from Model.ModelConf
    :return:
    """

    params.loc[params['formula'].notna() & params['set_start'] == 1, 'value'] \
        = params[params['formula'].notna() & params['set_start'] == 1]['formula'].apply(
        lambda x: exec_execute(ds, x)).values

    return params


@profile
def _var_aggregate(ds: pd.DataFrame
                   , var: str
                   , params_subset: pd.DataFrame
                   , index_vars: str):
    """
    If `var` column to be treated as a number of time series rather than single series / vector, group it by
    `index_vars` + date

    :param ds: dataset
    :param var: variable name
    :param params_subset: subset of params related to `var`
    :param index_vars: split column into smaller series by this grouping vars
    :return:
    """

    index_vars_list = ast.literal_eval('[' + index_vars + ',"date"' + ']')
    index_vars_list = list(set(index_vars_list))

    index_aggr = params_subset[params_subset['index_aggr'].notna()]['index_aggr'].iloc[0]

    if pd.isna(index_aggr):
        tmp = 0
        fermatrica_error('Aggregation function should be set in "index_aggr" column if index_vars is not "bs_key"')

    elif index_aggr == 'sum_kpi_coef':

        cols = copy.deepcopy(index_vars_list)
        cols.append(var)
        cols.append('kpi_coef')

        tmp = select_eff(ds, cols).copy(deep=False)
        tmp['tmp_col'] = tmp[var] * tmp['kpi_coef']
        tmp = tmp.groupby(index_vars_list, as_index=False).tmp_col.sum()

        tmp.rename(columns={'tmp_col': var}, inplace=True)

    elif index_aggr == 'sum_kpi_coef_master':

        cols = copy.deepcopy(index_vars_list)
        cols.append(var)
        cols.append('kpi_coef_master')

        tmp = select_eff(ds, cols).copy(deep=False)
        tmp['tmp_col'] = tmp[var] * tmp['kpi_coef_master']
        tmp = tmp.groupby(index_vars_list, as_index=False).tmp_col.sum()

        tmp.rename(columns={'tmp_col': var}, inplace=True)

    elif index_aggr == 'max':

        cols = copy.deepcopy(index_vars_list)
        cols.append(var)

        tmp = select_eff(ds, cols).copy(deep=False)
        tmp = tmp.groupby(index_vars_list)[var].max()
        tmp = tmp.reset_index(name=var)
        tmp.fillna(0, inplace=True)

    elif index_aggr == 'mean':

        cols = copy.deepcopy(index_vars_list)
        cols.append(var)

        tmp = select_eff(ds, cols).copy(deep=False)
        tmp = tmp.groupby(index_vars_list)[var].mean()
        tmp = tmp.reset_index(name=var)
        tmp.fillna(0, inplace=True)

    elif index_aggr == 'min':

        cols = copy.deepcopy(index_vars_list)
        cols.append(var)

        tmp = select_eff(ds, cols).copy(deep=False)
        tmp = tmp.groupby(index_vars_list)[var].min()
        tmp = tmp.reset_index(name=var)
        tmp.fillna(0, inplace=True)

    elif index_aggr == 'mfreq':

        cols = copy.deepcopy(index_vars_list)
        cols.append(var)

        tmp = select_eff(ds, cols).copy(deep=False)
        tmp = tmp.groupby(index_vars_list)[var].agg(lambda x: pd.Series.mode(x).iat[0])
        tmp = tmp.reset_index(name=var)
        tmp.fillna(0, inplace=True)

    else:
        tmp = ds.groupby(index_vars_list)[var].sum()
        tmp = tmp.reset_index(name=var)

    if isinstance(tmp, pd.DataFrame):
        tmp.iloc[0, 0] = tmp.iloc[0, 0]
    else:
        tmp.iloc[0] = tmp.iloc[0]
        tmp = tmp.copy(deep=True)

    return tmp


@profile
def _check_type(ds: pd.DataFrame
                , var: str | int):
    """
    Check and cleanse types of `var` variable to name it uniformly numeric and `np.ndarray`.
    Works by reference, no value is returned

    :param ds: dataset
    :param var: variable name
    :return: void
    """

    if var not in ds.columns.tolist():
        fermatrica_error(var + ' should be among data frame columns')
    elif ds[var].dtype.name not in ['float64', 'float']:
        ds[var] = ds[var].to_numpy().astype(float)


@profile
def _transform_all(ds: pd.DataFrame
                   , params: pd.DataFrame
                   , model_obj: "ModelObj"
                   , set_start: bool = False):
    """
    Middle-level function to run all transformation defined in `params` with `ds` dataset

    :param ds: dataset
    :param params: params table from Model.conf.params (and maybe after some changes)
    :param model_obj: ModelObj from model (model.obj)
    :param set_start: calculate starting values according to 'formula' in params table
    :return:
    """

    var_fun_uniques = params[['variable', 'fun']].drop_duplicates()

    if 'wrk_index' not in ds.columns:
        ds['wrk_index'] = ds.index

    if model_obj is not None and model_obj.adhoc_code_src is not None:
        for k, v in model_obj.adhoc_code_src.items():
            fr_cur_name = inspect.currentframe().f_globals['__name__']
            import_module_from_string(k, v, fr_cur_name)

    # ds.iloc[0, 0] = ds.iloc[0, 0]  # consolidation

    for ind, var_fun in var_fun_uniques.iterrows():

        params_subset = params[(params['variable'] == var_fun['variable']) & (params['fun'] == var_fun['fun'])]
        if set_start:
            params_subset = _params_set_start(ds, params_subset)

        var = var_fun['variable']
        fun_name = var_fun['fun']  # params_subset['fun'].iloc[0]

        if fun_name == 'follow_me':
            if set_start:
                params = params[~((params['variable'] == var_fun['variable']) & (params['fun'] == var_fun['fun']))]
                params = pd.concat([params, params_subset], ignore_index=True)
        
            if params_subset['if_active'].unique()[0] != 1: 
                continue

            pars_to_follow = [x.strip() for x in params_subset[params_subset.arg == 'which']['value'].values[0].split(',')]
            who_following = [x.strip() for x in params_subset[params_subset.arg == 'who']['value'].values[0].split(',')]
            what_following = params_subset[params_subset.arg == 'what']['value'].values[0].strip()

            followed = params[(params.variable == params_subset.variable.unique()[0]) & (params.fun == what_following) & (params.arg.isin(pars_to_follow))]

            for par in pars_to_follow:
                # mask to select rows of following features with parameter ==  par
                follow_mask = ((params.fun == what_following) & (params.arg == par) & (params.variable.isin(who_following)))

                params.loc[follow_mask,'value'] = followed.loc[(followed.arg == par), 'value'].values[0]  
                params.loc[(follow_mask) & (params.value < params.lower),'value'] = params.loc[(follow_mask) & (params.value < params.lower),'lower']
                params.loc[(follow_mask) & (params.value > params.upper),'value'] = params.loc[(follow_mask) & (params.value > params.upper),'upper']
            
            continue
        
        var_new = var + '_' + fun_name

        if re.search(r'\badhoc_[0-9A-z_]+$', fun_name):

            # arbitrary transformations and instructions

            eval(fun_name)(ds, params_subset, index_vars=params_subset['index_vars'].iloc[0])

        elif re.search(r'\bprice$', fun_name):

            eval(fun_name)(ds, var_raw=var, params_subset=params_subset)

        elif re.search(r'\bfe_model_', fun_name):

            # feature engineering (e.g. cleanse) models

            model_obj.models = eval(fun_name)(ds
                                              , model_obj
                                              , params_subset
                                              , index_vars=params_subset['index_vars'].iloc[0])

        elif params_subset[(params_subset['index_free_var'].isna()) | (params_subset['index_free_var'] == '')][
            'index_free_var'].shape[0] > 0:
            
            # if variable to be tuned uniformly with the dataset
            if var_new in ds.columns.tolist():
                ds.loc[:, var_new] = eval(fun_name)(ds
                                                    , var
                                                    , params_subset
                                                    , index_vars=params_subset['index_vars'].iloc[0])
            else:
                ds[var_new] = eval(fun_name)(ds, var, params_subset, index_vars=params_subset['index_vars'].iloc[0])

        else:

            # if variable params to be tuned separately by index_free_var

            params_subset = params_subset.copy()
            index_free_var = \
                params_subset[(params_subset['index_free_var'].notna()) & (params_subset['index_free_var'] != '')][
                    'index_free_var'].iloc[0]

            index_free_var = re.sub(r"___.+$", '', index_free_var)

            params_subset["index_free_var_work"] = params_subset["index_free_var"].apply(
                lambda x: re.sub(r"^.+___", '', x))

            # add another columns if used in standard transformations
            cols = [var, 'date', 'listed', 'media', 'kpi_coef', 'kpi_coef_master', 'wrk_index']
            cols_total = ds.columns.tolist()
            cols = [x for x in cols if x in cols_total]

            index_vars = params_subset['index_vars'].iloc[0]
            if isinstance(index_vars, str):
                cols = cols + ast.literal_eval('[' + index_vars + ']')

            var_vals = params_subset.loc[(params_subset['type'] == 'str') & (params_subset['value'].notna()), 'value']
            if len(var_vals) > 0:
                for x in var_vals:
                    if isinstance(x, str) and x in ds.columns.tolist():
                        cols = cols + [x]

            cols = list(set(cols))

            cols = [x for x in cols if x != index_free_var]

            # pandas >= 3 excludes grouping columns from frames passed to apply unless
            # they are explicitly selected; index_free_var is read inside the lambda
            # to pick the per-key parameter rows below
            grp = ds.groupby([index_free_var], sort=True, as_index=True)[[index_free_var] + cols]

            rtrn = grp.apply(lambda x: pd.Series(eval(fun_name)(x, var, params_subset[
                params_subset['index_free_var_work'] == x[index_free_var].iloc[0]
                ], index_vars=params_subset['index_vars'].iloc[0])))

            if len(rtrn) < ds.shape[0]:
                # new stack implementation (future_stack=True, default in pandas >= 3)
                # keeps NaN cells in place; dropna=False is rejected under pandas >= 3
                rtrn = rtrn.stack(future_stack=True)

            # main now

            rtrn_ind = grp.indices
            rtrn_ind = np.concatenate([rtrn_ind[x] for x in sorted(rtrn_ind.keys())])  # less reliable, but much faster

            rtrn = rtrn.set_axis(rtrn_ind).astype(float)
            ds[var_new] = rtrn

        ds.iloc[0, 0] = ds.iloc[0, 0]  # consolidation
        
        if var_new in ds.columns.tolist():
            ds[var_new] = ds[var_new].values.copy()

        if set_start:
            params = params[~((params['variable'] == var_fun['variable']) & (params['fun'] == var_fun['fun']))]
            params = pd.concat([params, params_subset], ignore_index=True)

    if set_start:
        return model_obj, params
    else:
        return model_obj


"""
Time series transformations
"""


@profile
def lag(ds: pd.DataFrame
        , var: str
        , params_subset: pd.DataFrame
        , index_vars: str | float = np.nan):
    """
    Transformation: lag

    Interface is standard among all transformation functions

    :param ds: dataset
    :param var: raw variable string name
    :param params_subset: subset of `params` frame containing params specific for `var` variable and current
        transformation
    :param index_vars: string containing variable name or names to apply transformation separately, e.g.
        `"bs_key"` or `"superbrand", "master"`
    :return: transformed time series
    """

    params_dict = fermatrica.basics.basics.params_to_dict(params_subset)
    n = round(params_dict['n'])

    if n == 0:
        return ds[var]
    elif n < 0:
        fermatrica_error('Number of lags should be non-negative')

    if 'cval' not in params_dict:
        params_dict['cval'] = 0

    cval = float(params_dict['cval'])

    cols = [var, 'date']

    if isinstance(index_vars, str):
        index_vars_list = ast.literal_eval('[' + index_vars + ']')
        cols = index_vars_list + cols

    cols = list(set(cols))

    var_series = ds[var]
    var_arr = var_series.to_numpy(dtype=np.float64, copy=False)

    if pd.isna(index_vars):
        rtrn = ftr.lag(var_arr, None, n, cval)
        rtrn = pd.Series(rtrn, index=ds.index, name=var)
        src_dtype = var_series.dtype

    elif index_vars == '"bs_key"':
        group_codes = ds.groupby(index_vars_list, sort=False).ngroup().to_numpy()
        rtrn = ftr.lag(var_arr, group_codes, n, cval)
        rtrn = pd.Series(rtrn, index=ds.index, name=var)
        src_dtype = var_series.dtype

    elif index_vars == '"date"':
        tmp = _var_aggregate(ds, var, params_subset, index_vars)
        src_dtype = tmp[var].dtype
        tmp_vals = tmp[var].to_numpy(dtype=np.float64, copy=False)
        tmp_vals = ftr.lag(tmp_vals, None, n, cval)
        tmp[var] = tmp_vals

        rtrn = select_eff(ds, index_vars_list + ['wrk_index'])
        rtrn = \
            pd.merge(rtrn, tmp, how='left', on=index_vars_list, sort=False).set_index('wrk_index')[var]

    else:
        tmp = _var_aggregate(ds, var, params_subset, index_vars)
        src_dtype = tmp[var].dtype
        tmp_codes = tmp.groupby(index_vars_list, sort=False).ngroup().to_numpy()
        tmp_vals = tmp[var].to_numpy(dtype=np.float64, copy=False)
        tmp[var] = ftr.lag(tmp_vals, tmp_codes, n, cval)

        index_vars_list = ast.literal_eval('[' + index_vars + ',"date"' + ']')

        rtrn = select_eff(ds, index_vars_list + ['wrk_index'])
        rtrn = \
            pd.merge(rtrn, tmp, how='left', on=index_vars_list, sort=False).set_index('wrk_index')[var]

    # preserve source dtype when it is integer and result is representable as integer
    if pd.api.types.is_integer_dtype(src_dtype):
        rtrn_arr = rtrn.to_numpy()
        if np.all(np.isfinite(rtrn_arr)) and np.all(rtrn_arr == rtrn_arr.astype(np.int64)):
            rtrn = rtrn.astype(src_dtype)

    return rtrn


@profile
def adstock(ds: pd.DataFrame
            , var: str
            , params_subset: pd.DataFrame
            , index_vars: str | float = np.nan):
    """
    Transformation: geometric adstock, weighted by (1 - a). Could be useful to preserve scale (more or less)
    Interface is standard among all transformation functions

    :param ds: dataset
    :param var: raw variable string name
    :param params_subset: subset of `params` frame containing params specific for `var` variable and current
        transformation
    :param index_vars: string containing variable name or names to apply transformation separately, e.g.
        `"bs_key"` or `"superbrand", "master"`
    :return: transformed time series
    """

    # check data type etc.

    _check_type(ds, var)

    if isinstance(index_vars, str):
        index_vars_list = ast.literal_eval('[' + index_vars + ']')

    # run

    params_dict = fermatrica.basics.basics.params_to_dict(params_subset)
    a = float(params_dict["a"])

    if pd.isna(index_vars):
        rtrn = ftr.adstock(ds[var].to_numpy(dtype=np.float64), a, weighted=True)

    elif index_vars == '"bs_key"':
        group_codes = ds.groupby(index_vars_list, sort=False).ngroup().to_numpy()
        rtrn = ftr._apply_grouped(
            ds[var].to_numpy(dtype=np.float64), group_codes,
            lambda x: ftr.adstock(x, a, weighted=True))
        rtrn = pd.Series(rtrn, index=ds.index, name=var)

    elif index_vars == '"date"':
        tmp = _var_aggregate(ds, var, params_subset, index_vars)

        tmp[var] = ftr.adstock(tmp[var].to_numpy(dtype=np.float64), a, weighted=True)

        rtrn = select_eff(ds, index_vars_list + ['wrk_index'])
        rtrn = \
            pd.merge(rtrn, tmp, how='left', on=index_vars_list, sort=False).set_index('wrk_index')[var]

    else:
        tmp = _var_aggregate(ds, var, params_subset, index_vars)

        index_vars_list = ast.literal_eval('[' + index_vars + ']')
        tmp_codes = tmp.groupby(index_vars_list, sort=False).ngroup().to_numpy()
        tmp[var] = ftr._apply_grouped(
            tmp[var].to_numpy(dtype=np.float64), tmp_codes,
            lambda x: ftr.adstock(x, a, weighted=True))

        index_vars_list = ast.literal_eval('[' + index_vars + ',"date"' + ']')

        rtrn = select_eff(ds, index_vars_list + ['wrk_index'])
        rtrn = \
            pd.merge(rtrn, tmp, how='left', on=index_vars_list, sort=False).set_index(
                'wrk_index')[var]

    return rtrn


@profile
def adstockp(ds: pd.DataFrame
             , var: str
             , params_subset: pd.DataFrame
             , index_vars: str | float = np.nan):
    """
    Transformation: pure geometric adstock.
    Interface is standard among all transformation functions

    :param ds: dataset
    :param var: raw variable string name
    :param params_subset: subset of `params` frame containing params specific for `var` variable and current
        transformation
    :param index_vars: string containing variable name or names to apply transformation separately, e.g.
        `"bs_key"` or `"superbrand", "master"`
    :return: transformed time series
    """

    # check data type

    _check_type(ds, var)

    if isinstance(index_vars, str):
        index_vars_list = ast.literal_eval('[' + index_vars + ']')

    # run

    params_dict = fermatrica.basics.basics.params_to_dict(params_subset)
    a = float(params_dict["a"])

    if pd.isna(index_vars):
        rtrn = ftr.adstock(ds[var].to_numpy(dtype=np.float64), a, weighted=False)

    elif index_vars == '"bs_key"':
        group_codes = ds.groupby(index_vars_list, sort=False).ngroup().to_numpy()
        rtrn = ftr._apply_grouped(
            ds[var].to_numpy(dtype=np.float64), group_codes,
            lambda x: ftr.adstock(x, a, weighted=False))
        rtrn = pd.Series(rtrn, index=ds.index, name=var)

    elif index_vars == '"date"':
        tmp = _var_aggregate(ds, var, params_subset, index_vars)

        tmp[var] = ftr.adstock(tmp[var].to_numpy(dtype=np.float64), a, weighted=False)

        rtrn = select_eff(ds, index_vars_list + ['wrk_index'])
        rtrn = \
            pd.merge(rtrn, tmp, how='left', on=index_vars_list, sort=False).set_index('wrk_index')[var]

    else:
        tmp = _var_aggregate(ds, var, params_subset, index_vars)

        index_vars_list = ast.literal_eval('[' + index_vars + ']')
        tmp_codes = tmp.groupby(index_vars_list, sort=False).ngroup().to_numpy()
        tmp[var] = ftr._apply_grouped(
            tmp[var].to_numpy(dtype=np.float64), tmp_codes,
            lambda x: ftr.adstock(x, a, weighted=False))

        index_vars_list = ast.literal_eval('[' + index_vars + ',"date"' + ']')

        rtrn = select_eff(ds, index_vars_list + ['wrk_index'])
        rtrn = \
            pd.merge(rtrn, tmp, how='left', on=index_vars_list, sort=False).set_index(
                'wrk_index')[var]

    return rtrn


@profile
def adstockd(ds: pd.DataFrame
              , var: str
              , params_subset: pd.DataFrame
              , index_vars: str | float = np.nan):
    """
    Transformation: geometric adstock with two weights, weighted by (1 - a1) and (1 - a2) respectively.
    Could be useful to preserve scale (more or less).
    Linear combination of two adstocks, useful to prevent sign switch of one of the components.

    Interface is standard among all transformation functions

    :param ds: dataset
    :param var: raw variable string name
    :param params_subset: subset of `params` frame containing params specific for `var` variable and current
        transformation
    :param index_vars: string containing variable name or names to apply transformation separately, e.g.
        `"bs_key"` or `"superbrand", "master"`
    :return: transformed time series
    """

    # check data type

    _check_type(ds, var)

    if isinstance(index_vars, str):
        index_vars_list = ast.literal_eval('[' + index_vars + ']')

    # run

    params_dict = fermatrica.basics.basics.params_to_dict(params_subset)
    a1, a2, w2 = float(params_dict["a1"]), float(params_dict["a2"]), float(params_dict["w2"])

    if pd.isna(index_vars):
        rtrn = ftr.adstockd(ds[var].to_numpy(dtype=np.float64), a1, a2, w2, weighted=True)

    elif index_vars == '"date"':
        tmp = _var_aggregate(ds, var, params_subset, index_vars)

        tmp[var] = ftr.adstockd(tmp[var].to_numpy(dtype=np.float64), a1, a2, w2, weighted=True)

        rtrn = select_eff(ds, index_vars_list + ['wrk_index'])
        rtrn = \
            pd.merge(rtrn, tmp, how='left', on=index_vars_list, sort=False).set_index('wrk_index')[var]

    elif index_vars == '"bs_key"':
        group_codes = ds.groupby(index_vars_list, sort=False).ngroup().to_numpy()
        rtrn = ftr._apply_grouped(
            ds[var].to_numpy(dtype=np.float64), group_codes,
            lambda x: ftr.adstockd(x, a1, a2, w2, weighted=True))
        rtrn = pd.Series(rtrn, index=ds.index, name=var)

    else:
        tmp = _var_aggregate(ds, var, params_subset, index_vars)

        index_vars_list = ast.literal_eval('[' + index_vars + ']')
        tmp_codes = tmp.groupby(index_vars_list, sort=False).ngroup().to_numpy()
        tmp[var] = ftr._apply_grouped(
            tmp[var].to_numpy(dtype=np.float64), tmp_codes,
            lambda x: ftr.adstockd(x, a1, a2, w2, weighted=True))

        index_vars_list = ast.literal_eval('[' + index_vars + ',"date"' + ']')

        rtrn = select_eff(ds, index_vars_list + ['wrk_index'])
        rtrn = \
            pd.merge(rtrn, tmp, how='left', on=index_vars_list, sort=False).set_index(
                'wrk_index')[var]

    return rtrn


@profile
def adstockpd(ds: pd.DataFrame
              , var: str
              , params_subset: pd.DataFrame
              , index_vars: str | float = np.nan):
    """
    Transformation: pure geometric adstock with two weights. Effectively linear combination of two adstocks,
    useful to prevent sign switch of one of the components.

    Interface is standard among all transformation functions

    :param ds: dataset
    :param var: raw variable string name
    :param params_subset: subset of `params` frame containing params specific for `var` variable and current
        transformation
    :param index_vars: string containing variable name or names to apply transformation separately, e.g.
        `"bs_key"` or `"superbrand", "master"`
    :return: transformed time series
    """

    # check data type

    _check_type(ds, var)

    if isinstance(index_vars, str):
        index_vars_list = ast.literal_eval('[' + index_vars + ']')

    # run

    params_dict = fermatrica.basics.basics.params_to_dict(params_subset)
    a1, a2, w2 = float(params_dict["a1"]), float(params_dict["a2"]), float(params_dict["w2"])

    if pd.isna(index_vars):
        rtrn = ftr.adstockd(ds[var].to_numpy(dtype=np.float64), a1, a2, w2, weighted=False)

    elif index_vars == '"date"':
        tmp = _var_aggregate(ds, var, params_subset, index_vars)

        tmp[var] = ftr.adstockd(tmp[var].to_numpy(dtype=np.float64), a1, a2, w2, weighted=False)

        rtrn = select_eff(ds, index_vars_list + ['wrk_index'])
        rtrn = \
            pd.merge(rtrn, tmp, how='left', on=index_vars_list, sort=False).set_index('wrk_index')[var]

    elif index_vars == '"bs_key"':
        group_codes = ds.groupby(index_vars_list, sort=False).ngroup().to_numpy()
        rtrn = ftr._apply_grouped(
            ds[var].to_numpy(dtype=np.float64), group_codes,
            lambda x: ftr.adstockd(x, a1, a2, w2, weighted=False))
        rtrn = pd.Series(rtrn, index=ds.index, name=var)

    else:
        tmp = _var_aggregate(ds, var, params_subset, index_vars)

        index_vars_list = ast.literal_eval('[' + index_vars + ']')
        tmp_codes = tmp.groupby(index_vars_list, sort=False).ngroup().to_numpy()
        tmp[var] = ftr._apply_grouped(
            tmp[var].to_numpy(dtype=np.float64), tmp_codes,
            lambda x: ftr.adstockd(x, a1, a2, w2, weighted=False))

        index_vars_list = ast.literal_eval('[' + index_vars + ',"date"' + ']')

        rtrn = select_eff(ds, index_vars_list + ['wrk_index'])
        rtrn = \
            pd.merge(rtrn, tmp, how='left', on=index_vars_list, sort=False).set_index(
                'wrk_index')[var]

    return rtrn


@profile
def dwbl(ds: pd.DataFrame
         , var: str
         , params_subset: pd.DataFrame
         , index_vars: str | float = np.nan):
    """
    Transformation: Weibull-based time series transformation. More sophisticated substitute to
    geometric adstock, but not especially human-friendly.

    IMPORTANT. This implementation could be sensitive to new periods added to series.

    Interface is standard among all transformation functions

    :param ds: dataset
    :param var: raw variable string name
    :param params_subset: subset of `params` frame containing params specific for `var` variable and current
        transformation
    :param index_vars: string containing variable name or names to apply transformation separately, e.g.
        `"bs_key"` or `"superbrand", "master"`
    :return: transformed time series
    """

    # check data type

    _check_type(ds, var)

    if isinstance(index_vars, str):
        index_vars_list = ast.literal_eval('[' + index_vars + ']')

    # run

    params_dict = fermatrica.basics.basics.params_to_dict(params_subset)
    kernel = ftr._compute_weibull_kernel(params_dict)

    if pd.isna(index_vars):
        rtrn = ftr._weibull_convolve(ds[var].to_numpy(dtype=np.float64), kernel, if_weight=True)

    elif index_vars == '"bs_key"':
        group_codes = ds.groupby(index_vars_list, sort=False).ngroup().to_numpy()
        rtrn = ftr._apply_grouped(
            ds[var].to_numpy(dtype=np.float64), group_codes,
            lambda x: ftr._weibull_convolve(x, kernel, if_weight=True))
        rtrn = pd.Series(rtrn, index=ds.index, name=var)

    elif index_vars == '"date"':
        tmp = _var_aggregate(ds, var, params_subset, index_vars)

        tmp[var] = ftr._weibull_convolve(tmp[var].to_numpy(dtype=np.float64), kernel, if_weight=True)

        rtrn = select_eff(ds, index_vars_list + ['wrk_index'])
        rtrn = \
            pd.merge(rtrn, tmp, how='left', on=index_vars_list, sort=False).set_index('wrk_index')[var]

    else:
        tmp = _var_aggregate(ds, var, params_subset, index_vars)

        index_vars_list = ast.literal_eval('[' + index_vars + ']')
        index_vars_list = list(set(index_vars_list))
        tmp_codes = tmp.groupby(index_vars_list, sort=False).ngroup().to_numpy()
        tmp[var] = ftr._apply_grouped(
            tmp[var].to_numpy(dtype=np.float64), tmp_codes,
            lambda x: ftr._weibull_convolve(x, kernel, if_weight=True))

        index_vars_list = ast.literal_eval('[' + index_vars + ',"date"' + ']')
        index_vars_list = list(set(index_vars_list))

        rtrn = select_eff(ds, index_vars_list + ['wrk_index'])
        rtrn = \
            pd.merge(rtrn, tmp, how='left', on=index_vars_list, sort=False).set_index('wrk_index')[var]


    return rtrn


@profile
def dwblp(ds: pd.DataFrame
          , var: str
          , params_subset: pd.DataFrame
          , index_vars: str | float = np.nan):
    """
    Transformation: Weibull-based time series transformation. More sophisticated substitute to
    geometric adstock, but not especially human-friendly.

    "Pure" flavour, non-sensitive to new periods added to series.

    Interface is standard among all transformation functions.

    :param ds: dataset
    :param var: raw variable string name
    :param params_subset: subset of `params` frame containing params specific for `var` variable and current
        transformation
    :param index_vars: string containing variable name or names to apply transformation separately, e.g.
        `"bs_key"` or `"superbrand", "master"`
    :return: transformed time series
    """

    # check data type

    _check_type(ds, var)

    if isinstance(index_vars, str):
        index_vars_list = ast.literal_eval('[' + index_vars + ']')

    # run

    params_dict = fermatrica.basics.basics.params_to_dict(params_subset)
    kernel = ftr._compute_weibull_kernel(params_dict)

    if pd.isna(index_vars):
        rtrn = ftr._weibull_convolve(ds[var].to_numpy(dtype=np.float64), kernel)

    elif index_vars == '"bs_key"':
        group_codes = ds.groupby(index_vars_list, sort=False).ngroup().to_numpy()
        rtrn = ftr._apply_grouped(
            ds[var].to_numpy(dtype=np.float64), group_codes,
            lambda x: ftr._weibull_convolve(x, kernel))
        rtrn = pd.Series(rtrn, index=ds.index, name=var)

    elif index_vars == '"date"':
        tmp = _var_aggregate(ds, var, params_subset, index_vars)

        tmp[var] = ftr._weibull_convolve(tmp[var].to_numpy(dtype=np.float64), kernel)

        rtrn = select_eff(ds, index_vars_list + ['wrk_index'])
        rtrn = \
            pd.merge(rtrn, tmp, how='left', on=index_vars_list, sort=False).set_index('wrk_index')[var]

    else:
        tmp = _var_aggregate(ds, var, params_subset, index_vars)

        index_vars_list = ast.literal_eval('[' + index_vars + ']')
        index_vars_list = list(set(index_vars_list))
        tmp_codes = tmp.groupby(index_vars_list, sort=False).ngroup().to_numpy()
        tmp[var] = ftr._apply_grouped(
            tmp[var].to_numpy(dtype=np.float64), tmp_codes,
            lambda x: ftr._weibull_convolve(x, kernel))

        index_vars_list = ast.literal_eval('[' + index_vars + ',"date"' + ']')
        index_vars_list = list(set(index_vars_list))

        rtrn = select_eff(ds, index_vars_list + ['wrk_index'])
        rtrn = \
            pd.merge(rtrn, tmp, how='left', on=index_vars_list, sort=False).set_index('wrk_index')[var]

    return rtrn


@profile
def mar(ds: pd.DataFrame
        , var: str
        , params_subset: pd.DataFrame
        , index_vars: str | float = np.nan):
    """
    Transformation: right moving average

    Interface is standard among all transformation functions

    :param ds: dataset
    :param var: raw variable string name
    :param params_subset: subset of `params` frame containing params specific for `var` variable and current
        transformation
    :param index_vars: string containing variable name or names to apply transformation separately, e.g.
        `"bs_key"` or `"superbrand", "master"`
    :return: transformed time series
    """

    params_dict = fermatrica.basics.basics.params_to_dict(params_subset)

    n = params_dict['n']
    if isinstance(n, float):
        n = int(round(n))

    algo = params_dict.get('algo', 'precise')

    cols = [var, 'date']
    if isinstance(index_vars, str):
        index_vars_list = ast.literal_eval('[' + index_vars + ']')
        cols = index_vars_list + cols
    cols = list(set(cols))

    var_series = ds[var]
    var_arr = var_series.to_numpy(dtype=np.float64, copy=False)
    valid_mask = ~np.isnan(var_arr)
    src_dtype = var_series.dtype

    if pd.isna(index_vars):
        rtrn_arr = ftr.mar(var_arr, None, valid_mask, n, algo)
        rtrn = pd.Series(rtrn_arr, index=ds.index, name=var + '_tmp')

    elif index_vars == '"bs_key"':
        index_vars_list = ast.literal_eval('[' + index_vars + ']')
        group_codes = ds.groupby(index_vars_list, sort=False).ngroup().to_numpy()
        rtrn_arr = ftr.mar(var_arr, group_codes, valid_mask, n, algo)
        rtrn = pd.Series(rtrn_arr, index=ds.index, name=var + '_tmp')

    elif index_vars == '"date"':
        ds_tmp = select_eff(ds, cols + ['wrk_index'])
        tmp = _var_aggregate(ds_tmp, var, params_subset, index_vars)

        tmp_vals = tmp[var].to_numpy(dtype=np.float64, copy=False)
        tmp_valid = ~np.isnan(tmp_vals)
        tmp[var] = ftr.mar(tmp_vals, None, tmp_valid, n, algo)

        rtrn = \
            pd.merge(ds_tmp[index_vars_list + ['wrk_index']], tmp, how='left', on=index_vars_list, sort=False).set_index(
                'wrk_index')[var]

    else:
        if 'index_aggr' in params_subset and params_subset['index_aggr'].iloc[0] == 'sum_kpi_coef':
            ds_tmp = select_eff(ds, cols + ['wrk_index', 'kpi_coef'])
        elif 'index_aggr' in params_subset and params_subset['index_aggr'].iloc[0] == 'sum_kpi_coef_master':
            ds_tmp = select_eff(ds, cols + ['wrk_index', 'kpi_coef_master'])
        else:
            ds_tmp = select_eff(ds, cols + ['wrk_index'])
            
        tmp = _var_aggregate(ds_tmp, var, params_subset, index_vars)

        index_vars_list = ast.literal_eval('[' + index_vars + ']')
        tmp_codes = tmp.groupby(index_vars_list, sort=False).ngroup().to_numpy()
        tmp_vals = tmp[var].to_numpy(dtype=np.float64, copy=False)
        tmp_valid = ~np.isnan(tmp_vals)
        tmp[var] = ftr.mar(tmp_vals, tmp_codes, tmp_valid, n, algo)

        index_vars_list = ast.literal_eval('[' + index_vars + ',"date"' + ']')

        rtrn = \
            pd.merge(ds_tmp[index_vars_list + ['wrk_index']], tmp, how='left', on=index_vars_list, sort=False).set_index(
                'wrk_index')[var]

    # preserve source dtype when n == 0 (pass-through) and source is integer
    if n == 0 and pd.api.types.is_integer_dtype(src_dtype):
        rtrn_arr = rtrn.to_numpy()
        if np.all(np.isfinite(rtrn_arr)) and np.all(rtrn_arr == rtrn_arr.astype(np.int64)):
            rtrn = rtrn.astype(src_dtype)

    if params_dict.get('if_scale', False):
        mask = ds['listed'].isin([2, 3]) & (ds['media'] == 1)
        rtrn = rtrn / rtrn.loc[mask].mean()

    return rtrn


@profile
def infl(
        ds: pd.DataFrame
        , var: str
        , params_subset: pd.DataFrame
        , index_vars: str | float = np.nan):
    """
    Transformation: inflation. Interface is standard among all transformation functions

    :param ds: dataset
    :param var: raw variable string name
    :param params_subset: subset of `params` frame containing params specific for `var` variable and current
        transformation
    :param index_vars: string containing variable name or names to apply transformation separately, e.g.
        `"bs_key"` or `"superbrand", "master"`
    :return: transformed time series
    """

    # check data type

    _check_type(ds, var)

    # run

    params_dict = fermatrica.basics.basics.params_to_dict(params_subset)

    infl = params_dict['infl']
    if type(infl) == int:
        infl = float(infl)

    rtrn = ds.loc[:, var] * (infl ** (ds['date'].dt.year - params_dict['year_start']))
    rtrn.loc[(rtrn.isna()) | (np.isinf(rtrn))] = 0

    return rtrn


@profile
def age(ds: pd.DataFrame
        , var: str
        , params_subset: pd.DataFrame
        , index_vars: str | float = np.nan):
    """
    Transformation: age of the entity

    Interface is standard among all transformation functions

    :param ds: dataset
    :param var: raw variable string name
    :param params_subset: subset of `params` frame containing params specific for `var` variable and current
        transformation
    :param index_vars: string containing variable name or names to apply transformation separately, e.g.
        `"bs_key"` or `"superbrand", "master"`
    :return: transformed time series
    """

    cols = [var, 'date', 'listed']
    if isinstance(index_vars, str):
        index_vars_list = ast.literal_eval('[' + index_vars + ']')
        cols = index_vars_list + cols
    cols = list(set(cols))

    ds_tmp = select_eff(ds, cols)

    mask = ~ds_tmp[var].isna()

    if pd.isna(index_vars):
        rtrn = ds_tmp.loc[mask, :].groupby(var).cumcount().rename(var)
        rtrn = ds_tmp.join(rtrn, how='left', rsuffix='_tmp')[var + '_tmp']

    elif index_vars == '"bs_key"':
        rtrn = ds_tmp.loc[mask, :].groupby(index_vars_list + [var]).cumcount().reset_index(name=var)
        rtrn = ds_tmp.join(rtrn, how='left', rsuffix='_tmp')[var + '_tmp']

    else:
        tmp = _var_aggregate(ds_tmp.loc[mask, :], var, params_subset, index_vars)

        tmp.sort_values(by=index_vars_list + ['date'], ignore_index=True, inplace=True)
        tmp[var] = tmp.groupby(index_vars_list + [var]).cumcount()

        index_vars_list = ast.literal_eval('[' + index_vars + ',"date"' + ']')

        rtrn = select_eff(ds, index_vars_list + ['wrk_index'])
        rtrn = \
            pd.merge(rtrn, tmp, how='left', on=index_vars_list, sort=False).set_index(
                'wrk_index')[var]

    rtrn[rtrn.isna()] = 0

    return rtrn


@profile
def scale(ds: pd.DataFrame
          , var: str
          , params_subset: pd.DataFrame
          , index_vars: str | float = np.nan):
    """
    Transformation: classic scaling: (X - median(X)) / STD(X)

    Interface is standard among all transformation functions

    :param ds: dataset
    :param var: raw variable string name
    :param params_subset: subset of `params` frame containing params specific for `var` variable and current
        transformation
    :param index_vars: string containing variable name or names to apply transformation separately, e.g.
        `"bs_key"` or `"superbrand", "master"`
    :return: transformed time series
    """

    params_dict = fermatrica.basics.basics.params_to_dict(params_subset)
    listed = ast.literal_eval('[' + params_dict['listed'] + ']')

    if isinstance(index_vars, str):
        index_vars_list = ast.literal_eval('[' + index_vars + ']')

    if pd.isna(index_vars):
        ds_tmp = select_eff(ds, [var, 'listed'])
        mask = ds_tmp['listed'].isin(listed)

        rtrn = ftr.scale_classic_median(ds_tmp[var], mask)

    else:
        ds_tmp = select_eff(ds, ([var, 'listed'] + index_vars_list))
        ds_tmp['mask'] = ds_tmp['listed'].isin(listed)

        rtrn = groupby_eff(ds_tmp, index_vars_list, [var, 'mask'], None)[[var, 'mask']]
        rtrn = rtrn.apply(lambda x: ftr.scale_classic_median(x[var], x['mask']))

        if isinstance(rtrn, pd.DataFrame) and len(rtrn) == 1:
            rtrn = rtrn.squeeze()
        else:
            rtrn.index = rtrn.index.get_level_values(-1)

        rtrn.sort_index(inplace=True)

    return rtrn


"""
Cross-sectional transformations
"""


@profile
def share(ds: pd.DataFrame
          , var: str
          , params_subset: pd.DataFrame
          , index_vars: str | float = np.nan):
    """
    Transformation: share (share of voice, share os spends, share of sales etc.)

    Interface is standard among all transformation functions

    :param ds: dataset
    :param var: raw variable string name
    :param params_subset: subset of `params` frame containing params specific for `var` variable and current
        transformation
    :param index_vars: string containing variable name or names to apply transformation separately, e.g.
        `"bs_key"` or `"superbrand", "master"`
    :return: transformed time series
    """

    params_dict = fermatrica.basics.basics.params_to_dict(params_subset)
    if pd.notna(index_vars):
        index_vars_list = ast.literal_eval('[' + index_vars + ']')

    rtrn = np.nan

    coef_var = params_dict['coef_var']
    include_coef_var = params_dict['include_coef_var']

    if (pd.isna(coef_var)) and (pd.isna(index_vars)):
        rtrn = ds.loc[:, var] / sum(ds.loc[:, var])

    elif (not pd.isna(coef_var)) and (pd.isna(index_vars)) and (include_coef_var is True):
        rtrn = (ds.loc[:, var] * ds.loc[:, coef_var]) / (sum(ds.loc[:, var] * ds.loc[:, coef_var]))

    elif (not pd.isna(coef_var)) and (pd.isna(index_vars)) and (include_coef_var is not True):
        rtrn = (ds.loc[:, var]) / (sum(ds.loc[:, var] * ds.loc[:, coef_var]))

    elif (pd.isna(coef_var)) and (not pd.isna(index_vars)):

        cols = copy.deepcopy(index_vars_list)
        cols.append(var)
        tmp = select_eff(ds, cols)
        tmp = tmp.groupby(index_vars_list, as_index=False)[var].sum()

        ds['tmp'] = np.nan
        ds_tmp = select_eff(ds, index_vars_list + ['wrk_index']).\
            merge(tmp, how='left', on=index_vars_list, sort=False).set_index('wrk_index')
        ds.loc[:, 'tmp'] = ds_tmp.loc[:, var]

        rtrn = ds.loc[:, var] / ds.loc[:, 'tmp']
        del ds['tmp']

    elif (not pd.isna(coef_var)) and (not pd.isna(index_vars)) and include_coef_var:

        cols = copy.deepcopy(index_vars_list)
        cols.extend([var, coef_var])
        tmp = select_eff(ds, cols)
        tmp['tmp_mul'] = tmp.loc[:, var] * tmp.loc[:, coef_var]
        tmp = tmp.groupby(index_vars_list, as_index=False).tmp_mul.sum()

        ds['tmp'] = np.nan
        ds_tmp = select_eff(ds, index_vars_list + ['wrk_index']).\
            merge(tmp, how='left', on=index_vars_list, sort=False).set_index('wrk_index')
        ds.loc[:, 'tmp'] = ds_tmp.loc[:, 'tmp_mul']

        rtrn = ds.loc[:, var] * ds.loc[:, coef_var] / ds.loc[:, 'tmp']
        del ds['tmp']

    elif (not pd.isna(coef_var)) and (not pd.isna(index_vars)) and (include_coef_var is not True):

        cols = copy.deepcopy(index_vars_list)
        cols.extend([var, coef_var])
        tmp = select_eff(ds, cols)
        tmp['tmp_mul'] = tmp.loc[:, var] * tmp.loc[:, coef_var]
        tmp = tmp.groupby(index_vars_list, as_index=False).tmp_mul.sum()

        ds['tmp'] = np.nan
        ds_tmp = select_eff(ds, index_vars_list + ['wrk_index']).\
            merge(tmp, how='left', on=index_vars_list, sort=False).set_index('wrk_index')
        ds.loc[:, 'tmp'] = ds_tmp.loc[:, 'tmp_mul']

        rtrn = ds.loc[:, var] / ds.loc[:, 'tmp']
        del ds['tmp']

    rtrn[rtrn.isna()] = 0

    return rtrn


"""
Saturation transformations
"""


@profile
def expm1(ds: pd.DataFrame
          , var: str
          , params_subset: pd.DataFrame
          , index_vars: str | float = np.nan):
    """
    Transformation: exponential transformation minus 1 (to restore after logarithm(x + 1))

    Interface is standard among all transformation functions

    :param ds: dataset
    :param var: raw variable string name
    :param params_subset: subset of `params` frame containing params specific for `var` variable and current
        transformation
    :param index_vars: string containing variable name or names to apply transformation separately, e.g.
        `"bs_key"` or `"superbrand", "master"`
    :return: transformed time series
    """

    params_dict = fermatrica.basics.basics.params_to_dict(params_subset)

    rtrn = np.exp(ds[var] * params_dict['product']) - 1

    return rtrn


@profile
def expm1scaled(ds: pd.DataFrame
                , var: str
                , params_subset: pd.DataFrame
                , index_vars: str | float = np.nan):
    """
    Transformation: exponential transformation minus 1 (to restore after logarithm(x + 1)) with scaling.
    Scaling is essential to keep coefficients in outer layer of the model more or less valid

    Interface is standard among all transformation functions

    :param ds: dataset
    :param var: raw variable string name
    :param params_subset: subset of `params` frame containing params specific for `var` variable and current
        transformation
    :param index_vars: string containing variable name or names to apply transformation separately, e.g.
        `"bs_key"` or `"superbrand", "master"`
    :return: transformed time series
    """

    params_dict = fermatrica.basics.basics.params_to_dict(params_subset)
    listed = ast.literal_eval('[' + str(params_dict['listed']) + ']')
    product = params_dict['product']

    if isinstance(index_vars, str):
        index_vars_list = ast.literal_eval('[' + index_vars + ']')

    if pd.isna(index_vars):
        ds_tmp = select_eff(ds, [var, 'listed'])
        mask = ds_tmp['listed'].isin(listed)

        ds_tmp.loc[:, 'tmp'] = np.exp(ds_tmp[var] * product) - 1

        rtrn = ftr.scale_classic_median(ds_tmp['tmp'], mask)

    else:
        ds_tmp = select_eff(ds, ([var, 'listed'] + index_vars_list))
        ds_tmp['mask'] = ds_tmp['listed'].isin(listed)

        ds_tmp.loc[:, 'tmp'] = np.exp(ds_tmp[var] * product) - 1

        rtrn = groupby_eff(ds_tmp, index_vars_list, ['tmp', 'mask'], None)[['tmp', 'mask']]
        rtrn = rtrn.apply(lambda x: ftr.scale_classic_median(x['tmp'], x['mask']))

        if isinstance(rtrn, pd.DataFrame) and len(rtrn) == 1:
            rtrn = rtrn.squeeze()
        else:
            rtrn.index = rtrn.index.get_level_values(-1)

        rtrn.sort_index(inplace=True)

    return rtrn


@profile
def softmaxfull(ds: pd.DataFrame
                , var: str
                , params_subset: pd.DataFrame
                , index_vars: str | float = np.nan):
    """
    Transformation: flavour of logistic saturation curve with unit-specific `avg` and "abstract" `lambda`.
    The more is lambda, less steep is the curve.

    If `ec50` is included in params, `avg` should be treated as fixed variable similar to `std`

    Interface is standard among all transformation functions

    :param ds: dataset
    :param var: raw variable string name
    :param params_subset: subset of `params` frame containing params specific for `var` variable and current
        transformation
    :param index_vars: string containing variable name or names to apply transformation separately, e.g.
        `"bs_key"` or `"superbrand", "master"`
    :return: transformed time series
    """

    # check data type

    _check_type(ds, var)

    # run

    params_dict = fermatrica.basics.basics.params_to_dict(params_subset)

    if params_dict['std'] == 0:
        fermatrica_error('Error in: softmax function, variable: ' + var + ', `std` = 0')

    params_dict['avg_level'] = copy.deepcopy(params_dict['avg'])
    if 'ec50' in params_dict:
        params_dict['avg'] = params_dict['avg'] * params_dict['ec50']

    x_input = ds.loc[:, var].values.copy()
    x_input = np.asarray(x_input, dtype=np.float64)
    
    rtrn = ftr.softmax(x_input, params_dict)
    
    rtrn = np.asarray(rtrn, dtype=np.float64).copy()

    # weights (i.e. vertical scale) could be important for linear combinations and LHS variables

    if 'scale' in params_dict:

        mask = ds['listed'] == 2
        if not isinstance(rtrn, pd.Series):
            rtrn = pd.Series(rtrn, index=ds.index, copy=True)
        rtrn = ftr.scale_level(rtrn, mask, params_dict)

        if isinstance(rtrn, pd.Series):
            rtrn = rtrn.values.copy()
        else:
            rtrn = np.asarray(rtrn, dtype=np.float64).copy()

    return rtrn


@profile
def softmax(ds: pd.DataFrame
            , var: str
            , params_subset: pd.DataFrame
            , index_vars: str | float = np.nan):

    _check_type(ds, var)

    params_dict = fermatrica.basics.basics.params_to_dict(params_subset)

    if params_dict['std'] == 0:
        fermatrica_error('Error in: softmax function, variable: ' + var + ', `std` = 0')

    params_dict['avg_level'] = copy.deepcopy(params_dict['avg'])
    if 'ec50' in params_dict:
        params_dict['avg'] = params_dict['avg'] * params_dict['ec50']

    x_input = ds.loc[:, var].values.copy()
    x_input = np.asarray(x_input, dtype=np.float64)
    
    rtrn = ftr.softmax(x_input, params_dict) - ftr.softmax(np.array([0], dtype=np.float64), params_dict)[0]
    
    rtrn = np.asarray(rtrn, dtype=np.float64).copy()

    if 'scale' in params_dict:
        mask = ds['listed'] == 2
        if not isinstance(rtrn, pd.Series):
            rtrn = pd.Series(rtrn, index=ds.index, copy=True)
        rtrn = ftr.scale_level(rtrn, mask, params_dict)
        
        if isinstance(rtrn, pd.Series):
            rtrn = rtrn.values.copy()
        else:
            rtrn = np.asarray(rtrn, dtype=np.float64).copy()

    return rtrn


@profile
def logistic(ds: pd.DataFrame
             , var: str
             , params_subset: pd.DataFrame
             , index_vars: str | float = np.nan):
    """
    Transformation: logistic saturation curve.
    Interface is standard among all transformation functions

    :param ds: dataset
    :param var: raw variable string name
    :param params_subset: subset of `params` frame containing params specific for `var` variable and current
        transformation
    :param index_vars: string containing variable name or names to apply transformation separately, e.g.
        `"bs_key"` or `"superbrand", "master"`
    :return: transformed time series
    """

    # check data type

    _check_type(ds, var)

    # run

    params_dict = fermatrica.basics.basics.params_to_dict(params_subset)
    rtrn = ftr.logistic(ds[var], params_dict)

    return rtrn


@profile
def gompertz(ds: pd.DataFrame
             , var: str
             , params_subset: pd.DataFrame
             , index_vars: str | float = np.nan):
    """
    Transformation: Gompertz asymmetric sigmoid

    Interface is standard among all transformation functions

    :param ds: dataset
    :param var: raw variable string name
    :param params_subset: subset of `params` frame containing params specific for `var` variable and current
        transformation
    :param index_vars: string containing variable name or names to apply transformation separately, e.g.
        `"bs_key"` or `"superbrand", "master"`
    :return: transformed time series
    """

    params_dict = fermatrica.basics.basics.params_to_dict(params_subset)

    rtrn = ftr.gompertz(ds[var], params_dict)

    return rtrn


@profile
def adbudg(ds: pd.DataFrame
           , var: str
           , params_subset: pd.DataFrame
           , index_vars: str | float = np.nan):
    """
    Transformation: saturation curve with both power / logarithm and S-curve versions
    depending on param values.

    Interface is standard among all transformation functions

    :param ds: dataset
    :param var: raw variable string name
    :param params_subset: subset of `params` frame containing params specific for `var` variable and current
        transformation
    :param index_vars: string containing variable name or names to apply transformation separately, e.g.
        `"bs_key"` or `"superbrand", "master"`
    :return: transformed time series
    """

    # check data type

    _check_type(ds, var)

    # run

    params_dict = fermatrica.basics.basics.params_to_dict(params_subset)
    rtrn = ftr.adbudg(ds[var], params_dict)

    return rtrn


@profile
def power(ds: pd.DataFrame
          , var: str
          , params_subset: pd.DataFrame
          , index_vars: str | float = np.nan):
    """
    Transformation: power

    Interface is standard among all transformation functions

    :param ds: dataset
    :param var: raw variable string name
    :param params_subset: subset of `params` frame containing params specific for `var` variable and current
        transformation
    :param index_vars: string containing variable name or names to apply transformation separately, e.g.
        `"bs_key"` or `"superbrand", "master"`
    :return: transformed time series
    """

    # check data type

    _check_type(ds, var)

    # run

    params_dict = fermatrica.basics.basics.params_to_dict(params_subset)
    rtrn = ds[var] ** params_dict['pwr']

    return rtrn


@profile
def gaussian(ds: pd.DataFrame
             , var: str
             , params_subset: pd.DataFrame
             , index_vars: str | float = np.nan):
    """
    Transformation: Gaussian (bell-curve, normal distribution). Useful to describe life cycle

    Interface is standard among all transformation functions

    :param ds: dataset
    :param var: raw variable string name
    :param params_subset: subset of `params` frame containing params specific for `var` variable and current
        transformation
    :param index_vars: string containing variable name or names to apply transformation separately, e.g.
        `"bs_key"` or `"superbrand", "master"`
    :return: transformed time series
    """

    params_dict = fermatrica.basics.basics.params_to_dict(params_subset)

    rtrn = ftr.gaussian(ds[var], params_dict)

    if 'if_scale_mean' in params_dict and params_dict['if_scale_mean']:
        mask = (ds['listed'].isin([2, 3]))
        rtrn = rtrn - rtrn.loc[mask].mean()

    return rtrn


@profile
def gaussianzero(ds: pd.DataFrame
                 , var: str
                 , params_subset: pd.DataFrame
                 , index_vars: str | float = np.nan):
    """
    Transformation: Gaussian (bell-curve, normal distribution) scaled minus 0-value

    Interface is standard among all transformation functions

    :param ds: dataset
    :param var: raw variable string name
    :param params_subset: subset of `params` frame containing params specific for `var` variable and current
        transformation
    :param index_vars: string containing variable name or names to apply transformation separately, e.g.
        `"bs_key"` or `"superbrand", "master"`
    :return: transformed time series
    """

    params_dict = fermatrica.basics.basics.params_to_dict(params_subset)

    rtrn = ftr.gaussian(ds[var], params_dict) - ftr.gaussian(0, params_dict)

    return rtrn


@profile
def lorentzian(ds: pd.DataFrame
               , var: str
               , params_subset: pd.DataFrame
               , index_vars: str | float = np.nan):
    """
    Transformation: Lorentzian (more "pointed" bell-curve)

    Interface is standard among all transformation functions

    :param ds: dataset
    :param var: raw variable string name
    :param params_subset: subset of `params` frame containing params specific for `var` variable and current
        transformation
    :param index_vars: string containing variable name or names to apply transformation separately, e.g.
        `"bs_key"` or `"superbrand", "master"`
    :return: transformed time series
    """

    params_dict = fermatrica.basics.basics.params_to_dict(params_subset)

    rtrn = ftr.lorentzian(ds[var], params_dict)

    return rtrn


@profile
def loggaussian(ds: pd.DataFrame
                , var: str
                , params_subset: pd.DataFrame
                , index_vars: str | float = np.nan):
    """
    Transformation: log-gaussian (asymmetric bell-curve)

    Interface is standard among all transformation functions

    :param ds: dataset
    :param var: raw variable string name
    :param params_subset: subset of `params` frame containing params specific for `var` variable and current
        transformation
    :param index_vars: string containing variable name or names to apply transformation separately, e.g.
        `"bs_key"` or `"superbrand", "master"`
    :return: transformed time series
    """

    params_dict = fermatrica.basics.basics.params_to_dict(params_subset)

    rtrn = params_dict['offset'] + ftr.log_gaussian(ds[var], params_dict)

    return rtrn


"""
Complex transformations
"""


@profile
def arl(ds: pd.DataFrame
        , var: str
        , params_subset: pd.DataFrame
        , index_vars: str | float = np.nan
        , adb: bool = True):
    """
    Transformation: complex transformation, combining saturation, weighted geometric adstock, lag (fixed order).
    Be very careful using ARL and ARLP functions with panels due to risk of overaggregating and ambiguous behavior

    Interface is standard among all transformation functions

    :param ds: dataset
    :param var: raw variable string name
    :param params_subset: subset of `params` frame containing params specific for `var` variable and current
        transformation
    :param index_vars: string containing variable name or names to apply transformation separately, e.g.
        `"bs_key"` or `"superbrand", "master"`
    :param adb: use logistic or adbudg saturation. Set as separate variable for historical reason,
        use `arlp()` function if `adb=False` is required
    :return: transformed time series
    """

    # check data type

    _check_type(ds, var)

    # run

    params_dict = fermatrica.basics.basics.params_to_dict(params_subset)

    # ds['tmp'] = ds[var].copy(deep=True)
    ds['tmp'] = ds[var].copy(deep=True)
    ds.iloc[0, 0] = ds.iloc[0, 0]  # consolidation

    # saturation

    if params_dict.get('steep', 0) != 0:
        if adb:
            result = ftr.adbudg(ds['tmp'], params_dict)
        else:
            result = ftr.logistic(ds['tmp'], params_dict)

        if isinstance(result, pd.Series):
            ds['tmp'] = result.values.copy()
        else:
            ds['tmp'] = np.asarray(result, dtype=np.float64).copy()

        ds.iloc[0, 0] = ds.iloc[0, 0]  # consolidation

    # adstock

    rtrn = adstock(ds, 'tmp', params_subset[params_subset['arg'] == 'a'], index_vars)
    # rtrn = pd.Series(rtrn.values.copy(), index=rtrn.index, name=rtrn.name)

    if isinstance(rtrn, pd.Series):
        ds['tmp'] = rtrn.copy(deep=True)
    else:
        ds['tmp'] = np.asarray(rtrn, dtype=np.float64).copy()
    ds.iloc[0, 0] = ds.iloc[0, 0]  # consolidation

    # prevent overaggregating

    if 'index_aggr' in params_subset.columns:
        index_aggr = params_subset.loc[params_subset['index_aggr'].notna(), 'index_aggr']

        if len(index_aggr) > 0 and index_aggr.iloc[0] not in ['max', 'sum_kpi_coef', 'sum_kpi_coef_master']:
            fermatrica_error('ARL and ARLP functions do not allow aggregating other than ' +
                             '`max`, `sum_kpi_coef`, `sum_kpi_coef_master`. Otherwise consistency of series ' +
                             'would be lost due to overaggregating and/or ambitious aggregation logic')

    # lag

    if 'n_lag' in params_dict:

        params_subset = params_subset.copy()  # important because of renaming `n_lag` param

        if 'cval' not in params_dict:
            params_subset = pd.concat([params_subset
                                          , pd.DataFrame({'arg': ['cval'], 'value': [np.nan]})]
                                      , axis=0)
        params_subset.loc[params_subset['arg'] == 'n_lag', 'arg'] = 'n'

        rtrn = lag(ds, 'tmp', params_subset, index_vars)

        if isinstance(rtrn, pd.Series):
            ds['tmp'] = rtrn.copy(deep=True)
        else:
            ds['tmp'] = np.asarray(rtrn, dtype=np.float64).copy()

        ds.iloc[0, 0] = ds.iloc[0, 0]  # consolidation

    # finalize

    rtrn = ds['tmp'].copy(deep=True)
    ds.iloc[0, 0] = ds.iloc[0, 0]  # consolidation

    del ds['tmp']  # much faster than .drop()

    return rtrn


@profile
def arlp(ds: pd.DataFrame
         , var: str
         , params_subset: pd.DataFrame
         , index_vars: str | float = np.nan):
    """
    Transformation: complex transformation, combining saturation, weighted geometric adstock, lag (fixed order).
    "Pure" flavour with S-version of saturation only.

    Be very careful using ARL and ARLP functions with panels due to risk of overaggregating and ambiguous behavior

    Interface is standard among all transformation functions

    :param ds: dataset
    :param var: raw variable string name
    :param params_subset: subset of `params` frame containing params specific for `var` variable and current
        transformation
    :param index_vars: string containing variable name or names to apply transformation separately, e.g.
        `"bs_key"` or `"superbrand", "master"`
    :return: transformed time series
    """

    rtrn = arl(ds, var, params_subset, index_vars, adb=False)

    return rtrn


@profile
def arld(ds: pd.DataFrame
        , var: str
        , params_subset: pd.DataFrame
        , index_vars: str | float = np.nan
        , adb: bool = True):
    """
    Transformation: complex transformation, combining saturation, weighted geometric adstock with two weights, lag (fixed order).
    Be very careful using ARL, ARLP and ARLD functions with panels due to risk of overaggregating and ambiguous behavior

    Interface is standard among all transformation functions

    :param ds: dataset
    :param var: raw variable string name
    :param params_subset: subset of `params` frame containing params specific for `var` variable and current
        transformation
    :param index_vars: string containing variable name or names to apply transformation separately, e.g.
        `"bs_key"` or `"superbrand", "master"`
    :param adb: use logistic or adbudg saturation. Set as separate variable for historical reason,
        create and use `arlpd()` function if `adb=False` is required
    :return: transformed time series
    """

    # check data type

    _check_type(ds, var)

    # run

    params_dict = fermatrica.basics.basics.params_to_dict(params_subset)

    # ds['tmp'] = ds[var].copy(deep=True)
    ds['tmp'] = ds[var].copy(deep=True)
    ds.iloc[0, 0] = ds.iloc[0, 0]  # consolidation

    # saturation

    if params_dict.get('steep', 0) != 0:
        if adb:
            result = ftr.adbudg(ds['tmp'], params_dict)
        else:
            result = ftr.logistic(ds['tmp'], params_dict)

        if isinstance(result, pd.Series):
            ds['tmp'] = result.values.copy()
        else:
            ds['tmp'] = np.asarray(result, dtype=np.float64).copy()

        ds.iloc[0, 0] = ds.iloc[0, 0]  # consolidation

    # adstock

    rtrn = adstockd(ds, 'tmp', params_subset[params_subset['arg'].isin(['a1', 'a2', 'w2'])], index_vars)
    # rtrn = pd.Series(rtrn.values.copy(), index=rtrn.index, name=rtrn.name)

    if isinstance(rtrn, pd.Series):
        ds['tmp'] = rtrn.copy(deep=True)
    else:
        ds['tmp'] = np.asarray(rtrn, dtype=np.float64).copy()

    ds.iloc[0, 0] = ds.iloc[0, 0]  # consolidation
    
    # prevent overaggregating

    if 'index_aggr' in params_subset.columns:
        index_aggr = params_subset.loc[params_subset['index_aggr'].notna(), 'index_aggr']

        if len(index_aggr) > 0 and index_aggr.iloc[0] not in ['max', 'sum_kpi_coef', 'sum_kpi_coef_master']:
            fermatrica_error('ARL, ARLP and ARLD functions do not allow aggregating other than ' +
                             '`max`, `sum_kpi_coef`, `sum_kpi_coef_master`. Otherwise consistency of series ' +
                             'would be lost due to overaggregating and/or ambitious aggregation logic')

    # lag

    if 'n_lag' in params_dict:

        params_subset = params_subset.copy()  # important because of renaming `n_lag` param

        if 'cval' not in params_dict:
            params_subset = pd.concat([params_subset
                                          , pd.DataFrame({'arg': ['cval'], 'value': [np.nan]})]
                                      , axis=0)
        params_subset.loc[params_subset['arg'] == 'n_lag', 'arg'] = 'n'

        rtrn = lag(ds, 'tmp', params_subset, index_vars)

        ds['tmp'] = rtrn.copy()
        ds.iloc[0, 0] = ds.iloc[0, 0]  # consolidation

    # finalize

    if isinstance(ds['tmp'], pd.Series):
        rtrn = ds['tmp'].copy(deep=True)
    else:
        rtrn = np.asarray(ds['tmp'], dtype=np.float64).copy()
        
    ds.iloc[0, 0] = ds.iloc[0, 0]  # consolidation

    del ds['tmp']  # much faster than .drop()

    return rtrn


@profile
def price(ds: pd.DataFrame
          , var_raw: str
          , params_subset: pd.DataFrame
          ):
    """
    Transformation: complex transformation to calculate relative prices

    Interface is standard among all transformation functions

    :param ds: dataset
    :param var: raw variable string name
    :param params_subset: subset of `params` frame containing params specific for `var` variable and current
        transformation
    :param index_vars: string containing variable name or names to apply transformation separately, e.g.
        `"bs_key"` or `"superbrand", "master"`
    :return: transformed time series
    """

    params_dict = fermatrica.basics.basics.params_to_dict(params_subset)

    threshold_var = ast.literal_eval(params_dict['threshold_var'])
    share_threshold = params_dict['share_threshold']
    index_vars_time = ast.literal_eval('[' + params_dict['index_vars_time'] + ']')
    year_start = params_dict['year_start']
    if_clutter = params_dict['if_clutter']
    inn_coef = params_dict['inn_coef']

    if pd.notna(params_dict['index_vars_dim']):
        index_vars_dim = ast.literal_eval('[' + params_dict['index_vars_dim'] + ']')
    else:
        index_vars_dim = []

    weighted_by = params_dict['weighted_by']
    if isinstance(weighted_by, str):
        weighted_by = re.sub(r'"', '', weighted_by)

    index_vars_list = index_vars_dim + index_vars_time

    if len(index_vars_dim) == 0:
        index_vars_dim = ["media"]

    if "media" not in ds.columns:
        ds['media'] = 'mock'

    if index_vars_dim == ["media"]:
        var = var_raw + "_rel_total"
    else:
        var = var_raw + "_rel_" + '_'.join(index_vars_dim)

    if 'tmp' in ds.columns:
        del ds['tmp']

    if pd.isna(weighted_by):

        ds['tmp'] = np.nan
        cols = index_vars_list + [var_raw, threshold_var, 'listed', 'is_inn', 'date']
        cols = list(set(cols))
        ds_tmp = select_eff(ds, cols)

        ds_tmp.loc[:, 'tmp'] = ds_tmp.loc[:, var_raw].array * (1 + ds_tmp.loc[:, 'is_inn'].array * inn_coef)

        mask = (ds_tmp[threshold_var] > share_threshold) & ((ds_tmp['date'].dt.year < year_start)
                                                            | ds_tmp['listed'].isin([2, 3, 4]))
        rtrn = ds_tmp[mask].groupby(index_vars_list)
        rtrn = rtrn.tmp.mean()

        ds_tmp = select_eff(ds, cols)
        ds_tmp = ds_tmp.merge(rtrn.reset_index(name='tmp'), how='left', on=index_vars_list,
                              sort=False, copy=False)

        ds.loc[mask, 'tmp'] = ds_tmp.loc[mask, 'tmp']

        ds['tmp_2'] = np.nan
        mask = ((ds[var_raw] > 0) & (ds['date'].dt.year < year_start)) | (ds['listed'].isin([2, 3, 4]))

        rtrn = ds_tmp[mask].groupby(index_vars_list)
        rtrn = rtrn.tmp.mean()

        ds_tmp = select_eff(ds, index_vars_list)
        ds_tmp = ds_tmp.merge(rtrn.reset_index(name='tmp_2'), how='left', on=index_vars_list,
                              sort=False, copy=False)

        ds.loc[mask, 'tmp_2'] = ds_tmp.loc[mask, 'tmp_2']

    else:

        ds['tmp'] = np.nan
        cols = index_vars_list + [var_raw, threshold_var, weighted_by, 'listed', 'is_inn', 'date']
        cols = list(set(cols))
        ds_tmp = select_eff(ds, cols)

        ds_tmp.loc[:, 'tmp'] = ds_tmp.loc[:, var_raw].array * (1 + ds_tmp.loc[:, 'is_inn'].array * inn_coef)

        mask = (ds_tmp[threshold_var] > share_threshold) & ((ds_tmp['date'].dt.year < year_start)
                                                            | ds_tmp['listed'].isin([2, 3, 4]))

        rtrn = weighted_mean_group(ds_tmp[mask], 'tmp', weighted_by, index_vars_list)

        ds_tmp = select_eff(ds, cols)
        ds_tmp = ds_tmp.merge(rtrn.reset_index(name='tmp'), how='left', on=index_vars_list,
                              sort=False, copy=False)

        ds.loc[mask, 'tmp'] = ds_tmp.loc[mask, 'tmp']

        ds['tmp_2'] = np.nan
        mask = ((ds[var_raw] > 0) & (ds['date'].dt.year < year_start)) | (ds['listed'].isin([2, 3, 4]))

        rtrn = weighted_mean_group(ds_tmp[mask], 'tmp', weighted_by, index_vars_list)

        ds_tmp = select_eff(ds, index_vars_list)
        ds_tmp = ds_tmp.merge(rtrn.reset_index(name='tmp_2'), how='left', on=index_vars_list,
                              sort=False, copy=False)

        ds.loc[mask, 'tmp_2'] = ds_tmp.loc[mask, 'tmp_2']

    ds.loc[ds['tmp_2'].isna(), 'tmp'] = ds.loc[ds['tmp_2'].isna(), var_raw] * (
            1 + ds.loc[ds['tmp_2'].isna(), 'is_inn'] * inn_coef)
    del ds['tmp_2']

    mask = ((ds[var_raw] > 0) & (ds['date'].dt.year < year_start)) | (ds['listed'].isin([2, 3, 4]))

    rtrn = groupby_eff(ds, index_vars_list, ['tmp'], mask, sort=False).mean()
    #
    ds_tmp = select_eff(ds, index_vars_list)
    ds_tmp = ds_tmp.merge(rtrn.reset_index(), how='left', on=index_vars_list,
                          sort=False, copy=False)

    ds.loc[mask, 'tmp_mean'] = ds_tmp.loc[mask, 'tmp']

    ds.loc[ds['tmp'].isna(), 'tmp'] = ds.loc[ds['tmp'].isna(), 'tmp_mean']

    del ds['tmp_mean']

    ds.loc[mask, var] = ds.loc[mask, 'tmp'] - ds.loc[mask, var_raw] * (1 + ds.loc[mask, 'is_inn'] * inn_coef)
    ds.loc[mask, var] = ds.loc[mask, var] / ds.loc[mask, 'tmp']

    del ds['tmp']

    mask = ds['listed'].isin([2, 3])
    rtrn = groupby_eff(ds, index_vars_dim, [var], mask, sort=False)[var].min()
    ds_tmp = select_eff(ds, index_vars_dim)
    ds_tmp = ds_tmp.merge(rtrn.reset_index(name='tmp'), how='left', on=index_vars_dim,
                          sort=False, copy=False)
    ds['tmp'] = ds_tmp['tmp']

    ds.loc[:, var + '_positive'] = ds.loc[:, var] - ds.loc[:, 'tmp']
    del ds['tmp']

    mask = ds['listed'].isin([2, 3])
    rtrn = groupby_eff(ds, index_vars_dim, [var], mask, sort=False)[var].max()
    ds_tmp = select_eff(ds, index_vars_dim)
    ds_tmp = ds_tmp.merge(rtrn.reset_index(name='tmp'), how='left', on=index_vars_dim,
                          sort=False, copy=False)
    ds['tmp'] = ds_tmp['tmp']

    ds.loc[:, var + '_negative'] = ds.loc[:, var] - ds.loc[:, 'tmp']
    del ds['tmp']

    if if_clutter:

        if index_vars_dim == ['media']:
            var_more = var_raw + '_clutter_total'
        else:
            var_more = var_raw + "_clutter_" + '_'.join(index_vars_dim)

        mask = ((ds[var_raw] > 0) & (ds['date'].dt.year < year_start)) | (ds['listed'].isin([2, 3, 4]))

        rtrn = groupby_eff(ds, index_vars_list, [var], mask, sort=False).transform(lambda x: ftr.sum_inv_exp_dist(x))

        ds_tmp = select_eff(ds, index_vars_list)
        ds_tmp = ds_tmp.join(rtrn, how='left', sort=False)
        ds.loc[mask, var_more] = ds_tmp.loc[mask, var]

        mask = ds['listed'].isin([2, 3])
        rtrn = groupby_eff(ds, index_vars_dim, [var_more], mask, sort=False)[var_more].min()
        ds_tmp = select_eff(ds, index_vars_dim)
        ds_tmp = ds_tmp.merge(rtrn.reset_index(name='tmp'), how='left', on=index_vars_dim,
                              sort=False, copy=False)
        ds['tmp'] = ds_tmp['tmp']

        rtrn = groupby_eff(ds, index_vars_dim, ['tmp'], sort=False, mask=None).transform(lambda x: x.fillna(x.mean()))

        ds_tmp = select_eff(ds, index_vars_dim)
        ds_tmp = ds_tmp.join(rtrn, how='left', sort=False)
        ds['tmp'] = ds_tmp['tmp']

        ds.loc[:, var_more] = ds.loc[:, var_more] - ds.loc[:, 'tmp']
        del ds['tmp']

    pass

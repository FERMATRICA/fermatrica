"""
Transformation functions per se: working (mostly) with vectors. Returned value is typically vector or series
containing new, transformed variable.

For medium- and high-level wrappers see `fermatrica.model.transform`.

-----------------------------------------------

The function `recursive_filter` has the specific license, different form the license of the other source code of the file
and of the whole project. The function `recursive_filter` is adopted with some modification from `statsmodels` library
and is redistributed under BSD-3-Clause License:

"
Copyright (C) 2006, Jonathan E. Taylor
All rights reserved.

Copyright (c) 2006-2008 Scipy Developers.
All rights reserved.

Copyright (c) 2009-2018 statsmodels Developers.
All rights reserved.


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of statsmodels nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL STATSMODELS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
"
"""


import math
import re

from line_profiler_pycharm import profile
import numpy as np
import pandas as pd

from scipy.stats import dweibull
import scipy.signal as signal
from statsmodels.tools.validation.validation import array_like


"""
Time series transformations
"""


@profile
def recursive_filter(x: pd.Series | np.ndarray | list
                     , ar_coeff: pd.Series | np.ndarray | list
                     , init=None):
    """
    Autoregressive, or recursive, filtering.
    More performance efficient version derived from statsmodel's.

    Computes the recursive filter ::

        y[n] = ar_coeff[0] * y[n-1] + ...
                + ar_coeff[n_coeff - 1] * y[n - n_coeff] + x[n]

    where n_coeff = len(n_coeff).

    :param x: array_like
        Time-series 1d data as Pandas Series or Numpy array
    :param ar_coeff: array_like
        AR coefficients in reverse time order. See Notes for details.
    :param init: array_like
        Initial values of the time-series prior to the first value of y.
        The default is zero.
    :return: array_like
        Filtered array, number of columns determined by x and ar_coeff. If x
        is a pandas object than a Series is returned.

    """

    if isinstance(x, pd.Series):
        x = x.to_numpy(dtype=np.double, copy=True)
    elif not isinstance(x, np.ndarray):
        x = np.asarray(x, dtype=np.double, order=None)

    if isinstance(ar_coeff, pd.Series):
        ar_coeff = ar_coeff.to_numpy(dtype=np.double, copy=True)
    elif not isinstance(ar_coeff, np.ndarray):
        ar_coeff = np.asarray(ar_coeff, dtype=np.double, order=None)

    if init is not None:  # integer init are treated differently in lfiltic
        init = array_like(init, 'init')
        if len(init) != len(ar_coeff):
            raise ValueError("ar_coeff must be the same length as init")

    if init is not None:
        zi = signal.lfiltic([1], np.r_[1, -ar_coeff], init, x)
    else:
        zi = None

    y = signal.lfilter([1.], np.r_[1, -ar_coeff], x, zi=zi)

    if init is not None:
        result = y[0]
    else:
        result = y

    return result


@profile
def weibull_multi_response(x: pd.Series | np.ndarray
                           , params_dict: dict
                           , if_weight: bool = False):
    """
    Weibull-based time series transformation. More sophisticated substitute to
    geometric adstock, but not especially human-friendly.

    :param x: vector / series to apply transformation
    :param params_dict: transformation function params
    :param if_weight:
    :return:
    """

    if isinstance(x, pd.Series):
        x = x.to_numpy().astype(float)

    if np.count_nonzero(x) == 0:
        return x

    idxs = np.nonzero(x)[0]
    ks = x[x != 0]

    if if_weight:
        ks = ks / sum(ks)  # evil

    if 'scale' not in params_dict:
        params_dict['scale'] = 1

    weibull_x = dweibull.pdf(range(1, round(params_dict['dur']) + 1), c=params_dict['shape'], scale=params_dict['scale'])
    weibull_x[np.isinf(weibull_x)] = 0

    rtrn_len = len(x)
    idxs_len = len(idxs)

    rtrn = np.zeros(rtrn_len * (idxs_len + 1)).reshape(-1, (idxs_len + 1))

    for i in range(0, idxs_len):

        tmp = np.zeros(rtrn_len)

        len_new = rtrn_len - idxs[i]
        if len_new > len(weibull_x):
            len_new = len(weibull_x)

        tmp[idxs[i]:(idxs[i] + len_new)] = (weibull_x * ks[i])[:len_new]

        rtrn[:, (i+1)] = tmp

    rtrn = rtrn.sum(axis=1, keepdims=True)

    if sum(np.isfinite(rtrn))[0] != len(x):
        rtrn = np.repeat(0, len(x), axis=0)

    return rtrn.flatten()


"""
Cross-sectional & non-dimension-specific transformations
"""


@profile
def sum_inv_exp_dist(x: pd.Series | np.ndarray
                     # , wght=None
                     , product=1
                     ):
    """
    Sum of inverted exponented distance. Used in price calculation to find brand clusters by price

    :param x: vector / series to apply transformation
    :param product:
    :return:
    """

    if isinstance(x, pd.Series):
        x1 = x.to_numpy()
    else:
        x1 = x

    m = np.zeros((len(x1), len(x1)))

    for i in range(len(x1)):
        for j in range(i, len(x1)):
            m[j, i] = math.dist([x1[i]], [x1[j]])

    m = m + m.T
    m = 1 / (np.exp(m) * product)

    m = pd.DataFrame(m)
    m.replace([np.inf, -np.inf], 0, inplace=True)
    if isinstance(x, pd.Series):
        m.index = x.index

    # if not pd.isna(wght):
    #     m = m * wght

    m = m.sum(axis=1)

    return m


@profile
def scale_classic_median(x: pd.Series | np.ndarray
                         , mask: pd.Series | np.ndarray
                         ):
    """
    Classic scaling: (X - median(X)) / STD(X).

    Infinite values created after calculation are replaced with raw values: effectively if raw vector
    is constant and standard deviation is 0, the whole vector to be returned as is.

    :param x: vector / series to apply transformation
    :param mask:
    :return:
    """

    if_array = False
    if isinstance(x, np.ndarray):
        x = pd.Series(x)
        if_array = True

    if isinstance(mask, pd.Series):
        mask = mask.values

    x_r = (x - x.loc[mask].median()) / x.loc[mask].std()

    mask = x_r.isin([np.inf, -np.inf])
    x_r.loc[mask] = x.loc[mask]
    x_r.loc[x_r.isna()] = 0

    if if_array:
        x_r = x_r.to_numpy()

    rtrn = x_r

    return rtrn


def scale_level(x: pd.Series | np.ndarray
                , mask: pd.Series | np.ndarray
                , params_dict: dict):
    """

    :param x: 
    :param mask:
    :param params_dict: transformation function params
    :return:
    """

    w = 1.0

    if params_dict['scale'] == 'max':
        w = x.loc[mask].max()

    elif params_dict['scale'] == 'min':
        w = x.loc[mask].min()

    elif params_dict['scale'] == 'avg_max':
        w = x.loc[mask].max()
        x = x * params_dict['avg_level']

    elif params_dict['scale'] == 'avg_min':
        w = x.loc[mask].min()
        x = x * params_dict['avg_level']

    if w not in [0, 1]:
        x = x / w

    return x

"""
Saturation transformations
"""


@profile
def adbudg(x: pd.Series | np.ndarray
           , params_dict: dict):
    """
    Saturation curve with both power / logarithm and S-curve versions depending on param values.

    :param x: vector / series to apply transformation
    :param params_dict: transformation function params
    :return:
    """

    if isinstance(x, pd.Series):
        x = x.array

    with np.errstate(divide='ignore', invalid='ignore'):
        summand = np.sign(x / (params_dict['level'] * params_dict['ec50'])) * \
                  abs(x / (params_dict['level'] * params_dict['ec50'])) ** (-params_dict['steep'])

    rtrn = params_dict['level'] / (1 + summand)
    rtrn = np.where(x == 0, 0, rtrn)

    return rtrn


@profile
def logistic(x: pd.Series | np.ndarray
             , params_dict: dict):
    """
    Logistic saturation curve.

    :param x: vector / series to apply transformation
    :param params_dict: transformation function params
    :return:
    """

    if isinstance(x, pd.Series):
        x = x.array

    rtrn = params_dict['level'] / (
            1 + np.exp((-params_dict['steep']) * (x / params_dict['level'] - params_dict['ec50']))) - params_dict[
               'level'] / (
                   1 + np.exp(params_dict['steep'] * params_dict['ec50']))

    return rtrn


@profile
def softmax(x: pd.Series | np.ndarray
            , params_dict: dict):
    """
    Flavour of logistic saturation curve with unit-specific `avg` and "abstract" `lambda`.
    The more is lambda, less steep is the curve.

    :param x: vector / series to apply transformation
    :param params_dict: transformation function params
    :return:
    """

    if isinstance(x, pd.Series):
        x = x.array

    rtrn = (x - params_dict['avg']) / (params_dict['lambda'] * (params_dict['std'] / (2 * np.pi)))
    rtrn = 1 / (1 + np.exp(-rtrn))

    return rtrn


@profile
def gompertz(x: pd.Series | np.ndarray
             , params_dict: dict):
    """
    Gompertz asymmetric sigmoid.

    :param x: vector / series to apply transformation
    :param params_dict: transformation function params
    :return:
    """

    if pd.isna(params_dict['avg']):
        avg = np.mean(x)
    else:
        avg = params_dict['avg']

    if pd.isna(params_dict['std']):
        std = np.std(x)
    else:
        std = params_dict['std']

    rtrn = params_dict['a'] * np.exp(-params_dict['b'] * np.exp(-params_dict['c'] * (x - avg) / std))

    return rtrn


@profile
def gaussian(x: pd.Series | np.ndarray
             , params_dict: dict):
    """
    Gaussian (bell-curve, normal distribution).

    :param x: vector / series to apply transformation
    :param params_dict: transformation function params
    :return:
    """

    rtrn = params_dict['amplitude'] * np.exp(-np.log(2) * (x - params_dict['expvalue']) ** 2 / params_dict['hwhm'] ** 2)

    return rtrn


@profile
def log_gaussian(x: pd.Series | np.ndarray
                 , params_dict: dict):
    """
    Log-gaussian (asymmetric bell-curve)

    :param x: vector / series to apply transformation
    :param params_dict: transformation function params
    :return:
    """

    rtrn = params_dict['amplitude'] * np.exp(-np.log(2) * (np.log(x) - np.log(params_dict['expvalue'])) ** 2 / np.log(params_dict['hwhm']) ** 2) / x

    return rtrn


@profile
def lorentzian(x: pd.Series | np.ndarray
               , params_dict: dict):
    """
    Lorentzian (more "pointed" bell-curve).

    :param x: vector / series to apply transformation
    :param params_dict: transformation function params
    :return:
    """

    rtrn = params_dict['amplitude'] / (1 + (x - params_dict['expvalue']) ** 2 / params_dict['hwhm'] ** 2)

    return rtrn


"""
Complex transformations
"""


@profile
def tv_clipl_affinity(ds: pd.DataFrame
                      , params_dict: dict
                      , cln_tv: list
                      , cln_aff: list
                      , cln_clipl: list
                      , tv_pattern: str):
    """
    Complex function to weight TV OTS by brands' affinity and clip duration

    :param ds: dataset
    :param params_dict:
    :param cln_tv: names of the TV OTS columns
    :param cln_aff: names of the columns with brands' affinity data
    :param cln_clipl: names of the clip duration columns
    :param tv_pattern: specific regex pattern containing media type and placement type, e.g. 'sponsor_nat'
    :return:
    """

    if len([i for i in cln_tv if (bool(re.search(tv_pattern, i)))]) == 0:
        return pd.Series(0, index=ds.index)

    def affinity_series(s: str):

        s_pattern = re.sub('(^(.+_)*ots_)|(_rolik|_sponsor|_nat|_local)', '', s)

        s_aff = [i for i in cln_aff if (bool(re.search(s_pattern, i)))
                 or (bool(re.search('m55plus', s)) and bool(re.search('m55_64', i)))
                 or (bool(re.search('f55plus', s)) and bool(re.search('f55_64', i)))
                 or (bool(re.search('m14_24', s)) and bool(re.search('m18_24', i)))
                 or (bool(re.search('f14_24', s)) and bool(re.search('f18_24', i)))]

        s_clipl = [i for i in cln_clipl if (s_pattern in i) and (tv_pattern in i)]

        if len(s_aff) == 0 or len(s_clipl) == 0:
            # rtrn = rep(0, ds.shape[0])
            rtrn = pd.Series(0, index=ds.index)
        else:
            # rtrn = ds.loc[:, s] * (ds.loc[:, s_aff] ** params_dict['power_coef']).iloc[:, 0] *\
            #        (ds.loc[:, s_clipl] ** params_dict['clipl_curve']).iloc[:, 0]
            rtrn = ds[s].array * (ds[s_aff[0]].array ** params_dict['power_coef']) * (ds[s_clipl[0]].array ** params_dict['clipl_curve'])
            rtrn = pd.Series(rtrn, index=ds.index)

        return rtrn

    rtrn = []
    for col in cln_tv:
        if re.search(tv_pattern, col):
            tmp = affinity_series(col)
            rtrn.append(tmp)
    rtrn = pd.concat(rtrn, axis=1)
    rtrn = rtrn.sum(axis=1)

    return rtrn


@profile
def tv_affinity(ds: pd.DataFrame
                , params_dict: dict
                , cln_tv: list
                , cln_aff: list
                , tv_pattern: str):
    """
    Complex function to weight TV OTS by brands' affinity. The difference with `tv_affinity_clipl`
    is ignoring clip duration here.

    :param ds: dataset
    :param params_dict:
    :param cln_tv: names of the TV OTS columns
    :param cln_aff: names of the columns with brands' affinity data
    :param cln_clipl: names of the clip duration columns
    :param tv_pattern: specific regex pattern containing media type and placement type, e.g. 'sponsor_nat'
    :return:
    """

    if len([i for i in cln_tv if (bool(re.search(tv_pattern, i)))]) == 0:
        return pd.Series(0, index=ds.index)

    def affinity_series(s: str):

        s_pattern = re.sub('(^(.+_)*ots_)|(_rolik|_sponsor|_nat|_local)', '', s)

        s_aff = [i for i in cln_aff if (bool(re.search(s_pattern, i)))
                 or (bool(re.search('m55plus', s)) and bool(re.search('m55_64', i)))
                 or (bool(re.search('f55plus', s)) and bool(re.search('f55_64', i)))
                 or (bool(re.search('m14_24', s)) and bool(re.search('m18_24', i)))
                 or (bool(re.search('f14_24', s)) and bool(re.search('f18_24', i)))]

        if len(s_aff) == 0:
            rtrn = pd.Series(0, index=ds.index)
        else:
            rtrn = ds[s] * (ds[s_aff[0]].array ** params_dict['power_coef'])

        return rtrn

    rtrn = []
    for col in cln_tv:
        if re.search(tv_pattern, col):
            tmp = affinity_series(col)
            rtrn.append(tmp)
    rtrn = pd.concat(rtrn, axis=1)
    rtrn = rtrn.sum(axis=1)

    return rtrn

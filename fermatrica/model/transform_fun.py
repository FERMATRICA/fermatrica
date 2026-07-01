"""
Transformation functions per se: working (mostly) with vectors. Returned value is typically vector or series
containing new, transformed variable.

For medium- and high-level wrappers see `fermatrica.model.transform`.
"""


import logging
import math
import re

from line_profiler_pycharm import profile
import numpy as np
import pandas as pd

try:
    import bottleneck as bn
    _BN_AVAILABLE = True
except ImportError:
    _BN_AVAILABLE = False

_logger = logging.getLogger(__name__)

import scipy.signal as signal
from statsmodels.tools.validation.validation import array_like

from fermatrica.basics.basics import fermatrica_error


"""
Grouping helpers
"""


def _group_sort_info(group_codes: np.ndarray):
    """
    For an integer-encoded group label array, return the stable-sort permutation,
    its inverse, the sorted codes, and the boundary positions where each group
    starts in sorted order (with the total length appended).

    Used to broadcast per-group vectorized operations on a flat numpy array
    while preserving the original row order of the source frame.
    """

    order = np.argsort(group_codes, kind='stable')
    inv = np.empty_like(order)
    inv[order] = np.arange(order.shape[0], dtype=order.dtype)

    sorted_codes = group_codes[order]
    n_total = sorted_codes.shape[0]

    if n_total == 0:
        boundaries = np.array([0], dtype=np.intp)
    else:
        is_new = np.empty(n_total, dtype=bool)
        is_new[0] = True
        is_new[1:] = sorted_codes[1:] != sorted_codes[:-1]
        boundaries = np.flatnonzero(is_new).astype(np.intp)
        boundaries = np.append(boundaries, np.intp(n_total))

    return order, inv, boundaries


def _apply_grouped(arr: np.ndarray
                   , group_codes: np.ndarray
                   , fn) -> np.ndarray:
    """
    Apply `fn(slice) -> np.ndarray` independently to each group's contiguous rows
    of `arr`, in the original row order, and return results in the original row order.

    Replaces `groupby(...)[var].transform(lambda x: fn(x.to_numpy()))` without
    per-group pandas overhead.  O(N log N) for the stable sort; O(N) thereafter.

    `fn` must accept and return a 1-D float64 numpy array of the same length.
    Within-group order passed to `fn` matches the original frame order because
    `np.argsort(..., kind='stable')` preserves the relative order of equal elements.
    """

    order, inv, boundaries = _group_sort_info(group_codes)
    sorted_arr = arr[order]
    sorted_out = np.empty_like(sorted_arr, dtype=np.float64)

    for i in range(boundaries.shape[0] - 1):
        s = int(boundaries[i])
        e = int(boundaries[i + 1])
        sorted_out[s:e] = fn(sorted_arr[s:e])

    return sorted_out[inv]


"""
Time series transformations
"""


def lag(arr: np.ndarray
            , group_codes: np.ndarray | None
            , n: int
            , cval: float = 0.0) -> np.ndarray:
    """
    Order-preserving positional lag of `arr` by `n` rows.

    Behaves like `pd.Series.shift(n, fill_value=cval)` when `group_codes` is None,
    and like `groupby(...)[var].shift(n, fill_value=cval)` (preserving original
    row order; the lag is positional within each group as the rows appear in
    the source frame) when `group_codes` is provided.

    Pure-numpy implementation: O(N) plus an O(N log N) stable sort when grouped.
    """

    n_total = arr.shape[0]

    if n == 0:
        return arr.astype(np.float64, copy=True)

    if group_codes is None:
        out = np.empty(n_total, dtype=np.float64)
        if n >= n_total:
            out[:] = cval
        else:
            out[:n] = cval
            out[n:] = arr[:-n]
        return out

    order, inv, boundaries = _group_sort_info(group_codes)
    sorted_arr = arr[order]

    arange = np.arange(n_total, dtype=np.intp)
    group_start = np.repeat(boundaries[:-1], np.diff(boundaries))
    pos = arange - group_start

    mask = pos >= n
    src_idx = np.where(mask, arange - n, 0)
    sorted_out = np.where(mask, sorted_arr[src_idx], np.float64(cval))

    return sorted_out[inv]


def mar(arr: np.ndarray
            , group_codes: np.ndarray | None
            , valid_mask: np.ndarray
            , n: int
            , algo: str = 'precise') -> np.ndarray:
    """
    Right moving average with window `n` and `min_count=1`, NaN-aware via an
    explicit `valid_mask`. Output positions where `valid_mask` is False are
    filled with NaN.

    When `group_codes` is provided, the moving average is computed independently
    within each group, on the subset of rows where `valid_mask` is True,
    in the original row order of the source frame.

    algo:
      'precise' (default) — uses `pd.Series.rolling`, bit-exact with the
          original mar_eff-based implementation.
      'fast'    — uses `bottleneck.move_mean` (requires bottleneck installed);
          ~2-5x faster but accumulates floating-point error O(N * eps * max_val)
          over long series (~1e-8 for typical media data). Safe when downstream
          transformations are not near-zero or near-infinity.
    """

    n_total = arr.shape[0]
    out = np.full(n_total, np.nan, dtype=np.float64)

    if isinstance(n, float):
        n = int(round(n))

    if n == 0:
        out[valid_mask] = arr[valid_mask]
        return out

    valid_idx = np.flatnonzero(valid_mask)
    if valid_idx.size == 0:
        return out

    valid_arr = arr[valid_idx].astype(np.float64, copy=False)

    use_bn = algo == 'fast' and _BN_AVAILABLE

    if algo == 'fast' and not _BN_AVAILABLE:
        _logger.warning(
            "mar: algo='fast' requested but 'bottleneck' is not installed; "
            "falling back to algo='precise'. Install bottleneck to enable the fast path."
        )

    def _rolling(a: np.ndarray) -> np.ndarray:
        if use_bn:
            return bn.move_mean(a, window=n, min_count=1)
        return pd.Series(a).rolling(window=n, min_periods=1).mean().to_numpy()

    if group_codes is None:
        out[valid_idx] = _rolling(valid_arr)
        return out

    valid_codes = group_codes[valid_idx]
    order, _, boundaries = _group_sort_info(valid_codes)
    sorted_arr = valid_arr[order]
    sorted_idx = valid_idx[order]
    sorted_out = np.empty_like(sorted_arr, dtype=np.float64)

    for i in range(boundaries.shape[0] - 1):
        s = int(boundaries[i])
        e = int(boundaries[i + 1])
        sorted_out[s:e] = _rolling(sorted_arr[s:e])

    out[sorted_idx] = sorted_out
    return out


def adstock(x: np.ndarray
            , a: float
            , weighted: bool = True) -> np.ndarray:
    
    """Geometric adstock via lfilter. x must be a float64 numpy array."""

    result = signal.lfilter([1.0], [1.0, -float(a)], x)

    if weighted:
        result *= (1.0 - float(a))

    return result


def adstockd(x: np.ndarray
             , a1: float
             , a2: float
             , w2: float
             , weighted: bool = True) -> np.ndarray:
    
    """Double geometric adstock via two lfilter calls, linearly combined."""

    a1, a2, w2 = float(a1), float(a2), float(w2)
    r1 = signal.lfilter([1.0], [1.0, -a1], x)
    r2 = signal.lfilter([1.0], [1.0, -a2], x)

    if weighted:
        r1 *= (1.0 - a1)
        r2 *= (1.0 - a2)

    return r1 * (1.0 - w2) + r2 * w2


def _compute_weibull_kernel(params_dict: dict) -> np.ndarray:
    """Compute the causal double-Weibull response kernel from params."""
    c = np.float64(params_dict['shape'])
    scale = np.float64(params_dict.get('scale', 1))
    dur = int(round(params_dict['dur']))
    if c <= 0 or scale <= 0 or dur <= 0:
        return np.zeros(0)
    t = np.arange(1.0, dur + 1.0, dtype=np.float64)
    t_s = t / scale
    kernel = (c / (2.0 * scale)) * t_s ** (c - 1.0) * np.exp(-(t_s ** c))
    kernel[np.isinf(kernel)] = 0.0
    return kernel


def _weibull_convolve(x: np.ndarray
                      , kernel: np.ndarray
                      , if_weight: bool = False) -> np.ndarray:
    """Apply causal convolution of x with a precomputed Weibull kernel, truncated to len(x)."""
    rtrn_len = len(x)

    if len(kernel) == 0 or np.count_nonzero(x) == 0:
        return np.zeros(rtrn_len)

    if if_weight:
        nonzero_mask = x != 0
        x = x.copy()
        x[nonzero_mask] /= x[nonzero_mask].sum()

    rtrn = np.convolve(x, kernel, mode='full')[:rtrn_len]

    if not np.all(np.isfinite(rtrn)):
        return np.zeros(rtrn_len)

    return rtrn


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
    elif not isinstance(x, np.ndarray):
        x = np.asarray(x, dtype=float)

    kernel = _compute_weibull_kernel(params_dict)
    return _weibull_convolve(x, kernel, if_weight=if_weight)


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

    if params_dict['scale'] in ['max', 'max_rat']:
        w = x.loc[mask].max()

    if params_dict['scale'] in ['mean', 'mean_rat']:
        w = x.loc[mask].mean()

    if params_dict['scale'] in ['median', 'median_rat']:
        w = x.loc[mask].median()

    elif params_dict['scale'] in ['min', 'min_rat']:
        w = x.loc[mask].min()

    elif params_dict['scale'] in ['avg_max', 'avg_max_rat']:
        w = x.loc[mask].max()
        x = x * params_dict['avg_level']

    elif params_dict['scale'] in ['avg_min', 'avg_min_rat']:
        w = x.loc[mask].min()
        x = x * params_dict['avg_level']

    # to restore scale (important for LHS)

    if w not in [0, 1] and not params_dict['scale'].endswith('_rat'):
        x = x * w

    # to raw data scale

    if w not in [0, 1] and params_dict['scale'].endswith('_rat'):
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
        x = x.to_numpy(dtype=np.float64, copy=True)
    else:
        x = np.asarray(x, dtype=np.float64)

    x = x.copy()

    rtrn = np.zeros_like(x, dtype=np.float64)
    mask = (x != 0) & np.isfinite(x)
    
    if not np.any(mask):
        return rtrn
    
    x_masked = x[mask]
    denominator = params_dict['level'] * params_dict['ec50']
    
    ratio = x_masked / denominator
    sign_ratio = np.sign(ratio)
    abs_ratio = np.abs(ratio)

    summand = sign_ratio * (abs_ratio ** (-params_dict['steep']))

    finite_mask = np.isfinite(summand).copy()
    summand[~finite_mask] = np.float64(0.0)

    rtrn[mask] = params_dict['level'] / (1 + summand)

    return rtrn.copy()


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

    if params_dict['lambda'] == 0:
        fermatrica_error('Lambda parameter for softmax transformation cannot be 0')

    if isinstance(x, pd.Series):
        x = x.to_numpy(dtype=np.float64, copy=True)
    else:
        x = np.asarray(x, dtype=np.float64)
    
    x = x.copy()
    
    avg = np.float64(params_dict['avg'])
    std = np.float64(params_dict['std'])
    lmbda = np.float64(params_dict['lambda'])
    
    x_normalized = (x - avg) / (lmbda * std / (2 * np.pi))
    
    exp_term = np.exp(-x_normalized)
    # exp_term = np.clip(exp_term, 0, 1e308)
    
    denominator = np.float64(1.0) + exp_term
    # denominator = np.clip(denominator, 1e-308, np.inf)
    
    rtrn = np.float64(1.0) / denominator
    
    # rtrn[~np.isfinite(rtrn)] = 0
    
    return rtrn.copy()


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
                      , cln_aff: list | None = None
                      , cln_clipl: list | None = None
                      , tv_pattern: str | None = None):
    """
    Complex function to weight TV OTS by brands' affinity and clip duration

    :param ds: dataset
    :param params_dict:
    :param cln_tv: names of the TV OTS columns
    :param cln_aff: names of the columns with brands' affinity data. If omitted, affinity will not be taken into account
                    and only clip duration will be used for weighting.
    :param cln_clipl: names of the clip duration columns. If omitted, they will be generated from TV OTS columns names
                      by replacing 'ots' with 'clipl' and removing placement type and media type info.
    :param tv_pattern: specific regex pattern containing media type and placement type, e.g. 'sponsor_nat'
    :return:
    """

    if tv_pattern is None:
        tv_pattern = '[a-z0-9]+'

    if len([i for i in cln_tv if (bool(re.search(tv_pattern, i)))]) == 0:
        return pd.Series(0, index=ds.index)
    
    if cln_clipl is None:
        cln_clipl = [re.sub(r'((_|^)*ots_)', '_clipl_', x) for x in cln_tv]
        cln_clipl = [re.sub(r'^_', '', x) for x in cln_clipl]

    def weight_series(s: str):

        s_pattern = re.sub(r'(^(.+_)*ots_)|(_rolik|_sponsor|_nat|_local)|_' + tv_pattern, '', s)
        clipl_pattern = re.sub(r'((_|^)*ots_)', '_clipl_', s)
        clipl_pattern = re.sub(r'^_', '', clipl_pattern)

        if cln_aff is not None:
            s_aff = [i for i in cln_aff if (bool(re.search(s_pattern, i)))
                    or (bool(re.search('m55plus', s)) and bool(re.search('m55_64', i)))
                    or (bool(re.search('f55plus', s)) and bool(re.search('f55_64', i)))
                    or (bool(re.search('m14_24', s)) and bool(re.search('m18_24', i)))
                    or (bool(re.search('f14_24', s)) and bool(re.search('f18_24', i)))]
        else:
            s_aff = []

        s_clipl = [i for i in cln_clipl if (s_pattern in i) and i == clipl_pattern]

        if len(s_clipl) == 0 or (cln_aff is not None and len(s_aff) == 0):
            rtrn = pd.Series(0, index=ds.index)
        elif len(s_clipl) != 0 and cln_aff is None:
            rtrn = ds[s].array * (ds[s_clipl[0]].array ** params_dict['clipl_curve'])
            rtrn = pd.Series(rtrn, index=ds.index)
        else:
            rtrn = ds[s].array * (ds[s_aff[0]].array ** params_dict['power_coef']) * (ds[s_clipl[0]].array ** params_dict['clipl_curve'])
            rtrn = pd.Series(rtrn, index=ds.index)

        return rtrn

    rtrn = []
    for col in cln_tv:
        if re.search(tv_pattern, col):
            tmp = weight_series(col)
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
    :param tv_pattern: specific regex pattern containing media type and placement type, e.g. 'sponsor_nat'
    :return:
    """

    if len([i for i in cln_tv if (bool(re.search(tv_pattern, i)))]) == 0:
        return pd.Series(0, index=ds.index)

    def affinity_series(s: str):

        s_pattern = re.sub(r'(^(.+_)*ots_)|(_rolik|_sponsor|_nat|_local)|_' + tv_pattern, '', s)

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

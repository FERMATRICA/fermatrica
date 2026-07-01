import logging
import re

import numpy as np
import pandas as pd

from fermatrica_utils import groupby_eff
from fermatrica.model.model import Model


def pre_reporting(model: "Model"
                  , ds: pd.DataFrame
                  , model_rep
                  ):
    """
    Prepare dataset for reporting / option run.
    Back-fill future-period variables for listed=4 rows.

    - Zero out media variables for listed=4 (future)
    - Forward-fill price and trend variables per bs_key
    """

    # zero out media for future
    zeroes_pattern = (r'^(brnd_|reg_|e_com|pos|pr_costs)')
    non_zeroes_pattern = r'(opt_media_ssn)'

    cln = [x for x in ds.columns.tolist()
           if re.search(zeroes_pattern, x) and not re.search(non_zeroes_pattern, x)]

    mask = (ds['superbrand'] == model.conf.target_superbrand) & (ds['listed'].isin([0, 4]))

    for cl in cln:
        ds.loc[mask, cl] = 0

    #

    cols = [col for col in ds.columns if re.search(r'opt_media_ssn', col)]

    for col in cols:
        ds.loc[mask & (ds[col].isna()), col] = 0.0

    #

    listed_4_len = ds[ds['listed'] == 4]['date'].unique().shape[0]
    avg_period = 4

    fwd_mask = (ds['listed'] > 1)

    # price_index
    ds.loc[ds['listed'] == 4, 'price_index'] = np.nan
    rtrn = groupby_eff(ds, ['bs_key'], ['price_index'], fwd_mask, sort=False)
    rtrn = rtrn.tail(listed_4_len + avg_period)
    rtrn['price_index'] = rtrn.groupby('bs_key')['price_index'].transform(
        lambda x: x.fillna(x.mean()))
    rtrn = rtrn.groupby('bs_key').tail(listed_4_len)['price_index']
    ds.loc[rtrn.index, 'price_index'] = rtrn

    # kpi_coef
    ds.loc[ds['listed'] == 4, 'kpi_coef'] = np.nan
    rtrn = groupby_eff(ds, ['bs_key'], ['kpi_coef'], fwd_mask, sort=False)
    ds.loc[fwd_mask, 'kpi_coef'] = rtrn['kpi_coef'].ffill()

    return model, ds, model_rep

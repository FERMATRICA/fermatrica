# file to store adhoc transformations functions (if required)

import fermatrica as fm
import pandas as pd


def adhoc_bht_mult(ds: pd.DataFrame
                    , params_subset: pd.DataFrame
                    , index_vars=None):

    params_dict = fm.basics.basics.params_to_dict(params_subset)

    ds['brnd_tv_ots_clip_softmax_bht'] = ds['brnd_tv_ots_clip_softmax'].array * \
        ((1 - ds['aba_lag_softmax'].array) * (1 - params_dict['offset']) + params_dict['offset'])

    ds['brnd_ots_ooh_softmax_bht'] = ds['brnd_ots_ooh_softmax'].array * \
        ((1 - ds['aba_lag_softmax'].array) * (1 - params_dict['offset']) + params_dict['offset'])

    pass


def adhoc_unparam_fin(ds: pd.DataFrame
                      , params_subset: pd.DataFrame
                      , index_vars=None):

    ds['price_ret_volat'] = ds.loc[:, 'price_ret_mar'] - ds.loc[:, 'price_ret']
    ds['price_ret_volat_rel'] = ds.loc[:, 'price_ret_volat'] / ds.loc[:, 'price_ret_mar']

    pass

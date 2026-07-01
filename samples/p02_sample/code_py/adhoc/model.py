import pandas as pd


def adhoc_unparam_fin(ds: pd.DataFrame
                      , params_subset: pd.DataFrame
                      , index_vars=None):

    ds['price_ret_volat'] = ds.loc[:, 'price_ret_mar'] - ds.loc[:, 'price_ret']
    ds['price_ret_volat_rel'] = ds.loc[:, 'price_ret_volat'] / ds.loc[:, 'price_ret_mar']

    pass


def adhoc_create_di2(ds: pd.DataFrame
                     , params_subset: pd.DataFrame
                     , index_vars=None):

    ds['di_soph_ssn_median_sm_2'] = ds.loc[:, 'di_soph_ssn_median_sm']

    pass

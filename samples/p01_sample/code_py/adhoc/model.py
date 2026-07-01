# file to store adhoc transformations functions (if required)

import pandas as pd

def adhoc_unparam_fin(ds: pd.DataFrame
                      , params_subset: pd.DataFrame
                      , index_vars=None):

    ds['price_ret_volat'] = ds.loc[:, 'price_ret_mar'] - ds.loc[:, 'price_ret']
    ds['price_ret_volat_rel'] = ds.loc[:, 'price_ret_volat'] / ds.loc[:, 'price_ret_mar']

    pass

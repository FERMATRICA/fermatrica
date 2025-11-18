import yaml
import os

import pandas as pd
import numpy as np
import pytest
from contextlib import nullcontext

import fermatrica.model.model_conf as mc
from fermatrica.basics.basics import FermatricaError

import tests.basics.basics as bs

yaml.add_constructor('!pd.RangeIndex', bs.construct_pandas_index)
yaml.add_constructor('!pd.Series', bs.construct_pandas_series)
yaml.add_constructor('!pd.DataFrame', bs.construct_pandas_dataframe)
yaml.add_constructor('!np.array', bs.construct_numpy_array)


def loc_parametrize_class(file_name):

    params = bs.parametrize_class('model_conf/'+file_name)

    for par in params:
        if par[0]['ds'] is not None:
            par[0]['ds'] = data(par[0]['ds'])

        if os.path.basename(os.getcwd()) != 'test_model':
            print(os.path.basename(__file__))
            par[0]['path'] = 'tests\\test_model\\' + par[0]['path']

    return params

def data(name):

    # get data from yaml if conditions are met (
    pth = os.path.join(os.path.abspath(os.path.join(__file__, "..")),
                       'test_cases/model_conf/data.yaml')
    with open(pth, mode='rt', encoding='utf-8') as yaml_io:
        yaml_data = yaml.full_load(yaml_io)
        item = yaml_data[0]
        df = pd.DataFrame(item[name])

    # type convertion
    df['date'] = pd.to_datetime(df['date'])

    return df

class TestMCInit:

    @pytest.mark.parametrize("input_params,expected_result,exp_err", loc_parametrize_class('mc.__init__.yaml'))
    def test_mc_init(self, input_params, expected_result, exp_err, if_debug):

        # get data from yaml if conditions are met (containing "testing case")
        # calc result
        with exp_err:
            if os.path.basename(os.getcwd()) == 'tests':
                os.chdir("test_model")
            result = mc.ModelConf(**input_params)

            if if_debug:
                for key, value in expected_result.items():
                    fun = value[0]
                    exp_res = value[1]
                    res = getattr(result, key)
                    try:
                        fun(res, exp_res)
                    except:
                        breakpoint()
            else:
                for key, value in expected_result.items():
                    fun = value[0]
                    exp_res = value[1]
                    res = getattr(result, key)
                    fun(res, exp_res)
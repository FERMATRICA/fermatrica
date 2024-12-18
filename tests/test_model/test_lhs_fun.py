import yaml
import os

import pandas as pd
import numpy as np
import pytest
from contextlib import nullcontext

import fermatrica.model.lhs_fun as lhs
from fermatrica.basics.basics import FermatricaError

from tests.basics.basics import construct_pandas_index, construct_pandas_series, construct_pandas_dataframe, construct_numpy_array


yaml.add_constructor('!pd.RangeIndex', construct_pandas_index)
yaml.add_constructor('!pd.Series', construct_pandas_series)
yaml.add_constructor('!pd.DataFrame', construct_pandas_dataframe)
yaml.add_constructor('!np.array', construct_numpy_array)


def parametrize_params(file_name):
    pth = os.path.join(os.path.abspath(os.path.join(__file__, "..")),
                           'test_cases/lhs_fun', file_name)
    params = []
    with open(pth, mode='rt', encoding='utf-8') as yaml_io:
        yaml_data = yaml.full_load(yaml_io)
        for item in yaml_data:
            if 'testing case' in item['description']:
                # get data
                input_params = item['params']

                expected_result = item['expected_result']

                fun_dict = {
                    pd.DataFrame: lambda result, expected_result: pd.testing.assert_frame_equal(result, expected_result),
                    pd.Series: lambda result, expected_result: pd.testing.assert_series_equal(result, expected_result),
                    np.ndarray: lambda result, expected_result: np.testing.assert_array_equal(result, expected_result)}
                fun = fun_dict[type(item['expected_result'])]

                if item['exp_err']:
                    exp_err = pytest.raises(eval(item['exp_err']))
                else:
                    exp_err = nullcontext()

                params.append((input_params, expected_result, fun, exp_err))
    return params


class TestMarketResizeFlex:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params('market_resize_flex.yaml'))
    def test_market_resize_flex(self, input_params, expected_result, fun, exp_err, if_debug):

        # get data from yaml if conditions are met (containing "testing case")
        # calc result
        with exp_err:
            result = lhs.market_resize_flex(**input_params)

            if if_debug:
                try:
                    fun(np.round(result,5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result,5), expected_result)


class TestLogExp:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params('log_exp.yaml'))
    def test_log_exp(self, input_params, expected_result, fun, exp_err, if_debug):

        # get data from yaml if conditions are met (containing "testing case")
        # calc result
        with exp_err:
            result = lhs.log_exp(**input_params)

            if if_debug:
                try:
                    fun(np.round(result,5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result,5), expected_result)
import yaml
import os

import pandas as pd
import numpy as np
import pytest
from contextlib import nullcontext

from fermatrica.basics.basics import FermatricaError

def construct_pandas_index(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> pd.Index:
    return pd.RangeIndex(**loader.construct_mapping(node))

def construct_pandas_series(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> pd.Series:
    return pd.Series(**loader.construct_mapping(node, deep=True))

def construct_pandas_dataframe(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> pd.DataFrame:
    return pd.DataFrame(**loader.construct_mapping(node, deep=True))

def construct_numpy_array(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> np.array:
    return np.array(**loader.construct_mapping(node, deep=True))

def construct_numpy_ndarray(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> np.ndarray:
    return np.ndarray(**loader.construct_mapping(node, deep=True))

def assertNone(a):
    assert a is None

def assertEq(a, b):
    assert a == b

def parametrize_fun(file_name):
    pth = os.path.join(os.path.abspath(os.path.join(__file__, "..")),
                       'test_model/test_cases', file_name)
    params = []
    with open(pth, mode='rt', encoding='utf-8') as yaml_io:
        yaml_data = yaml.full_load(yaml_io)
        for item in yaml_data:
            if 'testing case' in item['description']:
                # get data
                input_params = item['params']

                expected_result = item['expected_result']

                fun_dict = {
                    pd.DataFrame: lambda result, expected_result: pd.testing.assert_frame_equal(result,
                                                                                                expected_result),
                    pd.Series: lambda result, expected_result: pd.testing.assert_series_equal(result, expected_result),
                    np.ndarray: lambda result, expected_result: np.testing.assert_array_equal(result, expected_result)}

                if type(expected_result) in fun_dict.keys():
                    fun = fun_dict[type(expected_result)]
                elif expected_result is None:
                    fun = lambda result, expected_result: assertNone(result)
                else:
                    fun = lambda result, expected_result: assertEq(result, expected_result)

                if item['exp_err']:
                    exp_err = pytest.raises(eval(item['exp_err']))
                else:
                    exp_err = nullcontext()

                params.append((input_params, expected_result, fun, exp_err))
    return params


def parametrize_class(file_name):
    pth = os.path.join(os.path.abspath(os.path.join(__file__, "..", '..')),
                           'test_model/test_cases', file_name)
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

                if expected_result is not None:
                    for key, value in expected_result.items():
                        if type(value) in fun_dict.keys():
                            expected_result[key] = [fun_dict[type(value)], value]
                        elif value is None:
                            expected_result[key] = [lambda result, expected_result: assertNone(result), value]
                        else:
                            expected_result[key] = [lambda result, expected_result: assertEq(result, expected_result), value]

                if item['exp_err']:
                    exp_err = pytest.raises(eval(item['exp_err']))
                else:
                    exp_err = nullcontext()

                params.append((input_params,
                               expected_result,
                               exp_err))
    return params
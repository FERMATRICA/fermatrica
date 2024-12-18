import yaml
import os

import pandas as pd
import numpy as np
import pytest
from contextlib import nullcontext

import fermatrica.model.transform_fun as tr
from fermatrica.basics.basics import FermatricaError

from tests.basics.basics import construct_pandas_index, construct_pandas_series, construct_pandas_dataframe, construct_numpy_array


yaml.add_constructor('!pd.RangeIndex', construct_pandas_index)
yaml.add_constructor('!pd.Series', construct_pandas_series)
yaml.add_constructor('!pd.DataFrame', construct_pandas_dataframe)
yaml.add_constructor('!np.array', construct_numpy_array)

def parametrize_params(file_name):
    pth = os.path.join(os.path.abspath(os.path.join(__file__, "..")),
                           'test_cases/transform_fun', file_name)
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

class TestRecursiveFilter:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params('recursive_filter.yaml'))
    def test_recursive_filter(self, input_params, expected_result, fun, exp_err, if_debug):

        # get data from yaml if conditions are met (containing "testing case")
        # calc result
        with exp_err:
            result = tr.recursive_filter(**input_params)

            if if_debug:
                try:
                    fun(np.round(result,5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result,5), expected_result)


class TestWeibullMultiResponse:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params('weibull_multi_response.yaml'))
    def test_weibull_multi_response(self, input_params, expected_result, fun, exp_err, if_debug):

        # get data from yaml if conditions are met (containing "testing case")
        # calc result
        with exp_err:
            result = tr.weibull_multi_response(**input_params)

            if if_debug:
                try:
                    fun(np.round(result,5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result,5), expected_result)


class TestSumInvExpDist:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params('sum_inv_exp_dist.yaml'))
    def test_sum_inv_exp_dist(self, input_params, expected_result, fun, exp_err, if_debug):

        # get data from yaml if conditions are met (containing "testing case")
        # calc result
        with exp_err:
            result = tr.sum_inv_exp_dist(**input_params)

            if if_debug:
                try:
                    fun(np.round(result,5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result,5), expected_result)


class TestScaleClassicMedian:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params('scale_classic_median.yaml'))
    def test_scale_classic_median(self, input_params, expected_result, fun, exp_err, if_debug):

        # get data from yaml if conditions are met (containing "testing case")
        # calc result
        with exp_err:
            result = tr.scale_classic_median(**input_params)

            if if_debug:
                try:
                    fun(np.round(result,5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result,5), expected_result)


class TestAdbudg:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params('adbudg.yaml'))
    def test_adbudg(self, input_params, expected_result, fun, exp_err, if_debug):

        # get data from yaml if conditions are met (containing "testing case")
        # calc result
        with exp_err:
            result = tr.adbudg(**input_params)

            if if_debug:
                try:
                    fun(np.round(result,5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result,5), expected_result)


class TestLogistic:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params('logistic.yaml'))
    def test_logistic(self, input_params, expected_result, fun, exp_err, if_debug):

        # get data from yaml if conditions are met (containing "testing case")
        # calc result
        with exp_err:
            result = tr.logistic(**input_params)

            if if_debug:
                try:
                    fun(np.round(result,5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result,5), expected_result)


class TestSoftmax:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params('softmax.yaml'))
    def test_softmax(self, input_params, expected_result, fun, exp_err, if_debug):

        # get data from yaml if conditions are met (containing "testing case")
        # calc result
        with exp_err:
            result = tr.softmax(**input_params)

            if if_debug:
                try:
                    fun(np.round(result,5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result,5), expected_result)


class TestGompertz:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params('gompertz.yaml'))
    def test_gompertz(self, input_params, expected_result, fun, exp_err, if_debug):

        # get data from yaml if conditions are met (containing "testing case")
        # calc result
        with exp_err:
            result = tr.gompertz(**input_params)

            if if_debug:
                try:
                    fun(np.round(result,5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result,5), expected_result)


class TestGaussian:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params('gaussian.yaml'))
    def test_gaussian(self, input_params, expected_result, fun, exp_err, if_debug):

        # get data from yaml if conditions are met (containing "testing case")
        # calc result
        with exp_err:
            result = tr.gaussian(**input_params)

            if if_debug:
                try:
                    fun(np.round(result,5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result,5), expected_result)


class TestLogGaussian:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params('log_gaussian.yaml'))
    def test_log_gaussian(self, input_params, expected_result, fun, exp_err, if_debug):

        # get data from yaml if conditions are met (containing "testing case")
        # calc result
        with exp_err:
            result = tr.log_gaussian(**input_params)

            if if_debug:
                try:
                    fun(np.round(result,5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result,5), expected_result)


class TestLorentzian:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params('lorentzian.yaml'))
    def test_lorentzian(self, input_params, expected_result, fun, exp_err, if_debug):

        # get data from yaml if conditions are met (containing "testing case")
        # calc result
        with exp_err:
            result = tr.lorentzian(**input_params)

            if if_debug:
                try:
                    fun(np.round(result,5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result,5), expected_result)


class TestTvCliplAffinity:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params('tv_clipl_affinity.yaml'))
    def test_tv_clipl_affinity(self, input_params, expected_result, fun, exp_err, if_debug):

        # get data from yaml if conditions are met (containing "testing case")
        # calc result
        with exp_err:
            result = tr.tv_clipl_affinity(**input_params)

            if if_debug:
                try:
                    fun(np.round(result,5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result,5), expected_result)


class TestTvAffinity:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params('tv_affinity.yaml'))
    def test_tv_affinity(self, input_params, expected_result, fun, exp_err, if_debug):

        # get data from yaml if conditions are met (containing "testing case")
        # calc result
        with exp_err:
            result = tr.tv_affinity(**input_params)

            if if_debug:
                try:
                    fun(np.round(result,5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result,5), expected_result)


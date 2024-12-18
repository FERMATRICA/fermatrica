import yaml
import os

import pandas as pd
import numpy as np
import pytest
from contextlib import nullcontext

import fermatrica.model.transform as tr
from fermatrica.basics.basics import FermatricaError

from tests.basics.basics import construct_pandas_index


yaml.add_constructor('!pd.RangeIndex', construct_pandas_index)


def parametrize_params(file_name):
    pth = os.path.join(os.path.abspath(os.path.join(__file__, "..")),
                           'test_cases/transform', file_name)
    params = []
    with open(pth, mode='rt', encoding='utf-8') as yaml_io:
        yaml_data = yaml.full_load(yaml_io)
        for item in yaml_data:
            if 'testing case' in item['description']:
                #get data
                data = eval(item['data'])()

                # get params
                params_subset = pd.DataFrame(item['params_subset'])

                # check result
                type_dict = {
                    'dataframe': pd.DataFrame,
                    'series': pd.Series,
                    'array': np.array}
                type = type_dict[item['expected_type']]
                expected_result = type(**item['expected_result'])
                # type conversion

                if item['expected_type'] == 'dataframe' and 'date' in expected_result.columns:
                    expected_result['date'] = pd.to_datetime(expected_result['date'])

                # assertion function
                fun_dict = {'dataframe': lambda result, expected_result: pd.testing.assert_frame_equal(result, expected_result),
                            'series': lambda result, expected_result: pd.testing.assert_series_equal(result, expected_result),
                            'array': lambda result, expected_result: np.testing.assert_array_equal(result, expected_result)}
                fun = fun_dict[item['expected_type']]

                # waiting for an error
                if item['exp_err']:
                    exp_err = pytest.raises(eval(item['exp_err']))
                else:
                    exp_err = nullcontext()

                params.append((data, params_subset, item['index_vars'], expected_result, fun, exp_err))
    return params


def sample_data():

    # get data from yaml if conditions are met (
    pth = os.path.join(os.path.abspath(os.path.join(__file__, "..")),
                       'test_cases/transform/data.yaml')
    with open(pth, mode='rt', encoding='utf-8') as yaml_io:
        yaml_data = yaml.full_load(yaml_io)
        item = yaml_data[0]
        df = pd.DataFrame(item['sample_data'])

    # type convertion
    df['date'] = pd.to_datetime(df['date'])

    return df


def duppl_data():

    # get data from yaml if conditions are met (
    pth = os.path.join(os.path.abspath(os.path.join(__file__, "..")),
                       'test_cases/transform/data.yaml')
    with open(pth, mode='rt', encoding='utf-8') as yaml_io:
        yaml_data = yaml.full_load(yaml_io)
        item = yaml_data[1]
        df = pd.DataFrame(item['duppl_data'])

    # type convertion
    df['date'] = pd.to_datetime(df['date'])

    return df

def mfreq_data():

    # get data from yaml if conditions are met (
    pth = os.path.join(os.path.abspath(os.path.join(__file__, "..")),
                       'test_cases/transform/data.yaml')
    with open(pth, mode='rt', encoding='utf-8') as yaml_io:
        yaml_data = yaml.full_load(yaml_io)
        item = yaml_data[2]
        df = pd.DataFrame(item['mfreq_data'])

    # type convertion
    df['date'] = pd.to_datetime(df['date'])

    return df


class TestVarAggregate:

    @pytest.mark.parametrize("data,params_subset,index_vars,expected_result,fun,exp_err", parametrize_params('_var_aggregate.yaml'))
    def test_var_aggregate(self, data, params_subset, index_vars, expected_result, fun, exp_err, if_debug):

        # get data from yaml if conditions are met (containing "testing case")
        # calc result
        with exp_err:
            result = tr._var_aggregate(data, 'var', params_subset, index_vars)
            result = result[['superbrand', 'date', 'var']].sort_values(['superbrand', 'date'], ignore_index=True)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)


class TestCheckType:

    @pytest.mark.parametrize("data,params_subset,index_vars,expected_result,fun,exp_err", parametrize_params('_check_type.yaml'))
    def test_check_type(self, data, params_subset, index_vars, expected_result, fun, exp_err, if_debug):

        # get data from yaml if conditions are met (containing "testing case")
        # calc result
        with exp_err:
            tr._check_type(data, 'var')

            result = data['var']

            if if_debug:
                try:
                    fun(result, expected_result)
                    assert result.dtype.name in ['float64', 'float']
                except:
                    breakpoint()
            else:
                fun(result, expected_result)
                assert result.dtype.name in ['float64', 'float']

    @pytest.mark.parametrize("data,params_subset,index_vars,expected_result,fun,exp_err", parametrize_params('_check_type.yaml'))
    def test_check_type_fail(self, data, params_subset, index_vars, expected_result, fun, exp_err, if_debug):

        # get data from yaml if conditions are met (containing "testing case")
        # calc result
        with pytest.raises(FermatricaError):
            tr._check_type(data, 'vars')

            result = data['var']

            if if_debug:
                try:
                    fun(result, expected_result)
                    assert result.dtype.name in ['float64', 'float']
                except:
                    breakpoint()
            else:
                fun(result, expected_result)
                assert result.dtype.name in ['float64', 'float']


class TestLag:

    @pytest.mark.parametrize("data,params_subset,index_vars,expected_result,fun,exp_err", parametrize_params('lag.yaml'))
    def test_lag(self, data, params_subset, index_vars, expected_result, fun, exp_err, if_debug):

        with exp_err:
            result = tr.lag(data, 'var', params_subset, index_vars)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)


class TestAdstock:

    @pytest.mark.parametrize("data,params_subset,index_vars,expected_result,fun,exp_err", parametrize_params('adstock.yaml'))
    def test_adstock(self, data, params_subset, index_vars, expected_result, fun, exp_err, if_debug):

        with exp_err:
            result = tr.adstock(data, 'var', params_subset, index_vars)

            if if_debug:
                try:
                    fun(np.round(result, 2), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result, 2), expected_result)

class TestAdstockp:

    @pytest.mark.parametrize("data,params_subset,index_vars,expected_result,fun,exp_err", parametrize_params('adstockp.yaml'))
    def test_adstockp(self, data, params_subset, index_vars, expected_result, fun, exp_err, if_debug):

        with exp_err:
            result = tr.adstockp(data, 'var', params_subset, index_vars)

            if if_debug:
                try:
                    fun(np.round(result, 2), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result, 2), expected_result)


class TestAdstockpd:

    @pytest.mark.parametrize("data,params_subset,index_vars,expected_result,fun,exp_err", parametrize_params('adstockpd.yaml'))
    def test_adstockpd(self, data, params_subset, index_vars, expected_result, fun, exp_err, if_debug):

        with exp_err:
            result = tr.adstockpd(data, 'var', params_subset, index_vars)

            if if_debug:
                try:
                    fun(np.round(result, 2), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result, 2), expected_result)


class TestDWBL:

    @pytest.mark.parametrize("data,params_subset,index_vars,expected_result,fun,exp_err", parametrize_params('dwbl.yaml'))
    def test_dwbl(self, data, params_subset, index_vars, expected_result, fun, exp_err, if_debug):

        with exp_err:
            result = tr.dwbl(data, 'var', params_subset, index_vars)

            if if_debug:
                try:
                    fun(np.round(result, 5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result, 5), expected_result)


class TestDWBLP:

    @pytest.mark.parametrize("data,params_subset,index_vars,expected_result,fun,exp_err", parametrize_params('dwblp.yaml'))
    def test_dwblp(self, data, params_subset, index_vars, expected_result, fun, exp_err, if_debug):

        with exp_err:
            result = tr.dwblp(data, 'var', params_subset, index_vars)

            if if_debug:
                try:
                    fun(np.round(result, 5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result, 5), expected_result)


class TestMAR:

    @pytest.mark.parametrize("data,params_subset,index_vars,expected_result,fun,exp_err", parametrize_params('mar.yaml'))
    def test_mar(self, data, params_subset, index_vars, expected_result, fun, exp_err, if_debug):

        with exp_err:
            result = tr.mar(data, 'var', params_subset, index_vars)

            if if_debug:
                try:
                    fun(np.round(result, 5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result, 5), expected_result)


class TestINFL:

    @pytest.mark.parametrize("data,params_subset,index_vars,expected_result,fun,exp_err", parametrize_params('infl.yaml'))
    def test_infl(self, data, params_subset, index_vars, expected_result, fun, exp_err, if_debug):

        with exp_err:
            result = tr.infl(data, 'var', params_subset, index_vars)

            if if_debug:
                try:
                    fun(np.round(result, 5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result, 5), expected_result)


class TestAGE:

    @pytest.mark.parametrize("data,params_subset,index_vars,expected_result,fun,exp_err", parametrize_params('age.yaml'))
    def test_age(self, data, params_subset, index_vars, expected_result, fun, exp_err, if_debug):

        with exp_err:
            result = tr.age(data, 'var', params_subset, index_vars)

            if if_debug:
                try:
                    fun(np.round(result, 5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result, 5), expected_result)

    # def test_age_basic(self, duppl_data):
    #     params_subset = pd.DataFrame({})
    #     result = tr.age(duppl_data, 'var', params_subset)
    #     expected_result = pd.Series([0.,  0.5,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,
    #                                  10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20.,
    #                                  21., 22., 23.], name='var_tmp')
    #     pd.testing.assert_series_equal(result, expected_result)


class TestScale:

    @pytest.mark.parametrize("data,params_subset,index_vars,expected_result,fun,exp_err", parametrize_params('scale.yaml'))
    def test_scale(self, data, params_subset, index_vars, expected_result, fun, exp_err, if_debug):

        with exp_err:
            result = tr.scale(data, 'var', params_subset, index_vars)

            if if_debug:
                try:
                    fun(np.round(result, 5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result, 5), expected_result)


class TestShare:

    @pytest.mark.parametrize("data,params_subset,index_vars,expected_result,fun,exp_err", parametrize_params('share.yaml'))
    def test_share(self, data, params_subset, index_vars, expected_result, fun, exp_err, if_debug):

        with exp_err:
            result = tr.share(data, 'var', params_subset, index_vars)

            if if_debug:
                try:
                    fun(np.round(result, 5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result, 5), expected_result)


class TestExpm1:

    @pytest.mark.parametrize("data,params_subset,index_vars,expected_result,fun,exp_err", parametrize_params('expm1.yaml'))
    def test_expm1(self, data, params_subset, index_vars, expected_result, fun, exp_err, if_debug):

        with exp_err:
            result = tr.expm1(data, 'var', params_subset, index_vars)

            if if_debug:
                try:
                    fun(np.round(result, 5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result, 5), expected_result)


class TestExpm1scaled:

    @pytest.mark.parametrize("data,params_subset,index_vars,expected_result,fun,exp_err", parametrize_params('expm1scaled.yaml'))
    def test_expm1scaled(self, data, params_subset, index_vars, expected_result, fun, exp_err, if_debug):

        with exp_err:
            result = tr.expm1scaled(data, 'var', params_subset, index_vars)

            # for read null
            expected_result = expected_result.astype('float64')

            if if_debug:
                try:
                    fun(np.round(result, 5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result, 5), expected_result)


class TestSoftmaxfull:

    @pytest.mark.parametrize("data,params_subset,index_vars,expected_result,fun,exp_err", parametrize_params('softmaxfull.yaml'))
    def test_softmaxfull(self, data, params_subset, index_vars, expected_result, fun, exp_err, if_debug):

        with exp_err:
            result = tr.softmaxfull(data, 'var', params_subset, index_vars)

            # for read null
            expected_result = expected_result.astype('float64')

            if if_debug:
                try:
                    fun(np.round(result, 8), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result, 8), expected_result)


class TestSoftmax:

    @pytest.mark.parametrize("data,params_subset,index_vars,expected_result,fun,exp_err", parametrize_params('softmax.yaml'))
    def test_softmax(self, data, params_subset, index_vars, expected_result, fun, exp_err, if_debug):

        with exp_err:
            result = tr.softmax(data, 'var', params_subset, index_vars)

            # for read null
            #expected_result = expected_result.astype('float64')

            if if_debug:
                try:
                    fun(np.round(result, 8), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result, 8), expected_result)


class TestLogistic:

    @pytest.mark.parametrize("data,params_subset,index_vars,expected_result,fun,exp_err", parametrize_params('logistic.yaml'))
    def test_logistic(self, data, params_subset, index_vars, expected_result, fun, exp_err, if_debug):

        with exp_err:
            result = tr.logistic(data, 'var', params_subset, index_vars)

            # for read null
            expected_result = expected_result.astype('float64')

            if if_debug:
                try:
                    fun(np.round(result, 5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result, 5), expected_result)


class TestGompertz:

    @pytest.mark.parametrize("data,params_subset,index_vars,expected_result,fun,exp_err", parametrize_params('gompertz.yaml'))
    def test_gompertz(self, data, params_subset, index_vars, expected_result, fun, exp_err, if_debug):

        with exp_err:
            result = tr.gompertz(data, 'var', params_subset, index_vars)

            if if_debug:
                try:
                    fun(np.round(result, 5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result, 5), expected_result)

class TestAdbudg:

    @pytest.mark.parametrize("data,params_subset,index_vars,expected_result,fun,exp_err", parametrize_params('adbudg.yaml'))
    def test_adbudg(self, data, params_subset, index_vars, expected_result, fun, exp_err, if_debug):

        with exp_err:
            result = tr.adbudg(data, 'var', params_subset, index_vars)

            if if_debug:
                try:
                    fun(np.round(result, 5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result, 5), expected_result)

class TestPower:

    @pytest.mark.parametrize("data,params_subset,index_vars,expected_result,fun,exp_err", parametrize_params('power.yaml'))
    def test_power(self, data, params_subset, index_vars, expected_result, fun, exp_err, if_debug):

        with exp_err:
            result = tr.power(data, 'var', params_subset, index_vars)

            # for read null and inf
            expected_result = expected_result.astype('float64')

            if if_debug:
                try:
                    fun(np.round(result, 5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result, 5), expected_result)


class TestGaussian:

    @pytest.mark.parametrize("data,params_subset,index_vars,expected_result,fun,exp_err", parametrize_params('gaussian.yaml'))
    def test_gaussian(self, data, params_subset, index_vars, expected_result, fun, exp_err, if_debug):

        with exp_err:
            result = tr.gaussian(data, 'var', params_subset, index_vars)

            # for read null and inf and e-05
            expected_result = expected_result.astype('float64')

            if if_debug:
                try:
                    fun(np.round(result, 5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result, 5), expected_result)


class TestGaussianzero:

    @pytest.mark.parametrize("data,params_subset,index_vars,expected_result,fun,exp_err", parametrize_params('gaussianzero.yaml'))
    def test_gaussianzero(self, data, params_subset, index_vars, expected_result, fun, exp_err, if_debug):

        with exp_err:
            result = tr.gaussianzero(data, 'var', params_subset, index_vars)

            if if_debug:
                try:
                    fun(np.round(result, 5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result, 5), expected_result)


class TestLorentzian:

    @pytest.mark.parametrize("data,params_subset,index_vars,expected_result,fun,exp_err", parametrize_params('lorentzian.yaml'))
    def test_lorentzian(self, data, params_subset, index_vars, expected_result, fun, exp_err, if_debug):

        with exp_err:
            result = tr.lorentzian(data, 'var', params_subset, index_vars)

            if if_debug:
                try:
                    fun(np.round(result, 5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result, 5), expected_result)


class TestLoggaussian:

    @pytest.mark.parametrize("data,params_subset,index_vars,expected_result,fun,exp_err", parametrize_params('loggaussian.yaml'))
    def test_loggaussian(self, data, params_subset, index_vars, expected_result, fun, exp_err, if_debug):

        with exp_err:
            result = tr.loggaussian(data, 'var', params_subset, index_vars)

            # for read null and inf and e-05
            expected_result = expected_result.astype('float64')

            if if_debug:
                try:
                    fun(np.round(result, 5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result, 5), expected_result)


class TestArl:

    @pytest.mark.parametrize("data,params_subset,index_vars,expected_result,fun,exp_err", parametrize_params('arl.yaml'))
    def test_arl(self, data, params_subset, index_vars, expected_result, fun, exp_err, if_debug):

        with exp_err:
            result = tr.arl(data, 'var', params_subset, index_vars)

            # for read null and inf and e-05
            expected_result = expected_result.astype('float64')

            if if_debug:
                try:
                    fun(np.round(result, 5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result, 5), expected_result)


class TestArlp:

    @pytest.mark.parametrize("data,params_subset,index_vars,expected_result,fun,exp_err", parametrize_params('arlp.yaml'))
    def test_arlp(self, data, params_subset, index_vars, expected_result, fun, exp_err, if_debug):

        with exp_err:
            result = tr.arlp(data, 'var', params_subset, index_vars)

            # for read null and inf and e-05
            expected_result = expected_result.astype('float64')

            if if_debug:
                try:
                    fun(np.round(result, 5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result, 5), expected_result)

class TestPrice:

    @pytest.mark.parametrize("data,params_subset,index_vars,expected_result,fun,exp_err", parametrize_params('price.yaml'))
    def test_price(self, data, params_subset, index_vars, expected_result, fun, exp_err, if_debug):

        with exp_err:
            tr.price(data, 'var', params_subset)
            result = data.iloc[:, 10:]

            if if_debug:
                try:
                    fun(np.round(result,5), expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(result,5), expected_result)

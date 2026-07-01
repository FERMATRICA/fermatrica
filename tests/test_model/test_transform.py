import yaml
import os

import pandas as pd
import numpy as np
import pytest
from contextlib import nullcontext

import fermatrica.model.transform as tr
from fermatrica.basics.basics import FermatricaError

from tests.basics.basics import construct_pandas_index
from tests.conftest import if_debug


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
                    'array': np.array,
                }
                type = type_dict[item['expected_type']]

                if isinstance(item['expected_result'], list):
                    expected_result = item['expected_result']
                else:
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


def follow_me_data():

    pth = os.path.join(os.path.abspath(os.path.join(__file__, "..")),
                       'test_cases/transform/data.yaml')
    with open(pth, mode='rt', encoding='utf-8') as yaml_io:
        yaml_data = yaml.full_load(yaml_io)
        item = yaml_data[3]
        df = pd.DataFrame(item['follow_me_data'])

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


class TestAdstockd:

    @pytest.mark.parametrize("data,params_subset,index_vars,expected_result,fun,exp_err", parametrize_params('adstockd.yaml'))
    def test_adstockd(self, data, params_subset, index_vars, expected_result, fun, exp_err, if_debug):

        with exp_err:
            result = tr.adstockd(data, 'var', params_subset, index_vars)

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


class TestAge:

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


class TestArld:

    @pytest.mark.parametrize("data,params_subset,index_vars,expected_result,fun,exp_err", parametrize_params('arld.yaml'))
    def test_arld(self, data, params_subset, index_vars, expected_result, fun, exp_err, if_debug):

        with exp_err:
            result = tr.arld(data, 'var', params_subset, index_vars)

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


class TestFollowMe:

    @staticmethod
    def _make_row(variable, fun, arg, value, if_active=1, lower=np.nan, upper=np.nan):
        return {
            'variable': variable, 'fun': fun, 'arg': arg, 'value': value,
            'if_active': if_active, 'lower': lower, 'upper': upper,
            'index_vars': np.nan, 'index_free_var': np.nan,
            'formula': np.nan, 'set_start': 0,
        }

    def test_basic_value_copy(self, if_debug):
        params = pd.DataFrame([
            self._make_row('var1', 'adstock', 'a', 0.5, lower=0.0, upper=1.0),
            self._make_row('var1', 'follow_me', 'which', 'a'),
            self._make_row('var1', 'follow_me', 'who',   'var2'),
            self._make_row('var1', 'follow_me', 'what',  'adstock'),
            self._make_row('var2', 'adstock', 'a', 0.1, lower=0.0, upper=1.0),
        ])
        ds = follow_me_data()

        tr._transform_all(ds, params, model_obj=None, set_start=False)

        result = params.loc[
            (params['variable'] == 'var2') & (params['fun'] == 'adstock') & (params['arg'] == 'a'),
            'value'
        ].values[0]

        if if_debug:
            try:
                assert result == 0.5
            except:
                breakpoint()
        else:
            assert result == 0.5

    def test_multiple_followers(self, if_debug):
        params = pd.DataFrame([
            self._make_row('var1', 'adstock', 'a', 0.5),
            self._make_row('var1', 'follow_me', 'which', 'a'),
            self._make_row('var1', 'follow_me', 'who',   'var2, var3'),
            self._make_row('var1', 'follow_me', 'what',  'adstock'),
            self._make_row('var2', 'adstock', 'a', 0.1),
            self._make_row('var3', 'adstock', 'a', 0.2),
        ])
        ds = follow_me_data()

        tr._transform_all(ds, params, model_obj=None, set_start=False)

        result_var2 = params.loc[
            (params['variable'] == 'var2') & (params['fun'] == 'adstock') & (params['arg'] == 'a'),
            'value'
        ].values[0]
        result_var3 = params.loc[
            (params['variable'] == 'var3') & (params['fun'] == 'adstock') & (params['arg'] == 'a'),
            'value'
        ].values[0]

        if if_debug:
            try:
                assert result_var2 == 0.5
                assert result_var3 == 0.5
            except:
                breakpoint()
        else:
            assert result_var2 == 0.5
            assert result_var3 == 0.5

    def test_multiple_params(self, if_debug):
        # lag has two natural params: n and cval; follow_me must copy both
        params = pd.DataFrame([
            self._make_row('var1', 'lag', 'n',    2.0),
            self._make_row('var1', 'lag', 'cval', 0.5),
            self._make_row('var1', 'follow_me', 'which', 'n, cval'),
            self._make_row('var1', 'follow_me', 'who',   'var2'),
            self._make_row('var1', 'follow_me', 'what',  'lag'),
            self._make_row('var2', 'lag', 'n',    1.0),
            self._make_row('var2', 'lag', 'cval', 0.0),
        ])
        ds = follow_me_data()

        tr._transform_all(ds, params, model_obj=None, set_start=False)

        result_n = params.loc[
            (params['variable'] == 'var2') & (params['fun'] == 'lag') & (params['arg'] == 'n'),
            'value'
        ].values[0]
        result_cval = params.loc[
            (params['variable'] == 'var2') & (params['fun'] == 'lag') & (params['arg'] == 'cval'),
            'value'
        ].values[0]

        if if_debug:
            try:
                assert result_n == 2.0
                assert result_cval == 0.5
            except:
                breakpoint()
        else:
            assert result_n == 2.0
            assert result_cval == 0.5

    def test_clamp_to_lower(self, if_debug):
        # leader a=0.1 is below follower lower=0.3, so result must be clamped to 0.3
        params = pd.DataFrame([
            self._make_row('var1', 'adstock', 'a', 0.1),
            self._make_row('var1', 'follow_me', 'which', 'a'),
            self._make_row('var1', 'follow_me', 'who',   'var2'),
            self._make_row('var1', 'follow_me', 'what',  'adstock'),
            self._make_row('var2', 'adstock', 'a', 0.5, lower=0.3, upper=1.0),
        ])
        ds = follow_me_data()

        tr._transform_all(ds, params, model_obj=None, set_start=False)

        result = params.loc[
            (params['variable'] == 'var2') & (params['fun'] == 'adstock') & (params['arg'] == 'a'),
            'value'
        ].values[0]

        if if_debug:
            try:
                assert result == 0.3
            except:
                breakpoint()
        else:
            assert result == 0.3

    def test_clamp_to_upper(self, if_debug):
        # leader a=0.9 is above follower upper=0.5, so result must be clamped to 0.5
        params = pd.DataFrame([
            self._make_row('var1', 'adstock', 'a', 0.9),
            self._make_row('var1', 'follow_me', 'which', 'a'),
            self._make_row('var1', 'follow_me', 'who',   'var2'),
            self._make_row('var1', 'follow_me', 'what',  'adstock'),
            self._make_row('var2', 'adstock', 'a', 0.3, lower=0.0, upper=0.5),
        ])
        ds = follow_me_data()

        tr._transform_all(ds, params, model_obj=None, set_start=False)

        result = params.loc[
            (params['variable'] == 'var2') & (params['fun'] == 'adstock') & (params['arg'] == 'a'),
            'value'
        ].values[0]

        if if_debug:
            try:
                assert result == 0.5
            except:
                breakpoint()
        else:
            assert result == 0.5

    def test_inactive_skipped(self, if_debug):
        # if_active=0 on follow_me rows: follower value must remain unchanged
        params = pd.DataFrame([
            self._make_row('var1', 'adstock', 'a', 0.5),
            self._make_row('var1', 'follow_me', 'which', 'a',      if_active=0),
            self._make_row('var1', 'follow_me', 'who',   'var2',   if_active=0),
            self._make_row('var1', 'follow_me', 'what',  'adstock', if_active=0),
            self._make_row('var2', 'adstock', 'a', 0.1),
        ])
        ds = follow_me_data()

        tr._transform_all(ds, params, model_obj=None, set_start=False)

        result = params.loc[
            (params['variable'] == 'var2') & (params['fun'] == 'adstock') & (params['arg'] == 'a'),
            'value'
        ].values[0]

        if if_debug:
            try:
                assert result == 0.1
            except:
                breakpoint()
        else:
            assert result == 0.1

    def test_leader_transform_not_in_params_raises(self, if_debug):
        # follow_me points what=adbudg, but var1 has no adbudg entry in params
        # -> followed DataFrame is empty -> IndexError on .values[0]
        params = pd.DataFrame([
            self._make_row('var1', 'adstock', 'a', 0.5),
            self._make_row('var1', 'follow_me', 'which', 'a'),
            self._make_row('var1', 'follow_me', 'who',   'var2'),
            self._make_row('var1', 'follow_me', 'what',  'adbudg'),
            self._make_row('var2', 'adstock', 'a', 0.1),
        ])
        ds = follow_me_data()

        with pytest.raises(IndexError):
            tr._transform_all(ds, params, model_obj=None, set_start=False)

    def test_follower_not_in_params_is_noop(self, if_debug):
        # follow_me points who=var_ghost which has no adstock entry in params
        # -> follow_mask is all-False -> silent no-op, no error raised
        params = pd.DataFrame([
            self._make_row('var1', 'adstock', 'a', 0.5),
            self._make_row('var1', 'follow_me', 'which', 'a'),
            self._make_row('var1', 'follow_me', 'who',   'var_ghost'),
            self._make_row('var1', 'follow_me', 'what',  'adstock'),
            self._make_row('var2', 'adstock', 'a', 0.1),
        ])
        ds = follow_me_data()

        tr._transform_all(ds, params, model_obj=None, set_start=False)

        result_var2 = params.loc[
            (params['variable'] == 'var2') & (params['fun'] == 'adstock') & (params['arg'] == 'a'),
            'value'
        ].values[0]

        if if_debug:
            try:
                assert result_var2 == 0.1
            except:
                breakpoint()
        else:
            assert result_var2 == 0.1

    def test_follower_column_missing_from_ds_raises(self, if_debug):
        # var2 is in params (with adstock) but its column is absent from ds
        # -> follow_me updates params fine, then adstock for var2 raises FermatricaError
        params = pd.DataFrame([
            self._make_row('var1', 'adstock', 'a', 0.5),
            self._make_row('var1', 'follow_me', 'which', 'a'),
            self._make_row('var1', 'follow_me', 'who',   'var2'),
            self._make_row('var1', 'follow_me', 'what',  'adstock'),
            self._make_row('var2', 'adstock', 'a', 0.1),
        ])
        ds = follow_me_data()[['date', 'var1']]  # var2 and var3 intentionally dropped

        with pytest.raises(FermatricaError):
            tr._transform_all(ds, params, model_obj=None, set_start=False)

    def test_set_start_returns_updated_params(self, if_debug):
        # when set_start=True, _transform_all returns (model_obj, params);
        # the returned params must contain the follow_me copy;
        # the original params object must remain unmodified (local rebind isolation)
        params = pd.DataFrame([
            self._make_row('var1', 'adstock', 'a', 0.5, lower=0.0, upper=1.0),
            self._make_row('var1', 'follow_me', 'which', 'a'),
            self._make_row('var1', 'follow_me', 'who',   'var2'),
            self._make_row('var1', 'follow_me', 'what',  'adstock'),
            self._make_row('var2', 'adstock', 'a', 0.1, lower=0.0, upper=1.0),
        ])
        ds = follow_me_data()

        _, returned_params = tr._transform_all(ds, params, model_obj=None, set_start=True)

        result_returned = returned_params.loc[
            (returned_params['variable'] == 'var2') & (returned_params['fun'] == 'adstock') & (returned_params['arg'] == 'a'),
            'value'
        ].values[0]
        result_original = params.loc[
            (params['variable'] == 'var2') & (params['fun'] == 'adstock') & (params['arg'] == 'a'),
            'value'
        ].values[0]

        if if_debug:
            try:
                assert result_returned == 0.5
                assert result_original == 0.1
            except:
                breakpoint()
        else:
            assert result_returned == 0.5
            assert result_original == 0.1

    def test_multiple_follow_me_entries(self, if_debug):
        # var1 drives var2.adstock and var3 drives var2.lag independently;
        # both follow_me entries must be processed without interfering with each other
        params = pd.DataFrame([
            self._make_row('var1', 'adstock', 'a', 0.5),
            self._make_row('var1', 'follow_me', 'which', 'a'),
            self._make_row('var1', 'follow_me', 'who',   'var2'),
            self._make_row('var1', 'follow_me', 'what',  'adstock'),
            self._make_row('var2', 'lag', 'n', 1.0),
            self._make_row('var2', 'adstock', 'a', 0.1),
            self._make_row('var3', 'follow_me', 'which', 'n'),
            self._make_row('var3', 'follow_me', 'who',   'var2'),
            self._make_row('var3', 'follow_me', 'what',  'lag'),
            self._make_row('var3', 'lag', 'n', 3.0),
        ])
        ds = follow_me_data()

        tr._transform_all(ds, params, model_obj=None, set_start=False)

        result_adstock = params.loc[
            (params['variable'] == 'var2') & (params['fun'] == 'adstock') & (params['arg'] == 'a'),
            'value'
        ].values[0]
        result_lag = params.loc[
            (params['variable'] == 'var2') & (params['fun'] == 'lag') & (params['arg'] == 'n'),
            'value'
        ].values[0]

        if if_debug:
            try:
                assert result_adstock == 0.5
                assert result_lag == 3.0
            except:
                breakpoint()
        else:
            assert result_adstock == 0.5
            assert result_lag == 3.0

    def test_follower_rows_before_leader_rows(self, if_debug):
        # follower var2 rows appear first in params, leader var1 rows (including follow_me) come after;
        # var_fun_uniques preserves this order so var2.adstock is iterated before var1.follow_me;
        # follow_me reads the leader value directly from params (not from iteration state),
        # so the copy must succeed regardless of row order
        params = pd.DataFrame([
            self._make_row('var2', 'adstock', 'a', 0.1, lower=0.0, upper=1.0),
            self._make_row('var1', 'adstock', 'a', 0.5, lower=0.0, upper=1.0),
            self._make_row('var1', 'follow_me', 'which', 'a'),
            self._make_row('var1', 'follow_me', 'who',   'var2'),
            self._make_row('var1', 'follow_me', 'what',  'adstock'),
        ])
        ds = follow_me_data()

        tr._transform_all(ds, params, model_obj=None, set_start=False)

        result = params.loc[
            (params['variable'] == 'var2') & (params['fun'] == 'adstock') & (params['arg'] == 'a'),
            'value'
        ].values[0]

        if if_debug:
            try:
                assert result == 0.5
            except:
                breakpoint()
        else:
            assert result == 0.5

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


class TestMCChecks:
    """Focused tests for each ModelConf check method."""

    _MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'test_cases', 'model_conf', 'models')

    _PARAMS_COLS = ['variable', 'fun', 'arg', 'type', 'fixed', 'if_active',
                    'lower', 'upper', 'value', 'index_vars', 'index_aggr',
                    'index_free_var', 'set_start']

    @classmethod
    def _write_fixture(cls, filename, rows):
        path = os.path.join(cls._MODEL_DIR, filename)
        params_df = pd.DataFrame(rows, columns=cls._PARAMS_COLS)
        setup_df = pd.DataFrame({'key': [], 'value': []})
        rhs_df = pd.DataFrame({'token': ['feature_x'], 'fixed': [1], 'if_active': [1]})
        with pd.ExcelWriter(path, engine='openpyxl') as w:
            params_df.to_excel(w, sheet_name='params', index=False)
            rhs_df.to_excel(w, sheet_name='RHS', index=False)
            setup_df.to_excel(w, sheet_name='setup', index=False)
        return path

    @classmethod
    def setup_class(cls):
        cls._path_if_active_unstable = cls._write_fixture(
            'check_if_active_unstable.xlsx', [
                dict(variable='feature_a', fun='adstock', arg='coef', type='float64',
                     fixed=0, if_active=1, lower=None, upper=None, value=0.5,
                     index_vars=None, index_aggr=None, index_free_var=None, set_start=0),
                dict(variable='feature_a', fun='adstock', arg='lag', type='float64',
                     fixed=0, if_active=0, lower=None, upper=None, value=2.0,
                     index_vars=None, index_aggr=None, index_free_var=None, set_start=0),
            ])

        cls._path_set_start_fixed = cls._write_fixture(
            'check_set_start_fixed.xlsx', [
                dict(variable='feature_a', fun='adstock', arg='coef', type='float64',
                     fixed=0, if_active=1, lower=-1.0, upper=1.0, value=0.5,
                     index_vars=None, index_aggr=None, index_free_var=None, set_start=1),
            ])

        cls._path_opt_bounds_lower = cls._write_fixture(
            'check_opt_bounds_lower.xlsx', [
                dict(variable='feature_a', fun='adstock', arg='coef', type='float64',
                     fixed=0, if_active=1, lower=2.0, upper=5.0, value=0.5,
                     index_vars=None, index_aggr=None, index_free_var=None, set_start=0),
            ])

        cls._path_follow_me = cls._write_fixture(
            'check_follow_me.xlsx', [
                dict(variable='feature_a', fun='follow_me', arg='who', type='str',
                     fixed=1, if_active=1, lower=None, upper=None, value='feature_b',
                     index_vars=None, index_aggr=None, index_free_var=None, set_start=0),
                dict(variable='feature_a', fun='follow_me', arg='what', type='str',
                     fixed=1, if_active=1, lower=None, upper=None, value='adstock',
                     index_vars=None, index_aggr=None, index_free_var=None, set_start=0),
                dict(variable='feature_a', fun='follow_me', arg='which', type='str',
                     fixed=1, if_active=1, lower=None, upper=None, value='coef',
                     index_vars=None, index_aggr=None, index_free_var=None, set_start=0),
                dict(variable='feature_b', fun='adstock', arg='coef', type='float64',
                     fixed=0, if_active=1, lower=-1.0, upper=1.0, value=0.5,
                     index_vars=None, index_aggr=None, index_free_var=None, set_start=0),
            ])

    # ── _check_if_if_active_stable ────────────────────────────────────────────

    def test_if_active_unstable_raises_error(self):
        with pytest.raises(FermatricaError):
            mc.ModelConf(path=self._path_if_active_unstable)

    # ── _check_if_set_start_fixed ─────────────────────────────────────────────

    def test_set_start_unfixed_gets_fixed(self):
        result = mc.ModelConf(path=self._path_set_start_fixed, if_stable=False)
        mask = result.params['set_start'] == 1
        assert (result.params.loc[mask, 'fixed'] == 1).all()

    # ── _check_if_following_fixed ─────────────────────────────────────────────

    def test_follow_me_fixes_followed_param(self):
        result = mc.ModelConf(path=self._path_follow_me, if_stable=False)
        mask = ((result.params['variable'] == 'feature_b') &
                (result.params['fun'] == 'adstock') &
                (result.params['arg'] == 'coef'))
        assert result.params.loc[mask, 'fixed'].values[0] == 1

    # ── _check_opt_bounds ─────────────────────────────────────────────────────

    def test_opt_bounds_lower_exceeds_value(self):
        result = mc.ModelConf(path=self._path_opt_bounds_lower, if_stable=False)
        mask = (result.params['if_active'] == 1) & (result.params['fixed'] == 0)
        assert float(result.params.loc[mask, 'lower'].values[0]) == pytest.approx(0.5)


class TestMCFactsDefaults:

    _MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'test_cases', 'model_conf', 'models')

    _PARAMS_COLS = ['variable', 'fun', 'arg', 'type', 'fixed', 'if_active',
                    'lower', 'upper', 'value', 'index_vars', 'index_aggr',
                    'index_free_var', 'set_start']

    @classmethod
    def _write(cls, filename, setup_rows):
        path = os.path.join(cls._MODEL_DIR, filename)
        params_df = pd.DataFrame([], columns=cls._PARAMS_COLS)
        setup_df = pd.DataFrame(setup_rows, columns=['key', 'value'])
        rhs_df = pd.DataFrame({'token': ['feature_x'], 'fixed': [1], 'if_active': [1]})
        with pd.ExcelWriter(path, engine='openpyxl') as w:
            params_df.to_excel(w, sheet_name='params', index=False)
            rhs_df.to_excel(w, sheet_name='RHS', index=False)
            setup_df.to_excel(w, sheet_name='setup', index=False)
        return path

    def test_defaults(self):
        path = self._write('kpi_facts_defaults.xlsx', [])
        conf = mc.ModelConf(path=path, if_stable=False)
        assert conf.kpi_name == 'sales'
        assert conf.currency == 'RUB'
        assert conf.if_value_applicable is True
        assert conf.display_units == 'packs'

    def test_keys_read_and_parsed(self):
        path = self._write('kpi_facts_set.xlsx', [
            ('kpi_name', 'brand_awareness'),
            ('currency', 'EUR'),
            ('if_value_applicable', 'False'),
            ('display_units', 'pct'),
        ])
        conf = mc.ModelConf(path=path, if_stable=False)
        assert conf.kpi_name == 'brand_awareness'
        assert conf.currency == 'EUR'
        assert conf.if_value_applicable is False
        assert conf.display_units == 'pct'
import yaml
import os
import pandas as pd
import numpy as np
import pytest
from contextlib import nullcontext

import fermatrica.evaluation.metrics as metrics
from fermatrica.basics.basics import FermatricaError

from tests.conftest import if_debug


def parametrize_params(file_name):
    pth = os.path.join(os.path.abspath(os.path.join(__file__, '..')),
                           'test_cases/metrics', file_name)
    params = []
    with open(pth, mode='rt', encoding='utf-8') as yaml_io:
        yaml_data = yaml.full_load(yaml_io)
        for item in yaml_data:
            if 'testing case' in item['description']:
                obs = item['obs']
                pred = item['pred']
                group = item.get('group', None)
                reduce = item.get('reduce', None)
                if_mean = item.get('if_mean', True)
                adj_val = item.get('adj_val', None)
                expected_result = item['expected_result']
                
                if item['exp_err']:
                    exp_err = pytest.raises(eval(item['exp_err']))
                else:
                    exp_err = nullcontext()
                
                expected_type = item.get('expected_type', None)
                params.append((obs, pred, group, reduce, if_mean, adj_val, expected_result, exp_err, expected_type))
    return params


class TestRSquared:
    
    @pytest.mark.parametrize('obs,pred,group,reduce,if_mean,adj_val,expected_result,exp_err,expected_type', 
                           parametrize_params('r_squared.yaml'))
    def test_r_squared(self, obs, pred, group, reduce, if_mean, adj_val, expected_result, exp_err, expected_type, if_debug):
        
        with exp_err:
            result = metrics.r_squared(obs, pred)

            if str(expected_result) == 'nan':
                assert str(result) == 'nan'
            else:
                assert result == pytest.approx(expected_result, abs=1e-8)


class TestRMSE:
    
    @pytest.mark.parametrize('obs,pred,group,reduce,if_mean,adj_val,expected_result,exp_err,expected_type', 
                           parametrize_params('rmse.yaml'))
    def test_rmse(self, obs, pred, group, reduce, if_mean, adj_val, expected_result, exp_err, expected_type, if_debug):
        
        with exp_err:
            result = metrics.rmse(obs, pred)

            if str(expected_result) == 'nan':
                assert str(result) == 'nan'
            else:
                assert result == pytest.approx(expected_result, abs=1e-8)


class TestMSE:
    
    @pytest.mark.parametrize('obs,pred,group,reduce,if_mean,adj_val,expected_result,exp_err,expected_type', 
                           parametrize_params('mse.yaml'))
    def test_mse(self, obs, pred, group, reduce, if_mean, adj_val, expected_result, exp_err, expected_type, if_debug):
        
        with exp_err:
            result = metrics.mse(obs, pred)

            if str(expected_result) == 'nan':
                assert str(result) == 'nan'
            else:
                assert result == pytest.approx(expected_result, abs=1e-8)


class TestMAPE:
    
    @pytest.mark.parametrize('obs,pred,group,reduce,if_mean,adj_val,expected_result,exp_err,expected_type', 
                           parametrize_params('mape.yaml'))
    def test_mape(self, obs, pred, group, reduce, if_mean, adj_val, expected_result, exp_err, expected_type, if_debug):
        
        with exp_err:
            result = metrics.mape(obs, pred)

            if str(expected_result) == 'nan':
                assert str(result) == 'nan'
            else:
                assert result == pytest.approx(expected_result, abs=1e-8)


class TestMAPEF:
    
    @pytest.mark.parametrize('obs,pred,group,reduce,if_mean,adj_val,expected_result,exp_err,expected_type', 
                           parametrize_params('mapef.yaml'))
    def test_mapef(self, obs, pred, group, reduce, if_mean, adj_val, expected_result, exp_err, expected_type, if_debug):
        
        with exp_err:
            result = metrics.mapef(obs, pred)

            if str(expected_result) == 'nan':
                assert str(result) == 'nan'
            else:
                assert result == pytest.approx(expected_result, abs=1e-8)


class TestMAPEAdj:
    
    @pytest.mark.parametrize('obs,pred,group,reduce,if_mean,adj_val,expected_result,exp_err,expected_type', 
                           parametrize_params('mape_adj.yaml'))
    def test_mape_adj(self, obs, pred, group, reduce, if_mean, adj_val, expected_result, exp_err, expected_type, if_debug):
        
        with exp_err:
            result = metrics.mape_adj(obs, pred, adj_val=adj_val)

            if str(expected_result) == 'nan':
                assert str(result) == 'nan'
            else:
                assert result == pytest.approx(expected_result, abs=1e-8)


class TestSMAPE:
    
    @pytest.mark.parametrize('obs,pred,group,reduce,if_mean,adj_val,expected_result,exp_err,expected_type', 
                           parametrize_params('smape.yaml'))
    def test_smape(self, obs, pred, group, reduce, if_mean, adj_val, expected_result, exp_err, expected_type, if_debug):
        
        with exp_err:
            result = metrics.smape(obs, pred)

            if str(expected_result) == 'nan':
                assert str(result) == 'nan'
            else:
                assert result == pytest.approx(expected_result, abs=1e-8)


class TestEOM:
    
    @pytest.mark.parametrize('obs,pred,group,reduce,if_mean,adj_val,expected_result,exp_err,expected_type', 
                           parametrize_params('eom.yaml'))
    def test_eom(self, obs, pred, group, reduce, if_mean, adj_val, expected_result, exp_err, expected_type, if_debug):
        
        with exp_err:
            result = metrics.eom(obs, pred)

            if str(expected_result) == 'nan':
                assert str(result) == 'nan'
            else:
                assert result == pytest.approx(expected_result, abs=1e-8)


class TestZTest:
    
    @pytest.mark.parametrize('obs,pred,group,reduce,if_mean,adj_val,expected_result,exp_err,expected_type', 
                           parametrize_params('ztest.yaml'))
    def test_ztest(self, obs, pred, group, reduce, if_mean, adj_val, expected_result, exp_err, expected_type, if_debug):
        
        with exp_err:
            z, p = metrics.ztest(obs, pred)
            if str(expected_result['expected_z']) == 'nan':
                assert str(z) == 'nan'
            else:
                assert z == pytest.approx(expected_result['expected_z'], abs=1e-8)
            if str(expected_result['expected_p']) == 'nan':
                assert str(p) == 'nan'
            else:
                assert p == pytest.approx(expected_result['expected_p'], abs=1e-8)


class TestRSquaredGroup:
    
    @pytest.mark.parametrize('obs,pred,group,reduce,if_mean,adj_val,expected_result,exp_err,expected_type', 
                           parametrize_params('r_squared_group.yaml'))
    def test_r_squared_group(self, obs, pred, group, reduce, if_mean, adj_val, expected_result, exp_err, expected_type, if_debug):
        
        with exp_err:
            if group is None:
                pytest.skip('Group parameter is required for group functions')
            
            if reduce is not None:
                result = metrics.r_squared_group(obs, pred, group, reduce=reduce, if_mean=if_mean)
            else:
                result = metrics.r_squared_group(obs, pred, group, if_mean=if_mean)
            
            if expected_type == 'series':
                assert isinstance(result, pd.Series)
            else:
                if str(expected_result) == 'nan':
                    assert str(result) == 'nan'
                else:
                    assert result == pytest.approx(expected_result, abs=1e-8)


class TestRMSEGroup:
    
    @pytest.mark.parametrize('obs,pred,group,reduce,if_mean,adj_val,expected_result,exp_err,expected_type', 
                           parametrize_params('rmse_group.yaml'))
    def test_rmse_group(self, obs, pred, group, reduce, if_mean, adj_val, expected_result, exp_err, expected_type, if_debug):
        
        with exp_err:
            if group is None:
                pytest.skip('Group parameter is required for group functions')
            
            if reduce is not None:
                result = metrics.rmse_group(obs, pred, group, reduce=reduce, if_mean=if_mean)
            else:
                result = metrics.rmse_group(obs, pred, group, if_mean=if_mean)
            
            if expected_type == 'series':
                assert isinstance(result, pd.Series)
            else:
                if str(expected_result) == 'nan':
                    assert str(result) == 'nan'
                else:
                    assert result == pytest.approx(expected_result, abs=1e-8)


class TestMAPEFGroup:
    
    @pytest.mark.parametrize('obs,pred,group,reduce,if_mean,adj_val,expected_result,exp_err,expected_type', 
                           parametrize_params('mapef_group.yaml'))
    def test_mapef_group(self, obs, pred, group, reduce, if_mean, adj_val, expected_result, exp_err, expected_type, if_debug):
        
        with exp_err:
            if group is None:
                pytest.skip('Group parameter is required for group functions')
            
            if reduce is not None:
                result = metrics.mapef_group(obs, pred, group, reduce=reduce, if_mean=if_mean)
            else:
                result = metrics.mapef_group(obs, pred, group, if_mean=if_mean)
            
            if expected_type == 'series':
                assert isinstance(result, pd.Series)
            else:
                if str(expected_result) == 'nan':
                    assert str(result) == 'nan'
                else:
                    assert result == pytest.approx(expected_result, abs=1e-8)


class TestMAPEAdjGroup:
    
    @pytest.mark.parametrize('obs,pred,group,reduce,if_mean,adj_val,expected_result,exp_err,expected_type', 
                           parametrize_params('mape_adj_group.yaml'))
    def test_mape_adj_group(self, obs, pred, group, reduce, if_mean, adj_val, expected_result, exp_err, expected_type, if_debug):
        
        with exp_err:
            if group is None:
                pytest.skip('Group parameter is required for group functions')
            
            if reduce is not None:
                result = metrics.mape_adj_group(obs, pred, group, adj_val=adj_val, reduce=reduce, if_mean=if_mean)
            else:
                result = metrics.mape_adj_group(obs, pred, group, adj_val=adj_val, if_mean=if_mean)
            
            if expected_type == 'series':
                assert isinstance(result, pd.Series)
            else:
                if str(expected_result) == 'nan':
                    assert str(result) == 'nan'
                else:
                    assert result == pytest.approx(expected_result, abs=1e-8)


class TestEOMGroup:
    
    @pytest.mark.parametrize('obs,pred,group,reduce,if_mean,adj_val,expected_result,exp_err,expected_type', 
                           parametrize_params('eom_group.yaml'))
    def test_eom_group(self, obs, pred, group, reduce, if_mean, adj_val, expected_result, exp_err, expected_type, if_debug):
        
        with exp_err:
            if group is None:
                pytest.skip('Group parameter is required for group functions')
            
            if reduce is not None:
                result = metrics.eom_group(obs, pred, group, reduce=reduce, if_mean=if_mean)
            else:
                result = metrics.eom_group(obs, pred, group, if_mean=if_mean)
            
            if expected_type == 'series':
                assert isinstance(result, pd.Series)
            else:
                if str(expected_result) == 'nan':
                    assert str(result) == 'nan'
                else:
                    assert result == pytest.approx(expected_result, abs=1e-8)


class TestZTestGroup:
    
    @pytest.mark.parametrize('obs,pred,group,reduce,if_mean,adj_val,expected_result,exp_err,expected_type', 
                           parametrize_params('ztest_group.yaml'))
    def test_ztest_group(self, obs, pred, group, reduce, if_mean, adj_val, expected_result, exp_err, expected_type, if_debug):
        
        with exp_err:
            if group is None:
                pytest.skip('Group parameter is required for group functions')
            
            if reduce is not None:
                result = metrics.ztest_group(obs, pred, group, reduce=reduce, if_mean=if_mean)
            else:
                result = metrics.ztest_group(obs, pred, group, if_mean=if_mean)
            
            if expected_type == 'series':
                assert isinstance(result, pd.Series)
            else:
                if hasattr(result, 'apply'):
                    result = result.apply(lambda tup: tup[0]).mean()
                if str(expected_result) == 'nan':
                    assert str(result) == 'nan'
                else:
                    assert result == pytest.approx(expected_result, abs=1e-8)

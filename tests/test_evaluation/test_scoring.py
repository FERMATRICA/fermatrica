import yaml
import os
import pandas as pd
import numpy as np
import pytest
from contextlib import nullcontext

import fermatrica.evaluation.scoring as scoring
from fermatrica.basics.basics import FermatricaError

from tests.conftest import if_debug


def parametrize_params(file_name):
    pth = os.path.join(os.path.abspath(os.path.join(__file__, '..')),
                           'test_cases/scoring', file_name)
    params = []
    with open(pth, mode='rt', encoding='utf-8') as yaml_io:
        yaml_data = yaml.full_load(yaml_io)
        for item in yaml_data:
            if 'testing case' in item['description']:
                # get params
                x = item.get('x', None)
                if_invert = item.get('if_invert', 0)
                width = item.get('width', 1)
                obs = item.get('obs', None)
                pred = item.get('pred', None)
                
                # expected result
                expected_result = item['expected_result']
                
                # waiting for an error
                if item['exp_err']:
                    exp_err = pytest.raises(eval(item['exp_err']))
                else:
                    exp_err = nullcontext()
                
                params.append((x, if_invert, width, obs, pred, expected_result, exp_err))
    return params


class TestBellCurve:
    
    @pytest.mark.parametrize('x,if_invert,width,obs,pred,expected_result,exp_err', 
                           parametrize_params('bell_curve.yaml'))
    def test_bell_curve(self, x, if_invert, width, obs, pred, expected_result, exp_err, if_debug):
        
        with exp_err:
            result = scoring.bell_curve(x, if_invert=if_invert, width=width)
            
            if str(expected_result) == 'nan':
                assert str(result) == 'nan'
            else:
                assert np.allclose(result, expected_result, atol=1e-8)


class TestBellCurveInvert:
    
    @pytest.mark.parametrize('x,if_invert,width,obs,pred,expected_result,exp_err', 
                           parametrize_params('bell_curve_invert.yaml'))
    def test_bell_curve_invert(self, x, if_invert, width, obs, pred, expected_result, exp_err, if_debug):
        
        with exp_err:
            result = scoring.bell_curve(x, if_invert=if_invert, width=width)

            if str(expected_result) == 'nan':
                assert str(result) == 'nan'
            else:
                assert np.allclose(result, expected_result, atol=1e-8)


class TestRSquaredScoring:
    
    @pytest.mark.parametrize('x,if_invert,width,obs,pred,expected_result,exp_err', 
                           parametrize_params('r_squared_scoring.yaml'))
    def test_r_squared_scoring(self, x, if_invert, width, obs, pred, expected_result, exp_err, if_debug):
        
        with exp_err:
            result = scoring.r_squared(obs, pred)

            if str(expected_result) == 'nan':
                assert str(result) == 'nan'
            else:
                assert result == pytest.approx(expected_result, abs=1e-8)


class TestRMSEScoring:
    
    @pytest.mark.parametrize('x,if_invert,width,obs,pred,expected_result,exp_err', 
                           parametrize_params('rmse_scoring.yaml'))
    def test_rmse_scoring(self, x, if_invert, width, obs, pred, expected_result, exp_err, if_debug):
        
        with exp_err:
            result = scoring.rmse(obs, pred)

            if str(expected_result) == 'nan':
                assert str(result) == 'nan'
            else:
                assert result == pytest.approx(expected_result, abs=1e-8)


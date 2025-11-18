import os
import yaml
import pandas as pd
import numpy as np
import pytest
from datetime import datetime

import fermatrica as fm
import fermatrica_rep as fmr

from tests.conftest import if_debug


def load_test_cases():
    """Load test cases from YAML file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, "test_cases", "integration_p00_sample.yaml")
    
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def get_test_case(case_name):
    """Get specific test case by name"""
    test_cases = load_test_cases()
    for case in test_cases['test_cases']:
        if case['name'] == case_name:
            return case
    return None


class TestP00SampleIntegration:
    """
    Integration test based on p00_sample example.
    
    Test verifies the complete data processing cycle:
    1. Data loading
    2. Model creation
    3. Data transformation
    4. Predictions
    5. Quality metrics verification
    """
    
    def test_data_loading(self, prepared_data):
        """Test data loading and preparation"""
        test_case = get_test_case("data_preparation")
        expected = test_case['expected']
        
        assert not prepared_data.empty, "Data not loaded"
        
        # Check required columns
        for col in expected['required_columns']:
            assert col in prepared_data.columns, f"Column {col} missing"
        
        # Check that we have data for all periods
        assert len(prepared_data[prepared_data['listed'] == 2]) > 0, "No training data"
        assert len(prepared_data[prepared_data['listed'] == 3]) > 0, "No test data"
    
    def test_model_creation(self, base_model):
        """Test model creation"""
        test_case = get_test_case("basic_model_creation")
        expected = test_case['expected']
        
        assert base_model is not None, "Model not created"
        assert hasattr(base_model, 'conf'), "Model missing configuration"
        assert hasattr(base_model.conf, 'Y_var'), "Target variable not set"
    
    def test_model_transformation(self, trained_model):
        """Test model transformation"""
        test_case = get_test_case("model_transformation")
        expected = test_case['expected']
        
        model, pred = trained_model
        assert model is not None, "Model transformation failed"
    
    def test_predictions(self, trained_model, prepared_data):
        """Test prediction generation"""
        test_case = get_test_case("predictions_generation")
        expected = test_case['expected']
        
        model, pred = trained_model
        
        assert pred is not None, "Predictions not generated"
        assert len(pred) == len(prepared_data.loc[prepared_data['listed'] == 2])\
            , "Number of train predictions doesn't match train data size"

        dt_pred = fm.predict_ext(model, prepared_data)
        assert len(dt_pred) == len(prepared_data)\
            , "Number of all predictions doesn't match data size"

        mask = dt_pred['listed'] == 4
        assert (len(dt_pred[mask & (dt_pred['predicted'].notna())]) ==
                len(prepared_data[mask]))\
            , "Some future predictions are None"

        # Check that predictions are not all zero
        assert not np.all(pred == 0), "All predictions are zero"
    
    def test_model_metrics(self, test_metrics):
        """Test model quality metrics"""
        test_case = get_test_case("metrics_calculation")
        expected = test_case['expected']
        
        # Check that metrics are calculated
        assert 'rmse' in test_metrics, "RMSE not calculated"
        assert 'score' in test_metrics, "Overall score not calculated"
        
        # Check reasonableness of values using YAML expectations
        assert test_metrics['rmse'] > 0, f"RMSE should be positive: {test_metrics['rmse']}"
        assert not np.isinf(test_metrics['score']), f"Score should not be infinite: {test_metrics['score']}"
        
        # Check metrics if available using YAML expectations
        if 'r_squared_train' in test_metrics:
            r_squared_expected = expected['r_squared_train']
            r_squared_calculated = test_metrics['r_squared_train']
            assert test_metrics['r_squared_train'] == pytest.approx(r_squared_expected, abs=1e-8), \
                f"R² for training {r_squared_calculated} not equal to expected {r_squared_expected}"

        if 'mape_train' in test_metrics:
            mape_expected = expected['mape_train']
            mape_calculated = test_metrics['mape_train']
            assert test_metrics['mape_train'] == pytest.approx(mape_expected, abs=1e-8), \
                f"MAPE for training {mape_calculated} not equal to expected {mape_expected}"
        
        if 'mape_test' in test_metrics:
            mape_expected = expected['mape_test']
            mape_calculated = test_metrics['mape_test']
            assert test_metrics['mape_test'] == pytest.approx(mape_expected, abs=1e-8), \
                f"MAPE for testing {mape_calculated} not equal to expected {mape_expected}"
    
    def test_model_consistency(self, trained_model, prepared_data):
        """Test model consistency"""
        test_case = get_test_case("model_consistency")
        expected = test_case['expected']
        
        model, pred = trained_model
        
        # Check extended predictions
        dt_pred = fm.predict_ext(model, prepared_data)
        assert dt_pred is not None, "Extended predictions failed"
        assert not dt_pred.empty, "Extended predictions are empty"
    
    def test_model_objects(self, trained_model):
        """Test model objects"""
        model, pred = trained_model
        
        assert hasattr(model, 'obj'), "Model missing obj attribute"
        assert hasattr(model.obj, 'models'), "obj missing models attribute"
        assert 'main' in model.obj.models, "Main model missing"
        
        # Check that main model has summary
        main_model = model.obj.models['main']
        assert hasattr(main_model, 'summary'), "Main model missing summary"
    
    # @pytest.mark.skipif(if_debug, reason="Skipped in debug mode")
    def test_performance_metrics(self, test_metrics):
        """Test performance and specific metric values"""
        # Check specific metric values if available
        if 'r_squared_train' in test_metrics:
            r_squared_train = test_metrics['r_squared_train']
            mape_train = test_metrics['mape_train']
            
            # Check reasonableness of values
            assert 0 <= r_squared_train <= 1, f"R² for training out of reasonable range: {r_squared_train}"
            assert mape_train < 1000, f"MAPE for training too high: {mape_train}"
        
        if 'mape_test' in test_metrics:
            mape_test = test_metrics['mape_test']
            assert mape_test < 1000, f"MAPE for testing too high: {mape_test}"
        
        # Overall metrics
        rmse = test_metrics['rmse']
        score = test_metrics['score']
        
        # Check reasonableness of values
        assert rmse > 0, f"RMSE should be positive: {rmse}"
        assert not np.isinf(score), f"Score should not be infinite: {score}"
    
    def test_data_structure(self, prepared_data):
        """Test data structure"""
        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(prepared_data['date']), "Column date should be datetime"
        assert pd.api.types.is_numeric_dtype(prepared_data['listed']), "Column listed should be numeric"
        assert pd.api.types.is_numeric_dtype(prepared_data['price']), "Column price should be numeric"
        
        # Check uniqueness of values
        assert prepared_data['superbrand'].nunique() == 1, "Should be only one superbrand"
        assert prepared_data['bs_key'].nunique() == 1, "Should be only one bs_key"
        
        # Check range of listed values
        assert set(prepared_data['listed'].unique()).issubset({1, 2, 3, 4}), "Invalid values in listed column"
    
    def test_model_configuration(self, base_model):
        """Test model configuration"""
        # Check main configuration parameters
        assert hasattr(base_model.conf, 'Y_var'), "Target variable not set"
        assert hasattr(base_model.conf, 'price_var'), "Price variable not set"
        
        # Check model parameters
        assert hasattr(base_model.conf, 'params'), "Model parameters missing"
        assert not base_model.conf.params.empty, "Model parameters are empty"
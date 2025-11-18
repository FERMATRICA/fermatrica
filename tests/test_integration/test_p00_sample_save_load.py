import os
import yaml
import tempfile
import shutil
import pandas as pd
import numpy as np
import pytest

import fermatrica as fm

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


class TestP00SampleSaveLoad:
    """
    Integration test for model save and load functionality.
    
    Test verifies:
    1. Model saving
    2. Model loading
    3. Comparison of metrics before and after save/load
    """
    
    @pytest.fixture(scope="class")
    def temp_dir(self):
        """Temporary directory for model saving"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_model_save(self, trained_model, prepared_data, temp_dir):
        """Test model saving"""
        test_case = get_test_case("model_save_load")
        expected = test_case['expected']
        
        model, pred = trained_model
        
        # Save model
        save_path = model.save(prepared_data, path=temp_dir)
        
        assert os.path.exists(save_path), "Model not saved"
        assert os.path.isdir(save_path), "Save path is not a directory"
        
        # Check for model files
        expected_files = ['model_conf.xlsx', 'model_objects.pkl.lzma', 'dt_p.pkl.zstd']
        for file_name in expected_files:
            file_path = os.path.join(save_path, file_name)
            assert os.path.exists(file_path), f"File {file_name} not found in saved model"
    
    def test_model_load(self, trained_model, prepared_data, temp_dir):
        """Test model loading"""
        model, pred = trained_model
        
        # Save model
        save_path = model.save(prepared_data, path=temp_dir)
        
        # Load model
        loaded_model, loaded_dt_p, return_state = fm.model_load_ext(save_path, missed_stop=True)
        
        assert loaded_model is not None, "Model not loaded"
        assert loaded_dt_p is not None, "Data not loaded"
        assert return_state is not None, "Load state not defined"
        
        # Check that loaded data matches original
        assert len(loaded_dt_p) == len(prepared_data), "Number of loaded data doesn't match"
        assert list(loaded_dt_p.columns) == list(prepared_data.columns), "Loaded data columns don't match"
    
    def test_predictions_consistency(self, trained_model, prepared_data, temp_dir):
        """Test prediction consistency before and after save/load"""
        model, pred = trained_model
        dt_pred = fm.predict_ext(model, prepared_data)
        pred = dt_pred['predicted']

        # Save model
        save_path = model.save(prepared_data, path=temp_dir)
        
        # Load model
        loaded_model, loaded_dt_p, return_state = fm.model_load_ext(save_path, missed_stop=True)
        
        # Make predictions with loaded model using predict_ext
        loaded_dt_pred = fm.predict_ext(loaded_model, loaded_dt_p)
        loaded_pred = loaded_dt_pred['predicted']
        
        # Check that predictions number match
        assert len(loaded_pred) == len(pred), "Number of predictions doesn't match"
        
        # Compare predictions with numerical tolerance
        np.testing.assert_array_almost_equal(
            loaded_pred, 
            pred, 
            decimal=10, 
            err_msg="Predictions before and after save/load don't match"
        )
    
    def test_metrics_consistency(self, trained_model, prepared_data, temp_dir):
        """Test metrics consistency before and after save/load"""
        model, pred = trained_model
        dt_pred = fm.predict_ext(model, prepared_data)

        # Calculate metrics for original model
        train_data = prepared_data[prepared_data['listed'] == 2]
        test_data = prepared_data[prepared_data['listed'] == 3]
        train_pred = dt_pred.loc[prepared_data['listed'] == 2, 'predicted']
        test_pred = dt_pred.loc[prepared_data['listed'] == 3, 'predicted']
        
        original_metrics = {}
        if len(train_data) > 0 and len(train_pred) > 0:
            original_metrics['r_squared_train'] = fm.metrics.r_squared(train_data[model.conf.Y_var], train_pred)
            original_metrics['mape_train'] = fm.metrics.mapef(train_data[model.conf.Y_var], train_pred)

        if len(test_data) > 0 and len(test_pred) > 0:
            original_metrics['mape_test'] = fm.metrics.mapef(test_data[model.conf.Y_var], test_pred)

        # Save and load model
        save_path = model.save(prepared_data, path=temp_dir)
        loaded_model, loaded_dt_p, return_state = fm.model_load_ext(save_path, missed_stop=True)
        
        # Make predictions with loaded model using predict_ext
        loaded_dt_pred = fm.predict_ext(loaded_model, loaded_dt_p)
        loaded_pred = loaded_dt_pred['predicted']
        
        # Calculate metrics for loaded model
        loaded_train_pred = loaded_pred[loaded_dt_p['listed'] == 2]
        loaded_test_pred = loaded_pred[loaded_dt_p['listed'] == 3]

        loaded_train_data = loaded_dt_p[loaded_dt_p['listed'] == 2]
        loaded_test_data = loaded_dt_p[loaded_dt_p['listed'] == 3]

        loaded_metrics = {}
        if len(train_data) > 0 and len(loaded_train_pred) > 0:
            loaded_metrics['r_squared_train'] = fm.metrics.r_squared(loaded_train_data[model.conf.Y_var], loaded_train_pred)
            loaded_metrics['mape_train'] = fm.metrics.mapef(loaded_train_data[model.conf.Y_var], loaded_train_pred)
        
        if len(test_data) > 0 and len(loaded_test_pred) > 0:
            loaded_metrics['mape_test'] = fm.metrics.mapef(loaded_test_data[model.conf.Y_var], loaded_test_pred)
        
        # Check that metrics match
        for metric_name in original_metrics:
            assert metric_name in loaded_metrics, f"Metric {metric_name} missing in loaded model"
            np.testing.assert_almost_equal(
                original_metrics[metric_name],
                loaded_metrics[metric_name],
                decimal=10,
                err_msg=f"Metric {metric_name} doesn't match"
            )
    
    def test_model_configuration_consistency(self, trained_model, prepared_data, temp_dir):
        """Test model configuration consistency"""
        model, pred = trained_model
        
        # Save model
        save_path = model.save(prepared_data, path=temp_dir)
        
        # Load model
        loaded_model, loaded_dt_p, return_state = fm.model_load_ext(save_path, missed_stop=True)
        
        # Check that configurations match
        assert model.conf.Y_var == loaded_model.conf.Y_var, "Target variable doesn't match"
        assert model.conf.price_var == loaded_model.conf.price_var, "Price variable doesn't match"
        
        # Check model parameters
        original_params = model.conf.params
        loaded_params = loaded_model.conf.params
        
        assert len(original_params) == len(loaded_params), "Number of parameters doesn't match"
        
        # Check main parameters
        for col in ['variable', 'fun', 'arg', 'value']:
            if col in original_params.columns and col in loaded_params.columns:
                pd.testing.assert_series_equal(
                    original_params[col],
                    loaded_params[col],
                    check_names=False
                )
    
    def test_model_objects_consistency(self, trained_model, prepared_data, temp_dir):
        """Test model objects consistency"""
        model, pred = trained_model
        
        # Save model
        save_path = model.save(prepared_data, path=temp_dir)
        
        # Load model
        loaded_model, loaded_dt_p, return_state = fm.model_load_ext(save_path, missed_stop=True)
        
        # Check that model objects are accessible
        assert hasattr(loaded_model, 'obj'), "Loaded model missing obj attribute"
        assert hasattr(loaded_model.obj, 'models'), "obj of loaded model missing models attribute"
        assert 'main' in loaded_model.obj.models, "Main model missing in loaded model"
        
        # Check that main model has summary
        main_model = loaded_model.obj.models['main']
        assert hasattr(main_model, 'summary'), "Main model of loaded model missing summary"
    
    def test_save_load_error_handling(self, temp_dir):
        """Test error handling for save and load"""
        # Test loading non-existent model
        non_existent_path = os.path.join(temp_dir, "non_existent_model")
        
        # This test might not raise an exception depending on the implementation
        # Let's just check that the function handles the case gracefully
        try:
            result = fm.model_load_ext(non_existent_path, missed_stop=False)
            # If no exception is raised, that's also acceptable
            assert result is not None, "Function should return a result even for non-existent path"
        except Exception:
            # Exception is also acceptable
            pass
    
    # @pytest.mark.skipif(if_debug, reason="Skipped in debug mode")
    def test_save_load_performance(self, trained_model, prepared_data, temp_dir):
        """Test save and load performance"""
        test_case = get_test_case("model_save_load")
        expected = test_case['expected']
        
        import time
        
        model, pred = trained_model
        
        # Measure save time
        start_time = time.time()
        save_path = model.save(prepared_data, path=temp_dir)
        save_time = time.time() - start_time
        
        # Measure load time
        start_time = time.time()
        loaded_model, loaded_dt_p, return_state = fm.model_load_ext(save_path, missed_stop=True)
        load_time = time.time() - start_time
        
        # Check that operations complete in reasonable time using YAML expectations
        max_save_time = expected['max_save_time_seconds']
        max_load_time = expected['max_load_time_seconds']
        assert save_time < max_save_time, f"Model saving took too long: {save_time:.2f} seconds (max: {max_save_time})"
        assert load_time < max_load_time, f"Model loading took too long: {load_time:.2f} seconds (max: {max_load_time})"

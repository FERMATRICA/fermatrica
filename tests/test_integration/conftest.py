import os
import pandas as pd
import pytest

import fermatrica as fm


@pytest.fixture(scope="session")
def sample_data_path():
    """Path to p00_sample data"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(current_dir) == 'test_integration':
        current_dir = os.path.join(current_dir, "..", "..")
    if os.path.basename(current_dir) == 'tests':
        current_dir = os.path.join(current_dir, "..")

    return os.path.join(current_dir, "samples", "p00_sample")


@pytest.fixture(scope="session")
def model_conf_path(sample_data_path):
    """Path to model configuration"""
    return os.path.join(sample_data_path, "code_py", "model_data", "model_conf.xlsx")


@pytest.fixture(scope="session")
def data_path(sample_data_path):
    """Path to data file"""
    return os.path.join(sample_data_path, "data", "data_processed", "data_for_sample_p00_w.xlsx")


@pytest.fixture(scope="session")
def adhoc_model_code(sample_data_path):
    """Path to adhoc model module"""
    adhoc_path = os.path.join(sample_data_path, "code_py", "adhoc")
    if os.path.exists(os.path.join(adhoc_path, "model.py")):
        import sys
        sys.path.insert(0, adhoc_path)
        import model as adhoc_model
        return [adhoc_model]
    return []


@pytest.fixture(scope="session")
def prepared_data(data_path):
    """Prepared data for testing"""
    # Check if data file exists
    if not os.path.exists(data_path):
        pytest.skip(f"Data file not found: {data_path}")
    
    # Load data
    dt_p = pd.read_excel(data_path, parse_dates=['date'])
    
    # Settings as in example
    product_price = 50000
    conversion_rate = 1
    
    # Sync with FERMATRICA data structure
    dt_p['superbrand'] = 'brand_x'
    dt_p['bs_key'] = 1
    
    # Rename date column
    dt_p.rename({'Unnamed: 0': 'date'}, axis=1, inplace=True)
    
    # Remove Intercept if exists
    if 'Intercept' in dt_p.columns:
        del dt_p['Intercept']
    
    # Add product price
    dt_p['price'] = product_price * conversion_rate
    
    # Split into periods
    dt_p['listed'] = 1
    dt_p.loc[(dt_p['date'] > '2013-06-01') & (dt_p['date'] <= '2017-07-31'), 'listed'] = 2
    dt_p.loc[(dt_p['date'] > '2017-07-31') & (dt_p['date'] <= '2017-10-30'), 'listed'] = 3
    dt_p.loc[(dt_p['date'] > '2017-10-30'), 'listed'] = 4
    
    return dt_p


@pytest.fixture(scope="session")
def base_model(model_conf_path, prepared_data, adhoc_model_code):
    """Base model without training"""
    # Check if model config file exists
    if not os.path.exists(model_conf_path):
        pytest.skip(f"Model configuration file not found: {model_conf_path}")
    
    model = fm.Model(
        path=model_conf_path,
        adhoc_code=adhoc_model_code,
        ds=prepared_data
    )
    return model


@pytest.fixture(scope="session")
def trained_model(base_model, prepared_data):
    """Trained model"""
    # Transformation
    model = fm.transform(
        ds=prepared_data,
        model=base_model,
        set_start=True,
        if_by_ref=True
    )
    
    # Training
    pred, pred_raw, model = fm.fit_predict(
        prepared_data, 
        model, 
        if_fit=True, 
        return_full=True
    )
    
    return model, pred


@pytest.fixture(scope="session")
def test_metrics(trained_model, prepared_data):
    """Calculated metrics for testing"""
    model, pred = trained_model
    dt_pred = fm.predict_ext(model, prepared_data)
    
    # Get training and test data
    train_data = prepared_data[prepared_data['listed'] == 2]
    test_data = prepared_data[prepared_data['listed'] == 3]
    
    # Predictions for training and testing
    train_pred = dt_pred.loc[prepared_data['listed'] == 2, 'predicted']
    test_pred = dt_pred.loc[prepared_data['listed'] == 3, 'predicted']
    
    metrics = {}
    
    # Training metrics
    if len(train_data) > 0:
        metrics['r_squared_train'] = fm.metrics.r_squared(train_data[model.conf.Y_var], train_pred)
        metrics['mape_train'] = fm.metrics.mapef(train_data[model.conf.Y_var], train_pred)
    
    # Testing metrics
    if len(test_data) > 0:
        metrics['mape_test'] = fm.metrics.mapef(test_data[model.conf.Y_var], test_pred)
    
    # Overall metrics
    all_data = prepared_data[prepared_data['listed'].isin([2, 3])]
    all_pred = dt_pred.loc[prepared_data['listed'].isin([2, 3]), 'predicted']
    
    if len(all_data) > 0:
        metrics['rmse'] = fm.metrics.rmse(all_data[model.conf.Y_var], all_pred)
        metrics['score'] = fm.scoring(all_data, all_pred, model)
    
    return metrics

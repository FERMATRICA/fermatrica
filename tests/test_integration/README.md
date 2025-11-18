 # FERMATRICA Integration Tests

This directory contains integration tests for the FERMATRICA library.

## Description

Integration tests verify the complete data processing cycle, including:
- Data loading and preparation
- Model creation and configuration
- Data transformation
- Model training
- Prediction generation
- Quality metrics calculation
- Model saving and loading

## Test Structure

### test_p00_sample_integration.py
Main integration test based on the `p00_sample` example. Verifies:
- Data loading from `data_for_sample_p00_w.xlsx`
- Model creation with configuration from `model_conf.xlsx`
- Data transformation
- Prediction generation
- Quality metrics calculation and verification (R², MAPE, RMSE)
- Model consistency

### test_p00_sample_save_load.py
Additional test for model save and load functionality:
- Saving trained model
- Loading saved model
- Comparing predictions before and after save/load
- Verifying metrics consistency
- Checking model configuration preservation

### test_cases/integration_p00_sample.yaml
File with test cases and expected values for integration tests.

## Running Tests

### Run all integration tests
```bash
pytest tests/test_integration/ -v
```

### Run specific test
```bash
pytest tests/test_integration/test_p00_sample_integration.py -v
pytest tests/test_integration/test_p00_sample_save_load.py -v
```

### Run in debug mode
```bash
pytest tests/test_integration/ -v --tb=short
```

## Requirements

To run integration tests, you need:
1. Installed FERMATRICA library
2. Access to sample data in `samples/p00_sample/`
3. Dependencies from `requirements.txt`

## Test Data

Tests use data from the `p00_sample` example:
- `data/data_processed/data_for_sample_p00_w.xlsx` - source data
- `code_py/model_data/model_conf.xlsx` - model configuration
- `code_py/adhoc/model.py` - additional model functions (if available)

## Expected Results

### Quality Metrics
- R² for training: 0.0 - 1.0
- MAPE for training: < 1000%
- MAPE for testing: < 1000%
- RMSE: > 0
- Overall score: finite value

### Functionality
- Successful data loading
- Model creation with correct configuration
- Transformation execution
- Prediction generation
- Model saving and loading without quality loss

## Debugging

If tests fail:
1. Check data availability in `samples/p00_sample/`
2. Ensure all dependencies are installed
3. Verify model configuration in `model_conf.xlsx`
4. Run tests with `-v` flag for detailed output

## Adding New Tests

To add new integration tests:
1. Create new test file in `tests/test_integration/`
2. Add test cases to `test_cases/`
3. Update this README file
4. Ensure tests pass locally before committing

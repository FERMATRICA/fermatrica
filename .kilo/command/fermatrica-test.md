---
description: Run FERMATRICA tests with proper configuration
---

# Run FERMATRICA Tests

You are a FERMATRICA testing expert. Run the appropriate tests for the changes made and analyze failures.

## Test Commands

```bash
# All tests
pytest tests/

# Specific module
pytest tests/test_model/test_transform.py -v

# Integration tests (end-to-end pipeline)
pytest tests/test_integration/ -v

# With debug mode (drops into breakpoint() on assertion failure)
pytest tests/ --if-debug

# Verbose with shortened traceback
pytest tests/test_model/ -v --tb=short

# Run and stop at first failure
pytest tests/test_model/test_transform_fun.py -x
```

## What to test when changing code

| Change location | Tests to run |
|-----------------|-------------|
| `fermatrica/model/transform_fun.py` | `test_transform_fun.py` |
| `fermatrica/model/transform.py` | `test_transform.py` |
| `fermatrica/model/lhs_fun.py` | `test_lhs_fun.py` |
| `fermatrica/model/model_conf.py` | `test_model_conf.py` |
| `fermatrica/evaluation/metrics.py` | `test_metrics.py` |
| `fermatrica/evaluation/scoring.py` | `test_scoring.py` |
| `fermatrica/model/predict.py` | `test_integration/` |
| `fermatrica/optim/` | `test_integration/` |
| Any change affecting pipeline | `test_integration/` (full pipeline) |

## Test Architecture

### YAML-driven parametrization
Test cases are defined in `tests/**/test_cases/*.yaml` with custom YAML constructors (`!pd.DataFrame`, `!pd.Series`, `!np.array`). Each test case entry has:
- `description`: test name
- `params`: input values
- `expected_result`: expected output
- `exp_err`: `pytest.raises(FermatricaError)` or `nullcontext()`

### Integration tests
Session-scoped fixtures in `tests/test_integration/conftest.py`:
- `prepared_data` — loads p00 sample data
- `base_model` — creates `fm.Model()` from config
- `trained_model` — runs transform + fit, returns `(model, pred)`
- `test_metrics` — computes R², MAPE, RMSE, score

### Debug flag
Root `conftest.py` provides `if_debug` fixture. When `True`, drops into `breakpoint()` on failure:
```bash
pytest tests/ --if-debug
```

## Common test patterns to check for

### Expected error testing
```yaml
exp_err: !pytest_raises
  type: FermatricaError
  match: "Model definition : Params :"
```

### DataFrame comparison
```yaml
expected_result: !pd.DataFrame
  data: [[1, 2], [3, 4]]
  columns: [a, b]
```

### Parametrization helper
```python
from tests.basics.basics import parametrize_fun
test_cases, ids = parametrize_fun("test_cases/transform/lag.yaml", "test_cases/transform/adstock.yaml")
```

## When tests fail

1. **Read the assertion error carefully** — note the expected vs actual values
2. **Check if the test expectations need updating** (new behavior is correct) or code needs fixing
3. **Integration test failures** usually indicate a pipeline break — run transform + fit_predict manually to reproduce
4. **YAML test failures** — check if `expected_result` in the YAML matches the new code behavior
5. **Use `--if-debug`** to get an interactive debugger on assertion failures

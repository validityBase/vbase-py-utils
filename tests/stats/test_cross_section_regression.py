import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal, assert_frame_equal

from src.cross_section_regression import (
    run_cross_sectional_regression,
    run_monthly_factor_returns
)

# --- Helpers for generating test data ---
def make_simple_cs():
    # Two assets, single factor
    assets = ['A', 'B']
    # True factor return f = 2.0
    f = 2.0
    X = pd.DataFrame({'F': [1.0, 1.0]}, index=assets)
    y = pd.Series([f * 1.0, f * 1.0], index=assets)
    w = pd.Series([1.0, 1.0], index=assets)
    return y, X, w, f


def make_monthly_cs(n_assets=4, n_factors=2, n_periods=3, seed=1):
    np.random.seed(seed)
    assets = [f'A{i}' for i in range(n_assets)]
    factors = [f'F{j}' for j in range(n_factors)]
    periods = [f'2025-0{p+1}' for p in range(n_periods)]
    # exposures MultiIndex
    idx = pd.MultiIndex.from_product([periods, assets], names=['period', 'asset'])
    exposures = pd.DataFrame(np.random.randn(len(idx), n_factors), index=idx, columns=factors)
    # true factor returns
    true_fr = pd.DataFrame(np.random.randn(n_periods, n_factors)*0.1, index=periods, columns=factors)
    # returns
    data = {}
    for per in periods:
        Xmat = exposures.xs(per, level=0).to_numpy()
        fvals = true_fr.loc[per].to_numpy()
        noise = np.random.randn(n_assets) * 1e-6
        data[per] = pd.Series(Xmat.dot(fvals) + noise, index=assets)
    returns = pd.DataFrame(data)
    # weights df
    weights = pd.DataFrame(1.0, index=idx, columns=['w'])
    return returns, exposures, weights, true_fr

# --- Tests for run_cross_sectional_regression ---
def test_cross_section_regression_accuracy():
    y, X, w, true_f = make_simple_cs()
    est = run_cross_sectional_regression(y, X, w)
    assert isinstance(est, pd.Series)
    # Only one factor
    assert est.index.tolist() == ['F']
    # Approx equal
    assert pytest.approx(true_f, rel=1e-6) == est['F']
    # Name attribute
    assert est.name == 'factor_returns'


def test_cross_section_regression_index_mismatch_error():
    y = pd.Series([1.0, 2.0], index=['A', 'B'])
    X = pd.DataFrame({'F': [1, 2]}, index=['A', 'C'])
    w = pd.Series([1.0, 1.0], index=['A', 'B'])
    with pytest.raises(ValueError):
        run_cross_sectional_regression(y, X, w)

# --- Tests for run_monthly_factor_returns ---
def test_monthly_returns_shape_and_values():
    returns, exposures, weights, true_fr = make_monthly_cs()
    est_df = run_monthly_factor_returns(returns, exposures, weights)
    # same index and columns
    assert est_df.shape == true_fr.shape
    assert list(est_df.index) == list(true_fr.index)
    assert list(est_df.columns) == list(true_fr.columns)
    # values close
    diff = (est_df - true_fr).abs().to_numpy().max()
    assert diff < 1e-3


def test_monthly_returns_missing_exposures_results_nan():
    returns, exposures, weights, true_fr = make_monthly_cs()
    # drop exposures for second period
    period_to_drop = true_fr.index[1]
    exposures2 = exposures.drop(period_to_drop, level=0)
    est2 = run_monthly_factor_returns(returns, exposures2, weights)
    # that row should be all NaN
    assert est2.loc[period_to_drop].isna().all()

if __name__ == '__main__':
    pytest.main()
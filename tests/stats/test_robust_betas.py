"""Unit tests for the robust timeseries regression module"""

import unittest

import numpy as np
import pandas as pd

from vbase_utils.stats.robust_betas import robust_betas

# Constants for test data generation
# Standard deviation of factor returns
STD_FACT_RETS = 0.01
# Standard deviation of asset returns
STD_ASSET_RETS = 0.005

# Default delta for floating point comparisons
DEFAULT_DELTA = 0.2


class TestRobustBetas(unittest.TestCase):
    """Unit tests for the robust_betas function"""

    @classmethod
    def setUpClass(cls):
        """Set random seed and create common variables."""
        np.random.seed(42)
        cls.n_timestamps = 100
        cls.spy_returns = np.random.normal(0, STD_FACT_RETS, cls.n_timestamps)

    def test_single_asset(self):
        """Test beta estimation for a single ETF with known beta."""
        asset_returns = 1.5 * self.spy_returns + np.random.normal(
            0, STD_ASSET_RETS, self.n_timestamps
        )
        df_asset_rets = pd.DataFrame({"Asset1": asset_returns})
        df_fact_rets = pd.DataFrame({"SPY": self.spy_returns})
        beta_matrix = robust_betas(df_asset_rets, df_fact_rets, half_life=30)
        self.assertAlmostEqual(
            beta_matrix.loc["SPY", "Asset1"], 1.5, delta=DEFAULT_DELTA
        )

    def test_multiple_assets(self):
        """Test beta estimation for multiple ETFs with known betas."""
        asset_returns_1 = 1.2 * self.spy_returns + np.random.normal(
            0, STD_ASSET_RETS, self.n_timestamps
        )
        asset_returns_2 = 0.8 * self.spy_returns + np.random.normal(
            0, STD_ASSET_RETS, self.n_timestamps
        )
        df_asset_rets = pd.DataFrame(
            {"Asset1": asset_returns_1, "Asset2": asset_returns_2}
        )
        df_fact_rets = pd.DataFrame({"SPY": self.spy_returns})
        beta_matrix = robust_betas(df_asset_rets, df_fact_rets, half_life=30)
        self.assertAlmostEqual(
            beta_matrix.loc["SPY", "Asset1"], 1.2, delta=DEFAULT_DELTA
        )
        self.assertAlmostEqual(
            beta_matrix.loc["SPY", "Asset2"], 0.8, delta=DEFAULT_DELTA
        )

    def test_multiple_factors(self):
        """Test beta estimation with multiple factors in df_fact_rets."""
        iwm_returns = np.random.normal(0, STD_FACT_RETS, self.n_timestamps)
        asset_returns = (
            1.2 * self.spy_returns
            + 0.5 * iwm_returns
            + np.random.normal(0, STD_ASSET_RETS, self.n_timestamps)
        )
        df_asset_rets = pd.DataFrame({"Asset1": asset_returns})
        df_fact_rets = pd.DataFrame({"SPY": self.spy_returns, "IWM": iwm_returns})
        beta_matrix = robust_betas(df_asset_rets, df_fact_rets, half_life=30)
        self.assertAlmostEqual(
            beta_matrix.loc["SPY", "Asset1"], 1.2, delta=DEFAULT_DELTA
        )
        self.assertAlmostEqual(
            beta_matrix.loc["IWM", "Asset1"], 0.5, delta=DEFAULT_DELTA
        )

    def test_lambda_parameter(self):
        """Test beta estimation using lambda_ instead of half_life."""
        asset_returns = 1.5 * self.spy_returns + np.random.normal(
            0, STD_ASSET_RETS, self.n_timestamps
        )
        df_asset_rets = pd.DataFrame({"Asset1": asset_returns})
        df_fact_rets = pd.DataFrame({"SPY": self.spy_returns})
        beta_matrix = robust_betas(df_asset_rets, df_fact_rets, lambda_=0.985)
        self.assertAlmostEqual(
            beta_matrix.loc["SPY", "Asset1"], 1.5, delta=DEFAULT_DELTA
        )

    def test_empty_data(self):
        """Test handling of empty input DataFrames."""
        df_asset_rets = pd.DataFrame()
        df_fact_rets = pd.DataFrame()
        with self.assertRaises(ValueError):
            robust_betas(df_asset_rets, df_fact_rets, half_life=30)

    def test_mismatched_timestamps(self):
        """Test handling of df_asset_rets and df_fact_rets with different row counts."""
        asset_returns = 1.5 * self.spy_returns[:90] + np.random.normal(
            0, STD_ASSET_RETS, 90
        )
        df_asset_rets = pd.DataFrame({"Asset1": asset_returns})
        df_fact_rets = pd.DataFrame({"SPY": self.spy_returns})
        with self.assertRaises(ValueError):
            robust_betas(df_asset_rets, df_fact_rets, half_life=30)

    def test_invalid_half_life(self):
        """Test handling of negative or zero half_life."""
        asset_returns = 1.5 * self.spy_returns
        df_asset_rets = pd.DataFrame({"Asset1": asset_returns})
        df_fact_rets = pd.DataFrame({"SPY": self.spy_returns})
        for invalid_half_life in [0, -1]:
            with self.assertRaises(ValueError):
                robust_betas(df_asset_rets, df_fact_rets, half_life=invalid_half_life)

    def test_invalid_lambda(self):
        """Test handling of invalid lambda_ values."""
        asset_returns = 1.5 * self.spy_returns + np.random.normal(
            0, STD_ASSET_RETS, self.n_timestamps
        )
        df_asset_rets = pd.DataFrame({"Asset1": asset_returns})
        df_fact_rets = pd.DataFrame({"SPY": self.spy_returns})
        for invalid_lambda in [0, 1.5]:
            with self.assertRaises(ValueError):
                robust_betas(df_asset_rets, df_fact_rets, lambda_=invalid_lambda)

    def test_no_variation_in_x(self):
        """Test handling of df_fact_rets with zero variance."""
        spy_constant = np.ones(self.n_timestamps) * 0.01  # Constant returns
        asset_returns = 1.5 * spy_constant + np.random.normal(
            0, STD_ASSET_RETS, self.n_timestamps
        )
        df_asset_rets = pd.DataFrame({"Asset1": asset_returns})
        df_fact_rets = pd.DataFrame({"SPY": spy_constant})
        with self.assertRaises(ValueError):
            robust_betas(df_asset_rets, df_fact_rets, half_life=30)

    def test_outlier_heavy_data(self):
        """Test robustness with significant outliers."""
        asset_returns = 1.5 * self.spy_returns + np.random.normal(
            0, STD_ASSET_RETS, self.n_timestamps
        )
        asset_returns[::10] += 0.1  # Add large outliers every 10th point
        df_asset_rets = pd.DataFrame({"Asset1": asset_returns})
        df_fact_rets = pd.DataFrame({"SPY": self.spy_returns})
        beta_matrix = robust_betas(df_asset_rets, df_fact_rets, half_life=30)
        self.assertAlmostEqual(
            beta_matrix.loc["SPY", "Asset1"], 1.5, delta=DEFAULT_DELTA * 2
        )  # Allow larger delta due to outliers

    def test_insufficient_timestamps(self):
        """Test handling of insufficient timestamps after cleaning."""
        short_spy_returns = np.random.normal(0, STD_FACT_RETS, 5)
        short_asset_returns = 1.5 * short_spy_returns + np.random.normal(
            0, STD_ASSET_RETS, 5
        )
        df_asset_rets = pd.DataFrame({"Asset1": short_asset_returns})
        df_fact_rets = pd.DataFrame({"SPY": short_spy_returns})
        # When insufficient timestamps are available, a warning is issued
        # and a DataFrame with all NaN betas is returned.
        beta_matrix = robust_betas(
            df_asset_rets, df_fact_rets, half_life=2, min_timestamps=10
        )
        # Check that the betas are all NaN.
        self.assertTrue(beta_matrix.isna().all().all())


if __name__ == "__main__":
    unittest.main()

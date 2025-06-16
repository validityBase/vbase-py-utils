"""Unit tests for the pit_robust_betas function."""

import unittest

import numpy as np
import pandas as pd

from vbase_utils.stats.pit_robust_betas import pit_robust_betas

# Constants for test data generation
# Standard deviation of factor returns
STD_FACT_RETS = 0.01
# Standard deviation of asset returns
STD_ASSET_RETS = 0.005
# Default delta for floating point comparisons
DEFAULT_DELTA = 0.2


class TestPitRobustBetas(unittest.TestCase):
    """Unit tests for the pit_robust_betas function."""

    @classmethod
    def setUpClass(cls):
        """Set random seed and create common variables."""
        np.random.seed(42)
        cls.n_timestamps = 100
        cls.dates = pd.date_range("2023-01-01", periods=cls.n_timestamps)
        cls.spy_returns = pd.Series(
            np.random.normal(0, STD_FACT_RETS, cls.n_timestamps),
            index=cls.dates,
            name="SPY",
        )

    def setUp(self):
        """Set up test fixtures."""
        # Create factor returns DataFrame
        self.df_fact_rets = pd.DataFrame({"SPY": self.spy_returns})

        # Create asset returns with known betas
        asset1_rets = 1.5 * self.spy_returns + np.random.normal(
            0, STD_ASSET_RETS, self.n_timestamps
        )
        asset2_rets = 0.8 * self.spy_returns + np.random.normal(
            0, STD_ASSET_RETS, self.n_timestamps
        )
        self.df_asset_rets = pd.DataFrame(
            {"Asset1": asset1_rets, "Asset2": asset2_rets}, index=self.dates
        )

    def test_basic_functionality(self):
        """Test basic functionality with single factor and multiple assets."""
        results = pit_robust_betas(self.df_asset_rets, self.df_fact_rets, half_life=30)

        # Check structure of results
        self.assertIn("df_betas", results)
        self.assertIn("df_hedge_rets_by_fact", results)
        self.assertIn("df_hedge_rets", results)
        self.assertIn("df_asset_resids", results)

        # Check betas DataFrame structure
        df_betas = results["df_betas"]
        self.assertEqual(df_betas.index.names, ["timestamp", "factor"])
        self.assertEqual(set(df_betas.columns), {"Asset1", "Asset2"})
        self.assertEqual(set(df_betas.index.get_level_values("factor")), {"SPY"})

        # Check hedge returns by factor DataFrame structure
        df_hedge_rets_by_fact = results["df_hedge_rets_by_fact"]
        self.assertEqual(df_hedge_rets_by_fact.index.names, ["timestamp", "factor"])
        self.assertEqual(set(df_hedge_rets_by_fact.columns), {"Asset1", "Asset2"})
        self.assertEqual(
            set(df_hedge_rets_by_fact.index.get_level_values("factor")), {"SPY"}
        )

        # Check hedge returns DataFrame structure
        df_hedge_rets = results["df_hedge_rets"]
        self.assertEqual(df_hedge_rets.index.name, "timestamp")
        self.assertEqual(set(df_hedge_rets.columns), {"Asset1", "Asset2"})

        # Check asset residuals DataFrame structure
        df_asset_resids = results["df_asset_resids"]
        self.assertEqual(df_asset_resids.index.name, "timestamp")
        self.assertEqual(set(df_asset_resids.columns), {"Asset1", "Asset2"})

        # Check beta values (using last timestamp for stability)
        last_betas = df_betas.xs(df_betas.index.get_level_values("timestamp")[-1])
        self.assertAlmostEqual(
            last_betas.loc["SPY", "Asset1"], 1.5, delta=DEFAULT_DELTA
        )
        self.assertAlmostEqual(
            last_betas.loc["SPY", "Asset2"], 0.8, delta=DEFAULT_DELTA
        )

    def test_multiple_factors(self):
        """Test regression with multiple factors."""
        # Add second factor
        iwm_returns = pd.Series(
            np.random.normal(0, STD_FACT_RETS, self.n_timestamps),
            index=self.dates,
            name="IWM",
        )
        df_fact_rets = pd.DataFrame({"SPY": self.spy_returns, "IWM": iwm_returns})

        # Create asset returns dependent on both factors
        asset_rets = (
            1.2 * self.spy_returns
            + 0.5 * iwm_returns
            + np.random.normal(0, STD_ASSET_RETS, self.n_timestamps)
        )
        df_asset_rets = pd.DataFrame({"Asset1": asset_rets}, index=self.dates)

        results = pit_robust_betas(df_asset_rets, df_fact_rets, half_life=30)

        # Check beta values
        last_betas = results["df_betas"].xs(
            results["df_betas"].index.get_level_values("timestamp")[-1]
        )
        self.assertAlmostEqual(
            last_betas.loc["SPY", "Asset1"], 1.2, delta=DEFAULT_DELTA
        )
        self.assertAlmostEqual(
            last_betas.loc["IWM", "Asset1"], 0.5, delta=DEFAULT_DELTA
        )

        # Check hedge returns by factor
        df_hedge_rets_by_fact = results["df_hedge_rets_by_fact"]
        self.assertEqual(
            set(df_hedge_rets_by_fact.index.get_level_values("factor")), {"SPY", "IWM"}
        )

    def test_rebalance_time_index(self):
        """Test using custom rebalance time index."""
        # Create monthly rebalance dates
        rebalance_dates = pd.date_range("2023-01-01", "2023-12-31", freq="ME")

        results = pit_robust_betas(
            self.df_asset_rets,
            self.df_fact_rets,
            half_life=30,
            rebalance_time_index=rebalance_dates,
        )

        # Check that betas have been expanded to the asset returns index.
        self.assertEqual(
            set(results["df_betas"].index.get_level_values("timestamp")),
            set(self.df_asset_rets.index),
        )
        # Check that the betas are constant between the rebalance dates.
        # A simple check is that the number of non-NA and non-zero differences
        # is less than 10% of the total number of elements.
        self.assertGreater(
            (results["df_betas"].diff() == 0).sum().sum()
            + results["df_betas"].isna().sum().sum() / results["df_betas"].size,
            0.9,
        )

    def test_empty_data(self):
        """Test handling of empty input DataFrames."""
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            pit_robust_betas(empty_df, self.df_fact_rets)

        with self.assertRaises(ValueError):
            pit_robust_betas(self.df_asset_rets, empty_df)

    def test_invalid_index(self):
        """Test handling of non-DatetimeIndex."""
        df = pd.DataFrame({"A": [1, 2, 3]}, index=[1, 2, 3])
        with self.assertRaises(ValueError):
            pit_robust_betas(df, self.df_fact_rets)

        with self.assertRaises(ValueError):
            pit_robust_betas(self.df_asset_rets, df)

    def test_mismatched_timestamps(self):
        """Test handling of non-overlapping timestamps."""
        df_asset_rets = pd.DataFrame(
            {"A": [1, 2, 3]}, index=pd.date_range("2023-01-01", periods=3)
        )
        df_fact_rets = pd.DataFrame(
            {"B": [4, 5, 6]}, index=pd.date_range("2023-02-01", periods=3)
        )

        with self.assertRaises(ValueError):
            pit_robust_betas(df_asset_rets, df_fact_rets)

    def test_hedge_returns_calculation(self):
        """Test that hedge returns are correctly calculated using previous betas."""
        results = pit_robust_betas(self.df_asset_rets, self.df_fact_rets, half_life=30)

        # Get a specific timestamp (skip first as it has no hedge returns)
        timestamp = results["df_hedge_rets"].index[50]
        prev_timestamp = results["df_betas"].index.get_level_values("timestamp")[
            results["df_betas"].index.get_level_values("timestamp").get_loc(timestamp)
            - 1
        ]

        # Calculate expected hedge returns for Asset1
        prev_betas = results["df_betas"].xs(prev_timestamp)["Asset1"]
        expected_hedge_ret = (
            -1 * prev_betas["SPY"] * self.df_fact_rets.loc[timestamp, "SPY"]
        )

        # Compare with actual hedge returns
        actual_hedge_ret = results["df_hedge_rets"].loc[timestamp, "Asset1"]
        self.assertAlmostEqual(
            actual_hedge_ret, expected_hedge_ret, delta=DEFAULT_DELTA
        )

        # Check hedge returns by factor
        actual_hedge_ret_by_fact = results["df_hedge_rets_by_fact"].xs(timestamp)[
            "Asset1"
        ]["SPY"]
        self.assertAlmostEqual(
            actual_hedge_ret_by_fact, expected_hedge_ret, delta=DEFAULT_DELTA
        )

    def test_asset_residuals_calculation(self):
        """Test that asset residuals are correctly calculated."""
        results = pit_robust_betas(self.df_asset_rets, self.df_fact_rets, half_life=30)

        # Get a specific timestamp (skip first as it has no residuals)
        timestamp = results["df_asset_resids"].index[50]

        # Calculate expected residual for Asset1
        expected_resid = (
            self.df_asset_rets.loc[timestamp, "Asset1"]
            + results["df_hedge_rets"].loc[timestamp, "Asset1"]
        )

        # Compare with actual residual
        actual_resid = results["df_asset_resids"].loc[timestamp, "Asset1"]
        self.assertAlmostEqual(actual_resid, expected_resid, delta=DEFAULT_DELTA)


if __name__ == "__main__":
    unittest.main()

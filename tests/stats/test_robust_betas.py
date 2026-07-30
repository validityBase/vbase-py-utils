"""Unit tests for the robust timeseries regression module"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from tests.stats._robust_betas_fixtures import (
    STD_ASSET_RETS,
    STD_FACT_RETS,
    make_fit_first_call_error_side_effect,
    make_multi_asset_ret_frames,
    make_multi_factor_ret_frames,
    make_single_asset_ret_frames,
)
from vbase_utils.stats._huber_rlm import fit_huber_rlm_params as real_fit
from vbase_utils.stats.robust_betas import (
    NEAR_ZERO_VARIANCE_THRESHOLD,
    robust_betas,
)

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
        df_asset_rets, df_fact_rets = make_single_asset_ret_frames(
            self.spy_returns, self.n_timestamps
        )
        beta_matrix = robust_betas(df_asset_rets, df_fact_rets, half_life=30)
        self.assertAlmostEqual(
            beta_matrix.loc["SPY", "Asset1"], 1.5, delta=DEFAULT_DELTA
        )

    def test_multiple_assets(self):
        """Test beta estimation for multiple ETFs with known betas."""
        df_asset_rets, df_fact_rets = make_multi_asset_ret_frames(
            self.spy_returns, self.n_timestamps
        )
        beta_matrix = robust_betas(df_asset_rets, df_fact_rets, half_life=30)
        self.assertAlmostEqual(
            beta_matrix.loc["SPY", "Asset1"], 1.2, delta=DEFAULT_DELTA
        )
        self.assertAlmostEqual(
            beta_matrix.loc["SPY", "Asset2"], 0.8, delta=DEFAULT_DELTA
        )

    def test_multiple_factors(self):
        """Test beta estimation with multiple factors in df_fact_rets."""
        df_asset_rets, df_fact_rets = make_multi_factor_ret_frames(
            self.spy_returns, self.n_timestamps
        )
        beta_matrix = robust_betas(df_asset_rets, df_fact_rets, half_life=30)
        self.assertAlmostEqual(
            beta_matrix.loc["SPY", "Asset1"], 1.2, delta=DEFAULT_DELTA
        )
        self.assertAlmostEqual(
            beta_matrix.loc["IWM", "Asset1"], 0.5, delta=DEFAULT_DELTA
        )

    def test_lambda_parameter(self):
        """Test beta estimation using lambda_ instead of half_life."""
        df_asset_rets, df_fact_rets = make_single_asset_ret_frames(
            self.spy_returns, self.n_timestamps
        )
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

    def test_mismatched_index(self):
        """Test handling of df_asset_rets and df_fact_rets with different index."""
        asset_returns = 1.5 * self.spy_returns + np.random.normal(
            0, STD_ASSET_RETS, self.n_timestamps
        )
        df_asset_rets = pd.DataFrame(
            {"Asset1": asset_returns},
            index=pd.date_range("2023-01-01", periods=self.n_timestamps),
        )
        df_fact_rets = pd.DataFrame(
            {"SPY": self.spy_returns},
            index=pd.date_range("2023-02-01", periods=self.n_timestamps),
        )
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

    def test_rlm_fit_error_returns_nan_for_affected_asset(self):
        """When the fit raises for one asset,
        the function does not raise and that asset's betas are NaN."""
        df_asset_rets, df_fact_rets = make_multi_asset_ret_frames(
            self.spy_returns, self.n_timestamps
        )

        with patch(
            "vbase_utils.stats.robust_betas.fit_huber_rlm_params",
            side_effect=make_fit_first_call_error_side_effect(real_fit),
        ):
            beta_matrix = robust_betas(df_asset_rets, df_fact_rets, half_life=30)

        self.assertTrue(beta_matrix["Asset1"].isna().all())
        self.assertFalse(beta_matrix["Asset2"].isna().all())

    def test_non_finite_factor_rows_are_deleted_not_the_window(self):
        """NaN or inf in a factor row drops that row, matching R's na.omit.

        Rows on which any factor is non-finite are excluded from the regression
        for every asset (listwise deletion); the rest of the window is still fit.
        Previously a single bad row discarded the whole window.
        """
        for bad_value, position in ((np.nan, 5), (np.inf, 10)):
            with self.subTest(bad_value=bad_value):
                df_asset_rets, df_fact_rets = make_single_asset_ret_frames(
                    self.spy_returns, self.n_timestamps
                )
                df_fact_rets.iloc[position, 0] = bad_value

                beta_matrix = robust_betas(df_asset_rets, df_fact_rets, half_life=30)

                # Betas are produced, and recover the true coefficient.
                self.assertFalse(beta_matrix.isna().any().any())
                self.assertAlmostEqual(
                    beta_matrix.loc["SPY", "Asset1"], 1.5, delta=DEFAULT_DELTA
                )

    def test_deletion_matches_masking_the_bad_row_by_hand(self):
        """Deleting the row inside the fit equals blanking it in both inputs.

        Pins the semantics exactly rather than approximately: the admitted rows
        are the intersection of each asset's defined rows with the rows on which
        every factor is finite, so making the row undefined for the asset instead
        must produce identical betas. The weights are indexed by position in the
        full window in both cases, which is what keeps the decay anchored to
        calendar time rather than to the surviving row count.
        """
        position = 7
        df_asset_rets, df_fact_rets = make_single_asset_ret_frames(
            self.spy_returns, self.n_timestamps
        )

        df_fact_hole = df_fact_rets.copy()
        df_fact_hole.iloc[position, 0] = np.nan
        from_factor_hole = robust_betas(df_asset_rets, df_fact_hole, half_life=30)

        df_asset_hole = df_asset_rets.copy()
        df_asset_hole.iloc[position, 0] = np.nan
        from_asset_hole = robust_betas(df_asset_hole, df_fact_rets, half_life=30)

        np.testing.assert_array_equal(
            from_factor_hole.to_numpy(), from_asset_hole.to_numpy()
        )

    def test_min_timestamps_counts_rows_surviving_deletion(self):
        """A long window with too few complete rows yields no betas.

        The gate must measure the rows the regression can use, not the window
        length: a factor whose history starts late leaves a long window with only
        a handful of complete rows.
        """
        df_asset_rets, df_fact_rets = make_single_asset_ret_frames(
            self.spy_returns, self.n_timestamps
        )
        # Only the last 9 rows are complete, in a window of n_timestamps.
        df_fact_rets.iloc[:-9, 0] = np.nan

        beta_matrix = robust_betas(
            df_asset_rets, df_fact_rets, half_life=30, min_timestamps=10
        )
        self.assertTrue(beta_matrix.isna().all().all())

        # One more complete row clears the gate.
        df_fact_rets_ok = df_fact_rets.copy()
        df_fact_rets_ok.iloc[-10, 0] = self.spy_returns[-10]
        beta_matrix_ok = robust_betas(
            df_asset_rets, df_fact_rets_ok, half_life=30, min_timestamps=10
        )
        self.assertFalse(beta_matrix_ok.isna().all().all())

    def test_flat_factor_over_surviving_rows_degrades(self):
        """Near-zero variance over the admitted rows yields no betas, not a raise.

        A factor can vary across its own column and still be flat over one
        window's surviving rows, if its variation sits in the rows deletion
        removed. That is a local, recoverable condition -- the next window admits
        more rows -- so it must not kill the run the way a flat input column does.
        """
        n = 40
        dates = pd.date_range("2023-01-01", periods=n)
        # F1 is constant on the rows F2 defines, and varies only where F2 is NaN.
        f1 = np.full(n, 0.01)
        f1[30:] = np.random.normal(0, STD_FACT_RETS, n - 30)
        f2 = np.random.normal(0, STD_FACT_RETS, n)
        f2[30:] = np.nan
        df_fact_rets = pd.DataFrame({"F1": f1, "F2": f2}, index=dates)
        df_asset_rets = pd.DataFrame(
            {"Asset1": np.random.normal(0, STD_ASSET_RETS, n)}, index=dates
        )

        # The whole-column check must not fire: F1 does vary over its own values.
        self.assertGreater(df_fact_rets["F1"].var(), NEAR_ZERO_VARIANCE_THRESHOLD)

        beta_matrix = robust_betas(
            df_asset_rets, df_fact_rets, half_life=30, min_timestamps=10
        )
        self.assertTrue(beta_matrix.isna().all().all())

    def test_with_nan_asset_returns(self):
        """NaN in asset returns must not cause shape mismatch when weighting const column."""
        asset_returns = np.concatenate(
            [
                np.full(20, np.nan),
                1.5 * self.spy_returns[20:] + np.random.normal(0, STD_ASSET_RETS, 80),
            ]
        )
        df_asset_rets = pd.DataFrame({"Asset1": asset_returns})
        df_fact_rets = pd.DataFrame({"SPY": self.spy_returns})
        beta_matrix = robust_betas(df_asset_rets, df_fact_rets, half_life=30)
        self.assertAlmostEqual(
            beta_matrix.loc["SPY", "Asset1"], 1.5, delta=DEFAULT_DELTA
        )


if __name__ == "__main__":
    unittest.main()

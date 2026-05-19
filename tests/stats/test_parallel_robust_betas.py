"""Unit tests for parallel_robust_betas parity with robust_betas."""

import unittest

import numpy as np
import pandas as pd

from tests.stats.test_robust_betas import (
    STD_ASSET_RETS,
    STD_FACT_RETS,
    make_multi_asset_ret_frames,
    make_multi_factor_ret_frames,
    make_single_asset_ret_frames,
)
from vbase_utils.stats.parallel_robust_betas import parallel_robust_betas
from vbase_utils.stats.robust_betas import robust_betas


def _assert_parallel_matches_serial(
    df_asset_rets: pd.DataFrame,
    df_fact_rets: pd.DataFrame,
    **kwargs: object,
) -> None:
    """Assert parallel and serial robust beta functions return identical results."""
    serial = robust_betas(df_asset_rets, df_fact_rets, **kwargs)
    parallel = parallel_robust_betas(df_asset_rets, df_fact_rets, **kwargs)
    pd.testing.assert_frame_equal(serial, parallel, rtol=1e-9, atol=1e-9)


class TestParallelRobustBetas(unittest.TestCase):
    """Verify parallel_robust_betas matches robust_betas across representative inputs."""

    @classmethod
    def setUpClass(cls):
        """Set random seed and create common variables."""
        np.random.seed(42)
        cls.n_timestamps = 100
        cls.spy_returns = np.random.normal(0, STD_FACT_RETS, cls.n_timestamps)

    def test_matches_single_asset(self):
        """Single-asset betas match between parallel and serial."""
        df_asset_rets, df_fact_rets = make_single_asset_ret_frames(
            self.spy_returns, self.n_timestamps
        )
        _assert_parallel_matches_serial(df_asset_rets, df_fact_rets, half_life=30)

    def test_matches_multiple_assets(self):
        """Multi-asset betas match between parallel and serial."""
        df_asset_rets, df_fact_rets = make_multi_asset_ret_frames(
            self.spy_returns, self.n_timestamps
        )
        _assert_parallel_matches_serial(df_asset_rets, df_fact_rets, half_life=30)

    def test_matches_multiple_factors(self):
        """Multi-factor betas match between parallel and serial."""
        df_asset_rets, df_fact_rets = make_multi_factor_ret_frames(
            self.spy_returns, self.n_timestamps
        )
        _assert_parallel_matches_serial(df_asset_rets, df_fact_rets, half_life=30)

    def test_matches_lambda_parameter(self):
        """Lambda-weighted betas match between parallel and serial."""
        df_asset_rets, df_fact_rets = make_single_asset_ret_frames(
            self.spy_returns, self.n_timestamps
        )
        _assert_parallel_matches_serial(df_asset_rets, df_fact_rets, lambda_=0.985)

    def test_matches_outlier_heavy_data(self):
        """Outlier-heavy betas match between parallel and serial."""
        asset_returns = 1.5 * self.spy_returns + np.random.normal(
            0, STD_ASSET_RETS, self.n_timestamps
        )
        asset_returns[::10] += 0.1
        df_asset_rets = pd.DataFrame({"Asset1": asset_returns})
        df_fact_rets = pd.DataFrame({"SPY": self.spy_returns})
        _assert_parallel_matches_serial(df_asset_rets, df_fact_rets, half_life=30)

    def test_matches_insufficient_timestamps(self):
        """All-NaN betas match between parallel and serial when data is too short."""
        short_spy_returns = np.random.normal(0, STD_FACT_RETS, 5)
        short_asset_returns = 1.5 * short_spy_returns + np.random.normal(
            0, STD_ASSET_RETS, 5
        )
        df_asset_rets = pd.DataFrame({"Asset1": short_asset_returns})
        df_fact_rets = pd.DataFrame({"SPY": short_spy_returns})
        _assert_parallel_matches_serial(
            df_asset_rets, df_fact_rets, half_life=2, min_timestamps=10
        )

    def test_matches_nan_asset_returns(self):
        """Betas with leading NaN asset returns match between parallel and serial."""
        asset_returns = np.concatenate(
            [
                np.full(20, np.nan),
                1.5 * self.spy_returns[20:] + np.random.normal(0, STD_ASSET_RETS, 80),
            ]
        )
        df_asset_rets = pd.DataFrame({"Asset1": asset_returns})
        df_fact_rets = pd.DataFrame({"SPY": self.spy_returns})
        _assert_parallel_matches_serial(df_asset_rets, df_fact_rets, half_life=30)

    def test_shared_validation_errors(self):
        """Parallel path raises the same validation errors as serial (shared setup)."""
        df_asset_rets = pd.DataFrame()
        df_fact_rets = pd.DataFrame()
        with self.assertRaises(ValueError):
            parallel_robust_betas(df_asset_rets, df_fact_rets, half_life=30)

        asset_returns = 1.5 * self.spy_returns[:90] + np.random.normal(
            0, STD_ASSET_RETS, 90
        )
        df_asset_rets = pd.DataFrame({"Asset1": asset_returns})
        df_fact_rets = pd.DataFrame({"SPY": self.spy_returns})
        with self.assertRaises(ValueError):
            parallel_robust_betas(df_asset_rets, df_fact_rets, half_life=30)

        spy_constant = np.ones(self.n_timestamps) * 0.01
        asset_returns = 1.5 * spy_constant + np.random.normal(
            0, STD_ASSET_RETS, self.n_timestamps
        )
        df_asset_rets = pd.DataFrame({"Asset1": asset_returns})
        df_fact_rets = pd.DataFrame({"SPY": spy_constant})
        with self.assertRaises(ValueError):
            parallel_robust_betas(df_asset_rets, df_fact_rets, half_life=30)


if __name__ == "__main__":
    unittest.main()

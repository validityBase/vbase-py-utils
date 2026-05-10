"""Regression tests: pit_robust_betas(parallel=True) vs pit_robust_betas(parallel=False).

Both code paths use the same RLM + HuberT + exponential weighting algorithm.
Results must be numerically identical (within floating-point rounding) regardless
of whether the parallel or serial implementation is used.
"""

import unittest

import numpy as np
import pandas as pd
import pytz

from vbase_utils.stats.pit_robust_betas import pit_robust_betas

ET = pytz.timezone("America/New_York")


def _make_synthetic_returns(
    n: int = 300,
    true_beta: float = 0.7,
    noise_scale: float = 0.005,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic asset returns with a known linear factor structure.

    asset_ret[t] = 0.0001 + true_beta * factor_ret[t] + epsilon[t]
    Returns DataFrame with columns ['FACTOR', 'ASSET'] and tz-aware DatetimeIndex (ET).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n, freq="B", tz=ET)
    factor_ret = rng.normal(0, 0.01, n)
    epsilon = rng.normal(0, noise_scale, n)
    asset_ret = 0.0001 + true_beta * factor_ret + epsilon
    return pd.DataFrame({"FACTOR": factor_ret, "ASSET": asset_ret}, index=dates)


def _run_parallel(
    df: pd.DataFrame,
    min_ts: int = 50,
    half_life: float = 50.0,
    asset_cols: list[str] | None = None,
) -> dict:
    if asset_cols is None:
        asset_cols = ["ASSET"]
    return pit_robust_betas(
        df_asset_rets=df[asset_cols],
        df_fact_rets=df[["FACTOR"]],
        half_life=half_life,
        min_timestamps=min_ts,
        rebalance_time_index=df.index,
        parallel=True,
    )


def _run_serial(
    df: pd.DataFrame,
    min_ts: int = 50,
    half_life: float = 50.0,
    asset_cols: list[str] | None = None,
) -> dict:
    if asset_cols is None:
        asset_cols = ["ASSET"]
    return pit_robust_betas(
        df_asset_rets=df[asset_cols],
        df_fact_rets=df[["FACTOR"]],
        half_life=half_life,
        min_timestamps=min_ts,
        rebalance_time_index=df.index,
        parallel=False,
    )


class TestParallelVsSerialRegression(unittest.TestCase):
    """Verify parallel and serial implementations produce identical results."""

    @classmethod
    def setUpClass(cls):
        cls.df = _make_synthetic_returns(n=300, true_beta=0.7, seed=42)
        cls.parallel = _run_parallel(cls.df)
        cls.serial = _run_serial(cls.df)

    def test_output_keys_match(self):
        """Both return dicts with the same keys."""
        expected_keys = {
            "df_betas",
            "df_hedge_rets_by_fact",
            "df_hedge_rets",
            "df_asset_resids",
        }
        self.assertEqual(set(self.parallel.keys()), expected_keys)
        self.assertEqual(set(self.serial.keys()), expected_keys)

    def test_betas_close(self):
        """df_betas['ASSET'] values agree within atol=1e-6."""
        p = self.parallel["df_betas"]["ASSET"].dropna()
        s = self.serial["df_betas"]["ASSET"].dropna()
        np.testing.assert_allclose(
            p.values,
            s.values,
            atol=1e-6,
            rtol=0,
            err_msg="df_betas['ASSET'] diverges between parallel and serial",
        )

    def test_residuals_close(self):
        """df_asset_resids['ASSET'] values agree within atol=1e-6."""
        p = self.parallel["df_asset_resids"]["ASSET"].dropna()
        s = self.serial["df_asset_resids"]["ASSET"].dropna()
        np.testing.assert_allclose(
            p.values,
            s.values,
            atol=1e-6,
            rtol=0,
            err_msg="df_asset_resids['ASSET'] diverges between parallel and serial",
        )

    def test_hedge_returns_close(self):
        """df_hedge_rets['ASSET'] values agree within atol=1e-6."""
        p = self.parallel["df_hedge_rets"]["ASSET"].dropna()
        s = self.serial["df_hedge_rets"]["ASSET"].dropna()
        np.testing.assert_allclose(
            p.values,
            s.values,
            atol=1e-6,
            rtol=0,
            err_msg="df_hedge_rets['ASSET'] diverges between parallel and serial",
        )

    def test_index_shapes_match(self):
        """Both implementations return DataFrames with equal shape and index."""
        p_resids = self.parallel["df_asset_resids"]
        s_resids = self.serial["df_asset_resids"]
        self.assertEqual(p_resids.shape, s_resids.shape)
        pd.testing.assert_index_equal(p_resids.index, s_resids.index)

        p_betas = self.parallel["df_betas"]
        s_betas = self.serial["df_betas"]
        self.assertEqual(p_betas.shape, s_betas.shape)
        pd.testing.assert_index_equal(p_betas.index, s_betas.index)

    def test_multiple_assets(self):
        """With 3 assets, all betas and resids agree between parallel and serial."""
        rng = np.random.default_rng(99)
        n = 300
        dates = pd.date_range("2022-01-03", periods=n, freq="B", tz=ET)
        factor_ret = rng.normal(0, 0.01, n)
        df = pd.DataFrame(
            {
                "FACTOR": factor_ret,
                "ASSET": 0.0001 + 0.5 * factor_ret + rng.normal(0, 0.005, n),
                "ASSET2": 0.0002 + 0.8 * factor_ret + rng.normal(0, 0.005, n),
                "ASSET3": 0.0000 + 1.2 * factor_ret + rng.normal(0, 0.005, n),
            },
            index=dates,
        )
        asset_cols = ["ASSET", "ASSET2", "ASSET3"]
        parallel = _run_parallel(df, min_ts=50, half_life=50.0, asset_cols=asset_cols)
        serial = _run_serial(df, min_ts=50, half_life=50.0, asset_cols=asset_cols)

        for col in ["ASSET", "ASSET2", "ASSET3"]:
            p_betas = parallel["df_betas"][col].dropna()
            s_betas = serial["df_betas"][col].dropna()
            np.testing.assert_allclose(
                p_betas.values,
                s_betas.values,
                atol=1e-6,
                rtol=0,
                err_msg=f"df_betas['{col}'] diverges (multi-asset)",
            )
            p_resids = parallel["df_asset_resids"][col].dropna()
            s_resids = serial["df_asset_resids"][col].dropna()
            np.testing.assert_allclose(
                p_resids.values,
                s_resids.values,
                atol=1e-6,
                rtol=0,
                err_msg=f"df_asset_resids['{col}'] diverges (multi-asset)",
            )


if __name__ == "__main__":
    unittest.main()

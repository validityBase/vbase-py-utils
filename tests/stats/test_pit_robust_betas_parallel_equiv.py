"""Equivalence tests for the parallel path of pit_robust_betas.

The parallel=True path fans the per-asset RLM fits out across processes via
parallel_robust_betas (BLAS pinned per worker). Its output must be identical to
the parallel=False (serial robust_betas) path.

The gate attempts exact equality first; if cross-process floating-point
nondeterminism makes it flaky it can be relaxed to atol=1e-6 (the tolerance used
by tests/stats/test_compare_parallel_serial_betas.py).
"""

import unittest
import warnings

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from vbase_utils.stats.pit_robust_betas import pit_robust_betas


def _make_returns(
    n: int,
    betas: dict[str, list[float]],
    factors: list[str],
    seed: int,
    noise: float = 0.005,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build synthetic factor and asset returns with a known linear structure.

    Args:
        n: Number of timestamps.
        betas: Mapping asset name -> list of true betas (one per factor).
        factors: Factor column names.
        seed: RNG seed for determinism.
        noise: Idiosyncratic noise scale.

    Returns:
        Tuple ``(df_asset_rets, df_fact_rets)``.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    fact = {f: rng.normal(0, 0.01, n) for f in factors}
    df_fact = pd.DataFrame(fact, index=dates)
    assets = {}
    for asset, asset_betas in betas.items():
        signal = sum(b * df_fact[f].values for b, f in zip(asset_betas, factors))
        assets[asset] = 0.0001 + signal + rng.normal(0, noise, n)
    df_asset = pd.DataFrame(assets, index=dates)
    return df_asset, df_fact


def _assert_equiv(parallel: dict, serial: dict) -> None:
    """Assert parallel and serial result dicts match bit for bit.

    Exact equality holds because both paths pin BLAS to a single thread, so the
    LAPACK reduction order inside the IRLS loop is the same in the parent as in
    the workers. Without that pinning the two differ at ~1e-14 and the result
    also depends on the machine's ambient thread count, so this assertion is the
    regression gate for the pinning.
    """
    for key in ("df_betas", "df_asset_resids"):
        assert_frame_equal(parallel[key], serial[key], check_exact=True)


class TestPitParallelEquivalence(unittest.TestCase):
    """parallel=True must match parallel=False for df_betas and df_asset_resids."""

    def test_single_factor_multi_asset(self):
        """1-factor, 3-asset panel: parallel == serial."""
        df_asset, df_fact = _make_returns(
            n=120,
            betas={"A": [0.5], "B": [0.8], "C": [1.2]},
            factors=["FACTOR"],
            seed=42,
        )
        common = {
            "df_asset_rets": df_asset,
            "df_fact_rets": df_fact,
            "half_life": 50.0,
            "min_timestamps": 20,
            "rebalance_time_index": df_asset.index,
        }
        parallel = pit_robust_betas(**common, parallel=True)
        serial = pit_robust_betas(**common, parallel=False)
        _assert_equiv(parallel, serial)

    def test_two_factor(self):
        """2-factor panel: parallel == serial."""
        df_asset, df_fact = _make_returns(
            n=120,
            betas={"A": [0.5, -0.3], "B": [1.1, 0.4]},
            factors=["F1", "F2"],
            seed=7,
        )
        common = {
            "df_asset_rets": df_asset,
            "df_fact_rets": df_fact,
            "half_life": 60.0,
            "min_timestamps": 20,
            "rebalance_time_index": df_asset.index,
        }
        parallel = pit_robust_betas(**common, parallel=True)
        serial = pit_robust_betas(**common, parallel=False)
        _assert_equiv(parallel, serial)

    def test_two_factor_with_holes(self):
        """Panel with deleted rows: parallel == serial, bit for bit.

        Complete-case deletion is applied in both fit paths, so this is the gate
        on their agreeing about which rows a window admits. Both must weight the
        full window and then mask; pre-filtering rows in either path would change
        the multiply order and break exact equality.
        """
        df_asset, df_fact = _make_returns(
            n=120,
            betas={"A": [0.5, -0.3], "B": [1.1, 0.4]},
            factors=["F1", "F2"],
            seed=11,
        )
        # F2 keeps a different calendar: scattered missing prints, plus a late
        # start so the leading windows are missing the column entirely.
        df_fact.iloc[:25, 1] = np.nan
        df_fact.iloc[[40, 41, 77, 98], 1] = np.nan
        common = {
            "df_asset_rets": df_asset,
            "df_fact_rets": df_fact,
            "half_life": 60.0,
            "min_timestamps": 20,
            "rebalance_time_index": df_asset.index,
        }
        parallel = pit_robust_betas(**common, parallel=True)
        serial = pit_robust_betas(**common, parallel=False)
        _assert_equiv(parallel, serial)
        # Guard against the test passing because nothing was ever fit.
        self.assertGreater(parallel["df_betas"].notna().to_numpy().sum(), 0)

    def test_rebalance_subset(self):
        """A rebalance_time_index subset: parallel == serial."""
        df_asset, df_fact = _make_returns(
            n=120,
            betas={"A": [0.7], "B": [1.0]},
            factors=["FACTOR"],
            seed=123,
        )
        # Rebalance only every 5th timestamp, from index 40 onward.
        reb = df_asset.index[40::5]
        common = {
            "df_asset_rets": df_asset,
            "df_fact_rets": df_fact,
            "half_life": 40.0,
            "min_timestamps": 15,
            "rebalance_time_index": reb,
        }
        parallel = pit_robust_betas(**common, parallel=True)
        serial = pit_robust_betas(**common, parallel=False)
        _assert_equiv(parallel, serial)

    def test_fill_missing_betas_equiv(self):
        """fill_missing_betas=True: parallel == serial, including date-absent NaNs."""
        df_asset, df_fact = _make_returns(
            n=120,
            betas={"A": [0.6], "B": [0.9]},
            factors=["FACTOR"],
            seed=99,
        )
        # Asset B only has data in the second half of the window, so on early
        # rebalance dates B is dropped (all-NaN) and must remain NaN after the
        # per-date fill. This exercises the load-bearing per-date fill semantics.
        df_asset.loc[df_asset.index[:60], "B"] = np.nan
        common = {
            "df_asset_rets": df_asset,
            "df_fact_rets": df_fact,
            "half_life": 50.0,
            "min_timestamps": 20,
            "rebalance_time_index": df_asset.index,
            "fill_missing_betas": True,
        }
        parallel = pit_robust_betas(**common, parallel=True)
        serial = pit_robust_betas(**common, parallel=False)
        _assert_equiv(parallel, serial)

        # On the first rebalance date, B has no data and must stay NaN (not
        # filled to 1.0), while A is present.
        first_ts = df_asset.index[0]
        first_row = parallel["df_betas"].xs(first_ts, level="timestamp")
        self.assertTrue(first_row["B"].isna().all())

    def test_all_nan_asset_stays_nan(self):
        """An all-NaN asset column stays NaN in both paths."""
        df_asset, df_fact = _make_returns(
            n=120,
            betas={"A": [0.7], "DEAD": [0.0]},
            factors=["FACTOR"],
            seed=5,
        )
        df_asset["DEAD"] = np.nan
        common = {
            "df_asset_rets": df_asset,
            "df_fact_rets": df_fact,
            "half_life": 50.0,
            "min_timestamps": 20,
            "rebalance_time_index": df_asset.index,
        }
        parallel = pit_robust_betas(**common, parallel=True)
        serial = pit_robust_betas(**common, parallel=False)
        _assert_equiv(parallel, serial)
        self.assertTrue(parallel["df_betas"]["DEAD"].isna().all())

    def test_insufficient_timestamps(self):
        """Fewer than min_timestamps rows: all-NaN betas, no sim run, equal paths."""
        df_asset, df_fact = _make_returns(
            n=8,
            betas={"A": [0.7], "B": [1.0]},
            factors=["FACTOR"],
            seed=11,
        )
        common = {
            "df_asset_rets": df_asset,
            "df_fact_rets": df_fact,
            "half_life": 5.0,
            "min_timestamps": 10,
            "rebalance_time_index": df_asset.index,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parallel = pit_robust_betas(**common, parallel=True)
            serial = pit_robust_betas(**common, parallel=False)
        _assert_equiv(parallel, serial)
        self.assertTrue(parallel["df_betas"].isna().all().all())


if __name__ == "__main__":
    unittest.main()

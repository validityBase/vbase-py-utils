"""Unit tests for the pit_robust_betas function."""

import unittest

import numpy as np
import pandas as pd

from vbase_utils.stats import pit_robust_betas as pit_robust_betas_module
from vbase_utils.stats import robust_betas as robust_betas_module
from vbase_utils.stats.pit_robust_betas import pit_robust_betas

# Constants for test data generation
# Standard deviation of factor returns
STD_FACT_RETS = 0.01
# Standard deviation of asset returns
STD_ASSET_RETS = 0.005
# Default delta for floating point comparisons
DEFAULT_DELTA = 0.2


class _PanelFixture(unittest.TestCase):
    """Shared single-factor panel fixture. Holds no tests of its own."""

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


class TestPitRobustBetas(_PanelFixture):
    """Unit tests for the pit_robust_betas function."""

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

    def test_carry_forward_preserves_factor_identity(self):
        """Carried-forward betas stay on their own factor.

        df_betas is indexed by (timestamp, factor), so a plain ffill(axis=0)
        walks the flattened rows and fills the first factor of a carry-forward
        date from the LAST factor of the previous rebalance date -- silently
        swapping betas across factors. The fill must be per factor.
        """
        # Two factors with clearly separated betas, so a cross-factor fill is
        # unambiguous rather than a near-miss on noisy estimates.
        iwm_returns = pd.Series(
            np.random.normal(0, STD_FACT_RETS, self.n_timestamps),
            index=self.dates,
            name="IWM",
        )
        df_fact_rets = pd.DataFrame({"SPY": self.spy_returns, "IWM": iwm_returns})
        asset_rets = (
            2.0 * self.spy_returns
            + 0.2 * iwm_returns
            + np.random.normal(0, STD_ASSET_RETS, self.n_timestamps)
        )
        df_asset_rets = pd.DataFrame({"Asset1": asset_rets}, index=self.dates)

        # Sparse rebalance: most dates carry forward, exercising the fill.
        rebalance_dates = pd.DatetimeIndex([self.dates[49], self.dates[74]])
        df_betas = pit_robust_betas(
            df_asset_rets,
            df_fact_rets,
            half_life=30,
            rebalance_time_index=rebalance_dates,
        )["df_betas"]

        # Every date between two rebalances must repeat that rebalance's betas
        # factor for factor.
        for rebalance_date, next_date in (
            (self.dates[49], self.dates[74]),
            (self.dates[74], self.dates[-1] + pd.Timedelta(days=1)),
        ):
            expected = df_betas.xs(rebalance_date, level="timestamp")
            carried = self.dates[
                (self.dates > rebalance_date) & (self.dates < next_date)
            ]
            self.assertGreater(len(carried), 0)
            for timestamp in carried:
                pd.testing.assert_frame_equal(
                    df_betas.xs(timestamp, level="timestamp"), expected
                )

        # Guard the estimates themselves: a factor swap would also show up as
        # SPY and IWM trading values.
        betas_at_rebalance = df_betas.xs(self.dates[49], level="timestamp")
        self.assertAlmostEqual(
            betas_at_rebalance.loc["SPY", "Asset1"], 2.0, delta=DEFAULT_DELTA
        )
        self.assertAlmostEqual(
            betas_at_rebalance.loc["IWM", "Asset1"], 0.2, delta=DEFAULT_DELTA
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

    def test_rebalance_before_data_does_not_crash(self):
        """Test that no crash occurs when rebalance timestamps precede all data."""
        # All rebalance dates are before the data starts (2023-01-01),
        # so sim() skips the callback for every timestamp and returns {}.
        # Previously caused KeyError: 'betas' at sim_results["betas"].
        early_dates = pd.date_range("2022-01-01", periods=5)

        results = pit_robust_betas(
            self.df_asset_rets,
            self.df_fact_rets,
            half_life=30,
            rebalance_time_index=early_dates,
        )

        self.assertIn("df_betas", results)
        self.assertTrue(results["df_betas"].isna().all().all())

    def test_fill_missing_betas(self):
        """fill_missing_betas=True replaces NaN betas with 1.0 when valid betas exist."""
        n = 60
        dates = pd.date_range("2023-01-01", periods=n)
        np.random.seed(99)
        spy_rets = pd.Series(np.random.normal(0, STD_FACT_RETS, n), index=dates)
        df_fact_rets = pd.DataFrame({"SPY": spy_rets})

        # Asset1: always valid, beta ~1.5
        asset1 = 1.5 * spy_rets + np.random.normal(0, STD_ASSET_RETS, n)
        # Asset2: NaN for first 40 timestamps → only 1-9 valid observations at timestamps 40-48
        asset2_vals = np.full(n, np.nan)
        asset2_vals[40:] = 0.8 * spy_rets.values[40:] + np.random.normal(
            0, STD_ASSET_RETS, 20
        )
        asset2 = pd.Series(asset2_vals, index=dates)
        df_asset_rets = pd.DataFrame({"Asset1": asset1, "Asset2": asset2})

        # Without fill: timestamps 40-48 (Asset2 has <10 valid obs) → Asset2 beta is NaN
        results_no_fill = pit_robust_betas(
            df_asset_rets,
            df_fact_rets,
            half_life=30,
            min_timestamps=10,
            fill_missing_betas=False,
        )
        mid_date = dates[45]  # Asset2 has 6 valid obs here, insufficient
        betas_no_fill = results_no_fill["df_betas"].xs(mid_date)
        self.assertTrue(np.isnan(betas_no_fill.loc["SPY", "Asset2"]))

        # With fill: Asset2 NaN betas are replaced with 1.0
        results_with_fill = pit_robust_betas(
            df_asset_rets,
            df_fact_rets,
            half_life=30,
            min_timestamps=10,
            fill_missing_betas=True,
        )
        betas_with_fill = results_with_fill["df_betas"].xs(mid_date)
        self.assertEqual(betas_with_fill.loc["SPY", "Asset2"], 1.0)
        # Asset1 betas should still be real (not 1.0)
        self.assertAlmostEqual(
            betas_with_fill.loc["SPY", "Asset1"], 1.5, delta=DEFAULT_DELTA
        )

    def test_hedge_returns_use_same_factor_from_prior_date(self):
        """Hedge weights must come from the SAME factor one timestamp back.

        df_betas is indexed by (timestamp, factor), so a positional shift(1)
        hedges a date's first factor with the previous date's LAST factor.
        Single-factor panels cannot detect that (one row per timestamp), so this
        exercises two factors and asserts the exact identity

            df_hedge_rets_by_fact[t, f] == -1 * df_betas[t-1, f] * fact_rets[t, f]

        at every timestamp and factor, with an exact tolerance rather than
        DEFAULT_DELTA -- the cross-factor error is far smaller than 0.2 and a
        loose delta hides it.
        """
        iwm_returns = pd.Series(
            np.random.normal(0, STD_FACT_RETS, self.n_timestamps),
            index=self.dates,
            name="IWM",
        )
        df_fact_rets = pd.DataFrame({"SPY": self.spy_returns, "IWM": iwm_returns})
        # Betas differ strongly across factors, so a cross-factor swap is visible.
        df_asset_rets = pd.DataFrame(
            {
                "Asset1": 2.5 * self.spy_returns
                - 1.5 * iwm_returns
                + np.random.normal(0, STD_ASSET_RETS, self.n_timestamps)
            },
            index=self.dates,
        )

        results = pit_robust_betas(df_asset_rets, df_fact_rets, half_life=30)
        betas = results["df_betas"]["Asset1"]
        hedge_rets = results["df_hedge_rets_by_fact"]["Asset1"]

        checked = 0
        for i in range(1, len(self.dates)):
            for factor in ("SPY", "IWM"):
                prev_beta = betas.loc[(self.dates[i - 1], factor)]
                if np.isnan(prev_beta):
                    continue
                self.assertAlmostEqual(
                    hedge_rets.loc[(self.dates[i], factor)],
                    -1 * prev_beta * df_fact_rets.loc[self.dates[i], factor],
                    places=12,
                    msg=(
                        f"hedge return at {self.dates[i]} for {factor} does not "
                        f"use that factor's beta from {self.dates[i - 1]}"
                    ),
                )
                checked += 1
        # Guard against the loop silently checking nothing.
        self.assertGreater(checked, 100)

    def test_hedge_returns_shift_matches_single_factor_case(self):
        """A two-factor panel whose second factor is unused must hedge like K=1.

        Cross-checks the per-factor shift from a different angle: adding a
        factor the asset has no exposure to must not disturb the first factor's
        hedge returns, which a positional shift does by rotating betas.
        """
        results_k1 = pit_robust_betas(
            self.df_asset_rets, self.df_fact_rets, half_life=30
        )
        hedge_k1 = results_k1["df_hedge_rets_by_fact"].xs("SPY", level="factor")[
            "Asset1"
        ]

        # Same SPY exposure, plus an independent factor with zero true beta.
        noise_returns = pd.Series(
            np.random.normal(0, STD_FACT_RETS, self.n_timestamps),
            index=self.dates,
            name="NOISE",
        )
        df_fact_rets_k2 = pd.DataFrame(
            {"SPY": self.spy_returns, "NOISE": noise_returns}
        )
        results_k2 = pit_robust_betas(self.df_asset_rets, df_fact_rets_k2, half_life=30)
        hedge_k2 = results_k2["df_hedge_rets_by_fact"].xs("SPY", level="factor")[
            "Asset1"
        ]

        # The SPY betas shift by one timestamp in both runs, so the SPY hedge
        # returns must track each other closely. A positional shift makes the
        # K=2 run use NOISE's beta here, which breaks the relationship.
        common = hedge_k1.dropna().index.intersection(hedge_k2.dropna().index)
        self.assertGreater(len(common), 50)
        np.testing.assert_allclose(
            hedge_k2.loc[common].to_numpy(),
            hedge_k1.loc[common].to_numpy(),
            rtol=0.05,
            atol=1e-4,
            err_msg="adding an unrelated factor changed the SPY hedge returns",
        )


class TestPitRobustBetasFactorCoverage(_PanelFixture):
    """Behaviour when assets or factors are absent from part of the panel.

    The point-in-time mask drops all-NaN columns, so an asset that has not
    listed and a factor whose history has not started both disappear from the
    window. These cover what the betas and hedge totals do in those cases.
    """

    def _staggered_panel(self, first_listing):
        """Asset panel whose assets all start at ``first_listing``, factors from 0.

        Aligning a short asset history onto a longer factor index leaves leading
        all-NaN asset rows, which sim() strips with dropna(axis=1, how="all").
        """
        asset_vals = np.full((self.n_timestamps, 2), np.nan)
        asset_vals[first_listing:, 0] = (
            1.5 * self.spy_returns.to_numpy()[first_listing:]
        ) + np.random.normal(0, STD_ASSET_RETS, self.n_timestamps - first_listing)
        asset_vals[first_listing:, 1] = (
            0.8 * self.spy_returns.to_numpy()[first_listing:]
        ) + np.random.normal(0, STD_ASSET_RETS, self.n_timestamps - first_listing)
        return pd.DataFrame(asset_vals, index=self.dates, columns=["Asset1", "Asset2"])

    def test_no_live_assets_at_early_dates_does_not_raise(self):
        """Leading dates where no asset has data yield NaN betas, not an error.

        sim() drops asset columns that are all-NaN over the current window, so a
        panel whose assets list after the factors presents the callback with a
        (rows, 0) asset frame at its leading dates. That must degrade to no
        betas for those dates -- as the insufficient-timestamps and non-finite-
        factor cases do -- rather than aborting the whole simulation.
        """
        first_listing = 20
        df_asset_rets = self._staggered_panel(first_listing)

        results = pit_robust_betas(df_asset_rets, self.df_fact_rets, half_life=30)
        df_betas = results["df_betas"]

        # Dates before any asset lists carry no betas.
        self.assertTrue(
            df_betas.loc[: self.dates[first_listing - 1]].isna().all().all()
        )
        # Betas appear once assets have enough observations, and are sane.
        last_betas = df_betas.xs(self.dates[-1])
        self.assertAlmostEqual(
            last_betas.loc["SPY", "Asset1"], 1.5, delta=DEFAULT_DELTA
        )
        self.assertAlmostEqual(
            last_betas.loc["SPY", "Asset2"], 0.8, delta=DEFAULT_DELTA
        )

    def test_no_live_assets_matches_trimmed_rebalance_index(self):
        """Skipping the empty leading dates must not perturb later betas.

        Running over the whole index must give the same betas on the dates a run
        that rebalances only after the listings also covers -- i.e. the skipped
        dates contribute nothing rather than shifting the estimates.
        """
        first_listing = 20
        df_asset_rets = self._staggered_panel(first_listing)

        full = pit_robust_betas(df_asset_rets, self.df_fact_rets, half_life=30)
        trimmed = pit_robust_betas(
            df_asset_rets,
            self.df_fact_rets,
            half_life=30,
            rebalance_time_index=self.dates[first_listing:],
        )
        covered = self.dates[first_listing:]
        np.testing.assert_array_equal(
            full["df_betas"].loc[covered].to_numpy(),
            trimmed["df_betas"].loc[covered].to_numpy(),
        )

    def test_no_live_assets_warns_once(self):
        """The no-live-assets warning is emitted once, not once per date."""
        df_asset_rets = self._staggered_panel(20)
        # The flag is process-wide; clear it so the count is meaningful here.
        # pylint: disable=protected-access
        robust_betas_module._NO_ASSET_COLUMNS_WARNED["warned"] = False

        with self.assertLogs(
            "vbase_utils.stats.robust_betas", level="WARNING"
        ) as captured:
            pit_robust_betas(df_asset_rets, self.df_fact_rets, half_life=30)

        no_asset_warnings = [
            record
            for record in captured.records
            if "No asset has any data" in record.getMessage()
        ]
        self.assertEqual(len(no_asset_warnings), 1)

    def _late_factor_panel(self, factor_start):
        """Panel whose second factor starts late, so it drops out of early windows.

        sim() masks point-in-time and removes all-NaN columns, so a factor whose
        history has not begun is absent from the early windows entirely. Those
        windows now yield no betas at all rather than a smaller model on the
        remaining factors.
        """
        iwm_vals = np.random.normal(0, STD_FACT_RETS, self.n_timestamps)
        asset_vals = (
            1.4 * self.spy_returns.to_numpy()
            + 0.9 * iwm_vals
            + np.random.normal(0, STD_ASSET_RETS, self.n_timestamps)
        )
        iwm_col = iwm_vals.copy()
        iwm_col[:factor_start] = np.nan
        df_fact_rets = pd.DataFrame(
            {"SPY": self.spy_returns.to_numpy(), "IWM": iwm_col}, index=self.dates
        )
        df_asset_rets = pd.DataFrame({"Asset1": asset_vals}, index=self.dates)
        return df_asset_rets, df_fact_rets

    def _scattered_holes_panel(self, hole_positions):
        """Two-factor panel where the second factor is missing on scattered dates.

        Models factors that keep different calendars: the factor exists over the
        whole sample but has no print on the other's market holidays.
        """
        iwm_vals = np.random.normal(0, STD_FACT_RETS, self.n_timestamps)
        asset_vals = (
            1.4 * self.spy_returns.to_numpy()
            + 0.9 * iwm_vals
            + np.random.normal(0, STD_ASSET_RETS, self.n_timestamps)
        )
        iwm_col = iwm_vals.copy()
        iwm_col[hole_positions] = np.nan
        df_fact_rets = pd.DataFrame(
            {"SPY": self.spy_returns.to_numpy(), "IWM": iwm_col}, index=self.dates
        )
        df_asset_rets = pd.DataFrame({"Asset1": asset_vals}, index=self.dates)
        return df_asset_rets, df_fact_rets

    def test_window_missing_a_factor_produces_no_betas_for_any_factor(self):
        """A factor with no data yet blocks the window instead of shrinking the model.

        Before its history starts the factor's column is absent from the window
        entirely, so there is no NaN row to delete. Fitting the remaining factors
        would silently estimate a different model than the caller asked for, so
        the window yields nothing and every factor's beta stays NaN.
        """
        df_asset_rets, df_fact_rets = self._late_factor_panel(60)

        results = pit_robust_betas(df_asset_rets, df_fact_rets, half_life=30)
        probe = self.dates[45]

        for factor in ("SPY", "IWM"):
            self.assertTrue(
                np.isnan(results["df_betas"].loc[(probe, factor), "Asset1"]),
                f"{factor} must have no beta while another factor is absent",
            )
        # Nothing to hedge with, so the date carries no hedge return or residual.
        self.assertTrue(np.isnan(results["df_hedge_rets"].loc[probe, "Asset1"]))
        self.assertTrue(np.isnan(results["df_asset_resids"].loc[probe, "Asset1"]))

    def test_betas_resume_once_every_factor_has_enough_data(self):
        """Betas start once the late factor has min_timestamps complete rows.

        Complete-case deletion removes every pre-listing row from the window, so
        the usable sample restarts at the listing date and betas appear once it
        reaches min_timestamps.
        """
        factor_start = 60
        min_timestamps = 10
        df_asset_rets, df_fact_rets = self._late_factor_panel(factor_start)

        results = pit_robust_betas(
            df_asset_rets,
            df_fact_rets,
            half_life=30,
            min_timestamps=min_timestamps,
        )
        betas = results["df_betas"]

        # One row short of the minimum: still nothing.
        too_early = self.dates[factor_start + min_timestamps - 2]
        self.assertTrue(np.isnan(betas.loc[(too_early, "SPY"), "Asset1"]))
        # The first date with enough complete rows produces betas for both factors.
        first_fit = self.dates[factor_start + min_timestamps - 1]
        for factor in ("SPY", "IWM"):
            self.assertFalse(
                np.isnan(betas.loc[(first_fit, factor), "Asset1"]),
                f"{factor} must have a beta once every factor has data",
            )

    def test_late_factor_does_not_freeze_the_other_factors(self):
        """A late-listing factor must not stop estimation for the whole run.

        Regression test for the defect this replaced: once the late factor's first
        value arrived, its column survived the point-in-time all-NaN drop but the
        window still held every earlier NaN row. The window was discarded on that
        basis, and because the window only grows those rows never aged out -- so
        every later window was discarded too. The other factors' betas froze at
        their last pre-listing values and the late factor never got a beta at all.
        """
        factor_start = 60
        df_asset_rets, df_fact_rets = self._late_factor_panel(factor_start)

        results = pit_robust_betas(df_asset_rets, df_fact_rets, half_life=30)
        betas = results["df_betas"]
        after = self.dates[factor_start + 15 :]

        spy_after = betas.xs("SPY", level="factor").loc[after, "Asset1"].dropna()
        iwm_after = betas.xs("IWM", level="factor").loc[after, "Asset1"].dropna()

        # Both factors are estimated after the late one lists.
        self.assertGreater(len(spy_after), 20)
        self.assertGreater(len(iwm_after), 20)
        # And the estimates keep moving rather than being carried forward from a
        # single frozen date. Under the old behavior each of these was 1.
        self.assertGreater(spy_after.nunique(), 20)
        self.assertGreater(iwm_after.nunique(), 20)
        # The betas recover the panel's true coefficients.
        self.assertAlmostEqual(spy_after.iloc[-1], 1.4, delta=DEFAULT_DELTA)
        self.assertAlmostEqual(iwm_after.iloc[-1], 0.9, delta=DEFAULT_DELTA)

    def test_scattered_factor_holes_do_not_stop_estimation(self):
        """Isolated missing factor dates cost those rows, not the window.

        The old behavior discarded any window containing a non-finite factor
        value, so a factor keeping a different calendar from the others froze
        estimation from its first missing print onward.
        """
        holes = [12, 33, 34, 57, 81]
        df_asset_rets, df_fact_rets = self._scattered_holes_panel(holes)

        results = pit_robust_betas(df_asset_rets, df_fact_rets, half_life=30)
        spy = results["df_betas"].xs("SPY", level="factor")["Asset1"].dropna()
        iwm = results["df_betas"].xs("IWM", level="factor")["Asset1"].dropna()

        # Estimation continues past every hole, including the last one.
        self.assertIn(self.dates[-1], spy.index)
        self.assertGreater(spy.nunique(), 50)
        self.assertGreater(iwm.nunique(), 50)
        self.assertAlmostEqual(spy.iloc[-1], 1.4, delta=DEFAULT_DELTA)
        self.assertAlmostEqual(iwm.iloc[-1], 0.9, delta=DEFAULT_DELTA)

    def test_incomplete_factor_window_warns_once(self):
        """The missing-factor condition warns once per process, not per date."""
        df_asset_rets, df_fact_rets = self._late_factor_panel(60)

        module = "vbase_utils.stats.pit_robust_betas"
        # pylint: disable=protected-access
        pit_robust_betas_module._INCOMPLETE_FACTORS_WARNED["warned"] = False
        with self.assertLogs(module, level="WARNING") as captured:
            pit_robust_betas(df_asset_rets, df_fact_rets, half_life=30)

        warnings = [
            record
            for record in captured.records
            if "factor(s) have no data on or before" in record.getMessage()
        ]
        self.assertEqual(len(warnings), 1)

    def test_factor_with_no_data_anywhere_raises(self):
        """A factor that is NaN across the whole panel is a caller error.

        Under the all-factors rule such a column would make every window
        incomplete, silently voiding the entire run, so it fails up front.
        """
        df_asset_rets, df_fact_rets = self._late_factor_panel(60)
        df_fact_rets["DEAD"] = np.nan

        with self.assertRaises(ValueError) as ctx:
            pit_robust_betas(df_asset_rets, df_fact_rets, half_life=30)
        self.assertIn("DEAD", str(ctx.exception))

    def test_flat_factor_across_the_panel_raises(self):
        """A factor that never moves is collinear with the intercept."""
        df_asset_rets, df_fact_rets = self._late_factor_panel(60)
        df_fact_rets["FLAT"] = 0.01

        with self.assertRaises(ValueError) as ctx:
            pit_robust_betas(df_asset_rets, df_fact_rets, half_life=30)
        self.assertIn("FLAT", str(ctx.exception))

    def test_duplicate_asset_names_raise(self):
        """Repeated asset names make the betas panel ambiguous.

        The panel is filled through {label: position} maps, which keep only a
        duplicate's last occurrence, so one asset's betas would be silently lost.
        """
        df_asset_rets, df_fact_rets = self._scattered_holes_panel([10])
        df_asset_rets["Asset1_copy"] = df_asset_rets["Asset1"]
        df_asset_rets.columns = ["Asset1", "Asset1"]

        with self.assertRaises(ValueError) as ctx:
            pit_robust_betas(df_asset_rets, df_fact_rets, half_life=30)
        self.assertIn("df_asset_rets", str(ctx.exception))
        self.assertIn("Asset1", str(ctx.exception))

    def test_duplicate_factor_names_raise(self):
        """Repeated factor names silently produced a wrong beta before this.

        The unwritten factor row was forward-filled from an earlier window, so a
        stale beta flowed into df_hedge_rets and df_asset_resids as an ordinary
        number -- no NaN and no warning to signal it.
        """
        df_asset_rets, df_fact_rets = self._scattered_holes_panel([10])
        df_fact_rets.columns = ["SPY", "SPY"]

        with self.assertRaises(ValueError) as ctx:
            pit_robust_betas(df_asset_rets, df_fact_rets, half_life=30)
        self.assertIn("df_fact_rets", str(ctx.exception))
        self.assertIn("SPY", str(ctx.exception))

    def test_duplicate_rebalance_timestamps_raise(self):
        """Repeated rebalance dates are rejected with a message naming the input.

        This already failed, but as "cannot handle a non-unique multi-index!"
        raised from inside pandas on the later reindex, with nothing pointing at
        the offending argument.
        """
        df_asset_rets, df_fact_rets = self._scattered_holes_panel([10])
        reb = self.dates[[30, 40, 40, 50]]

        with self.assertRaises(ValueError) as ctx:
            pit_robust_betas(
                df_asset_rets, df_fact_rets, half_life=30, rebalance_time_index=reb
            )
        self.assertIn("rebalance_time_index", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

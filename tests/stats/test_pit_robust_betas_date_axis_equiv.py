"""Equivalence tests for the date-axis parallel path of pit_robust_betas.

``parallel_axis="date"`` replaces the ``sim()`` loop outright rather than
parallelizing inside it, so the two paths derive the same window rules from
different data: ``sim()`` masks and drops all-NaN columns per date, while
``_parallel_betas_by_date.precompute_date_betas`` answers every check once from cumulative row-level
facts. A divergence between them does not crash -- it silently returns different
betas -- so this suite is the gate, and it asserts bit-identity of **all four**
returned frames rather than just the betas.

Each test is named for the rule it pins. The cases were chosen by reading the two
paths against each other, not by coverage: they are the points where the same
question is answered by different code.
"""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from vbase_utils.stats.pit_robust_betas import pit_robust_betas

from ._robust_betas_fixtures import make_linear_ret_frames as _make_returns

# Worker count for the parallel runs. More than one so the pool really forks
# (n_jobs=1 takes joblib's sequential backend, which is covered separately).
N_JOBS = 2


def _make_hard_panel(n: int = 160, seed: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Panel exercising every branch the clean one does not.

    A clean panel has complete factors live from row 0, so listwise deletion
    never fires and the factor-liveness gate never fires -- exactly the two
    branches where the date axis is most likely to diverge. This adds scattered
    non-finite factor rows, a factor that lists a third of the way in, an asset
    that never lists at all and an asset with too few rows to ever be fit.
    """
    df_asset, df_fact = _make_returns(
        n=n,
        betas={
            "A": [0.6, -0.2],
            "B": [1.0, 0.5],
            "DEAD": [0.3, 0.3],
            "THIN": [0.9, 0.1],
        },
        factors=["F1", "F2"],
        seed=seed,
    )
    rng = np.random.default_rng(seed + 1)
    # Scattered incomplete factor rows -> listwise deletion.
    df_fact.iloc[[7, 31, 62, 95, 130], 0] = np.nan
    # An inf, not just a NaN: complete_rows uses isfinite, the all-NaN column
    # drop uses notna, and the two must not be conflated.
    df_fact.iloc[44, 1] = np.inf
    # F2 lists a third of the way in -> no betas at all before it is live.
    df_fact.iloc[: n // 3, 1] = np.nan
    # An asset with no data anywhere, and one with a handful of late prints.
    df_asset["DEAD"] = np.nan
    df_asset["THIN"] = np.nan
    df_asset.iloc[-4:, df_asset.columns.get_loc("THIN")] = rng.normal(0, 0.01, 4)
    return df_asset, df_fact


def _assert_equiv(date_axis: dict, serial: dict) -> None:
    """Assert the date-axis and serial result dicts match bit for bit.

    All four frames, not just the betas: ``df_hedge_rets_by_fact`` and
    ``df_hedge_rets`` are where a beta that is NaN on one path and a number on
    the other stops being visible as a NaN and starts being an ordinary-looking
    return.
    """
    assert set(date_axis) == set(serial)
    for key in sorted(serial):
        assert_frame_equal(date_axis[key], serial[key], check_exact=True)


def _run_both(common: dict, **date_kwargs) -> tuple[dict, dict]:
    """Run the date axis and the serial reference over the same inputs."""
    serial = pit_robust_betas(**common, parallel=False)
    date_axis = pit_robust_betas(
        **common,
        parallel=True,
        parallel_axis="date",
        n_jobs=date_kwargs.pop("n_jobs", N_JOBS),
        **date_kwargs,
    )
    return date_axis, serial


class TestDateAxisEquivalence(unittest.TestCase):
    """parallel_axis="date" must match parallel=False, bit for bit."""

    def test_single_factor_multi_asset(self):
        """Clean 1-factor panel: the baseline shape."""
        df_asset, df_fact = _make_returns(
            n=120, betas={"A": [0.5], "B": [0.8], "C": [1.2]}, factors=["F"], seed=42
        )
        common = {
            "df_asset_rets": df_asset,
            "df_fact_rets": df_fact,
            "half_life": 50.0,
            "min_timestamps": 20,
        }
        _assert_equiv(*_run_both(common))

    def test_multi_factor(self):
        """K > 1 pins the cross-factor row ordering in betas_buf.

        The buffer is addressed as ``row * n_facts + factor_position``. With one
        factor every ordering agrees, so this is the only shape that can catch a
        transposed or rotated write -- the same class of defect the per-factor
        ffill and shift already had.
        """
        df_asset, df_fact = _make_returns(
            n=140,
            betas={"A": [0.5, -0.3, 0.1], "B": [1.1, 0.4, -0.6]},
            factors=["F1", "F2", "F3"],
            seed=7,
        )
        common = {
            "df_asset_rets": df_asset,
            "df_fact_rets": df_fact,
            "half_life": 60.0,
            "min_timestamps": 20,
        }
        _assert_equiv(*_run_both(common))

    def test_single_asset_single_factor(self):
        """Degenerate shapes: one column on each side."""
        df_asset, df_fact = _make_returns(
            n=90, betas={"ONLY": [0.7]}, factors=["F"], seed=13
        )
        common = {
            "df_asset_rets": df_asset,
            "df_fact_rets": df_fact,
            "half_life": 40.0,
            "min_timestamps": 15,
        }
        _assert_equiv(*_run_both(common))

    def test_hard_panel(self):
        """Listwise deletion, a late factor, a dead asset and a thin asset."""
        df_asset, df_fact = _make_hard_panel()
        common = {
            "df_asset_rets": df_asset,
            "df_fact_rets": df_fact,
            "half_life": 60.0,
            "min_timestamps": 20,
        }
        _assert_equiv(*_run_both(common))

    def test_late_listing_factor(self):
        """A factor that lists late yields no betas until it is live.

        sim() drops the all-NaN factor column, so the callback sees fewer factors
        than the caller asked for and returns nothing; the date path answers the
        same question as ``i < all_facts_live_from``. Getting this wrong produces
        betas from a smaller model that look entirely reasonable.
        """
        df_asset, df_fact = _make_returns(
            n=130,
            betas={"A": [0.5, -0.4], "B": [0.9, 0.2]},
            factors=["F1", "F2"],
            seed=17,
        )
        df_fact.iloc[:55, 1] = np.nan
        common = {
            "df_asset_rets": df_asset,
            "df_fact_rets": df_fact,
            "half_life": 45.0,
            "min_timestamps": 15,
        }
        date_axis, serial = _run_both(common)
        _assert_equiv(date_axis, serial)
        # The gate really fired, so the equality above is not vacuous.
        early = serial["df_betas"].loc[: df_fact.index[54]]
        self.assertTrue(early.isna().all().all())

    def test_rebalance_timestamp_before_panel_start(self):
        """A rebalance timestamp earlier than every panel row.

        searchsorted lands it at position -1, which addresses the *last* panel
        row while slicing a zero-length window; unguarded, that fits a
        (0, n_facts+1) design, which pinv answers with zeros. The reindex onto
        the panel's timestamps drops the row before it reaches the output, so
        this is a regression test rather than a reproduction: it pins that the
        row stays absent and that nothing else shifts.
        """
        df_asset, df_fact = _make_returns(
            n=80, betas={"A": [0.6], "B": [1.3]}, factors=["F"], seed=23
        )
        rebalance = pd.DatetimeIndex([pd.Timestamp("2019-01-02")]).append(
            df_asset.index
        )
        common = {
            "df_asset_rets": df_asset,
            "df_fact_rets": df_fact,
            "half_life": 30.0,
            "min_timestamps": 10,
            "rebalance_time_index": rebalance,
        }
        date_axis, serial = _run_both(common)
        _assert_equiv(date_axis, serial)
        self.assertNotIn(
            pd.Timestamp("2019-01-02"),
            date_axis["df_betas"].index.get_level_values("timestamp"),
        )

    def test_rebalance_timestamps_absent_from_panel_index(self):
        """Rebalance timestamps that fall between panel rows.

        sim() masks with ``searchsorted(side="right")`` and the date path uses
        the same call, so both must land on the last panel row at or before the
        timestamp -- not on the nearest one, and not on the next.
        """
        df_asset, df_fact = _make_returns(
            n=100, betas={"A": [0.5], "B": [0.9]}, factors=["F"], seed=29
        )
        # Weekend dates: never in a business-day index, always strictly between
        # two panel rows.
        rebalance = pd.DatetimeIndex(
            [ts + pd.Timedelta(days=1) for ts in df_asset.index[40:60:3]]
        )
        common = {
            "df_asset_rets": df_asset,
            "df_fact_rets": df_fact,
            "half_life": 30.0,
            "min_timestamps": 10,
            "rebalance_time_index": rebalance,
        }
        _assert_equiv(*_run_both(common))

    def test_rebalance_index_strict_subset(self):
        """A sparse rebalance index pins row addressing in the buffer.

        With every timestamp a rebalance date the buffer row equals the panel
        row, so an off-by-one in either is invisible. A subset separates them.
        """
        df_asset, df_fact = _make_returns(
            n=120,
            betas={"A": [0.5, 0.2], "B": [0.9, -0.7]},
            factors=["F1", "F2"],
            seed=31,
        )
        common = {
            "df_asset_rets": df_asset,
            "df_fact_rets": df_fact,
            "half_life": 40.0,
            "min_timestamps": 15,
            "rebalance_time_index": df_asset.index[30::7],
        }
        _assert_equiv(*_run_both(common))

    def test_min_timestamps_boundary(self):
        """Assets with exactly min_timestamps - 1 and exactly min_timestamps rows.

        The asset path counts ``isfinite(y) & complete_rows`` inside the worker;
        the date path reads a cumulative count of the same intersection. The
        boundary is the only place a >= / > confusion shows up.
        """
        n, min_ts = 60, 12
        df_asset, df_fact = _make_returns(
            n=n,
            betas={"SHORT": [0.5], "EXACT": [0.9], "FULL": [1.1]},
            factors=["F"],
            seed=37,
        )
        df_asset.iloc[: n - (min_ts - 1), df_asset.columns.get_loc("SHORT")] = np.nan
        df_asset.iloc[: n - min_ts, df_asset.columns.get_loc("EXACT")] = np.nan
        common = {
            "df_asset_rets": df_asset,
            "df_fact_rets": df_fact,
            "half_life": 30.0,
            "min_timestamps": min_ts,
        }
        date_axis, serial = _run_both(common)
        _assert_equiv(date_axis, serial)
        # The boundary is where it was intended: EXACT is fit on the last date,
        # SHORT never is.
        last = serial["df_betas"].xs(df_asset.index[-1], level="timestamp")
        self.assertFalse(np.isnan(last["EXACT"]).all())
        self.assertTrue(np.isnan(last["SHORT"]).all())

    def test_factor_flat_over_a_prefix_raises_on_both_paths(self):
        """A factor flat over a prefix must raise on both paths, not one.

        The serial path's whole-column variance check raises *before* the
        min-timestamps gate and before the complete-row variance check. Scoring
        it only over the dates those two admit -- which is what the reference
        prototype did -- lets the flat prefix skip the raise entirely and return
        quietly-NaN early dates instead of failing the run.
        """
        df_asset, df_fact = _make_returns(
            n=120, betas={"A": [0.5], "B": [0.9]}, factors=["F"], seed=41
        )
        # Constant, not absent: the column is live from row 0, so the liveness
        # gate does not fire and the variance check is what must catch it.
        df_fact.iloc[:70, 0] = 0.004
        common = {
            "df_asset_rets": df_asset,
            "df_fact_rets": df_fact,
            "half_life": 40.0,
            "min_timestamps": 15,
        }
        with self.assertRaises(ValueError) as serial_err:
            pit_robust_betas(**common, parallel=False)
        with self.assertRaises(ValueError) as date_err:
            pit_robust_betas(
                **common, parallel=True, parallel_axis="date", n_jobs=N_JOBS
            )
        self.assertIn("near-zero variance", str(serial_err.exception))
        self.assertEqual(str(serial_err.exception), str(date_err.exception))

    def test_fill_missing_betas_with_unlisted_asset(self):
        """fill_missing_betas fills only the assets live at that date.

        The serial path fills inside the per-date callback, on the columns sim()
        handed it, so an asset that has not listed keeps its NaN. Filling the
        whole buffer after the loop instead hands every unlisted asset a beta of
        1.0 -- a different answer, and one that flows into df_hedge_rets as an
        ordinary-looking number.
        """
        n, min_ts = 100, 10
        df_asset, df_fact = _make_returns(
            n=n,
            betas={"EARLY": [0.6], "THIN": [0.4], "LATE": [1.2]},
            factors=["F"],
            seed=43,
        )
        # THIN lists immediately but has too few prints to be fit: live, so the
        # fill reaches it. LATE has not listed at all: not live, so the fill must
        # not reach it. The two are only distinguishable if the fill is
        # restricted to the columns sim() would have passed.
        df_asset.iloc[3:, df_asset.columns.get_loc("THIN")] = np.nan
        df_asset.iloc[: n // 2, df_asset.columns.get_loc("LATE")] = np.nan
        common = {
            "df_asset_rets": df_asset,
            "df_fact_rets": df_fact,
            "half_life": 30.0,
            "min_timestamps": min_ts,
            "fill_missing_betas": True,
        }
        date_axis, serial = _run_both(common)
        _assert_equiv(date_axis, serial)
        # Dates on which EARLY is fit but neither of the others can be: from the
        # first fittable window to just before LATE lists.
        early = date_axis["df_betas"].loc[
            df_asset.index[min_ts - 1] : df_asset.index[n // 2 - 2]
        ]
        self.assertTrue(early["EARLY"].notna().all())
        self.assertTrue((early["THIN"] == 1.0).all())
        self.assertTrue(early["LATE"].isna().all())

    def test_fill_missing_betas_on_hard_panel(self):
        """The fill against a panel that produces NaN betas for several reasons."""
        df_asset, df_fact = _make_hard_panel()
        common = {
            "df_asset_rets": df_asset,
            "df_fact_rets": df_fact,
            "half_life": 60.0,
            "min_timestamps": 20,
            "fill_missing_betas": True,
        }
        _assert_equiv(*_run_both(common))

    def test_sequential_backend(self):
        """n_jobs=1 takes joblib's sequential backend, which skips initializer.

        The worker-global panel is populated by the pool initializer, so an
        effective worker count of 1 must not go through the pool at all or every
        task raises KeyError.
        """
        df_asset, df_fact = _make_returns(
            n=90, betas={"A": [0.5], "B": [1.1]}, factors=["F"], seed=47
        )
        common = {
            "df_asset_rets": df_asset,
            "df_fact_rets": df_fact,
            "half_life": 30.0,
            "min_timestamps": 10,
        }
        _assert_equiv(*_run_both(common, n_jobs=1))

    def test_blocks_per_worker_does_not_change_results(self):
        """Block count is a scheduling knob, not a numerical one."""
        df_asset, df_fact = _make_hard_panel()
        common = {
            "df_asset_rets": df_asset,
            "df_fact_rets": df_fact,
            "half_life": 60.0,
            "min_timestamps": 20,
        }
        serial = pit_robust_betas(**common, parallel=False)
        for blocks_per_worker in (1, 4, 16):
            with self.subTest(blocks_per_worker=blocks_per_worker):
                _assert_equiv(
                    pit_robust_betas(
                        **common,
                        parallel=True,
                        parallel_axis="date",
                        n_jobs=N_JOBS,
                        blocks_per_worker=blocks_per_worker,
                    ),
                    serial,
                )

    def test_return_hedge_rets_by_fact_false(self):
        """The by-factor frame is dropped on both paths alike."""
        df_asset, df_fact = _make_returns(
            n=100,
            betas={"A": [0.5, 0.3], "B": [1.0, -0.2]},
            factors=["F1", "F2"],
            seed=53,
        )
        common = {
            "df_asset_rets": df_asset,
            "df_fact_rets": df_fact,
            "half_life": 40.0,
            "min_timestamps": 15,
            "return_hedge_rets_by_fact": False,
        }
        date_axis, serial = _run_both(common)
        self.assertNotIn("df_hedge_rets_by_fact", date_axis)
        _assert_equiv(date_axis, serial)

    def test_lambda_instead_of_half_life(self):
        """The decay can be given either way; both paths must build one weight set."""
        df_asset, df_fact = _make_returns(
            n=110, betas={"A": [0.7], "B": [1.4]}, factors=["F"], seed=59
        )
        common = {
            "df_asset_rets": df_asset,
            "df_fact_rets": df_fact,
            "lambda_": 0.985,
            "min_timestamps": 15,
        }
        _assert_equiv(*_run_both(common))


class TestDateAxisGuards(unittest.TestCase):
    """Input guards must fire identically whichever axis is selected."""

    def setUp(self):
        self.df_asset, self.df_fact = _make_returns(
            n=60, betas={"A": [0.5], "B": [0.9]}, factors=["F"], seed=61
        )
        self.common = {
            "df_asset_rets": self.df_asset,
            "df_fact_rets": self.df_fact,
            "half_life": 30.0,
            "min_timestamps": 10,
        }

    def test_unknown_parallel_axis_rejected(self):
        """An unrecognized axis names the input rather than falling back."""
        with self.assertRaises(ValueError) as err:
            pit_robust_betas(
                **self.common, parallel=True, parallel_axis="assets"  # type: ignore[arg-type]
            )
        self.assertIn("parallel_axis", str(err.exception))
        self.assertIn("assets", str(err.exception))

    def test_unknown_parallel_axis_rejected_when_serial(self):
        """A misspelled axis is a caller error even when it would be ignored."""
        with self.assertRaises(ValueError):
            pit_robust_betas(
                **self.common, parallel=False, parallel_axis="Date"  # type: ignore[arg-type]
            )

    def test_duplicate_asset_names_rejected(self):
        """The duplicate-name guard fires before either fit path runs."""
        df_asset = self.df_asset.copy()
        df_asset.columns = ["A", "A"]
        with self.assertRaises(ValueError) as err:
            pit_robust_betas(
                df_asset_rets=df_asset,
                df_fact_rets=self.df_fact,
                half_life=30.0,
                min_timestamps=10,
                parallel=True,
                parallel_axis="date",
            )
        self.assertIn("duplicate column name", str(err.exception))

    def test_duplicate_rebalance_timestamps_rejected(self):
        """Duplicate rebalance timestamps are rejected on the date axis too."""
        rebalance = self.df_asset.index.append(self.df_asset.index[:1])
        with self.assertRaises(ValueError) as err:
            pit_robust_betas(
                **self.common,
                parallel=True,
                parallel_axis="date",
                rebalance_time_index=rebalance,
            )
        self.assertIn("duplicate timestamp", str(err.exception))

    def test_missing_decay_control_rejected(self):
        """Neither half_life nor lambda_ raises rather than deriving garbage.

        The date path builds its own weight series instead of calling
        exponential_weights, so it validates through the same helper or not at
        all.
        """
        with self.assertRaises(ValueError) as err:
            pit_robust_betas(
                df_asset_rets=self.df_asset,
                df_fact_rets=self.df_fact,
                min_timestamps=10,
                parallel=True,
                parallel_axis="date",
            )
        self.assertIn("half_life or lambda_", str(err.exception))

    def test_non_positive_half_life_rejected(self):
        """A non-positive half_life raises rather than producing garbage weights."""
        with self.assertRaises(ValueError) as err:
            pit_robust_betas(
                df_asset_rets=self.df_asset,
                df_fact_rets=self.df_fact,
                half_life=-5.0,
                min_timestamps=10,
                parallel=True,
                parallel_axis="date",
            )
        self.assertIn("half_life must be positive", str(err.exception))


if __name__ == "__main__":
    unittest.main()

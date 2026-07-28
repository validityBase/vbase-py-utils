"""Unit tests for the sim module."""

import unittest
from typing import Dict

import numpy as np
import pandas as pd

from vbase_utils.sim import sim


class TestSim(unittest.TestCase):
    """Test cases for the sim module."""

    def setUp(self):
        """Set up test fixtures."""
        self.dates = pd.date_range("2023-01-01", periods=5)
        self.df1 = pd.DataFrame({"A": [1, 2, 3, 4, 5]}, index=self.dates)
        self.df2 = pd.DataFrame({"B": [10, 20, 30, 40, 50]}, index=self.dates)
        self.series = pd.Series([100, 200, 300, 400, 500], index=self.dates, name="C")
        self.sample_data = {"df1": self.df1, "df2": self.df2, "series": self.series}
        self.time_index = pd.date_range("2023-01-01", periods=5)

    def test_basic_functionality(self):
        """Test basic functionality of the sim function."""

        def callback(data: Dict[str, pd.DataFrame | pd.Series]) -> Dict[str, pd.Series]:
            df1 = data["df1"]
            df2 = data["df2"]
            series = data["series"]
            return {
                "values": pd.Series(
                    [
                        df1["A"].tail(1).values[0],
                        df2["B"].tail(1).values[0],
                        (df1["A"].tail(1) + df2["B"].tail(1) + series.tail(1)).values[
                            0
                        ],
                    ],
                    index=["A", "B", "all"],
                )
            }

        result = sim(self.sample_data, callback, self.time_index)

        self.assertIsInstance(result, dict)
        self.assertIn("values", result)
        self.assertIsInstance(result["values"], pd.DataFrame)
        self.assertEqual(len(result["values"]), len(self.time_index))
        pd.testing.assert_index_equal(result["values"].index, self.time_index)
        self.assertIn("all", result["values"].columns)

        # Check first row (should only have first day's data)
        self.assertEqual(result["values"].iloc[0]["all"], 111)  # 1 + 10 + 100

        # Check last row (should have all data)
        self.assertEqual(result["values"].iloc[-1]["all"], 555)  # 5 + 50 + 500

    def test_invalid_index(self):
        """Test that an integer Index (not DatetimeIndex or MultiIndex) raises ValueError."""
        df = pd.DataFrame({"A": [1, 2, 3]}, index=[1, 2, 3])
        time_index = pd.date_range("2023-01-01", periods=3)

        # pylint: disable=unused-argument
        def callback(data: Dict[str, pd.DataFrame | pd.Series]) -> Dict[str, pd.Series]:
            return {"values": pd.Series([1, 2, 3])}

        with self.assertRaisesRegex(ValueError, "must have a DatetimeIndex"):
            sim({"df": df}, callback, time_index)

    def test_callback_not_dict(self):
        """Test that callback returning non-dict raises ValueError."""

        # pylint: disable=unused-argument
        def callback(data: Dict[str, pd.DataFrame | pd.Series]) -> pd.Series:
            return pd.Series([1, 2, 3])

        with self.assertRaisesRegex(
            ValueError, "must return a dictionary of pandas Series"
        ):
            sim({"df1": self.df1}, callback, self.time_index)

    def test_callback_not_series(self):
        """Test that callback returning dict with non-Series values raises ValueError."""

        # pylint: disable=unused-argument
        def callback(
            data: Dict[str, pd.DataFrame | pd.Series],
        ) -> Dict[str, pd.DataFrame | pd.Series]:
            return {"values": {"A": [1, 2, 3]}}  # type: ignore

        with self.assertRaisesRegex(
            ValueError, "must return a dictionary of pandas Series or DataFrames"
        ):
            sim({"df1": self.df1}, callback, self.time_index)

    def test_callback_exception(self):
        """Test that callback exceptions are properly handled."""

        def callback(data: Dict[str, pd.DataFrame | pd.Series]) -> Dict[str, pd.Series]:
            raise ValueError("Test error")

        with self.assertRaisesRegex(ValueError, "Error processing timestamp"):
            sim({"df1": self.df1}, callback, self.time_index)

    def test_empty_data(self):
        """Test with empty data dictionary."""

        # pylint: disable=unused-argument
        def callback(data: Dict[str, pd.DataFrame | pd.Series]) -> Dict[str, pd.Series]:
            return {"values": pd.Series([1, 2, 3])}

        result = sim({}, callback, self.time_index)

        # If all input data is empty, the callback will be skipped.
        self.assertEqual(result, {})

    def test_data_masking(self):
        """Test that data is properly masked at each timestamp."""

        def callback(data: Dict[str, pd.DataFrame | pd.Series]) -> Dict[str, pd.Series]:
            df1 = data["df1"]
            # Return the length of available data at each timestamp
            return {"data_length": pd.Series([len(df1)], index=["length"])}

        result = sim(self.sample_data, callback, self.time_index)

        # Check that data length increases with each timestamp
        self.assertTrue(result["data_length"]["length"].is_monotonic_increasing)
        self.assertEqual(result["data_length"]["length"].iloc[0], 1)  # First timestamp
        self.assertEqual(result["data_length"]["length"].iloc[-1], 5)  # Last timestamp

    def test_missing_data(self):
        """Test handling of data with missing values."""
        df1 = pd.DataFrame({"A": [1, np.nan, 3, 4, 5]}, index=self.dates)
        df2 = pd.DataFrame({"B": [10, 20, np.nan, 40, 50]}, index=self.dates)

        def callback(data: Dict[str, pd.DataFrame | pd.Series]) -> Dict[str, pd.Series]:
            df1 = data["df1"]
            df2 = data["df2"]
            return {"result": pd.Series(df1["A"] + df2["B"], index=["sum"])}

        result = sim({"df1": df1, "df2": df2}, callback, self.time_index)
        self.assertIsInstance(result, dict)
        self.assertIn("result", result)
        self.assertIsInstance(result["result"], pd.DataFrame)
        self.assertEqual(len(result["result"]), len(self.time_index))
        self.assertTrue(
            pd.isna(result["result"]["sum"].iloc[1])
        )  # Should be NaN where either input is NaN

    def test_column_masking(self):
        """Test that columns with data only after a timestamp are removed by masking."""
        # Create a DataFrame with columns that start at different timestamps
        dates = pd.date_range("2023-01-01", periods=5)
        df = pd.DataFrame(
            {
                "early_col": [1, 2, 3, 4, 5],  # Data from start
                "mid_col": [None, None, 3, 4, 5],  # Data starts at index 2
                "late_col": [None, None, None, 4, 5],  # Data starts at index 3
            },
            index=dates,
        )

        def callback(data: Dict[str, pd.DataFrame | pd.Series]) -> Dict[str, pd.Series]:
            df = data["df"]
            # Return the columns available at each timestamp
            return {
                "columns": pd.Series([",".join(sorted(df.columns))], index=["cols"])
            }

        result = sim({"df": df}, callback, dates)

        # Check that columns are properly masked at each timestamp
        self.assertEqual(
            result["columns"]["cols"].iloc[0], "early_col"
        )  # Only early_col at t=0
        self.assertEqual(
            result["columns"]["cols"].iloc[1], "early_col"
        )  # Only early_col at t=1
        self.assertEqual(
            result["columns"]["cols"].iloc[2], "early_col,mid_col"
        )  # early_col and mid_col at t=2
        self.assertEqual(
            result["columns"]["cols"].iloc[3], "early_col,late_col,mid_col"
        )  # All columns at t=3
        self.assertEqual(
            result["columns"]["cols"].iloc[4], "early_col,late_col,mid_col"
        )  # All columns at t=4

    def test_callback_returns_empty_series(self):
        """Test that a callback returning an empty Series produces an empty DataFrame."""

        # pylint: disable=unused-argument
        def callback(data: Dict[str, pd.DataFrame | pd.Series]) -> Dict[str, pd.Series]:
            return {"weights": pd.Series([], dtype=float)}

        result = sim(self.sample_data, callback, self.time_index)

        self.assertIn("weights", result)
        self.assertTrue(result["weights"].empty)
        pd.testing.assert_index_equal(result["weights"].index, self.time_index)

    def test_callback_returns_empty_dataframe(self):
        """Test that a callback returning an empty DataFrame produces an empty DataFrame."""

        # pylint: disable=unused-argument
        def callback(
            data: Dict[str, pd.DataFrame | pd.Series],
        ) -> Dict[str, pd.DataFrame]:
            return {"predictions": pd.DataFrame()}

        result = sim(self.sample_data, callback, self.time_index)

        # Empty DataFrames carry no row index, so the output is empty with no index.
        self.assertIn("predictions", result)
        self.assertTrue(result["predictions"].empty)

    def test_on_result_streaming_mode(self):
        """on_result sink is called with the authoritative loop timestamp;
        sim() returns an empty dict while side effects capture each result."""
        # Extend time_index one day past the data end so the final loop
        # timestamp (2023-01-06) differs from the masked data's last row
        # (2023-01-05), confirming the sink receives the loop timestamp.
        time_index = pd.DatetimeIndex(
            list(self.time_index) + [self.time_index[-1] + pd.Timedelta(days=1)]
        )
        sink_calls: list[tuple[pd.Timestamp, dict]] = []

        def callback(data: Dict[str, pd.DataFrame | pd.Series]) -> Dict[str, pd.Series]:
            return {"v": pd.Series([data["df1"]["A"].iloc[-1]], index=["val"])}

        returned = sim(
            self.sample_data,
            callback,
            time_index,
            on_result=lambda ts, r: sink_calls.append((ts, r)),
        )

        # (2) Returned dict is empty — nothing is accumulated.
        self.assertEqual(returned, {})
        # Sink is called once per timestamp.
        self.assertEqual(len(sink_calls), len(time_index))
        for i, (ts, result_dict) in enumerate(sink_calls):
            # (1) Sink receives the loop timestamp, not masked data's last index.
            self.assertEqual(ts, time_index[i])
            self.assertIn("v", result_dict)
            self.assertIsInstance(result_dict["v"], pd.Series)
        # At the final step the loop timestamp (2023-01-06) is past the data;
        # the callback still ran and the sink captured the last data value.
        self.assertEqual(sink_calls[-1][0], time_index[-1])
        self.assertEqual(sink_calls[-1][1]["v"].iloc[0], 5.0)

    def test_callback_returns_empty_for_early_timestamps(self):
        """Test that empty Series results become NaN rows alongside valid rows."""
        cutoff = self.dates[2]  # 2023-01-03; first 2 timestamps return empty

        def callback(data: Dict[str, pd.DataFrame | pd.Series]) -> Dict[str, pd.Series]:
            latest_ts = data["df1"].index[-1]
            if latest_ts < cutoff:
                return {"result": pd.Series([], dtype=float)}
            return {"result": pd.Series([data["df1"]["A"].iloc[-1]], index=["value"])}

        result = sim(self.sample_data, callback, self.time_index)

        # All timestamps appear; empty-Series timestamps produce NaN rows.
        self.assertIn("result", result)
        self.assertEqual(len(result["result"]), len(self.time_index))
        self.assertTrue(result["result"]["value"].iloc[:2].isna().all())
        self.assertTrue(result["result"]["value"].iloc[2:].notna().all())


class TestSimMultiIndex(unittest.TestCase):
    """Test cases for sim() with MultiIndex (cross-sectional panel) inputs.

    The canonical use-case is a (date, symbol) panel where masking on the first
    (date) level gives the callback a causal view of the universe at each step.
    """

    def setUp(self):
        """Set up a balanced (date, symbol) MultiIndex panel."""
        self.dates = pd.date_range("2023-01-01", periods=4, freq="W")
        self.symbols = ["AAPL", "MSFT", "GOOG"]
        mi = pd.MultiIndex.from_product(
            [self.dates, self.symbols], names=["date", "symbol"]
        )
        n = len(mi)
        self.panel = pd.DataFrame(
            {
                "feature": np.arange(n, dtype=float),
                "factor": np.ones(n),
            },
            index=mi,
        )

    def test_multiindex_masking_row_count(self):
        """Callback sees only rows whose first-level date is <= the current timestamp."""
        n_rows: list[int] = []

        def callback(data: Dict[str, pd.DataFrame | pd.Series]) -> dict:
            n_rows.append(len(data["panel"]))
            return {}

        sim({"panel": self.panel}, callback, self.dates)

        n_sym = len(self.symbols)
        # Each successive timestamp exposes one more week (n_sym rows).
        self.assertEqual(n_rows, [n_sym * (i + 1) for i in range(len(self.dates))])

    def test_multiindex_masking_excludes_future(self):
        """No row with a future date is visible in the masked panel."""
        max_dates_seen: list[pd.Timestamp] = []

        def callback(data: Dict[str, pd.DataFrame | pd.Series]) -> dict:
            max_dates_seen.append(data["panel"].index.get_level_values("date").max())
            return {}

        sim({"panel": self.panel}, callback, self.dates)

        for i, seen_max in enumerate(max_dates_seen):
            self.assertLessEqual(
                seen_max,
                self.dates[i],
                msg=f"At step {i} (timestamp {self.dates[i].date()}), "
                f"future date {seen_max.date()} appeared in the panel.",
            )

    def test_multiindex_callback_sees_growing_window(self):
        """The number of distinct dates in the panel grows by one at each step."""
        n_dates_visible: list[int] = []

        def callback(data: Dict[str, pd.DataFrame | pd.Series]) -> dict:
            n_dates_visible.append(
                data["panel"].index.get_level_values("date").nunique()
            )
            return {}

        sim({"panel": self.panel}, callback, self.dates)

        self.assertEqual(n_dates_visible, list(range(1, len(self.dates) + 1)))

    def test_multiindex_result_accumulation_wide(self):
        """
        A symbol-indexed Series returned by the callback is accumulated as a row
        in a wide (date × symbol) DataFrame — one row per timestamp.
        """

        def callback(
            data: Dict[str, pd.DataFrame | pd.Series],
        ) -> Dict[str, pd.Series]:
            panel = data["panel"]
            t = panel.index.get_level_values("date").max()
            xs = panel[panel.index.get_level_values("date") == t]["feature"]
            xs = xs.copy()
            xs.index = xs.index.get_level_values("symbol")
            return {"xs": xs}

        result = sim({"panel": self.panel}, callback, self.dates)

        self.assertIn("xs", result)
        wide = result["xs"]
        # Row index = timestamps processed; column index = symbols.
        self.assertEqual(list(wide.index), list(self.dates))
        self.assertEqual(sorted(wide.columns.tolist()), sorted(self.symbols))

    def test_multiindex_invalid_first_level_type(self):
        """MultiIndex whose first level is not a DatetimeIndex raises ValueError."""
        mi = pd.MultiIndex.from_tuples(
            [(1, "AAPL"), (1, "MSFT"), (2, "AAPL"), (2, "MSFT")],
            names=["int_date", "symbol"],
        )
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}, index=mi)
        time_index = pd.date_range("2023-01-01", periods=2)

        def callback(_data: Dict[str, pd.DataFrame | pd.Series]) -> dict:
            return {}

        with self.assertRaisesRegex(ValueError, "first level must be a DatetimeIndex"):
            sim({"panel": df}, callback, time_index)

    def test_multiindex_dropna_removes_all_nan_column(self):
        """
        dropna(axis=1, how='all') applies to MultiIndex DataFrames: a column
        that is entirely NaN in the masked window is dropped from the data
        passed to the callback.
        """
        n_sym = len(self.symbols)
        n_dates = len(self.dates)
        # sparse_col is NaN for the first two date slices, present from date 3 onward.
        sparse_values = [np.nan] * (n_sym * 2) + [1.0] * (n_sym * (n_dates - 2))
        df = pd.DataFrame(
            {
                "always_present": np.arange(len(self.panel), dtype=float),
                "sparse_col": sparse_values,
            },
            index=self.panel.index,
        )
        cols_seen: list[list[str]] = []

        def callback(data: Dict[str, pd.DataFrame | pd.Series]) -> dict:
            cols_seen.append(sorted(data["panel"].columns.tolist()))
            return {}

        sim({"panel": df}, callback, self.dates)

        # Dates 0–1: sparse_col is entirely NaN in the masked window → dropped.
        self.assertEqual(cols_seen[0], ["always_present"])
        self.assertEqual(cols_seen[1], ["always_present"])
        # Date 2+: sparse_col has values → retained.
        self.assertEqual(cols_seen[2], ["always_present", "sparse_col"])
        self.assertEqual(cols_seen[3], ["always_present", "sparse_col"])

    def test_multiindex_empty_callback_result(self):
        """A callback that always returns {} leaves the sim() result empty."""

        def callback(_data: Dict[str, pd.DataFrame | pd.Series]) -> dict:
            return {}

        result = sim({"panel": self.panel}, callback, self.dates)

        self.assertEqual(result, {})

    def test_multiindex_mixed_with_datetimeindex(self):
        """
        A MultiIndex panel and a plain DatetimeIndex Series can be passed together;
        each is masked independently by its own index type.
        """
        market = pd.Series(
            np.arange(len(self.dates), dtype=float),
            index=self.dates,
            name="market_return",
        )
        n_panel_rows: list[int] = []
        market_vals: list[float] = []

        def callback(data: Dict[str, pd.DataFrame | pd.Series]) -> dict:
            n_panel_rows.append(len(data["panel"]))
            market_vals.append(float(data["market"].iloc[-1]))
            return {}

        sim({"panel": self.panel, "market": market}, callback, self.dates)

        n_sym = len(self.symbols)
        self.assertEqual(
            n_panel_rows, [n_sym * (i + 1) for i in range(len(self.dates))]
        )
        # market[i] == i (0-based) — each step exposes one more market observation.
        self.assertEqual(market_vals, list(range(len(self.dates))))


if __name__ == "__main__":
    unittest.main()

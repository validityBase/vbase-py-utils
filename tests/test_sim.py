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
        """Test that non-DatetimeIndex raises ValueError."""
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


if __name__ == "__main__":
    unittest.main()

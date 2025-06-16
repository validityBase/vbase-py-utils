import unittest
import numpy as np
import pandas as pd

from vbase_utils.stats.cross_section_regression import (
    run_cross_sectional_regression,
    calculate_factor_returns,
)

class TestRunCrossSectionalRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        cls.n = 100
        cls.assets = [f"A{i}" for i in range(5)]
        cls.factors = ["f1", "f2", "f3"]

        cls.true_params = np.array([0.5, -1.2, 2.0])

        cls.X = pd.DataFrame(
            np.random.randn(cls.n, len(cls.factors)),
            index=cls.assets * (cls.n // len(cls.assets)),
            columns=cls.factors,
        )
        cls.X.index = pd.Index(cls.X.index[: cls.n], name="asset")

        noise = np.random.normal(0, 0.01, cls.n)
        y_vals = cls.X.to_numpy().dot(cls.true_params) + noise
        cls.y = pd.Series(y_vals, index=cls.X.index, name="ret")

        cls.w = pd.Series(1.0, index=cls.X.index, name="w")

    def test_perfect_fit(self):
        """output perfect true_params"""
        est = run_cross_sectional_regression(
            asset_returns=self.y,
            factor_loadings=self.X,
            weights=self.w,
            huber_t=1.345,
        )
        for i, f in enumerate(self.factors):
            self.assertAlmostEqual(est[f], self.true_params[i], delta=1e-2)

    def test_mismatched_indices(self):
        """indices mismatched ValueError"""
        wrong_y = self.y.copy()
        wrong_y.index = pd.Index(["X"] * len(wrong_y), name=self.y.index.name)

        with self.assertRaises(ValueError):
            run_cross_sectional_regression(wrong_y, self.X, self.w)
        with self.assertRaises(ValueError):
            run_cross_sectional_regression(self.y, self.X, self.w.iloc[:-1])

    def test_empty_input(self):
        """empty input ValueError"""
        empty_s = pd.Series(dtype=float)
        empty_df = pd.DataFrame(dtype=float)
        with self.assertRaises(ValueError):
            run_cross_sectional_regression(empty_s, empty_df, empty_s)

    def test_extreme_outliers(self):
        """robust on outliers"""
        y2 = self.y.copy()
        idxs = np.arange(0, len(y2), 10)
        y2.iloc[idxs] += 100
        est2 = run_cross_sectional_regression(self.y, self.X, self.w)
        est_out = run_cross_sectional_regression(y2, self.X, self.w)
        for f in self.factors:
            self.assertAlmostEqual(est2[f], est_out[f], delta=0.5)

class TestCalculateFactorReturns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(1)
        cls.periods = pd.date_range("2021-01-01", periods=4, freq="D")
        cls.assets = ["A", "B", "C"]
        cls.factors = ["f1", "f2"]

        # MultiIndex exposures: (period, asset)
        idx = pd.MultiIndex.from_product(
            [cls.periods, cls.assets], names=("period", "asset")
        )
        rows = []
        # exposures: A→[1,0], B→[0,1], C→[1,1]
        for _ in cls.periods:
            rows.extend([[1, 0], [0, 1], [1, 1]])
        cls.exposures_df = pd.DataFrame(rows, index=idx, columns=cls.factors)

        # set weights all 1
        cls.weights_df = pd.DataFrame(1.0, index=idx, columns=["w"])

        # returns = exposures · [2,3]  -> A:2, B:3, C:5
        ret_rows = []
        for _ in cls.periods:
            ret_rows.append([2.0, 3.0, 5.0])
        cls.returns_df = pd.DataFrame(
            ret_rows, index=cls.periods, columns=cls.assets
        )

    def test_basic_factor_returns(self):
        fr = calculate_factor_returns(
            self.returns_df, self.exposures_df, self.weights_df
        )
        self.assertEqual(fr.shape, (len(self.periods), len(self.factors)))
        for d in self.periods:
            self.assertAlmostEqual(fr.loc[d, "f1"], 2.0, delta=1e-5)
            self.assertAlmostEqual(fr.loc[d, "f2"], 3.0, delta=1e-5)

    def test_empty_inputs(self):
        with self.assertRaises(ValueError):
            calculate_factor_returns(pd.DataFrame(), self.exposures_df, self.weights_df)
        with self.assertRaises(ValueError):
            calculate_factor_returns(self.returns_df, pd.DataFrame(), self.weights_df)
        with self.assertRaises(ValueError):
            calculate_factor_returns(self.returns_df, self.exposures_df, pd.DataFrame())

    def test_missing_period(self):
        # delete period 1
        bad_exp = self.exposures_df.drop(self.exposures_df.index[:3])
        fr = calculate_factor_returns(self.returns_df, bad_exp, self.weights_df)
        self.assertTrue(fr.iloc[0].isna().all())
        for d in self.periods[1:]:
            self.assertFalse(fr.loc[d].isna().any())

    def test_extreme_outliers(self):
        ret2 = self.returns_df.copy()
        ret2.iat[2, 0] += 100.0
        fr2 = calculate_factor_returns(ret2, self.exposures_df, self.weights_df)
        for d in self.periods:
            # allow delta<=0.5
            self.assertAlmostEqual(fr2.loc[d, "f1"], 2.0, delta=0.5)
            self.assertAlmostEqual(fr2.loc[d, "f2"], 3.0, delta=0.5)

if __name__ == "__main__":
    unittest.main()

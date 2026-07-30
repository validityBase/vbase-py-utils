"""Guard: the hand-rolled Huber RLM must match statsmodels RLM bit-for-bit.

The runtime betas path uses the pure-numpy hand-rolled fit
(:func:`vbase_utils.stats._huber_rlm.fit_huber_rlm_params`) and no longer depends
on statsmodels. statsmodels is a test-only dependency kept solely so this test can
assert the hand-rolled fit stays numerically equivalent to
``statsmodels.RLM(..., M=HuberT()).fit()`` -- the reference it was derived from.

If this test fails, the hand-rolled fit has drifted from statsmodels and must be
reconciled before shipping.
"""

import unittest

import numpy as np
import pandas as pd
import statsmodels.api as sm

from vbase_utils.stats._huber_rlm import fit_huber_rlm_params
from vbase_utils.stats.robust_betas import (
    check_min_timestamps_series,
    exponential_weights,
    robust_betas,
)

# Bit-faithful tolerance: the two implementations agree to ~1e-10 in practice.
EQUIV_ATOL = 1e-8


def _build_nontrivial_panel(t_n=180, a_n=40, f_n=3, seed=7):
    """Panel where the robust fit does real work (outliers, collinearity, NaNs)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=t_n, freq="B")
    fac = rng.normal(0, 0.01, (t_n, f_n))
    if f_n >= 2:
        # Near-collinear factor pair stresses the pinv/SVD path where a naive
        # normal-equations solver would drift from statsmodels.
        fac[:, 1] = fac[:, 0] + rng.normal(0, 0.0003, t_n)
    df_fact = pd.DataFrame(fac, index=dates, columns=[f"F{i}" for i in range(f_n)])

    betas = rng.normal(1.0, 0.3, (f_n, a_n))
    signal = fac @ betas
    scale = np.where(np.arange(t_n)[:, None] < t_n // 2, 0.005, 0.012)
    asset = signal + rng.standard_t(3, (t_n, a_n)) * scale
    # Inject point outliers so Huber weights bite.
    n_out = max(1, int(0.03 * t_n * a_n))
    asset[rng.integers(0, t_n, n_out), rng.integers(0, a_n, n_out)] += rng.normal(
        0, 0.08, n_out
    )
    # Ragged: NaN a short prefix of every 4th asset.
    asset[:12, ::4] = np.nan
    df_asset = pd.DataFrame(asset, index=dates, columns=[f"A{i}" for i in range(a_n)])
    return df_asset, df_fact


def _statsmodels_betas(df_asset, df_fact, half_life, min_timestamps):
    """Reference betas computed directly with statsmodels RLM, per asset.

    Feeds the fitter the exact inputs robust_betas builds (sqrt-time-weighted y,
    add_constant with a manually weighted constant, NaN mask).
    """
    weights = exponential_weights(len(df_asset), half_life=half_life)
    sqrt_weights = np.sqrt(weights)
    x_weighted = df_fact.multiply(sqrt_weights, axis=0)
    out = pd.DataFrame(index=df_fact.columns, columns=df_asset.columns, dtype=float)
    for asset in df_asset.columns:
        y_weighted = df_asset[asset].to_numpy() * sqrt_weights
        y_f, mask = check_min_timestamps_series(y_weighted, min_timestamps)
        if y_f.size == 0:
            continue
        x_w_const = sm.add_constant(x_weighted.loc[mask])
        x_w_const["const"] = x_w_const["const"] * sqrt_weights[mask]
        params = sm.RLM(y_f, x_w_const, M=sm.robust.norms.HuberT()).fit().params
        out[asset] = params  # label-aligned; the "const" row is dropped
    return out


class TestHandrolledVsStatsmodels(unittest.TestCase):
    """Assert the hand-rolled Huber RLM equals statsmodels."""

    def test_single_fit_matches_statsmodels(self):
        """A single non-trivial fit matches statsmodels to bit-faithful tolerance."""
        rng = np.random.default_rng(0)
        n, p = 120, 3
        x = rng.normal(0, 0.01, (n, p))
        x[:, 0] = 1.0  # constant column
        x[:, 2] = x[:, 1] + rng.normal(0, 3e-4, n)  # near-collinear
        y = x @ rng.normal(1, 0.3, p) + rng.standard_t(3, n) * 0.006
        y[rng.integers(0, n, 12)] += 0.08  # outliers

        p_sm = sm.RLM(y, x, M=sm.robust.norms.HuberT()).fit().params
        p_hr = fit_huber_rlm_params(y, x)
        np.testing.assert_allclose(p_hr, p_sm, atol=EQUIV_ATOL)

    def test_robust_betas_matches_statsmodels(self):
        """robust_betas (hand-rolled) matches a direct statsmodels computation."""
        df_asset, df_fact = _build_nontrivial_panel()
        half_life, min_timestamps = 40.0, 30

        got = robust_betas(
            df_asset, df_fact, half_life=half_life, min_timestamps=min_timestamps
        )
        ref = _statsmodels_betas(df_asset, df_fact, half_life, min_timestamps)

        # Same NaN placement and same values where computed.
        self.assertTrue((got.isna().to_numpy() == ref.isna().to_numpy()).all())
        np.testing.assert_allclose(
            got.to_numpy(dtype=float),
            ref.to_numpy(dtype=float),
            atol=EQUIV_ATOL,
            equal_nan=True,
        )


if __name__ == "__main__":
    unittest.main()

"""Shared test fixtures for robust_betas and parallel_robust_betas tests."""

from typing import Callable

import numpy as np
import pandas as pd

STD_FACT_RETS = 0.01
STD_ASSET_RETS = 0.005


def make_single_asset_ret_frames(
    spy_returns: np.ndarray, n_timestamps: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build single-asset and single-factor return DataFrames."""
    asset_returns = 1.5 * spy_returns + np.random.normal(
        0, STD_ASSET_RETS, n_timestamps
    )
    df_asset_rets = pd.DataFrame({"Asset1": asset_returns})
    df_fact_rets = pd.DataFrame({"SPY": spy_returns})
    return df_asset_rets, df_fact_rets


def make_multi_asset_ret_frames(
    spy_returns: np.ndarray, n_timestamps: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build two-asset and single-factor return DataFrames."""
    asset_returns_1 = 1.2 * spy_returns + np.random.normal(
        0, STD_ASSET_RETS, n_timestamps
    )
    asset_returns_2 = 0.8 * spy_returns + np.random.normal(
        0, STD_ASSET_RETS, n_timestamps
    )
    df_asset_rets = pd.DataFrame({"Asset1": asset_returns_1, "Asset2": asset_returns_2})
    df_fact_rets = pd.DataFrame({"SPY": spy_returns})
    return df_asset_rets, df_fact_rets


def make_fit_first_call_error_side_effect(real_fit: Callable) -> Callable:
    """Return a side-effect whose first call raises LinAlgError.

    Subsequent calls delegate to *real_fit* so only one asset gets NaN betas.
    Used to patch ``fit_huber_rlm_params`` in the serial and parallel paths.
    """
    call_count = [0]

    def fit_side_effect(*args, **kwargs):
        if call_count[0] == 0:
            call_count[0] += 1
            raise np.linalg.LinAlgError("singular matrix")
        call_count[0] += 1
        return real_fit(*args, **kwargs)

    return fit_side_effect


def make_multi_factor_ret_frames(
    spy_returns: np.ndarray, n_timestamps: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build single-asset and two-factor return DataFrames."""
    iwm_returns = np.random.normal(0, STD_FACT_RETS, n_timestamps)
    asset_returns = (
        1.2 * spy_returns
        + 0.5 * iwm_returns
        + np.random.normal(0, STD_ASSET_RETS, n_timestamps)
    )
    df_asset_rets = pd.DataFrame({"Asset1": asset_returns})
    df_fact_rets = pd.DataFrame({"SPY": spy_returns, "IWM": iwm_returns})
    return df_asset_rets, df_fact_rets


def make_linear_ret_frames(
    n: int,
    betas: dict[str, list[float]],
    factors: list[str],
    seed: int,
    noise: float = STD_ASSET_RETS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build synthetic factor and asset returns with a known linear structure.

    Shared by the two pit_robust_betas equivalence suites -- the asset axis's and
    the date axis's -- so that "identical inputs" means the same bytes rather
    than two copies of the same generator that can drift apart.

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
    fact = {f: rng.normal(0, STD_FACT_RETS, n) for f in factors}
    df_fact = pd.DataFrame(fact, index=dates)
    assets = {}
    for asset, asset_betas in betas.items():
        signal = sum(b * df_fact[f].values for b, f in zip(asset_betas, factors))
        assets[asset] = 0.0001 + signal + rng.normal(0, noise, n)
    df_asset = pd.DataFrame(assets, index=dates)
    return df_asset, df_fact

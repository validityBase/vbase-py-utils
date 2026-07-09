"""Parallel robust timeseries regression.

Thin wrapper over :func:`vbase_utils.stats._fast_betas.compute_betas_fast`, which
fans the per-asset Huber-RLM fits out across processes in chunked asset blocks
(one joblib task per block, numpy-only workers with BLAS pinned to a single
thread). See that module for the rationale; the fit itself is the pure-numpy
hand-rolled Huber RLM (bit-faithful to statsmodels).
"""

import logging

import pandas as pd

from vbase_utils.stats._fast_betas import compute_betas_fast

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# pylint: disable=too-many-arguments
def parallel_robust_betas(
    df_asset_rets: pd.DataFrame,
    df_fact_rets: pd.DataFrame,
    half_life: float | None = None,
    lambda_: float | None = None,
    min_timestamps: int = 10,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Perform robust regression (Huber RLM) with exponential time-weighting.

    Fits are parallelized across chunked asset blocks; the result is identical to
    the serial :func:`vbase_utils.stats.robust_betas.robust_betas`.

    Args:
        df_asset_rets: DataFrame of dependent returns with shape (n_timestamps, n_assets).
        df_fact_rets: DataFrame of factor returns with shape (n_timestamps, n_factors).
        half_life: Half-life in time units (e.g., days). Must be positive.
            Recommendations for half-life based on the horizon:
            | Horizon (days) | Recommended half-life (days) |
            |----------------|------------------------------|
            | 30             | 10                           |
            | 60             | 20                           |
            | 90             | 30                           |
            | 180            | 60                           |
            | 365            | 120                          |
        lambda_: Decay factor (e.g., 0.985). Must be between 0 and 1.
        min_timestamps: Minimum number of timestamps required for regression. Defaults to 10.
        n_jobs: Number of parallel jobs to run. -1 uses all available cores. Defaults to -1.

    Returns:
        DataFrame of shape (n_factors, n_assets) containing the computed betas.

    Raises:
        ValueError: If inputs are empty, have mismatched rows, excessive NaNs,
            or near-zero variance in df_fact_rets.
            Note: insufficient timestamps (< min_timestamps) returns an all-NaN
            beta matrix with a warning rather than raising.
    """
    return compute_betas_fast(
        df_asset_rets,
        df_fact_rets,
        half_life=half_life,
        lambda_=lambda_,
        min_timestamps=min_timestamps,
        n_jobs=n_jobs,
    )

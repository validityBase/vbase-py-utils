"""Parallel Robust timeseries regression module.

Note: ``pit_robust_betas`` no longer uses this asset-level parallel path; it
parallelizes over rebalance dates instead (see ``_pit_betas_parallel``). This
module is retained for backward compatibility for any direct importers.
"""

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed

from vbase_utils.stats.robust_betas import (
    check_min_timestamps_series,
    prepare_weighted_regression_inputs,
)

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _compute_single_asset_beta(
    asset: str,
    df_asset_rets: pd.DataFrame,
    x_weighted: pd.DataFrame,
    sqrt_weights: np.ndarray,
    min_timestamps: int,
):
    """Compute beta for a single asset - extracted for parallelization.

    Args:
        asset: Asset name/column identifier
        df_asset_rets: DataFrame of asset returns
        x_weighted: Weighted factor returns DataFrame
        sqrt_weights: Square root of exponential weights
        min_timestamps: Minimum timestamps required for regression

    Returns:
        Tuple of (asset, params) where params is None if insufficient data.
    """
    y = df_asset_rets[asset].values
    y_weighted: np.ndarray = y * sqrt_weights

    y_weighted_filtered, y_valid_mask = check_min_timestamps_series(
        y_weighted, min_timestamps
    )
    if y_weighted_filtered.size == 0:
        return asset, None

    y_weighted = y_weighted_filtered

    x_w_const: pd.DataFrame = sm.add_constant(x_weighted.loc[y_valid_mask])
    # Statsmodels does not apply weights to constant, apply manually.
    x_w_const["const"] = x_w_const["const"] * sqrt_weights[y_valid_mask]
    rlm_model = sm.RLM(y_weighted, x_w_const, M=sm.robust.norms.HuberT())
    rlm_results = rlm_model.fit()

    return asset, rlm_results.params


# The function must take a large number of arguments
# and consequently has a large number of local variables.
# pylint: disable=too-many-arguments, too-many-locals
def parallel_robust_betas(
    df_asset_rets: pd.DataFrame,
    df_fact_rets: pd.DataFrame,
    half_life: float | None = None,
    lambda_: float | None = None,
    min_timestamps: int = 10,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Perform robust regression (RLM) with exponential time-weighting.

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
    # prepare_weighted_regression_inputs validates inputs and initializes shared matrices.
    df_betas, sqrt_weights, x_weighted = prepare_weighted_regression_inputs(
        df_asset_rets, df_fact_rets, half_life, lambda_, min_timestamps
    )
    # If not enough timestamps, return the all-NaN beta matrix.
    if sqrt_weights is None or x_weighted is None:
        return df_betas

    # Use parallel processing for asset regressions
    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_single_asset_beta)(
            asset, df_asset_rets, x_weighted, sqrt_weights, min_timestamps
        )
        for asset in df_asset_rets.columns
    )

    # Fill results back into DataFrame
    for asset, params in results:
        if params is not None:
            df_betas[asset] = params

    return df_betas

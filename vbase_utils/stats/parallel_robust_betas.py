"""Parallel Robust timeseries regression module"""

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed

from vbase_utils.stats.robust_betas import (
    check_min_timestamps_series,
    exponential_weights,
    NEAR_ZERO_VARIANCE_THRESHOLD,
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
        ValueError: If inputs are empty, have insufficient data, mismatched rows,
            excessive NaNs, or near-zero variance in df_fact_rets.
    """
    # Check for empty inputs
    if df_asset_rets.empty:
        logger.error("Input DataFrame df_asset_rets is empty.")
        raise ValueError("Input DataFrame df_asset_rets is empty.")
    if df_fact_rets.empty:
        logger.error("Input DataFrame df_fact_rets is empty.")
        raise ValueError("Input DataFrame df_fact_rets is empty.")

    # Check for mismatched row counts
    if df_asset_rets.shape[0] != df_fact_rets.shape[0]:
        logger.error(
            "Mismatched row counts: df_asset_rets has %d rows, df_fact_rets has %d rows.",
            df_asset_rets.shape[0],
            df_fact_rets.shape[0],
        )
        raise ValueError(
            "Mismatched row counts: "
            f"df_asset_rets has {df_asset_rets.shape[0]} rows, "
            f"df_fact_rets has {df_fact_rets.shape[0]} rows."
        )

    # Make sure that the indices are the same.
    # We do not know at this level what is the best way to combine and align
    # the indices so must fail.
    if not df_asset_rets.index.equals(df_fact_rets.index):
        raise ValueError("df_asset_rets and df_fact_rets must have the same index.")

    n_timestamps, _ = df_asset_rets.shape

    df_betas: pd.DataFrame = pd.DataFrame(
        index=df_fact_rets.columns, columns=df_asset_rets.columns
    )

    # Check minimum timestamps
    if n_timestamps < min_timestamps:
        logger.warning(
            "Insufficient data: %d timestamps available, minimum required is %d.",
            n_timestamps,
            min_timestamps,
        )
        # Return a DataFrame with all NaNs.
        return df_betas

    # Check for near-zero variance in df_fact_rets
    if df_fact_rets.var().min() < NEAR_ZERO_VARIANCE_THRESHOLD:
        logger.error("One or more factors in df_fact_rets have near-zero variance.")
        raise ValueError("One or more factors in df_fact_rets have near-zero variance.")

    # Calculate weights
    weights: np.ndarray = exponential_weights(
        n_timestamps, half_life=half_life, lambda_=lambda_
    )
    sqrt_weights: np.ndarray = np.sqrt(weights)

    # Implement weighted regression for each asset
    # by multiplying the x and y matrices by the square root of the weights.
    x_weighted: pd.DataFrame = df_fact_rets.multiply(sqrt_weights, axis=0)

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

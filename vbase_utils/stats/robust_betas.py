"""Robust timeseries regression module"""

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# Threshold for near-zero variance in df_fact_rets.
NEAR_ZERO_VARIANCE_THRESHOLD = 1e-10


def exponential_weights(
    n: int,
    half_life: float | None = None,
    lambda_: float | None = None,
) -> np.ndarray:
    """Generate exponential decay weights for n time periods.

    Either half_life or lambda_ must be provided.
    If both are provided, lambda_ is used.

    Args:
        n: Number of time periods.
        half_life: Half-life in time units (e.g., days). Must be positive.
        lambda_: Decay factor (e.g., 0.985). Must be between 0 and 1.

    Returns:
        Normalized exponential decay weights as a numpy array.

    Raises:
        ValueError: If neither half_life nor lambda_ is provided.
        ValueError: If half_life is not positive or lambda_ is not between 0 and 1.
    """
    if half_life is None and lambda_ is None:
        raise ValueError("Either half_life or lambda_ must be provided.")
    if half_life is not None and half_life <= 0:
        raise ValueError("half_life must be positive.")
    if lambda_ is not None and not 0 < lambda_ < 1:
        raise ValueError("lambda_ must be between 0 and 1.")

    if lambda_ is None:
        lambda_ = np.exp(np.log(0.5) / half_life)

    weights: np.ndarray = lambda_ ** np.arange(n - 1, -1, -1)
    return weights / np.sum(weights)  # normalize


# The function must take a large number of arguments
# and consequently has a large number of local variables.
# pylint: disable=too-many-arguments, too-many-locals
def robust_betas(
    df_asset_rets: pd.DataFrame,
    df_fact_rets: pd.DataFrame,
    half_life: float | None = None,
    lambda_: float | None = None,
    min_timestamps: int = 10,
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

    Returns:
        DataFrame of shape (n_factors + 1, n_assets) containing the computed betas.

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
    for asset in df_asset_rets.columns:
        y: np.ndarray = df_asset_rets[asset].values
        y_weighted: np.ndarray = y * sqrt_weights

        x_w_const: pd.DataFrame = sm.add_constant(x_weighted)
        rlm_model = sm.RLM(y_weighted, x_w_const, M=sm.robust.norms.HuberT())
        rlm_results = rlm_model.fit()

        df_betas[asset] = rlm_results.params

    return df_betas

"""Robust timeseries regression module"""

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.typing import NDArray

from vbase_utils.stats._huber_rlm import fit_huber_rlm_params

# Supported robust-fit backends. "statsmodels" is the production default;
# "handrolled" uses the pure-numpy bit-faithful reimplementation and is opt-in
# for benchmarking (avoids per-fit statsmodels object overhead).
_BACKENDS = ("statsmodels", "handrolled")

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# Threshold for near-zero variance in df_fact_rets.
NEAR_ZERO_VARIANCE_THRESHOLD = 1e-10


def check_min_timestamps_series(
    arr: NDArray[np.floating], min_timestamps: int
) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """
    Filter a numpy array based on minimum number of defined (non-NaN) values.

    Args:
        arr: Numpy array to filter
        min_timestamps: Minimum number of defined values required

    Returns:
        tuple: Filtered array and boolean mask identifying non-NaN entries. If the
               minimum defined values condition is not met, both elements are empty arrays.
    """
    # Check for NaN values.
    mask = ~np.isnan(arr)
    # Count the number of defined values.
    defined_count = np.count_nonzero(mask)

    # If the number of defined values is greater than or equal to the minimum number of timestamps,
    # return the filtered array and mask.
    if defined_count >= min_timestamps:
        return arr[mask], mask

    # Otherwise, return empty arrays.
    empty_filtered = np.array([], dtype=arr.dtype)
    empty_mask = np.array([], dtype=bool)
    return empty_filtered, empty_mask


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


def _validate_beta_inputs(
    df_asset_rets: pd.DataFrame,
    df_fact_rets: pd.DataFrame,
    min_timestamps: int,
) -> tuple[int, pd.DataFrame, bool]:
    """Validate beta inputs and build the shared result DataFrame."""
    # Check for empty inputs.
    if df_asset_rets.empty:
        logger.error("Input DataFrame df_asset_rets is empty.")
        raise ValueError("Input DataFrame df_asset_rets is empty.")
    if df_fact_rets.empty:
        logger.error("Input DataFrame df_fact_rets is empty.")
        raise ValueError("Input DataFrame df_fact_rets is empty.")

    # Check for mismatched row counts.
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
        index=df_fact_rets.columns, columns=df_asset_rets.columns, dtype=float
    )

    # Check minimum timestamps.
    if n_timestamps < min_timestamps:
        logger.warning(
            "Insufficient data: %d timestamps available, minimum required is %d.",
            n_timestamps,
            min_timestamps,
        )
        # Not enough timestamps to perform regression.
        # Return the timestamp count and all-NaN beta matrix.
        return n_timestamps, df_betas, False

    # Check for near-zero variance in df_fact_rets.
    if df_fact_rets.var().min() < NEAR_ZERO_VARIANCE_THRESHOLD:
        logger.error("One or more factors in df_fact_rets have near-zero variance.")
        raise ValueError("One or more factors in df_fact_rets have near-zero variance.")

    return n_timestamps, df_betas, True


def prepare_weighted_regression_inputs(
    df_asset_rets: pd.DataFrame,
    df_fact_rets: pd.DataFrame,
    half_life: float | None,
    lambda_: float | None,
    min_timestamps: int,
) -> tuple[pd.DataFrame, np.ndarray | None, pd.DataFrame | None]:
    """Validate inputs and prepare shared weighted-regression matrices."""
    n_timestamps, df_betas, has_enough_timestamps = _validate_beta_inputs(
        df_asset_rets, df_fact_rets, min_timestamps
    )
    if not has_enough_timestamps:
        return df_betas, None, None

    # Calculate weights.
    weights: np.ndarray = exponential_weights(
        n_timestamps, half_life=half_life, lambda_=lambda_
    )
    sqrt_weights: np.ndarray = np.sqrt(weights)

    # Implement weighted regression for each asset
    # by multiplying the x and y matrices by the square root of the weights.
    x_weighted: pd.DataFrame = df_fact_rets.multiply(sqrt_weights, axis=0)
    return df_betas, sqrt_weights, x_weighted


# The function must take a large number of arguments
# and consequently has a large number of local variables.
# pylint: disable=too-many-arguments, too-many-locals
def robust_betas(
    df_asset_rets: pd.DataFrame,
    df_fact_rets: pd.DataFrame,
    half_life: float | None = None,
    lambda_: float | None = None,
    min_timestamps: int = 10,
    backend: str = "statsmodels",
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
        backend: Robust-fit backend, "statsmodels" (default, production path) or
            "handrolled" (pure-numpy bit-faithful reimplementation, opt-in for
            benchmarking). Numerically identical either way.

    Returns:
        DataFrame of shape (n_factors, n_assets) containing the computed betas.

    Raises:
        ValueError: If inputs are empty, have mismatched rows, excessive NaNs,
            or near-zero variance in df_fact_rets.
            Note: insufficient timestamps (< min_timestamps) returns an all-NaN
            beta matrix with a warning rather than raising.
    """
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; expected one of {_BACKENDS}.")

    # prepare_weighted_regression_inputs validates inputs and initializes shared matrices.
    df_betas, sqrt_weights, x_weighted = prepare_weighted_regression_inputs(
        df_asset_rets, df_fact_rets, half_life, lambda_, min_timestamps
    )
    # If not enough timestamps, return the all-NaN beta matrix.
    if sqrt_weights is None or x_weighted is None:
        return df_betas

    for asset in df_asset_rets.columns:
        y: np.ndarray = df_asset_rets[asset].values
        y_weighted: np.ndarray = y * sqrt_weights

        # Check if there are enough defined values to perform the regression.
        # If so, drop any NaN values and continue.
        y_filtered, valid_mask = check_min_timestamps_series(y_weighted, min_timestamps)
        if y_filtered.size == 0:
            # Not enough defined values to perform the regression.
            #  Skip regression for this asset.
            df_betas[asset] = np.nan
            continue

        # X is filtered according to the mask used to filter y.
        x_w_const: pd.DataFrame = sm.add_constant(x_weighted.loc[valid_mask])
        # Statsmodels does not apply weights to constant, apply manually.
        x_w_const["const"] = x_w_const["const"] * sqrt_weights[valid_mask]
        params = _fit_asset_params(asset, y_filtered, x_w_const, backend)
        if params is not None:
            df_betas[asset] = params

    return df_betas


def _fit_asset_params(
    asset: str,
    y_endog: np.ndarray,
    x_w_const: pd.DataFrame,
    backend: str,
) -> pd.Series | None:
    """Fit one asset's robust betas, returning params (indexed by design columns).

    Returns None if the fit raises a linear-algebra/zero-division error, so the
    caller leaves that asset's betas NaN. Both backends return a Series indexed by
    ``x_w_const.columns`` (including the "const" row), so label-aligned assignment
    into the factor-indexed beta frame drops the constant automatically.
    """
    if backend == "handrolled":
        try:
            params = fit_huber_rlm_params(y_endog, x_w_const.to_numpy())
        except (np.linalg.LinAlgError, ZeroDivisionError) as e:
            logger.exception("Error fitting RLM model for asset %s: %s", asset, e)
            return None
        return pd.Series(params, index=x_w_const.columns)

    rlm_model = sm.RLM(y_endog, x_w_const, M=sm.robust.norms.HuberT())
    try:
        return rlm_model.fit().params
    except (np.linalg.LinAlgError, ZeroDivisionError) as e:
        logger.exception("Error fitting RLM model for asset %s: %s", asset, e)
        return None

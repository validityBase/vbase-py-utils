"""Robust timeseries regression module"""

import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from vbase_utils.stats._huber_rlm import fit_huber_rlm_params

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# Threshold for near-zero variance in df_fact_rets.
NEAR_ZERO_VARIANCE_THRESHOLD = 1e-10

# The no-live-assets condition recurs on every leading rebalance date of a
# staggered panel, so the warning is emitted once per process; the per-window
# detail goes to DEBUG. Held in a dict so the flag can be flipped without a
# global statement.
_NO_ASSET_COLUMNS_WARNED = {"warned": False}


def _warn_no_asset_columns(timestamp) -> None:
    """Warn that a window contained no assets with data, at most once."""
    logger.debug(
        "No asset has any data on or before %s; no betas for this window.",
        timestamp,
    )
    if _NO_ASSET_COLUMNS_WARNED["warned"]:
        return
    _NO_ASSET_COLUMNS_WARNED["warned"] = True
    logger.warning(
        "No asset has any data on or before %s, so no betas are produced for "
        "that window; prior betas carry forward once assets have data. This is "
        "expected at the leading dates of a panel whose assets list later than "
        "the factors. Further occurrences are logged at DEBUG.",
        timestamp,
    )


def check_min_timestamps_series(
    arr: NDArray[np.floating], min_timestamps: int
) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """
    Filter a numpy array based on minimum number of defined (finite) values.

    Args:
        arr: Numpy array to filter
        min_timestamps: Minimum number of defined values required

    Returns:
        tuple: Filtered array and boolean mask identifying finite entries. If the
               minimum defined values condition is not met, both elements are empty arrays.
    """
    # Keep only finite values (drops NaN and +/-inf).
    mask = np.isfinite(arr)
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
    # A frame carrying rows but no columns is not a malformed input: callers
    # that mask point-in-time (sim(), via pit_robust_betas) drop asset columns
    # that are entirely NaN over the current window, so on a panel with
    # staggered listings every asset can disappear at an early rebalance date.
    # That case yields no betas and is handled after the shape checks below. A
    # frame with no rows at all is a genuine caller error and still raises.
    no_asset_columns = df_asset_rets.shape[0] > 0 and df_asset_rets.shape[1] == 0
    if df_asset_rets.empty and not no_asset_columns:
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

    # No asset has any data in this window. Return no betas rather than raising,
    # so a simulation continues past the leading dates of a staggered panel;
    # callers that forward-fill (e.g. pit_robust_betas) start carrying betas
    # once assets list. This mirrors the non-finite-factor handling below.
    if no_asset_columns:
        _warn_no_asset_columns(df_asset_rets.index[-1])
        return n_timestamps, df_betas, False

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

    # Non-finite factors (NaN/inf) are shared across all assets, and neither fit
    # path masks them out of the design matrix, so a single bad row invalidates
    # every fit over the window containing it. Return no betas for this date
    # rather than raising, so a simulation continues past dirty factor dates.
    # Callers that forward-fill (e.g. pit_robust_betas) carry the prior date's
    # betas over the gap; the date is not left NaN downstream.
    finite_rows = np.isfinite(df_fact_rets.to_numpy()).all(axis=1)
    if not finite_rows.all():
        bad_index = df_fact_rets.index[~finite_rows]
        logger.warning(
            "Non-finite (NaN/inf) factor value(s) at %d timestamp(s) "
            "(first %s, last %s); no new betas for %s, prior betas carry forward.",
            len(bad_index),
            bad_index[0],
            bad_index[-1],
            df_fact_rets.index[-1],
        )
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
) -> pd.DataFrame:
    """Perform robust regression (Huber RLM) with exponential time-weighting.

    Uses the numba/JIT hand-rolled Huber-t fit
    (:func:`vbase_utils.stats._huber_rlm.fit_huber_rlm_params`), which reproduces
    ``statsmodels.RLM(..., M=HuberT()).fit()`` bit-for-bit (guarded by
    ``tests/stats/test_handrolled_vs_statsmodels.py``).

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

    # Hoist both panels out of pandas once. Indexing them per asset through
    # pandas instead costs a boolean .loc plus a to_numpy for each design
    # matrix, which dominates the loop on wide panels. Indexing numpy directly
    # is bit-identical: the same buffers, the same elementwise products, in the
    # same order.
    y_all = df_asset_rets.to_numpy(dtype=np.float64)
    xw = x_weighted.to_numpy()
    # Results collect into a buffer and are wrapped once at the end; assets that
    # are skipped or whose fit fails keep the all-NaN row the frame starts with.
    out = df_betas.to_numpy(dtype=np.float64, copy=True)

    for j, asset in enumerate(df_asset_rets.columns):
        y: np.ndarray = y_all[:, j]
        y_weighted: np.ndarray = y * sqrt_weights

        # Check if there are enough defined values to perform the regression.
        # If so, drop any NaN values and continue.
        y_filtered, valid_mask = check_min_timestamps_series(y_weighted, min_timestamps)
        if y_filtered.size == 0:
            # Not enough defined values to perform the regression.
            #  Skip regression for this asset.
            continue

        # Design = [const, factors]. The constant column is the weighted 1s
        # (== sqrt_weights on the valid rows), matching the weighted regression.
        x_design = np.column_stack((sqrt_weights[valid_mask], xw[valid_mask]))
        params = _fit_asset_betas(asset, y_filtered, x_design)
        if params is not None:
            out[:, j] = params

    return pd.DataFrame(out, index=df_betas.index, columns=df_betas.columns)


def _fit_asset_betas(
    asset: str,
    y_endog: np.ndarray,
    x_design: np.ndarray,
) -> np.ndarray | None:
    """Fit one asset's robust betas via the hand-rolled Huber RLM.

    ``x_design`` is ``[const, factors]``; the returned array is the factor betas
    (constant dropped), positionally aligned with the factor columns. Returns
    None if the fit raises a linear-algebra/zero-division error, so the caller
    leaves that asset's betas NaN.
    """
    try:
        params = fit_huber_rlm_params(y_endog, x_design, label=asset)
    except (np.linalg.LinAlgError, ZeroDivisionError) as e:
        logger.exception("Error fitting RLM model for asset %s: %s", asset, e)
        return None
    return params[1:]

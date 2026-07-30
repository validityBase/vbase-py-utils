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

# Complete-case deletion removes rows on which any factor is non-finite, so on a
# panel whose factors keep different calendars it fires on every window. The
# summary warning is emitted once per process; per-window detail goes to DEBUG.
_DELETED_ROWS_WARNED = {"warned": False}


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


def _warn_deleted_rows(n_deleted: int, n_rows: int, timestamp) -> None:
    """Warn that complete-case deletion dropped rows from a window, at most once."""
    logger.debug(
        "Complete-case deletion dropped %d of %d rows from the window ending %s "
        "(non-finite factor value).",
        n_deleted,
        n_rows,
        timestamp,
    )
    if _DELETED_ROWS_WARNED["warned"]:
        return
    _DELETED_ROWS_WARNED["warned"] = True
    logger.warning(
        "Complete-case deletion dropped %d of %d rows from the window ending %s: "
        "a factor was non-finite (NaN/inf) on those dates, so they are excluded "
        "from the regression for every asset. This is expected when factors keep "
        "different calendars. Further occurrences are logged at DEBUG.",
        n_deleted,
        n_rows,
        timestamp,
    )


def finite_column_variances(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Per-column variance over each column's finite values.

    pandas' ``var()`` skips NaN but not +/-inf: with an inf present ``avg -
    values`` is ``inf - inf``, so the result is NaN plus a RuntimeWarning and an
    inf blinds the check rather than being ignored by it.

    Columns with fewer than two finite values yield NaN, which compares False
    against any threshold and so is never flagged. Too little data to judge a
    column is not the same as the column being flat, and the callers that care
    about emptiness check for it separately.
    """
    finite = np.isfinite(x)
    out = np.full(x.shape[1], np.nan)
    for j in range(x.shape[1]):
        col = x[finite[:, j], j]
        if col.size >= 2:
            out[j] = col.var(ddof=1)
    return out


def check_min_timestamps_series(
    arr: NDArray[np.floating],
    min_timestamps: int,
    extra_mask: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """
    Filter a numpy array based on minimum number of defined (finite) values.

    Args:
        arr: Numpy array to filter
        min_timestamps: Minimum number of defined values required
        extra_mask: Optional boolean mask ANDed with the finite mask, so a row is
            kept only when it is finite in ``arr`` *and* admitted by the caller.
            Used to intersect an asset's own defined rows with the rows on which
            every factor is finite (complete-case deletion), giving the row set
            the regression will actually use.

    Returns:
        tuple: Filtered array and boolean mask identifying kept entries. If the
               minimum defined values condition is not met, both elements are empty arrays.
    """
    # Keep only finite values (drops NaN and +/-inf).
    mask = np.isfinite(arr)
    if extra_mask is not None:
        # np.isfinite returns a fresh array, so this does not touch the caller's.
        mask &= extra_mask
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
) -> tuple[int, pd.DataFrame, NDArray[np.bool_] | None]:
    """Validate beta inputs and build the shared result DataFrame.

    Returns:
        ``(n_timestamps, df_betas, complete_rows)``. ``n_timestamps`` is the full
        window length, which the exponential weights are built on so the decay
        stays anchored to calendar time rather than to the surviving row count.
        ``complete_rows`` is the boolean mask of rows on which every factor is
        finite -- the rows the regression will use -- or None when no betas can
        be produced for this window, in which case ``df_betas`` is all-NaN.
    """
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
    # once assets list.
    if no_asset_columns:
        _warn_no_asset_columns(df_asset_rets.index[-1])
        return n_timestamps, df_betas, None

    x_fact = df_fact_rets.to_numpy()

    # A factor that never moves cannot be regressed on: its column is collinear
    # with the intercept, so the design is singular. Measured over each column's
    # own defined values, this is a property of the input rather than of any one
    # window, so it raises. The complete-case form of the same question is asked
    # below, on the rows the regression will actually use, and degrades instead.
    if np.any(finite_column_variances(x_fact) < NEAR_ZERO_VARIANCE_THRESHOLD):
        logger.error("One or more factors in df_fact_rets have near-zero variance.")
        raise ValueError("One or more factors in df_fact_rets have near-zero variance.")

    # Complete-case deletion (listwise deletion, as R's lm does under na.omit):
    # a row on which any factor is non-finite is dropped from the regression for
    # every asset. Factors are shared across assets, so the admitted row set is
    # shared too; each asset then narrows it further by its own defined rows.
    #
    # This replaces discarding the whole window on the first bad row. That
    # discard turned a late-listing factor into a permanent outage: once the
    # factor's first value arrives its column survives the point-in-time all-NaN
    # drop, but the window still holds every earlier NaN row, and because the
    # window only ever grows those rows never age out. Every subsequent window
    # was discarded, so the other factors' betas froze at their last pre-listing
    # values and the late factor never received a beta at all.
    complete_rows: NDArray[np.bool_] = np.asarray(
        np.isfinite(x_fact).all(axis=1), dtype=bool
    )
    n_complete = int(np.count_nonzero(complete_rows))
    if n_complete < n_timestamps:
        _warn_deleted_rows(
            n_timestamps - n_complete, n_timestamps, df_fact_rets.index[-1]
        )

    # Check minimum timestamps against the rows that survive deletion, not the
    # window length: a window can be long and still leave too few complete rows
    # to fit. Gating here also keeps the variance check below out of the region
    # where variance over a handful of rows is meaningless.
    if n_complete < min_timestamps:
        logger.warning(
            "Insufficient data: %d complete timestamp(s) available of %d in the "
            "window, minimum required is %d.",
            n_complete,
            n_timestamps,
            min_timestamps,
        )
        # Not enough timestamps to perform regression.
        # Return the timestamp count and all-NaN beta matrix.
        return n_timestamps, df_betas, None

    # Near-zero variance over the admitted rows. Unlike the whole-column check
    # above this is not a defect in the input: a factor can vary plenty and still
    # be flat across one window's surviving rows, if its variation happens to sit
    # in the rows deletion removed. That is a local, recoverable condition -- the
    # next window admits more rows -- so it degrades to no betas for this date
    # and lets the caller's forward fill carry the prior ones over the gap.
    #
    # ddof=1 matches pandas' var(), so a hole-free window is scored exactly as
    # before. It would divide by zero on a single admitted row, and one row is
    # degenerate regardless, so that case short-circuits.
    if (
        n_complete < 2
        or x_fact[complete_rows].var(axis=0, ddof=1).min()
        < NEAR_ZERO_VARIANCE_THRESHOLD
    ):
        logger.warning(
            "One or more factors have near-zero variance over the %d complete "
            "row(s) of the window ending %s, so its design is singular; no new "
            "betas for that window, prior betas carry forward.",
            n_complete,
            df_fact_rets.index[-1],
        )
        return n_timestamps, df_betas, None

    return n_timestamps, df_betas, complete_rows


def prepare_weighted_regression_inputs(
    df_asset_rets: pd.DataFrame,
    df_fact_rets: pd.DataFrame,
    half_life: float | None,
    lambda_: float | None,
    min_timestamps: int,
) -> tuple[
    pd.DataFrame,
    np.ndarray | None,
    pd.DataFrame | None,
    NDArray[np.bool_] | None,
]:
    """Validate inputs and prepare shared weighted-regression matrices.

    The fourth element is the complete-case row mask -- the rows on which every
    factor is finite. Both fit paths intersect it with each asset's own defined
    rows, so the mask is built once here rather than in each of them.
    """
    n_timestamps, df_betas, complete_rows = _validate_beta_inputs(
        df_asset_rets, df_fact_rets, min_timestamps
    )
    if complete_rows is None:
        return df_betas, None, None, None

    # Calculate weights over the full window, not over the surviving rows. The
    # half-life is stated in calendar time, so a deleted row must leave a hole in
    # the decay rather than compress it; the weights are subset by each fit path,
    # never rebuilt. They are also not renormalized after subsetting: scaling
    # every admitted row by one common factor scales y and the design alike and
    # leaves the Huber fit's betas unchanged, and the existing per-asset mask
    # already behaves this way.
    weights: np.ndarray = exponential_weights(
        n_timestamps, half_life=half_life, lambda_=lambda_
    )
    sqrt_weights: np.ndarray = np.sqrt(weights)

    # Implement weighted regression for each asset
    # by multiplying the x and y matrices by the square root of the weights.
    x_weighted: pd.DataFrame = df_fact_rets.multiply(sqrt_weights, axis=0)
    return df_betas, sqrt_weights, x_weighted, complete_rows


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
    df_betas, sqrt_weights, x_weighted, complete_rows = (
        prepare_weighted_regression_inputs(
            df_asset_rets, df_fact_rets, half_life, lambda_, min_timestamps
        )
    )
    # If no betas can be produced for this window, return the all-NaN matrix.
    if sqrt_weights is None or x_weighted is None or complete_rows is None:
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
        # If so, drop any NaN values and continue. The admitted rows are this
        # asset's own defined rows intersected with the complete-case rows, so a
        # date on which any factor is missing is excluded here for every asset --
        # listwise deletion, matching R's lm() under na.omit.
        y_filtered, valid_mask = check_min_timestamps_series(
            y_weighted, min_timestamps, extra_mask=complete_rows
        )
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
